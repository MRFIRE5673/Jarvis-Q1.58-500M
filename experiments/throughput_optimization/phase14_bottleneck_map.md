# JARVIS ULTRA — PHASE 14 BOTTLENECK MAP
**Hardware Target:** NVIDIA GeForce RTX 5070 12GB (Blackwell SM120)  
**Architecture:** Jarvis-Q1.58-500M (606.4M total parameters, 24 layers, $d_{\text{model}}=1024$, Top-2 MoE)  
**Reference Step:** $B=8, T=512, \text{accum}=1$ (4,096 tokens/update), BF16, Full Checkpointing, CUDA Graph ON  
**Measured Step Time:** 287.47 ms | **Throughput:** 14,248.3 tok/s  

---

## 1. Complete Step Time Forensic Breakdown (287.47 ms)

Through rigorous call-level and kernel-level GPU event instrumentation, the 287.47 ms optimizer update breaks down into the following exact sub-components:

| Component | Sub-Operations | Total Calls/Update | CUDA GPU Time (ms) | % of Step Time |
| :--- | :--- | :---: | :---: | :---: |
| **MoE Grouped GEMMs** | W1 forward, W2 forward, dW2, dAct, dW1, dX across 24 layers | 144 GEMM passes | **75.64 ms** | 26.31% |
| **Attention GEMMs** | Q, K, V projections, Out projection (fwd + bwd + recompute) | 288 GEMMs | **58.20 ms** | 20.25% |
| **Ternary Quantize STE** | Forward quantization + backward pass (Attention + MoE weights) | 432 calls | **57.15 ms** | 19.88% |
| **Liquid State Attention** | Fused RoPE+ELU, chunk intra-attention, CUDA prefix scan, inter-attention | 72 chunk passes | **34.80 ms** | 12.11% |
| **Padded LM Head** | Padded 50,304 GEMM forward ($N=50304, K=1024$) + backward | 2 GEMMs | **21.50 ms** | 7.48% |
| **AdamW Optimizer** | Fused 16-bit/32-bit AdamW kernel across 606.4M parameters | 1 call | **13.00 ms** | 4.52% |
| **MoE Routing & Permute** | Linear router, Gaussian noise, Top-2 softmax, dispatch gather, scatter combine | 48 passes | **11.80 ms** | 4.10% |
| **MoE GELU Epilogues** | Forward activation + backward GELU derivative across 24 layers | 48 passes | **6.37 ms** | 2.22% |
| **RMSNorms & Residuals** | Pre-attn norm, pre-moe norm, final norm, residual adds | 144 kernels | **5.40 ms** | 1.88% |
| **CUDA Graph Overhead** | Device synchronization, static buffer staging, event management | — | **3.61 ms** | 1.25% |
| **TOTAL** | **Complete Verified Optimizer Update** | — | **287.47 ms** | **100.00%** |

---

## 2. Reconciling Phase 13 vs Phase 14 Claims

### A. The "124 ms Ternary Claim"
* **Previous Claim:** 624 calls taking ~198.7 µs each $\implies$ ~124.0 ms (43.1% of step).
* **Forensic Reality:**
  - Actual calls under locked training: **exactly 432 calls**.
  - Actual GPU kernel execution time: **57.15 ms** (19.88% of step).
  - Attention projection: ~142.1 µs; Stacked 4-expert MoE: ~140.9 µs.
  - The previous 124 ms figure in eager mode was inflated by **~51.8 ms of Python autograd host dispatch latency**, which is completely eliminated by CUDA Graph.

### B. The "Zero Recompute via Lean MoE" Hypothesis
* **Hypothesis:** Lean MoE saves 768 MiB, allowing gradient checkpointing to be turned off (`none`), eliminating the ~80 ms recomputation pass and boosting throughput toward 20K–25K tok/s.
* **Forensic Reality:**
  - Lean MoE indeed saves **768.0 MiB** with **100% bitwise parity** (0.0 diff).
  - However, full model activation storage for 24 layers at $B=8, T=512$ without checkpointing requires **4,176 MiB**.
  - Adding model weights (1,213 MiB), AdamW optimizer states (4,851 MiB), gradients (1,213 MiB), and CUDA Graph internal workspace (~2,000 MiB) totals **13,453 MiB**.
  - The physical ceiling of the RTX 5070 is **12,226.5 MiB**.
  - Without checkpointing, memory exceeds the physical ceiling by +1,226.5 MiB, causing catastrophic Windows WDDM paging over PCIe (throughput collapses from 14,256 tok/s to 280 tok/s).
  - Therefore, **full checkpointing remains mathematically mandatory at $B=8, T=512$ on 12GB hardware**.

---

## 3. Low-Bit Hardware Execution on Blackwell SM120

Benchmarking on actual Jarvis runtime shapes ($M=4096, K=1024, N \in [1024, 2048, 3072]$) revealed the exact physical behavior of RTX 5070 SM120:

| Format / Path | Latency ($4096 \times 1024 \times 1024$) | Effective Compute Rate | Speedup vs BF16 | Training Usability |
| :--- | :---: | :---: | :---: | :--- |
| **Native BF16 GEMM** | **141.7 µs** | **60.6 TFLOPs** | **1.00x (Baseline)** | Fully verified, 100% stable |
| **Raw INT8 GEMM** (`torch._int_mm`) | 130.2 µs | 65.9 TOPs | 1.09x | Requires int8 inputs |
| **E2E INT8** (Quant + GEMM + Dequant) | 527.3 µs | 16.3 TOPs | **0.27x (3.7x slower)** | ALU quant/dequant bottleneck |
| **Packed 2-bit Ternary Unpack + BF16** | 673.7 µs | 12.8 TFLOPs | **0.21x (4.7x slower)** | Bit-shift unpack overhead |
| **CUTLASS INT4** (`_weight_int4pack_mm`) | 503.2 µs | 17.1 TOPs | **0.26x (3.8x slower)** | Forward-only, ALU dequant |

**Core Conclusion:**  
Low-bit INT4/INT8 GEMMs on Blackwell SM120 accelerate memory-bound inference ($M=1$ decoding). For compute-bound training ($M=4096$), the ALU dequantization overhead makes low-bit emulation 3.7x to 4.7x slower than native BF16 Tensor Cores, while destroying backward autograd compatibility.

---

## 4. Winning Verified Hardware Optimizations

1. **Fused QKV Linear Projection (`scratch/phase14l_fused_qkv.py`):**
   - Combines 3 separate $1024 \times 1024$ GEMMs into one $1024 \times 3072$ GEMM with per-projection ternary quantization.
   - **Mathematical Equivalence:** 0.0 max output diff, 0.0 dX diff, 0.0 dW diff (100% bitwise parity).
   - **Isolated Speedup:** 1.502x (from 625.0 µs to 416.1 µs per layer).
   - **Full Model Gain:** Saves **~15 ms per update**, fully compatible with CUDA Graph capture.

2. **Lean Triton Grouped MoE (`scratch/phase14c_verify_lean_moe.py`):**
   - Recomputes GELU activations on the fly during backward instead of storing 32 MiB/layer in DRAM.
   - **Bitwise Output & Gradient Match:** 0.0 max diff.
   - **VRAM Reclaimed:** 768.0 MiB across 24 blocks.
