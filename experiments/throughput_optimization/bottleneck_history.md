# Jarvis Throughput Optimization: Bottleneck History

This document tracks the iterative optimization progress across all optimization loops for the Jarvis-Q1.58-500M training engine on the RTX 5070 12GB.

---

## Optimization Iteration Summary Table

| Iteration | Optimization Description | Before tok/s | After tok/s | Speedup | Peak VRAM | Numerical Correctness | Decision |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Restore CUDA Attention & Fix Checkpoint VRAM Paging**<br>• Enabled `CUDAAssociativeLinearAttention`<br>• Replaced GPU-side optimizer compaction with CPU streaming<br>• Added `torch.cuda.empty_cache()` post-eval/ckpt | 374.0 tok/s | 1,467.9 tok/s | **3.92x** (+292.5%) | 9,422 MB (spikes eliminated) | **Verified**<br>• Max diff: $1.8 \times 10^{-5}$<br>• Zero NaNs / Infs | **KEEP** |
| **Phase 1A** | **Activation Stashing (Gradient Checkpointing OFF)**<br>• Evaluated disabling gradient checkpointing across 24 layers to eliminate 22.3% recomputation compute | 1,513.6 tok/s | 178.1 tok/s | **0.12x** (-88.2%) | 18,394 MB (+7.76 GB paging) | **Verified**<br>• Max diff: $1.5 \times 10^{-5}$ | **REVERT (KEEP CKPT ON)** |
| **Phase 1D/G**| **Packed 2-Bit Ternary CUDA Kernel (SM120)**<br>• Evaluated direct 2-bit packed ternary GEMM (0.25 B/weight) with register unpacking on CUDA cores | 1,546.3 tok/s | 171.6 tok/s | **0.11x** (-88.9%) | 17,256 MB (autograd tape) | **Verified**<br>• Max diff: $7.8 \times 10^{-3}$ | **REVERT (KEEP cuBLAS)** |
| **Phase 2** | **Blackwell FP8 Tensor Core Execution (`_scaled_mm`)**<br>• Evaluated native `float8_e4m3fn` Tensor Cores. Isolated GEMM: 1.16x–1.45x faster (51.3 TFLOPs). End-to-end layer training: 0.857 ms (BF16) vs 2.058 ms (FP8). | 1,513.6 tok/s | ~630 tok/s | **0.42x** (-58.3%) | 10,240 MB | **Verified**<br>• Output error: 2.7%<br>• Grad error: 2.8% | **REVERT (KEEP BF16 cuBLAS)** |
| **Phase 3** | **Micro-Batch Scaling ($B=4, \text{accum}=2$)**<br>• Doubled token batch dimension $M=1024 \to M=2048$ while holding effective update constant at 4,096 tokens<br>• Saturated 48 Blackwell SMs: GEMM utilization surged from 38% to 65–84% of peak<br>• Halved gradient accumulation loop iterations | 1,590.1 tok/s | 2,934.9 tok/s (3,050.3 steady) | **1.94x** (+94.3%) | 11,007 MB alloc / 11,946 MB res | **Verified**<br>• Zero NaNs / Infs<br>• Loss: 9.68 vs 9.70<br>• Zero paging | **KEEP** |
| **Phase 4** | **Triton Grouped MoE GEMM / Expert Dispatch**<br>• Fused all 4 expert GEMMs into a single unified kernel launch<br>• Eliminated 48 host-device synchronization stalls (`offsets_cpu`)<br>• Cut MoE launches from 44 to 12 per layer (2,112 to 576 per step) | 3,379.2 tok/s | 4,596.5 tok/s (4,660.6 peak) | **1.36x** (+36.0%) | 10,819 MB alloc / 11,180 MB res | **Verified**<br>• Max logit diff: 0.0<br>• Zero NaNs / Infs | **KEEP (NEW PRODUCTION BENCHMARK)** |
| *Phase 5* | *CUDA Graph Capture of Fixed-Shape Step* | 4,596.5 tok/s | — | — | — | Pending | Planned |

---

## Iteration 1 Deep Dive
- **Date:** September 12, 2026
- **Root Cause:** (1) Hardcoded `use_cuda_attn=False` forcing 768 unrolled einsum loop iterations per step. (2) Checkpoint serialization spiking VRAM to 12.99 GB, triggering WDDM PCIe memory paging.
- **Implemented Fixes:** Enabled `CUDAAssociativeLinearAttention`, CPU-streamed checkpointing, and cache flushing.
- **Result:** Throughput restored from 374.0 tok/s to **1,467.9 tok/s** (3.92x speedup).
- **Decision:** **KEEP**.

---

## Phase 1A Deep Dive: Activation Stashing
- **Date:** September 12, 2026
- **Hypothesis:** Disabling checkpointing eliminates 24-layer recomputation (saving 22.3% step compute).
- **Empirical Measurement:** Autograd graph consumed **+7,760.5 MB of persistent VRAM**. Peak VRAM reached **18,394 MB**, causing severe PCIe paging and dropping throughput from **1,513.6 tok/s to 178.1 tok/s (-88.2%)**.
- **Decision:** **REVERT / KEEP GRADIENT CHECKPOINTING ON.**

---

## Phase 1D/G Deep Dive: Packed Ternary CUDA Kernel
- **Date:** September 12, 2026
- **Hypothesis:** 2-bit packed ternary weights ($0.25$ bytes/param) reduce memory bandwidth by 8x.
- **Empirical Measurement:** Isolated kernel on $1024 \times 1024$ achieved **6.51 TFLOPs** vs. **25.81 TFLOPs** for dense cuBLAS BF16 on Blackwell Tensor Cores. Full model test dropped throughput to **171.6 tok/s**.
- **Root Cause:** Jarvis training is deeply compute-bound (1,139 FLOPs/Byte vs 114 hardware ridge point). Software unpacking on CUDA cores sacrifices 4x raw Tensor Core compute density.
- **Decision:** **REVERT / DEFER PACKED TERNARY KERNEL FROM PRODUCTION.**

---

## Phase 2 Deep Dive: Blackwell FP8 Tensor Core Execution
- **Date:** September 12, 2026
- **Hypothesis:** FP8 Tensor Cores deliver 2x higher theoretical throughput than BF16 (122.9 vs 61.4 TFLOPs).
- **Empirical Measurement:**
  - Isolated FP8 GEMM was **1.16x to 1.45x faster** (up to 51.27 TFLOPs on MoE shapes).
  - However, in full training (forward + backward), dynamic activation quantization, three `_scaled_mm` calls, and cuBLASLt transposed-stride memory copies increased layer training latency from **0.857 ms (BF16) $\to$ 2.058 ms (FP8)**.
- **Root Cause:** In eager execution, dynamic scaling and global-DRAM transposition copies take ~1.2 ms per layer, overshadowing the 0.08 ms GEMM compute.
- **Decision:** **REVERT / KEEP BF16 cuBLAS AS PRODUCTION DEFAULT.** Next optimization target is **Micro-batch tuning ($B=4, \text{accum}=2$)** and **MoE Grouped GEMM**.

---

## Phase 3 Deep Dive: Micro-Batch / Tensor-Core Utilization Sweep
- **Date:** September 12, 2026
- **Hypothesis:** At $B=2$ ($M=1024$), cuBLAS GEMM tile sizes under-saturate the 48 SMs of the RTX 5070 Blackwell GPU. Scaling to $B=4$ ($M=2048$) while keeping total tokens per optimizer update at 4,096 ($accum=2$) will dramatically improve Tensor Core arithmetic utilization without exceeding the 12,227 MiB physical VRAM ceiling.
- **Empirical Measurement:**
  - **$B=1, \text{accum}=8$:** 5.1352 s/step | **797.6 tok/s** | Peak Alloc: 9,819 MB | Peak Res: 10,190 MB (PASS, 0.50x speedup).
  - **$B=2, \text{accum}=4$ (Baseline):** 2.5760 s/step | **1,590.1 tok/s** | Peak Alloc: 10,214 MB | Peak Res: 10,520 MB (PASS, 1.00x).
  - **$B=4, \text{accum}=2$ (Winner):** 1.3956 s/step (1.3428s steady) | **2,934.9 tok/s (3,050.3 steady)** | Peak Alloc: 11,007 MB | Peak Res: 11,946 MB (PASS, 281 MB headroom, zero paging). **1.94x speedup (+94.3%)!**
  - **$B=8, \text{accum}=1$:** 1.5272 s/step | 2,682.0 tok/s | Peak Res: **12,780 MB** (FAILS: Exceeds 12,227 MiB physical limit $\implies$ Windows WDDM PCIe memory paging).
- **GEMM Scaling Data:**
  - Attention ($1024 \times 1024$): $M=1024 \implies 23.76\text{ TFLOPs}$ (38.7% peak) $\to$ $M=2048 \implies \mathbf{39.72\text{ TFLOPs}}$ (64.7% peak, 1.67x TC efficiency).
  - MoE Up ($1024 \times 2048$): $M=1024 \implies 39.99\text{ TFLOPs}$ (65.1% peak) $\to$ $M=2048 \implies \mathbf{51.09\text{ TFLOPs}}$ (83.2% peak).
  - MoE Down ($2048 \times 1024$): $M=1024 \implies 32.67\text{ TFLOPs}$ (53.2% peak) $\to$ $M=2048 \implies \mathbf{51.70\text{ TFLOPs}}$ (84.1% peak).
- **Root Cause of Speedup:**
  1. Higher Tensor Core wave quantization and arithmetic saturation across 48 SMs (38–65% $\to$ 65–84% of hardware ceiling).
  2. Halving gradient accumulation loops from 4 to 2, eliminating 50% of Python runtime dispatch overhead and intermediate tensor accumulation traffic.
- **Decision:** **KEEP AS NEW PRODUCTION DEFAULT.** Throughput nearly doubled to ~3,000 tok/s within physical VRAM boundaries.

---

## Phase 4 Deep Dive: MoE Grouped GEMM / Expert Dispatch
- **Date:** September 12, 2026
- **Hypothesis:** Sequential expert execution in MoE layers causes 48 CPU-GPU synchronization stalls per update (from `offsets_cpu = expert_offsets.cpu().numpy()`) and launches 1,152 separate cuBLAS GEMMs per update at suboptimal batch tile sizes. Fusing all 4 experts into a unified Triton Grouped GEMM will saturate Blackwell SM120, eliminate host stalls, and significantly accelerate full training throughput.
- **Empirical Measurement:**
  - **Single MoE Layer Compute:** 16.18 ms (Baseline) $\to$ **9.00 ms (Grouped)** (**1.80x faster**; Fwd: 1.84x, Bwd: 1.79x).
  - **Full Model Forward (B=4, T=512):** 146.58 ms $\to$ **103.19 ms** (**1.42x faster / -29.6% latency**).
  - **Full Training Step (4,096 tokens):** 1,212.11 ms (3,379.2 tok/s) $\to$ **891.12 ms (4,596.5 tok/s steady, 4,660.6 peak)** (**1.36x speedup / +36.0% throughput**).
  - **Peak VRAM:** Alloc: 10,819.4 MB | Res: 11,180.0 MB (Safe +1,047 MiB headroom to 12,227 MiB limit, **zero PCIe paging**).
  - **Kernel Launches:** 2,112 launches $\to$ **576 launches per step (3.67x reduction)**.
  - **Host Syncs:** 48 stalls $\to$ **0 stalls (100% eliminated)**.
- **Numerical Verification:**
  - Forward Output Max Logit Diff: **0.000000e+00** (Cosine similarity: 1.0000001).
  - Gradient Cosine Similarity: 0.9999986.
- **Root Cause of Speedup:**
  1. Fusing 4 expert passes into a single Triton Grouped GEMM with 2D grid scheduling keeps all 48 SMs fully occupied simultaneously (65.57 TFLOPs achieved).
  2. Complete eradication of 48 host-device synchronization stalls per update (`offsets` remains entirely on GPU).
  3. Elimination of PyTorch autograd tensor slice tracking graph overhead.
- **Decision:** **KEEP AND MERGE AS NEW PRODUCTION BENCHMARK.** Step time dropped below 900 ms, raising throughput to ~4,600 tok/s.


