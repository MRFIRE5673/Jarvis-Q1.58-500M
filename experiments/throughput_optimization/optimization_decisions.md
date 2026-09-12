# Jarvis Throughput Optimization: Formal Decision Log

This document records the formal decision framework, empirical evidence, and risk analysis for each optimization implemented in the Jarvis training engine.

---

## Decision Record 001: Restore CUDA Attention & CPU Checkpoint Streaming

### 1. Motivation
During the 50M-token baseline run, training throughput was ~374–403 tok/s (~10.9s per optimizer update), roughly 2x slower than prior benchmark projections (~745 tok/s). We needed to identify the exact cause, recover throughput, and protect the system against silent performance regressions.

### 2. Research & Empirical Evidence
1. **Code Audit:** In `experiments/architecture_matrix/train_1b_production.py`, line 216 instantiated the model with `use_cuda_attn=False`. This forced the attention backend to execute `AssociativeLinearAttention`, which loops 8 times per sequence over chunks ($T=512, cs=64$), launching 4 sequential `torch.einsum` operations per chunk.
2. **CUDA Kernel Differential:** `CUDAAssociativeLinearAttention` (`associative_attention_cuda/`) implements a fused RoPE/ELU kernel, reshapes chunk dimensions into batched Tensor Core operations (`torch.bmm`), and performs a fused associative recurrent scan.
3. **Memory Profile:** In `train_1b_production.py`, `save_checkpoint` duplicated optimizer states on the GPU into `bfloat16`, causing allocated VRAM to spike from 9,518 MB to 12,992 MB (exceeding the RTX 5070's 12,227 MiB limit). Windows WDDM silently paged VRAM across PCIe to host RAM, cutting subsequent training throughput in half.

### 3. Hypothesis
Enabling the precompiled `CUDAAssociativeLinearAttention` backend and moving optimizer checkpoint compaction to CPU host memory will:
1. Eliminate 768 unrolled chunk loop iterations per update, reducing single-block latency by ~48%.
2. Eliminate the 3.47 GB VRAM compaction spike, keeping peak reserved VRAM under 10.5 GB and preventing Windows WDDM PCIe paging.
3. Restore training throughput from ~374 tok/s to >1,400 tok/s.

### 4. Implementation
- **File Modified:** [`experiments/architecture_matrix/train_1b_production.py`](file:///e:/Jarvis-Q1.58-500M/experiments/architecture_matrix/train_1b_production.py)
- **Changes:**
  - Added CLI flag `--use-cuda-attn` (default `True`) and `--no-cuda-attn` for explicit fallback.
  - In `save_checkpoint`, moved tensor compaction directly to CPU:
    ```python
    opt_sd["state"][p_idx][k] = v.detach().to("cpu", dtype=torch.bfloat16)
    ```
  - Added `torch.cuda.empty_cache()` immediately following validation and checkpoint serialization.
  - Added instantaneous throughput reporting (`step_tok_s`) to log true per-step speed alongside cumulative average.

### 5. Correctness Verification
- **Numerical Agreement:** Tested forward output and gradient agreement between `AssociativeLinearAttention` and `CUDAAssociativeLinearAttention` on identical inputs ($B=2, T=512, d=1024$):
  - Max absolute difference in outputs: $1.8 \times 10^{-5}$ (within bfloat16 numerical precision tolerances).
  - Backward gradients: Zero NaNs, zero Infs.
- **Checkpoint Compatibility:** Verified that checkpoints saved with CPU-streamed compaction load cleanly, restore float32 optimizer states correctly, and produce valid loss values upon resumption.

### 6. Baseline vs. Optimized Measurements

| Metric | Baseline (Production Run) | Clean VRAM Reference Attn | Optimized Fast Path | Net Change |
| :--- | :---: | :---: | :---: | :---: |
| **Attention Fwd+Bwd (1 layer)** | 22.79 ms | 22.79 ms | 5.77 ms | **-74.7%** (3.95x speedup) |
| **Block Fwd+Bwd (1 layer, ckpt)** | 56.38 ms | 56.38 ms | 28.83 ms | **-48.9%** (1.96x speedup) |
| **Full Step Time (4,096 tokens)** | 10.95 s | 5.54 s | 2.79 s | **-74.5%** (3.92x speedup) |
| **Training Throughput** | 374.0 tok/s | 738.8 tok/s | 1,467.9 tok/s | **+292.5%** |
| **Peak Allocated VRAM** | 12,992 MB (paged) | 10,118 MB | 9,422 MB | **-3,570 MB** (spikes eliminated) |
| **Peak Reserved VRAM** | 13,032 MB | 11,200 MB | 10,430 MB | **-2,602 MB** (safe 1.8 GB headroom) |

### 7. Risk Analysis
- **Numerical Stability:** The CUDA attention kernel uses standard FP32 accumulation in the associative scan. Over 100 test steps, gradient norms and loss values matched the reference implementation.
- **Hardware Portability:** If CUDA compilation fails or dynamic libraries are missing on another environment, the script gracefully supports `--no-cuda-attn` to fall back to PyTorch reference attention.
- **VRAM Stability:** Peak allocated memory is 9,422 MB with reserved memory capped at 10,430 MB, leaving 1,797 MiB of unallocated safety buffer.

### 8. Final Decision
**KEEP AND MERGE.**  
CUDA attention and CPU-streamed checkpointing are confirmed correct, stable, and deliver an immediate **3.92x speedup** (+292.5% throughput), completely recovering and exceeding the target throughput.

---

## Decision Record 002: Activation Stashing (Disabling Gradient Checkpointing)

### 1. Motivation
The Phase 0 roofline analysis revealed that gradient checkpointing across 24 layers recomputes 640.9M FLOPs per token during the backward pass (22.3% of total step compute). If intermediate activations could be cached in VRAM without paging, training throughput could increase by ~15–25%.

### 2. Research & Empirical Evidence
Benchmarked the complete 606M model on RTX 5070 ($B=2, T=512$, accum=4, AdamW fused) with checkpointing ON vs. OFF.
- Checkpointing ON: 2.7062 s/step, **1,513.6 tok/s**, peak allocated 10,120.7 MB, peak reserved 10,548.0 MB.
- Checkpointing OFF: 22.9920 s/step, **178.1 tok/s**, peak allocated 17,881.2 MB (+7,760.5 MB), peak reserved 18,394.0 MB.

### 3. Root Cause
The full autograd tape across 24 layers (including attention chunk states, MoE scatter/gather buffers, expert expansions, and LSF decay tensors) requires **+7.76 GB of persistent activation memory**. Total memory reached 18.4 GB, vastly exceeding the 12,227 MiB physical VRAM limit. Windows WDDM paged gigabytes across PCIe to host RAM, causing an **88.2% performance collapse**.

### 4. Final Decision
**REJECT ACTIVATION STASHING. KEEP GRADIENT CHECKPOINTING MANDATORY ON 12GB HARDWARE.**

---

## Decision Record 003: Packed 2-Bit Ternary CUDA GEMM

### 1. Motivation
Current Jarvis layers store weights as floating point numbers scaled by $\alpha \in \{- \alpha, 0, +\alpha\}$. Packing weights into 2 bits per parameter ($0.25$ bytes/param) reduces weight memory bandwidth by 8.0x (e.g. 2,048 KB $\to$ 256 KB for $1024 \times 1024$).

### 2. Research & Empirical Evidence
Built and compiled a native Blackwell SM120 CUDA extension `ternary_gemm_cuda` implementing 2-bit packing $(00=0, 01=+1, 10=-1)$, fast decoding formula `(code & 1) - (code >> 1)`, shared memory tiling, and register unpacking.
- **Micro-benchmarks:**
  - Dense cuBLAS BF16 on Tensor Cores: **25.81 to 40.17 TFLOPs** (0.083 to 0.107 ms).
  - Packed Ternary on CUDA cores: **6.51 to 7.85 TFLOPs** (0.330 to 0.547 ms).
  - cuBLAS is **3.9x to 5.1x faster** on exact Jarvis shapes.
- **Full Model Test:** Replacing attention output projections with the packed ternary kernel reduced training throughput from **1,546.3 tok/s $\to$ 171.6 tok/s (0.11x)**.

### 3. Root Cause
Jarvis training has an arithmetic intensity of **1,139 to 3,876 FLOPs/Byte**, which is 10x to 34x higher than the RTX 5070 hardware ridge point (114.3 FLOPs/Byte). Because the workload is **compute-bound, not memory-bandwidth bound**, software SIMD unpacking on standard CUDA ALUs sacrifices ~4x raw compute density compared to dedicated hardware Tensor Cores.

### 4. Final Decision
**REVERT / DEFER CUSTOM PACKED KERNEL FROM PRODUCTION.**  
Preserve dense cuBLAS Tensor Core execution in production. Direct future low-precision acceleration toward hardware Tensor Core instructions (Native FP8 in Phase 3 and Grouped GEMM in Phase 2).
