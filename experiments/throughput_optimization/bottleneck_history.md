# Jarvis Throughput Optimization: Bottleneck History

This document tracks the iterative optimization progress across all optimization loops for the Jarvis-Q1.58-500M training engine on the RTX 5070 12GB.

---

## Optimization Iteration Summary Table

| Iteration | Optimization Description | Before tok/s | After tok/s | Speedup | Peak VRAM | Numerical Correctness | Decision |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Restore CUDA Attention & Fix Checkpoint VRAM Paging**<br>• Enabled `CUDAAssociativeLinearAttention`<br>• Replaced GPU-side optimizer compaction with CPU streaming<br>• Added `torch.cuda.empty_cache()` post-eval/ckpt | 374.0 tok/s | 1,467.9 tok/s | **3.92x** (+292.5%) | 9,422 MB (spikes eliminated) | **Verified**<br>• Max diff: $1.8 \times 10^{-5}$<br>• Zero NaNs / Infs | **KEEP** |
| **Phase 1A** | **Activation Stashing (Gradient Checkpointing OFF)**<br>• Evaluated disabling gradient checkpointing across 24 layers to eliminate 22.3% recomputation compute | 1,513.6 tok/s | 178.1 tok/s | **0.12x** (-88.2%) | 18,394 MB (+7.76 GB paging) | **Verified**<br>• Max diff: $1.5 \times 10^{-5}$ | **REVERT (KEEP CKPT ON)** |
| **Phase 1D/G**| **Packed 2-Bit Ternary CUDA Kernel (SM120)**<br>• Evaluated direct 2-bit packed ternary GEMM (0.25 B/weight) with register unpacking on CUDA cores | 1,546.3 tok/s | 171.6 tok/s | **0.11x** (-88.9%) | 17,256 MB (autograd tape) | **Verified**<br>• Max diff: $7.8 \times 10^{-3}$ | **REVERT (KEEP cuBLAS)** |
| *Phase 2* | *MoE Grouped / Fused GEMM (CUTLASS / CuTe / Triton)* | 1,467.9 tok/s | — | — | — | Pending | Planned |
| *Phase 3* | *Blackwell Native FP8 Tensor Cores (122.9–147.5 TFLOPs)* | 1,467.9 tok/s | — | — | — | Pending | Planned |
| *Phase 4* | *CUDA Graph Capture of Fixed-Shape Step* | — | — | — | — | Pending | Planned |

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
- **Empirical Measurement:** Autograd graph for non-linear neuromorphic layers (attention chunk states, MoE scatter/gather maps, LSF decay tensors) consumed **+7,760.5 MB of persistent VRAM**. Peak VRAM reached **18,394 MB**, causing severe PCIe paging and dropping throughput from **1,513.6 tok/s to 178.1 tok/s (-88.2%)**.
- **Decision:** **REVERT / KEEP GRADIENT CHECKPOINTING ON.**

---

## Phase 1D/G Deep Dive: Packed Ternary CUDA Kernel
- **Date:** September 12, 2026
- **Hypothesis:** 2-bit packed ternary weights ($0.25$ bytes/param) reduce memory bandwidth by 8x.
- **Empirical Measurement:** Isolated kernel on $1024 \times 1024$ achieved **6.51 TFLOPs** vs. **25.81 TFLOPs** for dense cuBLAS BF16 on Blackwell Tensor Cores. Full model test dropped throughput to **171.6 tok/s**.
- **Root Cause:** Jarvis training is deeply compute-bound (1,139 FLOPs/Byte vs 114 hardware ridge point). Software unpacking on CUDA cores sacrifices 4x raw Tensor Core compute density to save memory bandwidth that is not bottlenecking the system.
- **Decision:** **REVERT / DEFER PACKED TERNARY KERNEL FROM PRODUCTION.** Focus low-precision efforts on hardware FP8 Tensor Cores (Phase 3).
