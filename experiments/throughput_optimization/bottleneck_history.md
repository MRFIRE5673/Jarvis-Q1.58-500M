# Jarvis Throughput Optimization: Bottleneck History

This document tracks the iterative optimization progress across all optimization loops for the Jarvis-Q1.58-500M training engine on the RTX 5070 12GB.

---

## Optimization Iteration Summary Table

| Iteration | Optimization Description | Before tok/s | After tok/s | Speedup | Peak VRAM | Numerical Correctness | Decision |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Restore CUDA Attention & Fix Checkpoint VRAM Paging**<br>• Enabled `CUDAAssociativeLinearAttention` (fused RoPE/ELU + batched Tensor Core BMM + fused scan)<br>• Replaced GPU-side optimizer compaction with CPU-streamed compaction in `save_checkpoint`<br>• Added `torch.cuda.empty_cache()` after eval and checkpointing | 374.0 tok/s | 1,467.9 tok/s | **3.92x** (+292.5%) | 9,422 MB (spikes eliminated) | **Verified**<br>• Max diff vs PyTorch ref: $1.8 \times 10^{-5}$<br>• Zero NaNs / Infs<br>• Gradient norms identical | **KEEP** |
| *2* | *Profile Step Breakdown & Audit Remaining Compute* (MoE vs Attention vs Fusion) | 1,467.9 tok/s | — | — | — | In Progress | Planned |
| *3* | *torch.compile / AOTInductor Suitability for Fixed T=512* | — | — | — | — | Pending | Planned |
| *4* | *Tensor Core Utilization & BF16 Precision Audit* | — | — | — | — | Pending | Planned |
| *5* | *CUDA Attention Kernel Specialization for T=512* | — | — | — | — | Pending | Planned |
| *6* | *MoE Routing, Dispatch & Token Packing Fusion* | — | — | — | — | Pending | Planned |
| *7* | *Microbatch & Gradient Accumulation Grid Tuning* ($B=4, \text{accum}=2$) | — | — | — | — | Pending | Planned |
| *8* | *Packed 1.58-bit Ternary CUDA Kernel (Long-Term)* | — | — | — | — | Pending | Planned |

---

## Iteration 1 Deep Dive

- **Date:** September 12, 2026
- **Hypothesis:** Production training degraded to ~374 tok/s due to (1) algorithmic fallback to unrolled PyTorch einsum loops in `AssociativeLinearAttention` when `use_cuda_attn=False`, and (2) Windows WDDM PCIe paging caused by GPU-side optimizer compaction spiking VRAM above the 12,227 MiB physical ceiling.
- **Root Causes Confirmed:**
  1. `train_1b_production.py` had `use_cuda_attn=False` hardcoded.
  2. `save_checkpoint` duplicated 4.8 GB of optimizer states in VRAM, causing an immediate jump from 9,518 MB to 12,992 MB. PyTorch's reserved allocator held memory above 12.9 GB indefinitely, forcing WDDM to evict memory to host RAM.
- **Implemented Fixes:**
  - Defaulted `use_cuda_attn=True` in `train_1b_production.py` with `--no-cuda-attn` CLI fallback flag.
  - Implemented CPU-streamed state dictionary compaction (`v.detach().to("cpu", dtype=torch.bfloat16)`).
  - Added `torch.cuda.empty_cache()` post-evaluation and post-checkpoint.
  - Added instantaneous `step_tok_s` metric to production logging.
- **Measurements:**
  - Single-layer block fwd+bwd latency: 56.38 ms $\to$ 28.83 ms (**1.96x**)
  - Full model update time: 10.95 s $\to$ 2.79 s (**3.92x**)
  - Throughput: 374.0 tok/s $\to$ **1,467.9 tok/s**
  - Peak allocated VRAM: 12,992 MB $\to$ **9,422 MB**
- **Decision:** **KEEP**. Changes committed to `train_1b_production.py`.
