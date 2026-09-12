# Jarvis Training Throughput: Production vs. Fast Path Investigation

**Date:** September 12, 2026  
**Hardware:** NVIDIA GeForce RTX 5070 12GB (Blackwell Laptop GPU / Desktop Profile, 12,227 MiB dedicated VRAM)  
**Model Architecture:** Jarvis-Q1.58-500M (606.4M total parameters, 24 layers, $d_{\text{model}}=1024$, 16 heads, 4 experts Top-2, $T=512$, $B=2$, accum=4, effective batch 4,096 tokens/update)  
**Status:** Investigation Complete & Verified

---

## 1. Executive Summary & Root Cause Findings

During the 50M-token baseline pretraining run, training throughput stabilized at **~374–403 tok/s** (~10.9s per optimizer update), despite previous standalone component benchmarks projecting **~745–1,480 tok/s** (~2.75–5.4s per update).

Through systematic differential code audits, component profiling, and live micro-benchmarking, **three distinct root causes** were identified and proven:

### Root Cause 1: Attention Backend Fallback (`use_cuda_attn=False`)
- **What happened:** In `experiments/architecture_matrix/train_1b_production.py` (line 216), the model was explicitly instantiated with:
  ```python
  use_cuda_attn=False, # PyTorch vectorized fallback for stability
  ```
- **Consequence:** Jarvis fell back to `AssociativeLinearAttention` in `jarvis_model.py`. For every micro-batch and each of the 24 layers, it unrolled an 8-iteration Python `for` loop over chunks ($T=512$, $cs=64$). Each iteration launched 4 sequential `torch.einsum` operations. Across 4 gradient accumulation steps, this amounted to **768 chunk loop iterations in forward and 768 in backward**, creating hundreds of small CUDA kernel launches and intermediate tensor allocations.
- **The Fast Path:** `CUDAAssociativeLinearAttention` in `associative_attention_cuda/` fuses RoPE + ELU(x)+1 into a single CUDA kernel, reshapes all chunks across $(B \times H \times N_c = 256)$ into unified batch dimensions, executes **Tensor Core batched matrix multiplications (`torch.bmm`)**, and performs a fused recurrent state scan.
- **Measured Impact:** Restoring CUDA attention reduces single-layer checkpointed forward+backward latency from **56.38 ms $\to$ 28.83 ms (1.96x speedup)**.

---

### Root Cause 2: Checkpoint Compaction GPU VRAM Spike Triggering Windows WDDM PCIe Memory Paging
- **What happened:** In `train_1b_production.py`, `save_checkpoint` compacted model and optimizer states to `bfloat16` directly on the GPU:
  ```python
  opt_raw = optimizer.state_dict()
  for p_idx, p_state in opt_raw["state"].items():
      ... v.to(torch.bfloat16)
  ```
- **Consequence:** Compacting ~4.8 GB of float32 optimizer states (exp_avg, exp_avg_sq, momentum) on the GPU required an additional ~3.47 GB of transient VRAM allocation. At Step 3,750 (the first checkpoint save), allocated VRAM spiked from **9,518 MB $\to$ 12,992 MB**, exceeding the physical dedicated VRAM of the RTX 5070 (12,227 MiB).
- **The WDDM Paging Trap:** Under Windows WDDM, exceeding dedicated VRAM does not crash with CUDA OOM; instead, Windows silently evicts pages over PCIe to host system RAM. Crucially, PyTorch's caching allocator held the reserved pool at **12,966–13,032 MB** permanently because `torch.cuda.empty_cache()` was never invoked. This forced all subsequent training steps (from Step 3,750 to Step 10,500) to suffer high PCIe paging overhead, cutting training throughput from ~755 tok/s down to ~374 tok/s.
- **The Fix:** Stream state dictionaries directly to CPU memory (`v.detach().to("cpu", dtype=torch.bfloat16)`) during compaction, and invoke `torch.cuda.empty_cache()` after saving. This reduces the GPU VRAM spike from **+3,474 MB $\to$ +0.0 MB**, maintaining reserved memory at **~10,430 MB** (well below the 12,227 MiB ceiling).

---

### Root Cause 3: Cumulative Throughput Averaging in Production Logging
- **What happened:** In `train_1b_production.py`:
  ```python
  dt = now - t_start
  tok_s = (step - start_step) * tokens_per_step / max(dt, 1e-4)
  ```
- **Consequence:** `tok_s` computed the cumulative average throughput since script initialization `t_start`. Any pauses, holdout validation runs (which run 32 batches), or initial step delays permanently dragged down the reported metric, obscuring true instantaneous training speed.
- **The Fix:** Log both instantaneous throughput `tokens_per_step / t_step_dur` and cumulative average throughput.

---

## 2. Differential Code Path Matrix

| Feature / Subsystem | Slow Production Path | Fast Integrated Path | Impact on Latency |
| :--- | :--- | :--- | :--- |
| **Attention Backend** | `AssociativeLinearAttention` (PyTorch chunked einsum loops) | `CUDAAssociativeLinearAttention` (Fused RoPE/ELU + Batched BMM + Fused Scan) | **~2.75s per update savings (1.98x speedup)** |
| **MoE Backend** | `CUDASparseMoELayer` (CUDA dispatch + expert GEMMs) | `CUDASparseMoELayer` (Identical) | Neutral (both used CUDA MoE) |
| **LSF Backend** | `LiquidStateFusion` (PyTorch causal scan) | `LiquidStateFusion` (PyTorch causal scan) | Neutral (<4.4 ms/layer; <5% of step) |
| **Precision** | PyTorch AMP `torch.bfloat16` | PyTorch AMP `torch.bfloat16` | Neutral |
| **Checkpoint Compaction** | GPU-side in-place duplication (Peak: 12,992 MB) | CPU-streamed compaction (Peak: 9,422 MB) | **Eliminates WDDM PCIe paging; prevents 2x throttle** |
| **Dataloader** | `ShardedTokenDataset` | `ShardedTokenDataset` | Neutral (0.44 ms per update; <0.01% of step) |
| **Optimizer** | `AdamW(fused=True)` | `AdamW(fused=True)` | Neutral |

---

## 3. Verified Performance Recovery

Live benchmarks of the full 606M model (24 layers, $B=2, T=512$, accum=4, 4,096 tokens/update) on the RTX 5070 confirm:

```text
=====================================================================================
Configuration                                     | Step Time    | Throughput
-------------------------------------------------------------------------------------
Unoptimized Production Run (Paging + Ref Attn)    |  10.9500 s   |    374.0 tok/s
Reference Attn Without Paging (Clean VRAM)        |   5.5438 s   |    738.8 tok/s
Upgraded Production Engine (CUDA Attn + CPU Ckpt) |   2.7904 s   |  1,467.9 tok/s
-------------------------------------------------------------------------------------
End-to-End Throughput Gain: 3.92x (+292.5%)
```
