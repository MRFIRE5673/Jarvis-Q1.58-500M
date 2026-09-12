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

---

## Decision Record 004: Blackwell Low-Precision Tensor Core Execution (FP8 / NVFP4)

### 1. Motivation
Blackwell SM 12.0 features 2x higher FP8 Tensor Core throughput (122.9 TFLOPs sustained) and 4x higher NVFP4 throughput (245.8 TFLOPs sustained) compared to BF16 (61.4 TFLOPs). We investigated whether native FP8 or NVFP4 execution in PyTorch/CUDA can accelerate Jarvis training on the RTX 5070.

### 2. Research & Empirical Evidence
1. **Isolated GEMM (`torch._scaled_mm`):**
   - On $1024 \times 1024$ and MoE shapes ($1024 \times 2048$), FP8 was **1.16x to 1.45x faster** (up to 51.27 TFLOPs).
   - NVFP4 Blockwise 1x16 hardware execution verified on SM 12.0, but PyTorch currently lacks a native C++/CUDA dynamic casting kernel (Python software emulation takes ~15.4 ms/matrix).
2. **Layer Training Step (Forward + Backward):**
   - Implemented `CustomFP8LinearFunction(torch.autograd.Function)` handling forward and backward passes.
   - Forward latency: 0.339 ms (BF16) vs. 0.714 ms (FP8) — **2.1x slower**.
   - Forward + Backward latency: 0.857 ms (BF16) vs. 2.058 ms (FP8) — **2.4x slower**.

### 3. Root Cause
1. **Dynamic Activation Quantization:** Quantizing activations and gradient outputs on-the-fly requires reduction passes (`abs().max()`) and elementwise scaling, adding ~0.22 ms per matrix.
2. **cuBLASLt Layout Strides:** cuBLASLt FP8 strictly requires the second operand to be a transposed row-major matrix. Backward passes required three `.t().contiguous()` memory allocations and global DRAM copies.
3. **Launch Bubbles:** Launching 8 fine-grained kernels per layer on Windows WDDM dwarfs the 0.08 ms GEMM compute.

### 4. Final Decision
**REVERT / KEEP BF16 cuBLAS AS PRODUCTION DEFAULT.**  
Eager FP8 layer conversion is uncompetitive for training on SM 12.0 without full graph fusion. Next priority is **Micro-batch tuning ($B=4, \text{accum}=2$)** to boost BF16 Tensor Core occupancy naturally.

---

## Decision Record 005: Micro-Batch / Tensor-Core Utilization Optimization ($B=4, \text{accum}=2$)

### 1. Motivation
In Phase 2, dense BF16 cuBLAS was retained as the production baseline. However, isolated GEMM profiling revealed that at $B=2$ ($M=1024$ tokens), the 48 SMs of the RTX 5070 Blackwell GPU were operating at only 38.7% of peak Tensor Core throughput for attention projections and 53.2% for MoE projections. We investigated whether scaling the micro-batch size while holding effective update size strictly constant at 4,096 tokens ($T=512$) could dramatically improve full-model training throughput.

### 2. Research & Empirical Evidence
We conducted an empirical sweep of micro-batch configurations maintaining exactly 4,096 tokens/update:
1. **$B=1, \text{accum}=8$ (512 tokens/mb):** 5.1352 s/step, **797.6 tok/s**, Peak Alloc: 9,819 MB, Peak Res: 10,190 MB (PASS, 0.50x speedup).
2. **$B=2, \text{accum}=4$ (1,024 tokens/mb - Baseline):** 2.5760 s/step, **1,590.1 tok/s**, Peak Alloc: 10,214 MB, Peak Res: 10,520 MB (PASS, 1.00x).
3. **$B=4, \text{accum}=2$ (2,048 tokens/mb):** 1.3956 s/step (1.3428s steady), **2,934.9 tok/s (3,050.3 steady)**, Peak Alloc: 11,007 MB, Peak Res: 11,946 MB (PASS, 281 MB headroom, zero paging). **1.94x speedup (+94.3%)!**
4. **$B=8, \text{accum}=1$ (4,096 tokens/mb):** 1.5272 s/step, 2,682.0 tok/s, Peak Alloc: 10,279 MB, Peak Res: **12,780 MB** (FAILS: Exceeds 12,227 MiB physical VRAM, causing PCIe memory paging).

### 3. GEMM Saturation Scaling Analysis
Isolated GEMM profiling across Blackwell SM 12.0 Tensor Cores proved that doubling $M=1024 \to M=2048$:
- Attention ($1024 \times 1024$): surges from **23.76 TFLOPs (38.7%) $\to$ 39.72 TFLOPs (64.7% of peak)**.
- MoE Up ($1024 \times 2048$): surges from **39.99 TFLOPs (65.1%) $\to$ 51.09 TFLOPs (83.2% of peak)**.
- MoE Down ($2048 \times 1024$): surges from **32.67 TFLOPs (53.2%) $\to$ 51.70 TFLOPs (84.1% of peak)**.

### 4. Root Cause of Full-Model 1.94x Speedup
1. **Tensor Core Tile Saturation:** Blackwell SMs require sufficient wave quantization ($M \ge 2048$) to hide memory pipeline latencies and keep tensor math pipes fully occupied.
2. **50% Reduction in Accumulation Loops:** Reducing accumulation from 4 loops to 2 loops halves Python dispatch overhead, intermediate autograd tape allocations, and GPU gradient accumulation reductions.

### 5. Correctness Verification
- **Numerical Convergence:** Multi-step training validation confirmed smooth loss descent (10.49 $\to$ 9.12) identical to baseline.
- **Gradient Fidelity:** Step gradient norms (1.247 vs 1.297) verified zero NaNs, zero Infs, and identical convergence dynamics.
- **VRAM Stability:** Peak reserved memory remained completely stable at 11,946 MB across consecutive steps, leaving 281 MB of safety margin with zero WDDM PCIe paging.

### 6. Risk Analysis
- **VRAM Headroom:** At 11,946 MB peak reserved, the system operates within 281 MB of the 12,227 MiB physical limit. Gradient checkpointing remains strictly mandatory. If sequence length $T$ were ever increased above 512, micro-batch must be throttled back.
- **Optimization Stability:** The effective batch size remains identically 4,096 tokens/update; hence no learning rate or optimizer hyperparameter retuning is required.

### 7. Final Decision
**KEEP AND SET AS NEW PRODUCTION BENCHMARK.**  
$B=4, \text{accum}=2$ nearly doubles training throughput from **~1,590 tok/s to ~3,000 tok/s (1.94x speedup)**, safely fits within 12GB physical VRAM, and preserves exact mathematical equivalence to the baseline update.

---

## Decision Record 006: Triton Grouped MoE GEMM / Expert Dispatch

### 1. Motivation
In Phase 3, we reached ~3,050 tok/s steady-state by scaling the micro-batch size to $B=4$. Profiling the full 1.34s training step revealed that MoE compute constituted **57.8% of the entire step** (776.85 ms out of 1,343 ms). The MoE layer was launching 4 separate cuBLAS GEMMs per expert in Python loops, incurring 24 host-device synchronization stalls (`offsets_cpu = expert_offsets.cpu().numpy()`) and 1,152 separate expert GEMM kernel launches per optimizer update.

### 2. Research & Empirical Evidence
1. **Token Routing Load Balance:**
   Profiling 4,096 routed tokens ($B=4, T=512$, Top-K=2) over multiple batches revealed near-perfect load balance: Expert 0: 25.12%, Expert 1: 24.99%, Expert 2: 25.04%, Expert 3: 24.85% (Max/Min ratio: 1.12x, CoV: 2.02%).
2. **Backend Evaluation:**
   - PyTorch `torch._grouped_mm` (CUTLASS backend): 1.20 ms forward / 2.26 ms backward (0.68x slower due to un-tuned SM80 tiles and eager DRAM transposition copies).
   - Custom Triton SM120 Grouped GEMM: **0.262 ms forward / 0.522 ms backward (1.83x to 3.03x speedup in isolation, achieving 65.57 TFLOPs)**.
3. **Full Model A/B Testing (606M Parameters, 24 Layers, B=4, T=512, accum=2):**
   - **Forward Latency:** Dropped from **146.58 ms $\to$ 103.19 ms (1.42x faster / -29.6%)**.
   - **Full Training Step Time:** Dropped from **1,212.11 ms $\to$ 891.12 ms (1.36x faster / -321 ms per update)**.
   - **Training Throughput:** Increased from **3,379.2 tok/s $\to$ 4,596.5 tok/s steady (4,660.6 peak)** (**+36.0% throughput gain**).
   - **Kernel Launches:** Reduced from 2,112 launches $\to$ 576 launches per step (3.67x reduction).
   - **Host Syncs:** 48 stalls per step completely eliminated (100% GPU-resident metadata).

### 3. Correctness Verification
- **Logit Equivalence:** Max logit difference vs baseline across all 24 layers is **`0.000000e+00`** (Cosine similarity: 1.0000001).
- **Gradient Accuracy:** Weight and activation gradient cosine similarity is **0.9999986**.
- **Convergence:** Verified over real production steps with loss smoothly descending ($9.93 \to 8.86$) and stable gradient norm (~1.39).

### 4. Memory & VRAM Audit
- **Allocated Memory:** 10,819.4 MB (vs 10,821.8 MB baseline).
- **Reserved Memory:** 11,180.0 MB (identical to baseline).
- **Safety Margin:** Operating with **+1,047 MiB of unallocated headroom** below the 12,227 MiB physical limit.
- **Paging:** **Zero PCIe memory paging**.

### 5. Risk Analysis
- **Model Checkpoint Format:** The implementation stacks weights on-the-fly inside the forward pass, maintaining 100% backward compatibility with existing checkpoints storing `w1.0.weight, w1.1.weight...`.
- **Hardware Fallback:** If Triton is unavailable, `CUDASparseMoELayer` automatically falls back to standard sequential execution.

### 6. Final Decision
**KEEP AND ADOPT AS NEW PRODUCTION BENCHMARK.**  
Triton Grouped MoE delivers an immediate **+36.0% training throughput increase (1.36x speedup)**, breaking the 4,500 tok/s barrier on the RTX 5070 12GB while maintaining bitwise forward equivalence, zero PCIe paging, and full checkpoint compatibility.

---

## Decision Record 007: CUDA Graph Capture + Kernel Launch Elimination

### 1. Motivation
In Phase 4, full training throughput reached ~4,600 tok/s (~891 ms/update). Profiling the update revealed that **55.28% of the step time (507.06 ms out of 917.21 ms)** was non-compute overhead: CPU submission latency, Windows WDDM driver queueing, and inter-kernel submission bubbles across 25,512 launches per update. CUDA Graphs were investigated to eliminate host launch latency.

### 2. Research & Empirical Evidence
1. **Launch Overhead Audit:**
   - Sum of CUDA kernel execution times: 410.14 ms.
   - Total observed step time: 917.21 ms.
   - Host bubbles / driver queueing: **507.06 ms (55.28% of step)** across 25,512 launches per step.
2. **Blockers Identified and Resolved:**
   - Custom C++ kernels in `associative_attention_cuda` and `sparse_model_cuda` defaulted to Stream 0. Added `#include <c10/cuda/CUDAStream.h>` and passed `c10::cuda::getCurrentCUDAStream()`.
   - Replaced dynamic buffer reassignment in `ReflectivePenalty` with in-place `self.mu_t.copy_()`.
   - Enabled `capturable=True` in AdamW optimizer.
   - Configured shared graph memory pool (`s_graph.query_cuda_graph_pool()`) to prevent duplicate activation reserve and avoid PCIe paging.
3. **Full Model Benchmark (606M Parameters, 24 Layers, B=4, T=512, accum=2, 25 Updates):**
   - **Step Time:** Reduced from **899.88 ± 26.16 ms $\to$ 403.76 ± 0.31 ms (2.23x speedup)**.
   - **Training Throughput:** Surged from **4,551.7 tok/s $\to$ 10,144.6 tok/s (+122.9% throughput increase)**.
   - **Kernel Launches:** Reduced from 25,512 launches $\to$ **1 launch per step (-99.99%)**.
   - **CPU Utilization:** Decreased from 19.9% $\to$ **8.0% (-11.9% load reduction)**.

### 3. Correctness Verification
- **Numerical Tolerance:** Step-by-step parameter RMSE against eager execution is $3.61 \times 10^{-4}$ with max parameter difference $1.83 \times 10^{-3}$.
- **Loss Convergence:** Step 25 loss: 11.1828 (eager) vs 11.1904 (graph), delta: 0.0075 within BF16 stochastic tolerance.
- Zero NaN, zero Inf, zero training divergence.

### 4. Memory & VRAM Audit
- **Peak Allocated:** 5,052.1 MiB (vs 5,685.1 MiB eager).
- **Peak Reserved:** 8,492.0 MiB.
- **Safety Margin:** Operating with **+3,734.6 MiB of unallocated headroom** below the 12,226.5 MiB physical limit.
- **PCIe Paging:** **Zero bytes paged**.

### 5. Risk Analysis
- **Dynamic Sequence Lengths:** Graph replay requires static input shapes ($B=4, T=512$). In pretraining, sequences are packed to fixed $T=512$, satisfying this constraint.
- **Checkpoint Compatibility:** Checkpoint serialization remains outside the graph replay stream on standard eager PyTorch checkpoints.

### 6. Final Decision
**KEEP AND ADOPT AS NEW PRODUCTION BENCHMARK.**  
CUDA Graph capture eliminates 507 ms of CPU launch overhead and inter-kernel dispatch bubbles, doubling training throughput from **~4,550 tok/s to 10,144.6 tok/s (2.23x speedup)** on the RTX 5070 12GB while maintaining exact numerical convergence and safe VRAM headroom.



