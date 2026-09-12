# Phase 5: CUDA Graph Capture + Kernel Launch Overhead Elimination

## Executive Summary

Phase 5 investigated whether CUDA Graphs could eliminate the substantial CPU launch latency and inter-kernel submission bubbles on the RTX 5070 12GB (Blackwell SM120) under Windows WDDM without breaking PyTorch gradient checkpointing, Triton Grouped MoE, autograd, recurrent attention states, or AdamW optimizer updates.

Prior to Phase 5, Phase 4 achieved ~4,600 tok/s steady-state (~891 ms/update). However, fine-grained profiling in Part A revealed that **55.28% of the training update (507.06 ms out of 917.21 ms)** was consumed by CPU kernel submission latency, driver dispatch overhead, and inter-kernel bubbles, with **25,512 kernel launches per optimizer update**.

By propagating CUDA streams through all custom C++ extensions (`associative_attention_cuda`, `sparse_model_cuda`), updating buffers in-place, enabling `capturable=True` in fused AdamW, and binding a coordinated PyTorch graph memory pool (`query_cuda_graph_pool()`), the entire $B=4, T=512, \text{accum}=2$ (4,096 tokens/update) training pipeline was captured in a **single CUDA Graph**.

### Core Results

| Metric | Eager Baseline (Phase 4) | CUDA Graph Replay (Phase 5) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Update Step Time** | 899.88 ± 26.16 ms | **403.76 ± 0.31 ms** | **-496.12 ms (2.23x speedup)** |
| **Steady Throughput** | 4,551.7 tok/s | **10,144.6 tok/s** | **+5,592.9 tok/s (+122.9%)** |
| **CUDA Kernel Launches** | 25,512 launches / update | **1 graph launch / update** | **-25,511 launches (-99.99%)** |
| **Peak Allocated VRAM** | 5,685.1 MiB | **5,052.1 MiB** | -633.0 MiB |
| **Peak Reserved VRAM** | 6,004.0 MiB | **8,492.0 MiB** | +2,488.0 MiB (Private pool reuse) |
| **VRAM Headroom** | 6,222.6 MiB | **3,734.6 MiB** | Safe cap: 12,226.5 MiB (0 paging) |
| **CPU Utilization** | 19.9% | **8.0%** | -11.9% CPU load reduction |
| **Loss Delta (Step 25)** | 11.1828 | 11.1904 | 0.0075 (BF16 stochastic match) |
| **Numerical Equivalence**| Reference | Verified | Max param diff: 1.8e-3, RMSE: 3.6e-4 |
| **Final Decision** | — | **KEEP** | **Exceeds >=5% threshold by 24x** |

---

## Part A — Launch Overhead Profiling

A detailed audit of a full $B=4, T=512, \text{accum}=2$ optimizer update step was performed using CUDA events and PyTorch Profiler:

1. **Sum of all CUDA Kernel Execution Times:** 410.14 ms
2. **Observed GPU Wall Clock Time:** 917.21 ms
3. **CPU Submission Time:** 901.97 ms
4. **Inter-Kernel Bubbles / Driver Overhead:** **507.06 ms (55.28% of total step)**
5. **Total CUDA Kernel Launches:** **25,512 launches / update**

### Analysis
On Windows WDDM, kernel dispatch overhead through the DirectX Graphics Kernel (DXGK) layer imposes a 15–20 µs latency per kernel launch. Because the 24-layer Jarvis model features recurrent associative attention, ternary STE operations, Triton Grouped MoE, and gradient checkpointing, each accumulation microstep involves over 12,000 separate kernel dispatches (forward + backward). The GPU ran out of queued work repeatedly, creating 507 ms of dead air (bubbles). This conclusively satisfied the Part A condition (>2% launch overhead), justifying full CUDA Graph capture.

---

## Part B — CUDA Graph Feasibility & Blockers Resolved

Building an end-to-end captured graph required identifying and resolving several critical system blockers:

### Blocker 1: Missing CUDA Stream Propagation in Custom C++ Extensions
- **Problem:** In `associative_attention_cuda/associative_attention_cuda.cu` (4 kernels) and `sparse_model_cuda/sparse_model.cu` (6 kernels), kernel launches used `<<<blocks, threads_per_block>>>` without specifying a stream. In PyTorch extensions, unassigned kernel launches default to Stream 0. Capturing on a side stream while kernels hit Stream 0 violates CUDA graph invariants and caused `cudaErrorIllegalAddress` / stream capture illegal memory faults.
- **Resolution:** Added `#include <c10/cuda/CUDAStream.h>` and passed `c10::cuda::getCurrentCUDAStream()` into every `<<<grid, block, smem, stream>>>` invocation in both extensions, followed by full recompilation.

### Blocker 2: Buffer Re-allocation and Host Copy
- **Problem:**
  1. `ReflectivePenalty.mu_t` was reassigned as `self.mu_t = self.ema_decay * self.mu_t + ...`, creating a new tensor address outside the graph memory pool.
  2. `Jarvis.forward` instantiated `torch.tensor(0.0, device=idx.device)` on each call, which PyTorch attempted to synchronize from CPU.
- **Resolution:** Replaced `mu_t` assignment with in-place `self.mu_t.copy_(...)`, and replaced `torch.tensor(0.0)` with `torch.zeros((), device=idx.device, dtype=torch.float32)`.

### Blocker 3: AdamW Capturable Requirement
- **Problem:** Standard AdamW raised `Attempting CUDA graph capture of step() for an instance of AdamW but param_groups' capturable is False`.
- **Resolution:** Configured `torch.optim.AdamW(model.parameters(), lr=1.5e-4, fused=True, capturable=True)`.

### Blocker 4: Memory Pool Isolation & WDDM Paging
- **Problem:** A naive `CUDAGraph()` allocates a disjoint memory pool. In early tests, this duplicated activation reserves to 11,934 MiB, triggering Windows WDDM PCIe paging that degraded replay time to 2,400 ms.
- **Resolution:** Shared the warmup stream's memory pool with `pool = s_graph.query_cuda_graph_pool()`. This kept peak reserved VRAM at a stable **8,492.0 MiB**, with **3,734.6 MiB of headroom** and zero paging.

---

## Part C — Gradient Checkpointing Compatibility

PyTorch non-reentrant gradient checkpointing (`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`) was evaluated across all 24 layers inside the CUDA Graph capture stream.
- **Result:** **100% Compatible**.
- Non-reentrant checkpointing utilizes standard PyTorch autograd engine hooks without invoking CPU-GPU synchronizations or creating dynamic graph breaks.
- Recomputation forward and backward passes executed cleanly inside the captured graph stream.

---

## Part D — Triton Grouped MoE Compatibility

The Triton Grouped MoE kernels integrated in Phase 4 were audited during graph capture:
- Triton forward grouped GEMM (`_grouped_gemm_fwd_kernel`): Passed.
- Triton backward activation GEMM (`_grouped_gemm_bwd_kernel`): Passed.
- Triton backward weight GEMM (`_grouped_gemm_weight_kernel`): Passed.
- Ternary STE autograd functions: Passed.
- Expert prefix sums and routing maps: Metadata kernels execute with deterministic FIFO ordering on the capture stream; buffers remain static.

---

## Part E — Correctness Verification

Three consecutive production updates (4,096 tokens/step) were executed on identical batches comparing eager execution against CUDA Graph replay:

| Step | Max Parameter Difference | Root-Mean-Square Error (RMSE) | Status |
| :---: | :---: | :---: | :---: |
| **Step 1** | $1.0986 \times 10^{-3}$ | $2.7109 \times 10^{-4}$ | PASS |
| **Step 2** | $1.4648 \times 10^{-3}$ | $3.1878 \times 10^{-4}$ | PASS |
| **Step 3** | $1.8311 \times 10^{-3}$ | $3.6103 \times 10^{-4}$ | PASS |

After 25 full optimizer updates:
- Eager loss: **11.1828**
- CUDA Graph loss: **11.1904**
- Absolute delta: **0.0075** (within standard BF16 non-associative reduction tolerance).
- Zero NaN, zero Inf, zero training divergence.

---

## Part F — Performance & Benchmarking

25 steady-state optimizer updates (2 accumulation microsteps each, $B=4, T=512$, AdamW update, grad clip 1.0) were measured on identical pseudorandom seeds:

- **Current Eager Execution:**
  - Mean step time: **899.88 ± 26.16 ms**
  - Steady throughput: **4,551.7 tok/s**
  - Jitter: High variance (std 26.16 ms) due to OS/driver thread scheduling.
- **CUDA Graph Replay:**
  - Mean step time: **403.76 ± 0.31 ms**
  - Steady throughput: **10,144.6 tok/s**
  - Jitter: Virtually zero variance (std 0.31 ms) due to elimination of host thread interactions.
- **Speedup:** **2.23x (+122.9% throughput increase)**.

---

## Part G — VRAM & Headroom Audit

- **Physical GPU VRAM Cap:** 12,226.56 MiB (RTX 5070 12GB).
- **Eager Peak Allocated:** 5,685.1 MiB | **Peak Reserved:** 6,004.0 MiB.
- **CUDA Graph Peak Allocated:** 5,052.1 MiB | **Peak Reserved:** 8,492.0 MiB.
- **VRAM Delta (Reserved):** +2,488.0 MiB for pre-allocated static graph execution pool.
- **Remaining Unallocated Headroom:** **3,734.6 MiB**.
- **PCIe Paging:** **ZERO bytes paged**. The allocation is safely below the 11.5 GB threshold where Windows WDDM initiates shared system memory spillover.

---

## Part H — Decision

**DECISION: KEEP AND ADOPT AS PRODUCTION ENGINE.**
- Threshold for KEEP was $\ge 5\%$.
- Actual measured throughput increase is **+122.9% (2.23x speedup)**, breaking the 10,000 tok/s threshold (**10,144.6 tok/s**).
- All architectural invariants, ternary representations, checkpointing, and numerical stability are fully preserved.

---

## Part I — Profiling the Next Bottleneck

With CPU launch overhead completely eliminated, PyTorch Profiler was executed on the CUDA Graph replay to audit the true raw GPU hardware execution breakdown (total kernel time = **401.56 ms**):

| Rank | Kernel Category | CUDA Time (ms) | % of Step | Primary Operations |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Dense Attention & Output Projections** | ~91.0 ms | 22.7% | CUTLASS TensorOp BF16 GEMMs (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `lm_head`) |
| **2** | **Elementwise STE & Activation Kernels** | ~118.0 ms | 29.4% | Weight quantization, clamping, ELU+1, rotary embeddings, residual adds, RMSNorm |
| **3** | **Triton MoE Forward GEMM** | 64.57 ms | 16.1% | `_grouped_gemm_fwd_kernel` across 4 experts |
| **4** | **Triton MoE Weight Backward GEMM** | 23.26 ms | 5.8% | `_grouped_gemm_weight_kernel` (dW computation) |
| **5** | **Liquid State Fusion & Attention Recurrence**| ~20.0 ms | 5.0% | `liquid_state_fusion_forward/backward` and recurrent chunk scans |
| **6** | **Fused AdamW Optimizer Update** | 13.30 ms | 3.3% | Fused AdamW kernel across all model parameters |
| **7** | **MoE Metadata & Routing** | 9.86 ms | 2.5% | `moe_compute_metadata_kernel` |

### Primary Remaining Measured Bottleneck:
**Elementwise / Quantization Overhead (~29.4%) and Dense Projection GEMMs (~22.7%)**:
Now that launch overhead is 0, the dominant runtime is raw Tensor Core GEMMs (attention projections + MoE) and the memory-bound elementwise passes (ternary STE weight quantization and RMSNorm). Future work can explore fusing ternary quantization with GEMM input loading or fusing RMSNorm with linear projections.
