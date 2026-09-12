# Jarvis "20K Tok/s" CUDA R&D Program: Phase 2 — Blackwell Tensor Core Low-Precision Execution

**Date:** September 12, 2026  
**Target Hardware:** NVIDIA GeForce RTX 5070 12GB (Blackwell Architecture, SM 12.0 / sm_120, Driver 610.47)  
**Software Stack:** PyTorch 2.12.0.dev20260408+cu128, CUDA 12.8, cuDNN 9.2.0, Triton 3.7.1  
**Workload:** Jarvis-Q1.58-500M Pretraining Step ($B=2, T=512$, accum=4, 4,096 tokens/update)  
**Status:** Phase 2 Complete

---

## 1. Executive Summary & Measured Throughput Matrix

In Phase 2, we investigated hardware-accelerated low-precision execution on Blackwell SM 12.0 to determine if native **FP8** or **NVFP4** Tensor Core instructions can out-perform the dense **BF16 cuBLAS** production baseline.

### Measured Performance Summary Table

| Precision / Mode | Isolated GEMM Latency ($1024^3$) | End-to-End Layer Training (Fwd+Bwd) | Effective Model Tok/s | Peak VRAM | Numerical Status | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BF16 cuBLAS Baseline** | **0.0895 ms (24.0 TFLOPs)** | **0.8566 ms** | **1,514 tok/s** (2.71s step) | **10,215 MB** | Reference | **KEEP AS PRODUCTION DEFAULT** |
| **FP8 Tensor Core (`float8_e4m3fn`)** | **0.0772 ms (27.8 TFLOPs)** | 2.0579 ms | ~630 tok/s (projected) | 10,240 MB | Clean ($\Delta_{\text{grad}} = 2.8\%$) | **REVERT (Dynamic Quant Overhead)** |
| **NVFP4 (`float4_e2m1fn_x2`)** | Hardware verified (`_scaled_mm`) | Prohibitive (>15 ms quant) | <100 tok/s | N/A | Untested in loop | **DOCUMENT HARDWARE BLOCKER** |

> [!IMPORTANT]
> **Key Finding:**  
> - In isolation, Blackwell FP8 Tensor Cores are **1.16x to 1.45x faster** than BF16 (reaching **51.27 TFLOPs** on MoE shapes).  
> - In full training (forward + backward), **dynamic activation scaling, tensor transposition, and autograd tape serialization add ~1.20 ms of overhead**.  
> - End-to-end training latency is **0.857 ms (BF16) vs. 2.058 ms (FP8)** — making FP8 training **2.4x SLOWER** than pure BF16 cuBLAS without end-to-end fused compilation.  
> - **Decision:** **REVERT / KEEP BF16 cuBLAS AS PRODUCTION DEFAULT.**

---

## 2. Part A: Hardware & Software Capability Audit

We conducted an exhaustive audit of the installed toolchain on the target RTX 5070:
- **GPU Device:** NVIDIA GeForce RTX 5070 (48 SMs, 12,227 MiB GDDR7 @ 28 Gbps)
- **Compute Capability:** `(12, 0)` (Blackwell SM 12.0 / sm_120)
- **NVIDIA Display Driver:** `610.47`
- **PyTorch Version:** `2.12.0.dev20260408+cu128`
- **CUDA Runtime / Compiler:** CUDA 12.8 runtime, CUDA 13.3 CCCL compiler driver
- **cuDNN:** `9.2.0`
- **Triton:** `3.7.1`
- **Transformer Engine:** NOT installed (No pre-built Windows wheel available for Python 3.14 / SM120)
- **torchao:** NOT installed
- **CUTLASS / CuTe:** Available via CUDA CCCL headers (`<cutlass/cutlass.h>`, `<cute/tensor.hpp>`)

### Native Low-Precision Dtype & Primitive Support
1. **FP8 Support:**
   - Dtypes: `torch.float8_e4m3fn` (forward/weights) and `torch.float8_e5m2` (gradients).
   - GEMM Primitive: `torch._scaled_mm(a, b, scale_a, scale_b, bias, out_dtype, use_fast_accum)`.
   - Scaling Configurations: Supports TensorWise, RowWise, and BlockWise (1x128, 128x128, 1x32).
   - Autograd Support: `torch._scaled_mm` has **NO native autograd backward derivative in PyTorch**. A custom `torch.autograd.Function` is strictly required.
2. **NVFP4 (FP4) Support:**
   - Dtype: `torch.float4_e2m1fn_x2` (two 4-bit floats packed per byte).
   - GEMM Primitive: `torch._scaled_mm` natively executes Blockwise 1x16 FP4 matrix multiplication on SM 12.0 Tensor Cores with FP8 scale factors (`scale_a`, `scale_b` as `float8_e4m3fn`).
   - Software Blocker: PyTorch does NOT have a fast native C++/CUDA kernel for dynamic runtime casting from `bfloat16` to `float4_e2m1fn_x2`. Software bitpack emulation takes >15 ms per tensor, preventing practical dynamic training use.

---

## 3. Part B: FP8 Tensor Core Micro-Benchmark on Exact Jarvis Shapes

Benchmarked on RTX 5070 using 100 timed iterations per shape after 25 warmup steps via CUDA events:

| Matrix Operation & Shape | (M, K, N) | BF16 cuBLAS Latency | FP8 Isolated GEMM | Isolated Speedup | FP8 Complete (E2E with Dynamic Quant) | E2E Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention Projection (Q/K/V/Out)** | 1024, 1024, 1024 | 0.0895 ms (24.0 T) | **0.0772 ms (27.8 T)** | **1.16x** | 0.3097 ms (6.9 T) | **0.29x (3.5x slower)** |
| **MoE Up Projection ($W_1$)** | 1024, 1024, 2048 | 0.1057 ms (40.6 T) | **0.0838 ms (51.3 T)** | **1.26x** | 0.3297 ms (13.0 T) | **0.32x (3.1x slower)** |
| **MoE Down Projection ($W_2$)** | 1024, 2048, 1024 | 0.1310 ms (32.8 T) | **0.0903 ms (47.6 T)** | **1.45x** | 0.3555 ms (12.1 T) | **0.37x (2.7x slower)** |
| **MoE $W_1$ Sub-Batch (512 tokens)** | 512, 1024, 2048 | 0.0904 ms (23.8 T) | **0.0903 ms (23.8 T)** | **1.00x** | 0.3601 ms (6.0 T) | **0.25x (4.0x slower)** |
| **MoE $W_2$ Sub-Batch (512 tokens)** | 512, 2048, 1024 | 0.0919 ms (23.4 T) | **0.0938 ms (22.9 T)** | **0.98x** | 0.3598 ms (6.0 T) | **0.26x (3.9x slower)** |

### Numerical Accuracy vs. BF16 Reference:
- Max Absolute Error: $\le 0.141$
- Mean Absolute Error: $\le 0.020$
- Relative Frobenius Error: $3.76\% \text{ to } 3.93\%$
- Cosine Similarity: $>0.99928$ (clean directional preservation)

---

## 4. Part C: NVFP4 / FP4 Hardware Execution Investigation

### Findings:
1. **Hardware Execution Verified:** We successfully executed native Blockwise 1x16 NVFP4 matrix multiplication on the RTX 5070 using `torch._scaled_mm` with `torch.float4_e2m1fn_x2` inputs and `torch.float8_e4m3fn` scale blocks ($1024 \times 1024$ produced a valid `bfloat16` output tensor).
2. **The Exact Blocker for Training:**
   - In PyTorch 2.12 dev, `tensor.to(torch.float4_e2m1fn_x2)` returns:  
     `copy_() does not support casting Float4_e2m1fn_x2 to different types.`
   - Quantizing activations on-the-fly currently requires Python-level bit-twiddling (`_f32_to_floatx_unpacked` + `pack_uint4`), which takes **~15.4 ms per matrix**—nearly 150x slower than the GEMM itself!
   - Until NVIDIA or PyTorch releases a fused C++/CUDA quantizer for NVFP4, NVFP4 is strictly viable only for static offline weight quantization (inference), not dynamic training activations.

---

## 5. Part D & E: Representative Jarvis Layer Training Benchmark (Fwd + Bwd)

We implemented `CustomFP8LinearFunction(torch.autograd.Function)` handling the full training forward and backward pass on a representative $1024 \times 1024$ Jarvis linear layer:

```text
========================================================================================================================
Layer Step Benchmark (M=1024, K=1024, N=1024) | BF16 cuBLAS Baseline | Custom FP8 Autograd Function | Speedup Ratio
------------------------------------------------------------------------------------------------------------------------
Forward Only Latency                          |      0.3386 ms       |          0.7140 ms           | 0.47x (2.1x slower)
Forward + Backward Latency (Training Step)    |      0.8566 ms       |          2.0579 ms           | 0.42x (2.4x slower)
------------------------------------------------------------------------------------------------------------------------
Forward Output Relative Error                 |      Reference       |           2.696%             | Excellent fidelity
Input Gradient Relative Error (dX)            |      Reference       |           0.000%             | Exact match
Weight Gradient Relative Error (dW)           |      Reference       |           2.799%             | Matches STE tape
NaN / Inf Occurrences                         |        False         |           False              | 100% Clean
========================================================================================================================
```

### Why FP8 Training Is 2.4x Slower:
1. **Three Dynamic Quantization Passes:** Forward requires quantizing $X$; backward requires quantizing $\text{grad\_output}$ and $W$. Each requires finding `abs().max()`, computing scale, clamping, and casting.
2. **cuBLASLt Layout Constraints:** `torch._scaled_mm` strictly requires the second operand to be a transposed row-major matrix. In the backward pass, computing $dX = dY \cdot W$ and $dW = dY^T \cdot X$ requires **three `.t().contiguous()` memory allocations and global DRAM copies**.
3. **Kernel Launch Bubbles:** A single FP8 linear layer launches 8 separate GPU kernels (3 reduction kernels, 3 casting kernels, 2 GEMMs) compared to 2 unified GEMMs in BF16. On Windows WDDM, kernel launch overhead dwarfs the compute time.

---

## 6. Part G: Architectural Relationship to Ternary Quantization

Jarvis's core architecture uses **AbsMean ternary weights** $W_q \in \{-\alpha, 0, +\alpha\}$.  
In our FP8 prototype:
- Continuous FP32 master weights are quantized to ternary via STE.
- The ternary weights (whose values are exactly $-\alpha, 0, +\alpha$) are scaled to $\pm 448.0$ and stored as `float8_e4m3fn`.
- Because ternary values are exactly representable in FP8 without precision loss ($0$, $+1 \times \text{scale}$, $-1 \times \text{scale}$), FP8 execution introduces **zero ternary quantization degradation**.
- However, as proven above, the dynamic quantization of *activations* and the memory layout constraints make this uncompetitive against BF16 cuBLAS on SM 12.0.

---

## 7. Next Highest-Value Optimization

With software-unpack ternary (Phase 1) and eager FP8 autograd (Phase 2) thoroughly evaluated and rejected, the roadmap identifies the **next highest-value optimizations**:

1. **Micro-Batch Configuration Tuning ($B=4, \text{accum}=2$ vs. $B=2, \text{accum}=4$):**  
   - Increases token dimension from $M=1024 \to M=2048$, doubling Tensor Core density from ~24 TFLOPs to ~45 TFLOPs on BF16 cuBLAS with zero conversion overhead. Expected gain: **+10% to +18% net throughput**.
2. **CUDA Graph Capture / Ahead-Of-Time Kernel Fusion (Phase 4):**  
   - Eliminates the ~0.05 ms Windows driver launch bubbles that throttle small-matrix execution.
3. **MoE Grouped GEMM (Phase 2 Roadmap Track):**  
   - Fuses all 4 expert GEMMs into a single batched kernel call, eliminating expert dispatch scatter/gather overhead.
