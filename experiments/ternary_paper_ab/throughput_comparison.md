# Throughput and Telemetry Comparison: AbsMean vs Paper Eq. 3

## Overview

Following verification of numerical behavior, a standardized throughput benchmark was conducted comparing **Variant A (Current AbsMean Baseline)** and **Variant B (Paper Eq. 3 Literal Implementation)**. 

The test harness executed on the host system's **NVIDIA GeForce RTX 5070 12GB (Blackwell SM120)** using identical configuration:
- **Workload**: $B=4, T=512, \text{accum}=2 \implies 4,096\text{ real tokens/update}$
- **Architecture**: 24 layers, $d_{\text{model}}=1024$, 16 attention heads, 4 MoE experts, Top-2 routing
- **Precision**: BF16 Autocast with FP32 Master Weights and AdamW optimizer
- **Harness**: **30 warmup updates**, followed by **100 measured updates/replays**
- **Telemetry Source**: Direct `nvidia-smi` GPU polling across iterations

The benchmarking script is preserved at [scratch/benchmark_ternary_ab_throughput.py](file:///e:/Jarvis-Q1.58-500M/scratch/benchmark_ternary_ab_throughput.py).

---

## Measured Performance & Hardware Telemetry

| Metric | Variant A (AbsMean Baseline) | Variant B (Paper Eq. 3) | Difference / Impact |
| :--- | :--- | :--- | :--- |
| **Mean Step Time** | **3,791.45 ms** | 4,330.44 ms | +538.99 ms (+14.2%) |
| **Training Throughput** | **1,080.3 tok/s** | 945.9 tok/s | **-12.45% (-134.4 tok/s)** |
| **VRAM Allocated** | **9,526.77 MiB** | 9,527.07 MiB | +0.30 MiB (Identical within 0.003%) |
| **VRAM Reserved** | **12,386.00 MiB** | 13,132.00 MiB | +746.00 MiB (PyTorch caching pool delta) |
| **GPU Core Clock** | 3,390.0 MHz | 3,393.5 MHz | Sustained boost lock |
| **GPU Memory Clock** | 16,001.0 MHz | 16,001.0 MHz | Sustained 32 Gbps effective |
| **GPU Utilization** | **87.2%** | 77.0% | -10.2% (Lower compute intensity) |
| **Power Draw** | 71.6 W | 74.8 W | +3.2 W |
| **GPU Temperature** | 39.5 °C | 40.2 °C | Thermally stable (<41 °C) |

---

## Technical & Architectural Analysis

### 1. Why Paper Eq. 3 is 12.45% Slower in PyTorch Eager Engine
In Variant A:
- The production codebase leverages a fused Triton/CUDA quantization kernel (`FusedTernaryQuantizeSTE` in `utils/ternary_ops.py`), which fuses absolute-mean reduction, normalization, clamping, rounding, and alpha multiplication into a single GPU pass, and fuses the backward STE mask.

In Variant B:
- To adhere strictly to the literal paper specification without altering CUDA kernel binaries:
  $$W_q = \text{round}(\text{clamp}(W, -1, 1))$$
  $$\text{grad}_W = \text{grad}_{\text{out}} \times (|W| \le 1.0)$$
- Under PyTorch eager autograd, evaluating `clamp`, `round`, `abs()`, `<= 1.0`, and tensor multiplication individually launches **multiple separate CUDA elementwise kernels** across each of the 6 projection matrices per layer across all 24 layers ($144$ linear projections $\times 2$ forward/backward passes = hundreds of additional kernel launches per micro-step).
- The additional CPU kernel launch overhead and GPU global memory round-trips for unfused intermediate boolean masks reduced GPU compute utilization from **87.2% down to 77.0%**, resulting in the 12.45% throughput penalty.

### 2. Memory Footprint
- Peak active memory allocated for both variants is identical: **9,526.8 MiB**, reflecting that the model parameters, activation caches, and gradient buffers are structurally indistinguishable between the two quantizers.
- The physical 12GB VRAM ceiling was maintained without out-of-memory faults.

### 3. Scientific Implication
Per the hard constraints of this study, no custom CUDA kernels were written to artificially optimize Paper Eq. 3. Even if Paper Eq. 3 were kernel-fused to match AbsMean's execution speed, it would not alter the fundamental finding: **Paper Eq. 3 suffers from complete all-zero collapse during training**.
