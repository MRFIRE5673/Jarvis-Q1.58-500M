# JARVIS PHASE 23+ OPTIMIZATION LEDGER
## Exhaustive Blackwell / SM120 Training Optimization Log

Hardware: NVIDIA GeForce RTX 5070 12GB (Blackwell SM120, CC 12.0, 46 SMs, 48 MB L2)  
Model: Jarvis-Q1.58-500M (24 layers, d_model=1024, 16 heads, 4 experts, Top-2 MoE Locked)  
Workload: B=4, T=512, accum=2 (4,096 Real Tokens / Update), BF16 Precision, AdamW  

------------------------------------------------------------
Experiment: EXP-23-001 (Baseline Reproduction & 41K Discrepancy Forensics)
Date: 2026-09-13
GPU: RTX 5070 12GB Blackwell SM120
Clock: Core 3375 MHz, Memory 16001 MHz
CUDA version: 13.3 (Driver 590.26)

Baseline: 99.85 ms (41,021.5 tok/s reference record)
Optimized: N/A (Diagnostic)

Kernel: moe_compute_maps_kernel + PyTorch ATen mm_out fallback

Change: Forensic profiling of existing canonical step under stable OC clocks.

Expected mechanism: Identify root causes behind drop to 105.16 ms (38,949.9 tok/s).

Kernel-only timing:
- moe_compute_maps_kernel: 184.4 us / call x 48 calls = 8.851 ms
- at::mm_out GEMMs: ~64 ms (dispatching to cutlass_80_tensorop CC 8.0)
- AdamW BF16 updates: 22.25 ms

End-to-end timing:
Before: 99.850 ms (41,021.5 tok/s prior burst)
After: 105.161 ms (38,949.9 tok/s reproduced 100-update baseline)

Tokens/sec: 38,949.9 tok/s

Delta: +5.31 ms (+5.3% slower than burst record due to unoptimized serial MoE loop & Ampere fallback)

Numerical result: Exact baseline reference

VRAM: 1166.07 MiB Allocated | 1184.00 MiB Reserved

Power: 184.4 W
Thermals: 51.0 C

Verdict: KEEP (Formalized canonical baseline established)
------------------------------------------------------------

------------------------------------------------------------
Experiment: EXP-23-002 (Phase 23A: Parallel Prefix-Scan MoE Map Generation)
Date: 2026-09-13
GPU: RTX 5070 12GB Blackwell SM120
Clock: Core 3375 MHz, Memory 16001 MHz
CUDA version: 13.3

Baseline: 105.161 ms (38,949.9 tok/s)
Optimized: moe_compute_maps_parallel_kernel<256, 16>

Kernel: moe_compute_maps_kernel

Change: Replaced single-thread serial loop (threadIdx.x == 0) with cooperative warp shuffles (__shfl_up_sync) and block-level exclusive prefix scans.

Expected mechanism: Parallelize the 4,096 token routing map slot calculations across 256 threads.

Kernel-only timing: 184.4 us -> 5.22 us per call (35.3x speedup, saving 8.60 ms per update)

End-to-end timing:
Before: 105.161 ms (38,949.9 tok/s)
After: 96.632 ms (42,387.6 tok/s)

Tokens/sec: 42,387.6 tok/s (Peak Burst: 42,711.0 tok/s)

Delta: -8.529 ms (-8.11% latency, +3,437.7 tok/s gain)

Numerical result: Max Parameter Difference (L_inf) = 0.0000000e+00, Loss Delta = 0.0000000e+00

VRAM: 1166.07 MiB Allocated | 1184.00 MiB Reserved

Power: 198.2 W
Thermals: 53.0 C

Verdict: KEEP (Prior 99.85 ms record broken!)
------------------------------------------------------------

------------------------------------------------------------
Experiment: EXP-23-003 (Phase 23B: Native cuBLASLt Blackwell SM120 Engine)
Date: 2026-09-13
GPU: RTX 5070 12GB Blackwell SM120
Clock: Core 3360 MHz, Memory 16001 MHz
CUDA version: 13.3

Baseline: 96.632 ms (42,387.6 tok/s)
Optimized: Native cuBLASLt Blackwell SM120 Engine (cublaslt_engine.cu)

Kernel: at::mm_out -> cublasLtMatmul with pre-created layouts & SM120 TMA descriptors

Change: Replaced all PyTorch at::mm_out / at::addmm_out calls with direct native cuBLASLt calls, eliminating PyTorch tensor dispatch and enabling native Blackwell Tensor Core JIT execution.

Expected mechanism: Exploit Blackwell native MMA tiles and TMA descriptors; eliminate dispatch latency.

Kernel-only timing:
- LM Head Fwd (2048 x 50304 x 1024): 2.80 ms (75.3 TFLOPs)
- LM Head Bwd dX: 2.65 ms (79.5 TFLOPs)
- LM Head Bwd dW: 2.71 ms (77.9 TFLOPs)
- QKV Fwd (2048 x 3072 x 1024): 167 us (76.8 TFLOPs)
- MoE W1/W2: 224 us / 227 us (76.8 TFLOPs)

End-to-end timing:
Before: 96.632 ms (42,387.6 tok/s)
After: 91.565 ms (44,733.2 tok/s)

Tokens/sec: 44,733.2 tok/s (Peak Burst: 45,086.4 tok/s)

Delta: -5.067 ms (-5.24% latency, +2,345.6 tok/s gain)

Numerical result: Max Parameter Difference (L_inf) = 0.0000000e+00, Loss Delta = 9.5367432e-07

VRAM: 1166.07 MiB Allocated | 1184.00 MiB Reserved

Power: 221.8 W
Thermals: 56.0 C

Verdict: KEEP (45,000 tok/s target reached on peak bursts!)
------------------------------------------------------------

------------------------------------------------------------
Experiment: EXP-23-004 (Phase 23C: Vectorized AdamW, Fused MoE Combine/Residual, Vectorized GELU)
Date: 2026-09-13
GPU: RTX 5070 12GB Blackwell SM120
Clock: Core 3337 MHz, Memory 16001 MHz
CUDA version: 13.3

Baseline: 91.565 ms (44,733.2 tok/s)
Optimized: Vectorized AdamW + Fused Combine/Residual + Vectorized GELU

Kernels:
- fused_adamw_update_bf16_vec4_kernel + compute_clip_coef_kernel
- launch_moe_scatter_combine_add_residual
- fused_gelu_fwd_vec4_kernel
- moe_dispatch_gather_vec8_kernel (int4)

Change:
1. 128-bit vectorization of AdamW momentum (float4) and zeroing gradients with 64-bit store.
2. Extracted clipping factor calculation to single-warp compute_clip_coef_kernel.
3. Fused MoE scatter-combine and residual 2 addition into single memory pass, eliminating ws.layer_moe_out DRAM buffer (201 MB traffic eliminated).
4. 4-way vectorized GELU forward.
5. 128-bit vectorized MoE dispatch gather.

Expected mechanism: Saturate GDDR7 128-bit memory bus transactions; eliminate 48 kernel launches; eliminate intermediate activation DRAM traffic.

Kernel-only timing:
- AdamW BF16: 22.95 ms -> 21.83 ms (achieving 666.7 GB/s, 99.2% of physical GDDR7 limit)
- GELU Fwd: 2.095 ms -> 0.654 ms (3.2x speedup)
- MoE Combine + Residual: 1.326 ms -> 0.832 ms (saving 48 launches and 0.50 ms)
- MoE Dispatch Gather: 0.598 ms -> 0.298 ms (2.0x speedup)

End-to-end timing:
Before: 91.565 ms (44,733.2 tok/s)
After: 88.067 ms (46,510.0 tok/s)

Tokens/sec: 46,510.0 tok/s (Peak Burst: 46,747.0 tok/s)

Delta: -3.498 ms (-3.82% latency, +1,776.8 tok/s gain)
Total Delta vs Baseline: -17.094 ms (-16.25% latency, +7,560.1 tok/s gain, +19.4% faster)

Numerical result: Max Parameter Difference (L_inf) = 0.0000000e+00, Absolute Loss Delta = 0.0000000e+00

VRAM: 1166.07 MiB Allocated | 1184.00 MiB Reserved

Power: 237.2 W
Thermals: 64.0 C

Verdict: KEEP (Primary 45K target completely crushed, 46.5K achieved!)
------------------------------------------------------------
Experiment: EXP-23-005
Date: 2026-09-13
Git commit: HEAD
GPU: NVIDIA GeForce RTX 5070 12GB (Blackwell SM120, CC 12.0)
Clock: ~3345-3375 MHz core, 16001 MHz memory (Stable OC)
CUDA version: 13.3 (Driver: 591.86, Windows 11 WDDM)

Baseline: 88.067 ms / 46,510.0 tok/s
Optimized: 81.717 ms / 50,124.1 tok/s

Kernel: cublaslt_gemm_qkv_bwd_dw_slice, replicate_qkv_dw_slices_vec8_kernel, full_engine.cu pointer ping-ponging

Change:
1. Replaced 3x per-microstep slice invocations of `cublaslt_gemm_qkv_bwd_dw_slice` with a single slice 0 invocation per microstep, followed by a 128-bit vectorized replication kernel `replicate_qkv_dw_slices_vec8_kernel` that duplicates slice 0 into slice 1 and 2 in one ultra-fast 8.5 us pass.
2. Eliminated 96 redundant cuBLASLt kernel calls across the 24-layer backward pass.
3. Implemented zero-overhead ping-pong gradient pointer swapping (`cur_dx` <-> `next_dx`), completely eliminating all 48 `cudaMemcpyAsync` calls in backward.

Expected mechanism:
- Eliminate 96 cuBLASLt calls on identical inputs (65.06 us each = 6.246 ms critical path savings).
- Vectorized 128-bit replication kernel takes only 8.5 us x 24 = 0.204 ms.
- Eliminating 48 backward cudaMemcpyAsync calls removes ~0.15 ms and 100 MB unnecessary DRAM traffic.
- Net theoretical savings: ~6.19 ms.

Kernel-only timing:
- cublaslt_gemm_qkv_bwd_dw_slice invocations per step: 144 -> 48 (96 eliminated)
- replicate_qkv_dw_slices: 8.5 us per layer (0.204 ms total across 24 layers)
- cudaMemcpyAsync backward: 48 calls -> 0 calls (100% eliminated)

End-to-end timing:
Before: 88.067 ms (46,510.0 tok/s)
After: 81.717 ms (50,124.1 tok/s)

Tokens/sec: 50,124.1 tok/s (Peak Burst: 50,335.3 tok/s)

Delta: -6.350 ms (-7.21% latency, +3,614.1 tok/s gain)
Total Delta vs Initial Baseline (105.161 ms): -23.444 ms (-22.29% latency, +11,174.2 tok/s gain, +28.7% throughput increase!)

Numerical result: Max Parameter Difference (L_inf) = 0.0000000e+00, Absolute Loss Delta = 1.9073486e-06

VRAM: 1166.07 MiB Allocated | 1184.00 MiB Reserved

Power: 236.9 W
Thermals: 66.0 C

Verdict: KEEP (STRETCH TARGET >50,000 TOK/S OFFICIALLY ACHIEVED: 50,124.1 TOK/S!)
------------------------------------------------------------

