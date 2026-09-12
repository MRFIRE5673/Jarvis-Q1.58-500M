# Jarvis Training Throughput Optimization Suite

Welcome to the Jarvis-Q1.58-500M Throughput Optimization Suite. This directory contains benchmarks, profiling tools, decision logs, and roadmaps dedicated to maximizing training throughput on the NVIDIA GeForce RTX 5070 12GB.

---

## Directory Overview

- [`production_vs_fast_path.md`](file:///e:/Jarvis-Q1.58-500M/experiments/throughput_optimization/production_vs_fast_path.md): In-depth root-cause investigation explaining why the initial 50M production run operated at ~374 tok/s instead of >740 tok/s, detailing the attention fallback and WDDM memory paging findings.
- [`optimization_benchmark.md`](file:///e:/Jarvis-Q1.58-500M/experiments/throughput_optimization/optimization_benchmark.md): Comprehensive benchmark results, component latency breakdowns, and before/after measurements across attention, MoE, LSF, dataloading, and optimizer execution.
- [`bottleneck_history.md`](file:///e:/Jarvis-Q1.58-500M/experiments/throughput_optimization/bottleneck_history.md): Historical tracking table recording each optimization iteration, throughput progression, VRAM profile, and keep/revert decisions.
- [`optimization_decisions.md`](file:///e:/Jarvis-Q1.58-500M/experiments/throughput_optimization/optimization_decisions.md): Formal decision records detailing the hypothesis, research evidence, implementation, numerical verification, and risk analysis for every change.
- [`next_optimization_plan.md`](file:///e:/Jarvis-Q1.58-500M/experiments/throughput_optimization/next_optimization_plan.md): Prioritized, ranked roadmap for future optimization loops (microbatch tuning, profiling, Tensor Core audit, MoE dispatch, compilation, and ternary kernels).

---

## Key Experimental Results: Optimization Loop #1

Through systematic profiling and code auditing, two primary bottlenecks were resolved:
1. **Restored CUDA Attention:** Replaced Python-unrolled chunk loops and einsums with fused RoPE/ELU CUDA kernels and batched Tensor Core matrix multiplications (`CUDAAssociativeLinearAttention`).
2. **Eliminated VRAM Spikes & Paging:** Replaced in-place GPU optimizer state compaction with CPU-streamed compaction in `save_checkpoint`, preventing Windows WDDM from evicting memory across PCIe to system RAM.

### Summary Metrics

```text
====================================================================================
Configuration                                     | Step Time    | Throughput
------------------------------------------------------------------------------------
Baseline Production Run (Paging + Ref Attn)       |  10.9500 s   |    374.0 tok/s
Fast Path Production Engine (CUDA Attn + No Page) |   2.7904 s   |  1,467.9 tok/s
------------------------------------------------------------------------------------
Net Acceleration: 3.92x (+292.5% Throughput)
Peak VRAM Reduction: 12,992 MB -> 9,422 MB (spikes eliminated)
====================================================================================
```

---

## Experimental Protocol & Safety Guidelines

All throughput optimization work in Jarvis adheres to a strict scientific protocol:
1. **Lock Checkpoints:** Baseline checkpoints on drive `G:` are sacred and must never be modified or overwritten.
2. **Empirical Evidence First:** No optimization is merged based on theory alone; each change requires isolated micro-benchmarking and numerical correctness testing.
3. **Reversibility:** All engine modifications must remain modular, guarded behind CLI flags (e.g. `--no-cuda-attn`), and tracked via Git commits.
4. **Zero Architecture Changes:** Model architecture, hidden dimensions, head counts, and sequence lengths ($T=512$) remain strictly constant.
