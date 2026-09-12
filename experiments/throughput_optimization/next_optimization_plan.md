# Jarvis Training Throughput: Next Optimization Plan

**Date:** September 12, 2026  
**Current Baseline Throughput:** **1,467.9 tok/s** (2.7904 s/update @ 4,096 tokens/step, 9,422 MB allocated VRAM)  
**Target:** Maximize Jarvis-Q1.58-500M training efficiency on the RTX 5070 12GB before launching the future 1.0B token run.

---

## 1. Prioritized Optimization Ranking

All candidate optimizations are ranked based on:
1. **Measured Expected Speedup**
2. **Implementation Difficulty**
3. **Risk Profile**
4. **VRAM Impact**

| Rank | Candidate Optimization | Expected Speedup | Implementation Difficulty | Risk | VRAM Impact | Priority |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Micro-Batch Configuration Tuning ($B=4, \text{accum}=2$)** | **+10% to +18%** | Very Low (CLI flag / config) | Negligible | +800–1,200 MB (fits within 12GB) | **Immediate (Loop #7)** |
| **2** | **Detailed Profile of Restored Step Bottlenecks** | Diagnostic | Low (PyTorch profiler / CUDA events) | Zero | None | **Immediate (Loop #2)** |
| **3** | **Tensor Core & Precision Audit (GEMM alignments & FP32 promotions)** | **+5% to +10%** | Medium (Code audit & casting fixes) | Low | Neutral | **High (Loop #4)** |
| **4** | **MoE Routing & Token Packing Optimization** | **+8% to +15%** | Medium (Fused router / vectorized dispatch) | Low to Medium | Neutral to -100 MB | **High (Loop #6)** |
| **5** | **`torch.compile` / Inductor for Non-CUDA Layers** | **+5% to +15%** | Medium to High (Extension interoperability) | Medium (Compilation errors with custom CUDA) | +200–500 MB | **Medium (Loop #3)** |
| **6** | **CUDA Attention Kernel Specialization for $T=512$** | **+3% to +6%** | High (Custom CUDA C++/CUDA kernel edits) | Medium | Neutral | **Medium (Loop #5)** |
| **7** | **Liquid State Fusion (LSF) CUDA / Vectorization** | **<2%** | High | Low | Neutral | **Deprioritized** (LSF is only ~3.9% of step) |
| **8** | **Hardware-Packed 1.58-bit Ternary GEMM Kernel** | **+25% to +40%** | Very High (Custom GEMV/GEMM bit-manipulation) | High | -3,000 MB | **Long-Term R&D (Loop #8)** |

---

## 2. Detailed Roadmap & Execution Plan

### Optimization Loop #2 — Profile the Restored Fast Path
- **Objective:** Map every microsecond of the 2.7904s step to find the new dominant bottleneck.
- **Method:**
  - Instrument `train_1b_production.py` with `torch.profiler.profile` (CUDA activity, CPU activity, memory allocations).
  - Quantify breakdown between:
    1. Attention forward/backward (currently ~1,440 ms / 51.6%)
    2. MoE dispatch and expert GEMMs (currently ~884 ms / 31.7%)
    3. LSF causal scan (currently ~423 ms / 15.2%)
    4. Optimizer step and gradient clipping (~42 ms / 1.5%)
- **Deliverable:** Update `bottleneck_history.md` with precise percentage allocations.

---

### Optimization Loop #7 — Micro-Batch Tuning ($B=4, \text{accum}=2$)
- **Rationale:** Microbatch $B=2$ means Tensor Cores in attention and MoE operate on small batch dimensions ($M=1024$ for $T=512$). Increasing to $B=4$ doubles batch density ($M=2048$), which increases Tensor Core efficiency from ~45% to ~70% of theoretical peak, while halving gradient accumulation loop overhead.
- **Constraint:** Must fit within 12,227 MiB VRAM without triggering WDDM PCIe paging. Currently, peak allocated memory is 9,422 MB; testing $B=4$ with checkpointing is projected to require ~10,800 MB allocated, leaving ~1.4 GB safety margin.
- **Verification:** Run 5 benchmark steps with $B=4, \text{accum}=2$ (total 4,096 tokens). Check throughput and VRAM.

---

### Optimization Loop #4 — Tensor Core Utilization & BF16 Precision Audit
- **Objective:** Ensure all matrix multiplications meet Tensor Core alignment requirements (multiples of 8/16) and remain strictly in bfloat16.
- **Audit Points:**
  - Embedding lookup and vocabulary projection layer ($V=32,000$, $D=1024$): check for accidental FP32 promotion during cross-entropy loss.
  - MoE gating network (`top_k=2`, `num_experts=4`): ensure router output tensor remains contiguous and does not cause host-device synchronization during top-k selection.
  - Norm layers: RMSNorm / LayerNorm epsilon casting.

---

### Optimization Loop #6 — MoE Dispatch & Expert GEMM Fusion
- **Objective:** Optimize the MoE subsystem, which constitutes ~884 ms (31.7%) of the step.
- **Target Areas:**
  - Replace Python-level index scattering with fused CUDA scatter/gather.
  - Evaluate batched expert GEMMs vs sequential expert loops.
  - Verify zero memory allocation during expert token assignment.

---

### Optimization Loop #3 — `torch.compile` Suitability
- **Objective:** Investigate whether `torch.compile(mode="reduce-overhead")` or `torch.compile(backend="inductor")` can fuse LayerNorm, residual additions, and activation scaling.
- **Risk Mitigation:**
  - Custom CUDA extensions (`CUDAAssociativeLinearAttention` and `CUDASparseMoELayer`) cannot always be traced cleanly by AOTAutograd.
  - Strategy: Compile individual submodules (e.g. `LiquidStateFusion` or feedforward blocks) rather than the root model, testing for numerical stability and zero graph breaks.

---

### Optimization Loop #8 — Long-Term Packed 1.58-Bit Ternary Kernels
- **Status:** Dedicated Future R&D Track.
- **Scope:** True 1.58-bit ternary storage (2 bits per weight, 4 weights per byte) with custom CUDA decoding kernels to reduce weight memory footprint from 1.2 GB to ~150 MB and eliminate memory bandwidth bottlenecks.
- **Execution:** Deferred until standard floating-point CUDA path reaches its theoretical ceiling.

---

## 3. Strict Operating Rules
1. Never start a long training run or modify baseline checkpoints.
2. Every proposed optimization must be proven with an isolated micro-benchmark and numerical verification before merging.
3. If an optimization yields $<2\%$ improvement without memory benefits, discard it.
