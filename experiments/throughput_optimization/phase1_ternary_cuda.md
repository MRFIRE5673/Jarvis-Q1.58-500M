# Jarvis "20K Tok/s" CUDA R&D Program: Phase 1 — Packed Ternary CUDA GEMM & Activation-Stashing

**Date:** September 12, 2026  
**Hardware:** NVIDIA GeForce RTX 5070 12GB (Blackwell SM 12.0 / sm_120)  
**Workload:** Jarvis-Q1.58-500M Pretraining Step ($B=2, T=512$, accum=4, 4,096 tokens/update, BF16 AMP, AdamW Fused)  
**Status:** Phase 1 Complete

---

## 1. Executive Summary & Concise Final Table

In Phase 1, two major throughput hypotheses were subjected to rigorous empirical testing:
1. **Part A (Activation Stashing):** Can gradient checkpointing be safely removed to eliminate ~22% recomputation overhead?
2. **Part B–G (Packed Ternary CUDA GEMM):** Can direct consumption of 2-bit packed ternary weights ($0.25$ bytes/weight) out-perform dense BF16 matrix operations on Blackwell SM 12.0?

### Concise Final Results Table

| Experiment | Baseline tok/s | New tok/s | Speedup | Peak VRAM | Correct? | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Part A: Activation Stashing (Grad Checkpoint OFF)** | **1,513.6 tok/s** | 178.1 tok/s | **0.12x (-88.2%)** | 18,394 MB (+7.8 GB PCIe paging) | Yes ($\Delta < 1.5 \times 10^{-5}$) | **REVERT / KEEP CKPT ON** |
| **Part D: Isolated Ternary GEMM ($1024 \times 1024$)** | **25.81 TFLOPs** (cuBLAS) | 6.51 TFLOPs | **0.25x (-74.8%)** | 256 KB (8x less weight mem) | Yes ($\Delta < 7.8 \times 10^{-3}$) | **DOCUMENT / BENCHMARK** |
| **Part G: Full Model Test (Ternary Out-Proj)** | **1,546.3 tok/s** | 171.6 tok/s | **0.11x (-88.9%)** | 17,256 MB (autograd overhead) | Yes ($\Delta < 9.0 \times 10^{-4}$) | **REVERT TO DENSE cuBLAS** |

---

## 2. Part A: Activation Stashing Experiment

### The Hypothesis
From the Phase 0 roofline analysis, gradient checkpointing adds 640.9M FLOPs/tok of recomputation. Removing checkpointing would reduce step FLOPs from 2.873 to 2.232 GFLOPs/tok (-22.3% compute).

### Empirical Measurement
Running the exact 606M model with Checkpointing ON vs. OFF revealed that the autograd graph across 24 layers (including attention chunk states, MoE scatter/gather buffers, expert GEMMs, and LSF causal decay states) requires **+7,760.5 MB of persistent activation memory**:

```text
Metric                      | Checkpointing ON       | Checkpointing OFF (Stashing)
-----------------------------------------------------------------------------------
Optimizer Step Time         | 2.7062 s               | 22.9920 s (8.5x slower)
Throughput                  | 1,513.6 tok/s          | 178.1 tok/s (-88.2%)
Peak Allocated VRAM         | 10,120.7 MB            | 17,881.2 MB (+7.76 GB)
Peak Reserved VRAM          | 10,548.0 MB            | 18,394.0 MB (PCIe Paging!)
```

### Conclusion & Decision
On a 12GB GPU, activation stashing triggers massive Windows WDDM PCIe paging, collapsing throughput. **Gradient checkpointing is retained as mandatory.**

---

## 3. Part B & C: Packed Ternary Representation Design

### Representation Selected
- **2-bit Ternary Integer Encoding (4 weights per byte):**
  - `00` ($0$) $\to 0$
  - `01` ($1$) $\to +1$
  - `10` ($2$) $\to -1$
- **Vectorized Arithmetic Unpack Formula:**
  $$\text{Decoded Weight} = (\text{code} \& 1) - (\text{code} \gg 1)$$
  Verified on 1,048,576 elements with **0 errors (100% exact match)**.
- **Memory Footprint:** $0.25\text{ bytes/weight}$ (8.0x reduction vs BF16, 16.0x vs FP32).
- **Fast CUDA Fused Packing Kernel:** Quantizes FP32 master weights via AbsMean $\alpha = \text{mean}(|W|)$ and packs $1024 \times 1024$ into 256 KB in **$0.012\text{ ms}$**.

---

## 4. Part D: Exact Shape Micro-Benchmarks

Testing the dominant matrix shapes on the RTX 5070 12GB (SM 12.0 Blackwell):

```text
========================================================================================================================
Matrix Operation                  | Shape (M, K, N) | Dense cuBLAS BF16 | Packed Ternary CUDA | Compute Ratio (cuBLAS/Packed)
------------------------------------------------------------------------------------------------------------------------
Attention Projection (Q/K/V/Out) | 1024, 1024, 1024| 0.0832 ms (25.8 T) | 0.3296 ms (6.5 T)   | 3.96x faster (cuBLAS)
MoE Gate/Up Projection (W1)       | 1024, 1024, 2048| 0.1069 ms (40.2 T) | 0.5474 ms (7.9 T)   | 5.12x faster (cuBLAS)
MoE Down Projection (W2)          | 1024, 2048, 1024| 0.1226 ms (35.0 T) | 0.5830 ms (7.4 T)   | 4.76x faster (cuBLAS)
MoE W1 Sub-Batch (512 tokens)     |  512, 1024, 2048| 0.0829 ms (25.9 T) | 0.3321 ms (6.5 T)   | 4.01x faster (cuBLAS)
MoE W2 Sub-Batch (512 tokens)     |  512, 2048, 1024| 0.0810 ms (26.5 T) | 0.3294 ms (6.5 T)   | 4.07x faster (cuBLAS)
========================================================================================================================
```

---

## 5. Architectural Root Cause & Theoretical Insight

### Why Did 8x Memory Reduction Not Yield Speedup?
1. **Compute Density vs. Memory Bandwidth:**  
   - Dedicated **Blackwell Tensor Cores** deliver **512 FLOPs/cycle/SM** (dense BF16).
   - Standard **CUDA ALUs** (used for software unpacking and integer/float math) deliver only **128 FLOPs/cycle/SM**.
2. **Arithmetic Intensity Regime:**  
   - Jarvis training has an arithmetic intensity of **1,139 to 3,876 FLOPs/Byte**.
   - The RTX 5070 hardware ridge point is **114.3 FLOPs/Byte**.
   - Because Jarvis is **10x to 34x above the ridge point**, memory bandwidth is already plentiful. Trading away hardware Tensor Cores for software SIMD unpacking on CUDA cores sacrifices ~4x raw compute density to save bandwidth that wasn't bottlenecking the system.

---

## 6. Scientific Conclusion & Next Steps

1. **Keep Baseline Intact:** Dense BF16 with cuBLAS Tensor Cores remains the production training path.
2. **Path to 20K Tok/s:**  
   As proven in Phase 0, achieving 20K tok/s requires leveraging **hardware low-precision Tensor Cores**:
   - **Phase 2:** MoE Grouped GEMM (fusing expert dispatch and execution).
   - **Phase 3:** Blackwell Native **FP8 Tensor Cores** (122.9 to 147.5 hardware TFLOPs).
   - **Phase 4:** CUDA Graph capture of fixed-shape step to eliminate launch overheads.
