# Jarvis Throughput Optimization: Activation Stashing Benchmark (Part A)

**Date:** September 12, 2026  
**Hardware:** NVIDIA GeForce RTX 5070 12GB (Blackwell SM 12.0, 12,227 MiB Dedicated VRAM)  
**Model:** Jarvis-Q1.58-500M (24 layers, $d_{\text{model}}=1024$, 16 heads, 4 experts Top-2, $T=512$, $B=2$, accum=4, 4,096 tokens/update)  
**Experiment:** Benchmark Gradient Checkpointing ON vs. OFF (Activation Stashing)

---

## 1. Executive Summary & Decision

> [!CAUTION]
> **Definitive Finding:**  
> Disabling gradient checkpointing (activation stashing) on the 12GB RTX 5070 **fails catastrophically**.  
> While theoretical linear-transformer back-of-the-envelope estimates suggested intermediate activations might fit in VRAM, empirical measurement reveals that the complete autograd graph across 24 layers (including attention chunk states, MoE scatter/gather buffers, expert GEMM activations, and LSF decay tensors) requires **+7,760.5 MB of additional activation VRAM**.
>
> This pushed peak VRAM demand to **18,394 MB**, exceeding the 12,227 MiB physical ceiling by over 6.1 GB. Windows WDDM silently paged gigabytes of memory over PCIe to host RAM, causing step time to collapse from **2.71s $\to$ 22.99s** and throughput to plummet by **-88.2% (1,513.6 tok/s $\to$ 178.1 tok/s)**.

**Decision:** **REJECT ACTIVATION STASHING. KEEP GRADIENT CHECKPOINTING ON.**  
Gradient checkpointing is strictly mandatory to train Jarvis-Q1.58-500M within 12GB dedicated VRAM.

---

## 2. Empirical Benchmark Measurements

Both configurations were benchmarked under identical conditions using the full 606M model with BF16 AMP, fused AdamW, and 4,096 tokens per optimizer update:

| Metric | Checkpointing ON (Default) | Checkpointing OFF (Stashing) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Optimizer Step Time** | **2.7062 s** | **22.9920 s** | **0.12x (8.5x slower)** |
| **Training Throughput** | **1,513.6 tok/s** | **178.1 tok/s** | **-88.2%** |
| **Peak Forward Allocated VRAM** | 10,120.7 MB | 17,881.2 MB | **+7,760.5 MB** |
| **Peak Backward Allocated VRAM** | 10,214.7 MB | 17,975.3 MB | **+7,760.7 MB** |
| **Peak Optimizer Allocated VRAM** | 9,520.7 MB | 14,250.3 MB | **+4,729.6 MB** |
| **Peak Step Allocated VRAM** | 9,520.7 MB | 14,250.3 MB | **+4,729.6 MB** |
| **Peak Reserved VRAM** | **10,548.0 MB** | **18,394.0 MB** | **+7,846.0 MB (PCIe Paging)** |
| **Final Loss (Step 3)** | 9.6970 | 9.6970 | $\Delta = 1.50 \times 10^{-5}$ |
| **Final Gradient Norm** | 1.2972 | 1.2996 | $\Delta = 2.38 \times 10^{-3}$ |
| **NaN / Inf Detected** | **False (Clean)** | **False (Clean)** | Numerical Agreement |

---

## 3. Why Theoretical Activation Estimates Failed

Simplified transformer memory models assume activation memory is simply $O(B \cdot T \cdot d_{\text{model}} \cdot N_L)$. However, a production neuromorphic hybrid layer contains numerous non-linear intermediate operations whose backward graphs store substantial state:
1. **MoE Expert Expansion:** In each of the 24 layers, 2 active experts expand tokens from $d_{\text{model}}=1024 \to d_{\text{ffn}}=2048$. Forward activations before and after GELU are saved for backward.
2. **MoE Scatter/Gather Maps:** CUDA dispatch and scatter combine save index mappings, gate tensors, and permutation maps for backward.
3. **Associative Linear Attention:** RoPE rotation buffers, $Q, K, V$ chunk matrices, cross-chunk recurrent associative states, and decay matrices are all preserved on the autograd tape.
4. **Liquid State Fusion:** Decay matrices and membrane states across time are preserved for backward gradients.

Multiplying these state tensors across 24 layers produces **~7.76 GB of persistent activation memory**.

When combined with:
- Static model parameters: ~1.2 GB
- Fused AdamW optimizer states (FP32 moments): ~4.8 GB
- Static buffers & CUDA runtime: ~0.6 GB
- Total baseline footprint: ~6.6 GB

Adding 7.76 GB pushes total memory to **~14.4 GB allocated / 18.4 GB reserved**, which cannot fit in 12,227 MiB physical VRAM without extreme PCIe thrashing.

---

## 4. Engineering Takeaway

Gradient checkpointing trades ~22% recomputation compute to save ~7.76 GB of memory. On a 12GB GPU, this trade-off is not merely beneficial—it is the **enabling mechanism** that allows full-speed 1,514 tok/s training without falling into Windows WDDM PCIe paging traps.
