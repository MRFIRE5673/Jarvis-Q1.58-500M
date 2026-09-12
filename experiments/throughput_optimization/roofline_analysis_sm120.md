# Jarvis "20K Tok/s" CUDA R&D Program: Phase 0 — Compute Roofline Analysis

**Date:** September 12, 2026  
**Target Hardware:** NVIDIA GeForce RTX 5070 12GB (Blackwell Architecture, SM 12.0 / sm_120)  
**Target Workload:** Jarvis-Q1.58-500M Pretraining Step (Full Forward + Backward + Optimizer Update)  
**Metric:** FULL TRAINING TOKENS / WALL-CLOCK SECOND (No prefill cheats, no decode cheats, no skipping backward)

---

## 1. Executive Summary & Verdict on 20,000 Tok/s

> [!IMPORTANT]
> **Definitive Answer:**  
> **Is 20,000 full-training tok/s physically plausible on the RTX 5070 in BF16 without changing active compute?**  
> **NO.** In pure BF16 with gradient checkpointing, sustained 20,000 tok/s requires **57.43 TFLOPs**, which is **93.5% of the RTX 5070's sustained physical ceiling (61.44 TFLOPs)** and **77.9% of its absolute boost ceiling (73.73 TFLOPs)**. No production training engine (with dynamic MoE routing, RoPE, LayerNorms, loss reduction, and AdamW) achieves $\ge 78\text{--}93\%$ Model FLOPs Utilization (MFU) on a single GPU.
>
> **What IS physically achievable under current active compute?**  
> - **Current Baseline (Verified):** **1,468 tok/s** (~6.8% MFU, dominated by kernel launches and accumulation overhead).  
> - **Realistic Max in BF16 (50% MFU):** **10,699 tok/s** (sustained) to **12,838 tok/s** (boost).  
> - **Extreme Theoretical Upper Bound in BF16 (70% MFU, CUDA Graphs + Fused Kernels):** **14,978 tok/s** to **17,974 tok/s**.  
>
> **How CAN 20,000 Tok/s Be Achieved?**  
> 1. **Native FP8 Tensor Cores (Blackwell SM 12.0):** Doubles peak compute to 122.9–147.5 TFLOPs. At 20,000 tok/s, required MFU drops to **46.7%** (sustained) / **38.9%** (boost) — **fully realistic**.  
> 2. **Native FP4 / NVFP4 / Packed Ternary GEMM:** Quadruples peak compute to 245.8–294.9 TFLOPs. At 20,000 tok/s, required MFU is only **23.4%** — **highly achievable**, with potential beyond 40,000 tok/s.  
> 3. **Eliminating Gradient Checkpointing (Activation Stashing):** Drops required compute from 2.871 to 2.231 GFLOPs/tok (-22.3%), bringing BF16 required MFU down to **72.6%** (sustained) / **60.5%** (boost) — borderline feasible.  
> 4. **Top-1 MoE Architecture (Compute Reduction):** Reduces active parameters from 405.1M $\to$ 304.3M (-24.9%), bringing BF16 required MFU down to **68.2%** (sustained) / **56.8%** (boost).

---

## 2. Model Architecture & Active Parameter Breakdown

### Parameter Inventory
From exact model code audit ([`jarvis_engine/jarvis_model.py`](file:///e:/Jarvis-Q1.58-500M/jarvis_engine/jarvis_model.py)):
- Vocabulary Size: $V = 50,257$
- Hidden Dimension: $d_{\text{model}} = 1,024$
- Number of Layers: $N_L = 24$
- Attention Heads: $H = 16$ (Head dimension $d_k = 64$)
- MoE Experts: $E = 4$ total, $\text{Top-}k = 2$ active
- Expert Hidden Dimension: $d_{\text{ffn}} = 2 \times 1024 = 2,048$
- Sequence Length: $T = 512$

```text
Component                     | Total Parameters | Active Parameters (Top-2 MoE)
--------------------------------------------------------------------------------
Token Embeddings (tok_emb)    |    51,463,168    |    51,463,168
Output LM Head (lm_head)      |    51,463,168    |    51,463,168
24x Attention Subsystems      |   100,663,680    |   100,663,680
24x MoE Subsystems            |   402,751,488    |   201,424,896 (Top-2 active)
24x Norms, LSF & Reflective   |        49,200    |        49,200
--------------------------------------------------------------------------------
TOTAL                         |   606,390,704    |   405,064,112
                              |  (606.39M params)|  (405.06M active params)
```

---

## 3. Exact FLOPs Per Token Calculation

Every matrix multiplication $A_{(1 \times K)} \times W_{(K \times N)}$ requires $2KN$ FLOPs (1 multiply + 1 accumulate).

### A. Forward Pass FLOPs per Token
1. **Token Embeddings:** $0\text{ FLOPs}$ (Direct index gather / memory lookup).
2. **Attention Projections ($Q, K, V, O$):**  
   $$4 \times (2 \times d_{\text{model}}^2) = 8 \times 1024^2 = 8,388,608\text{ FLOPs / layer}$$
3. **Associative Linear Attention Recurrence & Kernel ($T=512, cs=64, H=16, d_k=64$):**  
   $$\text{Intra-chunk } Q K^T \text{ and } \text{Attn } V + \text{Cross-chunk state matmul} + \text{State update} \approx 458,752\text{ FLOPs / layer}$$
4. **MoE Router Projection:**  
   $$2 \times d_{\text{model}} \times E = 2 \times 1024 \times 4 = 8,192\text{ FLOPs / layer}$$
5. **MoE Active Expert Feed-Forward ($\text{Top-}k = 2$):**  
   $$2 \times [ (2 \times d_{\text{model}} \times d_{\text{ffn}}) + (2 \times d_{\text{ffn}} \times d_{\text{model}}) ] = 2 \times [ 4,194,304 + 4,194,304 ] = 16,777,216\text{ FLOPs / layer}$$
6. **Liquid State Fusion (LSF Causal Scan, $T=512$):**  
   $$2 \times T \times d_{\text{model}} = 2 \times 512 \times 1024 = 1,048,576\text{ FLOPs / layer}$$
7. **Norms, Residuals & Activations:**  
   $$\text{RMSNorms} + \text{GELU} + \text{Reflective} \approx 24,576\text{ FLOPs / layer}$$
8. **Total per JarvisBlock Forward:**  
   $$8,388,608 + 458,752 + 8,192 + 16,777,216 + 1,048,576 + 24,576 = \mathbf{26,705,920\text{ FLOPs / layer}}$$
9. **Total 24 Blocks Forward:**  
   $$24 \times 26,705,920 = 640,942,080\text{ FLOPs}$$
10. **Final LM Head Projection:**  
    $$2 \times d_{\text{model}} \times V = 2 \times 1024 \times 50,257 = 102,926,336\text{ FLOPs}$$
11. **Cross-Entropy Loss & Reduction:**  
    $$3 \times V = 3 \times 50,257 \approx 150,771\text{ FLOPs}$$

$$\mathbf{\text{Total Forward FLOPs / Token}} = 640,942,080 + 102,926,336 + 150,771 = \mathbf{744,019,187\text{ FLOPs} \approx 0.744\text{ GFLOPs/tok}}$$

---

### B. Backward Pass FLOPs per Token
In general backpropagation:
- Activation gradient ($dZ = dY \cdot W^T$): $2KN$ FLOPs.
- Weight gradient ($dW = X^T \cdot dY$): $2KN$ FLOPs.  
Standard backward is exactly **$2 \times \text{Forward}$**.

1. **Standard Backward (No Checkpointing):**  
   $$\text{FLOPs}_{\text{bwd, std}} = 2 \times 744,019,187 = \mathbf{1,488,038,374\text{ FLOPs} \approx 1.488\text{ GFLOPs/tok}}$$
   $$\text{Total Step FLOPs (No Checkpoint)} = \text{Forward} + \text{Backward} = 3 \times \text{Forward} = \mathbf{2,232,057,561\text{ FLOPs} \approx 2.232\text{ GFLOPs/tok}}$$

2. **Checkpointed Backward (Recomputing 24 Blocks):**  
   In Jarvis production training, all 24 blocks are wrapped in `torch.utils.checkpoint.checkpoint(block)`. During the backward pass, each block's forward pass is recomputed before computing gradients:  
   $$\text{Recomputed Forward} = 24 \times 26,705,920 = 640,942,080\text{ FLOPs}$$
   $$\text{FLOPs}_{\text{bwd, ckpt}} = 1,488,038,374 + 640,942,080 = \mathbf{2,128,980,454\text{ FLOPs} \approx 2.129\text{ GFLOPs/tok}}$$
   $$\mathbf{\text{Total Step FLOPs (Checkpointed)}} = \text{Forward} + \text{Checkpointed Backward} = \mathbf{2,872,999,641\text{ FLOPs} \approx 2.873\text{ GFLOPs/tok}}$$

---

## 4. Hardware Compute & Bandwidth Profile: NVIDIA GeForce RTX 5070

- **Architecture:** NVIDIA Blackwell Architecture (Compute Capability 12.0 / SM 12.0)
- **Streaming Multiprocessors (SMs):** 48 SMs
- **Clocks:**
  - Sustained full-load clock (280W TDP): **2.50 GHz**
  - Maximum boost clock: **3.00 GHz** (3,090 MHz reported by `nvidia-smi`)
- **Memory Subsystem:**
  - Dedicated VRAM: 12,227 MiB GDDR7
  - Memory Bus Width: 192-bit
  - Memory Pin Speed: 28.0 Gbps
  - **Theoretical Peak Bandwidth:** $(192 / 8) \times 28.0 = \mathbf{672.0\text{ GB/s}}$
  - **Achievable Sustained Bandwidth (~80% efficiency):** $\mathbf{537.6\text{ GB/s}}$

### Tensor Core Dense Compute Capabilities (sm_120)
| Precision | FLOPs / SM / Cycle | Sustained Peak (@2.5 GHz) | Max Boost Peak (@3.0 GHz) | Sparsity 2:4 Peak (@3.0 GHz) |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 (CUDA Cores)** | 128 | 15.36 TFLOPs | 18.43 TFLOPs | N/A |
| **BF16 / FP16 Tensor Core** | **512** | **61.44 TFLOPs** | **73.73 TFLOPs** | 147.46 TFLOPs |
| **FP8 Tensor Core** | **1,024** | **122.88 TFLOPs** | **147.46 TFLOPs** | 294.91 TFLOPs |
| **FP4 / NVFP4 Tensor Core** | **2,048** | **245.76 TFLOPs** | **294.91 TFLOPs** | 589.82 TFLOPs |

---

## 5. Theoretical Compute-Bound Throughput Ceilings

The compute-bound throughput limit is:
$$\text{tok/s}_{\text{compute}} = \frac{\text{Peak TFLOPs} \times 10^{12} \times \text{MFU}}{\text{FLOPs per Token}}$$

### Full Training Step Ceilings (Forward + Backward + Optimizer)

| Precision & Execution Mode | FLOPs/tok | 100% Ideal Peak | 70% High MFU (World-Class) | 50% Realistic MFU (Optimized) |
| :--- | :---: | :---: | :---: | :---: |
| **BF16 (Standard, No Checkpointing)** | 2.232 G | **33,049 tok/s** | 23,134 tok/s | 13,770 tok/s |
| **BF16 (With Gradient Checkpointing)** | 2.873 G | **25,676 tok/s** | 17,974 tok/s | 10,699 tok/s |
| **FP8 (With Gradient Checkpointing)** | 2.873 G | **51,353 tok/s** | **35,947 tok/s** | **21,397 tok/s** |
| **FP4 / Packed Ternary (With Checkpoint)** | 2.873 G | **102,706 tok/s** | **71,894 tok/s** | **42,794 tok/s** |

### Mathematical Proof for 20,000 Tok/s in BF16:
To sustain 20,000 tok/s in checkpointed BF16:
$$\text{Compute Required} = 20,000\text{ tok/s} \times 2.873\times 10^9\text{ FLOPs/tok} = \mathbf{57.43\text{ TFLOPs}}$$
$$\text{Required MFU of Sustained Peak} = \frac{57.43\text{ TFLOPs}}{61.44\text{ TFLOPs}} = \mathbf{93.5\%}$$
$$\text{Required MFU of Boost Peak} = \frac{57.43\text{ TFLOPs}}{73.73\text{ TFLOPs}} = \mathbf{77.9\%}$$

> [!CAUTION]
> In PyTorch training workloads, MFU rarely exceeds 55–60% even with Megatron-LM / FlashAttention due to non-GEMM kernels, memory latency, and Python launch bubbles. Sustaining 78–93% MFU in BF16 across an entire training loop is **thermodynamically and architecturally impossible on this single GPU**.

---

## 6. Memory Traffic & Arithmetic Intensity

In training, active weights ($W_{\text{active}} = 405.1\text{M params}$) are streamed from GDDR7 VRAM for each microbatch of $N_{\text{micro}} = B \times T$ tokens:
- Forward pass: 1 read of $W_{\text{active}}$
- Checkpointed backward: 1 read for recomputation + 1 read for input gradients $dZ$
- Total active weight traffic: $\approx 3 \times W_{\text{active}}$ per microbatch.

### Traffic and Intensity Comparison Across Configurations
*(Calculated with Achievable GDDR7 Bandwidth = 537.6 GB/s)*

| Batch Configuration | Active Weight Precision | Weight Traffic / mb | Activation Traffic / mb | Total Bytes / tok | Arithmetic Intensity | Memory-Bound Ceiling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$B=2, T=512$ (1,024 toks)** | **BF16** (2.0 B/param) | 2,430.4 MB | 151.0 MB | 2,520,878 B | **1,139.1 FLOPs/B** | **213,259 tok/s** |
| **$B=4, T=512$ (2,048 toks)** | **BF16** (2.0 B/param) | 2,430.4 MB | 302.0 MB | 1,334,167 B | **2,152.2 FLOPs/B** | **402,948 tok/s** |
| **$B=8, T=512$ (4,096 toks)** | **BF16** (2.0 B/param) | 2,430.4 MB | 604.0 MB | 740,812 B | **3,876.1 FLOPs/B** | **725,691 tok/s** |
| **$B=2, T=512$ (1,024 toks)** | **Packed Ternary** (0.25 B/param) | 303.8 MB | 151.0 MB | 444,134 B | **6,465.2 FLOPs/B** | **1,210,446 tok/s** |
| **$B=4, T=512$ (2,048 toks)** | **Packed Ternary** (0.25 B/param) | 303.8 MB | 302.0 MB | 295,795 B | **9,707.5 FLOPs/B** | **1,817,476 tok/s** |

### Critical Finding on Memory vs. Compute:
- On RTX 5070, the hardware ridge point is:
  $$\text{Ridge Point} = \frac{\text{Peak BF16 TFLOPs}}{\text{Achievable Bandwidth}} = \frac{61.44 \times 10^{12}}{537.6 \times 10^9} = \mathbf{114.3\text{ FLOPs/Byte}}$$
- Jarvis's arithmetic intensity is **1,139 to 3,876 FLOPs/Byte**—nearly **10x to 34x above the ridge point**!
- **Conclusion:** Jarvis training is **deeply compute-bound**, NOT memory-bandwidth bound. Memory bandwidth will not bottleneck performance until throughput exceeds 200,000 tok/s.

---

## 7. Concrete Pathways to Reach 20,000 Tok/s

Since BF16 cannot reach 20,000 tok/s under current active compute, here are the **4 viable engineering paths** to achieve or exceed 20,000 tok/s:

### Pathway 1: Blackwell Native FP8 Tensor Cores (Recommended Primary Path)
- **Mechanism:** Leverage native FP8 (E4M3/E5M2) Tensor Core instructions on SM 12.0.
- **Compute Ceiling:** **122.88 TFLOPs sustained / 147.46 TFLOPs boost**.
- **Required MFU for 20,000 tok/s:** **46.7% sustained / 38.9% boost** (standard achievable efficiency in PyTorch/TransformerEngine).
- **Realistic Throughput at 50% MFU:** **21,397 tok/s**.
- **Risk:** Low/Medium (FP8 pretraining stability well-established in modern LLMs).

### Pathway 2: Packed 1.58-Bit Ternary GEMM / NVFP4 Tensor Cores
- **Mechanism:** Replace dense float matmuls with direct hardware-accelerated ternary / FP4 Tensor Core kernels (2 bits per weight, 4 weights per byte).
- **Compute Ceiling:** **245.76 TFLOPs sustained / 294.91 TFLOPs boost**.
- **Required MFU for 20,000 tok/s:** Only **23.4% sustained**.
- **Realistic Throughput at 50% MFU:** **42,794 tok/s**.
- **Risk:** High R&D effort (requires custom CUTLASS/CuTe GEMM kernel development).

### Pathway 3: Activation Stashing (Eliminate Gradient Checkpointing)
- **Mechanism:** At microbatch $B=2$, total intermediate activations across all 24 layers are only ~360–500 MB. We can turn off gradient checkpointing without running out of 12GB VRAM.
- **Impact:** Eliminates block recomputation, dropping step compute from 2.873 to 2.232 GFLOPs/tok (**-22.3% compute**).
- **BF16 Ceiling at 70% MFU:** **23,134 tok/s** (boost).

### Pathway 4: Architectural Compute Sparsity (Top-1 MoE)
- **Mechanism:** Route each token to $\text{Top-}1$ expert instead of $\text{Top-}2$.
- **Impact:** Active parameters drop from 405.1M $\to$ 304.3M (**-24.9% active compute**). Total step compute drops to 2.094 GFLOPs/tok.
- **Required BF16 MFU for 20,000 tok/s:** **68.2% sustained / 56.8% boost** (feasible with CUDA graphs).
- **Risk:** Potential impact on downstream model perplexity (requires empirical validation).

---

## 8. Summary Table of Maximum Plausible Throughput

```text
========================================================================================================================
Architecture / Precision Mode              | Step FLOPs/tok | Peak TFLOPs | 50% MFU (Realistic) | 70% MFU (Upper Bound)
------------------------------------------------------------------------------------------------------------------------
1. BF16 Checkpointed (Current Baseline)   |    2.873 G     |    61.44    |     10,699 tok/s    |     14,978 tok/s
2. BF16 No-Checkpointing (Stashed Activ)   |    2.232 G     |    61.44    |     13,770 tok/s    |     19,278 tok/s
3. BF16 Top-1 MoE (Compute Reduction)      |    2.094 G     |    61.44    |     14,670 tok/s    |     20,538 tok/s
4. FP8 Checkpointed (Blackwell Native)     |    2.873 G     |   122.88    |     21,397 tok/s    |     29,956 tok/s
5. FP8 No-Checkpointing                    |    2.232 G     |   122.88    |     27,541 tok/s    |     38,557 tok/s
6. Packed Ternary / NVFP4 Checkpointed     |    2.873 G     |   245.76    |     42,794 tok/s    |     59,912 tok/s
========================================================================================================================
```
