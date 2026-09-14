# Jarvis-Specific Optimizer Analysis: Architectural & Hardware Deep-Dive

## Executive Summary

The **Jarvis-Q1.58-500M** architecture is a hybrid neuromorphic foundation model combining:
1. **AbsMean-scaled Ternary Weights (1.58-bit)** with Straight-Through Estimator (STE) training.
2. **Sparse Top-2 Mixture-of-Experts (MoE)** with 4 experts per layer across 24 layers.
3. **Chunked Associative Linear Attention** with per-token $O(N)$ recurrence and learnable decay $\gamma$.
4. **Liquid State Fusion (LSF)** with Leaky Integrate-and-Fire membrane dynamics and dynamic variance gating $\alpha$.
5. **A strictly constrained single-GPU hardware budget:** NVIDIA GeForce RTX 5070 (12 GB GDDR7 VRAM).

Applying any optimizer uniformly across this entire model is mathematically suboptimal and experimentally hazardous. Different components exhibit fundamentally distinct mathematical properties: continuous vs. quantized forward representations, dense vs. sparse routing updates, matrix vs. scalar parameter topologies, and high vs. low gradient densities.

This document provides an audited, component-by-component analysis of optimizer suitability, establishes the theoretical hypotheses regarding ternary quantization and sparse MoE, and quantifies modeled memory profiles on the RTX 5070 12GB.

---

## 1. Fine-Grained Component Analysis

The 606,391,512 parameters of Jarvis partition into five distinct functional categories:

```
Total Parameters: 606,391,512 (100.0%)
├── 2D Hidden Transformations (AbsMean Ternary STE) : 503,316,480 (83.00%)
│   ├── Attention Projections (Q, K, V, Out)        : 100,663,296 (16.60%)
│   └── MoE Expert Projections (W1, W2 x 4 experts) : 402,653,184 (66.40%)
├── Vocabulary Embeddings & LM Head (FP32/BF16)     : 102,926,336 (16.97%)
│   ├── Token Embedding (tok_emb)                  :  51,463,168 ( 8.49%)
│   └── Language Model Head (lm_head)              :  51,463,168 ( 8.49%)
├── MoE Routing Logits (Dense Float)                :      98,304 ( 0.016%)
│   └── 24 layers x (1024 -> 4)                    :      98,304 ( 0.016%)
├── Normalization Parameters (1D Vectors)           :      50,176 ( 0.008%)
│   └── RMSNorm weights (49 norms x 1024)          :      50,176 ( 0.008%)
└── Neuromorphic Recurrent / Gating Scalars         :         408 (<0.001%)
    ├── Associative Attention Decay (gamma_raw)     :         384 (<0.001%)
    └── Liquid State Membrane Scale (var_scale)     :          24 (<0.001%)
```

### 1.1 Dense Transformer Projections (Attention Q, K, V, Out)
- **Shape & Count:** 24 layers $\times$ 4 projections $\times$ $(1024 \times 1024) = 100,663,296$ parameters.
- **Topology:** Perfectly square 2D matrices ($1024 \times 1024$).
- **Quantization:** `TernaryLinear` with AbsMean scaling ($\alpha = \text{mean}(|W|)$) and STE backward pass.
- **Optimizer Suitability:**
  - **Muon:** **HIGH-PRIORITY HYPOTHESIS (GREEN).** Square matrices are the optimal input for the 5th-order Newton-Schulz iteration. The spectral norm regularization ensures that the singular values of the projection matrices remain bounded near 1.0, which on float models prevents representation rank collapse. Its behavior on ternary STE projections is an active hypothesis to be verified.
  - **AdamW:** Functional and proven stable; coordinate-wise scaling distorts the singular value spectrum.
  - **Lion:** Poor; coordinate sign updates create discretization chatter against ternary quantization.

### 1.2 MoE Expert Feed-Forward Matrices (W1, W2 across 4 Experts)
- **Shape & Count:** 24 layers $\times$ 4 experts $\times$ [W1: $(1024 \times 2048)$ + W2: $(2048 \times 1024)$] $= 402,653,184$ parameters (66.4% of total model parameters!).
- **Topology:** Non-square 2D matrices with aspect ratio $1:2$ and $2:1$.
- **Quantization:** `TernaryLinear` with AbsMean scaling and STE backward pass.
- **Activation Pattern:** Conditionally executed; each token routes to Top-2 of 4 experts.
- **Optimizer Suitability:**
  - **Muon:** **HIGH-PRIORITY HYPOTHESIS (GREEN).** Non-square matrices are handled via the aspect-ratio adjusted update scale: $\alpha(M, N) = 0.2 \times \max(1, \sqrt{M/N})$ (Moonshot AI, 2025). Proven on Moonlight 16B MoE in BF16; whether this translates to ternary MoE requires experimental testing.
  - **AdamW:** Prone to variance lag on unselected experts.
  - **Sophia-G:** Unstable; Hessian diagonal estimates become sparse and noisy when tokens are routed selectively.

### 1.3 MoE Router Parameters
- **Shape & Count:** 24 layers $\times$ `nn.Linear(1024, 4, bias=False)` $= 98,304$ parameters.
- **Topology:** Extreme aspect ratio ($4 \times 1024$). The matrix rank cannot exceed 4!
- **Optimizer Suitability:**
  - **Muon:** **INCOMPATIBLE (RED for this component).** Applying Newton-Schulz orthogonalization to a $4 \times 1024$ matrix is degenerate: it normalizes 4 singular values to 1 and forces all other 1020 dimensions to 0. Furthermore, router logits dictate token routing entropy via softmax; forcing unit spectral norm destroys routing temperature calibration.
  - **AdamW:** **MANDATORY (GREEN).** Coordinate-wise adaptive updates allow router weights to adjust their dynamic range smoothly to balance expert utilization.

### 1.4 Vocabulary Embeddings & LM Head
- **Shape & Count:** `tok_emb` $(50257 \times 1024)$ + `lm_head` $(1024 \times 50257) = 102,926,336$ parameters.
- **Gradient Sparsity:** Extremely sparse for embeddings (only token IDs present in the micro-batch receive non-zero gradients; at $T=512, B=2$, at most 1,024 rows out of 50,257 are non-zero per forward pass).
- **Optimizer Suitability:**
  - **Muon:** **CATASTROPHIC (RED for this component).** Newton-Schulz polynomial iteration computes $X (X^T X)^k$. Multiplying a sparse matrix by its Gramian densifies the update across ALL 50,257 vocabulary rows, adding artificial gradient noise to tens of thousands of tokens that never appeared in the batch!
  - **AdamW:** **MANDATORY (GREEN).** AdamW's coordinate-wise updates preserve gradient sparsity: parameters for unobserved tokens remain completely untouched (or receive pure weight decay if configured).

### 1.5 RMSNorm Weights (1D Vectors)
- **Shape & Count:** 24 blocks $\times$ 2 norms $\times$ 1024 + 1 final norm $\times$ 1024 $= 50,176$ parameters.
- **Topology:** 1D vectors ($d=1024$).
- **Optimizer Suitability:**
  - **Muon:** **INCOMPATIBLE.** Matrix orthogonalization is mathematically undefined for 1D vectors.
  - **AdamW:** **MANDATORY (GREEN).**

### 1.6 Neuromorphic Recurrent & Gating Parameters
- **Associative Attention Decay:** `gamma_raw` $\in \mathbb{R}^{16}$ per block (24 blocks $= 384$ parameters). Controls token memory retention $\gamma = \text{sigmoid}(\gamma_{\text{raw}}) \approx 0.95$.
- **Liquid State Fusion Scale:** `var_scale` $\in \mathbb{R}$ per block (24 blocks $= 24$ parameters). Modulates membrane leakiness $\alpha \in [0.1, 0.99]$.
- **Optimizer Suitability:**
  - **Muon:** **INCOMPATIBLE.** Scalars and small vectors cannot be orthogonalized.
  - **AdamW:** **MANDATORY (GREEN).** Recurrent decay parameters require small, stable gradient steps ($\sim 1.5 \times 10^{-4}$) to prevent sudden state divergence or vanishing memory.

---

## 2. The Hybrid Optimizer Formulation

Based on the mathematical realities of each component, **pure Muon cannot and should not be applied to Jarvis.** Instead, the highest-priority experimental formulation is a **Hybrid Muon + AdamW** architecture:

$$\theta = \Theta_{\text{2D-Hidden}} \cup \Theta_{\text{1D/Sparse}}$$

1. **Group 1: $\Theta_{\text{2D-Hidden}}$ (Muon)**
   - Includes: Attention projections ($W_q, W_k, W_v, W_{out}$) and MoE expert matrices ($W_1^{(e)}, W_2^{(e)}$).
   - Total Parameters: **503,316,480 (83.00%)**.
   - Optimizer: **Muon** with aspect-ratio scaling and decoupled weight decay.
   - Theoretical State Memory: **1 FP32 momentum buffer (4 bytes/param)** $= 2,013.27$ MB.

2. **Group 2: $\Theta_{\text{1D/Sparse}}$ (AdamW)**
   - Includes: `tok_emb`, `lm_head`, `router`, `norm1`, `norm2`, `final_norm`, `gamma_raw`, `var_scale`.
   - Total Parameters: **103,075,032 (17.00%)**.
   - Optimizer: **Fused AdamW**.
   - State Memory: **2 FP32 buffers (8 bytes/param)** $= 824.60$ MB.

### Combined Theoretical State Memory
$$\text{State}_{\text{Hybrid}} = 2013.27 \text{ MB} + 824.60 \text{ MB} = \mathbf{2,837.87 \text{ MB (~2.77 GB)}}$$
Compared to pure AdamW ($4,851.13$ MB / $4.74$ GB), the Hybrid configuration offers a **modeled reduction of 2,013.26 MB (~1.97 GB) in optimizer state storage.**

---

## 3. Special Jarvis Question 1: Ternary Weights & STE Compatibility (Critical Research Questions)

Jarvis trains ternary weights using the **AbsMean-scaled Straight-Through Estimator (STE)** formulation (`jarvis_engine/utils/ternary_ops.py`):
$$\alpha = \text{mean}(|W_{\text{FP32}}|)$$
$$W_{\text{norm}} = \frac{W_{\text{FP32}}}{\alpha + \epsilon}$$
$$W_q = \text{round}\Big(\text{clamp}(W_{\text{norm}}, -1.0, 1.0)\Big) \cdot \alpha$$

In the backward pass:
$$\frac{\partial \mathcal{L}}{\partial W_{\text{FP32}}} \approx \frac{\partial \mathcal{L}}{\partial W_q} \cdot \mathbf{1}\{|W_{\text{FP32}}| \le 1.0\}$$

### The Six Key Research Questions (Hypotheses to Test)

There is currently **zero direct published literature** examining Muon optimization on AbsMean ternary weights with STE gradient clipping. The following questions represent hypotheses that must be resolved empirically during post-50M bakeoff testing:

1. **Does matrix orthogonalization fight ternary quantization?**
   - *Question:* Muon projects updates onto an orthogonal polar factor $O_t = U V^T$, normalizing all singular values to 1.0. Ternary weights $W_q$ have a discrete, highly structured singular value spectrum. Does forcing unit spectral updates on master weights $W_{\text{FP32}}$ conflict with the natural low-rank or discrete structure that ternary networks converge toward?
2. **Does it change the master weight distributions?**
   - *Question:* AdamW updates coordinates independently, allowing master weights to form Gaussian or Laplace-like tails. Muon distributes energy across singular vectors uniformly. Will this broaden or narrow the master weight distribution?
3. **Does it change zero/$\pm 1$ proportions?**
   - *Question:* In AbsMean ternary quantization, roughly 30–50% of weights typically map to 0, with the remainder mapping to $+1$ or $-1$. If Muon shifts the continuous distribution, does the density of zeros (sparsity) increase or collapse?
4. **Does it increase saturation outside the STE mask ($|W| > 1.0$)?**
   - *Question:* If master weights drift beyond $\pm 1.0$, the STE gradient mask sets $\nabla_{W} = 0$. Does Muon's larger step size ($\eta \sim 2 \times 10^{-3}$ vs. AdamW $1.5 \times 10^{-4}$) push more weights into the saturated/dead zone, or does decoupled weight decay keep them bounded?
5. **Does it improve gradient diversity?**
   - *Question:* Because $O_t$ is a full-rank orthogonal matrix, every weight in the matrix receives an update even if its specific STE gradient coordinate was zero. Could this global spectral coupling provide useful gradient diversity and prevent dead neurons, or does it add unwanted noise to converged weights?
6. **Does it destabilize the AbsMean scale factor $\alpha$?**
   - *Question:* The layer activation scale depends on $\alpha = \text{mean}(|W|)$. If Muon alters weight magnitudes rapidly, $\alpha$ could oscillate across training steps, causing activation variance instability in subsequent layers and triggering reflective penalties.

**Conclusion:** None of these questions can be answered by theory alone. They define the exact empirical diagnostic metrics for the Stage 1 and Stage 2 bakeoff tests.

---

## 4. Special Jarvis Question 2: Sparse Mixture-of-Experts (MoE)

Jarvis utilizes 4 experts per layer with Top-2 routing. On any given token, exactly 2 experts are active and 2 are inactive.

### Dynamic Update Imbalance under AdamW
Under AdamW, each parameter maintains an independent second-moment accumulator:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
When an expert is inactive for several consecutive micro-batches:
1. Its gradient is zero: $g_t = 0$.
2. Its second moment decays: $v_t \to \beta_2^k v_0$.
3. When the router suddenly assigns tokens to that expert, $v_t$ has become small. The first update step evaluates $g_t / (\sqrt{v_t} + \epsilon)$, resulting in an abnormally large step.
4. Concurrently, decoupled weight decay $(1 - \eta \lambda)$ continues to shrink the inactive expert's master weights toward zero on every optimizer step.

### Evidence from Moonlight 16B (BF16 MoE)
In **Moonlight** (Moonshot AI, arXiv:2502.16982), researchers proved that Muon resolves this MoE instability in standard 16-bit floating-point networks:
- Muon has no second-moment denominator $v_t$, eliminating variance lag.
- Newton-Schulz iteration normalizes the singular values of the momentum matrix: $\sigma_i(O_t) = 1.0$.
- Across 5.7T tokens, Muon eliminated expert collapse and achieved $\approx 2.0\times$ computational efficiency over AdamW on compute-optimal pretraining curves.

*Audit Caveat:* Moonlight used BF16 weights and Multi-head Latent Attention (MLA). Transfer to Jarvis's AbsMean ternary experts is a promising research hypothesis, but not an established fact.

---

## 5. Special Jarvis Question 3: Recurrent Memory Dynamics

Jarvis contains two distinct recurrence mechanisms:
1. **Associative Linear Attention:**
   Per-token state recurrence: $S_t = \gamma S_{t-1} + v_t \otimes k_t^T$, where $\gamma = \text{sigmoid}(\gamma_{\text{raw}})$.
2. **Liquid State Fusion (LSF):**
   Membrane potential EMA: $H_t = \alpha H_{t-1} + (1 - \alpha) M_t$, where $\alpha = \alpha_{\text{min}} + (\alpha_{\text{max}} - \alpha_{\text{min}}) \sigma(-\text{var\_scale} \cdot \text{act\_var})$.

### Optimizer Sensitivity
- Both $\gamma$ and $\alpha$ operate as exponential recurrence bases. If an optimizer forces $\gamma_{\text{raw}}$ or $\text{var\_scale}$ to shift abruptly, the recurrence horizon $T_{\text{horizon}} \approx 1 / (1 - \gamma)$ shifts exponentially, causing massive loss spikes.
- In our proposed Hybrid configuration, `gamma_raw` and `var_scale` are strictly assigned to AdamW with small gradient clipping ($||g|| \le 1.0$), ensuring that these sensitive gating scalars evolve smoothly and monotonically.

---

## 6. Hardware Constraint: NVIDIA RTX 5070 (12 GB VRAM)

The NVIDIA GeForce RTX 5070 has **12,288 MB of physical GDDR7 memory**.
Under Windows 11, the maximum safe allocatable ceiling before CUDA Out-of-Memory (OOM) is **~11,200 MB**.

### Detailed Memory Model: AdamW Baseline vs. Hybrid Muon

*Baseline AdamW is EMPIRICALLY MEASURED at Step 5,600; Hybrid Muon is a modeled ESTIMATE.*

| Memory Component | Pure AdamW (Measured Baseline) | Hybrid Muon + AdamW (ESTIMATE) | Modeled Delta |
| :--- | :--- | :--- | :--- |
| **Model Parameters (BF16)** | 1,212.78 MB [Measured] | 1,212.78 MB | 0.0 MB |
| **Gradients (BF16)** | 1,212.78 MB [Measured] | 1,212.78 MB | 0.0 MB |
| **Optimizer States (FP32)** | **4,851.13 MB** (8B/param) [Measured] | **2,837.87 MB** (4B/8B hybrid) [ESTIMATE] | **-2,013.26 MB** |
| **Activation Cache (Grad Ckpt)** | ~1,850.00 MB [Measured] | ~1,850.00 MB [ESTIMATE] | 0.0 MB |
| **CUDA Context & Allocator** | ~230.00 MB [Measured] | ~250.00 MB [ESTIMATE] | +20.0 MB |
| **Temporary Workspace** | ~64.00 MB [Measured] | ~80.00 MB [ESTIMATE] | +16.0 MB |
| **Total Peak Allocated VRAM** | **9,420 MB (~9.20 GB)** [Measured] | **~7,450–7,800 MB** [ESTIMATE] | **~ -1,600 to -1,970 MB** |
| **Safe Headroom (to 11,200 MB)** | **1,780 MB (15.9%)** [Measured] | **~3,400–3,750 MB (30–33%)** [ESTIMATE] | **+1,600 to +1,970 MB** |

---

## 7. Implementation Feasibility Classification

| Optimizer | Hardware Feasibility | Codebase Integration Feasibility | Overall Category | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Hybrid Muon + AdamW** | **GREEN (~7.5–7.8 GB peak)** | **GREEN (Drop-in hybrid wrapper)** | **TOP EXPERIMENTAL CANDIDATE** | Modeled state reduction (-2 GB), negligible compute overhead (<1.5%), proven scaling on 16B BF16 MoE. |
| **Sophia-G** | **YELLOW (~10.9 GB peak)** | **YELLOW (Requires custom Hessian step)** | **WORTH TESTING** | Second-order adaptation; transient Gauss-Newton backpropagation spikes VRAM near 11 GB, leaving ~350 MB margin. |
| **Newton-Muon** | **GREEN (~9.0 GB peak)** | **RED (Complex activation covariance hooks)** | **EXPLORATORY** | Requires tracking layer input covariances under gradient checkpointing for modest (+4%) reported gain. |
| **MONA** | **YELLOW (~9.5 GB peak)** | **GREEN (Muon + gradient difference buffer)** | **EXPLORATORY** | Fits in memory, but extra gradient buffer eliminates Muon's VRAM savings. |
| **Lion** | **GREEN (~7.0 GB peak)** | **GREEN (Standard PyTorch optim)** | **NOT CURRENTLY WORTH TESTING** | Lowest VRAM, but sign-momentum risks severe chattering against AbsMean ternary STE quantization boundaries. |
| **SOAP** | **RED (>13.6 GB peak)** | **RED (Complex Kronecker decomposition)** | **REJECTED** | Immediate CUDA OOM on 12GB GPU. |
| **Schedule-Free AdamW** | **RED (>12.4 GB peak)** | **GREEN (Drop-in replacement)** | **REJECTED** | Dual iterate buffers exceed 12GB VRAM capacity. |
