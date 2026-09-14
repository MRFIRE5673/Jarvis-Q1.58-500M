# Deterministic Numerical Validation: Model A vs Model B

## Overview

To rigorously validate the numerical implementation differences between **Model A** (current production AbsMean baseline) and **Model B** (Paper Eq. 3 literal implementation), a controlled, deterministic single-step mini-test was conducted on identical hardware (NVIDIA GeForce RTX 5070 12GB).

The script executing this verification is located at [scratch/numerical_validation_mini_test.py](file:///e:/Jarvis-Q1.58-500M/scratch/numerical_validation_mini_test.py).

### Control Conditions
- **Base Checkpoint**: Bit-identical load from [experiment_start.pt](file:///e:/Jarvis-Q1.58-500M/experiments/ternary_paper_ab/experiment_start.pt)
- **Input Tensor**: $B = 2, T = 64$, fixed seed `torch.manual_seed(42)`
- **Targets**: Fixed seed token indices
- **Precision**: `torch.amp.autocast('cuda', dtype=torch.bfloat16)` with identical FP32 master weights
- **Routing**: Top-2 expert routing with fixed seed `100`

---

## Numerical Discrepancy Results

| Metric | Model A (AbsMean) | Model B (Paper Eq. 3) | Difference / Metric |
| :--- | :--- | :--- | :--- |
| **Forward Loss** | 13.88780 | 11.55628 | $\Delta = 2.33152$ |
| **Logits $L_\infty$ Difference** | — | — | **13.00000** |
| **Logits $L_2$ Norm Difference** | — | — | **11,661.21484** |
| **Gradient Max Absolute Delta ($\|g_A - g_B\|_\infty$)** | — | — | **0.26595** |

---

## Mathematical and Structural Analysis

### 1. Why Logits Differ Dramatically ($L_\infty = 13.0$, $L_2 = 11,661.2$)
In Model A:
- Each linear projection weight $W$ is scaled by its mean magnitude $\alpha = \text{mean}(|W|) \approx 0.0193$.
- $W_q \in \{-\alpha, 0, +\alpha\}$.
- Attention projections ($Q, K, V, \text{Out}$) and MoE expert projections ($W_1, W_2$) produce active hidden transformations with non-zero activations.

In Model B:
- Master FP32 weights have standard deviation $\sigma \approx 0.024$ and maximum magnitude $|W|_{\max} \approx 0.184$.
- Because $|W| < 0.50$ everywhere, $W_q = \text{round}(\text{clamp}(W, -1, 1)) \equiv 0.0000$.
- In every one of the 24 transformer blocks:
  $$\text{attn\_out}(x) = 0, \quad \text{moe\_out}(x) = 0$$
- The forward pass degenerates into a sequence of residual skips without attention mixing or feedforward modulation:
  $$x_{l+1} = x_l + 0 + 0 = x_l$$
- The output logits reflect solely the token embedding table mapped through the final RMSNorm and LM head projection:
  $$\text{logits}_B = \text{lm\_head}(\text{final\_norm}(\text{tok\_emb}(x)))$$
- This structural divergence creates an $L_\infty$ distance of $13.0$ and an $L_2$ norm difference of $11,661.2$ across the $2 \times 64 \times 50257$ logit tensor.

### 2. Gradient Behavior Under Eq. 5 STE
The backward gradient under Paper Eq. 5 is:
$$\frac{\partial \mathcal{L}}{\partial W_{\text{FP32}}} = \frac{\partial \mathcal{L}}{\partial W_q} \cdot \mathbf{1}_{\{|W_{\text{FP32}}| \le 1.0\}}$$

Because all master weights satisfy $|W_{\text{FP32}}| \le 0.184 < 1.0$, the clipping condition mask is:
$$\mathbf{1}_{\{|W_{\text{FP32}}| \le 1.0\}} \equiv 1.0 \quad (\text{100\% unclipped})$$

Therefore, the backward pass propagates gradients directly to $W_{\text{FP32}}$ without any clipping attenuation. However, the incoming gradient $\frac{\partial \mathcal{L}}{\partial W_q}$ is computed with respect to a forward pass where all downstream linear activations were zero. This creates a maximum parameter gradient discrepancy of $0.26595$.

---

## Verification of Correct Implementation
The goal of this numerical test was not to match outputs (since Model A and Model B have deliberately distinct forward equations), but to verify:
1. **Paper Eq. 3 is strictly obeyed**: Forward quantization produces integer values in $\{-1, 0, +1\}$ without any $\alpha$ multiplier.
2. **Backward STE operates directly on $W$**: Gradient mask applies to $|W| \le 1.0$ without normalization.
3. **No runtime divergence or NaN/Inf**: Both variants complete forward and backward passes stably without numerical faults.
