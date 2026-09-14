# Training Comparison: AbsMean Baseline (A) vs Paper Eq. 3 (B)

## Executive Summary

This report documents the empirical 100-update A/B training comparison on the canonical Jarvis Q1.58-500M architecture ($B=4, T=512, \text{accum}=2 \implies 4,096\text{ tokens/update}$, 24 layers, $d_{\text{model}}=1024$, 16 heads, 4 experts, Top-2 routing) executed on an NVIDIA GeForce RTX 5070 12GB GPU.

Both branches began from the exact same pre-trained master checkpoint ([experiment_start.pt](file:///e:/Jarvis-Q1.58-500M/experiments/ternary_paper_ab/experiment_start.pt)) and identical data batches / random seeds.

| Metric | Model A (AbsMean Baseline) | Model B (Paper Eq. 3) | Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Initial Val Loss (Step 0)** | 3.5310 | 10.9824 | **+7.4514 (+211.0%)** |
| **Initial Val PPL (Step 0)** | 34.16 | 58,829.00 | **1,722x Perplexity Explosion** |
| **Final Val Loss (Step 100)** | 3.8076 | 7.2498 | **+3.4422 (+90.4%)** |
| **Final Val PPL (Step 100)** | 45.04 | 1,407.86 | **31.3x Worse Perplexity** |
| **Final Train Loss (Step 100)** | 1.9855 | 5.5790 | **+3.5935 (+181.0%)** |
| **Zero Weight Percentage** | 29.83% - 29.90% | **100.00%** | **Total Quantization Collapse** |
| **Positive (+1) Percentage** | 35.03% - 35.07% | **0.00%** | All positive weights zeroed |
| **Negative (-1) Percentage** | 35.07% - 35.10% | **0.00%** | All negative weights zeroed |
| **Scaling Factor $\alpha$** | $0.01930 \to 0.01950$ | None (Paper Eq. 3 has no $\alpha$) | N/A |
| **Gradient Norm (Step 100)** | 1.515 | 1.430 | Both clipped at 1.0 |
| **NaN / Inf Occurrences** | 0 | 0 | None observed |
| **Text Generation Coherence** | Syntactic English / Code | Fragmented subword soup | Massive semantic degradation |

---

## Empirical Trajectory Progression

The detailed trajectory metrics recorded in [ab_results.csv](file:///e:/Jarvis-Q1.58-500M/experiments/ternary_paper_ab/ab_results.csv) are summarized below:

### Model A: AbsMean Baseline ($\alpha = \text{mean}(|W|), W_q \in \{-\alpha, 0, +\alpha\}$)

| Update | Train Loss | Val Loss | Val PPL | Grad Norm | Zero % | Pos % | Neg % | Mean $\alpha$ | Generated Sample Prefix |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | — | 3.5310 | 34.16 | 0.000 | 29.83% | 35.07% | 35.10% | 0.01930 | `The quantum computing architecture.` |
| **25** | 2.4233 | 3.7163 | 41.11 | 1.688 | 29.85% | 35.05% | 35.09% | 0.01936 | `The quantum computing architecture.` |
| **50** | 2.0554 | 3.8723 | 48.05 | 1.305 | 29.87% | 35.04% | 35.09% | 0.01941 | `The quantum computing architecture. self.client.get('/test_admin/admin/logout/')` |
| **75** | 2.8752 | 3.7718 | 43.46 | 1.592 | 29.88% | 35.04% | 35.08% | 0.01945 | `The quantum computing architecture.` |
| **100** | 1.9855 | 3.8076 | 45.04 | 1.515 | 29.90% | 35.03% | 35.07% | 0.01950 | `The quantum computing architecture (e-g) to use the` |

### Model B: Paper Eq. 3 ($W_q = \text{round}(\text{clamp}(W, -1, 1)) \in \{-1, 0, +1\}$)

| Update | Train Loss | Val Loss | Val PPL | Grad Norm | Zero % | Pos % | Neg % | Mean $\alpha$ | Generated Sample Prefix |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | — | 10.9824 | 58,829.00 | 0.000 | **100.00%** | **0.00%** | **0.00%** | 0.00000 | `The quantum computing architecture Andersen endsather wraps BIG34 apologtt 324*ummy448radorHandler` |
| **25** | 6.6324 | 8.3451 | 4,209.42 | 5.051 | **100.00%** | **0.00%** | **0.00%** | 0.00000 | `The quantum computing architecture Andersen endsatherpoolilt (""LR neutralityiletwriteelectric05L` |
| **50** | 6.5586 | 7.8309 | 2,517.28 | 1.429 | **100.00%** | **0.00%** | **0.00%** | 0.00000 | `The quantum computing architecture Andersen endsCODEwargs img').,ash plurality Forbidden Blast` |
| **75** | 6.3993 | 7.6445 | 2,089.08 | 1.487 | **100.00%** | **0.00%** | **0.00%** | 0.00000 | `The quantum computing architecture Andersen ends gens blasphemyconcatalogma 213ENSE established` |
| **100** | 5.5790 | 7.2498 | 1,407.86 | 1.430 | **100.00%** | **0.00%** | **0.00%** | 0.00000 | `The quantum computing architectureools requestfileobj = self.` |

---

## Root Cause Analysis: Total Quantization Collapse Under Eq. 3

### 1. The Discretization Threshold Problem
In Paper Eq. 3, the quantization function is:
$$W_q = \text{round}(\text{clamp}(W_{\text{FP32}}, -1, 1))$$
Because $\text{round}(x)$ maps any value in $(-0.50, +0.50)$ to $0$:
$$W_q = 0 \quad \forall W_{\text{FP32}} \in (-0.50, +0.50)$$

However, deep neural networks (including Jarvis 24-layer 500M) initialize linear projection weights with standard standard deviations scaled as $\sigma \approx \frac{1}{\sqrt{d_{\text{in}}}} \approx \frac{1}{\sqrt{1024}} \approx 0.03125$.
In the master pre-trained checkpoint:
- Attention QKV weights: $\text{std} = 0.0244$, $\max(|W|) = 0.1601 \ll 0.50$
- Attention Out proj: $\text{std} = 0.0246$, $\max(|W|) = 0.1654 \ll 0.50$
- MoE W1 expert gates: $\text{std} = 0.0249$, $\max(|W|) = 0.1837 \ll 0.50$
- MoE W2 expert projections: $\text{std} = 0.0205$, $\max(|W|) = 0.1610 \ll 0.50$

**Result**: Every single weight in every single ternary linear projection layer was strictly below $0.19$ in magnitude. As a consequence, **100.00% of all ternary weights collapsed to 0**.

### 2. Why AbsMean Prevents Collapse
Under AbsMean:
$$\alpha = \text{mean}(|W|) \approx 0.0193$$
$$W_{\text{norm}} = \frac{W}{\alpha}$$
Because $W$ is scaled by its own mean absolute value, $W_{\text{norm}}$ has mean absolute value equal to $1.0$. The values naturally span $[-6, +6]$, yielding:
- $\sim 30\%$ zeroes (values in $[-0.5\alpha, +0.5\alpha]$)
- $\sim 35\%$ $+1 \times \alpha$
- $\sim 35\%$ $-1 \times \alpha$
This preserves active signal propagation throughout all 24 layers of the transformer.

### 3. Why Model B Still Shows Some Loss Reduction
In Model B, train loss decreased from $6.63$ to $5.58$ and validation loss decreased from $10.98$ to $7.25$. 
Crucially, **Zero % remained 100.00% across all 100 steps**. Why did the loss improve at all?
- The model's token embeddings (`tok_emb`, continuous BF16), final RMSNorm (`final_norm`, continuous BF16), and language model projection head (`lm_head`, continuous BF16) are **not** ternary linear layers.
- With all 24 transformer layers outputting zero vectors (acting as dead bypasses or identity skip additions), the continuous parameters optimized unigram token frequency distributions (acting as an uncontextualized n-gram / bag-of-words language model).
- The language model capacity was completely truncated to a 0-layer linear model, explaining why perplexity could not improve past $1,407.86$ (compared to AbsMean's $45.04$).

---

## Scientific Conclusion

At standard initialization scales ($\sigma \approx 0.02 - 0.03$), **Paper Eq. 3 experiences an immediate, total quantization collapse to all zeroes**. 
AbsMean is not merely an engineering convenience; it is an essential mathematical scale adaptation mechanism required to match the dynamic range of continuous weight matrices to the discrete threshold $[-0.5, 0.5]$ of integer rounding.
