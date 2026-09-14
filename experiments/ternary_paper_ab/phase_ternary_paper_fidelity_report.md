# Phase Report: Jarvis Paper Ternary Quantization Fidelity A/B Experiment

## 1. Executive Summary

This study investigates whether the canonical Jarvis Q1.58-500M model should replace its current **AbsMean-scaled ternary quantization** with the **literal quantization equation specified in the original Jarvis research paper** (Paper Eq. 3):
$$W_q = \text{round}(\text{clamp}(W_{\text{FP32}}, -1, 1)) \in \{-1, 0, +1\}$$

This was conducted as a strict, controlled training/quality experiment on an NVIDIA GeForce RTX 5070 12GB GPU. Both variants started from the exact same pre-trained FP32 master checkpoint ([experiment_start.pt](file:///e:/Jarvis-Q1.58-500M/experiments/ternary_paper_ab/experiment_start.pt)), using identical data sequences, optimizer hyper-parameters, sequence lengths ($T=512$), and batch configurations ($B=4, \text{accum}=2 \implies 4,096\text{ tokens/update}$).

### Core Finding
Under standard continuous weight initialization scales ($\sigma \approx 0.02 - 0.03$), **Paper Eq. 3 suffers total, catastrophic zero-collapse (Outcome C)**:
- **100.00% of all ternary projection weights evaluate to 0** across all 24 layers.
- Master FP32 weights have $|W_{\text{FP32}}| \le 0.184 \ll 0.50$, falling entirely within the $(-0.50, +0.50)$ rounding dead zone.
- Validation perplexity exploded by **1,722x at Step 0** ($34.16 \to 58,829.00$) and remained **31.3x worse after 100 updates** ($45.04$ vs $1,407.86$).
- Generated text under Paper Eq. 3 degenerated into meaningless subword soup.
- **Scientific Conclusion**: AbsMean (or an equivalent scale adaptation mechanism) is mathematically indispensable for training deep ternary networks initialized with continuous variances. Jarvis must retain AbsMean for production training.

---

## 2. Theoretical Formulation & Paper Fidelity Classification

### Current Implementation (Model A: Production Baseline)
$$\alpha = \text{mean}(|W_{\text{FP32}}|)$$
$$W_{\text{norm}} = \frac{W_{\text{FP32}}}{\alpha}$$
$$W_q = \text{round}(\text{clamp}(W_{\text{norm}}, -1, 1)) \times \alpha \in \{-\alpha, 0, +\alpha\}$$
$$\text{STE Backward}: \frac{\partial \mathcal{L}}{\partial W_{\text{FP32}}} \approx \frac{\partial \mathcal{L}}{\partial W_q} \cdot \mathbf{1}_{\left\{\left|\frac{W}{\alpha}\right| \le 1\right\}}$$

**Classification**:
> *"Engineering/BitNet-style ternary variant; not literal Jarvis Eq. 3."*

### Paper-Faithful Implementation (Model B: Literal Eq. 3 & Eq. 5)
$$W_q = \text{round}(\text{clamp}(W_{\text{FP32}}, -1, 1)) \in \{-1, 0, +1\}$$
$$\text{STE Backward}: \frac{\partial \mathcal{L}}{\partial W_{\text{FP32}}} \approx \frac{\partial \mathcal{L}}{\partial W_q} \cdot \mathbf{1}_{\{|W_{\text{FP32}}| \le 1\}}$$

**Classification**:
> *"Direct implementation of the ternarization equation specified in Jarvis, subject only to the explicitly documented STE used for training."*

---

## 3. Pre-Flight Weight Scale Forensics

Before launching training on Model B, the pre-flight distribution of FP32 master weights across all architectural sub-modules was audited:

| Component | Mean Abs $\text{mean}(\|W\|)$ | Standard Dev $\sigma$ | Min Value | Max Value | $\%$ in $[-0.5, +0.5]$ | Zero % under Eq. 3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **tok_emb (Embeddings)** | 0.0401 | 0.0538 | -0.6865 | +0.6723 | 99.98% | *(Continuous BF16)* |
| **Block 0 QKV Proj** | 0.0194 | 0.0244 | -0.1601 | +0.1587 | **100.00%** | **100.00%** |
| **Block 0 Out Proj** | 0.0196 | 0.0246 | -0.1654 | +0.1493 | **100.00%** | **100.00%** |
| **Block 0 MoE W1 Expert 0** | 0.0198 | 0.0249 | -0.1837 | +0.1772 | **100.00%** | **100.00%** |
| **Block 0 MoE W2 Expert 0** | 0.0163 | 0.0205 | -0.1610 | +0.1578 | **100.00%** | **100.00%** |
| **Block 12 QKV Proj** | 0.0194 | 0.0244 | -0.1544 | +0.1522 | **100.00%** | **100.00%** |
| **Block 23 MoE W1 Expert 3** | 0.0198 | 0.0248 | -0.1748 | +0.1691 | **100.00%** | **100.00%** |
| **lm_head (LM Head)** | 0.0134 | 0.0175 | -0.1648 | +0.1627 | 100.00% | *(Continuous BF16)* |

**Key Diagnostic Insight**: In deep transformers, linear projection weights are initialized according to Xavier/He variance bounds ($\sigma \approx \frac{1}{\sqrt{d_{\text{in}}}} \approx 0.031$). Consequently, $100.00\%$ of all FP32 linear layer weights in the model lie strictly within $[-0.19, +0.19]$. Because $\text{round}(x) = 0$ for all $|x| < 0.5$, applying Eq. 3 without $\alpha$ scaling maps every single linear weight to $0.0$.

---

## 4. Empirical 100-Update Training Results

The 100-update A/B experiment was executed with $4,096$ real tokens per optimizer step ($409,600$ total tokens evaluated over $20$ holdout validation windows per checkpoint).

### Trajectory Comparison Table

| Step | Model A Train Loss | Model B Train Loss | Model A Val Loss | Model B Val Loss | Model A Val PPL | Model B Val PPL | Model A Zero % | Model B Zero % | Model A Mean $\alpha$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | — | — | **3.5310** | **10.9824** | **34.16** | **58,829.00** | 29.83% | **100.00%** | 0.01930 |
| **25** | 2.4233 | 6.6324 | **3.7163** | **8.3451** | **41.11** | **4,209.42** | 29.85% | **100.00%** | 0.01936 |
| **50** | 2.0554 | 6.5586 | **3.8723** | **7.8309** | **48.05** | **2,517.28** | 29.87% | **100.00%** | 0.01941 |
| **75** | 2.8752 | 6.3993 | **3.7718** | **7.6445** | **43.46** | **2,089.08** | 29.88% | **100.00%** | 0.01945 |
| **100** | **1.9855** | **5.5790** | **3.8076** | **7.2498** | **45.04** | **1,407.86** | 29.90% | **100.00%** | 0.01950 |

### Text Generation Quality at Step 100
- **Model A (AbsMean)**:
  `"The quantum computing architecture (e-g) to use the ..."`
  `"self.client.get('/test_admin/admin/logout/')"`
  *Output displays coherent English and Python syntax matching the training corpus.*
- **Model B (Paper Eq. 3)**:
  `"The quantum computing architectureools requestfileobj = self. ..."`
  `"The quantum computing architecture Andersen endsather wraps BIG34 apologtt 324*ummy448radorHandler..."`
  *Output consists of disconnected subword tokens reflecting only marginal unigram statistics.*

---

## 5. Deterministic Numerical Verification

A deterministic single-step mini-test ($B=2, T=64$, identical weights and seeds) recorded:
- Forward Loss Delta: $|13.88780 - 11.55628| = 2.33152$
- Logit Maximum Difference ($L_\infty$): **13.00000**
- Logit $L_2$ Difference: **11,661.21484**
- Gradient Max Absolute Difference ($L_\infty$): **0.26595**

In Model B, all 24 layers evaluate to identity bypasses ($W_q x = 0$). Logits are generated entirely by the continuous token embedding mapped directly through final RMSNorm and LM head projection.

---

## 6. Answers to Mandatory Evaluation Questions

### 1. What exactly does the paper specify?
The original Jarvis paper specifies:
$$W_f = \text{round}(\text{clamp}(W_{\text{FP32}}, -1, 1)) \in \{-1, 0, +1\}$$
with backward Straight-Through Estimator:
$$\frac{\partial \mathcal{L}}{\partial W_{\text{FP32}}} \approx \frac{\partial \mathcal{L}}{\partial W_f} \cdot \mathbf{1}_{\{|W_{\text{FP32}}| \le 1\}}$$
There is no AbsMean factor, no RMS scaling, and no channel/group scaling factor $\alpha$.

### 2. What exactly does the current implementation do?
The current baseline computes the dynamic mean absolute weight $\alpha = \text{mean}(|W|)$, normalizes weights $W_{\text{norm}} = W / \alpha$, and quantizes:
$$W_q = \text{round}(\text{clamp}(W_{\text{norm}}, -1, 1)) \times \alpha \in \{-\alpha, 0, +\alpha\}$$
with backward STE masked on $|W/\alpha| \le 1.0$.

### 3. How different are their ternary distributions?
Completely divergent:
- **Model A (AbsMean)**: Balanced ternary distribution: $29.9\%$ zero, $35.0\%$ positive ($+\alpha$), $35.1\%$ negative ($-\alpha$).
- **Model B (Paper Eq. 3)**: Complete collapse: **100.00% zero**, $0.00\%$ positive, $0.00\%$ negative.

### 4. Does Eq. 3 train?
Technically the optimizer updates continuous parameters without crashing, but **the ternary transformer backbone does NOT train**. 100% of all ternary linear weights remain permanently pinned at zero. Only the continuous token embeddings and LM head adapt, acting as a 0-layer unigram model.

### 5. Which has lower validation CE?
**Model A (AbsMean)**: Final validation CE is **3.8076** vs **7.2498** for Model B (+3.4422 higher / 90.4% worse for Eq. 3).

### 6. Which has lower PPL?
**Model A (AbsMean)**: Final validation perplexity is **45.04** vs **1,407.86** for Model B (**31.3x worse perplexity for Eq. 3**).

### 7. Which has better generation?
**Model A (AbsMean)**: Retains fluent, syntactically valid English and code structures. Model B emits broken unigram token soup.

### 8. Which has better long-context behavior?
**Model A (AbsMean)**: Model B zeroes all attention projection weights ($Q, K, V, \text{Out}$), eliminating all associative attention and recurrent liquid state memory.

### 9. Which has better throughput?
**Model A (AbsMean)** achieved higher throughput:
- **Variant A (AbsMean)**: **3,791.45 ms/update (1,080.3 tok/s)**
- **Variant B (Paper Eq. 3)**: **4,330.44 ms/update (945.9 tok/s)**
- **Delta**: **-12.45% throughput penalty for Paper Eq. 3**. Variant B executed unfused autograd operations (`clamp`, `round`, discrete tensor masks) across 144 linear projections per microstep, resulting in lower GPU compute intensity (77.0% vs 87.2% GPU utilization) and additional kernel launch overhead compared to the fused kernel in Variant A.

### 10. Which uses less VRAM?
**Identical**:
- **Variant A (AbsMean)**: **9,526.77 MiB** active allocated (12,386.00 MiB reserved)
- **Variant B (Paper Eq. 3)**: **9,527.07 MiB** active allocated (13,132.00 MiB reserved)
Peak active memory usage differs by less than 0.3 MiB (0.003%), as both variants maintain identical FP32 master weights, activations, and AdamW optimizer state buffers.

### 11. Does Eq. 3 create saturation/collapse?
Yes — **Eq. 3 produces total, immediate zero-collapse (Outcome C)** due to continuous weights lying well within $[-0.19, +0.19] \subset (-0.5, +0.5)$.

### 12. Does Eq. 3 require different initialization?
**Yes.** Standard Xavier/He initialization ($\sigma \approx 0.02 - 0.03$) guarantees zero collapse. Eq. 3 requires large-variance initialization ($\sigma \sim 0.65 - 0.80$) or discrete ternary initialization to avoid collapse.

### 13. Is AbsMean actually necessary for Jarvis?
**Yes.** AbsMean is mathematically essential to map continuous weight matrices to the discrete rounding threshold $\pm 0.50$ across deep architectures.

### 14. Should Jarvis remain AbsMean or move to literal Eq. 3?
**Jarvis MUST remain AbsMean for production training.** Adopting literal Eq. 3 at the current scale completely destroys model capacity.

### 15. What experiment should happen next?
A dedicated follow-up study: **"Large-Variance Scale-Calibrated Initialization for Literal Eq. 3"**, evaluating whether initializing weights at $\sigma \approx 0.70$ with layer-wise $1/\sqrt{d_{\text{model}}}$ gain compensation enables stable training without dynamic $\alpha$.
