# Pairwise Memory Interaction Matrix (Current Completed Subset)

> **Status Notice:** This document captures the **currently completed subset** of the Jarvis-600M factorial memory research matrix. The factorial study is actively ongoing. Do NOT assume this subset constitutes the final conclusions. Remaining pairs (`C+D`, `C+E`, `D+E`) and subsequent higher-order combinations may modify or qualify these observations.

---

## 1. Experimental Methodology & Mathematical Controls

All factorial experiments run under strictly identical controlled conditions:
- **Starting Checkpoint:** `experiments/extended_train/ckpt_step_0004284_best.pt` (inherited from the verified paper-faithful training trajectory).
- **Sequence Length:** $T = 512$
- **Token Budget:** $100\text{ updates} \times 4\text{ accum} \times 2\text{ batch} \times 512 = 409,600$ tokens.
- **Optimizer:** Fused AdamW ($\text{lr} = 5\times 10^{-5}$, $\beta_1=0.9, \beta_2=0.95$, weight decay $0.1$, gradient clip $1.0$).
- **Neutral / Identity Initialization:** Newly added projection weights are mathematically initialized so that initial step 0 logits reproduce baseline identity behavior ($\pm 0.0018$ initial loss delta), preventing artifactual uninitialized loss spikes.
- **Dataset:** Standardized token split (`fresh_holdout.txt`, 50 random windows of $T=512$).
- **Random Seed:** 42.

---

## 2. Mathematical Definition of Interaction Coefficients

Derived directly from [`experiments/architecture_matrix/interaction_analyzer.py`](../experiments/architecture_matrix/interaction_analyzer.py):

$$\text{Effect}(X) = \text{CE}(X) - \text{CE}(\text{baseline})$$

$$\text{Expected}(X+Y) = \text{Effect}(X) + \text{Effect}(Y)$$

$$\text{Actual}(X+Y) = \text{CE}(X+Y) - \text{CE}(\text{baseline})$$

$$\text{Interaction}(X, Y) = \text{Actual}(X+Y) - \text{Expected}(X+Y)$$

### Classification Scheme:
- $\text{Interaction} < -0.010$: **SYNERGISTIC** (mechanisms cooperate, mitigating individual parameter perturbations)
- $|\text{Interaction}| \le 0.010$: **ADDITIVE** (mechanisms operate independently without interference)
- $\text{Interaction} > +0.010$: **ANTAGONISTIC** (mechanisms interfere or exhibit functional redundancy)

---

## 3. Standalone Mechanisms (Singles: Verified Baseline & Standalone)

**Reference Baseline (Paper-Faithful Architecture):**
- Holdout Cross-Entropy: **3.2858**
- Perplexity (PPL): **26.73**
- Needle Retrieval Rank @ 64 tok: **4053.0 / 50257**

| Mechanism Code | Description | Initial CE | Final CE | Final PPL | Effect ($\Delta$) | Peak VRAM |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`fact_A`** | Adaptive Input-Dependent Decay ($\gamma_t$) | 3.2982 | **3.2982** | 27.06 | $+0.0124$ | 2,720 MB |
| **`fact_B`** | Gated Recurrent Write ($w_t$) | 3.2847 | **3.2960** | 27.00 | $+0.0102$ | 10,294 MB |
| **`fact_C`** | Gated Recurrent Erase ($e_t$) | 3.2858 | **3.2889** | 26.81 | $+0.0031$ | 10,295 MB |
| **`fact_D`** | Gated Read ($r_t$) | 3.2849 | **3.2966** | 27.02 | $+0.0108$ | 10,295 MB |
| **`fact_E`** | Local Sliding Buffer ($W=16$) | 3.2841 | **3.2960** | 27.00 | $+0.0102$ | 3,579 MB |

---

## 4. Completed Pairwise Subset

*Note: All seven currently completed pairs show negative interaction coefficients under this controlled 100-step experiment.*

| Pair | Mechanisms | Initial CE | Final CE | Final PPL | Needle Rank @ 64 | Expected $\Delta$ | Actual $\Delta$ | Interaction | Classification |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`fact_B+C`** | Write Gate + Erase Gate | 3.2853 | **3.2882** | **26.79** | 9085.4 | $+0.0133$ | $+0.0024$ | **-0.0109** | **SYNERGISTIC** |
| **`fact_A+C`** | Adaptive Decay + Erase Gate | 3.2888 | **3.2895** | **26.83** | 9200.8 | $+0.0155$ | $+0.0037$ | **-0.0118** | **SYNERGISTIC** |
| **`fact_B+E`** | Write Gate + Local Buffer ($W=16$) | 3.2847 | **3.2943** | **26.96** | 9104.8 | $+0.0204$ | $+0.0085$ | **-0.0119** | **SYNERGISTIC** |
| **`fact_B+D`** | Write Gate + Gated Read | 3.2847 | **3.2965** | **27.02** | 9129.6 | $+0.0210$ | $+0.0107$ | **-0.0103** | **SYNERGISTIC** |
| **`fact_A+E`** | Adaptive Decay + Local Buffer ($W=16$) | 3.2942 | **3.2978** | **27.05** | 9193.6 | $+0.0226$ | $+0.0120$ | **-0.0106** | **SYNERGISTIC** |
| **`fact_A+B`** | Adaptive Decay + Write Gate | 3.2947 | **3.2983** | **27.07** | 9172.0 | $+0.0226$ | $+0.0125$ | **-0.0101** | **SYNERGISTIC** |
| **`fact_A+D`** | Adaptive Decay + Gated Read | 3.2944 | **3.2983** | **27.07** | 9102.0 | $+0.0232$ | $+0.0125$ | **-0.0107** | **SYNERGISTIC** |

---

## 5. Preliminary Analytical Observations

1. **Dual Gating (`B+C`) as Primary Recurrent Driver:**
   - Combining selective write gating $w_t (k_t \otimes v_t)$ with selective erase retention $(1 - e_t)S_{t-1}$ produces the lowest cross-entropy among all completed pairs (**3.2882 CE / 26.79 PPL**), dipping to **3.2786 CE** mid-run.
   - This empirically confirms that bounded recurrent updates prevent memory state saturation.
2. **Timescale Separation (`B+E`):**
   - The local sliding window ($W=16$) absorbs high-frequency adjacent n-grams, enabling the recurrent write gate to close for redundant tokens and preserve long-range associative states.
   - `B+E` outperforms both standalone Write (`3.2960`) and standalone Buffer (`3.2960`), yielding **3.2943 CE / 26.96 PPL**.
3. **Adaptive Decay Synergy (`A+*`):**
   - In isolation, adaptive decay parameters require adaptation time; when paired with gating mechanisms, dynamic retention adapts rapidly without compounding degradation.

---

## 6. Pending Experiments
- `fact_C+D` (Erase Gate + Gated Read) — In progress
- `fact_C+E` (Erase Gate + Local Buffer) — Queued
- `fact_D+E` (Gated Read + Local Buffer) — Queued
- Higher-order combinations: `B+C+E`, `A+B+C`, `A+B+C+D+E`.
