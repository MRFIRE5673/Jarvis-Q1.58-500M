# JARVIS — TERNARY QUANTIZATION A/B EXPERIMENT PLAN
## Controlled Scientific Protocol for Quantizer Comparison

---

## 1. Experimental Objectives

To evaluate whether the literal ternarization equation specified in the Jarvis paper (Eq. 3) can substitute for the engineering AbsMean-scaled implementation in actual production training.

### Core Scientific Questions
1. Does Paper Eq. 3 train without numerical instability?
2. Does Paper Eq. 3 achieve competitive validation loss and perplexity?
3. Does Paper Eq. 3 suffer from all-zero collapse, saturation, or dead gradients?
4. What is the throughput and memory impact of removing the AbsMean reduction pass?

---

## 2. Controlled Independent & Dependent Variables

### Independent Variable (Strictly Isolated)
- **Ternary Quantizer**:
  - **Variant A**: AbsMean baseline ($\alpha = \text{mean}(|W|), W_q = \text{round}(\text{clamp}(W/\alpha, -1, 1)) \times \alpha$).
  - **Variant B**: Literal Paper Eq. 3 ($W_q = \text{round}(\text{clamp}(W, -1, 1))$).

### Controlled Constants (Strictly Identical)
- **Starting Weights**: Both variants start from the exact same checkpoint: `experiments/ternary_paper_ab/experiment_start.pt`.
- **Model Architecture**: Jarvis-Q1.58-500M (24 Layers, $d_{\text{model}}=1024$, 16 Heads, 4 MoE Experts, Top-2 Routing).
- **Training Data**: `jarvis_engine/data.txt` (tokenized via tiktoken `gpt2`).
- **Validation Data**: `jarvis_engine/fresh_holdout.txt` (50 held-out evaluation windows).
- **Data Ordering / Random Seed**: Fixed seed = 42 for both runs.
- **Batch Size & Tokens per Update**: $B=4, T=512, \text{accum}=2 \implies 4,096\text{ real tokens/update}$.
- **Optimizer**: Fused AdamW ($\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.01$).
- **Learning Rate**: Peak LR = $3 \times 10^{-4}$ with linear warmup.
- **Gradient Clipping**: Max norm = 1.0.
- **Training Precision**: BF16.
- **Training Duration**: 100 optimizer updates ($409,600$ real tokens).

---

## 3. Evaluation Schedule & Metrics

At steps 0, 25, 50, 75, and 100:
1. **Loss & Perplexity**:
   - Training cross-entropy loss.
   - Holdout validation cross-entropy loss.
   - Holdout validation perplexity ($\text{PPL} = \exp(\text{val\_loss})$).
2. **Optimization Stability**:
   - Gradient $L_2$ norm before clipping.
   - NaN / Inf detection count.
   - Cumulative loss change $\Delta \mathcal{L} = \mathcal{L}_{100} - \mathcal{L}_0$.
3. **Ternary Distribution Metrics**:
   - Percentage of zero weights.
   - Percentage of positive weights ($+1$).
   - Percentage of negative weights ($-1$).
   - Percentage of saturated/clipped weights ($|W| \ge 1.0$).
   - Mean $\alpha$ across layers (Variant A only).
4. **Qualitative Evaluation**:
   - Greedy / temperature-sampled generation output on prompt: *"The quantum computing architecture"*.
5. **Hardware Performance**:
   - Mean step latency (ms) and throughput (tok/s).
   - Peak VRAM allocated and reserved (MiB).
   - Core and memory clocks, temperature, and power.
