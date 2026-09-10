# Scientific Methodology & Controlled Experiment Protocol

## 1. Core Principles & Baseline Immutability

Jarvis research adheres to strict scientific control principles:
1. **Baseline Immutability:** The paper-faithful baseline checkpoint (`jarvis_engine/ckpt_step_0004209.pt`) and model definition (`AssociativeLinearAttention` in `jarvis_model.py`) remain completely untouched and immutable.
2. **Standardized Comparison Foundation:** All architectural variants, ablations, and factorial matrix experiments originate from the identical verified training checkpoint (`experiments/extended_train/ckpt_step_0004284_best.pt`), ensuring differences reflect architectural mechanics rather than starting weight variations.
3. **Fixed Token & Compute Budgets:** Each experiment consumes precisely **409,600 tokens** across 100 optimization steps ($100\text{ steps} \times 4\text{ accum} \times 2\text{ batch} \times 512\text{ tokens}$).
4. **No Fabricated Data:** Every recorded cross-entropy loss, perplexity, throughput, and needle rank is mechanically saved from automated test runs into JSON artifacts and logged in `experiments/research_database.json`.

---

## 2. Experimental Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Initial Checkpoint** | `ckpt_step_0004284_best.pt` | Verified 606M parameter checkpoint on clean data split |
| **Sequence Length ($T$)** | 512 | Matches standard pre-training context length |
| **Batch Size** | 2 sequences | GPU memory constraint on 12GB RTX 5070 |
| **Gradient Accumulation** | 4 steps | Effective batch = 8 sequences = 4,096 tokens/step |
| **Optimizer** | Fused AdamW | Single CUDA kernel launch per parameter group |
| **Learning Rate** | $5.0 \times 10^{-5}$ | Conservative fine-tuning rate preventing catastrophic forgetting |
| **Betas & Weight Decay** | $\beta_1=0.9, \beta_2=0.95, \lambda=0.1$ | Standard transformer pre-training settings |
| **Gradient Clipping** | Max norm 1.0 | Prevents gradient explosion in recurrent scans |
| **Precision** | BF16 (bfloat16) | Automatic mixed precision with zero underflow |
| **Random Seed** | 42 | Deterministic batch indexing and validation windows |

---

## 3. Mathematical Neutral / Identity Initialization

To eliminate artifactual step-0 loss spikes caused by randomly initialized weights, newly added memory gates are initialized to identity behavior:

- **Write Gate ($w_t$):**
  $$W_w = \mathbf{0}, \quad b_w = +4.0 \implies w_t = \sigma(4.0) \approx 0.982 \quad (\text{near fully open})$$
- **Erase Gate ($e_t$):**
  $$W_e = \mathbf{0}, \quad b_e = -4.0 \implies e_t = \sigma(-4.0) \approx 0.018 \implies 1 - e_t \approx 0.982 \quad (\text{near full retention})$$
- **Gated Read ($r_t$):**
  $$W_r = \mathbf{0}, \quad b_r = +4.0 \implies r_t = \sigma(4.0) \approx 0.982 \quad (\text{near transparent pass-through})$$
- **Adaptive Decay ($\gamma_t$):**
  $$W_\gamma = \mathbf{0}, \quad b_\gamma = \text{logit}\left(\frac{\gamma_{\text{base}} - \gamma_{\min}}{\gamma_{\max} - \gamma_{\min}}\right) \implies \gamma_t \equiv \gamma_{\text{base}} = 0.950$$
- **Local Buffer Fusion Gate ($g$):**
  $$g = \sigma(-3.0) \approx 0.047 \quad (\text{recurrent-dominant init, smooth ramp-up})$$

**Verification:** In initial zero-step holdout evaluation, models with newly added mechanisms achieve initial CE matching the paper baseline within $\pm 0.0018$ points (3.1548 vs 3.1566).

---

## 4. Interaction Coefficient Calculation

Derived directly from [`experiments/architecture_matrix/interaction_analyzer.py`](../experiments/architecture_matrix/interaction_analyzer.py):

$$\text{Effect}(X) = \text{CE}(X) - \text{CE}(\text{baseline})$$

$$\text{Expected}(X+Y) = \text{Effect}(X) + \text{Effect}(Y)$$

$$\text{Actual}(X+Y) = \text{CE}(X+Y) - \text{CE}(\text{baseline})$$

$$\text{Interaction}(X, Y) = \text{Actual}(X+Y) - \text{Expected}(X+Y)$$

### Classification Thresholds:
- **SYNERGISTIC ($\text{Interaction} < -0.010$):** The combination achieves substantially lower loss than the sum of independent effects, indicating constructive functional cooperation.
- **ADDITIVE ($|\text{Interaction}| \le 0.010$):** The mechanisms operate independently without mutual reinforcement or interference.
- **ANTAGONISTIC ($\text{Interaction} > +0.010$):** The combination underperforms expected independent effects, indicating functional redundancy or conflicting gradient updates.

---

## 5. Standardized Canonical Evaluation Suite

Every experiment undergoes an automated 4-stage evaluation post-training:
1. **Multi-Scale Language Modeling:** Holdout cross-entropy, perplexity, and bits-per-character across context lengths $T \in \{256, 512, 1024\}$ on 50 non-overlapping windows.
2. **Associative Needle Retrieval Stress Suite:**
   - Single-needle retrieval tested across distances $\{16, 64, 128, 256, 512\}$ tokens.
   - Measures Top-1 accuracy, target probability, and mean retrieval rank against the 50,257-token vocabulary.
   - Multi-needle interference and memory overwrite preference tests.
3. **Autoregressive Generation Quality:** 4-gram repetition rate, token entropy (bits), and Python syntax validity check.
4. **Hardware Performance & Resource Profiling:**
   - Prefill throughput (tok/s at $T=512$, batch=2).
   - Autoregressive decode latency (tok/s at $T=1$, batch=1).
   - Exact VRAM decomposition (parameter memory, activation memory, optimizer state, peak allocated).
