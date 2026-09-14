# Jarvis Future Optimizer Bakeoff: Controlled Experimental Protocol

## Executive Summary

This document specifies the audited, four-stage experimental protocol for evaluating candidate training optimizers for the **Jarvis-Q1.58-500M** foundation model. 

### Core Scientific Directive
**No optimizer will be deployed to production training based on paper claims or benchmarks from different architectures.** 

The current **50M-token AdamW baseline run (PID 28148)** serves as the inviolable control. Once that baseline run has completed its full 50,000,000-token trajectory, the candidate optimizers will be evaluated against it under strict, mathematically controlled isolation where **only the optimizer update rules and their calibrated hyperparameters vary**.

---

## 1. Experimental Candidates & Classification

The bakeoff evaluates candidate configurations categorized strictly by the audit criteria:

| Role | Candidate Name | Parameter Partitioning | Key Mechanism | Ranking Category |
| :--- | :--- | :--- | :--- | :--- |
| **CONTROL** | **AdamW Fused** | Unified across all 606.4M parameters | Coordinate-wise adaptive scaling ($\beta_1=0.9, \beta_2=0.95$) | **BASELINE_CONTROL** |
| **CANDIDATE 1** | **Hybrid Muon + AdamW** | 2D matrices (83%): Muon<br>1D/Embeddings (17%): AdamW | 5th-order Newton-Schulz spectral orthogonalization | **TOP EXPERIMENTAL CANDIDATE** |
| **CANDIDATE 2** | **Sophia-G** | Unified across all 606.4M parameters | Stochastic diagonal Gauss-Newton curvature clipping ($k=10$) | **WORTH TESTING** |
| **CANDIDATE 3** | **Newton-Muon** | 2D matrices: Newton-Muon<br>1D/Embeddings: AdamW | Input activation covariance right-preconditioning + Muon | **EXPLORATORY** |
| **CANDIDATE 4** | **MONA** | 2D matrices: MONA<br>1D/Embeddings: AdamW | Gradient difference EMA Nesterov acceleration + Muon | **EXPLORATORY** |

---

## 2. Inviolable Experimental Controls

To eliminate confounding variables, every candidate in every stage must adhere to the following identical conditions:

1. **Identical Starting Checkpoint:** All runs initialize from the identical master checkpoint weights:
   `experiments/extended_train/ckpt_step_0004284_best.pt`.
2. **Identical Dataset Slice:** Training tokens are read sequentially from `data/shards/train_shard_00000.bin` onwards using `ShardedTokenDataset(seed=42)`.
3. **Identical Tokenizer:** Standard GPT-2 Byte-Pair Encoding (50,257 vocabulary).
4. **Identical Model Architecture:** Exactly 606,391,512 parameters, 24 layers, $d_{\text{model}}=1024$, 16 heads, 4 experts, Top-2 routing, AbsMean ternary STE quantization, chunked associative linear attention ($C=64$), and Liquid State Fusion.
5. **Identical Batching & Sequence Geometry:**
   - Sequence length: $T = 512$
   - Micro-batch size: $B = 2$
   - Gradient accumulation steps: $4$
   - Effective batch size: $8$ sequences ($4,096$ tokens per optimizer update)
6. **Identical Evaluation Protocol:**
   - Validation set: `data/shards/val_shard_00000.bin` (held-out, never seen during training).
   - Evaluation cadence: Exactly every 1,250 steps (or end-of-stage).
   - Evaluation batch: 32 deterministic batches (32,768 validation tokens).
7. **Identical Compute Hardware:** Single NVIDIA GeForce RTX 5070 12GB running on PyTorch with BF16 mixed precision and CUDA 12.

---

## 3. Evaluation Metrics: Focus on Quality per GPU-Hour

The bakeoff focuses explicitly on real-world training efficiency:

$$\textbf{Primary Metric:} \quad \textbf{Validation CE vs. GPU Wall-Clock Hours}$$
$$\textbf{Secondary Metric:} \quad \textbf{Validation CE vs. Training Tokens}$$

### Full Diagnostic Suite (15 Metrics)
During all stages, the evaluation harness logs:
1. `train_ce`: Training cross-entropy loss.
2. `val_ce`: Validation cross-entropy on unseen shards.
3. `val_ppl`: Validation perplexity $\exp(\min(\text{val\_ce}, 20.0))$.
4. `wall_clock_sec`: Cumulative elapsed training time in seconds.
5. `tokens_per_sec`: End-to-end token throughput.
6. `step_time_ms`: Total latency per optimizer update step.
7. `optim_time_ms`: Time spent exclusively inside `optimizer.step()`.
8. `peak_vram_allocated_mb`: `torch.cuda.max_memory_allocated() / (1024 * 1024)`.
9. `vram_reserved_mb`: `torch.cuda.memory_reserved() / (1024 * 1024)`.
10. `grad_norm`: Global $L_2$ gradient norm before clipping.
11. `absmean_scale`: Mean absolute value $\alpha = \text{mean}(|W|)$ across ternary layers.
12. `ternary_sat_pct`: Fraction of master weights with $|W| > 1.0$ (measuring STE clipping saturation).
13. `router_entropy`: Shannon entropy of MoE routing probabilities: $-\sum P_i \log P_i$.
14. `l_balance`: MoE load balancing auxiliary loss ($f_i \cdot P_i$).
15. `l_reflect`: Reflective variance penalty.

---

## 4. Four-Stage Experimental Roadmap

```
Stage 1: Cheap Smoke Tests (50–100 Steps / ~400K Tokens / ~15–18 min)
         ├── Verify forward/backward/optimizer, VRAM fits, no NaNs/Infs
         └── Gate criteria: Zero errors, peak VRAM <= 11,000 MB, loss decreases
                                │
                                ▼
Stage 2: 10M-Token Comparison (2,441 Steps / 10.0M Tokens / ~6.7 hrs)
         ├── AdamW Control vs. Surviving Candidates
         └── Primary metric: Val CE vs. GPU Wall-Clock Hours
                                │
                                ▼
Stage 3: 50M-Token Head-to-Head (12,207 Steps / 50.0M Tokens / ~33.5 hrs)
         ├── Top 1–2 Finalists vs. Locked 50M AdamW Baseline Checkpoint
         └── Full metric suite: Val PPL, expert health, loss spike frequency
                                │
                                ▼
Stage 4: Production Deployment (Winner advances to 1.0B Token Run)
```

---

### Stage 1: Cheap Smoke Tests (50–100 Steps)
- **Objective:** Inexpensive verification of code and hardware compatibility before committing GPU hours.
- **Duration:** Exactly 100 optimizer steps ($409,600$ tokens).
- **Runtime:** $\approx 15–18$ minutes per candidate on RTX 5070.
- **Tested Candidates:**
  1. AdamW Control
  2. Hybrid Muon + AdamW
  3. Sophia-G
  4. Newton-Muon
- **Mandatory Gate Criteria to Advance to Stage 2:**
  1. Zero NaN / Inf values in loss, gradients, or weights across all 100 steps.
  2. Peak allocated VRAM $\le 11,000$ MB (safe margin below 12,288 MB).
  3. Training loss decreases: $\text{Loss}_{\text{step 100}} < \text{Loss}_{\text{step 0}}$.
  4. Gradient norm remains bounded: $0.2 \le ||g||_2 \le 5.0$.
  5. AbsMean ternary master weights retain normal distribution (no >10% drift in $\alpha$).
  6. MoE router entropy > 1.2 (no expert starved of tokens).
  7. Checkpoint save and reload produces identical loss on step 101.

### Stage 2: 10M-Token Bakeoff
- **Objective:** Evaluate sample efficiency and GPU-hour efficiency on candidates passing Stage 1.
- **Duration:** Exactly 2,441 optimizer steps ($10,000,000$ tokens).
- **Runtime:** $\approx 6.7$ hours per candidate.
- **Evaluation Cadence:** Validation evaluated at step 1,220 and step 2,441.
- **Gate Criteria to Advance to Stage 3:**
  - Candidate must achieve lower validation CE than the AdamW control at 10M tokens OR achieve equal validation CE in at least 20% less wall-clock time.
  - At most **two** candidates advance to Stage 3.

### Stage 3: 50M-Token Head-to-Head
- **Objective:** Direct head-to-head comparison against the active 50M baseline checkpoint.
- **Duration:** Exactly 12,207 optimizer steps ($50,000,000$ tokens).
- **Runtime:** $\approx 33.5$ hours per finalist.
- **Deliverables:** Comparative loss curves, expert singular value spectra, perplexity progression, and final checkpoint evaluations.

### Stage 4: Production Deployment
- The winning optimizer is integrated into `train_1b_production.py` to drive the full 1,000,000,000-token foundation run.

---

## 5. Calibrated Practical Hyperparameter Search Space

Rather than an unmanageable grid, search spaces are restricted to small, scientifically grounded candidate sets:

| Optimizer | Hyperparameter | Recommended Default | Test Candidates | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **AdamW (Control)** | Learning Rate ($\eta$) | $1.5 \times 10^{-4}$ | $[1.5 \times 10^{-4}]$ | Validated baseline setting |
| | Betas $(\beta_1, \beta_2)$ | $(0.90, 0.95)$ | Fixed | Standard LLM pretraining settings |
| | Weight Decay ($\lambda$) | $0.10$ | Fixed | Validated baseline setting |
| **Hybrid Muon** | Muon LR ($\eta_{\text{Muon}}$) | **$2.0 \times 10^{-3}$** | $[1.0 \times 10^{-3}, 2.0 \times 10^{-3}, 4.0 \times 10^{-3}]$ | Calibrated for AbsMean ternary master weights ($\sigma \approx 0.025$). Note: Standard float Muon uses $0.02$, which is $10\times$ too aggressive. |
| | AdamW LR ($\eta_{\text{AdamW}}$) | $1.5 \times 10^{-4}$ | $[1.5 \times 10^{-4}]$ | For embeddings, head, routers, norms, and recurrence gates |
| | Muon Momentum ($\beta$) | $0.95$ | $[0.95]$ | Standard Muon momentum |
| | Weight Decay ($\lambda$) | $0.05$ | $[0.01, 0.05]$ | Decoupled decay for 2D weights |
| | NS Iterations ($K$) | $5$ | $[5]$ | 5 iterations guarantee spectral polar factor convergence |
| | Aspect Scaling | Enabled | True | $\alpha(M, N) = 0.2 \max(1, \sqrt{M/N})$ for non-square MoE |
| **Sophia-G** | Learning Rate ($\eta$) | $3.0 \times 10^{-4}$ | $[1.5 \times 10^{-4}, 3.0 \times 10^{-4}]$ | Sophia typically supports $1.5\times$ to $2\times$ AdamW LR |
| | Hessian Interval ($k$) | $10$ steps | $[10, 15]$ | Controls frequency of Gauss-Newton backprop spikes |
| | Clip Threshold ($\rho$) | $0.05$ | $[0.05]$ | Maximum coordinate step bound |
| | Damping ($\gamma$) | $1.0 \times 10^{-2}$ | $[1e-2]$ | Regularization for near-zero curvature |
| **Newton-Muon** | Muon LR ($\eta$) | $2.0 \times 10^{-3}$ | $[2.0 \times 10^{-3}]$ | Same base scale as Muon |
| | Covariance EMA ($\beta_{\text{cov}}$) | $0.99$ | $[0.99]$ | Smoothing factor for input activation second moment |
| | Covariance Damping ($\epsilon$) | $1.0 \times 10^{-4}$ | $[1e-4]$ | Regularization for covariance matrix inversion |

---

## 6. Execution Safety Protocol

During any future execution of these bakeoff stages:
1. **The Live 50M Baseline Run Must Not Be Touched:** The active process (PID 28148) continues to completion without interruption.
2. **Dedicated Output Directories:** All bakeoff runs must write to isolated subdirectories under `experiments/optimizer_research/runs/` or on drive `G:\Jarvis_Training\optimizer_bakeoff\`.
3. **No File Overwriting:** Checkpoints from the live 50M run in `experiments/checkpoints_1b/` must never be modified or overwritten.
4. **Automated Resource Monitoring:** An automated script must log peak VRAM and GPU temperatures every 100 steps, gracefully aborting if VRAM exceeds 11,200 MB or temperature exceeds $80^\circ\text{C}$.

---

## 7. Binding Directive

> [!CAUTION]
> **NO IMPLEMENTATION SHOULD BEGIN UNTIL THE 50M ADAMW CONTROL RUN HAS COMPLETED.**
