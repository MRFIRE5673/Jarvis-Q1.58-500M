# Jarvis Advanced Training Optimizer Research

## 1. Executive Overview

This directory contains the audited research report, literature verification, and experimental bakeoff protocol for modern training optimizers evaluated for the **Jarvis-Q1.58-500M** foundation model.

### Absolute Baseline Constraint
The current **50M-token baseline training run (PID 28148)** on the NVIDIA GeForce RTX 5070 12GB is an active, inviolable control experiment. **It has remained completely untouched throughout this research.** No processes were stopped, paused, or modified; no configurations or checkpoints were altered; and no GPU memory was consumed that could interfere with its execution.

This research was conducted purely via static architectural analysis, mathematical modeling, and rigorous literature verification across top AI research organizations (Moonshot AI, Stanford, Google Brain/DeepMind, Meta FAIR, Microsoft Research, and Princeton).

---

## 2. Directory Structure

```
experiments/optimizer_research/
├── audit_report.md                    # Critical audit, evidence downgrades & literature check
├── README.md                          # Executive summary & candidate ranking (this file)
├── optimizer_landscape.md             # Modern optimizer landscape & theoretical breakdown
├── jarvis_optimizer_analysis.md       # Component-by-component analysis & 6 ternary hypotheses
├── future_bakeoff_plan.md             # 4-stage controlled experimental bakeoff protocol
├── optimizer_memory_estimates.json    # Audited VRAM budgets (measured vs. modeled estimates)
└── optimizer_candidate_matrix.csv     # 10-dimensional evaluation matrix across all candidates
```

---

## 3. Audited Optimizer Candidate Evaluation Matrix

*Baseline AdamW figures are EMPIRICALLY MEASURED on the active training run (PID 28148). All candidate figures are modeled ESTIMATES.*

| Optimizer | Evidence Level | Expected Benefit | Compute Overhead | VRAM Overhead vs. Baseline | LLM Suitability | MoE Suitability | Ternary Suitability | Jarvis Suitability | Ranking Category | Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AdamW (Fused)** | **[A - Established]** | Validated baseline reference; universal convergence guarantee | 0.0% (Control baseline) | 0 MB (4,851 MB states; 9,420 MB peak) [Measured] | High (Universal standard) | Moderate (Lagging $v_t$ on unselected experts) | High (Decoupled decay bounds master weights) | High (Current baseline; proven stability) | **BASELINE_CONTROL** | **GREEN** |
| **Hybrid Muon + AdamW** | **[A/B/C]** (Jordan 2024; Moonlight 2025; NanoGPT) | Reported 1.5×–2.0× token efficiency on float LLM/MoE; unverified on ternary | +0.8% to +2.5% (5 Newton-Schulz matmuls on 2D weights) | **~ -1,600 to -1,970 MB** peak (2,838 MB states; **~7,450–7,800 MB peak**) [ESTIMATE] | High (Validated up to 16B MoE / 5.7T tokens) | High (Equalizes expert singular values in float MoE) | Unverified Hypothesis (Spectral bounds vs. STE clipping) | **Highest Potential** (83% 2D params; significant VRAM relief) | **TOP EXPERIMENTAL CANDIDATE (Priority 1)** | **GREEN** |
| **Sophia-G** | **[B/C]** (Liu et al. Stanford 2023; mixed 3rd party) | Reported up to 1.5×–2.0× speedup via diagonal Hessian curvature preconditioning | +5.2% (Gauss-Newton backprop every $k=10$) | **+1,430 MB transient spike** (Surges to **~10,850 MB peak**) [ESTIMATE] | High on standard dense transformers | Moderate-Low (Hessian estimates noisy under sparse routing) | Questionable (Hessian of piecewise-linear STE is ill-defined) | Moderate (VRAM peak leaves narrow ~350 MB margin on 12GB GPU) | **WORTH TESTING (Priority 2)** | **YELLOW** |
| **Newton-Muon** | **[B]** (Du & Su 2026; Modded-NanoGPT) | Reported +4% wall-clock / 6% step reduction over pure Muon on 124M NanoGPT | +4.5% (Layer activation covariance inversion) | **-425 MB** (3,450 MB states; ~8,995 MB peak) [ESTIMATE] | Moderate-High (Small/medium pretraining) | Low-Moderate (Covariance tracking on dynamic tokens) | Unverified Hypothesis (Covariance of ternary master weights untested) | Moderate-Low (High hook complexity with grad checkpointing) | **EXPLORATORY (Priority 3)** | **YELLOW** |
| **MONA (Muon + Nesterov)**| **[B]** (arXiv:2605.26842 2026) | Reported faster escape from sharp local minima on MoE surfaces | +1.4% (Gradient difference EMA tracking) | **+60 MB** (4,851 MB states; ~9,480 MB peak) [ESTIMATE] | High (Designed for MoE pretraining) | High (Nesterov momentum mitigates routing latency) | Unverified Hypothesis (Similar to Muon; needs step tuning) | Moderate (Eliminates Muon's VRAM savings on 12GB) | **EXPLORATORY (Priority 4)** | **YELLOW** |
| **Lion** | **[A/B/C]** (Google Brain 2023) | Low memory (4 B/param); simple sign updates | -0.2% (Pure sign operation; no division) | **~ -2,400 MB** (2,425 MB states; ~7,020 MB peak) [ESTIMATE] | Moderate-High on vision / dense LLMs | Low-Moderate (Sign-flip oscillation on sparse experts) | **Low** (Severe risk: sign updates chattering on discrete ternary weights) | Low (High risk of optimization stalls and dead weights) | **NOT CURRENTLY WORTH TESTING** | **YELLOW** |
| **SOAP** | **[B/C]** (Meta FAIR / Princeton 2024) | Reported 1.3×–1.5× step reduction in eigenbasis | +8.0% (SVD / Eigendecomposition of Kronecker factors) | **+4,180 MB** (**>13,600 MB peak — EXCEEDS VRAM**) [ESTIMATE] | High on multi-GPU clusters | Moderate | Uncertain | **Unviable** (Immediate CUDA OOM on RTX 5070 12GB) | **REJECTED** | **RED** |
| **Schedule-Free AdamW** | **[B/C]** (Meta FAIR 2024) | Eliminates cosine schedule decay tuning; anytime stopping | +0.5% (Polyak iterate interpolation) | **+3,060 MB** (**>12,480 MB peak — EXCEEDS VRAM**) [ESTIMATE] | High for fixed compute exploration | Moderate | High (Same dynamics as AdamW) | **Unviable** (Iterate anchor buffer exceeds 12GB budget) | **REJECTED** | **RED** |

---

## 4. Top Candidates for Post-50M Evaluation

### Rank 1: Hybrid Muon + AdamW (Highest-Priority Experimental Candidate)
- **Status:** **TOP EXPERIMENTAL CANDIDATE (NOT YET PROVEN WINNER)**
- **Why It Is Prioritized:**
  1. **Substantial Modeled VRAM Relief:** Exactly 503,316,480 parameters (83.00% of Jarvis) reside in 2D linear weight matrices. Transitioning these matrices from AdamW (8 bytes/param) to Muon (4 bytes/param) reduces optimizer state memory from 4.85 GB to 2.84 GB, **estimated to free ~1.6 to 1.97 GB of physical VRAM** on the RTX 5070.
  2. **MoE Representation Scaling:** Proven on Moonlight 16B MoE (arXiv:2502.16982) in BF16, where 5th-order Newton-Schulz iteration equalized singular values across expert matrices ($\sigma_i(O_t) = 1.0$).
  3. **Critical Research Hypothesis:** Whether matrix orthogonalization works constructively with AbsMean ternary master weights or distorts the ternary representation is an unverified empirical question that must be tested during Stage 1 and Stage 2 bakeoff runs.

### Rank 2: Sophia-G (Secondary Candidate)
- **Status:** **WORTH TESTING**
- **Why It Is Ranked Second:**
  1. **Curvature Preconditioning:** Uses a stochastic diagonal Gauss-Newton Hessian estimator to clip updates based on local loss curvature, enabling paper-reported $1.5\times$ to $2\times$ faster convergence on dense transformer blocks.
  2. **The Hardware Limitation:** Evaluating the Gauss-Newton backward pass every $k=10$ steps creates a transient VRAM spike of $+1.4$ GB, pushing peak VRAM to **~10,850 MB (88.3% of capacity)** on the RTX 5070, leaving narrow ~350 MB margin.
  3. **Ternary Friction:** The piecewise-constant nature of AbsMean ternary quantization produces ill-defined Hessian curvature, making Gauss-Newton estimates noisy.

### Rank 3: Newton-Muon (Exploratory Candidate)
- **Status:** **EXPLORATORY**
- **Why It Is Ranked Third:**
  1. **Rigorous Theory:** Bridges the theoretical gap in Muon by right-preconditioning the gradient with the empirical input activation covariance $(Z Z^T)^{-1}$.
  2. **Why It Is Exploratory:** On Modded-NanoGPT, it yielded a 4% wall-clock improvement over pure Muon, but requires calculating and inverting covariance matrices per layer ($1024 \times 1024$). Under gradient checkpointing (`use_reentrant=False`), caching or recomputing input activations $Z$ adds substantial engineering complexity for a modest gain.

---

## 5. Post-50M Baseline Execution Roadmap

When the active 50M baseline training run completes its 12,207-step trajectory, the four-stage bakeoff will proceed as follows:

```
[Active 50M AdamW Baseline (PID 28148) Completes]
                         │
                         ▼
Stage 1: Cheap Smoke Tests (50–100 Steps / ~400K Tokens / ~15–18 min)
         ├── AdamW Control vs. Hybrid Muon vs. Sophia-G vs. Newton-Muon
         └── Test forward/backward/optimizer, peak VRAM, loss descent, no NaNs
                         │
                         ▼
Stage 2: 10M-Token Bakeoff (2,441 Steps / 10.0M Tokens / ~6.7 hrs)
         ├── Primary metric: Validation CE vs. GPU Wall-Clock Hours
         └── Candidates must beat AdamW or achieve 20%+ wall-clock speedup
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

## 6. Binding Directive

> [!CAUTION]
> **NO IMPLEMENTATION SHOULD BEGIN UNTIL THE 50M ADAMW CONTROL RUN HAS COMPLETED.**
