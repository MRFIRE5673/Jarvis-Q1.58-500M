# Jarvis Research Program & Documentation Hub

Welcome to the Jarvis-Q1.58-500M Research Program. This directory contains the scientific documentation, empirical interaction matrices, and methodology logs for our recurrent memory architecture investigations.

---

## Document Directory

1. **[Interaction Matrix (`interaction_matrix.md`)](interaction_matrix.md):**
   - Empirical interaction coefficients for the factorial memory combination matrix.
   - Comprehensive cross-entropy and perplexity tables for standalone mechanisms and the completed pairwise subset (`A+B`, `A+C`, `A+D`, `A+E`, `B+C`, `B+D`, `B+E`).
   - Synergy classifications and analytical observations.

2. **[Scientific Methodology (`methodology.md`)](methodology.md):**
   - Strict experimental controls, starting checkpoint immutability, and token budgets.
   - Mathematical definitions of interaction coefficients and neutral / identity initialization.
   - Canonical 4-stage evaluation suite protocol.

3. **[Architectural Decisions (`architecture_decisions.md`)](architecture_decisions.md):**
   - Exact distinction between the **Paper-Faithful Baseline** and **Research Variants**.
   - Model capacity verification (606M total parameters, 405M active/token).
   - Architectural Decision Records (ADRs) covering attention, quantization, MoE routing, and CUDA acceleration.

---

## Research Overview & Current Milestones

The Jarvis research journey follows a staged, empirical approach:

```
[Phase 1: Paper Baseline Audit] 
       │ (606M params, 405M active/token, 4,209 step verified baseline)
       ▼
[Phase 2: CUDA Engine Acceleration]
       │ (Audited CUDA Associative Attention, CUDA MoE, vectorized LSF)
       ▼
[Phase 3: Foundational Architecture Exploration]
       │ (16-experiment matrix: activations, FFN variants, aux-free bias, optimizers)
       ▼
[Phase 4: Factorial Memory Research (Current)]
       │ (Systematic investigation of mechanisms A, B, C, D, E)
       ├─ Singles: fact_A, fact_B, fact_C, fact_D, fact_E [Complete]
       ├─ Pairwise: 10 dual combinations [In Progress: 7/10 complete]
       └─ Higher-Order: Multi-way candidate synthesis [Queued]
```

### Current Status Highlights:
- **Baseline:** Holdout CE **3.2858** | PPL **26.73**.
- **Top Pairwise Candidate:** `fact_B+C` (Write Gate + Erase Gate) achieved **3.2882 CE** (**26.79 PPL**), dipping to **3.2786 CE** mid-run.
- **Synergy Signal:** All seven completed pairs exhibit negative interaction coefficients ($\approx -0.010$ to $-0.012$), demonstrating that selective memory gating mechanisms cooperate effectively rather than causing functional interference.
