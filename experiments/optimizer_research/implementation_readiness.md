# Jarvis Optimizer Infrastructure Implementation Readiness Report

**Date:** September 11, 2026  
**Status:** Infrastructure Prepared; Zero Training Experiments Executed  
**Live Baseline Run:** PID 28148 is active, untouched, and continuing its 50M training trajectory on the RTX 5070.

---

## 1. Files Created and Modified

### Created Modules
1. `jarvis_engine/optimizers/__init__.py`: Package initialization exposing `Muon`, `HybridMuonAdamW`, `zeropower_via_newtonschulz5`, `classify_parameter_groups`, and `build_optimizer`.
2. `jarvis_engine/optimizers/muon.py`: Pure PyTorch implementation of the 5th-order Newton-Schulz polar factor matrix orthogonalization with Moonshot aspect-ratio scaling.
3. `jarvis_engine/optimizers/parameter_groups.py`: Deterministic parameter classifier partitioning parameters into 2D hidden matrices (Muon) vs. vectors, scalars, embeddings, and routers (AdamW) with strict validation checks.
4. `jarvis_engine/optimizers/hybrid.py`: `HybridMuonAdamW` optimizer subclassing `torch.optim.Optimizer`, coordinating independent learning rates, weight decays, and updates across both groups.
5. `experiments/optimizer_research/config_hybrid_muon.yaml`: Research configuration example with clearly documented initial values.
6. `experiments/optimizer_research/reproducibility_plan.md`: Documented TODO and isolated future patch plan for RNG states (CPU, CUDA, Python, Dataset) and MoE router noise determinism.
7. `tests/test_optimizer_infrastructure.py`: Comprehensive CPU-only unit test suite covering shape handling, polar factor convergence, parameter grouping, ternary STE master weight updates, checkpoint serialization, and scheduler scaling.
8. `experiments/optimizer_research/parameter_group_inventory.csv`: Complete mapping of all 606,391,704 parameters across Jarvis.
9. `experiments/optimizer_research/optimizer_experiment_checklist.md`: 20-point scientific fairness checklist.
10. `experiments/optimizer_research/integration_readiness.md`: 13-section static code audit report.
11. `experiments/optimizer_research/future_bakeoff_plan.md`: 4-stage experimental protocol for post-50M baseline evaluation.

### Modified Files
- `experiments/architecture_matrix/train_1b_production.py`:
  - Added CLI options: `--optimizer {adamw, hybrid_muon}` (default: `adamw`), `--muon-lr` (default: `2.0e-3`), `--muon-wd` (default: `0.05`).
  - Added optimizer initialization branching via `jarvis_engine.optimizers`.
  - Added proportional learning rate scheduler scaling when using hybrid optimization.
  - **Regression Safety Guarantee:** When running with `--optimizer adamw` (the default!), all code paths remain 100% behaviorally identical to the existing baseline.

---

## 2. Exact Parameter Grouping

All 606,391,704 parameters are partitioned deterministically based on actual tensor properties and architectural rules:

1. **Group 0: `muon_2d_hidden` (503,316,480 parameters / 83.00%)**
   - Attention projections: $W_q, W_k, W_v, W_{\text{out}}$ ($1024 \times 1024 \times 24$ layers $= 100,663,296$ params).
   - MoE expert projections: $W_1$ ($2048 \times 1024 \times 4 \times 24 = 201,326,592$ params) and $W_2$ ($1024 \times 2048 \times 4 \times 24 = 201,326,592$ params).
   - Optimizer: Muon with 5th-order Newton-Schulz orthogonalization and aspect scaling.
2. **Group 1: `adamw_1d_and_special` (103,075,224 parameters / 17.00%)**
   - Vocabulary embeddings: `tok_emb.weight` ($50257 \times 1024 = 51,463,168$ params).
     - *Exclusion Rationale:* Sparse row gradients; Newton-Schulz polynomial iteration would densify updates across all 50,257 rows, corrupting unobserved tokens.
   - Language model head: `lm_head.weight` ($50257 \times 1024 = 51,463,168$ params).
     - *Exclusion Rationale:* Direct logit calibration; standard LLM practice preserves logit scales and temperature dynamics under AdamW.
   - MoE router projections: `router.weight` ($4 \times 1024 \times 24 = 98,304$ params).
     - *Exclusion Rationale:* Rank $\le 4$; Newton-Schulz forces singular values to 1.0, destroying routing entropy and causing expert starvation.
   - RMSNorm scale vectors: `norm1`, `norm2`, `final_norm` ($50,176$ params).
     - *Exclusion Rationale:* 1D vectors; matrix orthogonalization mathematically undefined.
   - Recurrent / LSF scalars: `gamma_raw` ($384$ params), `var_scale` ($24$ params).
     - *Exclusion Rationale:* 1D vectors and 0D scalars; govern temporal decay dynamics.

---

## 3. Unit Tests Performed & Results

All tests were executed strictly on **CPU** with **zero GPU allocation**:

```
tests/test_optimizer_infrastructure.py
├── test_newton_schulz_shapes ........................................... PASSED
│   └── Verified polar factor singular values contract to [0.60, 1.25] (condition number < 2.0)
├── test_muon_step_cpu .................................................. PASSED
│   └── Verified parameter updates, non-zero step, momentum buffer allocation
├── test_parameter_classification_synthetic ............................. PASSED
│   └── Verified exact classification across 2D weights, embeddings, norms, and scalars
├── test_ternary_ste_compatibility_cpu .................................. PASSED
│   └── Verified continuous master weight updates, dynamic AbsMean re-quantization
├── test_hybrid_muon_adamw_step_cpu ..................................... PASSED
│   └── Verified independent group updates, state tracking, absence of NaNs/Infs
├── test_scheduler_proportional_scaling_cpu ............................. PASSED
│   └── Verified proportional decay scaling preserves distinct base LRs
├── test_regression_adamw_default ....................................... PASSED
│   └── Verified build_optimizer with default options returns exact standard AdamW
└── test_checkpoint_serialization_and_model_only_init ................... PASSED
    └── Verified state_dict serialization and model-only initialization
```
**Test Result: 8 tests ran in 1.509s — OK (100% Passing).**

### Tests NOT Performed (By Strict Directive)
- No GPU training steps (smoke test, 100 steps, 1,000 steps).
- No GPU throughput or VRAM benchmarking.
- No dataset reading on GPU.
- **Reason:** PID 28148 is actively training and must not experience GPU memory competition.

---

## 4. Architectural Compatibility Analysis

### Ternary STE Compatibility
Muon operates on continuous master weights ($W_{\text{FP32}}$) during `optimizer.step()`. The forward pass re-quantizes master weights dynamically to $\{- \alpha, 0, +\alpha\}$ using dynamic AbsMean scaling $\alpha = \text{mean}(|W_{\text{FP32}}|)$. The forward pass remains 100% ternary. Neither `ternary_ops.py` nor `TernaryQuantizeSTE` were modified.

### MoE Compatibility
Each of the 4 experts has independent $W_1$ and $W_2$ weights. Inactive experts receive zero gradients; Muon preserves their momentum polar factor without the variance denominator collapse that affects AdamW. Routers ($4 \times 1024$) remain on AdamW to preserve routing entropy.

### Checkpoint Compatibility
`model.state_dict()` contains only parameter tensors and is 100% compatible between AdamW and Muon. Comparative bakeoff runs will use `--init-baseline` to initialize model weights while starting optimizer states fresh from step 0. AdamW momentum/second-moment states are never loaded into Muon.

### Scheduler Behavior
The training loop scales each parameter group's base learning rate by the schedule multiplier:
$$g[\text{'lr'}] = g[\text{'base\_lr'}] \cdot \left(\frac{\text{lr}}{\text{max\_lr}}\right)$$
This allows $\eta_{\text{Muon}} = 2.0 \times 10^{-3}$ and $\eta_{\text{AdamW}} = 1.5 \times 10^{-4}$ to decay synchronously throughout warmup and cosine decay.

### Gradient Clipping Semantics
Global gradient norm clipping (`max_norm = 1.0`) is preserved across all parameters after gradient accumulation and immediately prior to `optimizer.step()`. Both Muon and AdamW receive identically clipped gradients for fairness.

---

## 5. Known Risks & Mitigations

1. **Newton-Schulz Step Overhead:**
   - *Risk:* 5 iterations of matrix multiplications on large 2D weights ($2048 \times 1024$) may add step latency.
   - *Mitigation:* In Hybrid Muon, Newton-Schulz is executed once per optimizer step (not per micro-batch). With gradient accumulation = 4, this occurs only once every 4 forward/backward passes.
2. **Master Weight Saturation Under Polar Orthogonalization:**
   - *Risk:* Newton-Schulz forces orthogonal updates with norm $\sim \sqrt{N}$. If the master weights grow too large, STE clipping ($|W| > 1.0$) could saturate.
   - *Mitigation:* Decoupled weight decay ($\lambda = 0.05$) and calibrated base LR ($2.0 \times 10^{-3}$ rather than standard $0.02$) prevent unconstrained weight growth.
3. **MoE Router Noise Drift:**
   - *Risk:* `torch.randn_like` in router causes stochastic divergence across resumes.
   - *Mitigation:* Documented future patch in `reproducibility_plan.md` isolates router RNG via dedicated seeded generator.

---

## 6. Remaining Work (Post-50M Baseline Execution)

1. **Wait for PID 28148 Completion:** Allow the baseline training run to complete its 50M token target (~Step 12,207).
2. **Stage 1 Smoke Test (100 Steps, ~15 min):** Execute the 100-step smoke test on GPU with `--optimizer hybrid_muon` from `ckpt_step_0004284_best.pt`. Verify peak VRAM $\le 11,000\text{ MB}$ and loss descent.
3. **Stage 2 10M Bakeoff (2,441 Steps, ~6.7 hrs):** Run AdamW Control vs. Hybrid Muon on 10M tokens, measuring **Validation Cross-Entropy vs. GPU Wall-Clock Hours**.
4. **Stage 3 50M Head-to-Head (12,207 Steps, ~33.5 hrs):** Compare the winning candidate directly against the locked 50M AdamW baseline checkpoint.

---

## Final Status

**Optimizer infrastructure prepared; no training experiment executed.**
