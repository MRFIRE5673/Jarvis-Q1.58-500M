# Jarvis Optimizer Integration Readiness Audit

**Date:** September 11, 2026  
**Scope:** Static Code Audit & Integration Architecture Review  
**Subject:** Codebase Readiness for Future Post-50M Optimizer Bakeoffs  
**Live Baseline Status:** PID 28148 is active, untouched, and continuing its 50M baseline trajectory.

---

## 1. Current Optimizer Architecture

In the active production training pipeline (`experiments/architecture_matrix/train_1b_production.py`, lines 217–224):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True,
)
```

### Architectural Properties of the Current Baseline
1. **Single Unified Parameter Group:** All 606,391,704 parameters are placed into a single parameter group. No distinction is made between 2D weight matrices, 1D normalization vectors, embedding tables, router projections, or recurrent decay scalars.
2. **Fused CUDA Kernel Execution:** The optimizer utilizes PyTorch's native `fused=True` AdamW implementation, executing multi-tensor fused CUDA kernels on the RTX 5070 to minimize memory bus round-trips.
3. **State Memory Allocation:** Maintains two FP32 state tensors ($m_t, v_t$) for every trainable parameter ($8\text{ bytes/param}$), committing **$4,851.13\text{ MB}~(4.74\text{ GB})$** to static optimizer state memory.
4. **Global Learning Rate Schedule:** Learning rate is updated per step via a cosine decay function (`get_lr_cosine`), applied uniformly to all parameter groups via:
   ```python
   for g in optimizer.param_groups:
       g["lr"] = lr
   ```

---

## 2. Exact Parameter Classification

Every trainable parameter in Jarvis was audited from `jarvis_engine/jarvis_model.py`. The complete mapping is recorded in [`parameter_group_inventory.csv`](file:///e:/Jarvis-Q1.58-500M/experiments/optimizer_research/parameter_group_inventory.csv).

```
Total Parameters: 606,391,704 (100.0%)
├── Category A: 2D Hidden Transformations (AbsMean Ternary) : 503,316,480 (83.00%) -> Muon Candidate
│   ├── Attention Projections (Q, K, V, Out)                : 100,663,296 (16.60%)
│   └── MoE Expert Projections (W1, W2 x 4 experts)         : 402,653,184 (66.40%)
├── Category B: Vocabulary Embeddings & LM Head (FP32/BF16) : 102,926,336 (16.97%) -> Must Remain AdamW
│   ├── Token Embedding (tok_emb.weight)                    :  51,463,168 ( 8.49%)
│   └── Language Model Head (lm_head.weight)                :  51,463,168 ( 8.49%)
├── Category C: MoE Router Projections (Dense Float)        :      98,304 ( 0.016%) -> Must Remain AdamW
│   └── 24 layers x router.weight (4 x 1024)                :      98,304 ( 0.016%)
├── Category D: RMSNorm Scales (1D Vectors)                 :      50,176 ( 0.008%) -> Must Remain AdamW
│   ├── Pre-Attention RMSNorm (norm1.weight x 24)           :      24,576 ( 0.004%)
│   ├── Pre-MoE RMSNorm (norm2.weight x 24)                 :      24,576 ( 0.004%)
│   └── Final RMSNorm (final_norm.weight)                   :       1,024 (<0.001%)
└── Category E: Neuromorphic Recurrent / Gating Scalars     :         408 (<0.001%) -> Must Remain AdamW
    ├── Associative Attention Decay (gamma_raw x 24)        :         384 (<0.001%)
    └── Liquid State Membrane Scale (var_scale x 24)        :          24 (<0.001%)
```

### Detailed Parameter Analysis
- **Category A (2D Hidden Projections — 503.3M params / 83.0%):** Suitable for Muon. All matrices are 2D with dimensions $1024 \times 1024$ (attention), $2048 \times 1024$ ($w_1$), and $1024 \times 2048$ ($w_2$). These dimensions are highly favorable for 5-step Newton-Schulz polynomial iterations.
- **Category B (Embeddings & Head — 102.9M params / 17.0%):** Must remain on AdamW. Embeddings receive highly row-sparse gradients. Muon's polynomial iteration $X (X^T X)^k$ would densify updates across all 50,257 rows, injecting artificial gradient noise into unobserved tokens. The LM head must remain on AdamW to preserve logit scale and temperature calibration.
- **Category C (MoE Routers — 98.3K params / 0.016%):** Must remain on AdamW. Each router is a $4 \times 1024$ matrix with rank $\le 4$. Applying Newton-Schulz orthogonalization to a rank-4 matrix forces 4 singular values to 1.0 and eliminates all other dimensions, destroying softmax routing temperature and entropy dynamics.
- **Categories D & E (Norms, Recurrent Decay, LSF Scalars — 50.6K params / 0.008%):** Must remain on AdamW. Matrix orthogonalization is mathematically undefined for 1D vectors and scalars. Furthermore, $\gamma_{\text{raw}}$ and $\text{var\_scale}$ govern exponential recurrent horizons, requiring smooth, coordinate-dampened AdamW updates.

---

## 3. Proposed Muon / AdamW Parameter Grouping Scheme

To support Hybrid Muon without rewriting the model architecture, parameters must be separated into two distinct groups:

```python
def create_hybrid_muon_param_groups(model, muon_lr=2.0e-3, adamw_lr=1.5e-4, muon_wd=0.05, adamw_wd=0.10):
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1D vectors, scalars, embeddings, heads, and routers stay on AdamW
        if (param.ndim < 2 or 
            "tok_emb" in name or 
            "lm_head" in name or 
            "router" in name or 
            "norm" in name or 
            "gamma_raw" in name or 
            "var_scale" in name):
            adamw_params.append(param)
        else:
            # All 2D hidden linear weights (attention Q,K,V,Out and MoE W1,W2) go to Muon
            muon_params.append(param)
            
    param_groups = [
        {
            "name": "muon_2d_hidden",
            "params": muon_params,
            "optimizer": "muon",
            "lr": muon_lr,
            "weight_decay": muon_wd,
            "momentum": 0.95,
        },
        {
            "name": "adamw_1d_and_special",
            "params": adamw_params,
            "optimizer": "adamw",
            "lr": adamw_lr,
            "weight_decay": adamw_wd,
            "betas": (0.9, 0.95),
        }
    ]
    return param_groups
```

### Verification of Group Counts
- `muon_params`: Exactly 503,316,480 parameters (83.00%).
- `adamw_params`: Exactly 103,075,224 parameters (17.00%).
- Total sum: Exactly 606,391,704 parameters (100.00%).

---

## 4. Ternary STE Interaction Analysis

### The Exact Gradient Path
The mathematical dataflow of a ternary parameter during training proceeds as follows:

```
[FP32 Master Weight: W_FP32]
             │
             ▼
[TernaryQuantizeSTE Forward]
├── alpha = mean(|W_FP32|)                 (AbsMean scale factor)
├── W_norm = W_FP32 / alpha                (Normalized master weight)
└── W_q = round(clamp(W_norm, -1, 1)) * alpha (Quantized forward weight)
             │
             ▼
[Forward Matmul: Y = X @ W_q]
             │
             ▼
[Loss Calculation & Backprop]
             │
             ▼
[TernaryQuantizeSTE Backward]
├── grad_output = dL / dW_q
├── mask = (|W_FP32| <= 1.0).float()       (STE clipping boundary)
└── grad_W = grad_output * mask
             │
             ▼
[Stored in W_FP32.grad]
             │
             ▼
[Optimizer.step()]  <======================== MUON ENTERS EXACTLY HERE
├── Updates the continuous FP32 master weight W_FP32
└── Leaves the forward quantization mechanism completely intact
```

### Critical Findings on Ternary-Muon Interaction
1. **Muon Operates on Master Weights:** Muon updates the continuous master weight $W_{\text{FP32}}$ during `optimizer.step()`. At the start of the next forward pass, `TernaryQuantizeSTE.apply()` re-quantizes the newly updated master weight to ternary values. The forward pass remains 100% ternary.
2. **Update Happens Before Normalization:** The optimizer updates $W_{\text{FP32}}$. The AbsMean scale factor $\alpha = \text{mean}(|W_{\text{FP32}}|)$ is dynamically computed *after* the optimizer update, at the beginning of the subsequent forward pass.
3. **The Six Research Hypotheses for Post-50M Testing:**
   - *H1 (Quantization Conflict):* Does spectral orthogonalization fight the discrete ternary projection?
   - *H2 (Weight Distribution Shift):* Does Muon alter the kurtosis or tails of master weights?
   - *H3 (Zero/$\pm 1$ Sparsity):* Does Muon change the ratio of active vs. zero weights?
   - *H4 (Saturation Risk):* Does Muon's larger step size push more master weights beyond the $|W_{\text{FP32}}| \le 1.0$ STE clipping boundary?
   - *H5 (Gradient Diversity):* Does Muon's global singular-vector update revive stalled weights?
   - *H6 (Scale Stability):* Does Muon destabilize the layer AbsMean scale $\alpha$?

---

## 5. MoE Interaction Analysis

### MoE Codebase Inspection (`jarvis_engine/jarvis_model.py`)
- **Expert Architecture:** 4 independent experts per layer; Top-2 routing via `probs.topk(2, dim=-1)`.
- **Parameter Independence:** Experts do **NOT** share weights. Each expert $e \in \{0, 1, 2, 3\}$ has its own `w1[e]` (`TernaryLinear(1024, 2048)`) and `w2[e]` (`TernaryLinear(2048, 1024)`).
- **Gradient Behavior on Unused Experts:** In each micro-batch, tokens are routed conditionally:
  ```python
  for e in range(self.num_experts):
      mask = (flat_idx == e)
      if mask.any():
          xe = flat_x[mask]
          ye = self.w2[e](F.gelu(self.w1[e](xe)))
          flat_out[mask] = flat_gate[mask].unsqueeze(-1) * ye
  ```
  If an expert receives zero tokens in a micro-batch (`mask.any()` is False), `w1[e]` and `w2[e]` are **not called** in the autograd tape. Their gradient for that micro-batch is zero/None.
- **Optimizer State on Unused Experts:** Once allocated, PyTorch optimizers retain state tensors for all parameters. Under AdamW, if an expert receives sparse gradients, its second-moment accumulator $v_t$ decays toward zero, causing an erratic step when the expert is suddenly reactivated. Under Muon, momentum orthogonalization normalizes the polar factor ($\sigma_i(O_t) = 1.0$), which in BF16 MoEs (Moonlight) prevented representation collapse.
- **Load-Balancing Loss Flow:** Load-balance loss $L_{\text{bal}} = \alpha \cdot N_{\text{experts}} \sum f_i P_i$ flows directly through router logits $P_i$. The router parameters must remain on AdamW to maintain smooth logit adjustments.

---

## 6. Checkpoint Compatibility Analysis

### Model State vs. Optimizer State
1. **Model Checkpoints are 100% Interchangeable:**
   `model.state_dict()` contains only raw parameter tensors (`tok_emb.weight`, `blocks.0.attn.q_proj.weight`, etc.). It contains zero optimizer-specific data. A model trained with AdamW can be loaded into an experiment using Hybrid Muon, and vice-versa, with 100% bit-exact parameter fidelity.
2. **Optimizer State is Incompatible:**
   - AdamW state dictionary structure: `{"state": {p_id: {"step": int, "exp_avg": Tensor, "exp_avg_sq": Tensor}}}`.
   - Hybrid Muon state dictionary structure: Contains two parameter groups, with Group 0 storing only `{"step": int, "momentum": Tensor}` (no `exp_avg_sq`).
3. **Protocol for Fair Bakeoffs:**
   Future comparative bakeoffs will initialize from baseline checkpoints via `--init-baseline`:
   ```python
   ckpt = torch.load(init_from_baseline, map_location="cpu")
   model.load_state_dict(clean_sd, strict=False)
   # Optimizer states are initialized fresh at step 0!
   ```
   This loads identical starting model weights while giving every candidate optimizer a fresh, uncorrupted momentum initialization.

---

## 7. Resume Semantics & Reproducibility Audit

The existing checkpoint resume implementation (`load_checkpoint` in `train_1b_production.py`) was audited:

```
[SAVED & RESTORED]
├── step (int)
├── tokens_trained (int)
├── model_state_dict (compact BF16)
├── optimizer_state_dict (compact BF16)
├── dataloader_state (shard index, offset)
├── best_val_ce (float)
└── loss_history (list)

[NOT SAVED / NOT RESTORED - REPRODUCIBILITY RISKS]
├── torch.get_rng_state()        (CPU RNG state)
├── torch.cuda.get_rng_state()   (CUDA RNG state)
└── python / np random state     (Host RNG state)
```

### Identified Reproducibility Risk
In `jarvis_model.py` (line 220), Gaussian noise is injected into router logits during training:
```python
if self.training:
    logits = logits + torch.randn_like(logits) * self.noise_std
```
Because CUDA RNG states are not currently preserved in checkpoints, resuming mid-run causes the router's Gaussian noise sequence to diverge from an uninterrupted run.
- **Requirement for Post-50M Bakeoff:** The checkpointing code for bakeoff experiments should serialize `torch.cuda.get_rng_state()` to guarantee bit-exact resume reproducibility.

---

## 8. Learning Rate & Scheduler Interaction

In `train_1b_production.py` (lines 270–272):
```python
lr = get_lr_cosine(step, warmup_steps, total_steps, max_lr, min_lr)
for g in optimizer.param_groups:
    g["lr"] = lr
```

### The Architectural Limitation
Currently, the scheduler unconditionally overwrites `g["lr"] = lr` across all parameter groups.
- In Hybrid Muon, Muon requires $\eta_{\text{Muon}} \approx 2.0 \times 10^{-3}$, while AdamW requires $\eta_{\text{AdamW}} \approx 1.5 \times 10^{-4}$.
- If this loop executes unmodified, Muon would receive the tiny AdamW learning rate, or AdamW would receive the aggressive Muon learning rate.

### Required Scheduler Adaptation (To Be Implemented Post-50M)
Parameter groups must define group-specific base and minimum learning rates, updated via decay progress:
```python
def update_scheduler_step(optimizer, step, warmup_steps, total_steps):
    if step < warmup_steps:
        prog = (step + 1) / warmup_steps
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * prog
    else:
        decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        for g in optimizer.param_groups:
            g["lr"] = g["min_lr"] + coeff * (g["base_lr"] - g["min_lr"])
```

---

## 9. Gradient Clipping Behavior

In `train_1b_production.py` (lines 280–290):
```python
# 1. Micro-batch loop: accumulate unclipped gradients
for micro_idx in range(accum_steps):
    ...
    loss_to_back.backward()

# 2. Global gradient clipping across all parameters
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Parameter update
optimizer.step()
```

### Audit Findings on Gradient Clipping
1. **Clipping Timing:** Occurs **after gradient accumulation** across all 4 micro-batches and immediately before `optimizer.step()`. This is standard and correct.
2. **Global Norm Calculation:** Computes $\|g_{\text{global}}\|_2 = \sqrt{\sum \|g_p\|_2^2}$ across all 606M parameters simultaneously.
3. **Compatibility with Muon:**
   - For AdamW parameters (embeddings, routers, norms), global clipping directly scales down gradient spikes.
   - For Muon parameters, Newton-Schulz polynomial iteration maps singular values to 1.0, normalizing out the global scaling factor. However, global clipping still preserves relative gradient magnitudes across accumulated micro-batches and stabilizes the momentum accumulator $M_t$.
   - **Conclusion:** Global gradient clipping works cleanly with Hybrid Muon and requires zero modifications.

---

## 10. Benchmark Instrumentation Requirements

The current training loop records only total update duration (`t_step_dur = time.perf_counter() - t0_step`). It cannot isolate the computational cost of the optimizer.

### Required Instrumentation for Post-50M Bakeoffs
To measure **Quality per GPU-Hour** accurately, the training loop must record:
1. `t_data`: Time spent fetching batches from `train_loader.next_batch()`.
2. `t_fwd_bwd`: Time spent inside forward autocast and `loss.backward()`.
3. `t_clip`: Time spent inside `clip_grad_norm_`.
4. `t_optim`: Time spent exclusively inside `optimizer.step()`.
5. `t_eval`: Time spent evaluating validation shards.
6. `peak_vram`: Monitored via `torch.cuda.max_memory_allocated()`.

This instrumentation will be added to the bakeoff test harness without altering production training code.

---

## 11. Recommended Future Optimizer API

To support multiple optimizers cleanly without polluting `train_1b_production.py`, a modular optimizer factory should be introduced post-50M:

```
jarvis_engine/
└── optimizers/
    ├── __init__.py
    ├── optimizer_factory.py     # Clean factory returning optimizer & param groups
    └── hybrid_muon.py           # Self-contained PyTorch Muon implementation
```

### Minimal Training Command-Line Interface
```bash
# Baseline Control:
python train_bakeoff.py --optimizer adamw --max-lr 1.5e-4

# Hybrid Muon Candidate:
python train_bakeoff.py --optimizer hybrid_muon --muon-lr 2.0e-3 --adamw-lr 1.5e-4 --muon-wd 0.05

# Sophia-G Candidate:
python train_bakeoff.py --optimizer sophia_g --max-lr 3.0e-4 --hessian-interval 10
```

---

## 12. Risks and Blockers

| Area | Potential Risk | Blocker Severity | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Model Code** | Are parameter shapes or modules incompatible with Muon? | **NONE** | All 503.3M target parameters are clean 2D matrices. |
| **Quantization** | Does STE gradient clipping break Muon? | **NONE (Hypothesis to test)** | STE gradients flow directly into `weight.grad`; Muon updates master weights. |
| **MoE Routing** | Do sparse expert updates cause dimension errors? | **NONE** | Inactive experts have zero grads; Newton-Schulz operates per-matrix. |
| **Hardware** | Does Hybrid Muon exceed 12GB VRAM? | **NONE** | Hybrid Muon reduces state memory by ~1.97 GB vs. AdamW. |
| **Scheduler** | Does `train_1b_production.py` overwrite group LRs? | **MINOR CODE ADAPTATION** | Update scheduler loop to respect per-group `base_lr` post-50M. |
| **Instrumentation**| Is optimizer execution time measured? | **MINOR CODE ADAPTATION** | Add timing wrappers around `optimizer.step()` in bakeoff harness. |

**Zero structural or mathematical blockers exist in the Jarvis codebase.**

---

## 13. Minimal Implementation Plan (Post-50M Baseline)

```
[Phase 1: Baseline Completion]
└── Wait for live run PID 28148 to reach Step 12,207 (50M tokens). Lock ckpt_step_012207.pt.

[Phase 2: Bakeoff Infrastructure Preparation]
├── 1. Implement self-contained `optimizers/hybrid_muon.py` (~40 lines of PyTorch).
├── 2. Implement `optimizers/optimizer_factory.py` with parameter group separation.
└── 3. Create isolated bakeoff test harness `experiments/optimizer_research/train_bakeoff.py`.

[Phase 3: Stage 1 Smoke Tests (100 Steps / ~15 min each)]
├── Run AdamW Control vs. Hybrid Muon vs. Sophia-G from ckpt_step_0004284_best.pt.
└── Verify VRAM <= 11,000 MB, zero NaNs, and loss descent.

[Phase 4: Stage 2 Bakeoff (10M Tokens / ~6.7 hrs each)]
├── Run surviving candidates on 10M tokens.
└── Evaluate Validation Cross-Entropy vs. GPU Wall-Clock Hours.

[Phase 5: Stage 3 Finalist Bakeoff (50M Tokens / ~33.5 hrs)]
└── Compare top candidate head-to-head against the locked 50M AdamW baseline.
```

---

## Conclusion & Readiness Verdict

The static code audit confirms that Jarvis's model topology, ternary quantization engine, and sparse MoE layers are cleanly architected and fully compatible with future hybrid optimizer evaluation. The parameter inventory is precisely mapped, the gradient dataflow is mathematically verified, and all necessary adaptations have been designed.

**READY FOR IMPLEMENTATION AFTER 50M BASELINE COMPLETION**
