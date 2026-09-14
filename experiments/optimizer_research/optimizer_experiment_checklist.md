# Jarvis Future Optimizer Experiment Checklist: Scientific Fairness & Isolation Protocol

This checklist governs all future optimizer bakeoff experiments conducted after the active 50M-token AdamW baseline run (**PID 28148**) completes.

Every candidate optimizer experiment (Smoke Test, 10M Bakeoff, 50M Finalist Bakeoff) must verify and check off every item in this document before results can be accepted as scientifically valid.

---

## 1. Fixed Architectural & Hardware Controls (Must Be 100% Identical)

- [ ] **1. Identical Model Architecture:** Exactly 606,391,704 total parameters (503.3M 2D hidden ternary weights, 102.9M embedding/head weights, 98.3K router weights, 50.4K 1D/scalar weights). No layer count, head count, or hidden dimension changes.
- [ ] **2. Identical Starting Checkpoint:** All comparative runs must initialize from the **exact same master model weights**:
  `experiments/extended_train/ckpt_step_0004284_best.pt` (or the completed 50M baseline checkpoint `ckpt_step_012207.pt` if testing continual adaptation).
  - *Note:* Only model weights are loaded; optimizer states initialize fresh at step 0 for fair momentum warmup.
- [ ] **3. Identical Tokenizer:** Byte-Pair Encoding (GPT-2 tokenizer, vocabulary size: 50,257).
- [ ] **4. Identical Sequence Length:** $T = 512$ tokens per sequence.
- [ ] **5. Identical Batch Geometry & Token Cadence:**
  - Micro-batch size: $B = 2$ sequences per forward pass.
  - Gradient accumulation steps: $4$.
  - Effective update batch: $8$ sequences = **$4,096$ tokens per optimizer update**.
- [ ] **6. Identical Precision & Autocast:** BFloat16 mixed precision via `torch.amp.autocast("cuda", dtype=torch.bfloat16)` with unscaled backward pass (no GradScaler needed for BF16).
- [ ] **7. Identical Gradient Norm Clipping:** Global clipping across all parameters with `max_norm = 1.0` applied after gradient accumulation and immediately prior to `optimizer.step()`.
- [ ] **8. Identical Auxiliary Loss Weights:**
  - MoE load balancing weight: $\alpha = 0.01$.
  - Reflective penalty weight: $\lambda = 0.001$.
- [ ] **9. Identical Hardware Environment:** Single NVIDIA GeForce RTX 5070 12GB running on the same host environment, operating system, and PyTorch version.

---

## 2. Fixed Dataset & Evaluation Controls (Must Be 100% Identical)

- [ ] **10. Identical Training Dataset Streaming:** Sharded binary dataset streaming sequentially from `data/shards/train_shard_00000.bin` onwards via `ShardedTokenDataset(seed=42)`.
- [ ] **11. Identical Validation Set:** Held-out shard `data/shards/val_shard_00000.bin` (unseen during training).
- [ ] **12. Identical Validation Evaluation Cadence & Sample Size:**
  - Fixed sample size: Exactly 32 batches ($32,768$ tokens) evaluated deterministically without shuffling (`seed=1337`).
  - Validation execution: `model.eval()`, `torch.inference_mode()`, membrane states reset via `model.reset_state()`.
- [ ] **13. Identical Token Training Budget:**
  - Stage 1 Smoke Test: Exactly $100$ steps ($409,600$ tokens).
  - Stage 2 Bakeoff: Exactly $2,441$ steps ($10,000,000$ tokens).
  - Stage 3 Finalist Bakeoff: Exactly $12,207$ steps ($50,000,000$ tokens).

---

## 3. Storage & Process Isolation Protocol

- [ ] **14. Dedicated Storage Partition:** All bakeoff runs must write logs, temporary files, and checkpoints to isolated directories on drive `G:\Jarvis_Training\optimizer_bakeoff\<candidate_name>\`.
- [ ] **15. Zero Overwrite Guarantee:** Checkpoint files from completed production runs (`experiments/checkpoints_1b/`) must never be modified or overwritten.
- [ ] **16. Baseline Integrity:** No bakeoff experiment may be initiated until PID 28148 has completed its full run.

---

## 4. Optimizer Hyperparameter Documentation Table

For every experimental run, the operator must record the exact optimizer configuration in the run's metadata header:

| Hyperparameter | Recorded Setting for Run | Justification / Source |
| :--- | :--- | :--- |
| **Optimizer Name** | `[e.g., Hybrid_Muon_AdamW]` | Name of algorithm |
| **Primary Learning Rate** | `[e.g., 2.0e-3 for Muon]` | Scaled for master ternary weights ($\sigma \approx 0.025$) |
| **Secondary (AdamW) LR** | `[e.g., 1.5e-4]` | For 1D norms, embeddings, routers, gates |
| **Momentum / Betas** | `[e.g., beta=0.95 for Muon, (0.9, 0.95) for AdamW]` | Momentum smoothing |
| **Weight Decay** | `[e.g., 0.05 for Muon, 0.10 for AdamW]` | Decoupled decay |
| **Warmup Steps** | `[e.g., 200 steps for 10M, 2000 steps for 50M]` | Linear warmup schedule |
| **LR Decay Schedule** | `[e.g., Cosine decay to 10% min_lr]` | Matched decay schedule |
| **Special Hyperparameters**| `[e.g., NS iterations=5, aspect_scale=True]` | Algorithm-specific parameters |

---

## 5. Decision Metric Protocol

- [ ] **17. Primary Metric:** Validation Cross-Entropy vs. GPU Wall-Clock Hours (Total elapsed seconds from step 0).
- [ ] **18. Secondary Metric:** Validation Cross-Entropy vs. Training Tokens.
- [ ] **19. Resource Metrics Logged:** Peak VRAM (`torch.cuda.max_memory_allocated()`), step time (`step_time_ms`), optimizer execution time (`optim_time_ms`), and throughput (`tokens_per_sec`).
- [ ] **20. Representation Metrics Logged:** Global gradient norm (`grad_norm`), AbsMean scale factor ($\alpha$), master weight saturation fraction ($|W| > 1.0$), and MoE router entropy.
