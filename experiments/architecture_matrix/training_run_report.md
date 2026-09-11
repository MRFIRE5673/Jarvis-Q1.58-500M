# JARVIS 1.0B TOKEN TRAINING RUN & PREPARATION REPORT
======================================================

**Date:** September 11, 2026  
**Hardware Platform:** Single NVIDIA GeForce RTX 5070 (12,226.6 MB VRAM, GDDR7)  
**Baseline Model:** `experiments/extended_train/ckpt_step_0004284_best.pt` (Preserved and untouched)  
**Production Checkpoint Path:** `experiments/checkpoints_1b/`  
**Execution Status:** **PREPARATION COMPLETE / READY TO LAUNCH**  

---

## 1. SPECIFICATION & HYPERPARAMETER MANIFEST

- **Dataset:** FineWeb-Edu Sample-10BT (Curated Jarvis Billion-Token Pre-Training Corpus)
- **Dataset Storage Layout:** 20 Train Shards (`data/shards/train_shard_0000.bin` .. `0019.bin`) + 1 Val Shard (`data/shards/val_shard_0000.bin`)
- **Tokenizer:** GPT-2 `tiktoken` (`vocab_size = 50,257`, `<|endoftext|>` token ID `50,256`)
- **Architecture:** Jarvis-vNext (24 Layers, $d_{\text{model}}=1024$, 16 Attention Heads, RoPE, MoE 4 Experts Top-2, Liquid State Fusion)
- **Total Parameter Count:** **$606,391,704$ parameters (~$606.4$M)**
- **Active Parameters / Token:** **$353,600,000$ parameters (~$353.6$M)**
- **Training Tokens:** **$1,000,000,000$ tokens** (Verified bit-for-bit across 20 shards of 50M tokens each)
- **Validation Tokens:** **$1,588,263$ tokens** (Isolated holdout shard, 0 document overlap)
- **Sequence Length ($T$):** $512$ tokens
- **Effective Micro-Batch Size:** $2$ sequences ($1,024$ tokens / micro-batch)
- **Gradient Accumulation Steps:** $4$ accumulation steps
- **Tokens / Optimizer Update:** $2 \times 512 \times 4 = \mathbf{4,096}$ tokens / update
- **Total Updates Required for 1.0B Tokens:** **$244,140$ optimizer updates**
- **Precision:** BFloat16 mixed precision (`torch.amp.autocast("cuda", dtype=torch.bfloat16)`)
- **Optimizer:** `AdamW(fused=True, lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.1)`
- **Learning Rate Schedule:** Cosine decay with 2,000 warmup steps down to $1.5 \times 10^{-5}$ ($10\%$ floor)
- **Warmup Schedule:** Linear warmup over $2,000$ steps ($8,192,000$ tokens)
- **Gradient Clipping:** $\|g\|_2 \le 1.0$ maximum L2 norm
- **Checkpoint Frequency:** Every 2,500 steps (~10.24M tokens)
- **Evaluation Frequency:** Every 1,250 steps (~5.12M tokens) on holdout shard

---

## 2. EMPIRICAL BENCHMARKS & HARDWARE TELEMETRY

All benchmarks measured on the physical RTX 5070 using `experiments/architecture_matrix/test_training_pipeline.py`:

- **Average Step Latency:** **$5,388.6$ ms / step**
- **Raw Training Throughput:** **$760.1$ tokens / sec**
- **Realistic Effective Throughput:** **$684.1$ tokens / sec** (Factoring periodic validation, checkpoint serialization, and 90% duty cycle)
- **Peak Allocated VRAM:** **$9,420.4$ MB ($9.42$ GB)**
- **Peak Reserved VRAM:** **$10,596.0$ MB ($10.60$ GB)**
- **Free VRAM Headroom:** **$1,630.6$ MB ($13.3\%$ safety margin)**

---

## 3. ESTIMATED TRAINING SCHEDULE

| Milestone | Token Volume | Optimizer Updates | Duration (Hours) | Duration (Days) |
| :--- | :---: | :---: | :---: | :---: |
| **Milestone 1** | 100,000,000 | 24,414 | $40.6$ hrs | $1.69$ days |
| **Milestone 2** | 250,000,000 | 61,035 | $101.5$ hrs | $4.23$ days |
| **Milestone 3** | 500,000,000 | 122,070 | $203.0$ hrs | $8.46$ days |
| **Milestone 4 (Min Target)** | 800,000,000 | 195,312 | $324.8$ hrs | $13.53$ days |
| **Milestone 5 (Full Target)**| **1,000,000,000** | **244,140** | **$406.0$ hrs** | **$16.92$ days** |

---

## 4. CHECKPOINTING & RESUME VALIDATION TEST

- **Test Checkpoint Created:** `experiments/checkpoints_1b/smoke_test_resume.pt`
- **Saved Step:** Step 23 ($94,208$ tokens)
- **Teardown & Reinitialization:** Models, optimizers, and dataloaders were wiped from memory.
- **Resumption Validation:** Fresh model and dataloader loaded the checkpoint:
  - Dataloader resumed at shard 0, token offset $94,208$.
  - Next batch token IDs matched bit-for-bit with ground-truth expectation (`assert (x_expected == x_resumed).all()`).
  - 3 subsequent training steps executed smoothly: loss continued at $7.92 \to 7.95 \to 8.06$.
  - Zero NaN values, zero divergence.
- **Result:** **PASSED CLEANLY**.

## 5. LIVE RUN TELEMETRY & VALIDATION MILESTONES

- **Status:** **IN PROGRESS (Actively Training)**
- **Tokens Trained So Far:** **>5,120,000 tokens**
- **Sustained Throughput:** **741.0 tokens / sec**
- **VRAM Utilization:** **9,420 MB allocated / 10,636 MB reserved** (1.59 GB free headroom)
- **Step 1 Loss:** Cross-Entropy = **8.6143**
- **Step 1,250 Train Loss:** Cross-Entropy = **6.2596** (-2.3547 drop)
- **Step 1,250 Holdout Validation:**
  - **Validation Cross-Entropy:** **6.5345**
  - **Validation Perplexity:** **688.51**
  - **Best Checkpoint Saved:** `experiments/checkpoints_1b/ckpt_best_step_001250.pt`

---

## 6. FINAL STATUS

### **FINAL STATUS: ACTIVELY EXECUTING (HEALTHY & CONVERGING)**

The 1.0B token training run is progressing smoothly on the NVIDIA GeForce RTX 5070 with zero NaNs, zero Infs, and zero memory leaks. Checkpoints are automatically staged in `experiments/checkpoints_1b/`.

