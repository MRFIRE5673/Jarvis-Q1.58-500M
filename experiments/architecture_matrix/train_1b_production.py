# experiments/architecture_matrix/train_1b_production.py
"""
JARVIS 1.0 BILLION TOKEN PRODUCTION TRAINING PIPELINE
=====================================================
Production training engine for Jarvis 607M pre-training / continual learning
on 1,000,000,000 tokens of curated FineWeb-Edu shards.

Key Features:
- Zero-copy streaming dataloader (ShardedTokenDataset) with np.memmap.
- Decoupled Cross-Entropy, MoE load balancing, and reflective penalty tracking.
- BFloat16 mixed precision with gradient accumulation and gradient checkpointing.
- AdamW (fused) with cosine learning rate schedule and warmup.
- Gradient norm clipping (||g||_2 <= 1.0).
- Atomic rolling checkpoints storing model, optimizer, scheduler, and dataloader states.
- Dedicated validation on unseen holdout shards.
"""

import os
import sys
import math
import time
import json
import argparse
import torch
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")
SHARDS_DIR = os.path.join(DATA_DIR, "shards")
CHECKPOINT_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "checkpoints_1b")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import Jarvis
from data.streaming_dataloader import ShardedTokenDataset


def decoupled_forward(model, idx, targets=None, persist_state=False):
    B, T = idx.shape
    x = model.tok_emb(idx)

    start_pos = model._token_pos if persist_state else 0
    if persist_state:
        model._token_pos += T

    l_bal_total = torch.tensor(0.0, device=idx.device)
    l_ref_total = torch.tensor(0.0, device=idx.device)

    h_prevs = model._h_states if persist_state else [None] * len(model.blocks)
    new_h_states = []

    for i, block in enumerate(model.blocks):
        h_prev = h_prevs[i]
        if h_prev is not None and h_prev.shape[0] != B:
            h_prev = None

        if model.training:
            from torch.utils.checkpoint import checkpoint as grad_ckpt
            x, h_last, l_bal, l_ref = grad_ckpt(block, x, h_prev, start_pos, use_reentrant=False)
        else:
            x, h_last, l_bal, l_ref = block(x, h_prev, start_pos=start_pos)

        new_h_states.append(h_last.detach())
        l_bal_total = l_bal_total + l_bal
        l_ref_total = l_ref_total + l_ref

    model._h_states = new_h_states
    x = model.final_norm(x)
    logits = model.lm_head(x)

    total_loss, ce_loss = None, None
    if targets is not None:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss = ce_loss + 0.01 * l_bal_total + 0.001 * l_ref_total

    return logits, total_loss, ce_loss, l_bal_total, l_ref_total


@torch.inference_mode()
def evaluate_validation(model, val_loader, num_batches=32):
    model.eval()
    model.reset_state()
    ce_list = []
    
    for _ in range(num_batches):
        x, y = val_loader.next_batch()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, _, ce_loss, _, _ = decoupled_forward(model, x, targets=y, persist_state=False)
        ce_list.append(ce_loss.item())
        
    avg_ce = float(sum(ce_list) / max(len(ce_list), 1))
    ppl = math.exp(min(avg_ce, 20.0))
    model.train()
    return avg_ce, ppl


def get_lr_cosine(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(path, model, optimizer, step, tokens_trained, dataloader, best_val_ce, loss_history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    state = {
        "step": step,
        "tokens_trained": tokens_trained,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dataloader_state": dataloader.get_state(),
        "best_val_ce": best_val_ce,
        "loss_history": loss_history[-200:], # keep recent history
        "timestamp": time.time(),
    }
    torch.save(state, temp_path)
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)
    print(f"  [CHECKPOINT SAVED] {path} (Step {step:,}, Tokens {tokens_trained:,})")


def load_checkpoint(path, model, optimizer=None, dataloader=None, device="cuda"):
    print(f"Loading checkpoint from: {path}")
    state = torch.load(path, map_location=device)
    
    sd = state["model_state_dict"]
    clean_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        
    if dataloader is not None and "dataloader_state" in state:
        dataloader.load_state(state["dataloader_state"])
        print(f"  Dataloader resumed at shard {state['dataloader_state']['current_shard_idx']}, offset {state['dataloader_state']['current_offset']:,}")
        
    step = state.get("step", 0)
    tokens_trained = state.get("tokens_trained", 0)
    best_val_ce = state.get("best_val_ce", float("inf"))
    loss_history = state.get("loss_history", [])
    print(f"[OK] Checkpoint successfully loaded (Step {step:,}, Tokens {tokens_trained:,})")
    return step, tokens_trained, best_val_ce, loss_history


def train(
    max_tokens=1_000_000_000,
    seq_len=512,
    micro_batch=2,
    accum_steps=4,
    max_lr=1.5e-4,
    min_lr=1.5e-5,
    warmup_steps=2000,
    save_every_steps=2500,
    eval_every_steps=1250,
    resume_checkpoint=None,
    init_from_baseline=None,
    device="cuda",
):
    print("=" * 80)
    print(f"STARTING JARVIS 1.0B TOKEN TRAINING ENGINE")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    tokens_per_step = micro_batch * seq_len * accum_steps # 2 * 512 * 4 = 4096 tokens
    total_steps = max_tokens // tokens_per_step # 244,140 steps for 1.0B
    print(f"Configuration:")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Micro Batch:     {micro_batch}")
    print(f"  Gradient Accum:  {accum_steps}")
    print(f"  Tokens / Update: {tokens_per_step:,}")
    print(f"  Total Updates:   {total_steps:,} (for {max_tokens:,} tokens)")
    
    # Initialize model
    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=seq_len,
        use_cuda_attn=False, # PyTorch vectorized fallback for stability
        use_cuda_moe=True,
    ).to(device)
    
    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    
    # Initialize Dataloaders
    train_loader = ShardedTokenDataset(
        shards_dir=SHARDS_DIR,
        split="train",
        seq_len=seq_len,
        batch_size=micro_batch,
        device=device,
        seed=42,
    )
    
    val_loader = ShardedTokenDataset(
        shards_dir=SHARDS_DIR,
        split="val",
        seq_len=seq_len,
        batch_size=micro_batch,
        device=device,
        seed=1337,
        shuffle_shards=False,
    )
    
    start_step = 0
    tokens_trained = 0
    best_val_ce = float("inf")
    loss_history = []
    
    # Load checkpoint or initialize
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_step, tokens_trained, best_val_ce, loss_history = load_checkpoint(
            resume_checkpoint, model, optimizer, train_loader, device=device
        )
    elif init_from_baseline and os.path.exists(init_from_baseline):
        print(f"Initializing model weights from baseline checkpoint: {init_from_baseline}")
        ckpt = torch.load(init_from_baseline, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        clean_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        model.load_state_dict(clean_sd, strict=False)
        print("[OK] Baseline weights mapped successfully.")
        
    model.train()
    t_start = time.perf_counter()
    last_log_time = t_start
    
    print("\nBeginning training updates...")
    for step in range(start_step + 1, total_steps + 1):
        lr = get_lr_cosine(step, warmup_steps, total_steps, max_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
            
        optimizer.zero_grad(set_to_none=True)
        ce_accum = 0.0
        tot_accum = 0.0
        
        t0_step = time.perf_counter()
        for micro_idx in range(accum_steps):
            x, y = train_loader.next_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, tot_loss, ce_loss, _, _ = decoupled_forward(model, x, targets=y)
                
            loss_to_back = tot_loss / accum_steps
            loss_to_back.backward()
            
            ce_accum += ce_loss.item() / accum_steps
            tot_accum += tot_loss.item() / accum_steps
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        
        if math.isnan(tot_accum) or math.isnan(norm_val) or norm_val > 100.0:
            print(f"[FATAL ERROR] Loss or GradNorm exploded at step {step}: Loss={tot_accum}, Norm={norm_val}")
            raise RuntimeError("Divergence detected.")
            
        optimizer.step()
        t_step_dur = time.perf_counter() - t0_step
        
        tokens_trained += tokens_per_step
        loss_history.append({"step": step, "ce": ce_accum, "norm": norm_val, "lr": lr})
        
        # Periodic logging
        if step % 10 == 0 or step == 1 or step == total_steps:
            now = time.perf_counter()
            dt = now - t_start
            tok_s = (step - start_step) * tokens_per_step / max(dt, 1e-4)
            alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            print(
                f"step {step:06d}/{total_steps:06d}: train_ce {ce_accum:.4f} | lr {lr:.6f} | "
                f"grad_norm {norm_val:.3f} | {tok_s:,.0f} tok/s | VRAM: {alloc_mb:.0f}M alloc / {res_mb:.0f}M res",
                flush=True
            )
            
        # Periodic validation
        if step % eval_every_steps == 0:
            torch.cuda.synchronize()
            val_ce, val_ppl = evaluate_validation(model, val_loader)
            print(f"\n  -> [VAL @ Step {step:,}] Validation CE: {val_ce:.4f} | PPL: {val_ppl:.2f}", flush=True)
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_path = os.path.join(CHECKPOINT_DIR, f"ckpt_best_step_{step:06d}.pt")
                save_checkpoint(best_path, model, optimizer, step, tokens_trained, train_loader, best_val_ce, loss_history)
            print("", flush=True)
            
        # Periodic checkpoint
        if step % save_every_steps == 0 or step == total_steps:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_step_{step:06d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, tokens_trained, train_loader, best_val_ce, loss_history)
            
    print(f"\n[OK] Training completed! Total tokens trained: {tokens_trained:,}")


if __name__ == "__main__":
    train()
