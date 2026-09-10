# experiments/train_600m_extended.py
"""
Extended Controlled Training for Jarvis 606M Model
===================================================
Executes continued training from ckpt_step_0004209.pt with:
1. Decoupled pure Cross-Entropy loss tracking & reporting
2. Curated training corpus (data_clean.txt) with <|endoftext|> boundaries
3. Smooth cosine LR continuation with floor
4. Validation on independent fresh_holdout.txt every 25 steps
5. Periodic checkpointing into experiments/extended_train/
6. Zero corruption of baseline ckpt_step_0004209.pt
"""

import os
import sys
import math
import time
import glob
import json
import statistics
import argparse
import torch
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
SPARSE_DIR = os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")
ATTN_DIR = os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, SPARSE_DIR, ATTN_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken
from jarvis_model import Jarvis

SAVE_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train")
os.makedirs(SAVE_DIR, exist_ok=True)


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
        total_loss = ce_loss + l_bal_total + l_ref_total

    return logits, total_loss, ce_loss, l_bal_total, l_ref_total


@torch.inference_mode()
def evaluate_holdout(model, val_tokens, num_windows=50, seq_len=256, seed=42):
    model.eval()
    model.reset_state()
    max_start = len(val_tokens) - seq_len - 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    window_starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()

    ce_losses = []
    bal_losses = []
    ref_losses = []

    for start in window_starts:
        x = val_tokens[start : start + seq_len].unsqueeze(0)
        y = val_tokens[start + 1 : start + seq_len + 1].unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, tot, ce, bal, ref = decoupled_forward(model, x, targets=y, persist_state=False)
        ce_losses.append(ce.item())
        bal_losses.append(bal.item())
        ref_losses.append(ref.item())

    mean_ce = statistics.mean(ce_losses)
    ppl = math.exp(min(mean_ce, 100.0))
    model.train()
    model.reset_state()
    return mean_ce, ppl, statistics.mean(bal_losses), statistics.mean(ref_losses)


def train_extended(
    steps: int = 200,
    lr: float = 5e-5,
    warmup_steps: int = 25,
    seq_len: int = 512,
    batch_size: int = 2,
    accum_steps: int = 4,
    train_file: str = "data_clean.txt",
    save_every: int = 50,
    use_normalized_ste: bool = True,
):
    print("=" * 85)
    print("        JARVIS 606M EXTENDED CONTINUED TRAINING FROM BASELINE 4209")
    print(f"        Steps: {steps} | LR: {lr} | SeqLen: {seq_len} | Data: {train_file} | NormSTE: {use_normalized_ste}")
    print("=" * 85)

    if use_normalized_ste:
        from utils.ternary_ops import TernaryQuantizeSTE
        def normalized_backward(ctx, grad_output):
            w, = ctx.saved_tensors
            alpha = w.abs().mean().clamp(min=1e-8)
            mask = ((w / alpha).abs() <= 1.0).float()
            return grad_output * mask
        TernaryQuantizeSTE.backward = staticmethod(normalized_backward)

    device = "cuda"
    enc = tiktoken.get_encoding("gpt2")

    train_path = os.path.join(JARVIS_ENGINE, train_file) if not os.path.isabs(train_file) else train_file
    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        train_text = f.read()
    train_tokens = torch.tensor(enc.encode(train_text, allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device)
    print(f"Training tokens GPU-resident: {len(train_tokens):,}")

    val_path = os.path.join(JARVIS_ENGINE, "fresh_holdout.txt")
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_text = f.read()
    val_tokens = torch.tensor(enc.encode(val_text), dtype=torch.long, device=device)
    print(f"Validation tokens GPU-resident: {len(val_tokens):,}")

    # Model instantiation
    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=max(256, seq_len),
        use_cuda_attn=True,
        use_cuda_moe=True,
    ).to(device)

    # Strictly load baseline checkpoint
    base_ckpt_path = os.path.join(JARVIS_ENGINE, "ckpt_step_0004209.pt")
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    start_step = ckpt.get("step", 4209)
    print(f"[OK] Strictly loaded baseline checkpoint at step {start_step}")
    del ckpt, sd, new_sd
    torch.cuda.empty_cache()

    # Initial holdout evaluation
    init_ce, init_ppl, init_bal, init_ref = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=min(256, seq_len))
    print(f"[Initial Baseline Step {start_step}] Holdout CE: {init_ce:.4f} | PPL: {init_ppl:.2f} | L_bal: {init_bal:.4f} | L_ref: {init_ref:.4f}")

    # Fused AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

    def get_lr(step_idx):
        if step_idx < warmup_steps:
            return lr * (step_idx + 1) / warmup_steps
        progress = (step_idx - warmup_steps) / max(1, steps - warmup_steps)
        # Cosine decay with 20% floor
        return lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    _offsets = torch.arange(seq_len, device=device)
    def get_batch():
        ix = torch.randint(0, len(train_tokens) - seq_len - 1, (batch_size,), device=device)
        idx = ix.unsqueeze(1) + _offsets
        return train_tokens[idx], train_tokens[idx + 1]

    trajectory = []
    t_start = time.perf_counter()
    best_holdout_ce = init_ce
    best_ckpt_path = None

    for step_idx in range(1, steps + 1):
        global_step = start_step + step_idx
        cur_lr = get_lr(step_idx)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        ce_accum = 0.0
        tot_accum = 0.0

        for _ in range(accum_steps):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, tot_loss, ce_loss, _, _ = decoupled_forward(model, x, targets=y)
            (tot_loss / accum_steps).backward()
            ce_accum += ce_loss.item() / accum_steps
            tot_accum += tot_loss.item() / accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        optimizer.step()

        # Logging
        if step_idx % 5 == 0 or step_idx == 1 or step_idx == steps:
            dt = time.perf_counter() - t_start
            tok_s = (batch_size * accum_steps * seq_len * step_idx) / max(dt, 1e-4)
            alloc_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"step {global_step:05d} (+{step_idx:03d}): train_ce {ce_accum:.4f} | total_loss {tot_accum:.4f} | lr {cur_lr:.6f} | grad_norm {norm_val:.3f} | {tok_s:.0f} tok/s | VRAM {alloc_gb:.2f} GB")

        # Validation evaluation
        if step_idx % 25 == 0 or step_idx == steps:
            torch.cuda.synchronize()
            v_ce, v_ppl, v_bal, v_ref = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=min(256, seq_len))
            print(f"  -> [VALIDATION @ step {global_step:05d}] Holdout CE: {v_ce:.4f} (Delta: {v_ce - init_ce:+.4f}) | PPL: {v_ppl:.2f} (Delta: {v_ppl - init_ppl:+.2f})")

            rec = {
                "step": global_step,
                "relative_step": step_idx,
                "train_ce": ce_accum,
                "train_total_loss": tot_accum,
                "holdout_ce": v_ce,
                "holdout_ppl": v_ppl,
                "holdout_bal_loss": v_bal,
                "holdout_ref_loss": v_ref,
                "grad_norm": norm_val,
                "lr": cur_lr,
            }
            trajectory.append(rec)

            if v_ce < best_holdout_ce:
                best_holdout_ce = v_ce
                best_ckpt_path = os.path.join(SAVE_DIR, f"ckpt_step_{global_step:07d}_best.pt")
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": v_ce,
                    "val_perplexity": v_ppl,
                }, best_ckpt_path)
                print(f"     [NEW BEST] Saved best checkpoint: {os.path.basename(best_ckpt_path)}")

        # Periodic checkpoint
        if step_idx % save_every == 0 or step_idx == steps:
            save_path = os.path.join(SAVE_DIR, f"ckpt_step_{global_step:07d}.pt")
            torch.save({
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "val_loss": trajectory[-1]["holdout_ce"] if trajectory else None,
                "val_perplexity": trajectory[-1]["holdout_ppl"] if trajectory else None,
            }, save_path)
            print(f"  [CKPT] Saved checkpoint: {os.path.basename(save_path)}")

    # Final summary report
    summary = {
        "start_step": start_step,
        "end_step": start_step + steps,
        "total_new_steps": steps,
        "tokens_trained": steps * batch_size * accum_steps * seq_len,
        "train_corpus": train_file,
        "learning_rate": lr,
        "initial_holdout_ce": init_ce,
        "initial_holdout_ppl": init_ppl,
        "best_holdout_ce": best_holdout_ce,
        "best_holdout_ppl": math.exp(min(best_holdout_ce, 100.0)),
        "final_holdout_ce": trajectory[-1]["holdout_ce"],
        "final_holdout_ppl": trajectory[-1]["holdout_ppl"],
        "ce_improvement": init_ce - best_holdout_ce,
        "ppl_improvement": init_ppl - math.exp(min(best_holdout_ce, 100.0)),
        "best_checkpoint": best_ckpt_path,
        "trajectory": trajectory,
    }

    sum_path = os.path.join(SAVE_DIR, "extended_training_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85)
    print("EXTENDED TRAINING COMPLETE")
    print(f"  Initial Holdout CE: {init_ce:.4f} | PPL: {init_ppl:.2f}")
    print(f"  Best Holdout CE:    {best_holdout_ce:.4f} | PPL: {math.exp(min(best_holdout_ce, 100.0)):.2f}")
    print(f"  CE Improvement:     {init_ce - best_holdout_ce:+.4f}")
    print(f"  PPL Improvement:    {init_ppl - math.exp(min(best_holdout_ce, 100.0)):+.2f}")
    print(f"  Summary Report:     {sum_path}")
    print("=" * 85)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train-file", type=str, default="data_clean.txt")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--no-norm-ste", action="store_true", help="Disable normalized STE")
    args = parser.parse_args()

    train_extended(
        steps=args.steps,
        lr=args.lr,
        train_file=args.train_file,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        use_normalized_ste=not args.no_norm_ste,
    )
