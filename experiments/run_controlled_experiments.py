# experiments/run_controlled_experiments.py
"""
Controlled Hyperparameter & Architecture Experiments for Jarvis 606M
=====================================================================
Executes controlled 100-step training experiments from ckpt_step_0004209.pt:

Experiment 1: Data Curation (data.txt vs data_clean.txt)
Experiment 2: Ternary STE (Baseline STE vs Normalized STE)
Experiment 3: Learning Rate Sweep (1e-4 vs 2e-4 vs 3e-4 with warm continuation)
Experiment 4: Context Window Progression (T=256 vs T=512)

Strict rules:
1. NEVER overwrite ckpt_step_0004209.pt
2. Evaluate holdout CE and PPL on fresh_holdout.txt every 25 steps
3. Rank configurations strictly by HOLDOUT CE
4. Save results to individual experiment subdirectories
"""

import os
import sys
import math
import time
import json
import statistics
import argparse
import torch
import torch.nn as nn
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
    for start in window_starts:
        x = val_tokens[start : start + seq_len].unsqueeze(0)
        y = val_tokens[start + 1 : start + seq_len + 1].unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, _, ce_loss, _, _ = decoupled_forward(model, x, targets=y, persist_state=False)
        ce_losses.append(ce_loss.item())

    mean_ce = statistics.mean(ce_losses)
    ppl = math.exp(min(mean_ce, 100.0))
    model.train()
    model.reset_state()
    return mean_ce, ppl


def run_experiment(
    exp_name: str,
    train_file: str,
    lr: float = 2e-4,
    warmup_steps: int = 20,
    num_steps: int = 100,
    seq_len: int = 256,
    batch_size: int = 2,
    accum_steps: int = 4,
    use_normalized_ste: bool = False,
):
    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"  Data: {train_file} | LR: {lr} | SeqLen: {seq_len} | Steps: {num_steps} | NormSTE: {use_normalized_ste}")
    print("=" * 80)

    exp_dir = os.path.join(WORKSPACE_ROOT, "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    device = "cuda"

    # Load training tokens
    with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
        train_text = f.read()
    train_tokens = torch.tensor(enc.encode(train_text, allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device)
    print(f"Loaded {len(train_tokens):,} training tokens")

    # Load validation tokens
    val_path = os.path.join(JARVIS_ENGINE, "fresh_holdout.txt")
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_text = f.read()
    val_tokens = torch.tensor(enc.encode(val_text), dtype=torch.long, device=device)

    # Initialize model
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

    # Load baseline checkpoint
    ckpt_path = os.path.join(JARVIS_ENGINE, "ckpt_step_0004209.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    del ckpt, sd, new_sd
    torch.cuda.empty_cache()

    # Apply Normalized STE if requested
    if use_normalized_ste:
        from utils.ternary_ops import TernaryQuantizeSTE
        # Monkey-patch TernaryQuantizeSTE backward to normalize mask
        def normalized_backward(ctx, grad_output):
            w, = ctx.saved_tensors
            alpha = w.abs().mean().clamp(min=1e-8)
            mask = ((w / alpha).abs() <= 1.0).float()
            return grad_output * mask
        TernaryQuantizeSTE.backward = staticmethod(normalized_backward)

    # Optimizer with warm continuation schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    # Initial validation before step 0
    init_ce, init_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=min(256, seq_len))
    print(f"Step 000 (Initial Baseline): Holdout CE = {init_ce:.4f} | Holdout PPL = {init_ppl:.2f}")

    trajectory = [{
        "step": 0,
        "train_ce": None,
        "holdout_ce": init_ce,
        "holdout_ppl": init_ppl,
        "lr": 0.0,
    }]

    _offsets = torch.arange(seq_len, device=device)
    def get_batch():
        ix = torch.randint(0, len(train_tokens) - seq_len - 1, (batch_size,), device=device)
        idx = ix.unsqueeze(1) + _offsets
        return train_tokens[idx], train_tokens[idx + 1]

    model.train()
    t_start = time.perf_counter()

    for step in range(1, num_steps + 1):
        cur_lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        train_ce_accum = 0.0
        train_tot_accum = 0.0

        for _ in range(accum_steps):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, tot_loss, ce_loss, _, _ = decoupled_forward(model, x, targets=y)
            (tot_loss / accum_steps).backward()
            train_ce_accum += ce_loss.item() / accum_steps
            train_tot_accum += tot_loss.item() / accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 25 == 0 or step == num_steps:
            torch.cuda.synchronize()
            val_ce, val_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=min(256, seq_len))
            dt = time.perf_counter() - t_start
            tok_s = (step * batch_size * accum_steps * seq_len) / max(dt, 1e-4)
            print(f"  Step {step:03d}/{num_steps}: Train CE = {train_ce_accum:.4f} | Holdout CE = {val_ce:.4f} | PPL = {val_ppl:.2f} | {tok_s:.0f} tok/s")
            trajectory.append({
                "step": step,
                "train_ce": train_ce_accum,
                "train_total_loss": train_tot_accum,
                "holdout_ce": val_ce,
                "holdout_ppl": val_ppl,
                "lr": cur_lr,
            })

    # Save experiment report
    final_metrics = trajectory[-1]
    report = {
        "experiment": exp_name,
        "train_file": os.path.basename(train_file),
        "learning_rate": lr,
        "warmup_steps": warmup_steps,
        "total_steps": num_steps,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "effective_batch_tokens": batch_size * accum_steps * seq_len,
        "use_normalized_ste": use_normalized_ste,
        "initial_holdout_ce": init_ce,
        "initial_holdout_ppl": init_ppl,
        "final_holdout_ce": final_metrics["holdout_ce"],
        "final_holdout_ppl": final_metrics["holdout_ppl"],
        "ce_delta": final_metrics["holdout_ce"] - init_ce,
        "ppl_delta": final_metrics["holdout_ppl"] - init_ppl,
        "trajectory": trajectory,
    }

    report_path = os.path.join(exp_dir, "experiment_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save final checkpoint of this experiment
    ckpt_save_path = os.path.join(exp_dir, f"ckpt_exp_{exp_name}_final.pt")
    torch.save({
        "step": 4209 + num_steps,
        "model_state_dict": model.state_dict(),
        "val_loss": final_metrics["holdout_ce"],
        "val_perplexity": final_metrics["holdout_ppl"],
    }, ckpt_save_path)

    print(f"[OK] Experiment {exp_name} completed.")
    print(f"     Holdout CE:  {init_ce:.4f} -> {final_metrics['holdout_ce']:.4f} (Delta: {final_metrics['holdout_ce'] - init_ce:+.4f})")
    print(f"     Holdout PPL: {init_ppl:.2f} -> {final_metrics['holdout_ppl']:.2f} (Delta: {final_metrics['holdout_ppl'] - init_ppl:+.2f})")
    print(f"     Report:      {report_path}")
    print(f"     Checkpoint:  {ckpt_save_path}\n")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", choices=["all", "data", "ste", "lr", "context"])
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    results = {}
    train_orig = os.path.join(JARVIS_ENGINE, "data.txt")
    train_clean = os.path.join(JARVIS_ENGINE, "data_clean.txt")

    if args.exp in ["all", "data"]:
        # Experiment 1: Original data.txt vs Curated data_clean.txt
        res_data_orig = run_experiment("exp_data_orig", train_orig, lr=2e-4, num_steps=args.steps)
        res_data_clean = run_experiment("exp_data_clean", train_clean, lr=2e-4, num_steps=args.steps)
        results["exp_data_orig"] = res_data_orig
        results["exp_data_clean"] = res_data_clean

    if args.exp in ["all", "ste"]:
        # Experiment 2: Baseline STE vs Normalized STE with optimal LR 5e-5
        res_ste_norm = run_experiment("exp_ste_normalized", train_clean, lr=5e-5, num_steps=args.steps, use_normalized_ste=True)
        results["exp_ste_normalized"] = res_ste_norm

    if args.exp in ["all", "lr"]:
        # Experiment 3: LR 5e-5 vs 1e-4
        res_lr_5e5 = run_experiment("exp_lr_5e5", train_clean, lr=5e-5, num_steps=args.steps)
        res_lr_1e4 = run_experiment("exp_lr_1e4", train_clean, lr=1e-4, num_steps=args.steps)
        results["exp_lr_5e5"] = res_lr_5e5
        results["exp_lr_1e4"] = res_lr_1e4

    if args.exp in ["all", "context"]:
        # Experiment 4: T=512 context training with optimal LR 5e-5
        res_ctx_512 = run_experiment("exp_context_512", train_clean, lr=5e-5, seq_len=512, batch_size=2, accum_steps=4, num_steps=args.steps)
        results["exp_context_512"] = res_ctx_512

    # Compile Summary Table
    summary_path = os.path.join(WORKSPACE_ROOT, "experiments", "controlled_experiments_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print("               CONTROLLED EXPERIMENTS LEADERBOARD (Ranked by Holdout CE)")
    print("=" * 90)
    ranked = sorted(results.values(), key=lambda r: r["final_holdout_ce"])
    print(f"{'Rank':<5} | {'Experiment Name':<22} | {'Init CE':<8} | {'Final CE':<9} | {'CE Delta':<9} | {'Final PPL':<9}")
    print("-" * 90)
    for idx, r in enumerate(ranked, 1):
        print(f"{idx:<5} | {r['experiment']:<22} | {r['initial_holdout_ce']:<8.4f} | {r['final_holdout_ce']:<9.4f} | {r['ce_delta']:<+9.4f} | {r['final_holdout_ppl']:<9.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
