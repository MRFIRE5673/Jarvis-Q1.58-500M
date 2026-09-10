# experiments/architecture_matrix/run_longer_adaptation_e_only.py
"""
EXPERIMENT A: E-ONLY (W=16 LOCAL BUFFER) 500-STEP CONTROLLED ADAPTATION
========================================================================
Runs a controlled 500-step adaptation test from the paper-faithful baseline
checkpoint (ckpt_step_0004284_best.pt) for E-Only (W=16 Local Buffer).

Evaluates periodically at Steps: 0, 100, 250, 500.
Measures at every checkpoint:
1. T=512 Holdout CE & PPL
2. T=1024 Holdout CE & PPL
3. Canonical Associative Needle Retrieval Rank @ 64 tok
4. Memory Path Contributions (Local Buffer vs Recurrent Memory)
5. Training Loss Stability & Gradient Norm
6. VRAM Accounting (Allocated & Reserved)
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken
from jarvis_model import Jarvis
from modular_memory import build_modular_jarvis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def evaluate_holdout_multicontext(model, val_tokens, seq_len=512, num_windows=20, seed=42):
    model.eval()
    max_start = len(val_tokens) - seq_len - 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()
    
    losses = []
    for s in starts:
        x = val_tokens[s : s + seq_len].unsqueeze(0).to(DEVICE)
        y = val_tokens[s + 1 : s + seq_len + 1].unsqueeze(0).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(ce.item())
        
    mean_ce = float(np.mean(losses))
    ppl = float(math.exp(min(mean_ce, 50.0)))
    return mean_ce, ppl

@torch.inference_mode()
def evaluate_needle_rank(model, enc, corpus_tokens, distance=64, num_trials=10):
    model.eval()
    needle_target = " 42"
    target_id = enc.encode(needle_target)[0]
    
    needle = "The system authentication passcode is 42.\n"
    query = "\nWhat is the system authentication passcode? The system authentication passcode is"
    needle_toks = enc.encode(needle)
    query_toks = enc.encode(query)
    
    ranks = []
    max_idx = len(corpus_tokens) - distance - 100
    for trial in range(num_trials):
        start_idx = (trial * 13337 + distance * 97) % max(max_idx, 1)
        distractors = corpus_tokens[start_idx : start_idx + distance]
        full_toks = needle_toks + distractors + query_toks
        inp = torch.tensor([full_toks], dtype=torch.long, device=DEVICE)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
        last_logits = logits[0, -1, :]
        rank = (last_logits > last_logits[target_id]).sum().item() + 1
        ranks.append(rank)
        
    return float(np.mean(ranks))

@torch.inference_mode()
def collect_path_diagnostics(model, dummy_input):
    model.eval()
    local_norms = []
    rec_norms = []
    fuse_gates = []
    
    x = model.tok_emb(dummy_input)
    
    for block in model.blocks:
        x_norm = block.norm1(x)
        attn = block.attn
        
        fuse = torch.sigmoid(attn.blend_gate).view(-1).cpu().numpy()
        fuse_gates.extend(fuse)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, diag = attn(x_norm, collect_diagnostics=True)
            local_norms.append(diag["local_norm"])
            rec_norms.append(diag["recurrent_norm"])
            x, _, _, _ = block(x)

    mean_l_norm = float(np.mean(local_norms))
    mean_r_norm = float(np.mean(rec_norms))
    total_norm = max(mean_l_norm + mean_r_norm, 1e-6)
    
    return {
        "local_buffer_norm": mean_l_norm,
        "recurrent_memory_norm": mean_r_norm,
        "local_contribution_ratio": mean_l_norm / total_norm,
        "recurrent_contribution_ratio": mean_r_norm / total_norm,
        "fusion_gate_mean": float(np.mean(fuse_gates)),
    }


def main():
    print("=" * 85)
    print("EXPERIMENT A: E-ONLY (LOCAL BUFFER W=16) 500-STEP CONTROLLED ADAPTATION")
    print(f"Device: {DEVICE}")
    print("=" * 85)
    
    enc = tiktoken.get_encoding("gpt2")
    with open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore") as f:
        train_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    with open(os.path.join(JARVIS_ENGINE, "fresh_holdout.txt"), "r", encoding="utf-8", errors="ignore") as f:
        val_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    corpus_tokens = enc.encode(open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore").read(), allowed_special={"<|endoftext|>"})
    
    e_cfg = {
        "use_local_buffer": True,
        "local_window_size": 16,
        "use_adaptive_decay": False,
        "use_write_gate": False,
        "use_erase_gate": False,
        "use_gated_read": False,
        "fusion_option": 2,
    }
    
    model = build_modular_jarvis(config_dict=e_cfg, max_seq_len=512).to(DEVICE)
    
    # Load paper baseline checkpoint
    base_ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model_sd = model.state_dict()
    filtered_sd = {k: v for k, v in new_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered_sd, strict=False)
    
    # Apply neutral initialization
    for block in model.blocks:
        if hasattr(block.attn, "init_neutral"):
            block.attn.init_neutral()
            
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Loaded {len(filtered_sd)} baseline tensors with neutral identity initialization.")
    print(f"Total Parameters in E-Only Model: {total_p:,} (+384 parameters overhead)")
    
    lr = 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    batch_size = 2
    grad_accum_steps = 4
    seq_len = 512
    tokens_per_step = batch_size * grad_accum_steps * seq_len
    total_steps = 500
    
    eval_checkpoints = [0, 100, 250, 500]
    results_timeline = {}
    
    _offsets = torch.arange(seq_len, device=DEVICE)
    def get_batch():
        starts = torch.randint(0, len(train_tokens) - seq_len - 1, (batch_size,), device=DEVICE)
        idx = starts.unsqueeze(1) + _offsets.unsqueeze(0)
        x = train_tokens[idx]
        y = train_tokens[idx + 1]
        return x, y
    
    dummy_diag = torch.randint(0, 50257, (2, 512), device=DEVICE)
    
    print(f"\nTraining Plan: {total_steps} steps | Accum: {grad_accum_steps} | Tokens/step: {tokens_per_step:,} | Total: {total_steps * tokens_per_step:,} tokens")
    print(f"Checkpoints: {eval_checkpoints}")
    print("-" * 85)
    
    step_times = []
    grad_norms = []
    train_losses = []
    
    for step in range(total_steps + 1):
        if step in eval_checkpoints:
            t_eval_0 = time.perf_counter()
            ce_512, ppl_512 = evaluate_holdout_multicontext(model, val_tokens, seq_len=512, num_windows=20, seed=42 + step)
            ce_1024, ppl_1024 = evaluate_holdout_multicontext(model, val_tokens, seq_len=1024, num_windows=20, seed=42 + step)
            needle_rank = evaluate_needle_rank(model, enc, corpus_tokens, distance=64, num_trials=10)
            diag = collect_path_diagnostics(model, dummy_diag)
            vram_alloc = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
            vram_res = torch.cuda.max_memory_reserved(DEVICE) / (1024 * 1024)
            free_b, total_b = torch.cuda.mem_get_info(DEVICE)
            vram_driver = (total_b - free_b) / (1024 * 1024)
            eval_dt = time.perf_counter() - t_eval_0
            
            print(f"\n>>> [E-ONLY CHECKPOINT STEP {step:03d}/{total_steps}] (Eval Time: {eval_dt:.1f}s)")
            print(f"  T= 512: Holdout CE = {ce_512:.4f} | PPL = {ppl_512:.2f}")
            print(f"  T=1024: Holdout CE = {ce_1024:.4f} | PPL = {ppl_1024:.2f}")
            print(f"  Needle Rank @ 64: {needle_rank:.1f} / 50257")
            print(f"  Contributions: Local Buffer = {diag['local_contribution_ratio']*100:.1f}% | Recurrent = {diag['recurrent_contribution_ratio']*100:.1f}%")
            print(f"  VRAM: Allocated = {vram_alloc:.1f} MB | Reserved = {vram_res:.1f} MB | Driver Used = {vram_driver:.1f} MB / {total_b/(1024*1024):.1f} MB\n")
            
            results_timeline[f"step_{step}"] = {
                "step": step,
                "ce_512": ce_512,
                "ppl_512": ppl_512,
                "ce_1024": ce_1024,
                "ppl_1024": ppl_1024,
                "needle_rank_64": needle_rank,
                "diagnostics": diag,
                "vram_allocated_mb": vram_alloc,
                "vram_reserved_mb": vram_res,
                "vram_driver_used_mb": vram_driver,
                "vram_total_device_mb": total_b / (1024 * 1024),
            }
            
            # Save intermediate checkpoint
            ckpt_path = os.path.join(ARCH_DIR, f"ckpt_e_only_step{step:04d}.pt")
            torch.save({
                "step": step,
                "model_state_dict": {k: v.bfloat16() for k, v in model.state_dict().items()},
                "ce_512": ce_512,
                "ppl_512": ppl_512,
            }, ckpt_path)
            
        if step >= total_steps:
            break
            
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t_step_0 = time.perf_counter()
        accum_loss = 0.0
        
        for _ in range(grad_accum_steps):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * grad_accum_steps
            
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        torch.cuda.synchronize(DEVICE)
        dt = time.perf_counter() - t_step_0
        
        step_times.append(dt)
        grad_norms.append(gnorm)
        train_losses.append(accum_loss)
        
        if (step + 1) % 25 == 0:
            tok_s = tokens_per_step / max(np.mean(step_times[-25:]), 1e-5)
            print(f"  Step {step + 1:03d}/{total_steps}: Loss = {accum_loss:.4f} | GradNorm = {gnorm:.2f} | Speed = {tok_s:5.0f} tok/s | Latency = {dt*1000:5.1f} ms")

    print("=" * 85)
    print("E-ONLY 500-STEP ADAPTATION COMPLETED SUCCESSFULLY")
    print("=" * 85)
    
    out_file = os.path.join(ARCH_DIR, "longer_adaptation_e_only_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timeline": results_timeline,
            "training_dynamics": {
                "mean_grad_norm": float(np.mean(grad_norms)),
                "std_grad_norm": float(np.std(grad_norms)),
                "max_grad_norm": float(np.max(grad_norms)),
                "initial_loss": train_losses[0] if train_losses else 0.0,
                "final_loss": train_losses[-1] if train_losses else 0.0,
                "mean_step_time_ms": float(np.mean(step_times)) * 1000.0,
            }
        }, f, indent=2)
    print(f"\n[OK] E-only report saved to: {out_file}")

if __name__ == "__main__":
    main()
