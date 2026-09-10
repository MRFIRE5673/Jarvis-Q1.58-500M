# experiments/architecture_matrix/run_decoupled_gate_lr_bce.py
"""
EXPERIMENT C: GATE LEARNING RATE DECOUPLING (10x GATE LR) & CONTENT-DEPENDENCE
=============================================================================
Investigates whether write and erase gates fail to learn because the backbone
learning rate (5e-5) is too low for newly initialized projections.

Decoupled LR Configuration:
- Backbone LR: 5e-5 (constant, identical to baseline and standard B+C+E)
- Gate LR: 5e-4 (10x higher, with cosine decay schedule down to 5e-5)

Evaluates periodically at Steps: 0, 100, 250, 500.
Measures at every checkpoint:
1. T=512 Holdout CE & PPL
2. T=1024 Holdout CE & PPL
3. Canonical Associative Needle Retrieval Rank @ 64 tok
4. Memory Path Contributions (Local Buffer vs Recurrent Memory)
5. Comprehensive Gate Diagnostics:
   - mean, std, min, max, fraction >0.95, fraction <0.05
   - parameter displacement ||W_t - W_0||, ||b_t - b_0||
   - gradient norms ||grad(W_write)||, ||grad(W_erase)||
6. Gate Content-Dependence Test:
   - per-token variance
   - per-sequence variance Var_t(gate)
   - correlation with input norm ||x_t||
   - 10-bin activation distribution histogram
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
def collect_path_and_gate_diagnostics(model, dummy_input, initial_params):
    model.eval()
    local_norms = []
    rec_norms = []
    fuse_gates = []
    
    w_gate_vals = []
    e_gate_vals = []
    w_seq_vars = []
    e_seq_vars = []
    w_x_corrs = []
    e_x_corrs = []
    
    x = model.tok_emb(dummy_input)
    
    for block in model.blocks:
        x_norm = block.norm1(x)
        attn = block.attn
        
        # Extract gate activations directly
        # x_norm: (B, T, C)
        B, T, C = x_norm.shape
        x_tok_norm = torch.norm(x_norm, dim=-1).cpu().float().numpy() # (B, T)
        
        w_raw = attn.write_gate_proj(x_norm) # (B, T, H)
        w_act = torch.sigmoid(w_raw).cpu().float().numpy()
        
        e_raw = attn.erase_gate_proj(x_norm) # (B, T, H)
        e_act = torch.sigmoid(e_raw).cpu().float().numpy()
        
        w_gate_vals.append(w_act)
        e_gate_vals.append(e_act)
        
        # Per-sequence variance: variance over time dimension T
        # Var_t(w_act): shape (B, H)
        w_seq_vars.append(np.mean(np.var(w_act, axis=1)))
        e_seq_vars.append(np.mean(np.var(e_act, axis=1)))
        
        # Correlation with input norm ||x_t||
        # Compute correlation between x_tok_norm and w_act mean over heads
        w_mean_heads = np.mean(w_act, axis=-1) # (B, T)
        e_mean_heads = np.mean(e_act, axis=-1) # (B, T)
        
        # Flatten batch & tokens
        x_flat = x_tok_norm.flatten()
        w_flat = w_mean_heads.flatten()
        e_flat = e_mean_heads.flatten()
        
        # Safe Pearson correlation
        if np.std(w_flat) > 1e-7 and np.std(x_flat) > 1e-7:
            r_w = float(np.corrcoef(w_flat, x_flat)[0, 1])
        else:
            r_w = 0.0
            
        if np.std(e_flat) > 1e-7 and np.std(x_flat) > 1e-7:
            r_e = float(np.corrcoef(e_flat, x_flat)[0, 1])
        else:
            r_e = 0.0
            
        w_x_corrs.append(r_w)
        e_x_corrs.append(r_e)
        
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
    
    # Flatten all gate activations across layers: (N_layers, B, T, H) -> 1D
    all_w = np.concatenate([w.flatten() for w in w_gate_vals])
    all_e = np.concatenate([e.flatten() for e in e_gate_vals])
    
    # Compute parameter displacements from initialization
    w_weight_displacements = []
    w_bias_displacements = []
    e_weight_displacements = []
    e_bias_displacements = []
    
    for name, p in model.named_parameters():
        if "write_gate_proj.weight" in name:
            init_w = initial_params[name]
            w_weight_displacements.append(torch.norm(p - init_w).item())
        elif "write_gate_proj.bias" in name:
            init_b = initial_params[name]
            w_bias_displacements.append(torch.norm(p - init_b).item())
        elif "erase_gate_proj.weight" in name:
            init_w = initial_params[name]
            e_weight_displacements.append(torch.norm(p - init_w).item())
        elif "erase_gate_proj.bias" in name:
            init_b = initial_params[name]
            e_bias_displacements.append(torch.norm(p - init_b).item())
            
    # Histogram 10 bins [0.0, 0.1, ..., 1.0]
    w_hist, _ = np.histogram(all_w, bins=10, range=(0.0, 1.0))
    e_hist, _ = np.histogram(all_e, bins=10, range=(0.0, 1.0))
    
    return {
        "memory_paths": {
            "local_buffer_norm": mean_l_norm,
            "recurrent_memory_norm": mean_r_norm,
            "local_contribution_ratio": mean_l_norm / total_norm,
            "recurrent_contribution_ratio": mean_r_norm / total_norm,
            "fusion_gate_mean": float(np.mean(fuse_gates)),
        },
        "write_gate": {
            "mean": float(np.mean(all_w)),
            "std": float(np.std(all_w)),
            "min": float(np.min(all_w)),
            "max": float(np.max(all_w)),
            "fraction_gt_095": float(np.mean(all_w > 0.95)),
            "fraction_lt_005": float(np.mean(all_w < 0.05)),
            "mean_per_sequence_var": float(np.mean(w_seq_vars)),
            "mean_correlation_with_input_norm": float(np.mean(w_x_corrs)),
            "norm_w_weight_displacement": float(np.sqrt(np.sum(np.square(w_weight_displacements)))),
            "norm_w_bias_displacement": float(np.sqrt(np.sum(np.square(w_bias_displacements)))),
            "histogram_10bins": [int(h) for h in w_hist],
        },
        "erase_gate": {
            "mean": float(np.mean(all_e)),
            "std": float(np.std(all_e)),
            "min": float(np.min(all_e)),
            "max": float(np.max(all_e)),
            "fraction_gt_095": float(np.mean(all_e > 0.95)),
            "fraction_lt_005": float(np.mean(all_e < 0.05)),
            "mean_per_sequence_var": float(np.mean(e_seq_vars)),
            "mean_correlation_with_input_norm": float(np.mean(e_x_corrs)),
            "norm_e_weight_displacement": float(np.sqrt(np.sum(np.square(e_weight_displacements)))),
            "norm_e_bias_displacement": float(np.sqrt(np.sum(np.square(e_bias_displacements)))),
            "histogram_10bins": [int(h) for h in e_hist],
        }
    }


def main():
    print("=" * 90)
    print("EXPERIMENT C: B+C+E DECOUPLED GATE LR (10x GATE LR: 5e-4 vs BACKBONE 5e-5)")
    print(f"Device: {DEVICE}")
    print("=" * 90)
    
    enc = tiktoken.get_encoding("gpt2")
    with open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore") as f:
        train_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    with open(os.path.join(JARVIS_ENGINE, "fresh_holdout.txt"), "r", encoding="utf-8", errors="ignore") as f:
        val_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    corpus_tokens = enc.encode(open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore").read(), allowed_special={"<|endoftext|>"})
    
    bce_cfg = {
        "use_local_buffer": True,
        "local_window_size": 16,
        "use_adaptive_decay": False,
        "use_write_gate": True,
        "use_erase_gate": True,
        "use_gated_read": False,
        "fusion_option": 2,
    }
    
    model = build_modular_jarvis(config_dict=bce_cfg, max_seq_len=512).to(DEVICE)
    
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
    print(f"Total Parameters in B+C+E Model: {total_p:,} (+787,584 parameters overhead)")
    
    # Save initial parameter tensors for displacement tracking
    initial_params = {name: p.clone().detach() for name, p in model.named_parameters() if any(k in name for k in ["write_gate_proj", "erase_gate_proj"])}
    
    # Decouple parameter groups: Backbone vs Gates
    backbone_params = []
    gate_params = []
    for name, p in model.named_parameters():
        if any(k in name for k in ["write_gate_proj", "erase_gate_proj", "blend_gate"]):
            gate_params.append(p)
        else:
            backbone_params.append(p)
            
    backbone_lr = 5e-5
    gate_lr_initial = 5e-4
    total_steps = 500
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": gate_params, "lr": gate_lr_initial},
    ], fused=True)
    
    # Cosine decay schedule for gate LR (5e-4 -> 5e-5), constant 5e-5 for backbone
    def backbone_schedule(step):
        return 1.0
        
    def gate_schedule(step):
        min_ratio = 0.1  # 5e-4 * 0.1 = 5e-5
        progress = float(step) / float(max(1, total_steps))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[backbone_schedule, gate_schedule])
    
    batch_size = 2
    grad_accum_steps = 4
    seq_len = 512
    tokens_per_step = batch_size * grad_accum_steps * seq_len
    
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
    
    print(f"\nTraining Plan: {total_steps} steps | Accum: {grad_accum_steps} | Tokens/step: {tokens_per_step:,}")
    print(f"Decoupled LRs: Backbone = {backbone_lr:.1e} (constant) | Gates = {gate_lr_initial:.1e} -> {gate_lr_initial*0.1:.1e} (cosine)")
    print(f"Checkpoints: {eval_checkpoints}")
    print("-" * 90)
    
    step_times = []
    grad_norms = []
    w_gate_grad_norms = []
    e_gate_grad_norms = []
    train_losses = []
    
    for step in range(total_steps + 1):
        if step in eval_checkpoints:
            t_eval_0 = time.perf_counter()
            ce_512, ppl_512 = evaluate_holdout_multicontext(model, val_tokens, seq_len=512, num_windows=20, seed=42 + step)
            ce_1024, ppl_1024 = evaluate_holdout_multicontext(model, val_tokens, seq_len=1024, num_windows=20, seed=42 + step)
            needle_rank = evaluate_needle_rank(model, enc, corpus_tokens, distance=64, num_trials=10)
            diag = collect_path_and_gate_diagnostics(model, dummy_diag, initial_params)
            vram_alloc = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
            vram_res = torch.cuda.max_memory_reserved(DEVICE) / (1024 * 1024)
            free_b, total_b = torch.cuda.mem_get_info(DEVICE)
            vram_driver = (total_b - free_b) / (1024 * 1024)
            eval_dt = time.perf_counter() - t_eval_0
            
            cur_gate_lr = optimizer.param_groups[1]["lr"]
            
            print(f"\n>>> [10x GATE LR CHECKPOINT STEP {step:03d}/{total_steps}] (Gate LR: {cur_gate_lr:.2e} | Eval: {eval_dt:.1f}s)")
            print(f"  T= 512: Holdout CE = {ce_512:.4f} | PPL = {ppl_512:.2f}")
            print(f"  T=1024: Holdout CE = {ce_1024:.4f} | PPL = {ppl_1024:.2f}")
            print(f"  Needle Rank @ 64: {needle_rank:.1f} / 50257")
            print(f"  Write Gate: mean={diag['write_gate']['mean']:.4f} ± {diag['write_gate']['std']:.4f} | >0.95: {diag['write_gate']['fraction_gt_095']*100:.1f}% | ||dW||: {diag['write_gate']['norm_w_weight_displacement']:.4f}")
            print(f"  Erase Gate: mean={diag['erase_gate']['mean']:.4f} ± {diag['erase_gate']['std']:.4f} | <0.05: {diag['erase_gate']['fraction_lt_005']*100:.1f}% | ||dW||: {diag['erase_gate']['norm_e_weight_displacement']:.4f}")
            print(f"  Content Dep: Write SeqVar={diag['write_gate']['mean_per_sequence_var']:.6f}, Corr(x)={diag['write_gate']['mean_correlation_with_input_norm']:.4f} | Erase SeqVar={diag['erase_gate']['mean_per_sequence_var']:.6f}, Corr(x)={diag['erase_gate']['mean_correlation_with_input_norm']:.4f}")
            print(f"  Contributions: Local = {diag['memory_paths']['local_contribution_ratio']*100:.1f}% | Recurrent = {diag['memory_paths']['recurrent_contribution_ratio']*100:.1f}%")
            print(f"  VRAM: Allocated = {vram_alloc:.1f} MB | Reserved = {vram_res:.1f} MB | Driver Used = {vram_driver:.1f} MB / {total_b/(1024*1024):.1f} MB\n")
            
            results_timeline[f"step_{step}"] = {
                "step": step,
                "gate_lr": cur_gate_lr,
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
            
            # Save checkpoint
            ckpt_path = os.path.join(ARCH_DIR, f"ckpt_bce_10xgatelr_step{step:04d}.pt")
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
            
        # Extract gate grad norms before clipping
        w_grads = [p.grad.norm().item() for name, p in model.named_parameters() if "write_gate_proj.weight" in name and p.grad is not None]
        e_grads = [p.grad.norm().item() for name, p in model.named_parameters() if "erase_gate_proj.weight" in name and p.grad is not None]
        w_gnorm = float(np.sqrt(np.sum(np.square(w_grads)))) if w_grads else 0.0
        e_gnorm = float(np.sqrt(np.sum(np.square(e_grads)))) if e_grads else 0.0
        w_gate_grad_norms.append(w_gnorm)
        e_gate_grad_norms.append(e_gnorm)
        
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(DEVICE)
        dt = time.perf_counter() - t_step_0
        
        step_times.append(dt)
        grad_norms.append(gnorm)
        train_losses.append(accum_loss)
        
        if (step + 1) % 25 == 0:
            tok_s = tokens_per_step / max(np.mean(step_times[-25:]), 1e-5)
            print(f"  Step {step + 1:03d}/{total_steps}: Loss = {accum_loss:.4f} | GradNorm = {gnorm:.2f} (W_grad={w_gnorm:.4f}, E_grad={e_gnorm:.4f}) | Speed = {tok_s:5.0f} tok/s")

    print("=" * 90)
    print("10x GATE LR 500-STEP ADAPTATION COMPLETED SUCCESSFULLY")
    print("=" * 90)
    
    out_file = os.path.join(ARCH_DIR, "decoupled_gate_lr_bce_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timeline": results_timeline,
            "training_dynamics": {
                "mean_grad_norm": float(np.mean(grad_norms)),
                "mean_w_gate_grad_norm": float(np.mean(w_gate_grad_norms)),
                "mean_e_gate_grad_norm": float(np.mean(e_gate_grad_norms)),
                "initial_loss": train_losses[0] if train_losses else 0.0,
                "final_loss": train_losses[-1] if train_losses else 0.0,
                "mean_step_time_ms": float(np.mean(step_times)) * 1000.0,
            }
        }, f, indent=2)
    print(f"\n[OK] 10x Gate LR report saved to: {out_file}")

if __name__ == "__main__":
    main()
