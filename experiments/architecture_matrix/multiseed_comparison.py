# experiments/architecture_matrix/multiseed_comparison.py
"""
EXPERIMENT G: MULTI-SEED CONFIRMATION (Seeds: 42, 123, 456)
============================================================
Tests whether E-only and B+C+E exhibit statistically distinguishable
performance under independent initialization / data sampling across 3 random seeds:
- Seed 42
- Seed 123
- Seed 456

Runs 100-step controlled adaptation for both candidates and evaluates:
- T=512 Holdout CE & PPL
- T=1024 Holdout CE & PPL
- Associative Needle Retrieval Rank @ 64 tokens
- Performs Welch's two-sample t-test to determine if B+C adds statistically significant value over E.
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

def welch_t_test(x1, x2):
    n1, n2 = len(x1), len(x2)
    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / max(se, 1e-12)
    
    # Satterthwaite degrees of freedom
    df = (v1 / n1 + v2 / n2)**2 / max(( (v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1) ), 1e-12)
    
    # Two-tailed p-value using normal approximation for large df
    p_val = math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(df), float(p_val)

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
def evaluate_multicontext(model, val_tokens, seq_len=512, num_windows=20, seed=42):
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
def evaluate_needle_rank(model, enc, corpus_tokens, distance=64, num_trials=10, seed=42):
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
        start_idx = (seed * 997 + trial * 13337 + distance * 97) % max(max_idx, 1)
        distractors = corpus_tokens[start_idx : start_idx + distance]
        full_toks = needle_toks + distractors + query_toks
        inp = torch.tensor([full_toks], dtype=torch.long, device=DEVICE)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
        last_logits = logits[0, -1, :]
        rank = (last_logits > last_logits[target_id]).sum().item() + 1
        ranks.append(rank)
        
    return float(np.mean(ranks))

def train_and_eval_variant(candidate_name, config_dict, seed, train_tokens, val_tokens, corpus_tokens, enc, steps=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = build_modular_jarvis(config_dict=config_dict, max_seq_len=512).to(DEVICE)
    
    base_ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model_sd = model.state_dict()
    filtered_sd = {k: v for k, v in new_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered_sd, strict=False)
    
    for block in model.blocks:
        if hasattr(block.attn, "init_neutral"):
            block.attn.init_neutral()
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, fused=True)
    batch_size = 2
    grad_accum_steps = 4
    seq_len = 512
    
    _offsets = torch.arange(seq_len, device=DEVICE)
    g = torch.Generator(device="cpu").manual_seed(seed)
    
    def get_batch():
        starts = torch.randint(0, len(train_tokens) - seq_len - 1, (batch_size,), generator=g).to(DEVICE)
        idx = starts.unsqueeze(1) + _offsets.unsqueeze(0)
        x = train_tokens[idx]
        y = train_tokens[idx + 1]
        return x, y

    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    ce_512, ppl_512 = evaluate_multicontext(model, val_tokens, seq_len=512, num_windows=20, seed=seed)
    ce_1024, ppl_1024 = evaluate_multicontext(model, val_tokens, seq_len=1024, num_windows=20, seed=seed)
    needle_rank = evaluate_needle_rank(model, enc, corpus_tokens, distance=64, num_trials=10, seed=seed)
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return {
        "candidate": candidate_name,
        "seed": seed,
        "ce_512": ce_512,
        "ppl_512": ppl_512,
        "ce_1024": ce_1024,
        "ppl_1024": ppl_1024,
        "needle_rank_64": needle_rank,
    }

def main():
    print("=" * 85)
    print("EXPERIMENT G: MULTI-SEED CONFIRMATION (Seeds 42, 123, 456)")
    print("=" * 85)
    
    enc = tiktoken.get_encoding("gpt2")
    with open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore") as f:
        train_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    with open(os.path.join(JARVIS_ENGINE, "fresh_holdout.txt"), "r", encoding="utf-8", errors="ignore") as f:
        val_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    corpus_tokens = enc.encode(open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore").read(), allowed_special={"<|endoftext|>"})

    seeds = [42, 123, 456]
    
    e_cfg = {
        "use_local_buffer": True,
        "local_window_size": 16,
        "use_adaptive_decay": False,
        "use_write_gate": False,
        "use_erase_gate": False,
        "use_gated_read": False,
        "fusion_option": 2,
    }
    
    bce_cfg = {
        "use_local_buffer": True,
        "local_window_size": 16,
        "use_adaptive_decay": False,
        "use_write_gate": True,
        "use_erase_gate": True,
        "use_gated_read": False,
        "fusion_option": 2,
    }
    
    e_results = []
    bce_results = []
    
    for s in seeds:
        print(f"\n--- Running Seed {s} for E-only ---")
        res_e = train_and_eval_variant("E-only", e_cfg, s, train_tokens, val_tokens, corpus_tokens, enc)
        print(f"  E-only [Seed {s}]: CE@512={res_e['ce_512']:.4f} | CE@1024={res_e['ce_1024']:.4f} | NeedleRank={res_e['needle_rank_64']:.1f}")
        e_results.append(res_e)
        
        print(f"\n--- Running Seed {s} for B+C+E ---")
        res_bce = train_and_eval_variant("B+C+E", bce_cfg, s, train_tokens, val_tokens, corpus_tokens, enc)
        print(f"  B+C+E  [Seed {s}]: CE@512={res_bce['ce_512']:.4f} | CE@1024={res_bce['ce_1024']:.4f} | NeedleRank={res_bce['needle_rank_64']:.1f}")
        bce_results.append(res_bce)
        
    e_ce_512 = [r["ce_512"] for r in e_results]
    bce_ce_512 = [r["ce_512"] for r in bce_results]
    t_stat_512, df_512, p_val_512 = welch_t_test(e_ce_512, bce_ce_512)
    
    e_ce_1024 = [r["ce_1024"] for r in e_results]
    bce_ce_1024 = [r["ce_1024"] for r in bce_results]
    t_stat_1024, df_1024, p_val_1024 = welch_t_test(e_ce_1024, bce_ce_1024)
    
    e_needle = [r["needle_rank_64"] for r in e_results]
    bce_needle = [r["needle_rank_64"] for r in bce_results]
    t_stat_needle, df_needle, p_val_needle = welch_t_test(e_needle, bce_needle)
    
    summary = {
        "seeds": seeds,
        "e_only": {
            "ce_512_mean": float(np.mean(e_ce_512)),
            "ce_512_std": float(np.std(e_ce_512)),
            "ce_1024_mean": float(np.mean(e_ce_1024)),
            "ce_1024_std": float(np.std(e_ce_1024)),
            "needle_mean": float(np.mean(e_needle)),
            "needle_std": float(np.std(e_needle)),
            "runs": e_results,
        },
        "bce": {
            "ce_512_mean": float(np.mean(bce_ce_512)),
            "ce_512_std": float(np.std(bce_ce_512)),
            "ce_1024_mean": float(np.mean(bce_ce_1024)),
            "ce_1024_std": float(np.std(bce_ce_1024)),
            "needle_mean": float(np.mean(bce_needle)),
            "needle_std": float(np.std(bce_needle)),
            "runs": bce_results,
        },
        "welch_tests": {
            "ce_512": {"t_stat": float(t_stat_512), "p_val": float(p_val_512)},
            "ce_1024": {"t_stat": float(t_stat_1024), "p_val": float(p_val_1024)},
            "needle_rank": {"t_stat": float(t_stat_needle), "p_val": float(p_val_needle)},
        }
    }
    
    print("\n" + "=" * 85)
    print("MULTI-SEED CONFIRMATION SUMMARY")
    print("=" * 85)
    print(f"E-Only:  CE@512 = {summary['e_only']['ce_512_mean']:.4f} ± {summary['e_only']['ce_512_std']:.4f} | CE@1024 = {summary['e_only']['ce_1024_mean']:.4f} ± {summary['e_only']['ce_1024_std']:.4f} | Needle = {summary['e_only']['needle_mean']:.1f} ± {summary['e_only']['needle_std']:.1f}")
    print(f"B+C+E:   CE@512 = {summary['bce']['ce_512_mean']:.4f} ± {summary['bce']['ce_512_std']:.4f} | CE@1024 = {summary['bce']['ce_1024_mean']:.4f} ± {summary['bce']['ce_1024_std']:.4f} | Needle = {summary['bce']['needle_mean']:.1f} ± {summary['bce']['needle_std']:.1f}")
    print(f"Welch t-tests: CE@512 p = {p_val_512:.4f} | CE@1024 p = {p_val_1024:.4f} | Needle p = {p_val_needle:.4f}")
    
    out_file = os.path.join(ARCH_DIR, "multiseed_confirmation_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Multi-seed confirmation report saved to: {out_file}")

if __name__ == "__main__":
    main()
