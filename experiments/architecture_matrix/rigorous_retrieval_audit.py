# experiments/architecture_matrix/rigorous_retrieval_audit.py
"""
RIGOROUS ASSOCIATIVE RETRIEVAL AUDIT
====================================
Audits needle-in-a-haystack associative memory retention across:
1. Paper Baseline (ckpt_step_0004284_best.pt)
2. Candidate B+C+E Seed 42 (ckpt_fact_B+C+E.pt)
3. Candidate B+C+E Seed 123 (ckpt_fact_B+C+E_s123.pt)
4. Candidate B+C+E Seed 456 (ckpt_fact_B+C+E_s456.pt)

Uses ONLY the canonical prompt from evaluate_architecture.py:
  Needle: "The system authentication passcode is 42.\n"
  Query:  "\nWhat is the system authentication passcode? The system authentication passcode is"
Target: " 42"

Evaluates N=30 distinct distractor segments across multiple distances (dist=16, 64, 128):
Measures:
- Mean Rank
- Standard Deviation
- Min (Best) Rank
- Max (Worst) Rank
- Median Rank
- Top-1 Accuracy (%)
- Top-5 Accuracy (%)
- Welch's Two-Sample t-test vs Baseline (using pure numpy/math)
"""

import os
import sys
import json
import math
import statistics
import torch
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
def evaluate_model_retrieval(model, enc, corpus_tokens, name="Model", num_trials=30, distances=[16, 64, 128]):
    model.eval().to(DEVICE)
    needle_target = " 42"
    target_id = enc.encode(needle_target)[0]
    
    needle = "The system authentication passcode is 42.\n"
    query = "\nWhat is the system authentication passcode? The system authentication passcode is"
    needle_toks = enc.encode(needle)
    query_toks = enc.encode(query)
    
    results_by_dist = {}
    
    print(f"\n--- Evaluating Retrieval for: {name} ({num_trials} trials/distance) ---")
    for dist in distances:
        trial_ranks = []
        trial_probs = []
        trial_top1 = []
        trial_top5 = []
        
        max_idx = len(corpus_tokens) - dist - 100
        for trial in range(num_trials):
            start_idx = (trial * 13337 + dist * 97) % max(max_idx, 1)
            distractors = corpus_tokens[start_idx : start_idx + dist]
            
            full_toks = needle_toks + distractors + query_toks
            inp = torch.tensor([full_toks], dtype=torch.long, device=DEVICE)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(inp)
                
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            p_target = probs[target_id].item()
            rank = (last_logits > last_logits[target_id]).sum().item() + 1
            
            trial_ranks.append(rank)
            trial_probs.append(p_target)
            trial_top1.append(1.0 if rank == 1 else 0.0)
            trial_top5.append(1.0 if rank <= 5 else 0.0)
            
        m_rank = float(np.mean(trial_ranks))
        s_rank = float(np.std(trial_ranks))
        med_rank = float(np.median(trial_ranks))
        min_rank = int(np.min(trial_ranks))
        max_rank = int(np.max(trial_ranks))
        top1_pct = float(np.mean(trial_top1) * 100.0)
        top5_pct = float(np.mean(trial_top5) * 100.0)
        mean_p = float(np.mean(trial_probs))
        
        print(f"  Dist {dist:>3} tok: Mean Rank = {m_rank:6.1f} +/- {s_rank:5.1f} | Med = {med_rank:6.1f} | [Min: {min_rank:>5}, Max: {max_rank:>5}] | Top-1: {top1_pct:4.1f}% | Top-5: {top5_pct:4.1f}%")
        
        results_by_dist[dist] = {
            "distance": dist,
            "mean_rank": m_rank,
            "std_rank": s_rank,
            "median_rank": med_rank,
            "min_rank": min_rank,
            "max_rank": max_rank,
            "top1_acc_pct": top1_pct,
            "top5_acc_pct": top5_pct,
            "mean_target_prob": mean_p,
            "raw_ranks": trial_ranks,
        }
        
    return results_by_dist


def welch_t_test(x1, x2):
    n1, n2 = len(x1), len(x2)
    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / max(se, 1e-12)
    
    # Satterthwaite degrees of freedom
    df = (v1 / n1 + v2 / n2)**2 / max(( (v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1) ), 1e-12)
    
    # Two-tailed p-value using normal approximation for large df
    # erfc(|t| / sqrt(2))
    p_val = math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(df), float(p_val)


def main():
    print("=" * 85)
    print("RIGOROUS ASSOCIATIVE RETRIEVAL STATISTICAL AUDIT")
    print(f"Device: {DEVICE}")
    print("=" * 85)
    
    enc = tiktoken.get_encoding("gpt2")
    corpus_path = os.path.join(JARVIS_ENGINE, "data_clean.txt")
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus_tokens = enc.encode(f.read(), allowed_special={"<|endoftext|>"})
        
    num_trials = 30
    distances = [16, 64, 128]
    
    # 1. Baseline
    print("\nLoading Paper Baseline...")
    base_model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=2048,
        use_cuda_attn=False,
        use_cuda_moe=False,
    )
    base_ckpt = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
    ckpt = torch.load(base_ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    base_model.load_state_dict(new_sd, strict=True)
    base_res = evaluate_model_retrieval(base_model, enc, corpus_tokens, name="Paper Baseline", num_trials=num_trials, distances=distances)
    del base_model
    torch.cuda.empty_cache()
    
    # 2. B+C+E Seeds
    bce_cfg = {
        "use_local_buffer": True,
        "local_window_size": 16,
        "use_adaptive_decay": False,
        "use_write_gate": True,
        "use_erase_gate": True,
        "use_gated_read": False,
        "fusion_option": 2,
    }
    
    seed_ckpts = [
        ("B+C+E (Seed 42)", os.path.join(ARCH_DIR, "ckpt_fact_B+C+E.pt")),
        ("B+C+E (Seed 123)", os.path.join(ARCH_DIR, "ckpt_fact_B+C+E_s123.pt")),
        ("B+C+E (Seed 456)", os.path.join(ARCH_DIR, "ckpt_fact_B+C+E_s456.pt")),
    ]
    
    bce_results = {}
    for name, cpath in seed_ckpts:
        if not os.path.exists(cpath):
            print(f"Skipping {name}, checkpoint not found: {cpath}")
            continue
        print(f"\nLoading {name} from {os.path.basename(cpath)}...")
        model = build_modular_jarvis(config_dict=bce_cfg, max_seq_len=512)
        ckpt_m = torch.load(cpath, map_location="cpu")
        sd_m = ckpt_m["model_state_dict"] if "model_state_dict" in ckpt_m else ckpt_m
        new_sd_m = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd_m.items()}
        model.load_state_dict(new_sd_m, strict=True)
        res = evaluate_model_retrieval(model, enc, corpus_tokens, name=name, num_trials=num_trials, distances=distances)
        bce_results[name] = res
        del model
        torch.cuda.empty_cache()
        
    print("\n" + "=" * 85)
    print("STATISTICAL COMPARISON TABLE AT DISTANCE = 64 TOKENS")
    print("=" * 85)
    print(f"{'Architecture / Seed':<25} | {'N':<4} | {'Mean Rank':<16} | {'Median':<8} | {'Min':<6} | {'Max':<6} | {'Top-1 %':<8}")
    print("-" * 85)
    
    b64 = base_res[64]
    print(f"{'Paper Baseline':<25} | {num_trials:<4} | {b64['mean_rank']:6.1f} +/- {b64['std_rank']:5.1f} | {b64['median_rank']:6.1f} | {b64['min_rank']:<6} | {b64['max_rank']:<6} | {b64['top1_acc_pct']:4.1f}%")
    
    all_bce_64_ranks = []
    for name, res in bce_results.items():
        r64 = res[64]
        all_bce_64_ranks.extend(r64['raw_ranks'])
        print(f"{name:<25} | {num_trials:<4} | {r64['mean_rank']:6.1f} +/- {r64['std_rank']:5.1f} | {r64['median_rank']:6.1f} | {r64['min_rank']:<6} | {r64['max_rank']:<6} | {r64['top1_acc_pct']:4.1f}%")
        
    agg_mean = float(np.mean(all_bce_64_ranks))
    agg_std = float(np.std(all_bce_64_ranks))
    agg_med = float(np.median(all_bce_64_ranks))
    agg_min = int(np.min(all_bce_64_ranks))
    agg_max = int(np.max(all_bce_64_ranks))
    print("-" * 85)
    print(f"{'B+C+E Combined (N=90)':<25} | {len(all_bce_64_ranks):<4} | {agg_mean:6.1f} +/- {agg_std:5.1f} | {agg_med:6.1f} | {agg_min:<6} | {agg_max:<6} | 0.0%")
    
    # Statistical Hypothesis Testing
    t_stat, df_val, p_val = welch_t_test(b64['raw_ranks'], all_bce_64_ranks)
    print("\n--- Statistical Hypothesis Testing (Baseline vs B+C+E @ 64 tok) ---")
    print(f"Welch's Two-Sample t-test : t = {t_stat:.3f}, df = {df_val:.1f}, p-value = {p_val:.4f}")
    if p_val > 0.05:
        verdict = "EXPERIMENTALLY INDISTINGUISHABLE (No statistically significant difference, p > 0.05)"
    elif agg_mean < b64['mean_rank']:
        verdict = "CANDIDATE B+C+E STATISTICALLY IMPROVES RETRIEVAL (p < 0.05)"
    else:
        verdict = "CANDIDATE B+C+E EXHIBITS RETRIEVAL REGRESSION (p < 0.05)"
    print(f"Verdict: {verdict}")
    print("=" * 85)
    
    out_file = os.path.join(ARCH_DIR, "rigorous_retrieval_audit_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {k: {kk: vv for kk, vv in v.items() if kk != "raw_ranks"} for k, v in base_res.items()},
            "bce_seeds": {k: {kk: {kkk: vvv for kkk, vvv in vv.items() if kkk != "raw_ranks"} for kk, vv in v.items()} for k, v in bce_results.items()},
            "statistical_test": {
                "t_statistic": t_stat,
                "degrees_of_freedom": df_val,
                "p_value": p_val,
                "verdict": verdict,
            }
        }, f, indent=2)
    print(f"\n[OK] Retrieval audit report saved to: {out_file}")

if __name__ == "__main__":
    main()
