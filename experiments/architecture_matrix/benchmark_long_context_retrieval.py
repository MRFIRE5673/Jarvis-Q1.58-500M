# experiments/architecture_matrix/benchmark_long_context_retrieval.py
"""
EXTENDED LONG-CONTEXT & MEMORY INTERFERENCE STRESS SUITE
=========================================================
Benchmarks useful associative memory retention, multi-fact interference,
and overwrite dynamics across context depths:
T in [64, 128, 256, 512, 1024, 2048, 4096]
"""

import os
import sys
import math
import json
import statistics
import argparse
import torch
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken


@torch.inference_mode()
def run_long_context_stress_suite(model, enc, corpus_tokens, max_dist=4096, device="cuda"):
    model.eval()
    results = {}

    distances = [64, 128, 256, 512, 1024, 2048, 4096]
    distances = [d for d in distances if d <= max_dist]

    # Target definition
    needle_target = " 42"
    target_id = enc.encode(needle_target)[0]

    # 1. Single Needle Retention Curve
    single_retention = {}
    print("\n[1/3] Benchmarking Single Needle Retention Curve across Distances...")
    for dist in distances:
        trial_ranks, trial_probs, trial_top1, trial_top5 = [], [], [], []
        num_trials = 5

        for trial in range(num_trials):
            start_idx = (trial * 7919 + dist * 37) % max(len(corpus_tokens) - dist - 150, 1)
            distractors = corpus_tokens[start_idx : start_idx + dist]

            needle = "The system authentication passcode is 42.\n"
            query = "\nWhat is the system authentication passcode? The system authentication passcode is"

            full_toks = enc.encode(needle) + distractors + enc.encode(query)
            inp = torch.tensor([full_toks], dtype=torch.long, device=device)

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

        mean_rank = statistics.mean(trial_ranks)
        top1_pct = statistics.mean(trial_top1) * 100
        top5_pct = statistics.mean(trial_top5) * 100
        mean_p = statistics.mean(trial_probs)

        single_retention[f"dist_{dist}"] = {
            "distance": dist,
            "top1_acc_pct": top1_pct,
            "top5_acc_pct": top5_pct,
            "mean_target_prob": mean_p,
            "mean_rank": mean_rank,
        }
        print(f"  Distance {dist:4d} tokens: Mean Rank = {mean_rank:6.1f} | Top-5 Acc = {top5_pct:5.1f}% | Target P = {mean_p:.4f}")

    results["single_needle_retention_curve"] = single_retention

    # 2. Multi-Fact Interference Stress Test
    # 4 distinct facts inserted at spacing intervals
    print("\n[2/3] Benchmarking Multi-Fact Memory Interference...")
    keys = ["alpha", "beta", "gamma", "delta"]
    values = [" 101", " 202", " 303", " 404"]
    multi_fact_results = {}

    for trial in range(3):
        seq = []
        fact_tokens = []
        target_ids = []
        for k, v in zip(keys, values):
            fact_str = f"Record {k} identifier is{v}.\n"
            target_ids.append(enc.encode(v)[0])
            fact_tokens.append(enc.encode(fact_str))

        # Build sequence with intervening distractors
        for i in range(len(keys)):
            seq.extend(fact_tokens[i])
            # 64 tokens distractor between facts
            d_start = (trial * 4000 + i * 500) % (len(corpus_tokens) - 100)
            seq.extend(corpus_tokens[d_start : d_start + 64])

        # Query each key sequentially
        for i, (k, v) in enumerate(zip(keys, values)):
            q_str = f"\nWhat is Record {k} identifier? Record {k} identifier is"
            full_seq = seq + enc.encode(q_str)
            inp = torch.tensor([full_seq], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(inp)
            last_logits = logits[0, -1, :]
            tid = target_ids[i]
            r = (last_logits > last_logits[tid]).sum().item() + 1
            if k not in multi_fact_results:
                multi_fact_results[k] = []
            multi_fact_results[k].append(r)

    results["multi_fact_interference"] = {
        k: {"mean_rank": statistics.mean(ranks)} for k, ranks in multi_fact_results.items()
    }
    for k, v in results["multi_fact_interference"].items():
        print(f"  Key '{k}': Mean Target Rank = {v['mean_rank']:.1f}")

    # 3. Overwrite & Sequential Erase Fidelity Test
    # Variable X set to 10, overwritten to 20, overwritten to 30
    print("\n[3/3] Benchmarking Overwrite & Selective Erasure Fidelity...")
    target_v1 = enc.encode(" 10")[0]
    target_v2 = enc.encode(" 20")[0]
    target_v3 = enc.encode(" 30")[0]
    ranks_v3, ranks_v1 = [], []

    for trial in range(5):
        s1 = enc.encode("Variable X is initialized to 10.\n")
        d1 = corpus_tokens[(trial * 3000) % (len(corpus_tokens) - 200) : (trial * 3000) % (len(corpus_tokens) - 200) + 64]
        s2 = enc.encode("Variable X is updated to 20.\n")
        d2 = corpus_tokens[(trial * 5000) % (len(corpus_tokens) - 200) : (trial * 5000) % (len(corpus_tokens) - 200) + 64]
        s3 = enc.encode("Variable X is finalized to 30.\n")
        d3 = corpus_tokens[(trial * 7000) % (len(corpus_tokens) - 200) : (trial * 7000) % (len(corpus_tokens) - 200) + 64]
        q = enc.encode("\nWhat is the current final value of Variable X? The current final value of Variable X is")

        seq = s1 + d1 + s2 + d2 + s3 + d3 + q
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
        last_l = logits[0, -1, :]
        r3 = (last_l > last_l[target_v3]).sum().item() + 1
        r1 = (last_l > last_l[target_v1]).sum().item() + 1
        ranks_v3.append(r3)
        ranks_v1.append(r1)

    results["sequential_overwrite_fidelity"] = {
        "mean_rank_final_value_30": statistics.mean(ranks_v3),
        "mean_rank_stale_value_10": statistics.mean(ranks_v1),
        "erase_preference_ratio": statistics.mean(ranks_v1) / max(statistics.mean(ranks_v3), 1.0),
    }
    print(f"  Final Value (30) Mean Rank: {results['sequential_overwrite_fidelity']['mean_rank_final_value_30']:.1f}")
    print(f"  Stale Value (10) Mean Rank: {results['sequential_overwrite_fidelity']['mean_rank_stale_value_10']:.1f}")
    print(f"  Selective Erase Preference: {results['sequential_overwrite_fidelity']['erase_preference_ratio']:.2f}x (Higher = Better Overwrite)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="baseline")
    parser.add_argument("--max_dist", type=int, default=4096)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    with open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore") as f:
        corpus = enc.encode(f.read(), allowed_special={"<|endoftext|>"})

    from jarvis_research_controller import build_candidate_model
    # Load model
    model = build_candidate_model(args.model_type)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    model.cuda()

    results = run_long_context_stress_suite(model, enc, corpus, max_dist=args.max_dist)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved long-context results to: {args.output}")


if __name__ == "__main__":
    main()
