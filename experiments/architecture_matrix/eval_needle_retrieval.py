# experiments/architecture_matrix/eval_needle_retrieval.py
"""
Needle-in-a-Haystack Associative Retrieval Benchmark
====================================================
Evaluates associative attention and memory retention across token distances:
k in [16, 32, 64, 128, 256, 384, 512] tokens.

Measures:
- Top-1 Accuracy: Is target token ranked #1 at retrieval query?
- Target Probability: Softmax probability assigned to target token.
- Mean Rank: Rank of target token across all 50,257 vocabulary tokens.
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
SPARSE_DIR = os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")
ATTN_DIR = os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, SPARSE_DIR, ATTN_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken
from jarvis_model import Jarvis


@torch.inference_mode()
def evaluate_needle_retrieval(model, enc, distances=[16, 32, 64, 128, 256, 384, 512], num_trials=10, device="cuda"):
    print("=" * 80)
    print("      NEEDLE-IN-A-HAYSTACK ASSOCIATIVE RETRIEVAL BENCHMARK")
    print(f"      Distances: {distances} | Trials/Distance: {num_trials}")
    print("=" * 80)

    target_str = " 42"
    target_id = enc.encode(target_str)[0]

    # Load background distractors from data_clean.txt
    corpus_path = os.path.join(JARVIS_ENGINE, "data_clean.txt")
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus_text = f.read()
    corpus_tokens = enc.encode(corpus_text, allowed_special={"<|endoftext|>"})

    results = {}

    for dist in distances:
        trial_ranks = []
        trial_probs = []
        trial_top1 = []

        for trial in range(num_trials):
            # Select random distractor segment from corpus
            max_idx = len(corpus_tokens) - dist - 100
            start_idx = (trial * 7919 + dist * 31) % max_idx
            distractor_tokens = corpus_tokens[start_idx : start_idx + dist]

            # Construct needle and query
            needle = "The secret access code is 42.\n"
            query = "\nWhat is the secret access code? The secret access code is"

            needle_tokens = enc.encode(needle)
            query_tokens = enc.encode(query)

            full_tokens = needle_tokens + distractor_tokens + query_tokens
            inp = torch.tensor([full_tokens], dtype=torch.long, device=device)

            model.eval()
            model.reset_state()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(inp, persist_state=False)

            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)

            p_target = probs[target_id].item()
            rank = (last_logits > last_logits[target_id]).sum().item() + 1
            is_top1 = 1.0 if rank == 1 else 0.0

            trial_ranks.append(rank)
            trial_probs.append(p_target)
            trial_top1.append(is_top1)

        mean_rank = statistics.mean(trial_ranks)
        mean_prob = statistics.mean(trial_probs)
        top1_acc = statistics.mean(trial_top1) * 100

        print(f"  Distance {dist:>3} tokens: Top-1 Acc = {top1_acc:>5.1f}% | Target Prob = {mean_prob*100:>5.2f}% | Mean Rank = {mean_rank:>6.1f} / 50257")

        results[f"dist_{dist}"] = {
            "distance_tokens": dist,
            "top1_acc_pct": top1_acc,
            "mean_target_prob": mean_prob,
            "mean_rank": mean_rank,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=os.path.join(JARVIS_ENGINE, "ckpt_step_0004209.pt"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=1024,
        use_cuda_attn=True,
        use_cuda_moe=True,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"]
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)

    results = evaluate_needle_retrieval(model, enc, distances=[16, 32, 64, 128, 256, 512])

    out_file = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix", "needle_retrieval_baseline.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")


if __name__ == "__main__":
    main()
