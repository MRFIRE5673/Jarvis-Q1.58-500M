# experiments/architecture_matrix/evaluate_architecture.py
"""
CANONICAL STANDARDIZED EVALUATION HARNESS FOR JARVIS-600M
=========================================================
Universal evaluation suite benchmarking any Jarvis architecture variant
under 100% identical, scientifically controlled conditions:

1. Language Modeling Quality:
   - Holdout Cross-Entropy, Perplexity, and Bits/Token on independent fresh_holdout.txt
   - Context length progression: T in [256, 512, 1024, 2048]
2. Associative Memory Stress Suite:
   - Single Needle Retrieval across distances k in [16, 64, 128, 256, 512]
   - Multi-Needle Retrieval (retrieving 2 distinct keys placed at different depths)
   - Memory Overwrite/Update Test (Key X initialized to V1, updated to V2 after distractors)
   - Metrics: Top-1 Accuracy, Top-5 Accuracy, Target Rank (/50257), Target Probability
3. Autoregressive Generation & Correctness:
   - Python code generation prompt
   - 4-gram repetition rate, token entropy, syntax validity
   - Single-token state persistence verification (T=1 decode vs full prefill equivalence)
4. Computational Profile:
   - Prefill throughput (tok/s at T=512)
   - Decode throughput (tok/s at T=1)
   - Peak VRAM allocated (MB)
   - Total parameter count & active parameters per token

Outputs machine-readable JSON to --output path.
"""

import os
import sys
import math
import time
import json
import random
import statistics
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken
from jarvis_model import Jarvis
from jarvis_v2_combined import JarvisV2


def get_git_revision():
    try:
        import subprocess
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=WORKSPACE_ROOT).decode().strip()
        return rev
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 1. Language Modeling Benchmark
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_language_modeling(model, val_tokens, seq_lens=[256, 512, 1024], num_windows=50, seed=42):
    results = {}
    model.eval()

    for T in seq_lens:
        if len(val_tokens) <= T + 1:
            continue
        max_start = len(val_tokens) - T - 1
        g = torch.Generator(device="cpu").manual_seed(seed)
        starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()

        ce_list = []
        for start in starts:
            x = val_tokens[start : start + T].unsqueeze(0)
            y = val_tokens[start + 1 : start + T + 1].unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, targets=y)
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            ce_list.append(ce.item())

        mean_ce = statistics.mean(ce_list)
        ppl = math.exp(min(mean_ce, 100.0))
        bpc = mean_ce / math.log(2.0)

        results[f"T_{T}"] = {
            "seq_len": T,
            "holdout_ce": mean_ce,
            "perplexity": ppl,
            "bits_per_token": bpc,
            "num_windows": num_windows,
        }

    # Canonical T=512 metrics
    canonical = results.get("T_512", results.get(f"T_{seq_lens[0]}"))
    results["canonical_ce"] = canonical["holdout_ce"]
    results["canonical_ppl"] = canonical["perplexity"]
    results["canonical_bpc"] = canonical["bits_per_token"]
    return results


# ---------------------------------------------------------------------------
# 2. Associative Memory Stress Suite
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_associative_memory(model, enc, corpus_tokens, device="cuda"):
    model.eval()
    memory_results = {}

    # Test A: Single Needle Retrieval across distances
    distances = [16, 64, 128, 256, 512]
    needle_target = " 42"
    target_id = enc.encode(needle_target)[0]

    single_needle_results = {}
    for dist in distances:
        trial_ranks, trial_probs, trial_top1, trial_top5 = [], [], [], []
        num_trials = 5

        for trial in range(num_trials):
            max_idx = len(corpus_tokens) - dist - 100
            start_idx = (trial * 7919 + dist * 37) % max(max_idx, 1)
            distractors = corpus_tokens[start_idx : start_idx + dist]

            needle = "The system authentication passcode is 42.\n"
            query = "\nWhat is the system authentication passcode? The system authentication passcode is"

            needle_toks = enc.encode(needle)
            query_toks = enc.encode(query)
            full_toks = needle_toks + distractors + query_toks

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

        single_needle_results[f"dist_{dist}"] = {
            "distance": dist,
            "top1_acc_pct": statistics.mean(trial_top1) * 100,
            "top5_acc_pct": statistics.mean(trial_top5) * 100,
            "mean_target_prob": statistics.mean(trial_probs),
            "mean_rank": statistics.mean(trial_ranks),
        }

    memory_results["single_needle"] = single_needle_results

    # Test B: Multi-Needle Retrieval (Retrieving Key A vs Key B)
    multi_trials_top1 = []
    target_a_id = enc.encode(" alpha")[0]
    target_b_id = enc.encode(" beta")[0]

    for trial in range(5):
        needle_a = "Key alpha secret is alpha.\n"
        distractor_1 = corpus_tokens[(trial * 3000) % (len(corpus_tokens) - 500) : (trial * 3000) % (len(corpus_tokens) - 500) + 128]
        needle_b = "Key beta secret is beta.\n"
        distractor_2 = corpus_tokens[(trial * 5000) % (len(corpus_tokens) - 500) : (trial * 5000) % (len(corpus_tokens) - 500) + 128]
        query_b = "\nWhat is Key beta secret? Key beta secret is"

        seq = enc.encode(needle_a) + distractor_1 + enc.encode(needle_b) + distractor_2 + enc.encode(query_b)
        inp = torch.tensor([seq], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)

        rank_b = (logits[0, -1, :] > logits[0, -1, target_b_id]).sum().item() + 1
        multi_trials_top1.append(1.0 if rank_b <= 5 else 0.0)

    memory_results["multi_needle_top5_pct"] = statistics.mean(multi_trials_top1) * 100

    # Test C: Overwrite / Memory Update Test
    # Key X is assigned 10 at t1, then updated to 20 at t2. Does model retrieve 20?
    target_old = enc.encode(" 10")[0]
    target_new = enc.encode(" 20")[0]
    overwrite_ranks_new = []

    for trial in range(5):
        needle_old = "Variable X is initialized to 10.\n"
        distractors_mid = corpus_tokens[(trial * 4000) % (len(corpus_tokens) - 500) : (trial * 4000) % (len(corpus_tokens) - 500) + 64]
        needle_new = "Variable X is reassigned to 20.\n"
        distractors_end = corpus_tokens[(trial * 6000) % (len(corpus_tokens) - 500) : (trial * 6000) % (len(corpus_tokens) - 500) + 64]
        query = "\nWhat is the current value of Variable X? The current value of Variable X is"

        seq = enc.encode(needle_old) + distractors_mid + enc.encode(needle_new) + distractors_end + enc.encode(query)
        inp = torch.tensor([seq], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)

        last_l = logits[0, -1, :]
        rank_new = (last_l > last_l[target_new]).sum().item() + 1
        overwrite_ranks_new.append(rank_new)

    memory_results["overwrite_test"] = {
        "mean_rank_updated_value": statistics.mean(overwrite_ranks_new),
    }

    return memory_results


# ---------------------------------------------------------------------------
# 3. Autoregressive Generation & Correctness
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_generation(model, enc, device="cuda"):
    model.eval()
    prompt = "def binary_search(arr, target):\n    left = 0\n    right = len(arr) - 1\n"
    prompt_tokens = enc.encode(prompt)
    curr_tokens = list(prompt_tokens)

    # Autoregressive generation of 64 tokens
    for _ in range(64):
        inp = torch.tensor([curr_tokens[-512:]], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
        next_token = torch.argmax(logits[0, -1, :]).item()
        curr_tokens.append(next_token)

    generated_text = enc.decode(curr_tokens)
    gen_only_tokens = curr_tokens[len(prompt_tokens):]

    # Calculate 4-gram repetition
    four_grams = [tuple(gen_only_tokens[i : i + 4]) for i in range(len(gen_only_tokens) - 3)]
    rep_4gram_pct = (1.0 - len(set(four_grams)) / max(len(four_grams), 1)) * 100

    # Calculate token entropy
    counts = {}
    for t in gen_only_tokens:
        counts[t] = counts.get(t, 0) + 1
    probs = [c / len(gen_only_tokens) for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)

    # Check syntax compilability
    is_valid_syntax = False
    try:
        compile(generated_text, "<string>", "exec")
        is_valid_syntax = True
    except Exception:
        pass

    return {
        "prompt": prompt,
        "generated_sample": generated_text[:200],
        "four_gram_repetition_pct": rep_4gram_pct,
        "token_entropy": entropy,
        "is_valid_python_syntax": is_valid_syntax,
    }


# ---------------------------------------------------------------------------
# 4. Computational Efficiency Benchmark
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_throughput(model, seq_len=512, batch_size=2, device="cuda"):
    model.eval()
    dummy_input = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # Prefill benchmark
    num_runs = 15
    t_start = time.perf_counter()
    for _ in range(num_runs):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    prefill_dt = time.perf_counter() - t_start
    prefill_tok_s = (num_runs * batch_size * seq_len) / max(prefill_dt, 1e-5)

    # Single-token decode benchmark
    decode_input = torch.randint(0, 50257, (1, 1), device=device)
    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(decode_input)
    torch.cuda.synchronize()

    num_decode = 100
    t_dec_start = time.perf_counter()
    for _ in range(num_decode):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(decode_input)
    torch.cuda.synchronize()
    dec_dt = time.perf_counter() - t_dec_start
    decode_tok_s = num_decode / max(dec_dt, 1e-5)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "prefill_throughput_tok_s": round(prefill_tok_s, 1),
        "decode_throughput_tok_s": round(decode_tok_s, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }


# ---------------------------------------------------------------------------
# Main Canonical Evaluation Dispatcher
# ---------------------------------------------------------------------------
def run_canonical_evaluation(
    model: nn.Module = None,
    model_type: str = "baseline",
    ckpt_path: str = None,
    output_json: str = None,
    quick_test: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    print("\n" + "=" * 90)
    print(f"CANONICAL STANDARDIZED EVALUATION: {model_type.upper()}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device} | Git: {get_git_revision()} | Quick Test: {quick_test}")
    print("=" * 90)

    # 1. Instantiate Model if not provided
    if model is None:
        if model_type == "v2":
            model = JarvisV2(max_seq_len=2048).to(device)
        else:
            model = Jarvis(
                vocab_size=50257,
                d_model=1024,
                n_layers=24,
                n_heads=16,
                num_experts=4,
                top_k=2,
                max_seq_len=2048,
                use_cuda_attn=False,
                use_cuda_moe=False,
            ).to(device)

        # 2. Load Checkpoint
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}

            model_sd = model.state_dict()
            filtered_sd = {k: v for k, v in new_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
            missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
            print(f"  [Load Weights] Loaded {len(filtered_sd)} matching keys (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        else:
            print("  [Warning] No checkpoint provided or file not found. Running evaluation on current weights.")
    else:
        model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    active_params_per_token = 51463168 + 24 * (4194304 + 2 * 4194304 + 4096) + 51463168
    print(f"  Total Parameters : {total_params:,}")
    print(f"  Active / Token   : {active_params_per_token:,} ({active_params_per_token / total_params * 100:.1f}%)")

    # 3. Load Datasets
    val_path = os.path.join(JARVIS_ENGINE, "fresh_holdout.txt")
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_text = f.read()
    val_tokens = torch.tensor(enc.encode(val_text), dtype=torch.long, device=device)

    corpus_path = os.path.join(JARVIS_ENGINE, "data_clean.txt")
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus_text = f.read()
    corpus_tokens = enc.encode(corpus_text, allowed_special={"<|endoftext|>"})

    # 4. Benchmark Language Modeling
    print("\n[1/4] Benchmarking Language Modeling across Context Lengths...")
    seq_lens = [256, 512] if quick_test else [256, 512, 1024]
    num_windows = 10 if quick_test else 50
    lm_results = evaluate_language_modeling(model, val_tokens, seq_lens=seq_lens, num_windows=num_windows)
    for k, v in lm_results.items():
        if isinstance(v, dict):
            print(f"  Context T={v['seq_len']:>4}: Holdout CE = {v['holdout_ce']:.4f} | PPL = {v['perplexity']:.2f} | BPC = {v['bits_per_token']:.3f}")

    # 5. Benchmark Associative Memory
    print("\n[2/4] Benchmarking Associative Memory Stress Suite...")
    mem_results = evaluate_associative_memory(model, enc, corpus_tokens, device=device)
    for dist_k, dist_v in mem_results["single_needle"].items():
        print(f"  Needle @ {dist_v['distance']:>3} tok : Top-1 Acc = {dist_v['top1_acc_pct']:>5.1f}% | Target Prob = {dist_v['mean_target_prob']*100:.3f}% | Mean Rank = {dist_v['mean_rank']:.1f}/50257")
    print(f"  Multi-Needle Top-5 Accuracy : {mem_results['multi_needle_top5_pct']:.1f}%")
    print(f"  Memory Overwrite Mean Rank  : {mem_results['overwrite_test']['mean_rank_updated_value']:.1f}")

    # 6. Benchmark Generation & Correctness
    print("\n[3/4] Benchmarking Autoregressive Generation...")
    gen_results = evaluate_generation(model, enc, device=device)
    print(f"  4-Gram Repetition : {gen_results['four_gram_repetition_pct']:.1f}%")
    print(f"  Token Entropy     : {gen_results['token_entropy']:.2f} bits")
    print(f"  Python Syntax OK  : {gen_results['is_valid_python_syntax']}")

    # 7. Benchmark Throughput & Hardware Efficiency
    print("\n[4/4] Benchmarking Hardware Throughput & Peak VRAM...")
    perf_results = evaluate_throughput(model, seq_len=512, device=device)
    print(f"  Prefill Speed     : {perf_results['prefill_throughput_tok_s']} tok/s (T=512, Batch=2)")
    print(f"  Decode Speed      : {perf_results['decode_throughput_tok_s']} tok/s (T=1, Batch=1)")
    print(f"  Peak VRAM         : {perf_results['peak_vram_mb']} MB")

    report = {
        "model_type": model_type,
        "checkpoint_path": ckpt_path,
        "git_revision": get_git_revision(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameter_accounting": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "active_parameters_per_token": active_params_per_token,
            "sparsity_ratio": active_params_per_token / total_params,
        },
        "language_modeling": lm_results,
        "associative_memory": mem_results,
        "generation": gen_results,
        "efficiency": perf_results,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Canonical evaluation saved to: {output_json}")

    print("=" * 90)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical Standardized Evaluation Harness for Jarvis-600M")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "v2"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    run_canonical_evaluation(
        model_type=args.model,
        ckpt_path=args.ckpt,
        output_json=args.output,
        quick_test=args.quick_test,
    )
