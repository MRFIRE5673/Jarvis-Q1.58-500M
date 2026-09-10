# experiments/architecture_matrix/memory_v2/depth_retrieval_benchmark.py
"""
LONG-CONTEXT DEPTH-OF-RETRIEVAL BENCHMARK
=========================================
Evaluates information retention across context depths:
- Early (~10% depth): needle placed at 10% of total sequence length
- Middle (~50% depth): needle placed at 50% of total sequence length
- Late (~90% depth): needle placed at 90% of total sequence length

Tested across context lengths:
T in [512, 1024, 2048, 4096, 8192]

Architectures:
1. Paper Baseline (Locked reference)
2. E-W16 (Uniform sliding-window buffer)
3. Multi-Scale (W in {8, 16, 32} partitioned across heads)

Metrics:
- Needle target rank (1 = perfect, lower is better)
- Top-1 accuracy (%)
- Top-5 accuracy (%)
- Needle logit & probability
- Cross-entropy at depth
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn.functional as F
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")
REPORTS_DIR = os.path.join(ARCH_DIR, "reports")
MEM_V2_DIR = os.path.join(ARCH_DIR, "memory_v2")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR, MEM_V2_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken
from jarvis_model import Jarvis
from multiscale_memory import build_multiscale_jarvis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_baseline_weights(model):
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
    return model


@torch.inference_mode()
def evaluate_needle_at_depth(model, enc, corpus_tokens, context_len: int, depth_pct: float, num_trials: int = 5, seed: int = 42):
    """
    depth_pct: 0.10 (Early), 0.50 (Middle), 0.90 (Late)
    """
    model.eval()
    needle_target = " 42"
    target_id = enc.encode(needle_target)[0]
    
    needle_str = " The system authentication passcode is 42.\n"
    query_str = "\nWhat is the system authentication passcode? The system authentication passcode is"
    
    needle_toks = enc.encode(needle_str)
    query_toks = enc.encode(query_str)
    
    ranks = []
    top1_hits = []
    top5_hits = []
    target_probs = []
    
    total_distractor_len = context_len - len(needle_toks) - len(query_toks)
    if total_distractor_len < 10:
        total_distractor_len = 10
        
    pre_len = int(total_distractor_len * depth_pct)
    post_len = total_distractor_len - pre_len
    
    for trial in range(num_trials):
        start_idx = (seed * 1009 + trial * 1777 + int(depth_pct * 100) * 31) % max(len(corpus_tokens) - context_len - 100, 1)
        
        pre_distractors = corpus_tokens[start_idx : start_idx + pre_len]
        post_distractors = corpus_tokens[start_idx + pre_len : start_idx + pre_len + post_len]
        
        full_tokens = pre_distractors + needle_toks + post_distractors + query_toks
        # Exact length match
        full_tokens = full_tokens[:context_len]
        
        inp = torch.tensor([full_tokens], dtype=torch.long, device=DEVICE)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
            
        last_logits = logits[0, -1, :] # (V,)
        probs = F.softmax(last_logits, dim=-1)
        
        target_prob = probs[target_id].item()
        target_logit = last_logits[target_id].item()
        
        rank = (last_logits > target_logit).sum().item() + 1
        top1 = 1 if rank == 1 else 0
        top5 = 1 if rank <= 5 else 0
        
        ranks.append(rank)
        top1_hits.append(top1)
        top5_hits.append(top5)
        target_probs.append(target_prob)
        
    return {
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "top1_acc_pct": float(np.mean(top1_hits) * 100.0),
        "top5_acc_pct": float(np.mean(top5_hits) * 100.0),
        "mean_target_prob": float(np.mean(target_probs)),
    }


def run_depth_retrieval_study():
    print("=" * 85)
    print("RUNNING DEPTH-OF-RETRIEVAL RETENTION CURVE STUDY")
    print(f"Device: {DEVICE}")
    print("=" * 85)
    
    enc = tiktoken.get_encoding("gpt2")
    corpus_file = os.path.join(JARVIS_ENGINE, "data_clean.txt")
    with open(corpus_file, "r", encoding="utf-8", errors="ignore") as f:
        corpus_tokens = enc.encode(f.read(), allowed_special={"<|endoftext|>"})
        
    candidates = [
        ("Paper Baseline", lambda: Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=8192, use_cuda_attn=False, use_cuda_moe=False)),
        ("E-W16", lambda: build_multiscale_jarvis(window_config=16, max_seq_len=8192)),
        ("Multi-Scale", lambda: build_multiscale_jarvis(window_config=[8]*4 + [16]*6 + [32]*6, max_seq_len=8192)),
    ]
    
    context_lengths = [512, 1024, 2048, 4096, 8192]
    depths = [
        ("Early (~10%)", 0.10),
        ("Middle (~50%)", 0.50),
        ("Late (~90%)", 0.90),
    ]
    
    full_report = {}
    
    for cand_name, builder in candidates:
        print(f"\n>>> Evaluating Architecture: [{cand_name}]")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = builder().to(DEVICE)
        model = load_baseline_weights(model)
        
        cand_results = {}
        for T in context_lengths:
            cand_results[str(T)] = {}
            print(f"  Context T = {T:4d}:")
            for depth_name, depth_pct in depths:
                res = evaluate_needle_at_depth(model, enc, corpus_tokens, context_len=T, depth_pct=depth_pct, num_trials=5)
                cand_results[str(T)][depth_name] = res
                print(f"    {depth_name:14s} | Rank: {res['mean_rank']:6.1f} | Top-1: {res['top1_acc_pct']:4.1f}% | Top-5: {res['top5_acc_pct']:4.1f}% | P(42): {res['mean_target_prob']*100:6.3f}%")
                
        full_report[cand_name] = cand_results
        del model
        torch.cuda.empty_cache()
        
    # Save JSON report
    out_json = os.path.join(REPORTS_DIR, "depth_retrieval_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[OK] Depth retrieval report saved to: {out_json}")
    
    # Save Markdown summary table
    out_md = os.path.join(REPORTS_DIR, "depth_retrieval_table.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Depth-of-Retrieval Retention Curve Analysis\n\n")
        f.write("Evaluates information retention across sequence depths (Early ~10%, Middle ~50%, Late ~90%) and context lengths (T = 512 to 8192).\n\n")
        f.write("| Architecture | Context | Early (~10%) Rank | Middle (~50%) Rank | Late (~90%) Rank | Late P(target) | Retention Ratio (Late/Early) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for cand, res in full_report.items():
            for T in context_lengths:
                t_str = str(T)
                e_rank = res[t_str]["Early (~10%)"]["mean_rank"]
                m_rank = res[t_str]["Middle (~50%)"]["mean_rank"]
                l_rank = res[t_str]["Late (~90%)"]["mean_rank"]
                l_prob = res[t_str]["Late (~90%)"]["mean_target_prob"] * 100.0
                ratio = e_rank / max(l_rank, 1.0)
                f.write(f"| **{cand}** | T={T} | {e_rank:.1f} | {m_rank:.1f} | {l_rank:.1f} | {l_prob:.3f}% | {ratio:.2f}x |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            
    print(f"[OK] Markdown table saved to: {out_md}")
    return full_report


if __name__ == "__main__":
    run_depth_retrieval_study()
