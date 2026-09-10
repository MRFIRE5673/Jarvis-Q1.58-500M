# experiments/architecture_matrix/memory_v2/long_context_benchmark.py
"""
LONG-CONTEXT MEMORY BENCHMARK (T = 512, 1024, 2048, 4096, 8192)
================================================================
Comprehensively evaluates memory architectures across extended sequence lengths:
T in [512, 1024, 2048, 4096, 8192]

Architectures Tested:
1. Paper Baseline (frozen baseline reference)
2. E-W8  (Uniform W=8 sliding window)
3. E-W16 (Uniform W=16 sliding window)
4. E-W32 (Uniform W=32 sliding window)
5. Multi-Scale ([8]*4 + [16]*6 + [32]*6 across 16 heads)
6. State-Compacted (Grouped Recurrent Memory, 4 KV groups, 75% state reduction)

Metrics measured per architecture and context depth:
- Cross-Entropy (CE) and Perplexity (PPL)
- Associative Needle Retrieval Rank @ 64 tokens
- Recurrent State Memory Footprint (KB / sequence)
- Forward Step Latency (ms)
- Peak Allocated and Reserved VRAM (MB)
- Driver-Visible GPU Memory (MB)
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
from state_compaction import GroupedRecurrentMemoryAttention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def evaluate_context_length(model, val_tokens, seq_len: int, num_windows: int = 5, seed: int = 42):
    """Evaluates cross-entropy and perplexity at a specified sequence length."""
    model.eval()
    max_start = len(val_tokens) - seq_len - 1
    if max_start <= 0:
        return float("nan"), float("nan"), 0.0
        
    g = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()
    
    losses = []
    t_start = time.perf_counter()
    
    for s in starts:
        x = val_tokens[s : s + seq_len].unsqueeze(0).to(DEVICE)
        y = val_tokens[s + 1 : s + seq_len + 1].unsqueeze(0).to(DEVICE)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(ce.item())
        
    duration = time.perf_counter() - t_start
    mean_lat_ms = (duration / num_windows) * 1000.0
    mean_ce = float(np.mean(losses))
    ppl = float(math.exp(min(mean_ce, 50.0)))
    return mean_ce, ppl, mean_lat_ms


@torch.inference_mode()
def evaluate_needle_at_context(model, enc, corpus_tokens, context_len: int = 512, distance: int = 64, num_trials: int = 5):
    """Evaluates canonical associative needle retrieval embedded inside context_len."""
    model.eval()
    needle_target = " 42"
    target_id = enc.encode(needle_target)[0]
    
    needle = "The system authentication passcode is 42.\n"
    query = "\nWhat is the system authentication passcode? The system authentication passcode is"
    needle_toks = enc.encode(needle)
    query_toks = enc.encode(query)
    
    ranks = []
    for trial in range(num_trials):
        pad_needed = context_len - len(needle_toks) - distance - len(query_toks)
        pad_needed = max(pad_needed, 0)
        
        start_idx = (trial * 13337 + distance * 97) % max(len(corpus_tokens) - distance - pad_needed - 100, 1)
        prefix_padding = corpus_tokens[start_idx : start_idx + pad_needed]
        distractors = corpus_tokens[start_idx + pad_needed : start_idx + pad_needed + distance]
        
        full_toks = prefix_padding + needle_toks + distractors + query_toks
        inp = torch.tensor([full_toks], dtype=torch.long, device=DEVICE)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
            
        last_logits = logits[0, -1, :]
        rank = (last_logits > last_logits[target_id]).sum().item() + 1
        ranks.append(rank)
        
    return float(np.mean(ranks))


def load_baseline_checkpoint_weights(model):
    """Loads weights from locked paper baseline checkpoint."""
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


def build_candidate_model(name: str):
    """Factory for candidate models."""
    if name == "Paper Baseline":
        model = Jarvis(
            vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
            num_experts=4, top_k=2, max_seq_len=8192,
            use_cuda_attn=False, use_cuda_moe=False
        )
        base_ckpt = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
        ckpt = torch.load(base_ckpt, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=True)
        state_kb = (24 * 16 * 64 * 64 * 4) / 1024.0 # 6,144 KB
        return model.to(DEVICE), state_kb
        
    elif name == "E-W8":
        model = build_multiscale_jarvis(window_config=8, max_seq_len=8192).to(DEVICE)
        model = load_baseline_checkpoint_weights(model)
        state_kb = (24 * 16 * 64 * 64 * 4) / 1024.0
        return model, state_kb
        
    elif name == "E-W16":
        model = build_multiscale_jarvis(window_config=16, max_seq_len=8192).to(DEVICE)
        model = load_baseline_checkpoint_weights(model)
        state_kb = (24 * 16 * 64 * 64 * 4) / 1024.0
        return model, state_kb
        
    elif name == "E-W32":
        model = build_multiscale_jarvis(window_config=32, max_seq_len=8192).to(DEVICE)
        model = load_baseline_checkpoint_weights(model)
        state_kb = (24 * 16 * 64 * 64 * 4) / 1024.0
        return model, state_kb
        
    elif name == "Multi-Scale":
        # Multi-scale windows: [8]*4 + [16]*6 + [32]*6
        ms_cfg = [8]*4 + [16]*6 + [32]*6
        model = build_multiscale_jarvis(window_config=ms_cfg, max_seq_len=8192).to(DEVICE)
        model = load_baseline_checkpoint_weights(model)
        state_kb = (24 * 16 * 64 * 64 * 4) / 1024.0
        return model, state_kb
        
    elif name == "State-Compacted (GRM)":
        model = Jarvis(
            vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
            num_experts=4, top_k=2, max_seq_len=8192,
            use_cuda_attn=False, use_cuda_moe=False
        )
        base_ckpt = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
        ckpt = torch.load(base_ckpt, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=True)
        for block in model.blocks:
            grm = GroupedRecurrentMemoryAttention(d_model=1024, n_heads=16, n_kv_groups=4, local_window=16, max_seq_len=8192)
            grm.init_from_baseline_attn(block.attn)
            block.attn = grm
        state_kb = (24 * 4 * 64 * 64 * 4) / 1024.0 # 1,536 KB (4x smaller!)
        return model.to(DEVICE), state_kb
    else:
        raise ValueError(f"Unknown architecture: {name}")


def run_long_context_benchmark():
    print("=" * 90)
    print("RUNNING AUTOMATED LONG-CONTEXT MEMORY BENCHMARK (T = 512 to 8192)")
    print(f"Device: {DEVICE}")
    print("=" * 90)
    
    enc = tiktoken.get_encoding("gpt2")
    with open(os.path.join(JARVIS_ENGINE, "fresh_holdout.txt"), "r", encoding="utf-8", errors="ignore") as f:
        val_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    corpus_tokens = enc.encode(open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore").read(), allowed_special={"<|endoftext|>"})
    
    candidates = [
        "Paper Baseline",
        "E-W8",
        "E-W16",
        "E-W32",
        "Multi-Scale",
        "State-Compacted (GRM)",
    ]
    
    context_lengths = [512, 1024, 2048, 4096, 8192]
    
    all_results = {}
    
    for cand in candidates:
        print(f"\n>>> Evaluating Architecture: [{cand}]")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model, state_kb = build_candidate_model(cand)
        cand_metrics = {
            "recurrent_state_footprint_kb": state_kb,
            "context_evaluations": {},
        }
        
        for T in context_lengths:
            print(f"  Testing T = {T:4d} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            
            ce, ppl, lat_ms = evaluate_context_length(model, val_tokens, seq_len=T, num_windows=4, seed=42)
            needle_rank = evaluate_needle_at_context(model, enc, corpus_tokens, context_len=T, distance=64, num_trials=5)
            
            vram_alloc = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
            vram_res = torch.cuda.max_memory_reserved(DEVICE) / (1024 * 1024)
            free_b, total_b = torch.cuda.mem_get_info(DEVICE)
            vram_driver = (total_b - free_b) / (1024 * 1024)
            
            print(f"CE = {ce:.4f} | PPL = {ppl:5.2f} | Needle = {needle_rank:6.1f} | Lat = {lat_ms:5.1f} ms | VRAM = {vram_alloc:.0f} MB")
            
            cand_metrics["context_evaluations"][str(T)] = {
                "seq_len": T,
                "ce": ce,
                "ppl": ppl,
                "needle_rank_64": needle_rank,
                "step_latency_ms": lat_ms,
                "vram_allocated_mb": vram_alloc,
                "vram_reserved_mb": vram_res,
                "vram_driver_used_mb": vram_driver,
            }
            
        all_results[cand] = cand_metrics
        del model
        torch.cuda.empty_cache()

    # Save JSON report
    out_json = os.path.join(REPORTS_DIR, "long_context_scaling_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Long-context scaling report saved to: {out_json}")
    
    # Save Markdown report table
    out_md = os.path.join(REPORTS_DIR, "long_context_scaling_table.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Long-Context Scaling Benchmark (T = 512 to 8192)\n\n")
        f.write("| Architecture | State (KB/seq) | Metric | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |\n")
        f.write("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        
        for cand, res in all_results.items():
            state = f"{res['recurrent_state_footprint_kb']:.0f}"
            ce_vals = [f"{res['context_evaluations'][str(T)]['ce']:.4f}" for T in context_lengths]
            ppl_vals = [f"{res['context_evaluations'][str(T)]['ppl']:.2f}" for T in context_lengths]
            needle_vals = [f"{res['context_evaluations'][str(T)]['needle_rank_64']:.1f}" for T in context_lengths]
            lat_vals = [f"{res['context_evaluations'][str(T)]['step_latency_ms']:.1f}ms" for T in context_lengths]
            
            f.write(f"| **{cand}** | {state} KB | CE | {' | '.join(ce_vals)} |\n")
            f.write(f"| | | PPL | {' | '.join(ppl_vals)} |\n")
            f.write(f"| | | Needle Rank | {' | '.join(needle_vals)} |\n")
            f.write(f"| | | Latency | {' | '.join(lat_vals)} |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
            
    print(f"[OK] Markdown scaling table saved to: {out_md}")
    return all_results

if __name__ == "__main__":
    run_long_context_benchmark()
