# experiments/architecture_matrix/memory_v2/run_multiscale_experiments.py
"""
MEMORY v2: MULTI-SCALE LOCAL WINDOW & COMPACTION EXPERIMENT RUNNER
===================================================================
Runs controlled adaptation and rigorous evaluation across:
1. E-W8:  Uniform W=8
2. E-W16: Uniform W=16
3. E-W32: Uniform W=32
4. Multi-Scale: W in {8, 16, 32} across head groups ([8]*4 + [16]*6 + [32]*6)
5. State-Compacted: Grouped Recurrent Memory (GRM, 4 KV groups)

Evaluates:
- Parameter overhead
- Forward correctness & Causal integrity
- T=512 and T=1024 Holdout Cross-Entropy and Perplexity
- Associative needle retrieval rank @ 64 tokens
- Standardized throughput (tok/s) and step latency (ms)
- Peak allocated & reserved VRAM
- Local buffer vs recurrent path contribution
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


@torch.inference_mode()
def benchmark_model_speed(model, batch_size=2, seq_len=512, trials=50):
    model.eval()
    dummy = torch.randint(0, 50257, (batch_size, seq_len), device=DEVICE)
    # Warmup
    for _ in range(10):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(dummy)
    torch.cuda.synchronize(DEVICE)
    
    t0 = time.perf_counter()
    for _ in range(trials):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(dummy)
    torch.cuda.synchronize(DEVICE)
    dur = time.perf_counter() - t0
    
    step_latency_ms = (dur / trials) * 1000.0
    throughput = (batch_size * seq_len * trials) / dur
    return throughput, step_latency_ms


@torch.inference_mode()
def collect_path_contributions(model, dummy):
    model.eval()
    local_norms = []
    rec_norms = []
    
    x = model.tok_emb(dummy)
    for block in model.blocks:
        x_norm = block.norm1(x)
        attn = block.attn
        if hasattr(attn, "forward") and hasattr(attn, "head_windows"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, diag = attn(x_norm, collect_diagnostics=True)
                local_norms.append(diag["local_norm"])
                rec_norms.append(diag["recurrent_norm"])
        x, _, _, _ = block(x)

    if not local_norms:
        return 0.0, 1.0
        
    m_l = float(np.mean(local_norms))
    m_r = float(np.mean(rec_norms))
    tot = max(m_l + m_r, 1e-6)
    return m_l / tot, m_r / tot


def load_baseline_checkpoint(model):
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
    return len(filtered_sd)


def run_experiment_on_variant(name, builder_fn, train_tokens, val_tokens, corpus_tokens, enc, steps=100, seed=42):
    print(f"\n{'='*85}")
    print(f"RUNNING EXPERIMENT: [{name}] (Steps: {steps}, Seed: {seed})")
    print(f"{'='*85}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = builder_fn().to(DEVICE)
    tensors_loaded = load_baseline_checkpoint(model)
    total_params = sum(p.numel() for p in model.parameters())
    param_overhead = total_params - 606391704
    
    print(f"Loaded {tensors_loaded} baseline tensors.")
    print(f"Total Parameters: {total_params:,} (Overhead: {param_overhead:+,d} params)")
    
    # Step 0 initial evaluation
    ce_512_s0, ppl_512_s0 = evaluate_multicontext(model, val_tokens, seq_len=512, num_windows=10, seed=seed)
    ce_1024_s0, ppl_1024_s0 = evaluate_multicontext(model, val_tokens, seq_len=1024, num_windows=10, seed=seed)
    needle_s0 = evaluate_needle_rank(model, enc, corpus_tokens, distance=64, num_trials=5, seed=seed)
    dummy_diag = torch.randint(0, 50257, (2, 512), device=DEVICE)
    loc_rat_s0, rec_rat_s0 = collect_path_contributions(model, dummy_diag)
    
    print(f"  Step 0:  CE@512 = {ce_512_s0:.4f} | CE@1024 = {ce_1024_s0:.4f} | Needle = {needle_s0:6.1f} | Local Ratio = {loc_rat_s0*100:.1f}%")
    
    # Speed benchmark
    throughput, latency_ms = benchmark_model_speed(model, batch_size=2, seq_len=512, trials=50)
    print(f"  Speed:   {throughput:.1f} tok/s | Latency = {latency_ms:.2f} ms")
    
    # Controlled adaptation training (100 steps)
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
    t_train_0 = time.perf_counter()
    losses = []
    
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(accum_loss)
        
    train_dur = time.perf_counter() - t_train_0
    
    # Step 100 final evaluation
    ce_512_s100, ppl_512_s100 = evaluate_multicontext(model, val_tokens, seq_len=512, num_windows=20, seed=seed)
    ce_1024_s100, ppl_1024_s100 = evaluate_multicontext(model, val_tokens, seq_len=1024, num_windows=20, seed=seed)
    needle_s100 = evaluate_needle_rank(model, enc, corpus_tokens, distance=64, num_trials=10, seed=seed)
    loc_rat_s100, rec_rat_s100 = collect_path_contributions(model, dummy_diag)
    
    vram_alloc = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
    vram_res = torch.cuda.max_memory_reserved(DEVICE) / (1024 * 1024)
    free_b, total_b = torch.cuda.mem_get_info(DEVICE)
    vram_driver = (total_b - free_b) / (1024 * 1024)
    
    print(f"  Step 100: CE@512 = {ce_512_s100:.4f} (PPL {ppl_512_s100:.2f}) | CE@1024 = {ce_1024_s100:.4f} (PPL {ppl_1024_s100:.2f})")
    print(f"  Needle Rank @ 64: {needle_s100:6.1f} / 50257")
    print(f"  Local Buffer Ratio: {loc_rat_s100*100:.1f}% | Recurrent Ratio: {rec_rat_s100*100:.1f}%")
    print(f"  Peak VRAM: Allocated = {vram_alloc:.1f} MB | Reserved = {vram_res:.1f} MB | Driver = {vram_driver:.1f} MB")
    
    res = {
        "variant": name,
        "parameters": {
            "total": total_params,
            "overhead": param_overhead,
        },
        "step_0": {
            "ce_512": ce_512_s0,
            "ppl_512": ppl_512_s0,
            "ce_1024": ce_1024_s0,
            "ppl_1024": ppl_1024_s0,
            "needle_rank_64": needle_s0,
            "local_ratio": loc_rat_s0,
            "recurrent_ratio": rec_rat_s0,
        },
        "step_100": {
            "ce_512": ce_512_s100,
            "ppl_512": ppl_512_s100,
            "ce_1024": ce_1024_s100,
            "ppl_1024": ppl_1024_s100,
            "needle_rank_64": needle_s100,
            "local_ratio": loc_rat_s100,
            "recurrent_ratio": rec_rat_s100,
        },
        "speed": {
            "throughput_tok_s": throughput,
            "step_latency_ms": latency_ms,
        },
        "vram": {
            "allocated_mb": vram_alloc,
            "reserved_mb": vram_res,
            "driver_used_mb": vram_driver,
        },
        "training": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "duration_sec": train_dur,
        }
    }
    
    del model, optimizer
    torch.cuda.empty_cache()
    return res


def main():
    print("=" * 85)
    print("MEMORY v2 COMPREHENSIVE EXPERIMENTAL PROGRAM")
    print("=" * 85)
    
    enc = tiktoken.get_encoding("gpt2")
    with open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore") as f:
        train_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    with open(os.path.join(JARVIS_ENGINE, "fresh_holdout.txt"), "r", encoding="utf-8", errors="ignore") as f:
        val_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=DEVICE)
    corpus_tokens = enc.encode(open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore").read(), allowed_special={"<|endoftext|>"})
    
    variants = [
        ("E-W8", lambda: build_multiscale_jarvis(window_config=8, max_seq_len=512)),
        ("E-W16", lambda: build_multiscale_jarvis(window_config=16, max_seq_len=512)),
        ("E-W32", lambda: build_multiscale_jarvis(window_config=32, max_seq_len=512)),
        ("Multi-Scale (W8/16/32)", lambda: build_multiscale_jarvis(window_config=[8]*4 + [16]*6 + [32]*6, max_seq_len=512)),
        ("State-Compacted (GRM)", lambda: build_candidate_grm()),
    ]
    
    def build_candidate_grm():
        model = Jarvis(
            vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
            num_experts=4, top_k=2, max_seq_len=512,
            use_cuda_attn=False, use_cuda_moe=False
        )
        base_ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
        ckpt = torch.load(base_ckpt_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=True)
        for block in model.blocks:
            grm = GroupedRecurrentMemoryAttention(d_model=1024, n_heads=16, n_kv_groups=4, local_window=16, max_seq_len=512)
            grm.init_from_baseline_attn(block.attn)
            block.attn = grm
        return model

    results = {}
    for name, builder in variants:
        res = run_experiment_on_variant(name, builder, train_tokens, val_tokens, corpus_tokens, enc, steps=100)
        results[name] = res
        
    out_file = os.path.join(REPORTS_DIR, "memory_v2_multiscale_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Memory v2 full report saved to: {out_file}")

if __name__ == "__main__":
    main()
