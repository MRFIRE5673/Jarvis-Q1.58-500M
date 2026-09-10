# experiments/architecture_matrix/standardized_throughput_benchmark.py
"""
STANDARDIZED DETERMINISTIC THROUGHPUT & VRAM BENCHMARK
=====================================================
Audits throughput variation and measures exact inference latency,
tokens per second, and peak VRAM under strictly controlled conditions:
- Fixed batch B=2, sequence length T=512
- 20 warmup iterations (forces GPU clocks to settle at P0 state)
- Explicit torch.cuda.synchronize() before and after every measurement
- 100 timed iterations per trial across 5 repeated trials
- Apples-to-apples comparison of Baseline vs Candidate B+C+E
"""

import os
import sys
import time
import math
import torch
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import Jarvis
from modular_memory import build_modular_jarvis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_deterministic_benchmark(model, name="Model", batch_size=2, seq_len=512, warmup_steps=20, timed_steps=100, num_trials=5):
    model.eval().to(DEVICE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    
    dummy_input = torch.randint(0, 50257, (batch_size, seq_len), device=DEVICE)
    tokens_per_step = batch_size * seq_len
    total_tokens_per_trial = timed_steps * tokens_per_step
    
    print(f"\n[{name}] Warming up for {warmup_steps} iterations...")
    for _ in range(warmup_steps):
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(dummy_input)
    torch.cuda.synchronize(DEVICE)
    
    trial_throughput = []
    trial_step_times = []
    
    print(f"[{name}] Running {num_trials} trials of {timed_steps} steps ({total_tokens_per_trial:,} tokens/trial)...")
    for trial in range(num_trials):
        step_times = []
        torch.cuda.synchronize(DEVICE)
        t_trial_start = time.perf_counter()
        
        for _ in range(timed_steps):
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(dummy_input)
            torch.cuda.synchronize(DEVICE)
            step_times.append(time.perf_counter() - t0)
            
        t_trial_end = time.perf_counter()
        trial_duration = t_trial_end - t_trial_start
        tok_s = total_tokens_per_trial / trial_duration
        mean_step_ms = np.mean(step_times) * 1000.0
        
        trial_throughput.append(tok_s)
        trial_step_times.append(mean_step_ms)
        print(f"  Trial {trial + 1}/{num_trials}: Throughput = {tok_s:8.1f} tok/s | Mean Step Time = {mean_step_ms:6.2f} ms")
        
    peak_vram_allocated_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
    peak_vram_reserved_mb = torch.cuda.max_memory_reserved(DEVICE) / (1024 * 1024)
    
    mean_tok_s = float(np.mean(trial_throughput))
    std_tok_s = float(np.std(trial_throughput))
    min_tok_s = float(np.min(trial_throughput))
    max_tok_s = float(np.max(trial_throughput))
    mean_step = float(np.mean(trial_step_times))
    std_step = float(np.std(trial_step_times))
    
    print(f"[{name}] SUMMARY:")
    print(f"  Throughput : {mean_tok_s:8.1f} +/- {std_tok_s:5.1f} tok/s (Min: {min_tok_s:.1f}, Max: {max_tok_s:.1f})")
    print(f"  Step Latency: {mean_step:6.2f} +/- {std_step:5.2f} ms")
    print(f"  Peak VRAM Allocated: {peak_vram_allocated_mb:8.1f} MB | Reserved: {peak_vram_reserved_mb:8.1f} MB")
    
    return {
        "name": name,
        "mean_throughput_tok_s": mean_tok_s,
        "std_throughput_tok_s": std_tok_s,
        "min_throughput_tok_s": min_tok_s,
        "max_throughput_tok_s": max_tok_s,
        "mean_step_time_ms": mean_step,
        "std_step_time_ms": std_step,
        "peak_vram_allocated_mb": peak_vram_allocated_mb,
        "peak_vram_reserved_mb": peak_vram_reserved_mb,
    }


def main():
    print("=" * 90)
    print("STANDARDIZED DETERMINISTIC BENCHMARK: BASELINE vs E-ONLY vs B+C+E")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print("=" * 90)
    
    # 1. Benchmark Baseline
    print("\n>>> [1/3] Instantiating Paper Baseline...")
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
    base_ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    base_model.load_state_dict(new_sd, strict=True)
    
    base_res = run_deterministic_benchmark(base_model, name="Paper Baseline")
    del base_model
    torch.cuda.empty_cache()
    
    # 2. Benchmark E-Only (Local Buffer W=16)
    print("\n>>> [2/3] Instantiating E-Only (Local Buffer W=16)...")
    e_cfg = {
        "use_local_buffer": True,
        "local_window_size": 16,
        "use_adaptive_decay": False,
        "use_write_gate": False,
        "use_erase_gate": False,
        "use_gated_read": False,
        "fusion_option": 2,
    }
    e_model = build_modular_jarvis(config_dict=e_cfg, max_seq_len=512)
    # Load baseline weights and neutral init
    model_sd = e_model.state_dict()
    filtered_sd = {k: v for k, v in new_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    e_model.load_state_dict(filtered_sd, strict=False)
    for block in e_model.blocks:
        if hasattr(block.attn, "init_neutral"):
            block.attn.init_neutral()
            
    e_res = run_deterministic_benchmark(e_model, name="E-Only (W=16 Buffer)")
    del e_model
    torch.cuda.empty_cache()
    
    # 3. Benchmark Candidate B+C+E
    print("\n>>> [3/3] Instantiating Candidate B+C+E...")
    bce_cfg = {
        "use_local_buffer": True,
        "local_window_size": 16,
        "use_adaptive_decay": False,
        "use_write_gate": True,
        "use_erase_gate": True,
        "use_gated_read": False,
        "fusion_option": 2,
    }
    bce_model = build_modular_jarvis(config_dict=bce_cfg, max_seq_len=512)
    bce_ckpt_path = os.path.join(ARCH_DIR, "ckpt_fact_B+C+E_s123.pt")
    if not os.path.exists(bce_ckpt_path):
        bce_ckpt_path = os.path.join(ARCH_DIR, "ckpt_fact_B+C+E.pt")
    ckpt_bce = torch.load(bce_ckpt_path, map_location="cpu")
    sd_bce = ckpt_bce["model_state_dict"] if "model_state_dict" in ckpt_bce else ckpt_bce
    new_sd_bce = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd_bce.items()}
    bce_model.load_state_dict(new_sd_bce, strict=True)
    
    bce_res = run_deterministic_benchmark(bce_model, name="Candidate B+C+E")
    del bce_model
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 90)
    print("STANDARDIZED PERFORMANCE FAIRNESS COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Metric':<25} | {'Paper Baseline':<18} | {'E-Only (Buffer)':<18} | {'Candidate B+C+E':<18}")
    print("-" * 90)
    
    tp_base = f"{base_res['mean_throughput_tok_s']:.1f} +/- {base_res['std_throughput_tok_s']:.1f}"
    tp_e = f"{e_res['mean_throughput_tok_s']:.1f} +/- {e_res['std_throughput_tok_s']:.1f}"
    tp_bce = f"{bce_res['mean_throughput_tok_s']:.1f} +/- {bce_res['std_throughput_tok_s']:.1f}"
    print(f"{'Throughput (tok/s)':<25} | {tp_base:<18} | {tp_e:<18} | {tp_bce:<18}")
    
    st_base = f"{base_res['mean_step_time_ms']:.2f} ms"
    st_e = f"{e_res['mean_step_time_ms']:.2f} ms"
    st_bce = f"{bce_res['mean_step_time_ms']:.2f} ms"
    print(f"{'Step Latency (ms)':<25} | {st_base:<18} | {st_e:<18} | {st_bce:<18}")
    
    vr_base = f"{base_res['peak_vram_allocated_mb']:.1f} / {base_res['peak_vram_reserved_mb']:.1f}"
    vr_e = f"{e_res['peak_vram_allocated_mb']:.1f} / {e_res['peak_vram_reserved_mb']:.1f}"
    vr_bce = f"{bce_res['peak_vram_allocated_mb']:.1f} / {bce_res['peak_vram_reserved_mb']:.1f}"
    print(f"{'VRAM Alloc/Res (MB)':<25} | {vr_base:<18} | {vr_e:<18} | {vr_bce:<18}")
    print("=" * 90)
    
    import json
    out_path = os.path.join(ARCH_DIR, "standardized_throughput_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"baseline": base_res, "e_only": e_res, "candidate_bce": bce_res}, f, indent=2)
    print(f"\n[OK] Report saved to: {out_path}")

if __name__ == "__main__":
    main()
