# benchmark_production_ab.py
"""
End-to-End Production A/B Benchmark: Current Production vs Integrated CUDA Production
=====================================================================================
Strict 1-to-1 comparison between:
  1. BASELINE PRODUCTION (Pure PyTorch Associative Attention + Pure PyTorch Sparse MoE)
     use_cuda_attn=False, use_cuda_moe=False
  2. INTEGRATED PRODUCTION (CUDA Associative Attention + CUDA Sparse MoE)
     use_cuda_attn=True, use_cuda_moe=True

Specifications:
  - Architecture: Jarvis 606M (vocab 50257, d_model=1024, 24 layers, 16 heads, 4 experts, top_k=2, max_seq_len=256)
  - Workload: B=2, T=256, BF16 autocast, gradient_accumulation=4 (2048 tokens/update)
  - Optimization: Fused AdamW, lr=3e-4, grad_norm_clip=1.0, 24/24 layers gradient checkpointed
  - Benchmarking: 3 warmup updates + 20 timed updates in separate subprocesses
  - Measurements: Step time, tok/s, peak allocated & reserved VRAM, loss trajectory, correctness checks
"""

import os
import sys
import time
import math
import copy
import json
import argparse
import subprocess
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
JARVIS_ENGINE_PATH = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
SPARSE_DIR = os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")
ATTN_DIR = os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")

for p in [WORKSPACE_ROOT, SPARSE_DIR, ATTN_DIR, JARVIS_ENGINE_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.cuda.set_per_process_memory_fraction(0.92)

from jarvis_model import Jarvis


def run_benchmark_worker(mode: str, num_warmup: int = 3, num_updates: int = 20, seed: int = 42):
    """
    Runs benchmark in an isolated worker process.
    mode: 'baseline' (use_cuda_attn=False, use_cuda_moe=False)
          'integrated' (use_cuda_attn=True, use_cuda_moe=True)
    """
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA is required for benchmark."

    use_cuda = (mode == "integrated")

    # 1. Deterministic initialization
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=256,
        use_cuda_attn=use_cuda,
        use_cuda_moe=use_cuda
    ).to(device)

    status = model.get_backend_status()
    print(f"[{mode.upper()}] Backend status: {status}", flush=True)

    if use_cuda:
        assert status["attn_backend"] == "cuda" and status["moe_backend"] == "cuda"
    else:
        assert status["attn_backend"] == "pytorch" and status["moe_backend"] == "pytorch"

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    # 2. Data loader setup with exact determinism
    enc = tiktoken.get_encoding("gpt2")
    data_path = os.path.join(JARVIS_ENGINE_PATH, "data.txt")
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tokens = torch.tensor(enc.encode(text), dtype=torch.long, device=device)
    _offsets = torch.arange(256, device=device)

    BATCH_SIZE = 2
    ACCUM_STEPS = 4
    SEQ_LEN = 256
    TOKENS_PER_UPDATE = BATCH_SIZE * ACCUM_STEPS * SEQ_LEN  # 2048

    # Pre-generate batch indices with fixed seed for identical data across runs
    torch.manual_seed(seed + 100)
    torch.cuda.manual_seed_all(seed + 100)

    batch_indices = []
    total_microbatches = (num_warmup + num_updates) * ACCUM_STEPS
    for _ in range(total_microbatches):
        ix = torch.randint(0, len(tokens) - SEQ_LEN - 1, (BATCH_SIZE,), device=tokens.device)
        batch_indices.append(ix)

    batch_ptr = 0
    def get_deterministic_batch():
        nonlocal batch_ptr
        ix = batch_indices[batch_ptr]
        batch_ptr += 1
        idx = ix.unsqueeze(1) + _offsets[:SEQ_LEN]
        return tokens[idx], tokens[idx + 1]

    # 3. Warmup Phase
    print(f"[{mode.upper()}] Starting warmup ({num_warmup} updates)...", flush=True)
    for _ in range(num_warmup):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(ACCUM_STEPS):
            x, y = get_deterministic_batch()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
            (loss / ACCUM_STEPS).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # 4. Timed Updates
    print(f"[{mode.upper()}] Executing {num_updates} timed updates...", flush=True)
    step_times = []
    fwd_times = []
    bwd_times = []
    losses = []
    grad_norms = []

    for update_idx in range(1, num_updates + 1):
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        step_fwd_ms = 0.0
        step_bwd_ms = 0.0

        for micro_idx in range(ACCUM_STEPS):
            x, y = get_deterministic_batch()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
                scaled_loss = loss / ACCUM_STEPS
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            scaled_loss.backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            step_fwd_ms += (t1 - t0) * 1000.0
            step_bwd_ms += (t2 - t1) * 1000.0
            accum_loss += loss.item() / ACCUM_STEPS

        # Gradient clipping and optimizer step
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        torch.cuda.synchronize()
        t_step_end = time.perf_counter()

        step_duration = t_step_end - t_step_start
        step_times.append(step_duration)
        fwd_times.append(step_fwd_ms)
        bwd_times.append(step_bwd_ms)
        losses.append(accum_loss)
        grad_norms.append(total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm))

        tok_s = TOKENS_PER_UPDATE / step_duration
        print(f"[{mode.upper()}] Update {update_idx:02d}/{num_updates:02d}: {step_duration:.4f}s "
              f"({tok_s:5.1f} tok/s) | Fwd: {step_fwd_ms:6.1f}ms | Bwd: {step_bwd_ms:6.1f}ms | "
              f"Loss: {accum_loss:.4f} | GradNorm: {grad_norms[-1]:.3f}", flush=True)

    # 5. Peak memory and correctness verification
    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    # Check non-zero gradients on attention and MoE across all layers
    all_gamma_had_grad = all(
        b.attn.gamma_raw.grad is not None and b.attn.gamma_raw.grad.abs().sum().item() > 0
        for b in model.blocks
    )
    all_routers_had_grad = all(
        b.moe.router.weight.grad is not None and b.moe.router.weight.grad.abs().sum().item() > 0
        for b in model.blocks
    )
    all_experts_had_grad = all(
        all(e.weight.grad is not None and e.weight.grad.abs().sum().item() > 0 for e in b.moe.w1)
        for b in model.blocks
    )

    results = {
        "mode": mode,
        "num_updates": num_updates,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "step_times": step_times,
        "fwd_times": fwd_times,
        "bwd_times": bwd_times,
        "losses": losses,
        "grad_norms": grad_norms,
        "mean_step": statistics.mean(step_times),
        "median_step": statistics.median(step_times),
        "std_step": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "min_step": min(step_times),
        "max_step": max(step_times),
        "mean_tok_s": TOKENS_PER_UPDATE / statistics.mean(step_times),
        "median_tok_s": TOKENS_PER_UPDATE / statistics.median(step_times),
        "peak_alloc_gb": peak_alloc_gb,
        "peak_res_gb": peak_res_gb,
        "all_gamma_had_grad": all_gamma_had_grad,
        "all_routers_had_grad": all_routers_had_grad,
        "all_experts_had_grad": all_experts_had_grad,
        "backend_status": status,
    }

    print("\n__BENCHMARK_RESULT_START__")
    print(json.dumps(results))
    print("__BENCHMARK_RESULT_END__\n")
    return results


def run_isolated_subprocess(mode: str, num_warmup: int, num_updates: int, seed: int):
    """Launches benchmark in a fresh subprocess and extracts JSON result."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", mode,
        "--warmup", str(num_warmup),
        "--updates", str(num_updates),
        "--seed", str(seed),
        "--worker"
    ]
    print(f"\n========================================================")
    print(f"Launching isolated worker for: {mode.upper()}")
    print(f"========================================================")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(f"[{mode.upper()} STDERR]\n{proc.stderr}")
    if proc.returncode != 0:
        raise RuntimeError(f"Worker for {mode} failed with returncode {proc.returncode}")

    stdout = proc.stdout
    start_tag = "__BENCHMARK_RESULT_START__"
    end_tag = "__BENCHMARK_RESULT_END__"
    start_idx = stdout.find(start_tag)
    end_idx = stdout.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        raise RuntimeError(f"Could not find JSON result markers in worker output for {mode}")

    json_str = stdout[start_idx + len(start_tag):end_idx].strip()
    return json.loads(json_str)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Production A/B Benchmark")
    parser.add_argument("--mode", choices=["baseline", "integrated", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker", action="store_true", help="Internal flag for worker process")

    args = parser.parse_args()

    if args.worker:
        run_benchmark_worker(args.mode, args.warmup, args.updates, args.seed)
        return

    print("================================================================================")
    print("        JARVIS 606M PRODUCTION INTEGRATION A/B BENCHMARK (20 UPDATES)          ")
    print("================================================================================")
    print(f"Hardware: RTX 5070 12GB | sm_120 Blackwell | CUDA Toolkit 13.3")
    print(f"Workload: B=2, T=256, Accum=4 (2048 tokens/update) | BF16 | 24/24 Checkpointing")
    print(f"Optimizer: Fused AdamW (lr=3e-4, clip=1.0) | Seed={args.seed} | Updates={args.updates}")
    print("================================================================================")

    res_baseline = run_isolated_subprocess("baseline", args.warmup, args.updates, args.seed)
    res_integrated = run_isolated_subprocess("integrated", args.warmup, args.updates, args.seed)

    # Comparison summary
    base_mean = res_baseline["mean_step"]
    integ_mean = res_integrated["mean_step"]
    speedup = (base_mean - integ_mean) / base_mean * 100.0
    throughput_gain = (res_integrated["mean_tok_s"] - res_baseline["mean_tok_s"]) / res_baseline["mean_tok_s"] * 100.0

    print("\n" + "=" * 80)
    print("                  PRODUCTION A/B BENCHMARK FINAL RESULTS                        ")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Baseline (PyTorch)':<22} | {'Integrated (CUDA)':<22}")
    print("-" * 80)
    print(f"{'Mean Step Time (s)':<30} | {res_baseline['mean_step']:<22.4f} | {res_integrated['mean_step']:<22.4f}")
    print(f"{'Median Step Time (s)':<30} | {res_baseline['median_step']:<22.4f} | {res_integrated['median_step']:<22.4f}")
    print(f"{'Min Step Time (s)':<30} | {res_baseline['min_step']:<22.4f} | {res_integrated['min_step']:<22.4f}")
    print(f"{'Max Step Time (s)':<30} | {res_baseline['max_step']:<22.4f} | {res_integrated['max_step']:<22.4f}")
    print(f"{'Std Dev (s)':<30} | {res_baseline['std_step']:<22.4f} | {res_integrated['std_step']:<22.4f}")
    print(f"{'Throughput (tok/s)':<30} | {res_baseline['mean_tok_s']:<22.1f} | {res_integrated['mean_tok_s']:<22.1f}")
    print(f"{'Peak Allocated VRAM (GB)':<30} | {res_baseline['peak_alloc_gb']:<22.2f} | {res_integrated['peak_alloc_gb']:<22.2f}")
    print(f"{'Peak Reserved VRAM (GB)':<30} | {res_baseline['peak_res_gb']:<22.2f} | {res_integrated['peak_res_gb']:<22.2f}")
    print(f"{'Initial Loss (Update 1)':<30} | {res_baseline['losses'][0]:<22.4f} | {res_integrated['losses'][0]:<22.4f}")
    print(f"{'Final Loss (Update 20)':<30} | {res_baseline['losses'][-1]:<22.4f} | {res_integrated['losses'][-1]:<22.4f}")
    print(f"{'Mean Loss':<30} | {statistics.mean(res_baseline['losses']):<22.4f} | {statistics.mean(res_integrated['losses']):<22.4f}")
    print(f"{'All Gammas Had Grad':<30} | {str(res_baseline['all_gamma_had_grad']):<22} | {str(res_integrated['all_gamma_had_grad']):<22}")
    print(f"{'All Routers Had Grad':<30} | {str(res_baseline['all_routers_had_grad']):<22} | {str(res_integrated['all_routers_had_grad']):<22}")
    print(f"{'All Experts Had Grad':<30} | {str(res_baseline['all_experts_had_grad']):<22} | {str(res_integrated['all_experts_had_grad']):<22}")
    print("-" * 80)
    print(f"End-to-End Latency Reduction: {speedup:+.2f}% ({base_mean:.4f}s -> {integ_mean:.4f}s)")
    print(f"End-to-End Throughput Gain:   {throughput_gain:+.2f}% ({res_baseline['mean_tok_s']:.1f} -> {res_integrated['mean_tok_s']:.1f} tok/s)")
    print("=" * 80)

    # Save summary artifact
    summary_path = os.path.join(WORKSPACE_ROOT, "benchmark_production_ab_results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "baseline": res_baseline,
            "integrated": res_integrated,
            "speedup_percent": speedup,
            "throughput_gain_percent": throughput_gain
        }, f, indent=2)
    print(f"Detailed JSON results written to: {summary_path}")


if __name__ == "__main__":
    main()
