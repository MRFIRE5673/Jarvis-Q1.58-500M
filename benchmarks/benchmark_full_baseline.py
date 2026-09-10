# benchmarks/benchmark_full_baseline.py
"""
Complete Phase 2 Baseline Measurement Suite
===========================================
Measures all 7 distinct workloads with strict isolation:
1. Forward-only (micro-batch B=2, T=256)
2. Backward-only (micro-batch B=2, T=256)
3. Complete training step (accum=4, 2048 tokens, with optimizer & sync)
4. Inference prefill (B=1, T=256)
5. Autoregressive single-token decode (Prompt=128, Gen=64 tokens)
6. Batched prefill (B=4, 8, 16, T=256)
7. Batched decode (B=4, 8, Gen=32 tokens)
"""

import os
import sys
import time
import math
import statistics
import torch
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
SPARSE_DIR = os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")
ATTN_DIR = os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")
LIQUID_DIR = os.path.join(WORKSPACE_ROOT, "liquid_fusion_cuda")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, SPARSE_DIR, ATTN_DIR, LIQUID_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

from jarvis_model import Jarvis

def percentile(data, p):
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_d[int(k)]
    d0 = sorted_d[int(f)] * (c - k)
    d1 = sorted_d[int(c)] * (k - f)
    return d0 + d1

def compute_stats(data_list):
    return {
        "mean": statistics.mean(data_list),
        "median": statistics.median(data_list),
        "p10": percentile(data_list, 0.10),
        "p90": percentile(data_list, 0.90),
        "min": min(data_list),
        "max": max(data_list),
        "std": statistics.stdev(data_list) if len(data_list) > 1 else 0.0,
    }

def print_stat_line(label, stats, unit, tok_s=None):
    tok_s_str = f" | {tok_s:8.1f} tok/s" if tok_s is not None else ""
    print(f"{label:<36} | Mean: {stats['mean']:7.2f}{unit} | Median: {stats['median']:7.2f}{unit} | "
          f"P10: {stats['p10']:7.2f}{unit} | P90: {stats['p90']:7.2f}{unit} | "
          f"Min: {stats['min']:7.2f}{unit} | Max: {stats['max']:7.2f}{unit} | Std: {stats['std']:5.2f}{unit}{tok_s_str}")

def run_baseline_suite(num_iters=30):
    print("=" * 85)
    print("PHASE 2: COMPREHENSIVE MULTI-WORKLOAD PERFORMANCE BASELINE AUDIT")
    print(f"Iterations per benchmark: {num_iters} (after warmup)")
    print("=" * 85)

    device = "cuda"
    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=256,
        use_cuda_attn=True,
        use_cuda_moe=True
    ).to(device)

    # 1. Forward-Only Benchmark (Training Mode with grad_ckpt, B=2, T=256)
    print("\n[1/7] Measuring Forward-Only (Training Mode, B=2, T=256)...")
    model.train()
    x_train = torch.randint(0, 50257, (2, 256), device=device)
    y_train = torch.randint(0, 50257, (2, 256), device=device)
    # Warmup
    for _ in range(5):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, _ = model(x_train, targets=y_train)
    torch.cuda.synchronize()

    fwd_times_ms = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, _ = model(x_train, targets=y_train)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_times_ms.append((t1 - t0) * 1000.0)

    # 2. Backward-Only Benchmark (B=2, T=256)
    print("[2/7] Measuring Backward-Only (B=2, T=256)...")
    bwd_times_ms = []
    for _ in range(num_iters):
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x_train, targets=y_train)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bwd_times_ms.append((t1 - t0) * 1000.0)

    # 3. Complete Training Step (B=2, accum=4, 2048 tokens)
    print("[3/7] Measuring Complete Training Step (accum=4, 2048 tokens)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    step_times_s = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(4):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x_train, targets=y_train)
            (loss / 4.0).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times_s.append(t1 - t0)

    # 4. Inference Prefill (Eval Mode, B=1, T=256)
    print("[4/7] Measuring Inference Prefill (B=1, T=256)...")
    model.eval()
    x_inf1 = torch.randint(0, 50257, (1, 256), device=device)
    for _ in range(5):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(x_inf1)
    torch.cuda.synchronize()

    inf_prefill_ms = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(x_inf1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inf_prefill_ms.append((t1 - t0) * 1000.0)

    # 5. Autoregressive Single-Token Decode (T_prompt=128, Gen=64 tokens)
    print("[5/7] Measuring Autoregressive Single-Token Decode...")
    prompt_len = 128
    gen_len = 64
    x_prompt = torch.randint(0, 50257, (1, prompt_len), device=device)

    decode_token_latencies_ms = []
    first_token_latencies_ms = []

    for _ in range(10): # 10 decode runs
        model.reset_state()
        curr_seq = x_prompt.clone()

        # Prefill prompt
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(curr_seq, persist_state=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        first_token_latencies_ms.append((t1 - t0) * 1000.0)

        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr_seq = torch.cat([curr_seq, next_tok], dim=1)

        # Autoregressive loop
        for _ in range(gen_len - 1):
            ctx = curr_seq[:, -256:]
            torch.cuda.synchronize()
            t_tok0 = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(ctx, persist_state=False)
            torch.cuda.synchronize()
            t_tok1 = time.perf_counter()
            decode_token_latencies_ms.append((t_tok1 - t_tok0) * 1000.0)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_seq = torch.cat([curr_seq, next_tok], dim=1)

    # 6. Batched Prefill (B=4, 8, 16, T=256)
    print("[6/7] Measuring Batched Prefill (B=4, 8, 16)...")
    batched_prefill_results = {}
    for B in [4, 8, 16]:
        x_b = torch.randint(0, 50257, (B, 256), device=device)
        b_times_ms = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model(x_b)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            b_times_ms.append((t1 - t0) * 1000.0)
        batched_prefill_results[B] = b_times_ms

    # 7. Batched Decode (B=4, 8, Gen=32)
    print("[7/7] Measuring Batched Decode (B=4, 8)...")
    batched_decode_results = {}
    for B in [4, 8]:
        tok_lats = []
        x_b = torch.randint(0, 50257, (B, 128), device=device)
        for _ in range(5):
            curr_seq = x_b.clone()
            for _ in range(16):
                ctx = curr_seq[:, -256:]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits, _ = model(ctx)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                tok_lats.append((t1 - t0) * 1000.0)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_seq = torch.cat([curr_seq, next_tok], dim=1)
        batched_decode_results[B] = tok_lats

    # Print Summary Table
    print("\n" + "=" * 85)
    print("PHASE 2 COMPREHENSIVE BASELINE RESULTS")
    print("=" * 85)

    s_fwd = compute_stats(fwd_times_ms)
    s_bwd = compute_stats(bwd_times_ms)
    s_step = compute_stats(step_times_s)
    s_prefill = compute_stats(inf_prefill_ms)
    s_first = compute_stats(first_token_latencies_ms)
    s_dec = compute_stats(decode_token_latencies_ms)

    fwd_tok_s = 512.0 / (s_fwd['mean'] / 1000.0)
    train_tok_s = 2048.0 / s_step['mean']
    prefill_tok_s = 256.0 / (s_prefill['mean'] / 1000.0)
    decode_tok_s = 1000.0 / s_dec['mean']

    print_stat_line("1. Forward-Only (B=2, T=256)", s_fwd, " ms", tok_s=fwd_tok_s)
    print_stat_line("2. Backward-Only (B=2, T=256)", s_bwd, " ms")
    print_stat_line("3. Complete Training Step (2048 tok)", s_step, " s ", tok_s=train_tok_s)
    print_stat_line("4. Inference Prefill (B=1, T=256)", s_prefill, " ms", tok_s=prefill_tok_s)
    print_stat_line("5. First-Token Latency (T=128)", s_first, " ms")
    print_stat_line("6. Single-Token Decode (B=1)", s_dec, " ms", tok_s=decode_tok_s)

    for B, times in batched_prefill_results.items():
        st = compute_stats(times)
        b_tok_s = (B * 256.0) / (st['mean'] / 1000.0)
        print_stat_line(f"   Batched Prefill (B={B:2d}, T=256)", st, " ms", tok_s=b_tok_s)

    for B, times in batched_decode_results.items():
        st = compute_stats(times)
        b_tok_s = (B * 1000.0) / st['mean']
        print_stat_line(f"   Batched Decode (B={B:2d})", st, " ms", tok_s=b_tok_s)

    print("=" * 85)

if __name__ == '__main__':
    run_baseline_suite(num_iters=30)
