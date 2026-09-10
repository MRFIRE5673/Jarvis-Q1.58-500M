# reproducible_benchmark.py
"""
Jarvis / Model Performance Report & Forensic Benchmark Suite
============================================================
Accurately measures and prints:
- Genuine end-to-end training throughput (Forward + Backward + Optimizer + Sync)
- Comparative sub-workloads (Inference Prefill, Forward-Only, Tokenization)
- Loss metrics on training corpus and holdout validation corpus
- Peak VRAM footprint and hardware parameters
"""

import os
import sys
import time
import math
import statistics
import torch
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
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
torch.cuda.set_per_process_memory_fraction(0.92)

import tiktoken
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

def run_benchmark(num_warmup=5, num_measured=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA required for benchmark."

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_vram_gb = gpu_props.total_memory / (1024**3)

    # Load Model
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

    n_params, param_str = model.param_count()
    status = model.get_backend_status()

    # Load checkpoint weights if available
    ckpt_path = os.path.join(JARVIS_ENGINE, "ckpt_step_0004209.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        cleaned_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd, strict=True)
        del ckpt, sd, cleaned_sd
        torch.cuda.empty_cache()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    # Dataset Setup
    enc = tiktoken.get_encoding("gpt2")
    train_path = os.path.join(JARVIS_ENGINE, "data.txt")
    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        train_text = f.read()
    train_tokens = torch.tensor(enc.encode(train_text), dtype=torch.long, device=device)

    holdout_path = os.path.join(JARVIS_ENGINE, "fresh_holdout.txt")
    val_tokens = None
    if os.path.exists(holdout_path):
        with open(holdout_path, "r", encoding="utf-8", errors="ignore") as f:
            holdout_text = f.read()
        val_tokens = torch.tensor(enc.encode(holdout_text), dtype=torch.long, device=device)

    BATCH_SIZE = 2
    ACCUM_STEPS = 4
    SEQ_LEN = 256
    TOK_MICROBATCH = BATCH_SIZE * SEQ_LEN  # 512
    TOK_STEP = BATCH_SIZE * ACCUM_STEPS * SEQ_LEN  # 2048
    _offsets = torch.arange(SEQ_LEN, device=device)

    # 1. Measure Tokenization Throughput
    tok_sample = train_text[:10000]
    t0 = time.perf_counter()
    _ = enc.encode(tok_sample)
    t1 = time.perf_counter()
    tok_throughput = len(_) / max(t1 - t0, 1e-6)

    # 2. Warmup
    print(f"Executing {num_warmup} warmup steps...", flush=True)
    for _ in range(num_warmup):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(ACCUM_STEPS):
            ix = torch.randint(0, len(train_tokens) - SEQ_LEN - 1, (BATCH_SIZE,), device=device)
            idx = ix.unsqueeze(1) + _offsets
            x = train_tokens[idx]
            y = train_tokens[idx + 1]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
            (loss / ACCUM_STEPS).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # 3. Timed Training Updates
    print(f"Measuring {num_measured} full training updates...", flush=True)
    fwd_times_ms = []
    bwd_times_ms = []
    opt_times_ms = []
    data_times_ms = []
    sync_times_ms = []
    total_step_s = []
    losses = []

    fwd_ev0 = torch.cuda.Event(enable_timing=True)
    fwd_ev1 = torch.cuda.Event(enable_timing=True)
    bwd_ev0 = torch.cuda.Event(enable_timing=True)
    bwd_ev1 = torch.cuda.Event(enable_timing=True)
    opt_ev0 = torch.cuda.Event(enable_timing=True)
    opt_ev1 = torch.cuda.Event(enable_timing=True)

    for _ in range(num_measured):
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        # Data sampling
        t_d0 = time.perf_counter()
        batches = []
        for _ in range(ACCUM_STEPS):
            ix = torch.randint(0, len(train_tokens) - SEQ_LEN - 1, (BATCH_SIZE,), device=device)
            idx = ix.unsqueeze(1) + _offsets
            batches.append((train_tokens[idx], train_tokens[idx + 1]))
        torch.cuda.synchronize()
        t_d1 = time.perf_counter()
        data_times_ms.append((t_d1 - t_d0) * 1000.0)

        optimizer.zero_grad(set_to_none=True)
        step_fwd_ms = 0.0
        step_bwd_ms = 0.0
        step_loss = 0.0

        for micro_idx in range(ACCUM_STEPS):
            x, y = batches[micro_idx]

            # Forward
            fwd_ev0.record()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
                scaled_loss = loss / ACCUM_STEPS
            fwd_ev1.record()
            fwd_ev1.synchronize()
            step_fwd_ms += fwd_ev0.elapsed_time(fwd_ev1)
            step_loss += loss.item() / ACCUM_STEPS

            # Backward
            bwd_ev0.record()
            scaled_loss.backward()
            bwd_ev1.record()
            bwd_ev1.synchronize()
            step_bwd_ms += bwd_ev0.elapsed_time(bwd_ev1)

        fwd_times_ms.append(step_fwd_ms)
        bwd_times_ms.append(step_bwd_ms)
        losses.append(step_loss)

        # Optimizer
        opt_ev0.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        opt_ev1.record()
        opt_ev1.synchronize()
        opt_times_ms.append(opt_ev0.elapsed_time(opt_ev1))

        # Sync
        t_s0 = time.perf_counter()
        torch.cuda.synchronize()
        t_s1 = time.perf_counter()
        sync_times_ms.append((t_s1 - t_s0) * 1000.0)

        t_step_end = time.perf_counter()
        total_step_s.append(t_step_end - t_step_start)

    # 4. Measure Inference Prefill Throughput on same batch size and on large batch
    model.eval()
    with torch.no_grad():
        # Microbatch prefill (B=2, T=256)
        x_inf = train_tokens[:512].view(2, 256)
        t0 = time.perf_counter()
        for _ in range(10):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(x_inf)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inf_micro_tok_s = (512 * 10) / (t1 - t0)

        # Batched prefill (B=16, T=256 = 4096 tokens)
        x_batched = train_tokens[:4096].view(16, 256)
        t0 = time.perf_counter()
        for _ in range(10):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(x_batched)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inf_batched_tok_s = (4096 * 10) / (t1 - t0)

        # 5. Evaluate Validation Loss
        val_loss = None
        val_ppl = None
        if val_tokens is not None and len(val_tokens) > 257:
            val_losses = []
            for start_idx in range(0, min(len(val_tokens) - 257, 10 * 256), 256):
                xv = val_tokens[start_idx : start_idx + 256].unsqueeze(0)
                yv = val_tokens[start_idx + 1 : start_idx + 257].unsqueeze(0)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, l = model(xv, targets=yv)
                val_losses.append(l.item())
            val_loss = statistics.mean(val_losses)
            val_ppl = math.exp(min(val_loss, 100.0))

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_res = torch.cuda.max_memory_reserved() / (1024**3)

    mean_fwd_ms = statistics.mean(fwd_times_ms)
    mean_bwd_ms = statistics.mean(bwd_times_ms)
    mean_opt_ms = statistics.mean(opt_times_ms)
    mean_data_ms = statistics.mean(data_times_ms)
    mean_sync_ms = statistics.mean(sync_times_ms)
    mean_step_s = statistics.mean(total_step_s)
    true_tok_s = TOK_STEP / mean_step_s
    mean_train_loss = statistics.mean(losses)
    train_ppl = math.exp(min(mean_train_loss, 100.0))

    # Print Final Standardized Report
    print("\n" + "=" * 60)
    print("JARVIS / MODEL PERFORMANCE REPORT")
    print("=" * 60)
    print(f"GPU:                      {gpu_name}")
    print(f"GPU VRAM:                 {gpu_vram_gb:.2f} GB")
    print(f"\nParameters:               {n_params:,} ({param_str})")
    print(f"Precision:                BF16 Autocast (FP32 Master / Ternary Weights)")
    print(f"\nBatch:                    {BATCH_SIZE}")
    print(f"Sequence:                 {SEQ_LEN}")
    print(f"Gradient accumulation:    {ACCUM_STEPS}")
    print(f"\nTokens / microbatch:      {TOK_MICROBATCH}")
    print(f"Tokens / optimizer step:  {TOK_STEP}")
    print(f"\nForward:                  {mean_fwd_ms:.1f} ms")
    print(f"Backward:                 {mean_bwd_ms:.1f} ms")
    print(f"Optimizer:                {mean_opt_ms:.1f} ms")
    print(f"Data:                     {mean_data_ms:.2f} ms")
    print(f"Synchronization:          {mean_sync_ms:.3f} ms")
    print(f"Total step:               {mean_step_s:.4f} s ({mean_step_s * 1e6:,.0f} us)")
    print(f"\nTRUE TOKENS/SEC:          {true_tok_s:.1f} tokens/sec")
    print(f"\nTraining loss:            {mean_train_loss:.4f}")
    if val_loss is not None:
        print(f"Validation loss:          {val_loss:.4f}")
        print(f"Perplexity:               {val_ppl:.2f}")
    else:
        print(f"Validation loss:          N/A")
        print(f"Perplexity:               {train_ppl:.2f}")
    print(f"\nPeak VRAM:                {peak_alloc:.2f} GB (Reserved: {peak_res:.2f} GB)")
    print("=" * 60)

    print("\n--- COMPARATIVE WORKLOAD THROUGHPUT (DISAMBIGUATION) ---")
    print(f"1. Genuine End-to-End Training:      {true_tok_s:8.1f} tok/s  (Fwd + Bwd + Opt + Clip + Sync)")
    print(f"2. Forward-Only Inference (B=2):      {inf_micro_tok_s:8.1f} tok/s  (No Bwd, No Optimizer)")
    print(f"3. Batched Inference Prefill (B=16):  {inf_batched_tok_s:8.1f} tok/s  (Tensor Cores Saturated)")
    print(f"4. Tiktoken Tokenization (CPU):       {tok_throughput:8.1f} tok/s  (Host text encoding)")
    print("=" * 60)

if __name__ == '__main__':
    run_benchmark(num_warmup=5, num_measured=20)
