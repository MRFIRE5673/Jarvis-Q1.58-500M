# experiments/audit_baseline_600m.py
"""
Hard Baseline Audit for Jarvis 606M Model (ckpt_step_0004209.pt)
================================================================
Measures and records:
1. Strict checkpoint loading verification
2. Exact parameter count (total, active per token, dense, expert)
3. Decoupled Cross-Entropy vs MoE Load-Balance vs Reflective Loss on holdout (fresh_holdout.txt)
4. Pure holdout perplexity: exp(CE_loss)
5. Training loss on data.txt (decoupled)
6. Deterministic generation on 5 fixed benchmark prompts
7. Throughput (prefill & decode tok/s) and VRAM telemetry
8. Output saved to experiments/baseline/baseline_metrics.json
"""

import os
import sys
import math
import time
import json
import statistics
import torch
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
SPARSE_DIR = os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")
ATTN_DIR = os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, SPARSE_DIR, ATTN_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken
from jarvis_model import Jarvis

FIXED_BENCHMARK_PROMPTS = [
    {
        "id": "code_algo",
        "category": "Algorithm Code Generation",
        "prompt": "def binary_search(arr, target):\n    \"\"\"Find target in sorted array arr using binary search.\"\"\"\n"
    },
    {
        "id": "sys_text",
        "category": "Technical Factual Completion",
        "prompt": "In computer systems, virtual memory provides"
    },
    {
        "id": "reasoning_logic",
        "category": "Logical Reasoning",
        "prompt": "Question: If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly? Answer:"
    },
    {
        "id": "code_stdlib",
        "category": "Python Standard Library",
        "prompt": "import os\nimport sys\n\ndef get_file_stats(filepath):\n"
    },
    {
        "id": "stability_list",
        "category": "Repetition & Stability",
        "prompt": "The following is a list of distinct programming languages and their primary paradigms:\n1."
    }
]


def decoupled_forward(model, idx, targets=None, persist_state=False):
    """
    Evaluates Jarvis forward pass with decoupled loss computation:
    Returns: logits, total_loss, ce_loss, l_bal_total, l_ref_total
    """
    B, T = idx.shape
    x = model.tok_emb(idx)

    start_pos = model._token_pos if persist_state else 0
    if persist_state:
        model._token_pos += T

    l_bal_total = torch.tensor(0.0, device=idx.device)
    l_ref_total = torch.tensor(0.0, device=idx.device)

    h_prevs = model._h_states if persist_state else [None] * len(model.blocks)
    new_h_states = []

    for i, block in enumerate(model.blocks):
        h_prev = h_prevs[i]
        if h_prev is not None and h_prev.shape[0] != B:
            h_prev = None

        x, h_last, l_bal, l_ref = block(x, h_prev, start_pos=start_pos)
        new_h_states.append(h_last.detach())
        l_bal_total = l_bal_total + l_bal
        l_ref_total = l_ref_total + l_ref

    model._h_states = new_h_states
    x = model.final_norm(x)
    logits = model.lm_head(x)

    total_loss, ce_loss = None, None
    if targets is not None:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss = ce_loss + l_bal_total + l_ref_total

    return logits, total_loss, ce_loss, l_bal_total, l_ref_total


@torch.inference_mode()
def evaluate_corpus_decoupled(model, enc, file_path, num_windows=200, seq_len=256, seed=42, label="HOLDOUT"):
    print(f"[{label}] Evaluating {file_path} across {num_windows} windows (seq_len={seq_len}, seed={seed})...")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tokens = torch.tensor(enc.encode(text), dtype=torch.long, device="cuda")
    total_tokens = len(tokens)

    max_start = total_tokens - seq_len - 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    if total_tokens >= num_windows * (seq_len + 1):
        window_starts = [i * seq_len for i in range(num_windows)]
    else:
        window_starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()

    ce_losses = []
    bal_losses = []
    ref_losses = []
    total_losses = []

    model.eval()
    model.reset_state()

    for s in window_starts:
        x = tokens[s : s + seq_len].unsqueeze(0)
        y = tokens[s + 1 : s + seq_len + 1].unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, tot_l, ce_l, b_l, r_l = decoupled_forward(model, x, targets=y, persist_state=False)

        ce_losses.append(ce_l.item())
        bal_losses.append(b_l.item())
        ref_losses.append(r_l.item())
        total_losses.append(tot_l.item())

    mean_ce = statistics.mean(ce_losses)
    mean_bal = statistics.mean(bal_losses)
    mean_ref = statistics.mean(ref_losses)
    mean_tot = statistics.mean(total_losses)
    ppl = math.exp(min(mean_ce, 100.0))

    print(f"  -> Mean CE Loss:        {mean_ce:.4f}")
    print(f"  -> Pure Perplexity:     {ppl:.2f}")
    print(f"  -> MoE Balance Loss:    {mean_bal:.4f}")
    print(f"  -> Reflective Penalty:  {mean_ref:.4f}")
    print(f"  -> Total Combined Loss: {mean_tot:.4f}")

    return {
        "file": file_path,
        "total_tokens": total_tokens,
        "windows_evaluated": len(ce_losses),
        "mean_ce_loss": mean_ce,
        "perplexity": ppl,
        "mean_bal_loss": mean_bal,
        "mean_ref_loss": mean_ref,
        "mean_total_loss": mean_tot,
        "std_ce_loss": statistics.stdev(ce_losses) if len(ce_losses) > 1 else 0.0,
    }


def generate_text(model, enc, prompt, max_new_tokens=128, temperature=0.7, top_k=50, top_p=0.9, seed=42):
    torch.manual_seed(seed)
    model.eval()
    model.reset_state()

    prompt_tokens = enc.encode(prompt)
    curr_tokens = list(prompt_tokens)
    input_tensor = torch.tensor([curr_tokens], dtype=torch.long, device="cuda")

    # Prefill
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _, _, _, _ = decoupled_forward(model, input_tensor, persist_state=True)
    next_logits = logits[0, -1, :]

    gen_tokens = []
    for _ in range(max_new_tokens):
        if temperature == 0.0:
            next_token = torch.argmax(next_logits).item()
        else:
            scaled = next_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < v[-1]] = -float("Inf")
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled[indices_to_remove] = -float("Inf")

            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        gen_tokens.append(next_token)
        inp = torch.tensor([[next_token]], dtype=torch.long, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _, _, _ = decoupled_forward(model, inp, persist_state=True)
        next_logits = logits[0, -1, :]

    model.reset_state()
    return enc.decode(gen_tokens)


def measure_throughput(model, enc):
    print("\nMeasuring inference throughput & latency...")
    model.eval()
    model.reset_state()

    # Prefill B=1, T=256
    x = torch.randint(0, 50257, (1, 256), device="cuda")
    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = decoupled_forward(model, x, persist_state=False)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_iters = 30
    for _ in range(n_iters):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = decoupled_forward(model, x, persist_state=False)
    torch.cuda.synchronize()
    prefill_dt = (time.perf_counter() - t0) / n_iters
    prefill_tok_s = 256 / prefill_dt

    # Decode throughput (B=1, autoregressive step)
    model.reset_state()
    x_init = torch.randint(0, 50257, (1, 32), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = decoupled_forward(model, x_init, persist_state=True)

    single_tok = torch.tensor([[100]], device="cuda")
    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = decoupled_forward(model, single_tok, persist_state=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_decode = 100
    for _ in range(n_decode):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = decoupled_forward(model, single_tok, persist_state=True)
    torch.cuda.synchronize()
    decode_dt = (time.perf_counter() - t0) / n_decode
    decode_tok_s = 1.0 / decode_dt

    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    reserv_gb = torch.cuda.memory_reserved() / 1024**3
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    print(f"  Prefill Throughput (B=1, T=256): {prefill_tok_s:.1f} tok/s ({prefill_dt*1000:.2f} ms/pass)")
    print(f"  Decode Throughput (B=1, T=1):     {decode_tok_s:.1f} tok/s ({decode_dt*1000:.2f} ms/token)")
    print(f"  Allocated VRAM:                  {alloc_gb:.2f} GB")
    print(f"  Reserved VRAM:                   {reserv_gb:.2f} GB")
    print(f"  Peak VRAM:                       {peak_gb:.2f} GB")

    return {
        "prefill_tok_s": prefill_tok_s,
        "prefill_latency_ms": prefill_dt * 1000,
        "decode_tok_s": decode_tok_s,
        "decode_latency_ms": decode_dt * 1000,
        "allocated_vram_gb": alloc_gb,
        "reserved_vram_gb": reserv_gb,
        "peak_vram_gb": peak_gb,
    }


def main():
    print("=" * 80)
    print("      JARVIS 606M HARD BASELINE AUDIT (ckpt_step_0004209.pt)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA is required for baseline audit"

    ckpt_path = os.path.join(JARVIS_ENGINE, "ckpt_step_0004209.pt")
    assert os.path.exists(ckpt_path), f"Baseline checkpoint not found at {ckpt_path}"

    enc = tiktoken.get_encoding("gpt2")

    # 1. Instantiate Model
    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=256,
        use_cuda_attn=True,
        use_cuda_moe=True,
    ).to(device)

    # 2. Strict load verification
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                      for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, f"Strict load failed: missing={missing}, unexpected={unexpected}"
    step = ckpt.get("step", 4209)
    print(f"[OK] Checkpoint loaded strictly: step={step}, missing={len(missing)}, unexpected={len(unexpected)}")

    # 3. Parameter count analysis
    total_params = sum(p.numel() for p in model.parameters())
    emb_params = sum(p.numel() for p in model.tok_emb.parameters())
    head_params = sum(p.numel() for p in model.lm_head.parameters())
    expert_params = sum(sum(p.numel() for p in b.moe.w1.parameters()) + sum(p.numel() for p in b.moe.w2.parameters()) for b in model.blocks)
    router_params = sum(sum(p.numel() for p in b.moe.router.parameters()) for b in model.blocks)
    attn_params = sum(sum(p.numel() for p in b.attn.parameters()) for b in model.blocks)
    active_params_per_token = (total_params - expert_params) + (expert_params * 2 // 4)

    print(f"\nParameter Count Breakdown:")
    print(f"  Total Parameters:           {total_params:,}")
    print(f"  Embedding Parameters:       {emb_params:,}")
    print(f"  LM Head Parameters:         {head_params:,}")
    print(f"  Total Expert Parameters:    {expert_params:,} (4 experts x 24 layers)")
    print(f"  Router Parameters:          {router_params:,}")
    print(f"  Attention Parameters:       {attn_params:,}")
    print(f"  Active Parameters / Token:  {active_params_per_token:,} (Top-2 active)")

    backend_status = model.get_backend_status()
    print(f"\nBackends: Attn={backend_status['attn_backend'].upper()} ({backend_status['attn_cuda_blocks']}), MoE={backend_status['moe_backend'].upper()} ({backend_status['moe_cuda_blocks']})")

    # 4. Decoupled Holdout Evaluation (fresh_holdout.txt)
    val_file = os.path.join(JARVIS_ENGINE, "fresh_holdout.txt")
    val_metrics_200 = evaluate_corpus_decoupled(model, enc, val_file, num_windows=200, seq_len=256, seed=42, label="HOLDOUT-200")
    val_metrics_50  = evaluate_corpus_decoupled(model, enc, val_file, num_windows=50,  seq_len=256, seed=42, label="HOLDOUT-50")

    # 5. Decoupled Train Evaluation (data.txt)
    train_file = os.path.join(JARVIS_ENGINE, "data.txt")
    train_metrics_50 = evaluate_corpus_decoupled(model, enc, train_file, num_windows=50, seq_len=256, seed=42, label="TRAIN-50")

    # 6. Throughput & VRAM
    tp_metrics = measure_throughput(model, enc)

    # 7. Deterministic Benchmark Generation
    print("\nRunning deterministic benchmark generations across 5 fixed prompts...")
    gen_results = []
    for bp in FIXED_BENCHMARK_PROMPTS:
        print(f"\nPrompt [{bp['id']} - {bp['category']}]:")
        print(f"  Prompt Text: {repr(bp['prompt'])}")
        greedy_out = generate_text(model, enc, bp["prompt"], max_new_tokens=128, temperature=0.0, seed=42)
        sample_out = generate_text(model, enc, bp["prompt"], max_new_tokens=128, temperature=0.7, top_k=50, top_p=0.9, seed=42)
        print(f"  Greedy (T=0.0): {repr(greedy_out[:120])}...")
        print(f"  Sample (T=0.7): {repr(sample_out[:120])}...")
        gen_results.append({
            "id": bp["id"],
            "category": bp["category"],
            "prompt": bp["prompt"],
            "greedy_generation": greedy_out,
            "sample_generation": sample_out,
        })

    # 8. Compile Complete Baseline Metrics
    baseline_payload = {
        "checkpoint": os.path.basename(ckpt_path),
        "step": step,
        "strict_load_verified": True,
        "parameter_counts": {
            "total_params": total_params,
            "active_params_per_token": active_params_per_token,
            "embedding_params": emb_params,
            "lm_head_params": head_params,
            "expert_params": expert_params,
            "router_params": router_params,
            "attn_params": attn_params,
        },
        "backend_status": backend_status,
        "holdout_200_windows": val_metrics_200,
        "holdout_50_windows": val_metrics_50,
        "train_50_windows": train_metrics_50,
        "throughput_and_vram": tp_metrics,
        "generations": gen_results,
    }

    out_dir = os.path.join(WORKSPACE_ROOT, "experiments", "baseline")
    metrics_path = os.path.join(out_dir, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)
    print(f"\n[OK] Baseline metrics saved to {metrics_path}")

    # Text format of generations
    gen_text_path = os.path.join(out_dir, "baseline_generation.txt")
    with open(gen_text_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("JARVIS 606M BASELINE DETERMINISTIC GENERATION SAMPLES (Step 4209)\n")
        f.write("=" * 80 + "\n\n")
        for g in gen_results:
            f.write(f"--- [{g['id'].upper()}] {g['category']} ---\n")
            f.write(f"Prompt:\n{g['prompt']}\n\n")
            f.write(f"Greedy Output (T=0.0):\n{g['greedy_generation']}\n\n")
            f.write(f"Sampled Output (T=0.7, top_k=50, top_p=0.9):\n{g['sample_generation']}\n\n")
            f.write("-" * 80 + "\n\n")
    print(f"[OK] Baseline generation samples saved to {gen_text_path}")


if __name__ == "__main__":
    main()
