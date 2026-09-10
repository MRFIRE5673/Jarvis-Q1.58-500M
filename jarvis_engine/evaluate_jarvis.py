# evaluate_jarvis.py
"""
Jarvis 606M Production Evaluation & Interactive Inference Suite
===============================================================
Features:
  1. Interactive terminal CLI (`--interactive`) with streaming token output
  2. Direct prompt evaluation (`--prompt "..."`)
  3. Strict 606M checkpoint loading with full backend inspection
  4. Generation controls: --max_new_tokens, --temperature, --top_k, --top_p
  5. State isolation (`model.reset_state()` per prompt) and slash commands (/reset, /info, /quit)
  6. Quantitative holdout corpus evaluation & perplexity tracking
"""

import os
import sys
import math
import time
import argparse
import statistics
import torch
import torch.nn.functional as F
import tiktoken

# Ensure repo paths are in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JARVIS_ENGINE_PATH = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
SPARSE_DIR = os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")
ATTN_DIR = os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE_PATH, SPARSE_DIR, ATTN_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import Jarvis


def resolve_path(path_arg, default_name=None):
    """Resolves relative file paths whether running from root or jarvis_engine/."""
    if path_arg and os.path.exists(path_arg):
        return os.path.abspath(path_arg)
    candidates = []
    if path_arg:
        candidates.append(path_arg)
        candidates.append(os.path.join(JARVIS_ENGINE_PATH, path_arg))
        candidates.append(os.path.join(WORKSPACE_ROOT, path_arg))
    if default_name:
        candidates.append(default_name)
        candidates.append(os.path.join(JARVIS_ENGINE_PATH, default_name))
        candidates.append(os.path.join(WORKSPACE_ROOT, default_name))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return path_arg or default_name


def parse_args():
    parser = argparse.ArgumentParser(description="Jarvis 606M Production Evaluation and Interactive Suite")
    # Interactive / Generation controls
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Launch interactive conversation terminal")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to evaluate directly")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate per response (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0.0 for greedy decoding, default: 0.7)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering threshold (default: 50, <=0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus top-p filtering threshold (default: 0.9, 1.0 to disable)")
    parser.add_argument("--ckpt", type=str, default="ckpt_step_0004209.pt",
                        help="Path to model checkpoint")

    # Quantitative corpus evaluation arguments
    parser.add_argument("--eval-file", type=str, default="fresh_holdout.txt",
                        help="Path to genuine holdout text file (must NOT be data.txt)")
    parser.add_argument("--train-file", type=str, default="data.txt",
                        help="Path to training corpus file for comparison")
    parser.add_argument("--num-windows", type=int, default=200,
                        help="Number of evaluation windows (default: 200)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length of each evaluation window (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic window selection")
    parser.add_argument("--skip-train-eval", action="store_true",
                        help="Skip training corpus evaluation and only run holdout")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip text generation evaluation")
    return parser.parse_args()


def load_jarvis_model(ckpt_path, device="cuda"):
    resolved_ckpt = resolve_path(ckpt_path, "ckpt_step_0004209.pt")
    if not os.path.exists(resolved_ckpt):
        raise FileNotFoundError(f"Checkpoint file '{ckpt_path}' not found (searched '{resolved_ckpt}').")

    print("=" * 80)
    print("                    LOADING PRODUCTION JARVIS MODEL                     ")
    print("=" * 80)
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
    print(f"Model Architecture: 606M Jarvis ({n_params:,} parameters, {param_str})")
    print(f"Loading checkpoint from: {resolved_ckpt}...")
    ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                      for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"

    trained_step = ckpt.get("step", "unknown")
    print(f"Checkpoint loaded successfully: strict=True verified (trained step: {trained_step})")

    status = model.get_backend_status()
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Attention Backend:  {status['attn_backend'].upper()} ({status['attn_cuda_blocks']} blocks)")
    print(f"MoE Backend:        {status['moe_backend'].upper()} ({status['moe_cuda_blocks']} blocks)")
    print(f"Inference Device:   {device} ({device_name})")
    print(f"Inference Dtype:    torch.bfloat16 (with native FP32 accumulations)")
    print("=" * 80 + "\n")

    del ckpt, state_dict, new_state_dict
    torch.cuda.empty_cache()
    model.eval()
    return model, resolved_ckpt, trained_step


def print_model_info(model, ckpt_path, device="cuda"):
    status = model.get_backend_status()
    n_params, param_str = model.param_count()
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3) if device == "cuda" else 0.0
    vram_res = torch.cuda.memory_reserved() / (1024 ** 3) if device == "cuda" else 0.0

    print("\n--- Model & System Information ---")
    print(f"Checkpoint:          {ckpt_path}")
    print(f"Parameter Count:     {n_params:,} ({param_str})")
    print(f"Attention Backend:   {status['attn_backend'].upper()} ({status['attn_cuda_blocks']} blocks)")
    print(f"Sparse MoE Backend:  {status['moe_backend'].upper()} ({status['moe_cuda_blocks']} blocks)")
    print(f"Device:              {device} ({device_name})")
    print(f"Current VRAM Usage:  {vram_alloc:.2f} GB allocated / {vram_res:.2f} GB reserved")
    print(f"Liquid State Status: {sum(1 for h in model._h_states if h is not None)}/{len(model.blocks)} blocks active")
    print(f"Token Position:      {model._token_pos}")
    print("----------------------------------\n")


def sample_next_token(logits, temperature=0.7, top_k=50, top_p=0.9):
    """Samples next token with temperature, top-k, and nucleus top-p filtering."""
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-5)

    # Top-K filtering
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

    # Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any():
        return torch.argmax(logits, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate_streaming(
    model,
    enc,
    prompt_text: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda"
):
    """
    Generates text autoregressively and streams tokens to stdout in real-time.
    Resets model state before generation to prevent state leakage.
    Measures synchronized timing for accurate tok/s calculation.
    """
    model.eval()
    # Reset model state before prompt execution to prevent cross-prompt state pollution
    model.reset_state()

    prompt_ids = enc.encode(prompt_text)
    if len(prompt_ids) == 0:
        prompt_ids = [enc.eot_token]
    curr_tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_token_ids = []
    prev_decoded_len = 0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    for step_idx in range(max_new_tokens):
        # Jarvis sliding context window (supports sequence lengths up to 256)
        context = curr_tokens[:, -256:] if curr_tokens.size(1) > 256 else curr_tokens

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(context, persist_state=False)

        next_logits = logits[:, -1, :].float()
        next_tok = sample_next_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        tok_id = next_tok.item()

        # Check for EOS token
        if tok_id == enc.eot_token:
            break

        generated_token_ids.append(tok_id)
        curr_tokens = torch.cat([curr_tokens, next_tok], dim=1)

        # Stream decoded token delta to stdout
        full_gen_text = enc.decode(generated_token_ids)
        new_text = full_gen_text[prev_decoded_len:]
        sys.stdout.write(new_text)
        sys.stdout.flush()
        prev_decoded_len = len(full_gen_text)

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    sys.stdout.write("\n")
    sys.stdout.flush()

    duration = t_end - t_start
    n_tokens = len(generated_token_ids)
    tok_s = n_tokens / max(duration, 1e-5)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if device == "cuda" else 0.0

    stats = {
        "tokens_generated": n_tokens,
        "generation_time_s": duration,
        "tokens_per_sec": tok_s,
        "peak_vram_gb": peak_vram,
    }

    print("-" * 50)
    print(f"Tokens Generated:  {n_tokens}")
    print(f"Generation Time:   {duration:.3f} s")
    print(f"Throughput:        {tok_s:.1f} tok/s")
    print(f"Peak VRAM:         {peak_vram:.2f} GB")
    print("-" * 50)

    # Clean model state after response completion
    model.reset_state()
    return enc.decode(generated_token_ids), stats


def interactive_terminal(model, enc, args, ckpt_path, device="cuda"):
    """Interactive CMD/Terminal REPL loop."""
    print("================================================================================")
    print("                   JARVIS 606M INTERACTIVE TERMINAL                             ")
    print("================================================================================")
    print("Commands:")
    print("  /reset  - Clear model liquid state and position offsets")
    print("  /info   - Display current model architecture, backends, and memory usage")
    print("  /quit   - Exit the interactive terminal (or /exit)")
    print(f"Controls: max_new_tokens={args.max_new_tokens}, temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("Type your prompt and press Enter to stream completions in real-time.")
    print("================================================================================\n")

    while True:
        try:
            prompt = input("Jarvis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive terminal.")
            break

        if not prompt:
            continue

        cmd = prompt.lower()
        if cmd in ["/quit", "/exit"]:
            print("Exiting Jarvis terminal. Goodbye!")
            break
        elif cmd == "/reset":
            model.reset_state()
            print("  [State] Model liquid membrane states and positional offsets cleared.")
            continue
        elif cmd == "/info":
            print_model_info(model, ckpt_path, device)
            continue

        print("\nResponse:")
        generate_streaming(
            model=model,
            enc=enc,
            prompt_text=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )


@torch.no_grad()
def evaluate_corpus(model, file_path, enc, num_windows=200, seq_len=256, seed=42, label="HOLDOUT", device="cuda"):
    resolved_file = resolve_path(file_path)
    if not os.path.exists(resolved_file):
        raise FileNotFoundError(f"CRITICAL ERROR: {label} file '{file_path}' does not exist! "
                                f"Silently falling back to training data is strictly prohibited.")

    with open(resolved_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = torch.tensor(enc.encode(text), dtype=torch.long, device=device)
    total_tokens_in_file = len(tokens)

    required_tokens = seq_len + 1
    if total_tokens_in_file < required_tokens:
        raise ValueError(f"Corpus '{file_path}' has only {total_tokens_in_file} tokens, less than required {required_tokens}.")

    max_start = total_tokens_in_file - seq_len - 1
    if total_tokens_in_file >= num_windows * (seq_len + 1):
        window_starts = [i * seq_len for i in range(num_windows)]
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        window_starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()

    losses = []
    has_nan_or_inf = False

    for start_idx in window_starts:
        x = tokens[start_idx : start_idx + seq_len].unsqueeze(0)
        y = tokens[start_idx + 1 : start_idx + seq_len + 1].unsqueeze(0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, targets=y, persist_state=False)

        val = loss.item()
        if math.isnan(val) or math.isinf(val):
            has_nan_or_inf = True
        losses.append(val)

    mean_loss = statistics.mean(losses)
    std_loss = statistics.stdev(losses) if len(losses) > 1 else 0.0
    ppl = math.exp(min(mean_loss, 100.0))
    total_eval_tokens = num_windows * seq_len

    return {
        "label": label,
        "file": resolved_file,
        "corpus_total_tokens": total_tokens_in_file,
        "windows_evaluated": len(losses),
        "tokens_evaluated": total_eval_tokens,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "perplexity": ppl,
        "has_nan_inf": has_nan_or_inf,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    enc = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load production model
    model, ckpt_path, trained_step = load_jarvis_model(args.ckpt, device=device)

    # 2. Interactive Terminal Mode
    if args.interactive:
        interactive_terminal(model, enc, args, ckpt_path, device=device)
        return

    # 3. Single Prompt Evaluation Mode
    if args.prompt:
        print(f"Prompt: {args.prompt}\n")
        print("Response:")
        generate_streaming(
            model=model,
            enc=enc,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        return

    # 4. Standard Corpus Evaluation Mode
    eval_file_resolved = resolve_path(args.eval_file, "fresh_holdout.txt")
    train_file_resolved = resolve_path(args.train_file, "data.txt")

    print("=" * 90)
    print(f"  JARVIS 606M POST-TRAINING CORPUS EVALUATION")
    print(f"  Target Checkpoint: {ckpt_path} (Step {trained_step})")
    print(f"  Holdout Corpus:    {eval_file_resolved}")
    print(f"  Training Corpus:   {train_file_resolved}")
    print(f"  Eval Windows:      {args.num_windows} windows x {args.seq_len} tokens")
    print("=" * 90)

    train_res = None
    if not args.skip_train_eval and os.path.exists(train_file_resolved):
        print(f"\nEvaluating TRAINING CORPUS: {train_file_resolved}...", flush=True)
        train_res = evaluate_corpus(model, train_file_resolved, enc,
                                    num_windows=args.num_windows, seq_len=args.seq_len,
                                    seed=args.seed, label="TRAINING LOSS", device=device)

    print(f"\nEvaluating HOLDOUT CORPUS: {eval_file_resolved}...", flush=True)
    holdout_res = evaluate_corpus(model, eval_file_resolved, enc,
                                  num_windows=args.num_windows, seq_len=args.seq_len,
                                  seed=args.seed, label="HOLDOUT LOSS", device=device)

    print("\n" + "=" * 90)
    print("  QUANTITATIVE EVALUATION SUMMARY")
    print("=" * 90)
    header = f"{'Metric':<32} | {'TRAINING LOSS':<28} | {'HOLDOUT LOSS':<26}"
    print(header)
    print("-" * 90)
    t_mean = f"{train_res['mean_loss']:.4f}" if train_res else "N/A"
    h_mean = f"{holdout_res['mean_loss']:.4f}"
    print(f"{'Mean Cross-Entropy Loss':<32} | {t_mean:<28} | {h_mean:<26}")
    t_ppl = f"{train_res['perplexity']:.2f}" if train_res else "N/A"
    h_ppl = f"{holdout_res['perplexity']:.2f}"
    print(f"{'Perplexity (exp(loss))':<32} | {t_ppl:<28} | {h_ppl:<26}")
    print("=" * 90)


if __name__ == "__main__":
    main()
