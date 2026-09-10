import os
import sys
import math
import signal
import time
import glob
import argparse
import statistics
import torch

torch.backends.cuda.matmul.allow_tf32 = True   # faster bf16-equivalent matmuls
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False          # True crashes cuDNN with OOM when optimizer states loaded
torch.cuda.set_per_process_memory_fraction(0.92)  # ~10.74 GB hard cap

# Ensure jarvis_engine and CUDA extensions are reachable from sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [WORKSPACE_ROOT, SCRIPT_DIR, os.path.join(WORKSPACE_ROOT, "sparse_model_cuda"), os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken
from jarvis_model import Jarvis

# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------
def resolve_file(path_arg, default_name):
    if path_arg and os.path.exists(path_arg):
        return os.path.abspath(path_arg)
    candidates = [
        path_arg,
        os.path.join(SCRIPT_DIR, path_arg) if path_arg else None,
        os.path.join(WORKSPACE_ROOT, path_arg) if path_arg else None,
        os.path.join(SCRIPT_DIR, default_name),
        os.path.join(WORKSPACE_ROOT, default_name),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return path_arg or os.path.join(SCRIPT_DIR, default_name)

# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Jarvis 606M Production Training Loop")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of steps to run in this execution (default: continue to total_steps)")
    parser.add_argument("--total-steps", type=int, default=5000,
                        help="Total steps in LR schedule (default: 5000)")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps in LR schedule (default: 500)")
    parser.add_argument("--train-file", type=str, default="data.txt",
                        help="Path to training corpus file (default: data.txt)")
    parser.add_argument("--val-file", type=str, default="fresh_holdout.txt",
                        help="Path to holdout validation file (default: fresh_holdout.txt)")
    parser.add_argument("--val-every", type=int, default=25,
                        help="Evaluate validation loss/perplexity every N steps (default: 25)")
    parser.add_argument("--val-windows", type=int, default=50,
                        help="Number of holdout windows to evaluate (default: 50)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to specific checkpoint to resume from (default: latest checkpoint)")
    parser.add_argument("--ckpt-keep", type=int, default=3,
                        help="Number of recent checkpoints to retain (default: 3)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Base learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Micro-batch size per forward call (default: 2)")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps per update (default: 4)")
    return parser.parse_args()

args = parse_args()

# ---------------------------------------------------------------------------
# Checkpoint helpers — numbered filenames, immutable baseline protection
# ---------------------------------------------------------------------------
CKPT_DIR    = SCRIPT_DIR
CKPT_PREFIX = "ckpt_step_"
CKPT_KEEP   = args.ckpt_keep

def _ckpt_path(step):
    return os.path.join(CKPT_DIR, f"{CKPT_PREFIX}{step:07d}.pt")

def save_checkpoint(step, model, optimizer, val_loss=None, val_ppl=None):
    """Save to a new numbered file — excludes baseline checkpoints from purge."""
    path = _ckpt_path(step)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if val_loss is not None:
        payload["val_loss"] = float(val_loss)
    if val_ppl is not None:
        payload["val_perplexity"] = float(val_ppl)
    torch.save(payload, path)
    print(f"  -> checkpoint saved: {os.path.basename(path)}", flush=True)

    # Purge old checkpoints beyond the keep window, NEVER touching baseline checkpoints
    all_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, f"{CKPT_PREFIX}*.pt")))
    purge_candidates = [
        c for c in all_ckpts[:-CKPT_KEEP]
        if "baseline" not in os.path.basename(c) and "0004209" not in os.path.basename(c)
    ]
    for old in purge_candidates:
        try:
            os.remove(old)
        except OSError:
            pass   # AV still scanning — skip, next save will clean it up

def load_best_checkpoint(explicit_path=None):
    """Find the newest numbered checkpoint or load explicit_path; load to CPU."""
    if explicit_path:
        full_p = resolve_file(explicit_path, explicit_path)
        if os.path.exists(full_p):
            ckpt = torch.load(full_p, map_location="cpu", weights_only=False)
            print(f"Loaded specified checkpoint from {os.path.basename(full_p)} (step {ckpt['step']})", flush=True)
            return ckpt
        else:
            raise FileNotFoundError(f"Requested resume checkpoint not found: {explicit_path}")

    all_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, f"{CKPT_PREFIX}*.pt")))
    for path in reversed(all_ckpts):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            print(f"Loaded checkpoint from {os.path.basename(path)} (step {ckpt['step']})", flush=True)
            return ckpt
        except Exception as e:
            print(f"  [warn] {os.path.basename(path)} unreadable ({e}), trying next…", flush=True)
    # Fallback: legacy named checkpoints from the old save scheme
    for path in ["checkpoint_latest.pt", "checkpoint_prev.pt", "checkpoint_old.pt"]:
        full_p = os.path.join(CKPT_DIR, path)
        if os.path.exists(full_p):
            try:
                ckpt = torch.load(full_p, map_location="cpu", weights_only=False)
                print(f"Loaded legacy checkpoint {path} (step {ckpt['step']})", flush=True)
                return ckpt
            except Exception as e:
                print(f"  [warn] {path} unreadable ({e}), trying next…", flush=True)
    return None

# --- Model setup ---
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
).cuda()

n, s = model.param_count()
status = model.get_backend_status()
print(f"Jarvis Model: {s} | Backends: Attn={status['attn_backend'].upper()} ({status['attn_cuda_blocks']}), MoE={status['moe_backend'].upper()} ({status['moe_cuda_blocks']})", flush=True)

# --- Resume from checkpoint if one exists ---
start_step = 0
ckpt = load_best_checkpoint(args.resume_ckpt)
if ckpt is not None:
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                      for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    print(f"  [ckpt] strict=True load verified: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    start_step = ckpt["step"] + 1

# --- Dataset setup ---
enc = tiktoken.get_encoding("gpt2")

train_file_path = resolve_file(args.train_file, "data.txt")
print(f"Loading training data from: {train_file_path}...", flush=True)
with open(train_file_path, "r", encoding="utf-8", errors="ignore") as f:
    train_text = f.read()
tokens = torch.tensor(enc.encode(train_text), dtype=torch.long).cuda()
print(f"Training dataset: {len(tokens):,} tokens GPU-resident", flush=True)

# Holdout validation dataset setup
val_file_path = resolve_file(args.val_file, "fresh_holdout.txt")
val_tokens = None
if os.path.exists(val_file_path):
    print(f"Loading holdout validation data from: {val_file_path}...", flush=True)
    with open(val_file_path, "r", encoding="utf-8", errors="ignore") as f:
        val_text = f.read()
    val_tokens = torch.tensor(enc.encode(val_text), dtype=torch.long).cuda()
    print(f"Validation dataset: {len(val_tokens):,} tokens GPU-resident", flush=True)
else:
    print(f"  [warn] Validation file '{val_file_path}' not found! Validation metrics will be skipped.", flush=True)

# GPU-resident advanced-indexing batch sampler — zero CPU↔GPU transfers per step
_offsets = torch.arange(256, device="cuda")

def get_batch(batch_size=2, seq_len=256):
    ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,), device=tokens.device)
    idx = ix.unsqueeze(1) + _offsets[:seq_len]
    x = tokens[idx]
    y = tokens[idx + 1]
    return x, y

@torch.inference_mode()
def evaluate_validation(val_toks, num_windows=50, seq_len=256, seed=42):
    """Evaluates validation cross-entropy loss and perplexity on holdout windows."""
    if val_toks is None or len(val_toks) < seq_len + 1:
        return None, None
    model.eval()
    model.reset_state()
    total_len = len(val_toks)
    max_start = total_len - seq_len - 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    window_starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()

    losses = []
    for start_idx in window_starts:
        x = val_toks[start_idx : start_idx + seq_len].unsqueeze(0)
        y = val_toks[start_idx + 1 : start_idx + seq_len + 1].unsqueeze(0)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, targets=y, persist_state=False)
        losses.append(loss.item())

    mean_loss = statistics.mean(losses)
    val_ppl = math.exp(min(mean_loss, 100.0))
    model.train()
    model.reset_state()
    return mean_loss, val_ppl

# --- Optimizer: fused AdamW ---
BASE_LR      = args.lr
WARMUP_STEPS = args.warmup_steps
TOTAL_STEPS  = args.total_steps
SAVE_EVERY   = args.save_every
BATCH_SIZE   = args.batch_size
ACCUM_STEPS  = args.accum_steps
VAL_EVERY    = args.val_every

try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, fused=True)
    print("Using fused AdamW (fused=True)", flush=True)
except TypeError:
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    print("Using standard AdamW", flush=True)

# Optimizer state intentionally NOT loaded from checkpoint to protect 12GB VRAM cap
if ckpt is not None:
    print("  [ckpt] optimizer state intentionally fresh (12GB VRAM budget) — Adam allocates lazily", flush=True)

if ckpt is not None:
    del ckpt
torch.cuda.empty_cache()

# --- Graceful exit on Ctrl+C or SIGTERM ---
_current_step = start_step
_saving = False

def _handle_exit(sig, frame):
    global _saving
    if _saving:
        print("\n[!] Already saving — force quitting.", flush=True)
        sys.exit(1)
    _saving = True
    print(f"\nInterrupted at step {_current_step} — saving checkpoint…", flush=True)
    try:
        save_checkpoint(_current_step, model, optimizer)
    except Exception as e:
        print(f"  [!] Save failed ({e}) — model weights on GPU are lost.", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT,  _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)

# --- LR schedule: linear warmup + cosine decay ---
def get_lr(step):
    if step < WARMUP_STEPS:
        return BASE_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return BASE_LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

# Determine stop step
target_end_step = start_step + args.steps if args.steps is not None else TOTAL_STEPS
print(f"Starting training run: step {start_step} to {target_end_step - 1} ({target_end_step - start_step} updates)", flush=True)

# Initial validation evaluation before step 0
last_val_loss = None
last_val_ppl = None
if val_tokens is not None:
    init_val_loss, init_val_ppl = evaluate_validation(val_tokens, num_windows=args.val_windows)
    last_val_loss = init_val_loss
    last_val_ppl = init_val_ppl
    print(f"[Initial Validation] Step {start_step - 1}: Holdout Loss = {init_val_loss:.4f} | Perplexity = {init_val_ppl:.2f}", flush=True)

# --- Training loop ---
for step in range(start_step, target_end_step):
    t0 = time.perf_counter()
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_accum = torch.zeros((), device="cuda")
    for _ in range(ACCUM_STEPS):
        x, y = get_batch(BATCH_SIZE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
        (loss / ACCUM_STEPS).backward()
        loss_accum += loss.detach() / ACCUM_STEPS

    loss_val = loss_accum.item()
    if math.isnan(loss_val) or math.isinf(loss_val):
        raise FloatingPointError(f"Step {step}: Loss is NaN or Inf ({loss_val})!")

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    norm_val = total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm)
    if math.isnan(norm_val) or math.isinf(norm_val):
        raise FloatingPointError(f"Step {step}: Gradient norm is NaN or Inf ({norm_val})!")

    optimizer.step()
    _current_step = step

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tok_s = (BATCH_SIZE * ACCUM_STEPS * 256) / max(dt, 1e-4)
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3

    # Periodic progress logging
    if step % 5 == 0 or step == start_step or step == target_end_step - 1:
        print(f"step {step:05d}: train_loss {loss_val:.4f} | lr {lr:.6f} | grad_norm {norm_val:.3f} | {dt:.2f}s ({tok_s:.0f} tok/s) | VRAM {alloc:.2f}/{reserv:.2f} GB", flush=True)

    # Periodic validation evaluation
    is_last_step = (step == target_end_step - 1)
    if val_tokens is not None and ((step - start_step + 1) % VAL_EVERY == 0 or is_last_step):
        v_loss, v_ppl = evaluate_validation(val_tokens, num_windows=args.val_windows)
        last_val_loss = v_loss
        last_val_ppl = v_ppl
        print(f"  -> [VALIDATION] step {step:05d}: val_loss {v_loss:.4f} | perplexity {v_ppl:.2f}", flush=True)

    # Periodic checkpoint save
    if (step % SAVE_EVERY == 0 and step > 0) or is_last_step:
        save_checkpoint(step, model, optimizer, val_loss=last_val_loss, val_ppl=last_val_ppl)

print("Execution complete.", flush=True)