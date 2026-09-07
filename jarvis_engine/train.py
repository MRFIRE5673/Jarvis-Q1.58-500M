import os, math, signal, sys, time, glob
import torch

torch.backends.cuda.matmul.allow_tf32 = True   # faster bf16-equivalent matmuls
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False          # True crashes cuDNN with OOM when optimizer states loaded
torch.cuda.set_per_process_memory_fraction(0.92)  # ~10.74 GB hard cap

import tiktoken
from jarvis_model import Jarvis

# ---------------------------------------------------------------------------
# Checkpoint helpers — numbered filenames, ZERO os.rename calls
# Windows Defender locks .pt files for 10-60s after write, making os.rename
# impossible.  Writing to a *brand-new* filename sidesteps this entirely.
# ---------------------------------------------------------------------------
CKPT_DIR    = "."
CKPT_PREFIX = "ckpt_step_"
CKPT_KEEP   = 3            # keep the N most-recent checkpoints on disk

def _ckpt_path(step):
    return os.path.join(CKPT_DIR, f"{CKPT_PREFIX}{step:07d}.pt")

def save_checkpoint(step, model, optimizer):
    """Save to a new numbered file — no rename, no WinError 32."""
    path = _ckpt_path(step)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  -> checkpoint saved: {os.path.basename(path)}", flush=True)
    # Purge old checkpoints beyond the keep window
    all_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, f"{CKPT_PREFIX}*.pt")))
    for old in all_ckpts[:-CKPT_KEEP]:
        try:
            os.remove(old)
        except OSError:
            pass   # AV still scanning — skip, next save will clean it up

def load_best_checkpoint():
    """Find the newest numbered checkpoint; load to CPU to avoid VRAM double-copy."""
    # Try numbered checkpoints first (newest → oldest)
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
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                print(f"Loaded legacy checkpoint {path} (step {ckpt['step']})", flush=True)
                return ckpt
            except Exception as e:
                print(f"  [warn] {path} unreadable ({e}), trying next…", flush=True)
    return None

# --- Model setup ---
model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
               num_experts=4, top_k=2, max_seq_len=256).cuda()

n, s = model.param_count()
print(s, flush=True)

# --- Resume from checkpoint if one exists ---
start_step = 0
ckpt = load_best_checkpoint()
if ckpt is not None:
    state_dict = ckpt["model_state_dict"]
    # Strip torch.compile's _orig_mod. prefix if present
    new_state_dict = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                      for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  [ckpt] fresh-init keys (new arch): {len(missing)}", flush=True)
    if unexpected:
        print(f"  [ckpt] ignored old keys: {len(unexpected)}", flush=True)
    start_step = ckpt["step"] + 1

# --- Data setup ---
enc = tiktoken.get_encoding("gpt2")
with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
tokens = torch.tensor(enc.encode(text), dtype=torch.long).cuda()   # GPU-resident
print(f"Dataset: {len(tokens)} tokens", flush=True)

# GPU-resident advanced-indexing batch sampler — zero CPU↔GPU transfers per step
_offsets = torch.arange(256, device="cuda")   # reusable offset vector

def get_batch(batch_size=1, seq_len=256):
    ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,), device=tokens.device)
    idx = ix.unsqueeze(1) + _offsets[:seq_len]   # (B, seq_len)
    x = tokens[idx]
    y = tokens[idx + 1]
    return x, y

# --- Optimizer: fused AdamW (single CUDA kernel per step, no Python loop) ---
BASE_LR      = 3e-4
WARMUP_STEPS = 500
TOTAL_STEPS  = 5000
SAVE_EVERY   = 50
BATCH_SIZE   = 2
ACCUM_STEPS  = 4   # Effective batch = 8 sequences = 2048 tokens/update

try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, fused=True)
    print("Using fused AdamW", flush=True)
except TypeError:
    # fused= not available in older PyTorch builds
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    print("Using standard AdamW", flush=True)

# Optimizer state intentionally NOT loaded from checkpoint.
# Adam m1+m2 tensors (~4.85 GB FP32) + model weights + backward activations
# exceed the 12 GB VRAM budget on RTX 5070 Windows.
# Model weights ARE restored above — Adam reinitialises lazily on first step.
# Expect ~5-10 noisy steps then full recovery.
if ckpt is not None:
    print("  [ckpt] optimizer state skipped (VRAM budget) — Adam starts fresh", flush=True)

# Free the CPU-side checkpoint dict — model state dict already copied into GPU buffers.
if ckpt is not None:
    del ckpt
torch.cuda.empty_cache()

# --- Graceful exit on Ctrl+C or SIGTERM ---
_current_step = start_step
_saving = False   # guard against re-entrance during save

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
    # Cosine decay to 10% of peak LR
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return BASE_LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

# --- Training loop ---
for step in range(start_step, TOTAL_STEPS):
    t0 = time.perf_counter()
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Gradient accumulation: accumulate ACCUM_STEPS micro-batches before update
    optimizer.zero_grad(set_to_none=True)
    loss_accum = torch.zeros((), device="cuda")
    for _ in range(ACCUM_STEPS):
        x, y = get_batch(BATCH_SIZE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
        (loss / ACCUM_STEPS).backward()
        loss_accum += loss.detach() / ACCUM_STEPS

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    _current_step = step

    if step % 10 == 0:
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_s = (BATCH_SIZE * ACCUM_STEPS * 256) / max(dt, 1e-4)
        alloc  = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved()  / 1024**3
        print(f"step {step:05d}: loss {loss_accum.item():.4f} | lr {lr:.6f} | {dt:.2f}s ({tok_s:.0f} tok/s) "
              f"| VRAM {alloc:.2f}/{reserv:.2f} GB", flush=True)

    if step % SAVE_EVERY == 0 and step > 0:
        save_checkpoint(step, model, optimizer)

print("Training complete.", flush=True)
save_checkpoint(TOTAL_STEPS - 1, model, optimizer)