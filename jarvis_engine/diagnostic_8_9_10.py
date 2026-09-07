import sys
sys.path.insert(0, ".")
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from jarvis_model import Jarvis

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TEST 8: FULL 606M MODEL SANITY CHECK")
print("=" * 60)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

t_init_start = time.perf_counter()
model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
               num_experts=4, top_k=2, max_seq_len=256).to(device)
t_init = time.perf_counter() - t_init_start

param_count, param_str = model.param_count()
print(f"Instantiated: {param_str} in {t_init:.2f}s")
vram_init_alloc = torch.cuda.memory_allocated() / (1024**3)
vram_init_reserv = torch.cuda.memory_reserved() / (1024**3)
print(f"VRAM after init: Allocated={vram_init_alloc:.2f} GB, Reserved={vram_init_reserv:.2f} GB")

# Test batch: B=2, T=256
x = torch.randint(0, 50257, (2, 256), device=device)
y = torch.randint(0, 50257, (2, 256), device=device)

# Forward pass
torch.cuda.synchronize()
t_fwd_start = time.perf_counter()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits, loss = model(x, targets=y)
torch.cuda.synchronize()
t_fwd = time.perf_counter() - t_fwd_start

vram_fwd_alloc = torch.cuda.memory_allocated() / (1024**3)
vram_fwd_reserv = torch.cuda.memory_reserved() / (1024**3)

loss_val = loss.item()
has_nan = torch.isnan(logits).any().item() or math.isnan(loss_val)
has_inf = torch.isinf(logits).any().item() or math.isinf(loss_val)
logits_mean = logits.mean().item()
logits_std = logits.std().item()

print(f"Forward time: {t_fwd*1000:.1f} ms")
print(f"Loss: {loss_val:.4f}")
print(f"Logits: mean={logits_mean:.4f}, std={logits_std:.4f}, has_nan={has_nan}, has_inf={has_inf}")
print(f"VRAM after forward: Allocated={vram_fwd_alloc:.2f} GB, Reserved={vram_fwd_reserv:.2f} GB")

# Backward pass
torch.cuda.synchronize()
t_bwd_start = time.perf_counter()
loss.backward()
torch.cuda.synchronize()
t_bwd = time.perf_counter() - t_bwd_start

vram_bwd_alloc = torch.cuda.memory_allocated() / (1024**3)
vram_bwd_reserv = torch.cuda.memory_reserved() / (1024**3)

# Gradient norms and non-zero checks
total_params = 0
nonzero_grad_params = 0
for p in model.parameters():
    total_params += p.numel()
    if p.grad is not None and (p.grad != 0).any():
        nonzero_grad_params += p.numel()

pct_nonzero_grad = (nonzero_grad_params / total_params) * 100
grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

print(f"Backward time: {t_bwd*1000:.1f} ms")
print(f"VRAM after backward: Allocated={vram_bwd_alloc:.2f} GB, Reserved={vram_bwd_reserv:.2f} GB")
print(f"Global gradient norm (pre-clip): {grad_norm_before_clip:.4f}")
print(f"Parameters with non-zero gradient: {nonzero_grad_params}/{total_params} ({pct_nonzero_grad:.2f}%)")

test8_pass = (not has_nan) and (not has_inf) and (loss_val > 0) and (pct_nonzero_grad > 90) and (grad_norm_before_clip < 100.0)
print(f"TEST 8 VERDICT: {'PASS' if test8_pass else 'FAIL'}")

# Clean up gradients
model.zero_grad(set_to_none=True)

print("\n" + "=" * 60)
print("TEST 9: SHORT TRAINING SANITY RUN (50 STEPS)")
print("=" * 60)

# Setup data as in train.py
enc = tiktoken.get_encoding("gpt2")
with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
tokens = torch.tensor(enc.encode(text), dtype=torch.long, device=device)
print(f"Dataset resident on GPU: {len(tokens)} tokens")

_offsets = torch.arange(256, device=device)
def get_batch(batch_size=2, seq_len=256):
    ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,), device=tokens.device)
    idx = ix.unsqueeze(1) + _offsets[:seq_len]
    return tokens[idx], tokens[idx + 1]

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
BATCH_SIZE = 2
ACCUM_STEPS = 4
TOTAL_TEST_STEPS = 50

# Track expert utilization
expert_picks = torch.zeros(4, device=device)

loss_history = []
print(f"Running {TOTAL_TEST_STEPS} steps...")
for step in range(TOTAL_TEST_STEPS):
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = torch.zeros((), device=device)
    
    for _ in range(ACCUM_STEPS):
        bx, by = get_batch(BATCH_SIZE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(bx, targets=by)
        (loss / ACCUM_STEPS).backward()
        loss_accum += loss.detach() / ACCUM_STEPS

    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    optimizer.step()
    
    if step % 10 == 0 or step == TOTAL_TEST_STEPS - 1:
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_s = (BATCH_SIZE * ACCUM_STEPS * 256) / max(dt, 1e-4)
        alloc = torch.cuda.memory_allocated() / (1024**3)
        l_item = loss_accum.item()
        loss_history.append((step, l_item, gnorm, alloc, tok_s))
        print(f"step {step:03d} | loss={l_item:.4f} | gnorm={gnorm:.3f} | alloc={alloc:.2f}GB | tok/s={tok_s:.0f}")

initial_loss = loss_history[0][1]
final_loss = loss_history[-1][1]
loss_diff = final_loss - initial_loss
print(f"\nInitial Loss (step 0): {initial_loss:.4f} -> Final Loss (step {TOTAL_TEST_STEPS-1}): {final_loss:.4f} (delta: {loss_diff:.4f})")
has_downward_trend = final_loss < initial_loss
print(f"Downward trend observed: {has_downward_trend}")
test9_pass = has_downward_trend and not any(math.isnan(l[1]) for l in loss_history)
print(f"TEST 9 VERDICT: {'PASS' if test9_pass else 'FAIL'}")

print("\n" + "=" * 60)
print("TEST 10: THROUGHPUT BENCHMARK (.item() vs .detach())")
print("=" * 60)

# Benchmark Version A: loss.item() inside accumulation loop
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
trials = 5
for _ in range(trials):
    loss_acc_a = 0.0
    for _ in range(ACCUM_STEPS):
        bx, by = get_batch(BATCH_SIZE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(bx, targets=by)
        (loss / ACCUM_STEPS).backward()
        loss_acc_a += loss.item() / ACCUM_STEPS # Forces CPU sync
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
time_a = (time.perf_counter() - t0) / trials
tok_s_a = (BATCH_SIZE * ACCUM_STEPS * 256) / time_a

# Benchmark Version B: loss.detach() accumulated on CUDA
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(trials):
    loss_acc_b = torch.zeros((), device=device)
    for _ in range(ACCUM_STEPS):
        bx, by = get_batch(BATCH_SIZE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(bx, targets=by)
        (loss / ACCUM_STEPS).backward()
        loss_acc_b += loss.detach() / ACCUM_STEPS # Async tensor add
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
time_b = (time.perf_counter() - t0) / trials
tok_s_b = (BATCH_SIZE * ACCUM_STEPS * 256) / time_b

speedup_pct = ((time_a - time_b) / time_a) * 100
print(f"Version A (loss.item() inside loop):  {time_a*1000:.1f} ms/step ({tok_s_a:.0f} tok/s)")
print(f"Version B (loss.detach() on GPU):      {time_b*1000:.1f} ms/step ({tok_s_b:.0f} tok/s)")
print(f"Speed difference: {speedup_pct:+.2f}% speedup")
test10_verdict = "PASS"
print(f"TEST 10 VERDICT: {test10_verdict}")
