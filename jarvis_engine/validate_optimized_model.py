import sys
sys.path.insert(0, ".")
import math
import time
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from jarvis_model import Jarvis, LiquidStateFusion

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("=" * 60)
print("CHECK 1: NUMERICAL EQUIVALENCE VS SEQUENTIAL REFERENCE")
print("=" * 60)

liquid_prod = LiquidStateFusion(d_model=1024).to(device)

def sequential_reference(x, act_var, h_prev=None):
    B, T, C = x.shape
    if h_prev is None:
        h_prev = x.new_zeros(B, C)
    alpha_raw = torch.sigmoid(-liquid_prod.var_scale * act_var)
    alpha = liquid_prod.alpha_min + (liquid_prod.alpha_max - liquid_prod.alpha_min) * alpha_raw
    h_states = []
    h_cur = h_prev
    for t in range(T):
        h_cur = alpha * h_cur + (1.0 - alpha) * x[:, t, :]
        h_states.append(h_cur)
    h_out = torch.stack(h_states, dim=1)
    return h_out, h_cur

lengths = [20, 64, 65, 127, 128, 129, 256, 512]
all_pass_1 = True
for T in lengths:
    x = torch.randn(2, T, 1024, device=device)
    h = torch.randn(2, 1024, device=device)
    v = torch.tensor(0.75, device=device)
    
    with torch.no_grad():
        h_seq, last_seq = sequential_reference(x, v, h)
        h_prod, last_prod = liquid_prod(x, v, h)
        
    max_err = (h_seq - h_prod).abs().max().item()
    mean_err = (h_seq - h_prod).abs().mean().item()
    rel_err = max_err / (h_seq.abs().max().item() + 1e-8)
    close = torch.allclose(h_seq, h_prod, atol=1e-5, rtol=1e-5)
    print(f"  T={T:3d} | max_err={max_err:.2e} | mean_err={mean_err:.2e} | rel_err={rel_err:.2e} | allclose={close}")
    if not close:
        all_pass_1 = False

print(f"CHECK 1 RESULT: {'PASS' if all_pass_1 else 'FAIL'}")

print("\n" + "=" * 60)
print("CHECK 2: FORWARD/BACKWARD GRADIENT CHECKS")
print("=" * 60)

x_seq = torch.randn(2, 128, 1024, device=device, requires_grad=True)
h_seq = torch.randn(2, 1024, device=device, requires_grad=True)
v_seq = torch.tensor(0.65, device=device, requires_grad=True)

x_prod = x_seq.clone().detach().requires_grad_(True)
h_prod_t = h_seq.clone().detach().requires_grad_(True)
v_prod = v_seq.clone().detach().requires_grad_(True)

out_s, last_s = sequential_reference(x_seq, v_seq, h_seq)
loss_s = out_s.sum() + last_s.sum()
loss_s.backward()

out_p, last_p = liquid_prod(x_prod, v_prod, h_prod_t)
loss_p = out_p.sum() + last_p.sum()
loss_p.backward()

gx_err = (x_seq.grad - x_prod.grad).abs().max().item()
gh_err = (h_seq.grad - h_prod_t.grad).abs().max().item()
gv_err = (v_seq.grad - v_prod.grad).abs().max().item()

print(f"  dLoss/dx max error:       {gx_err:.2e}")
print(f"  dLoss/dh_prev max error:  {gh_err:.2e}")
print(f"  dLoss/dact_var max error: {gv_err:.2e}")

grad_close = torch.allclose(x_seq.grad, x_prod.grad, atol=1e-4, rtol=1e-4) and \
             torch.allclose(h_seq.grad, h_prod_t.grad, atol=1e-4, rtol=1e-4) and \
             torch.allclose(v_seq.grad, v_prod.grad, atol=1e-3, rtol=1e-3)

print(f"CHECK 2 RESULT: {'PASS' if grad_close else 'FAIL'}")

print("\n" + "=" * 60)
print("CHECK 3: FULL 606M MODEL SANITY CHECK")
print("=" * 60)

torch.cuda.empty_cache()
model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
               num_experts=4, top_k=2, max_seq_len=256).cuda()

bx = torch.randint(0, 50257, (2, 256), device=device)
by = torch.randint(0, 50257, (2, 256), device=device)

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits, loss = model(bx, targets=by)
torch.cuda.synchronize()
tfwd = time.perf_counter() - t0

has_nan = torch.isnan(logits).any().item() or math.isnan(loss.item())
has_inf = torch.isinf(logits).any().item() or math.isinf(loss.item())

torch.cuda.synchronize()
t0 = time.perf_counter()
loss.backward()
torch.cuda.synchronize()
tbwd = time.perf_counter() - t0

gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
alloc = torch.cuda.memory_allocated() / (1024**3)

print(f"  606M Forward time:  {tfwd*1000:.1f} ms")
print(f"  606M Backward time: {tbwd*1000:.1f} ms")
print(f"  Loss:               {loss.item():.4f}")
print(f"  NaN / Inf:          {has_nan} / {has_inf}")
print(f"  Pre-clip Grad Norm: {gnorm:.4f}")
print(f"  Active VRAM Alloc:  {alloc:.2f} GB")

check3_pass = (not has_nan) and (not has_inf) and (gnorm < 50.0) and (loss.item() > 0)
print(f"CHECK 3 RESULT: {'PASS' if check3_pass else 'FAIL'}")

print("\n" + "=" * 60)
print("CHECK 4: 50-STEP TRAINING SANITY TEST")
print("=" * 60)

enc = tiktoken.get_encoding("gpt2")
with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
tokens = torch.tensor(enc.encode(text), dtype=torch.long, device="cuda")

_offsets = torch.arange(256, device="cuda")
def get_batch(batch_size=2, seq_len=256):
    ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,), device=tokens.device)
    idx = ix.unsqueeze(1) + _offsets[:seq_len]
    return tokens[idx], tokens[idx + 1]

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
BATCH_SIZE = 2
ACCUM_STEPS = 4

loss_records = []
step_times_50 = []

for step in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    optimizer.zero_grad(set_to_none=True)
    loss_acc = torch.zeros((), device="cuda")
    for _ in range(ACCUM_STEPS):
        bx, by = get_batch(BATCH_SIZE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(bx, targets=by)
        (loss / ACCUM_STEPS).backward()
        loss_acc += loss.detach() / ACCUM_STEPS
        
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    optimizer.step()
    
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    step_times_50.append(dt)
    
    if step % 10 == 0 or step == 49:
        alloc_now = torch.cuda.memory_allocated() / (1024**3)
        l_val = loss_acc.item()
        loss_records.append(l_val)
        tok_s = 2048.0 / dt
        print(f"  Step {step:02d} | loss={l_val:.4f} | gnorm={gn:.3f} | time={dt:.3f}s ({tok_s:.1f} tok/s) | VRAM={alloc_now:.2f} GB")

init_loss = loss_records[0]
final_loss = loss_records[-1]
downward = final_loss < init_loss
print(f"  Loss trajectory: {init_loss:.4f} -> {final_loss:.4f} (delta: {final_loss - init_loss:+.4f})")
print(f"  Downward loss trend: {downward}")
check4_pass = downward and not any(math.isnan(l) for l in loss_records)
print(f"CHECK 4 RESULT: {'PASS' if check4_pass else 'FAIL'}")

print("\n" + "=" * 60)
print("CHECK 5 & 6: BENCHMARK PRODUCTION CONFIG & VRAM VERIFICATION")
print("=" * 60)

def get_gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return float(out.strip())
    except:
        return -1.0

# 10 benchmark steps
bench_times = []
bench_utils = []
allocs = []

for step in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss_acc = torch.zeros((), device="cuda")
    for _ in range(ACCUM_STEPS):
        bx, by = get_batch(BATCH_SIZE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(bx, targets=by)
        (loss / ACCUM_STEPS).backward()
        loss_acc += loss.detach() / ACCUM_STEPS
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bench_times.append(dt)
    u = get_gpu_util()
    if u >= 0: bench_utils.append(u)
    allocs.append(torch.cuda.memory_allocated() / (1024**3))
    print(f"  Bench Step {step+1:02d}: time={dt:.3f}s | tok/s={2048/dt:.1f} | GPU util={u:.0f}% | VRAM={allocs[-1]:.2f} GB")

avg_dt = sum(bench_times) / len(bench_times)
avg_toks = 2048.0 / avg_dt
avg_u = sum(bench_utils) / len(bench_utils) if bench_utils else -1.0
peak_vram = max(allocs)

print(f"\nFinal Production Metrics:")
print(f"  Average Optimizer Step Time: {avg_dt:.3f}s")
print(f"  Tokens/sec:                   {avg_toks:.1f} tok/s")
print(f"  Average GPU Utilization:      {avg_u:.1f}%")
print(f"  Peak VRAM Allocated:          {peak_vram:.2f} GB")

vram_in_range = (8.8 <= peak_vram <= 9.6)
print(f"  VRAM in ~9.2 GB range:        {vram_in_range}")

check5_6_pass = (avg_toks > 400.0) and vram_in_range
print(f"CHECK 5 & 6 RESULT: {'PASS' if check5_6_pass else 'FAIL'}")
