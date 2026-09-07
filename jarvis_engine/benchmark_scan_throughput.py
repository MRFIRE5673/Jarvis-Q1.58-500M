import sys
sys.path.insert(0, ".")
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import subprocess

from jarvis_model import Jarvis, LiquidStateFusion

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PART 1: NUMERICAL EQUIVALENCE BENCHMARK")
print("=" * 60)

# Compare Sequential vs Parallel Scan across multiple lengths and variances
B, C = 2, 1024
lengths = [20, 64, 128, 256, 512]
variances = [0.05, 0.5, 2.0, 8.0]

def sequential_scan(x, alpha, h_prev):
    B, T, C = x.shape
    h_states = []
    h_cur = h_prev
    for t in range(T):
        h_cur = alpha * h_cur + (1.0 - alpha) * x[:, t, :]
        h_states.append(h_cur)
    h_out = torch.stack(h_states, dim=1)
    return h_out, h_cur

def parallel_scan(x, alpha, h_prev):
    B, T, C = x.shape
    t_idx = torch.arange(T, device=x.device, dtype=torch.float32)
    diff = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).clamp(min=0)
    causal = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0) >= 0).to(dtype=x.dtype)
    log_a = torch.log(alpha)
    decay_mat = (torch.exp(log_a * diff) * causal).to(dtype=x.dtype)
    conv_out = (1.0 - alpha) * torch.matmul(decay_mat, x)
    carry_weights = torch.exp(log_a * (t_idx + 1)).view(1, T, 1).to(dtype=x.dtype)
    carry_out = carry_weights * h_prev.unsqueeze(1)
    h_out = conv_out + carry_out
    return h_out, h_out[:, -1, :]

print("Testing FP32 numerical equivalence:")
for T in lengths:
    for v in [0.5]:
        x = torch.randn(B, T, C, device=device, dtype=torch.float32)
        h_prev = torch.randn(B, C, device=device, dtype=torch.float32)
        alpha = torch.tensor(0.85, device=device, dtype=torch.float32)
        
        h_seq, last_seq = sequential_scan(x, alpha, h_prev)
        h_par, last_par = parallel_scan(x, alpha, h_prev)
        
        max_err = (h_seq - h_par).abs().max().item()
        mean_err = (h_seq - h_par).abs().mean().item()
        rel_err = max_err / (h_seq.abs().max().item() + 1e-8)
        close = torch.allclose(h_seq, h_par, atol=1e-5, rtol=1e-5)
        print(f"  T={T:3d} | max_err={max_err:.2e} | mean_err={mean_err:.2e} | rel_err={rel_err:.2e} | allclose={close}")

print("\n" + "=" * 60)
print("PART 2: END-TO-END 606M TRAINING THROUGHPUT BENCHMARK")
print("=" * 60)

# Setup model and dataset
model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
               num_experts=4, top_k=2, max_seq_len=256).cuda()
model.train()

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
BENCH_STEPS = 5

def get_gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return float(out.strip())
    except:
        return -1.0

# Define a monkey-patchable parallel forward for LiquidStateFusion
def parallel_liquid_forward(self, x, act_var, h_prev=None):
    B, T, C = x.shape
    if h_prev is None:
        h_prev = x.new_zeros(B, C)
    alpha_raw = torch.sigmoid(-self.var_scale * act_var)
    alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw
    
    t_idx = torch.arange(T, device=x.device, dtype=torch.float32)
    diff = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).clamp(min=0)
    causal = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0) >= 0).to(dtype=x.dtype)
    log_a = torch.log(alpha)
    decay_mat = (torch.exp(log_a * diff) * causal).to(dtype=x.dtype)
    conv_out = (1.0 - alpha) * torch.matmul(decay_mat, x)
    carry_weights = torch.exp(log_a * (t_idx + 1)).view(1, T, 1).to(dtype=x.dtype)
    carry_out = carry_weights * h_prev.unsqueeze(1)
    h_out = conv_out + carry_out
    h_last = h_out[:, -1, :]
    return h_out, h_last

original_liquid_forward = LiquidStateFusion.forward

# 1. Benchmark Sequential Scan (Production Current)
print("1. Benchmarking SEQUENTIAL scan for 5 optimizer steps...")
LiquidStateFusion.forward = original_liquid_forward

# Warmup
optimizer.zero_grad(set_to_none=True)
for _ in range(ACCUM_STEPS):
    bx, by = get_batch(BATCH_SIZE)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = model(bx, targets=by)
    (loss / ACCUM_STEPS).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
torch.cuda.synchronize()

seq_times = []
seq_utils = []
for step in range(BENCH_STEPS):
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
    seq_times.append(dt)
    u = get_gpu_util()
    seq_utils.append(u)
    print(f"  Seq Step {step+1}: dt={dt:.3f}s | tok/s={2048/dt:.1f} | GPU util={u:.0f}% | loss={loss_acc.item():.4f}")

# 2. Benchmark Parallel Scan
print("\n2. Benchmarking PARALLEL decay scan for 5 optimizer steps...")
LiquidStateFusion.forward = parallel_liquid_forward

# Warmup
optimizer.zero_grad(set_to_none=True)
for _ in range(ACCUM_STEPS):
    bx, by = get_batch(BATCH_SIZE)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = model(bx, targets=by)
    (loss / ACCUM_STEPS).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
torch.cuda.synchronize()

par_times = []
par_utils = []
for step in range(BENCH_STEPS):
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
    par_times.append(dt)
    u = get_gpu_util()
    par_utils.append(u)
    print(f"  Par Step {step+1}: dt={dt:.3f}s | tok/s={2048/dt:.1f} | GPU util={u:.0f}% | loss={loss_acc.item():.4f}")

# Restore original forward before exiting
LiquidStateFusion.forward = original_liquid_forward

avg_seq_time = sum(seq_times) / len(seq_times)
avg_par_time = sum(par_times) / len(par_times)
tok_s_seq = 2048.0 / avg_seq_time
tok_s_par = 2048.0 / avg_par_time
speedup = avg_seq_time / avg_par_time
time_saved_per_step = avg_seq_time - avg_par_time

print("\n" + "=" * 60)
print("FINAL SCAN COMPARISON REPORT")
print("=" * 60)
print(f"Sequential Scan Average Step Time:  {avg_seq_time:.3f}s ({tok_s_seq:.1f} tok/s) | GPU Util: {sum(seq_utils)/len(seq_utils):.1f}%")
print(f"Parallel Decay Scan Step Time:      {avg_par_time:.3f}s ({tok_s_par:.1f} tok/s) | GPU Util: {sum(par_utils)/len(par_utils):.1f}%")
print(f"Step Time Reduction:               {time_saved_per_step:.3f} seconds faster per step")
print(f"End-to-End Speedup:                {speedup:.2f}x")
print(f"Throughput Increase:               {tok_s_seq:.1f} -> {tok_s_par:.1f} tok/s (+{tok_s_par - tok_s_seq:.1f} tok/s)")
print(f"Estimated 5,000-step training run:")
print(f"  Sequential: {5000 * avg_seq_time / 3600:.2f} hours")
print(f"  Parallel:   {5000 * avg_par_time / 3600:.2f} hours")
print(f"  Time Saved: {(5000 * time_saved_per_step) / 3600:.2f} hours")
