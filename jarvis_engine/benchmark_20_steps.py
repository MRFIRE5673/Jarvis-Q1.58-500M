import sys
sys.path.insert(0, ".")
import math
import time
import torch
import tiktoken
import subprocess

from jarvis_model import Jarvis

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Setting up model and data with EXACT production train.py configuration...")

# 1. Model setup
model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
               num_experts=4, top_k=2, max_seq_len=256).cuda()
model.train()

# 2. Data setup
enc = tiktoken.get_encoding("gpt2")
with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
tokens = torch.tensor(enc.encode(text), dtype=torch.long, device="cuda")

_offsets = torch.arange(256, device="cuda")
def get_batch(batch_size=2, seq_len=256):
    ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,), device=tokens.device)
    idx = ix.unsqueeze(1) + _offsets[:seq_len]
    return tokens[idx], tokens[idx + 1]

# 3. Optimizer setup
BASE_LR = 3e-4
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, fused=True)
    print("Using fused AdamW")
except TypeError:
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    print("Using standard AdamW")

BATCH_SIZE = 2
ACCUM_STEPS = 4 # 2 * 4 * 256 = 2048 tokens per optimizer update

def get_gpu_utilization():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        gpu_u, mem_u = [float(x.strip()) for x in out.strip().split(",")]
        return gpu_u, mem_u
    except Exception:
        return -1.0, -1.0

# Warmup step (excluded from benchmark)
print("Running 1 warmup step...")
optimizer.zero_grad(set_to_none=True)
for _ in range(ACCUM_STEPS):
    x, y = get_batch(BATCH_SIZE)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = model(x, targets=y)
    (loss / ACCUM_STEPS).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
torch.cuda.synchronize()
print("Warmup complete.")

# Benchmark 20 optimizer steps
BENCH_STEPS = 20
print(f"\nStarting benchmark for {BENCH_STEPS} optimizer steps...")

step_times = []
gpu_utils = []

torch.cuda.synchronize()
bench_start = time.perf_counter()

for step in range(BENCH_STEPS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
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
    
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    step_times.append(dt)
    
    gpu_u, mem_u = get_gpu_utilization()
    if gpu_u >= 0:
        gpu_utils.append(gpu_u)
    
    tok_s = 2048.0 / dt
    alloc = torch.cuda.memory_allocated() / (1024**3)
    print(f"Step {step+1:02d}/{BENCH_STEPS:02d} | time={dt:.3f}s | tok/s={tok_s:.1f} | GPU util={gpu_u:.0f}% | VRAM={alloc:.2f}GB | loss={loss_accum.item():.4f}")

torch.cuda.synchronize()
bench_total_time = time.perf_counter() - bench_start

avg_step_time = sum(step_times) / len(step_times)
avg_tok_s = 2048.0 / avg_step_time
total_tokens = BENCH_STEPS * 2048
effective_tok_s = total_tokens / bench_total_time
avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else -1.0

print("\n" + "="*60)
print("BENCHMARK REPORT (EXACT PRODUCTION CONFIGURATION)")
print("="*60)
print(f"Total steps measured:          {BENCH_STEPS}")
print(f"Total tokens processed:        {total_tokens}")
print(f"Wall-clock time (excl. init):  {bench_total_time:.3f} seconds ({bench_total_time/60:.2f} minutes)")
print(f"Average optimizer step time:   {avg_step_time:.3f} seconds")
print(f"Tokens/sec (2048 / step_time): {avg_tok_s:.2f} tok/s")
print(f"Overall effective throughput:  {effective_tok_s:.2f} tok/s")
print(f"Average GPU Utilization:       {avg_gpu_util:.1f}%")
print(f"Min step time:                 {min(step_times):.3f}s ({2048/min(step_times):.1f} tok/s)")
print(f"Max step time:                 {max(step_times):.3f}s ({2048/max(step_times):.1f} tok/s)")
