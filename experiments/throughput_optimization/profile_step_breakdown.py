import os
import sys
import glob
import time
import json
import torch
import torch.nn as nn

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")
ROOT_LSF_DIR = os.path.join(WORKSPACE_ROOT, "liquid_fusion_cuda")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ROOT_LSF_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Configure CUDA bin DLL directory on Windows
if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        cands = sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"), reverse=True)
        if cands:
            cuda_home = cands[0]
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin")):
        try:
            os.add_dll_directory(os.path.join(cuda_home, "bin"))
        except Exception:
            pass

from jarvis_model import (
    JarvisBlock,
    AssociativeLinearAttention,
    CUDASparseMoELayer,
    SparseMoELayer,
    LiquidStateFusion,
    _HAS_CUDA_MOE,
    _HAS_CUDA_ATTN,
)
from data.streaming_dataloader import ShardedTokenDataset

def time_op(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    
    timings = []
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        timings.append(start_ev.elapsed_time(end_ev))
        
    timings.sort()
    return {
        "mean_ms": sum(timings) / len(timings),
        "median_ms": timings[len(timings) // 2],
        "min_ms": timings[0],
        "p95_ms": timings[int(len(timings) * 0.95)]
    }

def profile_components(B=2, T=512, D=1024, n_heads=16, num_experts=4, top_k=2):
    print("=" * 80)
    print(f"JARVIS TRAINING STEP BREAKDOWN PROFILING (Blackwell RTX 5070)")
    print(f"Batch={B}, SeqLen={T}, Dim={D}, Heads={n_heads}, Experts={num_experts}, TopK={top_k}, BF16 Autocast")
    print("=" * 80)
    
    device = "cuda"
    torch.manual_seed(42)
    
    # 1. Dataloader profiling
    print("\n[1/6] Profiling ShardedTokenDataset throughput...")
    shards_dir = os.path.join(DATA_DIR, "shards")
    val_shard = os.path.join(shards_dir, "val_shard_0000.bin")
    loader = ShardedTokenDataset(
        shards_dir=shards_dir,
        split="val",
        seq_len=T,
        batch_size=B,
        device=device,
        seed=42,
        shuffle_shards=False
    )
    t_load_start = time.perf_counter()
    for _ in range(100):
        x, y = loader.next_batch()
    torch.cuda.synchronize()
    t_load_ms = (time.perf_counter() - t_load_start) / 100 * 1000
    print(f"  Dataloader next_batch() average latency: {t_load_ms:.4f} ms per microbatch ({B*T/(t_load_ms/1000):,.0f} tok/s)")

    # 2. Block initialization (1 representative layer)
    print("\n[2/6] Initializing single production JarvisBlock...")
    block = JarvisBlock(
        d_model=D,
        n_heads=n_heads,
        num_experts=num_experts,
        top_k=top_k,
        max_seq_len=T,
        use_cuda_attn=False, # exactly matching train_1b_production.py
        use_cuda_moe=True,
    ).to(device)
    block.train()
    
    x = torch.randn(B, T, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    
    # 3. Component Forward Profiling
    print("\n[3/6] Profiling Component Forward passes (BF16 Autocast)...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # Attention forward
        norm1_x = block.norm1(x)
        t_attn_fwd = time_op(lambda: block.attn(norm1_x))
        print(f"  Associative Attention Forward : {t_attn_fwd['median_ms']:.4f} ms")
        
        # MoE forward
        norm2_x = block.norm2(x)
        t_moe_fwd = time_op(lambda: block.moe(norm2_x))
        print(f"  Sparse MoE Forward            : {t_moe_fwd['median_ms']:.4f} ms")
        
        # Liquid State Fusion forward (PyTorch scan)
        moe_out, _, _, act_var = block.moe(norm2_x)
        t_lsf_fwd_pt = time_op(lambda: block.liquid(moe_out, act_var, None))
        print(f"  LSF Forward (PyTorch Scan)    : {t_lsf_fwd_pt['median_ms']:.4f} ms")
        
        # Full Block forward
        t_block_fwd = time_op(lambda: block(x))
        print(f"  Full Block Forward (1 Layer)  : {t_block_fwd['median_ms']:.4f} ms")

    # 4. Component Backward Profiling
    print("\n[4/6] Profiling Component Forward + Backward passes...")
    def run_fwd_bwd():
        x_in = x.detach().clone().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out, _, l_bal, l_ref = block(x_in)
            loss = out.sum() + l_bal + l_ref
        loss.backward()
        
    t_block_fwd_bwd = time_op(run_fwd_bwd)
    t_block_bwd = t_block_fwd_bwd['median_ms'] - t_block_fwd['median_ms']
    print(f"  Full Block Fwd+Bwd (1 Layer)  : {t_block_fwd_bwd['median_ms']:.4f} ms (Bwd est: {t_block_bwd:.4f} ms)")

    # 5. Gradient Checkpointed Block (Production Path)
    print("\n[5/6] Profiling Gradient Checkpointing (use_reentrant=False)...")
    from torch.utils.checkpoint import checkpoint as grad_ckpt
    def run_grad_ckpt():
        x_in = x.detach().clone().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out, _, l_bal, l_ref = grad_ckpt(block, x_in, None, 0, use_reentrant=False)
            loss = out.sum() + l_bal + l_ref
        loss.backward()
        
    t_block_ckpt = time_op(run_grad_ckpt)
    print(f"  Checkpointed Block Fwd+Bwd    : {t_block_ckpt['median_ms']:.4f} ms per layer")

    # 6. Optimizer Step Profiling (scaled for 606M model)
    print("\n[6/6] Profiling Optimizer Step & Grad Clipping...")
    # Initialize AdamW with block params
    opt = torch.optim.AdamW(block.parameters(), lr=1.5e-4, fused=True)
    def run_clip_opt():
        torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    t_clip_opt = time_op(run_clip_opt)
    print(f"  Clip + Fused AdamW Step (Block): {t_clip_opt['median_ms']:.4f} ms")

    # Full Step Projection across 24 Layers, 4 Accumulation Steps
    print("\n" + "=" * 80)
    print("FULL STEP PROJECTION & BOTTLENECK BREAKDOWN")
    print("=" * 80)
    accum_steps = 4
    n_layers = 24
    
    total_data_time = t_load_ms * accum_steps
    total_ckpt_compute = t_block_ckpt['median_ms'] * n_layers * accum_steps
    # Estimate full model optimizer step (555 params, ~606M weights vs 25M in 1 block -> ~24x)
    total_opt_time = t_clip_opt['median_ms'] * n_layers
    
    projected_step_time_ms = total_data_time + total_ckpt_compute + total_opt_time
    projected_tok_s = (B * T * accum_steps) / (projected_step_time_ms / 1000)
    
    print(f"Component Breakdown (per 4,096-token update):")
    print(f"  - Data Loading (4 batches)    : {total_data_time:7.2f} ms ({total_data_time/projected_step_time_ms*100:4.1f}%)")
    print(f"  - 24-Layer Checkpointed Compute: {total_ckpt_compute:7.2f} ms ({total_ckpt_compute/projected_step_time_ms*100:4.1f}%)")
    print(f"  - Optimizer & Grad Clip       : {total_opt_time:7.2f} ms ({total_opt_time/projected_step_time_ms*100:4.1f}%)")
    print(f"  -------------------------------------------------------------")
    print(f"  Projected Step Time           : {projected_step_time_ms:7.2f} ms ({projected_step_time_ms/1000:.3f} s)")
    print(f"  Projected Throughput          : {projected_tok_s:7.1f} tok/s")
    
    # Layer Component Fractions
    print(f"\nWithin each Block ({t_block_fwd['median_ms']:.3f} ms forward):")
    print(f"  - Attention Fraction          : {t_attn_fwd['median_ms'] / t_block_fwd['median_ms'] * 100:5.1f}%")
    print(f"  - Sparse MoE Fraction         : {t_moe_fwd['median_ms'] / t_block_fwd['median_ms'] * 100:5.1f}%")
    print(f"  - LSF Scan Fraction           : {t_lsf_fwd_pt['median_ms'] / t_block_fwd['median_ms'] * 100:5.1f}%")

    out_file = r"G:\Jarvis_Training\run_50m_baseline\profiling\step_breakdown.json"
    with open(out_file, "w") as f:
        json.dump({
            "dataloader_ms": t_load_ms,
            "attn_fwd_ms": t_attn_fwd['median_ms'],
            "moe_fwd_ms": t_moe_fwd['median_ms'],
            "lsf_fwd_ms": t_lsf_fwd_pt['median_ms'],
            "block_fwd_ms": t_block_fwd['median_ms'],
            "block_ckpt_ms": t_block_ckpt['median_ms'],
            "projected_step_ms": projected_step_time_ms,
            "projected_tok_s": projected_tok_s,
        }, f, indent=2)
    print(f"\n[OK] Breakdown saved to {out_file}")

if __name__ == "__main__":
    profile_components()
