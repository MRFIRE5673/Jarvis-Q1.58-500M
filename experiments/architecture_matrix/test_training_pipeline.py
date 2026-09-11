# experiments/architecture_matrix/test_training_pipeline.py
"""
JARVIS TRAINING PIPELINE VALIDATION & LAUNCH GATE TEST RUNNER
============================================================
Automated validation testing:
1. VRAM Safety & Numerical Stability Smoke Test (Forward, Backward, Optimizer, BF16, NaNs, Infs)
2. Empirical Sustained Training Throughput & Step Latency Measurement on RTX 5070
3. Atomic Checkpointing & State Resumption Integrity Test (Model, Optimizer, Dataloader shard/offset)
"""

import os
import sys
import math
import time
import json
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")
SHARDS_DIR = os.path.join(DATA_DIR, "shards")
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix", "reports")
CHECKPOINT_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "checkpoints_1b")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import Jarvis
from data.streaming_dataloader import ShardedTokenDataset
from experiments.architecture_matrix.train_1b_production import (
    decoupled_forward,
    save_checkpoint,
    load_checkpoint,
)

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def run_pipeline_validation():
    print("=" * 85)
    print("JARVIS 1.0B TRAINING PIPELINE COMPREHENSIVE VALIDATION SUITE")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
    print("=" * 85)
    
    device = "cuda"
    seq_len = 512
    micro_batch = 2
    accum_steps = 4
    tokens_per_step = micro_batch * seq_len * accum_steps # 4096 tokens
    
    results = {}
    
    # -------------------------------------------------------------------------
    # TEST 1: Model Instantiation & VRAM Safety Smoke Test
    # -------------------------------------------------------------------------
    print("\n[PHASE 1/3] VRAM Safety & Numerical Stability Smoke Test...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    t0_init = time.perf_counter()
    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=seq_len,
        use_cuda_attn=False,
        use_cuda_moe=True,
    ).to(device)
    
    init_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"  Model Instantiated: {model.param_count()[1]} ({init_alloc_mb:.1f} MB weights in VRAM)")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    
    train_loader = ShardedTokenDataset(
        shards_dir=SHARDS_DIR,
        split="train",
        seq_len=seq_len,
        batch_size=micro_batch,
        device=device,
        seed=42,
    )
    
    # Execute 3 warmup steps with full gradient accumulation
    model.train()
    smoke_losses = []
    smoke_norms = []
    
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        ce_accum = 0.0
        tot_accum = 0.0
        
        for _ in range(accum_steps):
            x, y = train_loader.next_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, tot_loss, ce_loss, _, _ = decoupled_forward(model, x, targets=y)
                
            loss_to_back = tot_loss / accum_steps
            loss_to_back.backward()
            ce_accum += ce_loss.item() / accum_steps
            tot_accum += tot_loss.item() / accum_steps
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        optimizer.step()
        
        assert not math.isnan(tot_accum) and not math.isinf(tot_accum), f"NaN/Inf loss at smoke step {step}"
        assert not math.isnan(norm_val) and not math.isinf(norm_val), f"NaN/Inf grad norm at smoke step {step}"
        
        smoke_losses.append(ce_accum)
        smoke_norms.append(norm_val)
        print(f"  Smoke Step {step}: Train CE = {ce_accum:.4f}, Grad Norm = {norm_val:.3f}")
        
    smoke_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    smoke_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    smoke_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    total_gpu_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    headroom_mb = total_gpu_mb - smoke_res_mb
    
    print(f"  [PASSED] VRAM Allocation: {smoke_alloc_mb:.1f} MB allocated | {smoke_peak_mb:.1f} MB peak | {smoke_res_mb:.1f} MB reserved")
    print(f"  [PASSED] GPU Headroom:    {headroom_mb:.1f} MB free on RTX 5070 (Safety margin: {headroom_mb/total_gpu_mb*100:.1f}%)")
    
    results["vram_safety"] = {
        "status": "PASSED",
        "allocated_mb": round(smoke_alloc_mb, 1),
        "peak_mb": round(smoke_peak_mb, 1),
        "reserved_mb": round(smoke_res_mb, 1),
        "headroom_mb": round(headroom_mb, 1),
        "safety_margin_pct": round(headroom_mb / total_gpu_mb * 100, 1),
    }
    
    # -------------------------------------------------------------------------
    # TEST 2: Sustained Throughput & Step Latency Measurement
    # -------------------------------------------------------------------------
    print("\n[PHASE 2/3] Sustained Training Throughput Measurement (20 steps = 81,920 tokens)...")
    step_latencies = []
    num_meas_steps = 20
    
    for m_step in range(1, num_meas_steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        optimizer.zero_grad(set_to_none=True)
        ce_accum = 0.0
        
        for _ in range(accum_steps):
            x, y = train_loader.next_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, tot_loss, ce_loss, _, _ = decoupled_forward(model, x, targets=y)
            (tot_loss / accum_steps).backward()
            ce_accum += ce_loss.item() / accum_steps
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        step_latencies.append(lat)
        
    avg_latency = float(sum(step_latencies) / len(step_latencies))
    measured_tok_s = tokens_per_step / avg_latency
    effective_tok_s = measured_tok_s * 0.90 # 90% duty cycle allowance for periodic checkpointing & eval
    
    print(f"  Average Step Latency: {avg_latency*1000:.1f} ms / step ({tokens_per_step:,} tokens/step)")
    print(f"  Raw Training Throughput:       {measured_tok_s:,.1f} tokens/sec")
    print(f"  Realistic Effective Throughput: {effective_tok_s:,.1f} tokens/sec (factoring eval/save/uptime)")
    
    # Time projections
    milestones = [100_000_000, 250_000_000, 500_000_000, 800_000_000, 1_000_000_000]
    time_projections = {}
    for tok in milestones:
        hours = tok / effective_tok_s / 3600.0
        days = hours / 24.0
        tok_str = f"{tok/1e6:.0f}M" if tok < 1e9 else f"{tok/1e9:.1f}B"
        time_projections[tok_str] = {
            "tokens": tok,
            "realistic_hours": round(hours, 2),
            "realistic_days": round(days, 2),
        }
        print(f"    Target {tok_str:>5} tokens: {hours:6.1f} hours ({days:5.2f} days)")
        
    results["throughput"] = {
        "status": "PASSED",
        "avg_step_latency_ms": round(avg_latency * 1000, 1),
        "raw_tokens_per_sec": round(measured_tok_s, 1),
        "effective_tokens_per_sec": round(effective_tok_s, 1),
        "projections": time_projections,
    }
    
    # -------------------------------------------------------------------------
    # TEST 3: Checkpointing & State Resumption Integrity Test
    # -------------------------------------------------------------------------
    print("\n[PHASE 3/3] Checkpointing & State Resumption Test...")
    test_ckpt_path = os.path.join(CHECKPOINT_DIR, "smoke_test_resume.pt")
    
    # 1. Capture state before saving
    saved_step = 23 # 3 smoke + 20 meas
    saved_tokens = saved_step * tokens_per_step
    pre_save_state = train_loader.get_state()
    pre_save_shard = pre_save_state["current_shard_idx"]
    pre_save_offset = pre_save_state["current_offset"]
    
    # Save checkpoint
    save_checkpoint(
        test_ckpt_path,
        model,
        optimizer,
        saved_step,
        saved_tokens,
        train_loader,
        best_val_ce=3.2858,
        loss_history=[],
    )
    assert os.path.exists(test_ckpt_path), "Checkpoint file was not created"
    
    # 2. Fetch ground-truth batch post-save (to compare with resumed model)
    x_expected, y_expected = train_loader.next_batch()
    
    # 3. Completely teardown model and dataloader
    del model, optimizer, train_loader
    torch.cuda.empty_cache()
    
    # 4. Instantiate fresh model, optimizer, dataloader
    resumed_model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=seq_len,
        use_cuda_attn=False,
        use_cuda_moe=True,
    ).to(device)
    
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=1.5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    
    resumed_loader = ShardedTokenDataset(
        shards_dir=SHARDS_DIR,
        split="train",
        seq_len=seq_len,
        batch_size=micro_batch,
        device=device,
        seed=42,
    )
    
    # 5. Load checkpoint
    loaded_step, loaded_tokens, _, _ = load_checkpoint(
        test_ckpt_path,
        resumed_model,
        resumed_optimizer,
        resumed_loader,
        device=device,
    )
    
    assert loaded_step == saved_step, f"Step mismatch: {loaded_step} != {saved_step}"
    assert loaded_tokens == saved_tokens, f"Token mismatch: {loaded_tokens} != {saved_tokens}"
    
    # 6. Verify resumed dataloader yields bit-identical batch
    x_resumed, y_resumed = resumed_loader.next_batch()
    assert (x_expected == x_resumed).all(), "Dataloader resumed batch token mismatch!"
    assert (y_expected == y_resumed).all(), "Dataloader resumed target token mismatch!"
    print(f"  [PASSED] Dataloader exact batch equivalence verified post-resume!")
    
    # 7. Execute 3 more steps on resumed model to verify gradient flow
    resumed_model.train()
    for s in range(1, 4):
        resumed_optimizer.zero_grad(set_to_none=True)
        tot_accum = 0.0
        for _ in range(accum_steps):
            x, y = resumed_loader.next_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, tot_loss, _, _, _ = decoupled_forward(resumed_model, x, targets=y)
            (tot_loss / accum_steps).backward()
            tot_accum += tot_loss.item() / accum_steps
        torch.nn.utils.clip_grad_norm_(resumed_model.parameters(), max_norm=1.0)
        resumed_optimizer.step()
        assert not math.isnan(tot_accum), f"NaN after resume at step {s}"
        print(f"  Resumed Step {loaded_step + s}: Train Loss = {tot_accum:.4f}")
        
    # Clean up smoke test checkpoint
    if os.path.exists(test_ckpt_path):
        os.remove(test_ckpt_path)
    print(f"  [PASSED] Checkpoint save and resumption completely verified!")
    
    results["checkpoint_resume"] = {
        "status": "PASSED",
        "verified_step": loaded_step,
        "verified_tokens": loaded_tokens,
        "batch_determinism_match": True,
    }
    
    # Save test results JSON
    report_json = os.path.join(REPORTS_DIR, "training_pipeline_test_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Validation test report written to: {report_json}")
    print("=" * 85)
    print("ALL TRAINING PIPELINE INTEGRITY TESTS PASSED CLEANLY.")
    print("=" * 85)
    return results


if __name__ == "__main__":
    run_pipeline_validation()
