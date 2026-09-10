# experiments/architecture_matrix/jarvis_research_controller.py
"""
JARVIS RESEARCH CONTROLLER & ABLATION ENGINE
============================================
Automated execution framework for Phase 2 & 3 controlled comparisons:
- Delta-Rule Associative Memory
- Buffer Width Scaling (W=8, W=16, W=32)
- Fine-Grained MoE (8 experts under fixed budget)
- V2 Full Continued Training (100 steps)
- V2 Component Ablations (No-Buffer, No-Gated-Write, Baseline-GELU)

Tracks all metrics in research_database.json and leaderboard.md.
"""

import os
import sys
import math
import time
import json
import argparse
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tiktoken
from jarvis_model import Jarvis, JarvisBlock, RMSNorm
from jarvis_v2_combined import JarvisV2, JarvisV2Block, JarvisV2Attention, JarvisV2MoE
from attention_variants import (
    MultiTimescaleAttention,
    HybridRecurrentSlidingAttention,
    GatedWriteAssociativeAttention,
    DeltaAssociativeAttention,
    WriteEraseAssociativeAttention,
    HybridWriteEraseBufferAttention,
    AdaptiveDecayAssociativeAttention,
    GatedReadAssociativeAttention,
    EraseGateAssociativeAttention,
    AdaptiveWriteEraseBufferAttention,
)
from ffn_variants import MoELayerWithActivation
from moe_variants import AuxFreeBiasMoELayer, DeepSeekSharedMoELayer
from evaluate_architecture import run_canonical_evaluation

DB_PATH = os.path.join(WORKSPACE_ROOT, "experiments", "research_database.json")
LEADERBOARD_PATH = os.path.join(WORKSPACE_ROOT, "experiments", "leaderboard.md")


def load_research_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"experiments": {}, "leaderboard": []}


def save_research_database(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

    # Update Markdown Leaderboard
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        f.write("# JARVIS-600M RESEARCH LEADERBOARD\n\n")
        f.write("| Experiment ID | Category | Initial CE | Final CE | $\\Delta$ CE | Final PPL | Throughput | Needle Rank @ 64 |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for entry in sorted(db["leaderboard"], key=lambda x: x.get("final_ce", 999.0)):
            f.write(f"| `{entry['experiment_id']}` | {entry.get('category', 'Architecture')} | {entry.get('initial_ce', 0.0):.4f} | **{entry.get('final_ce', 0.0):.4f}** | {entry.get('ce_delta', 0.0):+.4f} | {entry.get('final_ppl', 0.0):.2f} | {entry.get('tok_s', 0):.0f} tok/s | {entry.get('needle_rank_64', 0):.1f} |\n")


def build_candidate_model(experiment_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if experiment_id == "baseline":
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
    elif experiment_id == "v2_full":
        model = JarvisV2(max_seq_len=512)
    elif experiment_id == "v2_write_erase":
        # V2 architecture with combined Local Buffer + Separate Write and Erase Gates
        model = JarvisV2(max_seq_len=512)
        for block in model.blocks:
            block.attn = HybridWriteEraseBufferAttention(d_model=1024, n_heads=16, window_size=16, max_seq_len=512)
    elif experiment_id == "exp_write_erase_gate":
        # Phase 1A: Baseline associative attention with Separate Write and Erase Gates
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = WriteEraseAssociativeAttention(d_model=1024, n_heads=16, max_seq_len=512)
    elif experiment_id == "exp_buffer_write_erase":
        # Phase 1A + Local Buffer: Separate Write/Erase Gates + Local Buffer W=16 on baseline
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = HybridWriteEraseBufferAttention(d_model=1024, n_heads=16, window_size=16, max_seq_len=512)
    elif experiment_id == "exp_adaptive_decay":
        # Phase 1B: Baseline associative attention with input-dependent adaptive decay
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = AdaptiveDecayAssociativeAttention(d_model=1024, n_heads=16, max_seq_len=512)
    elif experiment_id == "exp_gated_read":
        # Phase 1D: Baseline associative attention with gated read
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = GatedReadAssociativeAttention(d_model=1024, n_heads=16, max_seq_len=512)
    elif experiment_id == "exp_erase_gate":
        # Phase 2 Variant 4: Baseline associative attention with standalone Erase Gate
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = EraseGateAssociativeAttention(d_model=1024, n_heads=16, max_seq_len=512)
    elif experiment_id == "exp_delta_memory":
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = DeltaAssociativeAttention(d_model=1024, n_heads=16, max_seq_len=512)
    elif experiment_id == "exp_buffer_w8":
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = HybridRecurrentSlidingAttention(d_model=1024, n_heads=16, window_size=8, max_seq_len=512)
    elif experiment_id == "exp_buffer_w32":
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = HybridRecurrentSlidingAttention(d_model=1024, n_heads=16, window_size=32, max_seq_len=512)
    elif experiment_id == "v2_ablate_nobuffer":
        # V2 without sliding buffer (pure recurrent with multi-timescale + gated write)
        model = JarvisV2(max_seq_len=512)
        for block in model.blocks:
            block.attn = GatedWriteAssociativeAttention(d_model=1024, n_heads=16, max_seq_len=512)
    elif experiment_id == "v2_ablate_nogate":
        # V2 without write gate (pure buffer + multi-timescale)
        model = JarvisV2(max_seq_len=512)
        for block in model.blocks:
            block.attn = HybridRecurrentSlidingAttention(d_model=1024, n_heads=16, window_size=16, max_seq_len=512)
    elif experiment_id == "v2_ablate_gelu":
        # V2 with GELU instead of squared-ReLU
        model = JarvisV2(max_seq_len=512)
        for block in model.blocks:
            block.moe = MoELayerWithActivation(d_model=1024, activation="gelu")
    elif experiment_id == "exp_best_memory_combo":
        # Phase 2 Variant 10: Best Memory Combination (Adaptive Decay + Write/Erase + Gated Read + Local Buffer)
        model = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, num_experts=4, top_k=2, max_seq_len=512, use_cuda_attn=False, use_cuda_moe=False)
        for block in model.blocks:
            block.attn = AdaptiveWriteEraseBufferAttention(d_model=1024, n_heads=16, window_size=16, max_seq_len=512)
    else:
        raise ValueError(f"Unknown experiment_id: {experiment_id}")

    return model.to(device)


def run_controlled_experiment(
    experiment_id: str,
    category: str = "Attention",
    steps: int = 100,
    lr: float = 5e-5,
    seed: int = 42,
):
    print("\n" + "=" * 90)
    print(f"CONTROLLED EXPERIMENT RUNNER: {experiment_id}")
    print(f"Category: {category} | Steps: {steps} | LR: {lr} | Seed: {seed}")
    print("=" * 90)

    ckpt_save = os.path.join(ARCH_DIR, f"ckpt_{experiment_id}.pt")
    eval_json = os.path.join(ARCH_DIR, f"canonical_eval_{experiment_id}.json")
    db = load_research_database()

    # 1. Skip check: Do NOT rerun completed experiments
    if experiment_id in db.get("experiments", {}) and os.path.exists(ckpt_save) and os.path.exists(eval_json):
        print(f"[SKIP] Experiment {experiment_id} is already completed and recorded in research database.")
        print(f"       Checkpoint: {ckpt_save}")
        print(f"       Report:     {eval_json}")
        return db["experiments"][experiment_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    enc = tiktoken.get_encoding("gpt2")

    # Load clean data
    corpus_path = os.path.join(JARVIS_ENGINE, "data_clean.txt")
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        train_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device)

    val_path = os.path.join(JARVIS_ENGINE, "fresh_holdout.txt")
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_tokens = torch.tensor(enc.encode(f.read()), dtype=torch.long, device=device)

    # Build model
    model = build_candidate_model(experiment_id)

    # 2. Check for intermediate checkpoints to resume from
    import glob
    inter_ckpts = sorted(glob.glob(os.path.join(ARCH_DIR, f"ckpt_{experiment_id}_step*.pt")))
    start_step = 1
    last_val_ce, last_val_ppl = None, None

    if inter_ckpts:
        latest_inter = inter_ckpts[-1]
        ckpt = torch.load(latest_inter, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        # Extract step index from filename e.g. ckpt_v2_ablate_gelu_step25.pt
        try:
            step_str = latest_inter.split("_step")[-1].replace(".pt", "")
            completed_step = int(step_str)
            start_step = completed_step + 1
            last_val_ce = ckpt.get("val_loss")
            last_val_ppl = ckpt.get("val_ppl")
            print(f"  [Resume] Resuming {experiment_id} from {os.path.basename(latest_inter)} (step {completed_step})")
        except ValueError:
            pass
    else:
        # Load compatible baseline checkpoint
        ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        model_sd = model.state_dict()
        filtered_sd = {k: v for k, v in new_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        print(f"  [Load Weights] Loaded {len(filtered_sd)} keys (Filtered/Missing: {len(missing)})")

        if "buffer" in experiment_id:
            init_gammas = [2.94] * 8 + [5.50] * 8
            for block in model.blocks:
                if hasattr(block.attn, "gamma_raw"):
                    block.attn.gamma_raw.data.copy_(torch.tensor(init_gammas, dtype=torch.float32, device=device))
            print("  [Init Decay] Re-initialized 16 heads across 2 timescale bands for hybrid buffer")
        elif "v2" in experiment_id:
            init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
            for block in model.blocks:
                if hasattr(block.attn, "gamma_raw"):
                    block.attn.gamma_raw.data.copy_(torch.tensor(init_gammas, dtype=torch.float32, device=device))
            print("  [Init Decay] Re-initialized 16 heads across 4 timescale bands for Jarvis-V2")

    # Initial evaluation
    from run_track_experiments import evaluate_holdout
    init_ce, init_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=512, seed=seed)
    print(f"Step 000 (Initial): Holdout CE = {init_ce:.4f} | PPL = {init_ppl:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    _offsets = torch.arange(512, device=device)

    def get_batch():
        ix = torch.randint(0, len(train_tokens) - 513, (2,), device=device)
        idx = ix.unsqueeze(1) + _offsets
        return train_tokens[idx], train_tokens[idx + 1]

    model.train()
    t_start = time.perf_counter()
    val_ce, val_ppl = init_ce, init_ppl

    for step in range(start_step, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(4):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
            (loss / 4).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update dynamic expert bias if aux-free MoE
        for m in model.modules():
            if hasattr(m, "update_bias"):
                m.update_bias()

        if step % 25 == 0 or step == steps:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t_start
            tok_s = (step * 8 * 512) / max(dt, 1e-4)
            val_ce, val_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=512, seed=seed)
            print(f"  Step {step:03d}/{steps}: Holdout CE = {val_ce:.4f} | PPL = {val_ppl:.2f} | {tok_s:.0f} tok/s")

            # Save intermediate progress checkpoint
            inter_save = os.path.join(ARCH_DIR, f"ckpt_{experiment_id}_step{step:03d}.pt")
            torch.save({
                "step": 4284 + step,
                "model_state_dict": model.state_dict(),
                "val_loss": val_ce,
                "val_ppl": val_ppl,
            }, inter_save)

    # Save final checkpoint
    torch.save({"step": 4284 + steps, "model_state_dict": model.state_dict(), "val_loss": val_ce}, ckpt_save)
    print(f"  -> Saved final experiment checkpoint: {os.path.basename(ckpt_save)}")

    # Run Canonical Evaluation Harness
    canonical_report = run_canonical_evaluation(
        model=model,
        model_type="v2" if "v2" in experiment_id else "baseline",
        ckpt_path=ckpt_save,
        output_json=eval_json,
        quick_test=False,
    )

    final_ce = canonical_report["language_modeling"]["canonical_ce"]
    final_ppl = canonical_report["language_modeling"]["canonical_ppl"]
    needle_rank_64 = canonical_report["associative_memory"]["single_needle"]["dist_64"]["mean_rank"]
    prefill_speed = canonical_report["efficiency"]["prefill_throughput_tok_s"]

    # Record in database
    db = load_research_database()
    result_entry = {
        "experiment_id": experiment_id,
        "category": category,
        "initial_ce": init_ce,
        "final_ce": final_ce,
        "ce_delta": final_ce - init_ce,
        "final_ppl": final_ppl,
        "tok_s": prefill_speed,
        "needle_rank_64": needle_rank_64,
        "checkpoint": ckpt_save,
        "canonical_report": eval_json,
    }
    db["experiments"][experiment_id] = result_entry
    db["leaderboard"] = [e for e in db["leaderboard"] if e["experiment_id"] != experiment_id]
    db["leaderboard"].append(result_entry)
    save_research_database(db)

    # Clean up intermediate step checkpoints now that final checkpoint & eval are secured
    for old_inter in glob.glob(os.path.join(ARCH_DIR, f"ckpt_{experiment_id}_step*.pt")):
        try:
            os.remove(old_inter)
        except OSError:
            pass

    print(f"\n[OK] Completed {experiment_id}: Final CE = {final_ce:.4f} | Rank @ 64 = {needle_rank_64:.1f}")
    return result_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment ID to run")
    parser.add_argument("--category", type=str, default="Architecture")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    run_controlled_experiment(
        experiment_id=args.exp,
        category=args.category,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
