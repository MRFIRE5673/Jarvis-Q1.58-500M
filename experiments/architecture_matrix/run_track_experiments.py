# experiments/architecture_matrix/run_track_experiments.py
"""
Systematic Architecture Evolution Experiment Runner
====================================================
Runs controlled experiments across priority research tracks:
- Track A: Attention & Memory Retention (Multi-Timescale, Sliding Buffer, Write Gating)
- Track B: Modern Ternary Parameterization (Per-Channel Scaling, BitLinear)
- Track E: FFN Activations (squared-ReLU, SwiGLU under budget)
- Track F & G: MoE Evolution (DeepSeekMoE Shared Expert, Aux-Free Bias)
- Track M: Modern Optimizer (Muon + AdamW Hybrid)

Evaluates:
- Holdout Cross-Entropy & Perplexity on independent fresh_holdout.txt
- Needle-in-a-Haystack retrieval accuracy across distances
- Training stability & throughput
"""

import os
import sys
import math
import time
import json
import statistics
import argparse
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
from attention_variants import MultiTimescaleAttention, HybridRecurrentSlidingAttention, GatedWriteAssociativeAttention
from ternary_variants import PerChannelTernaryLinear, BitLinear
from ffn_variants import SquaredReLUExpert, SwiGLUExpert, MoELayerWithActivation
from moe_variants import DeepSeekSharedMoELayer, AuxFreeBiasMoELayer
from optimizers import create_hybrid_optimizer
from eval_needle_retrieval import evaluate_needle_retrieval


def build_custom_jarvis(
    attn_variant: str = "baseline",
    ffn_variant: str = "baseline",
    moe_variant: str = "baseline",
    ternary_variant: str = "baseline",
    d_model: int = 1024,
    n_layers: int = 24,
    n_heads: int = 16,
    max_seq_len: int = 512,
):
    model = Jarvis(
        vocab_size=50257,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        num_experts=4,
        top_k=2,
        max_seq_len=max_seq_len,
        use_cuda_attn=False,  # Use PyTorch module tree for custom blocks
        use_cuda_moe=False,
    )

    for b_idx in range(n_layers):
        block = model.blocks[b_idx]

        # 1. Custom Attention Subsystem
        if attn_variant == "multi_timescale":
            block.attn = MultiTimescaleAttention(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len)
        elif attn_variant == "sliding_buffer":
            block.attn = HybridRecurrentSlidingAttention(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len)
        elif attn_variant == "gated_write":
            block.attn = GatedWriteAssociativeAttention(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len)

        # 2. Custom MoE / Routing Subsystem
        if moe_variant == "deepseek_shared":
            block.moe = DeepSeekSharedMoELayer(d_model=d_model)
        elif moe_variant == "aux_free_bias":
            block.moe = AuxFreeBiasMoELayer(d_model=d_model)

        # 3. Custom FFN Activations (inside MoE experts)
        if ffn_variant == "squared_relu" and moe_variant == "baseline":
            block.moe = MoELayerWithActivation(d_model=d_model, activation="squared_relu")
        elif ffn_variant == "swiglu" and moe_variant == "baseline":
            block.moe = MoELayerWithActivation(d_model=d_model, activation="swiglu")

    return model


def load_compatible_weights(model, ckpt_path, attn_variant="baseline"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}

    # Filter out tensors whose shape does not match model's current parameter shape
    model_sd = model.state_dict()
    filtered_sd = {}
    mismatched = []
    for k, v in new_sd.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                filtered_sd[k] = v
            else:
                mismatched.append((k, v.shape, model_sd[k].shape))
        else:
            filtered_sd[k] = v

    # Load with strict=False to allow architectural replacements
    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if mismatched:
        print(f"  [Load Weights] Filtered {len(mismatched)} shape-mismatched parameters (e.g. {mismatched[0][0]})")
    print(f"  [Load Weights] Loaded from {os.path.basename(ckpt_path)} (matched: {len(filtered_sd) - len(unexpected)} keys, missing: {len(missing)})")

    # Apply specialized initialization for architectural variants where checkpoint had uniform values
    if attn_variant == "multi_timescale":
        init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        for block in model.blocks:
            block.attn.gamma_raw.data.copy_(torch.tensor(init_gammas, dtype=torch.float32))
        print("  [Init Decay] Re-initialized 16 heads across 4 timescale bands (0.85, 0.95, 0.99, 0.999)")
    elif attn_variant == "sliding_buffer":
        init_gammas = [2.94] * 8 + [5.50] * 8
        for block in model.blocks:
            block.attn.gamma_raw.data.copy_(torch.tensor(init_gammas, dtype=torch.float32))
        print("  [Init Decay] Re-initialized 16 heads across 2 timescale bands for hybrid buffer")
    elif attn_variant == "gated_write":
        init_gammas = [2.00] * 4 + [3.50] * 4 + [5.00] * 4 + [6.50] * 4
        for block in model.blocks:
            block.attn.gamma_raw.data.copy_(torch.tensor(init_gammas, dtype=torch.float32))
        print("  [Init Decay] Re-initialized 16 heads across 4 timescale bands for gated write")

    return model


@torch.inference_mode()
def evaluate_holdout(model, val_tokens, num_windows=50, seq_len=512, seed=42):
    model.eval()
    model.reset_state()
    max_start = len(val_tokens) - seq_len - 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    window_starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()

    ce_losses = []
    for start in window_starts:
        x = val_tokens[start : start + seq_len].unsqueeze(0)
        y = val_tokens[start + 1 : start + seq_len + 1].unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, targets=y, persist_state=False)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        ce_losses.append(ce.item())

    mean_ce = statistics.mean(ce_losses)
    ppl = math.exp(min(mean_ce, 100.0))
    model.train()
    model.reset_state()
    return mean_ce, ppl


def run_track_experiment(
    exp_id: str,
    attn_variant: str = "baseline",
    ffn_variant: str = "baseline",
    moe_variant: str = "baseline",
    ternary_variant: str = "baseline",
    use_muon: bool = False,
    steps: int = 50,
    lr: float = 5e-5,
    seq_len: int = 512,
):
    print("\n" + "=" * 85)
    print(f"EXPERIMENT: {exp_id}")
    print(f"  Attn: {attn_variant} | MoE: {moe_variant} | FFN: {ffn_variant} | Muon: {use_muon} | Steps: {steps}")
    print("=" * 85)

    device = "cuda"
    enc = tiktoken.get_encoding("gpt2")

    # Load tokens
    corpus_path = os.path.join(JARVIS_ENGINE, "data_clean.txt")
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        train_text = f.read()
    train_tokens = torch.tensor(enc.encode(train_text, allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device)

    val_path = os.path.join(JARVIS_ENGINE, "fresh_holdout.txt")
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_text = f.read()
    val_tokens = torch.tensor(enc.encode(val_text), dtype=torch.long, device=device)

    # Build model
    model = build_custom_jarvis(
        attn_variant=attn_variant,
        ffn_variant=ffn_variant,
        moe_variant=moe_variant,
        ternary_variant=ternary_variant,
        max_seq_len=seq_len,
    ).to(device)

    # Load best checkpoint weights
    ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(JARVIS_ENGINE, "ckpt_step_0004209.pt")
    load_compatible_weights(model, ckpt_path, attn_variant=attn_variant)

    # Initial evaluation
    init_ce, init_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=seq_len)
    print(f"Step 000 (Initial): Holdout CE = {init_ce:.4f} | PPL = {init_ppl:.2f}")

    # Optimizer setup
    if use_muon:
        muon_opt, adamw_opt = create_hybrid_optimizer(model, muon_lr=0.01, adamw_lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

    _offsets = torch.arange(seq_len, device=device)
    def get_batch():
        ix = torch.randint(0, len(train_tokens) - seq_len - 1, (2,), device=device)
        idx = ix.unsqueeze(1) + _offsets
        return train_tokens[idx], train_tokens[idx + 1]

    model.train()
    t_start = time.perf_counter()

    for step in range(1, steps + 1):
        if use_muon:
            muon_opt.zero_grad()
            adamw_opt.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)

        for _ in range(4):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
            (loss / 4).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if use_muon:
            muon_opt.step()
            adamw_opt.step()
        else:
            optimizer.step()

        # Update dynamic expert bias for aux-free MoE
        if moe_variant == "aux_free_bias":
            for m in model.modules():
                if isinstance(m, AuxFreeBiasMoELayer):
                    m.update_bias()

        if step % 25 == 0 or step == steps:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t_start
            tok_s = (step * 8 * seq_len) / max(dt, 1e-4)
            val_ce, val_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=seq_len)
            print(f"  Step {step:03d}/{steps}: Holdout CE = {val_ce:.4f} | PPL = {val_ppl:.2f} | {tok_s:.0f} tok/s")

    # Evaluate Needle-in-a-Haystack retrieval
    print("\nRunning Needle-in-a-Haystack retrieval...")
    needle_results = evaluate_needle_retrieval(model, enc, distances=[16, 64, 128, 256, 512], num_trials=5)

    final_ce, final_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=seq_len)
    ce_delta = final_ce - init_ce
    ppl_delta = final_ppl - init_ppl

    report = {
        "experiment_id": exp_id,
        "attn_variant": attn_variant,
        "ffn_variant": ffn_variant,
        "moe_variant": moe_variant,
        "use_muon": use_muon,
        "steps": steps,
        "initial_holdout_ce": init_ce,
        "final_holdout_ce": final_ce,
        "initial_holdout_ppl": init_ppl,
        "final_holdout_ppl": final_ppl,
        "ce_delta": ce_delta,
        "ppl_delta": ppl_delta,
        "needle_retrieval": needle_results,
    }

    out_file = os.path.join(ARCH_DIR, f"result_{exp_id}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save checkpoint
    ckpt_save = os.path.join(ARCH_DIR, f"ckpt_{exp_id}.pt")
    torch.save({"step": 4284 + steps, "model_state_dict": model.state_dict(), "val_loss": final_ce}, ckpt_save)

    print(f"\n[OK] {exp_id} Complete: CE Delta={ce_delta:+.4f} | PPL Delta={ppl_delta:+.2f}")
    return report


def check_existing_result(exp_id):
    path = os.path.join(ARCH_DIR, f"result_{exp_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [Skip] {exp_id} already completed (Final Holdout CE: {data['final_holdout_ce']:.4f})")
        return data
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="remaining", choices=["attention", "ffn", "moe", "muon", "remaining", "all"])
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    results = {}

    if args.track in ["attention", "all"]:
        for exp_name, variant in [("exp_multi_timescale", "multi_timescale"), ("exp_sliding_buffer", "sliding_buffer"), ("exp_gated_write", "gated_write")]:
            r = check_existing_result(exp_name)
            if r is None:
                r = run_track_experiment(exp_name, attn_variant=variant, steps=args.steps)
            results[exp_name] = r

    if args.track in ["ffn", "all"]:
        r_sq = check_existing_result("exp_squared_relu")
        if r_sq is None:
            r_sq = run_track_experiment("exp_squared_relu", ffn_variant="squared_relu", steps=args.steps)
        results["exp_squared_relu"] = r_sq

    if args.track in ["ffn", "remaining", "all"]:
        r_sw = check_existing_result("exp_swiglu")
        if r_sw is None:
            r_sw = run_track_experiment("exp_swiglu", ffn_variant="swiglu", steps=args.steps)
        results["exp_swiglu"] = r_sw

    if args.track in ["moe", "remaining", "all"]:
        r_moe = check_existing_result("exp_deepseek_shared")
        if r_moe is None:
            r_moe = run_track_experiment("exp_deepseek_shared", moe_variant="deepseek_shared", steps=args.steps)
        results["exp_deepseek_shared"] = r_moe

        r_bias = check_existing_result("exp_aux_free_bias")
        if r_bias is None:
            r_bias = run_track_experiment("exp_aux_free_bias", moe_variant="aux_free_bias", steps=args.steps)
        results["exp_aux_free_bias"] = r_bias

    if args.track in ["muon", "remaining", "all"]:
        r_muon = check_existing_result("exp_muon_hybrid")
        if r_muon is None:
            r_muon = run_track_experiment("exp_muon_hybrid", use_muon=True, steps=args.steps)
        results["exp_muon_hybrid"] = r_muon

    # Summary table
    print("\n" + "=" * 95)
    print("                  ARCHITECTURE MATRIX EXPERIMENTAL LEADERBOARD")
    print("=" * 95)
    print(f"{'Experiment ID':<25} | {'Initial CE':<10} | {'Final CE':<10} | {'CE Delta':<10} | {'PPL Delta':<10}")
    print("-" * 95)
    for k, v in results.items():
        print(f"{k:<25} | {v['initial_holdout_ce']:<10.4f} | {v['final_holdout_ce']:<10.4f} | {v['ce_delta']:<+10.4f} | {v['ppl_delta']:<+10.2f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
