# experiments/architecture_matrix/factorial_research_runner.py
"""
FACTORIAL & STAGED-ELIMINATION RESEARCH CONTROLLER FOR JARVIS-600M
===================================================================
Orchestrates controlled factorial experiments across five core memory mechanisms:
A = Adaptive Decay
B = Write Gate
C = Erase Gate
D = Gated Read
E = Local Buffer (W=16)

Fusion Options (when E is enabled):
1 = Additive (local + recurrent)
2 = Learned Per-Head Gate (default)
3 = Hierarchical Interaction

Instruments exact VRAM breakdown, gate activation distributions,
neutral initialization, and intermediate checkpointing every 25 steps.
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
from jarvis_model import Jarvis, RMSNorm
from modular_memory import ModularMemoryAttention, build_modular_jarvis
from evaluate_architecture import run_canonical_evaluation
from run_track_experiments import evaluate_holdout

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

    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        f.write("# JARVIS-600M RESEARCH LEADERBOARD\n\n")
        f.write("| Experiment ID | Category | Initial CE | Final CE | $\\Delta$ CE | Final PPL | Throughput | Needle Rank @ 64 |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for entry in sorted(db["leaderboard"], key=lambda x: x.get("final_ce", 999.0)):
            f.write(f"| `{entry['experiment_id']}` | {entry.get('category', 'Architecture')} | {entry.get('initial_ce', 0.0):.4f} | **{entry.get('final_ce', 0.0):.4f}** | {entry.get('ce_delta', 0.0):+.4f} | {entry.get('final_ppl', 0.0):.2f} | {entry.get('tok_s', 0):.0f} tok/s | {entry.get('needle_rank_64', 0):.1f} |\n")


def parse_combo_code(combo_str: str, fusion_option: int = 2):
    """
    Parses combo string like 'A', 'A+B', 'A+C+E', 'A+B+C+D+E'
    into configuration dictionary.
    """
    tokens = [t.strip().upper() for t in combo_str.replace(",", "+").split("+") if t.strip()]
    cfg = {
        "use_adaptive_decay": "A" in tokens,
        "use_write_gate": "B" in tokens,
        "use_erase_gate": "C" in tokens,
        "use_gated_read": "D" in tokens,
        "use_local_buffer": "E" in tokens,
        "local_window_size": 16,
        "fusion_option": fusion_option,
    }
    canonical_id = f"fact_{'+'.join(sorted(tokens))}"
    if cfg["use_local_buffer"] and fusion_option != 2:
        canonical_id += f"_fus{fusion_option}"
    return canonical_id, cfg


def get_vram_breakdown(model, optimizer=None):
    """Calculates detailed GPU VRAM profile in MB."""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    opt_mem = 0.0
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    opt_mem += (v.numel() * v.element_size()) / (1024 ** 2)

    act_mem = max(allocated - param_mem - opt_mem, 0.0)
    return {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "peak_mb": round(peak, 1),
        "param_mb": round(param_mem, 1),
        "optimizer_mb": round(opt_mem, 1),
        "activation_mb": round(act_mem, 1),
    }


def collect_model_diagnostics(model, val_tokens, device="cuda"):
    """
    Runs diagnostic forward pass on sample validation window to collect
    gate activation distributions and fusion statistics.
    """
    model.eval()
    toks = val_tokens[:512].unsqueeze(0)
    gammas, w_gates, e_gates, r_gates, local_norms, rec_norms, fusion_ratios = [], [], [], [], [], [], []

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model.tok_emb(toks)
            for block in model.blocks:
                normed_h = block.norm1(h)
                try:
                    attn_out, diag = block.attn(normed_h, collect_diagnostics=True)
                    gammas.append(diag["gamma_mean"])
                    w_gates.append(diag["write_gate_mean"])
                    e_gates.append(diag["erase_gate_mean"])
                    r_gates.append(diag["read_gate_mean"])
                    local_norms.append(diag["local_norm"])
                    rec_norms.append(diag["recurrent_norm"])
                    fusion_ratios.append(diag["fusion_ratio"])
                    h = h + attn_out
                except Exception:
                    h, _, _, _ = block(h)

    return {
        "mean_gamma": statistics.mean(gammas) if gammas else 0.95,
        "mean_write_gate": statistics.mean(w_gates) if w_gates else 1.0,
        "mean_erase_gate": statistics.mean(e_gates) if e_gates else 0.0,
        "mean_read_gate": statistics.mean(r_gates) if r_gates else 1.0,
        "local_to_recurrent_norm_ratio": statistics.mean(local_norms) / max(statistics.mean(rec_norms), 1e-6) if local_norms else 0.0,
        "mean_fusion_gate": statistics.mean(fusion_ratios) if fusion_ratios else 0.0,
        "recurrent_state_norm": statistics.mean(rec_norms) if rec_norms else 0.0,
    }


def run_factorial_experiment(
    combo_str: str,
    fusion_option: int = 2,
    steps: int = 100,
    lr: float = 5e-5,
    seed: int = 42,
    force: bool = False,
):
    exp_id, cfg = parse_combo_code(combo_str, fusion_option)
    if seed != 42:
        exp_id += f"_s{seed}"
        category = f"Factorial ({combo_str}, seed={seed})"
    else:
        category = f"Factorial ({combo_str})"

    print("\n" + "=" * 90)
    print(f"FACTORIAL EXPERIMENT: {exp_id}")
    print(f"Components: {combo_str} | Fusion Option: {fusion_option} | Steps: {steps} | Seed: {seed}")
    print("=" * 90)

    ckpt_save = os.path.join(ARCH_DIR, f"ckpt_{exp_id}.pt")
    eval_json = os.path.join(ARCH_DIR, f"canonical_eval_{exp_id}.json")
    db = load_research_database()

    # 1. Skip check
    if not force and exp_id in db.get("experiments", {}) and os.path.exists(ckpt_save) and os.path.exists(eval_json):
        print(f"[SKIP] Experiment {exp_id} is already completed in research database.")
        return db["experiments"][exp_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    enc = tiktoken.get_encoding("gpt2")

    # Load data
    with open(os.path.join(JARVIS_ENGINE, "data_clean.txt"), "r", encoding="utf-8", errors="ignore") as f:
        train_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device)
    with open(os.path.join(JARVIS_ENGINE, "fresh_holdout.txt"), "r", encoding="utf-8", errors="ignore") as f:
        val_tokens = torch.tensor(enc.encode(f.read(), allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device)

    # Build model
    model = build_modular_jarvis(config_dict=cfg, max_seq_len=512).to(device)

    # 2. Check for intermediate checkpoints
    import glob
    inter_ckpts = sorted(glob.glob(os.path.join(ARCH_DIR, f"ckpt_{exp_id}_step*.pt")))
    start_step = 1

    if inter_ckpts and not force:
        latest_inter = inter_ckpts[-1]
        ckpt = torch.load(latest_inter, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        try:
            step_str = latest_inter.split("_step")[-1].replace(".pt", "")
            completed_step = int(step_str)
            start_step = completed_step + 1
            print(f"  [Resume] Resuming {exp_id} from {os.path.basename(latest_inter)} (step {completed_step})")
        except ValueError:
            pass
    else:
        # Load baseline weights
        ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        model_sd = model.state_dict()
        filtered_sd = {k: v for k, v in new_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)

        # Apply neutral initialization to newly introduced parameters
        for block in model.blocks:
            if hasattr(block.attn, "init_neutral"):
                block.attn.init_neutral()

        inherited_count = sum(p.numel() for k, p in filtered_sd.items())
        total_count = sum(p.numel() for p in model.parameters())
        new_count = total_count - inherited_count
        print(f"  [Init Weights] Inherited: {inherited_count:,} params | Newly Initialized (Neutral): {new_count:,} params")

    # Initial evaluation
    init_ce, init_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=512, seed=seed)
    print(f"Step 000 (Initial Holdout): CE = {init_ce:.4f} | PPL = {init_ppl:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    _offsets = torch.arange(512, device=device)

    def get_batch():
        ix = torch.randint(0, len(train_tokens) - 513, (2,), device=device)
        idx = ix.unsqueeze(1) + _offsets
        return train_tokens[idx], train_tokens[idx + 1]

    model.train()
    t_start = time.perf_counter()
    val_ce, val_ppl = init_ce, init_ppl
    best_ce = init_ce
    best_step = 0

    for step in range(start_step, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(4):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
            (loss / 4).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 25 == 0 or step == steps:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t_start
            tok_s = (step * 8 * 512) / max(dt, 1e-4)
            val_ce, val_ppl = evaluate_holdout(model, val_tokens, num_windows=50, seq_len=512, seed=seed)
            if val_ce < best_ce:
                best_ce = val_ce
                best_step = step
            print(f"  Step {step:03d}/{steps}: Holdout CE = {val_ce:.4f} | PPL = {val_ppl:.2f} | {tok_s:.0f} tok/s (Best: {best_ce:.4f} @ step {best_step})", flush=True)

            save_sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}
            inter_save = os.path.join(ARCH_DIR, f"ckpt_{exp_id}_step{step:03d}.pt")
            # Remove earlier step checkpoints to preserve disk space
            for old_step in glob.glob(os.path.join(ARCH_DIR, f"ckpt_{exp_id}_step*.pt")):
                if old_step != inter_save:
                    try:
                        os.remove(old_step)
                    except OSError:
                        pass
            torch.save({
                "step": 4284 + step,
                "model_state_dict": save_sd,
                "val_loss": val_ce,
                "val_ppl": val_ppl,
            }, inter_save)

    # Save final checkpoint in BF16
    save_sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"step": 4284 + steps, "model_state_dict": save_sd, "val_loss": val_ce}, ckpt_save)
    print(f"  -> Saved final experiment checkpoint (BF16): {os.path.basename(ckpt_save)}")

    # VRAM and Gate Diagnostics
    vram_stats = get_vram_breakdown(model, optimizer)
    gate_diags = collect_model_diagnostics(model, val_tokens, device=device)
    print(f"  [VRAM Profile] Peak: {vram_stats['peak_mb']} MB (Param: {vram_stats['param_mb']} MB, Act: {vram_stats['activation_mb']} MB, Opt: {vram_stats['optimizer_mb']} MB)")
    print(f"  [Gate Diags] Gamma: {gate_diags['mean_gamma']:.3f} | Write: {gate_diags['mean_write_gate']:.3f} | Erase: {gate_diags['mean_erase_gate']:.3f} | Read: {gate_diags['mean_read_gate']:.3f} | Fusion: {gate_diags['mean_fusion_gate']:.3f}")

    # Canonical Standardized Evaluation
    canonical_report = run_canonical_evaluation(
        model=model,
        model_type="baseline",
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
        "experiment_id": exp_id,
        "category": category,
        "components": combo_str,
        "fusion_option": fusion_option,
        "seed": seed,
        "initial_ce": init_ce,
        "final_ce": final_ce,
        "ce_delta": final_ce - init_ce,
        "best_ce": best_ce,
        "best_step": best_step,
        "final_ppl": final_ppl,
        "tok_s": prefill_speed,
        "needle_rank_64": needle_rank_64,
        "vram": vram_stats,
        "diagnostics": gate_diags,
        "checkpoint": ckpt_save,
        "canonical_report": eval_json,
        "status": "COMPLETED",
    }
    db["experiments"][exp_id] = result_entry
    db["leaderboard"] = [e for e in db["leaderboard"] if e["experiment_id"] != exp_id]
    db["leaderboard"].append(result_entry)
    save_research_database(db)

    # Clean intermediate step checkpoints
    for old_inter in glob.glob(os.path.join(ARCH_DIR, f"ckpt_{exp_id}_step*.pt")):
        try:
            os.remove(old_inter)
        except OSError:
            pass

    print(f"[OK] Completed {exp_id}: Final CE = {final_ce:.4f} | PPL = {final_ppl:.2f} | Rank @ 64 = {needle_rank_64:.1f}")
    return result_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", type=str, required=True, help="Components to enable e.g. 'A', 'A+B', 'A+C+E'")
    parser.add_argument("--fusion", type=int, default=2, help="Fusion option (1=additive, 2=learned gate, 3=hierarchical)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_factorial_experiment(
        combo_str=args.combo,
        fusion_option=args.fusion,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
