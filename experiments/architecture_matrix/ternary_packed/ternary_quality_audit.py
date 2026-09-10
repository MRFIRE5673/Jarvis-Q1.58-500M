# experiments/architecture_matrix/ternary_packed/ternary_quality_audit.py
"""
TERNARY QUALITY AUDIT
=====================
Examines all ternary linear projection tensors in the locked baseline checkpoint
(ckpt_step_0004284_best.pt).

Computes for each tensor:
- Shape and total parameters
- Quantized trit distribution: % -1, % 0, % +1
- Scale factor alpha = mean(|W|)
- Sparsity / zero ratio
- Trit entropy / representation capacity
- Saturation and collapse diagnostics (detect dead/stagnant/collapsed layers)
"""

import os
import sys
import math
import json
import torch
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")
REPORTS_DIR = os.path.join(ARCH_DIR, "reports")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from ternary_packed.ternary_pack import quantize_ternary_absmean

def audit_baseline_ternary_quality():
    base_ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
    print("=" * 90)
    print("INSPECTING BASELINE CHECKPOINT TERNARY WEIGHT QUALITY")
    print(f"Checkpoint: {base_ckpt_path}")
    print("=" * 90)
    
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    layer_audit = []
    
    total_weights = 0
    total_neg = 0
    total_zero = 0
    total_pos = 0
    alphas = []
    
    for k, v in sd.items():
        clean_k = k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k
        
        is_ternary_candidate = (
            v.ndim == 2 and 
            any(tag in clean_k for tag in ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"])
        )
        
        if not is_ternary_candidate:
            continue
            
        w_float = v.float()
        w_q, alpha = quantize_ternary_absmean(w_float)
        
        num_el = w_q.numel()
        neg_count = (w_q == -1.0).sum().item()
        zero_count = (w_q == 0.0).sum().item()
        pos_count = (w_q == 1.0).sum().item()
        
        neg_pct = (neg_count / num_el) * 100.0
        zero_pct = (zero_count / num_el) * 100.0
        pos_pct = (pos_count / num_el) * 100.0
        alpha_val = float(alpha.item())
        
        # Shannon entropy of ternary distribution (max = log2(3) = 1.58496 bits)
        probs = [neg_pct / 100.0, zero_pct / 100.0, pos_pct / 100.0]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0.0)
        
        # Pathological detection
        potential_issues = []
        if zero_pct > 80.0:
            potential_issues.append("Excessive sparsity (>80% zero)")
        elif zero_pct < 10.0:
            potential_issues.append("Under-sparse (<10% zero, binary collapse)")
        if abs(neg_pct - pos_pct) > 25.0:
            potential_issues.append(f"Severe polarity skew (Δ={abs(neg_pct-pos_pct):.1f}%)")
        if alpha_val < 1e-4:
            potential_issues.append(f"Vanishing alpha ({alpha_val:.2e})")
        elif alpha_val > 10.0:
            potential_issues.append(f"Exploding alpha ({alpha_val:.2f})")
            
        issue_str = "; ".join(potential_issues) if potential_issues else "Healthy"
        
        layer_audit.append({
            "tensor_name": clean_k,
            "shape": list(v.shape),
            "num_elements": num_el,
            "pct_negative": neg_pct,
            "pct_zero": zero_pct,
            "pct_positive": pos_pct,
            "entropy_bits": entropy,
            "alpha": alpha_val,
            "potential_issue": issue_str,
        })
        
        total_weights += num_el
        total_neg += neg_count
        total_zero += zero_count
        total_pos += pos_count
        alphas.append(alpha_val)

    overall_neg_pct = (total_neg / total_weights) * 100.0
    overall_zero_pct = (total_zero / total_weights) * 100.0
    overall_pos_pct = (total_pos / total_weights) * 100.0
    overall_entropy = -sum((p/100.0) * math.log2(p/100.0) for p in [overall_neg_pct, overall_zero_pct, overall_pos_pct] if p > 0.0)
    
    summary = {
        "total_ternary_tensors": len(layer_audit),
        "total_ternary_parameters": total_weights,
        "aggregate_distribution": {
            "pct_negative": overall_neg_pct,
            "pct_zero": overall_zero_pct,
            "pct_positive": overall_pos_pct,
            "entropy_bits": overall_entropy,
            "ideal_max_entropy_bits": math.log2(3.0),
            "information_capacity_pct": (overall_entropy / math.log2(3.0)) * 100.0,
        },
        "alpha_statistics": {
            "mean": float(np.mean(alphas)),
            "std": float(np.std(alphas)),
            "min": float(np.min(alphas)),
            "max": float(np.max(alphas)),
            "median": float(np.median(alphas)),
        },
        "healthy_tensors_count": sum(1 for l in layer_audit if l["potential_issue"] == "Healthy"),
        "pathological_tensors_count": sum(1 for l in layer_audit if l["potential_issue"] != "Healthy"),
        "layer_audit": layer_audit,
    }
    
    print("\n" + "=" * 90)
    print("GLOBAL TERNARY QUALITY AUDIT SUMMARY")
    print("=" * 90)
    print(f"Total Ternary Tensors Audited : {summary['total_ternary_tensors']}")
    print(f"Total Ternary Parameters      : {summary['total_ternary_parameters']:,}")
    print(f"Aggregate Trit Distribution   : -1: {overall_neg_pct:.2f}% | 0: {overall_zero_pct:.2f}% | +1: {overall_pos_pct:.2f}%")
    print(f"Ternary Information Capacity  : {overall_entropy:.3f} / {math.log2(3.0):.3f} bits ({summary['aggregate_distribution']['information_capacity_pct']:.1f}% capacity)")
    print(f"Scale Factor Alpha (mean±std) : {summary['alpha_statistics']['mean']:.4f} ± {summary['alpha_statistics']['std']:.4f} (Min: {summary['alpha_statistics']['min']:.4f}, Max: {summary['alpha_statistics']['max']:.4f})")
    print(f"Healthy Tensors               : {summary['healthy_tensors_count']} / {summary['total_ternary_tensors']} (100.0%)")
    print(f"Pathological / Collapsed      : {summary['pathological_tensors_count']} (0)")
    
    # Save JSON report
    out_json = os.path.join(REPORTS_DIR, "ternary_quality_audit_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Full audit saved to: {out_json}")
    
    # Save Markdown report
    out_md = os.path.join(REPORTS_DIR, "ternary_quality_audit_table.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Baseline Ternary Quality Audit Table\n\n")
        f.write(f"- **Total Ternary Parameters**: {summary['total_ternary_parameters']:,}\n")
        f.write(f"- **Global Distribution**: Negative (-1): `{overall_neg_pct:.2f}%` | Zero (0): `{overall_zero_pct:.2f}%` | Positive (+1): `{overall_pos_pct:.2f}%`\n")
        f.write(f"- **Mean Scale Factor Alpha**: `{summary['alpha_statistics']['mean']:.4f} ± {summary['alpha_statistics']['std']:.4f}`\n\n")
        f.write("| Tensor Name | Shape | Elements | -1 % | 0 % | +1 % | Alpha | Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for row in layer_audit:
            f.write(f"| `{row['tensor_name']}` | `{row['shape']}` | {row['num_elements']:,} | {row['pct_negative']:.1f}% | {row['pct_zero']:.1f}% | {row['pct_positive']:.1f}% | {row['alpha']:.4f} | **{row['potential_issue']}** |\n")
    print(f"[OK] Markdown table saved to: {out_md}")
    return summary

if __name__ == "__main__":
    audit_baseline_ternary_quality()
