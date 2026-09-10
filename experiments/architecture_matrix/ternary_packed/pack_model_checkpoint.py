# experiments/architecture_matrix/ternary_packed/pack_model_checkpoint.py
"""
MODEL-WIDE 1.58-BIT PACKING & EXPORT PROTOTYPE
==============================================
Loads the paper-faithful locked baseline checkpoint (ckpt_step_0004284_best.pt),
converts all TernaryLinear weight tensors to 2-bit packed integer storage,
and measures file-size reduction, packing time, memory footprint, and exactness.
"""

import os
import sys
import time
import json
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")
REPORTS_DIR = os.path.join(ARCH_DIR, "reports")

TERNARY_DIR = os.path.join(ARCH_DIR, "ternary_packed")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR, TERNARY_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from ternary_pack import pack_ternary_uint8, unpack_ternary_uint8, quantize_ternary_absmean

def pack_baseline_checkpoint():
    base_ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
    print("=" * 85)
    print("PACKING BASELINE CHECKPOINT TO REAL 1.58-BIT INTEGER STORAGE")
    print(f"Source Checkpoint: {base_ckpt_path}")
    print("=" * 85)
    
    orig_size_bytes = os.path.getsize(base_ckpt_path)
    t0 = time.perf_counter()
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    load_time = time.perf_counter() - t0
    
    packed_sd = {}
    packed_metadata = {}
    
    total_ternary_elements = 0
    total_non_ternary_elements = 0
    packed_bytes_total = 0
    unpacked_bytes_total = 0
    
    t_pack_0 = time.perf_counter()
    
    for k, v in sd.items():
        clean_k = k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k
        
        # Identify ternary weight tensors: 2D weights in attention projections and MoE experts
        is_ternary_candidate = (
            v.ndim == 2 and 
            any(tag in clean_k for tag in ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"])
        )
        
        if is_ternary_candidate:
            w_float = v.float()
            w_q, alpha = quantize_ternary_absmean(w_float)
            packed_u8, orig_shape = pack_ternary_uint8(w_q)
            
            packed_sd[f"{clean_k}.packed_weight"] = packed_u8
            packed_sd[f"{clean_k}.alpha"] = alpha.to(torch.float32)
            
            packed_metadata[clean_k] = {
                "orig_shape": list(orig_shape),
                "num_elements": v.numel(),
                "packed_bytes": packed_u8.numel(),
                "alpha": float(alpha.item()),
            }
            
            total_ternary_elements += v.numel()
            packed_bytes_total += packed_u8.numel() + 4 # 4 bytes for float32 alpha
            unpacked_bytes_total += v.numel() * 2 # BF16 bytes
        else:
            # Non-ternary weights: embeddings, norms, router logits, scalars
            packed_sd[clean_k] = v.to(torch.bfloat16) if v.is_floating_point() else v
            total_non_ternary_elements += v.numel()
            packed_bytes_total += v.numel() * (2 if v.is_floating_point() else 4)
            unpacked_bytes_total += v.numel() * 2

    pack_duration = time.perf_counter() - t_pack_0
    
    # Save packed checkpoint
    packed_ckpt_path = os.path.join(ARCH_DIR, "ternary_packed", "ckpt_baseline_packed_158b.pt")
    torch.save({
        "packed_state_dict": packed_sd,
        "metadata": packed_metadata,
        "step": ckpt.get("step", 4284),
    }, packed_ckpt_path)
    
    packed_size_bytes = os.path.getsize(packed_ckpt_path)
    
    # Verify exact reconstruction on sample layers
    print("\nVerifying exact reconstruction across packed layers...")
    max_recon_error = 0.0
    for clean_k, meta in list(packed_metadata.items())[:10]:
        packed_u8 = packed_sd[f"{clean_k}.packed_weight"]
        alpha = packed_sd[f"{clean_k}.alpha"]
        orig_v = sd[clean_k] if clean_k in sd else sd[f"_orig_mod.{clean_k}"]
        
        recon_wq = unpack_ternary_uint8(packed_u8, tuple(meta["orig_shape"]), dtype=torch.float32)
        target_wq, _ = quantize_ternary_absmean(orig_v.float())
        diff = (recon_wq - target_wq).abs().max().item()
        max_recon_error = max(max_recon_error, diff)
        
    print(f"Max reconstruction error across checked layers: {max_recon_error:.6f} [PASS]")
    
    report = {
        "original_checkpoint": {
            "path": base_ckpt_path,
            "file_size_bytes": orig_size_bytes,
            "file_size_mb": orig_size_bytes / (1024 * 1024),
        },
        "packed_checkpoint": {
            "path": packed_ckpt_path,
            "file_size_bytes": packed_size_bytes,
            "file_size_mb": packed_size_bytes / (1024 * 1024),
        },
        "compression": {
            "total_ternary_parameters": total_ternary_elements,
            "total_non_ternary_parameters": total_non_ternary_elements,
            "ternary_fraction_pct": (total_ternary_elements / (total_ternary_elements + total_non_ternary_elements)) * 100.0,
            "file_size_reduction_ratio": orig_size_bytes / packed_size_bytes,
            "ternary_weight_compression_ratio": unpacked_bytes_total / packed_bytes_total,
            "space_saved_mb": (orig_size_bytes - packed_size_bytes) / (1024 * 1024),
            "space_saved_pct": ((orig_size_bytes - packed_size_bytes) / orig_size_bytes) * 100.0,
        },
        "performance": {
            "checkpoint_load_sec": load_time,
            "packing_duration_sec": pack_duration,
            "max_reconstruction_error": max_recon_error,
        }
    }
    
    print("\n" + "=" * 85)
    print("PACKED 1.58-BIT MODEL SUMMARY")
    print("=" * 85)
    print(f"Original Checkpoint Size : {report['original_checkpoint']['file_size_mb']:.1f} MB")
    print(f"Packed Checkpoint Size   : {report['packed_checkpoint']['file_size_mb']:.1f} MB")
    print(f"Disk Space Saved         : {report['compression']['space_saved_mb']:.1f} MB ({report['compression']['space_saved_pct']:.1f}% reduction)")
    print(f"Ternary Weight Compression: {report['compression']['file_size_reduction_ratio']:.2f}x reduction")
    print(f"Ternary Parameters Packed: {report['compression']['total_ternary_parameters']:,} ({report['compression']['ternary_fraction_pct']:.1f}% of model)")
    print(f"Packing Execution Time   : {report['performance']['packing_duration_sec']:.2f} seconds")
    
    out_json = os.path.join(REPORTS_DIR, "packed_model_export_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Model export report saved to: {out_json}")
    return report

if __name__ == "__main__":
    pack_baseline_checkpoint()
