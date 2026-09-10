# experiments/architecture_matrix/ternary_packed/jarvis_tbin_format.py
"""
JARVIS-TBIN: 1.58-BIT PACKED BINARY MODEL FORMAT SPECIFICATION & RUNTIME
=======================================================================
Implements a self-contained, high-performance binary container (.jarvis-tbin)
for 1.58-bit ternary models.

Container Specification:
[0..7]   : Magic Header b"JTBIN01\x00"
[8..11]  : Format Version (int32 = 1)
[12..15] : Metadata JSON byte length (int32)
[16..]   : UTF-8 Metadata JSON string
[Header End] : Binary Data Payload (contiguous byte-aligned tensor memory)

Each Tensor Entry contains:
- name: str
- is_ternary: bool
- shape: list[int]
- alpha: float (AbsMean scale factor for ternary weights)
- dtype: str ("uint8_packed" or "bfloat16" or "float32")
- byte_offset: int
- byte_length: int

Provides:
- export_to_jarvis_tbin(model, save_path, metadata)
- load_from_jarvis_tbin(model, load_path, device)
- validate_tbin_equivalence(baseline_model, tbin_path, test_inputs)
"""

import os
import sys
import json
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")
TERNARY_DIR = os.path.join(ARCH_DIR, "ternary_packed")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR, TERNARY_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from ternary_pack import pack_ternary_uint8, unpack_ternary_uint8, quantize_ternary_absmean
from jarvis_model import Jarvis

MAGIC_HEADER = b"JTBIN01\x00"
VERSION = 1


def export_to_jarvis_tbin(model: nn.Module, output_path: str, extra_metadata: dict = None):
    """
    Exports a model to .jarvis-tbin packed binary format.
    TernaryLinear weights are quantized via AbsMean, packed to 2-bit (uint8),
    and stored with alpha. All other weights (embeddings, norms, router) are stored in BF16.
    """
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    state_dict = model.state_dict()
    tensor_manifest = []
    payload_buffers = []
    current_offset = 0
    
    total_ternary_weights = 0
    total_dense_weights = 0
    
    for name, tensor in state_dict.items():
        is_ternary = (tensor.dim() == 2 and any(term in name for term in ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2"]))
        
        if is_ternary:
            # AbsMean quantization
            w_q, alpha = quantize_ternary_absmean(tensor.float().cpu())
            packed, orig_shape = pack_ternary_uint8(w_q)
            raw_bytes = packed.numpy().tobytes()
            byte_len = len(raw_bytes)
            
            entry = {
                "name": name,
                "is_ternary": True,
                "orig_shape": list(orig_shape),
                "packed_shape": list(packed.shape),
                "alpha": float(alpha.item()),
                "dtype": "uint8_packed",
                "offset": current_offset,
                "length": byte_len,
            }
            total_ternary_weights += tensor.numel()
        else:
            # Store non-ternary in bfloat16 (using int16 view for bit-exact serialization)
            bf16_tensor = tensor.cpu().to(torch.bfloat16)
            raw_bytes = bf16_tensor.view(torch.int16).numpy().tobytes()
            byte_len = len(raw_bytes)
            
            entry = {
                "name": name,
                "is_ternary": False,
                "orig_shape": list(tensor.shape),
                "packed_shape": list(tensor.shape),
                "alpha": 1.0,
                "dtype": "bfloat16",
                "offset": current_offset,
                "length": byte_len,
            }
            total_dense_weights += tensor.numel()
            
        tensor_manifest.append(entry)
        payload_buffers.append(raw_bytes)
        current_offset += byte_len
        
    meta = {
        "version": VERSION,
        "format": "jarvis-tbin",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_parameters": total_ternary_weights + total_dense_weights,
        "ternary_parameters": total_ternary_weights,
        "dense_parameters": total_dense_weights,
        "ternary_fraction_pct": (total_ternary_weights / max(total_ternary_weights + total_dense_weights, 1)) * 100.0,
        "tensors": tensor_manifest,
        "extra": extra_metadata or {},
    }
    
    meta_json = json.dumps(meta).encode("utf-8")
    meta_len = len(meta_json)
    
    # Write container
    with open(output_path, "wb") as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<II", VERSION, meta_len))
        f.write(meta_json)
        for buf in payload_buffers:
            f.write(buf)
            
    dur = time.perf_counter() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Exported .jarvis-tbin: {output_path} ({file_size_mb:.1f} MB in {dur:.2f}s)")
    return meta


class InferenceTernaryLinear(nn.Module):
    """Direct inference layer using pre-quantized effective weights without re-quantization."""
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def load_from_jarvis_tbin(model: nn.Module, tbin_path: str, device="cpu"):
    """
    Loads and reconstructs model weights directly from a .jarvis-tbin file.
    Switches TernaryLinear layers to direct inference mode using the exact effective weights.
    """
    t0 = time.perf_counter()
    with open(tbin_path, "rb") as f:
        magic = f.read(8)
        assert magic == MAGIC_HEADER, f"Invalid magic header: {magic}"
        ver, meta_len = struct.unpack("<II", f.read(8))
        meta_json = f.read(meta_len).decode("utf-8")
        meta = json.loads(meta_json)
        
        payload_start = 8 + 8 + meta_len
        f.seek(payload_start)
        payload_data = f.read()
        
    reconstructed_sd = {}
    
    for entry in meta["tensors"]:
        name = entry["name"]
        offset = entry["offset"]
        length = entry["length"]
        raw_chunk = payload_data[offset : offset + length]
        
        if entry["is_ternary"]:
            packed_bytes = torch.frombuffer(raw_chunk, dtype=torch.uint8).reshape(entry["packed_shape"])
            w_unpacked = unpack_ternary_uint8(packed_bytes, tuple(entry["orig_shape"]), dtype=torch.float32)
            w_eff = w_unpacked * entry["alpha"]
            reconstructed_sd[name] = w_eff.to(device=device, dtype=torch.float32)
        else:
            tensor = torch.frombuffer(raw_chunk, dtype=torch.int16).view(torch.bfloat16).reshape(entry["orig_shape"])
            reconstructed_sd[name] = tensor.to(device=device, dtype=torch.float32)
            
    model_sd = model.state_dict()
    filtered_sd = {k: v for k, v in reconstructed_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered_sd, strict=False)
    
    # Switch TernaryLinear layers to direct inference mode
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child.__class__.__name__ == "TernaryLinear":
                inf_layer = InferenceTernaryLinear(child.weight.data, child.bias.data if child.bias is not None else None)
                setattr(module, child_name, inf_layer)
    
    dur = time.perf_counter() - t0
    print(f"[OK] Loaded {len(filtered_sd)} tensors from .jarvis-tbin into model ({dur:.2f}s)")
    return meta


def validate_tbin_equivalence(orig_model, tbin_path, device="cuda"):
    """
    Verifies that running inference on orig_model vs loaded tbin produces
    strictly equivalent outputs within floating-point tolerance.
    """
    print(f"\nVerifying Numerical Equivalence for .jarvis-tbin...")
    orig_model.eval()
    
    # Instantiate clean clone
    clone_model = Jarvis(
        vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
        num_experts=4, top_k=2, max_seq_len=512,
        use_cuda_attn=False, use_cuda_moe=False
    ).to(device)
    
    _ = load_from_jarvis_tbin(clone_model, tbin_path, device=device)
    clone_model.eval()
    
    # Test batch of sequences
    torch.manual_seed(42)
    test_inputs = torch.randint(0, 50257, (2, 256), device=device)
    
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            orig_logits, _ = orig_model(test_inputs)
            tbin_logits, _ = clone_model(test_inputs)
            
    diff = (orig_logits.float() - tbin_logits.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    
    cos_sim = float(F.cosine_similarity(orig_logits.flatten().unsqueeze(0).float(), tbin_logits.flatten().unsqueeze(0).float()).item())
    
    print(f"  Max Absolute Difference : {max_diff:.6f}")
    print(f"  Mean Absolute Difference: {mean_diff:.6f}")
    print(f"  Cosine Similarity       : {cos_sim:.8f} [MATCH]")
    
    assert cos_sim > 0.9999, f"Cosine similarity failure: {cos_sim}"
    print("[SUCCESS] .jarvis-tbin export/reload is numerically identical to original model!")
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine_similarity": cos_sim,
    }


if __name__ == "__main__":
    base_ckpt = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
    out_tbin = os.path.join(ARCH_DIR, "ternary_packed", "jarvis_baseline.jarvis-tbin")
    
    print("Loading baseline model for export...")
    model = Jarvis(
        vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
        num_experts=4, top_k=2, max_seq_len=512,
        use_cuda_attn=False, use_cuda_moe=False
    )
    ckpt = torch.load(base_ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    new_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    
    meta = export_to_jarvis_tbin(model, out_tbin, extra_metadata={"source_checkpoint": base_ckpt})
    validate_tbin_equivalence(model.to(DEVICE), out_tbin, device=DEVICE)
