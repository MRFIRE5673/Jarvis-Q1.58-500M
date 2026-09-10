# experiments/architecture_matrix/ternary_packed/ternary_pack.py
"""
REAL 1.58-BIT / TERNARY PACKED STORAGE FOUNDATION
=================================================
Implements bit-level packed integer storage for ternary weights in {-1, 0, +1}.

Encoding (2-bit per trit):
  00 (0) =  0
  01 (1) = +1
  10 (2) = -1
  11 (3) = reserved / padding

Packing formats:
1. UINT8:  4 trits per byte   (4x compression over INT8, 8x over FP16, 16x over FP32)
2. UINT32: 16 trits per uint32 (vectorized tensor operations)

Provides:
- pack_ternary_tensor(w, alpha=None)
- unpack_ternary_tensor(packed_obj)
- round-trip exactness validation
- state_dict checkpoint conversion and export
- benchmark: storage, packing, unpacking, CPU/GPU footprint
"""

import os
import sys
import time
import math
import json
import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Vectorized shift tensors
SHIFTS_UINT8 = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
SHIFTS_UINT32 = torch.tensor([i * 2 for i in range(16)], dtype=torch.int64)
LOOKUP_MAP = torch.tensor([0.0, 1.0, -1.0, 0.0], dtype=torch.float32)


def quantize_ternary_absmean(weight: torch.Tensor):
    """
    Paper-faithful AbsMean quantization (BitNet 1.58b / Jarvis Eq. 3):
    alpha = mean(|W|)
    W_q = round(clamp(W / alpha, -1, 1))
    """
    alpha = weight.abs().mean().clamp(min=1e-8)
    w_scaled = weight / alpha
    w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))
    return w_q, alpha


def pack_ternary_uint8(w_q: torch.Tensor):
    """
    Packs a ternary tensor w_q in {-1, 0, +1} into uint8 storage (4 trits / byte).
    
    Args:
        w_q: Tensor of shape (..., K) with values in {-1, 0, 1}.
    Returns:
        packed: ByteTensor of shape (..., ceil(K/4))
        orig_shape: tuple of original tensor shape
    """
    orig_shape = w_q.shape
    orig_k = orig_shape[-1]
    pad_k = (4 - (orig_k % 4)) % 4
    
    device = w_q.device
    if pad_k > 0:
        pad_shape = list(orig_shape)
        pad_shape[-1] = pad_k
        w_padded = torch.cat([w_q, torch.zeros(pad_shape, dtype=w_q.dtype, device=device)], dim=-1)
    else:
        w_padded = w_q
        
    flat_w = w_padded.reshape(-1, 4)
    # Map {-1: 2, 0: 0, 1: 1}
    codes = torch.zeros_like(flat_w, dtype=torch.uint8)
    codes = torch.where(flat_w == 1.0, torch.tensor(1, dtype=torch.uint8, device=device), codes)
    codes = torch.where(flat_w == -1.0, torch.tensor(2, dtype=torch.uint8, device=device), codes)
    
    packed = (
        (codes[:, 0]) |
        (codes[:, 1] << 2) |
        (codes[:, 2] << 4) |
        (codes[:, 3] << 6)
    )
    
    new_shape = list(orig_shape[:-1]) + [w_padded.shape[-1] // 4]
    return packed.view(new_shape), orig_shape


def unpack_ternary_uint8(packed: torch.Tensor, orig_shape: tuple, dtype=torch.bfloat16):
    """
    Unpacks a uint8 tensor back to floating point ternary values {-1, 0, +1}.
    100% vectorized, zero Python loops.
    """
    device = packed.device
    flat_p = packed.reshape(-1, 1).to(torch.int32)
    shifts = SHIFTS_UINT8.to(device)
    
    # Extract 2-bit codes: (N, 4)
    codes = (flat_p >> shifts) & 0x03
    
    # Lookup values: 0->0.0, 1->1.0, 2->-1.0, 3->0.0
    lookup = LOOKUP_MAP.to(dtype=dtype, device=device)
    unpacked_flat = lookup[codes.long()]
    
    # Reshape and trim padding
    orig_k = orig_shape[-1]
    padded_k = ((orig_k + 3) // 4) * 4
    inter_shape = list(orig_shape[:-1]) + [padded_k]
    unpacked = unpacked_flat.reshape(inter_shape)
    
    if padded_k > orig_k:
        unpacked = unpacked[..., :orig_k]
        
    return unpacked.contiguous()


def pack_ternary_uint32(w_q: torch.Tensor):
    """
    Packs a ternary tensor into uint32 storage (16 trits / uint32).
    Optimal for 32-bit register operations and CUDA thread memory access.
    """
    orig_shape = w_q.shape
    orig_k = orig_shape[-1]
    pad_k = (16 - (orig_k % 16)) % 16
    
    device = w_q.device
    if pad_k > 0:
        pad_shape = list(orig_shape)
        pad_shape[-1] = pad_k
        w_padded = torch.cat([w_q, torch.zeros(pad_shape, dtype=w_q.dtype, device=device)], dim=-1)
    else:
        w_padded = w_q
        
    flat_w = w_padded.reshape(-1, 16)
    codes = torch.zeros_like(flat_w, dtype=torch.int64)
    codes = torch.where(flat_w == 1.0, torch.tensor(1, dtype=torch.int64, device=device), codes)
    codes = torch.where(flat_w == -1.0, torch.tensor(2, dtype=torch.int64, device=device), codes)
    
    shifts = SHIFTS_UINT32.to(device)
    packed_words = torch.sum(codes << shifts, dim=-1).to(torch.int32)
    
    new_shape = list(orig_shape[:-1]) + [w_padded.shape[-1] // 16]
    return packed_words.view(new_shape), orig_shape


def unpack_ternary_uint32(packed: torch.Tensor, orig_shape: tuple, dtype=torch.bfloat16):
    """
    Unpacks a uint32 tensor back to floating point ternary values {-1, 0, +1}.
    """
    device = packed.device
    flat_p = packed.reshape(-1, 1).to(torch.int64)
    shifts = SHIFTS_UINT32.to(device)
    
    codes = (flat_p >> shifts) & 0x03
    lookup = LOOKUP_MAP.to(dtype=dtype, device=device)
    unpacked_flat = lookup[codes.long()]
    
    orig_k = orig_shape[-1]
    padded_k = ((orig_k + 15) // 16) * 16
    inter_shape = list(orig_shape[:-1]) + [padded_k]
    unpacked = unpacked_flat.reshape(inter_shape)
    
    if padded_k > orig_k:
        unpacked = unpacked[..., :orig_k]
        
    return unpacked.contiguous()


class PackedTernaryLinear(torch.nn.Module):
    """
    Inference-ready Linear Layer with Packed 1.58-bit Weights.
    Stores weights packed in uint8 (2-bits/trit) with FP32/BF16 scale factor alpha.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.packed_k = (in_features + 3) // 4
        
        self.register_buffer("packed_weight", torch.zeros((out_features, self.packed_k), dtype=torch.uint8))
        self.register_buffer("alpha", torch.tensor(1.0, dtype=torch.float32))
        self.orig_shape = (out_features, in_features)
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    @classmethod
    def from_float_weight(cls, weight: torch.Tensor, bias: torch.Tensor = None):
        out_features, in_features = weight.shape
        layer = cls(in_features, out_features, bias=(bias is not None))
        w_q, alpha = quantize_ternary_absmean(weight)
        packed, _ = pack_ternary_uint8(w_q)
        layer.packed_weight.copy_(packed)
        layer.alpha.copy_(alpha.float())
        if bias is not None:
            layer.bias.copy_(bias.bfloat16())
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically unpack to current execution precision
        w_unpacked = unpack_ternary_uint8(self.packed_weight, self.orig_shape, dtype=x.dtype)
        # Scaled linear
        w_eff = w_unpacked * self.alpha.to(x.dtype)
        return torch.nn.functional.linear(x, w_eff, self.bias)


def test_roundtrip_correctness():
    print("=" * 80)
    print("TESTING 1.58-BIT PACKING ROUND-TRIP CORRECTNESS")
    print("=" * 80)
    
    test_shapes = [
        (64, 64),
        (1024, 1024),
        (2048, 1024),
        (1024, 2048),
        (1024, 64),
        (16, 1023), # Non-multiple of 4 & 16 to test padding boundary
    ]
    
    for shape in test_shapes:
        # Random ternary tensor
        trits = torch.randint(-1, 2, shape, dtype=torch.float32, device=DEVICE)
        
        # UINT8 test
        packed_u8, s_u8 = pack_ternary_uint8(trits)
        unpacked_u8 = unpack_ternary_uint8(packed_u8, s_u8, dtype=torch.float32)
        diff_u8 = (trits - unpacked_u8).abs().max().item()
        assert diff_u8 == 0.0, f"UINT8 Round-trip error on shape {shape}: max diff = {diff_u8}"
        
        # UINT32 test
        packed_u32, s_u32 = pack_ternary_uint32(trits)
        unpacked_u32 = unpack_ternary_uint32(packed_u32, s_u32, dtype=torch.float32)
        diff_u32 = (trits - unpacked_u32).abs().max().item()
        assert diff_u32 == 0.0, f"UINT32 Round-trip error on shape {shape}: max diff = {diff_u32}"
        
        print(f"  Shape {str(shape):15s} | UINT8 bytes: {packed_u8.numel():8d} | UINT32 words: {packed_u32.numel():6d} | Error: 0.0000 [PASS]")

    print("\n[OK] All round-trip tests passed with 100% mathematical exactness!")


def benchmark_storage_and_latency():
    print("\n" + "=" * 80)
    print("BENCHMARKING TERNARY STORAGE & RECONSTRUCTION LATENCY")
    print("=" * 80)
    
    # Realistic layer dimensions in Jarvis-600M:
    # d_model=1024, MLP intermediate=2048
    N, K = 2048, 1024
    num_weights = N * K
    
    # 1. Memory footprints
    fp32_bytes = num_weights * 4
    bf16_bytes = num_weights * 2
    int8_bytes = num_weights * 1
    packed_u8_bytes = (num_weights // 4) + 4 # packed weight + 4-byte float alpha
    
    print(f"Matrix Dimensions: [{N} x {K}] ({num_weights:,} elements)")
    print(f"  FP32 Storage:         {fp32_bytes / 1024:8.1f} KB (Reference: 1.00x)")
    print(f"  BF16 Storage:         {bf16_bytes / 1024:8.1f} KB (Compression: 2.00x)")
    print(f"  INT8 Dense Storage:   {int8_bytes / 1024:8.1f} KB (Compression: 4.00x)")
    print(f"  Packed 2-bit Storage: {packed_u8_bytes / 1024:8.1f} KB (Compression: {fp32_bytes / packed_u8_bytes:.2f}x vs FP32, {bf16_bytes / packed_u8_bytes:.2f}x vs BF16)")
    
    # 2. Timing benchmarks
    w = torch.randn((N, K), device=DEVICE, dtype=torch.bfloat16)
    w_q, alpha = quantize_ternary_absmean(w)
    
    # Warmup
    for _ in range(50):
        p, s = pack_ternary_uint8(w_q)
        u = unpack_ternary_uint8(p, s)
    torch.cuda.synchronize(DEVICE)
    
    # Packing benchmark
    trials = 200
    t0 = time.perf_counter()
    for _ in range(trials):
        p, s = pack_ternary_uint8(w_q)
    torch.cuda.synchronize(DEVICE)
    pack_time_ms = ((time.perf_counter() - t0) / trials) * 1000.0
    
    # Unpacking benchmark
    t0 = time.perf_counter()
    for _ in range(trials):
        u = unpack_ternary_uint8(p, s)
    torch.cuda.synchronize(DEVICE)
    unpack_time_ms = ((time.perf_counter() - t0) / trials) * 1000.0
    
    print(f"\nReconstruction Performance on {DEVICE}:")
    print(f"  Pack Latency   : {pack_time_ms:.3f} ms ({num_weights / (pack_time_ms * 1e-3) / 1e6:.1f} M weights/s)")
    print(f"  Unpack Latency : {unpack_time_ms:.3f} ms ({num_weights / (unpack_time_ms * 1e-3) / 1e6:.1f} M weights/s)")
    
    return {
        "matrix_shape": [N, K],
        "total_elements": num_weights,
        "fp32_bytes": fp32_bytes,
        "bf16_bytes": bf16_bytes,
        "packed_bytes": packed_u8_bytes,
        "compression_ratio_vs_fp32": fp32_bytes / packed_u8_bytes,
        "compression_ratio_vs_bf16": bf16_bytes / packed_u8_bytes,
        "pack_time_ms": pack_time_ms,
        "unpack_time_ms": unpack_time_ms,
    }

if __name__ == "__main__":
    test_roundtrip_correctness()
    res = benchmark_storage_and_latency()
    
    out_json = os.path.join(os.path.dirname(__file__), "..", "reports", "ternary_packed_storage_benchmark.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[OK] Benchmark report saved to: {out_json}")
