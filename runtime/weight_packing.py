# runtime/weight_packing.py
"""
Packed Ternary Representation Engine (BitNet 1.58b / 2-bit Packing)
===================================================================
Packs ternary weights {-1, 0, +1} at 4 weights per uint8 byte (2 bits/weight):
- 00 (0) ->  0
- 01 (1) -> +1
- 10 (2) -> -1
- 11 (3) -> Reserved / unassigned

Achieves:
- 16x storage and PCIe bandwidth compression vs FP32
- 8x compression vs BF16
- Enables complete 3.2B parameter model to stream in 800MB (29ms over PCIe Gen 4)
"""

import math
import time
import torch
import torch.nn as nn

def pack_ternary_tensor(w_ternary: torch.Tensor):
    """
    Packs a ternary tensor of values in {-1.0, 0.0, 1.0} into 2-bit uint8.
    w_ternary shape must have inner dimension divisible by 4.
    Returns:
        packed_bytes: uint8 tensor of shape (*shape[:-1], shape[-1] // 4)
        original_shape: tuple
    """
    orig_shape = w_ternary.shape
    assert orig_shape[-1] % 4 == 0, "Last dimension must be divisible by 4"
    flat = w_ternary.reshape(-1, 4)

    # Encode {-1 -> 2, 0 -> 0, +1 -> 1}
    # Using integer bit arithmetic
    code = torch.zeros_like(flat, dtype=torch.uint8)
    code[flat == 1.0] = 1
    code[flat == -1.0] = 2

    # Pack 4 2-bit codes into 1 byte
    # byte = (c0) | (c1 << 2) | (c2 << 4) | (c3 << 6)
    b0 = code[:, 0]
    b1 = code[:, 1] << 2
    b2 = code[:, 2] << 4
    b3 = code[:, 3] << 6
    packed_flat = b0 | b1 | b2 | b3

    new_shape = list(orig_shape)
    new_shape[-1] = new_shape[-1] // 4
    packed = packed_flat.reshape(new_shape)
    return packed, orig_shape

def unpack_ternary_tensor(packed: torch.Tensor, orig_shape: tuple, dtype=torch.bfloat16, device="cuda") -> torch.Tensor:
    """
    Fast GPU vectorized bit-shift unpacking from 2-bit uint8 back to ternary {-1, 0, +1}.
    """
    p_flat = packed.to(device).reshape(-1)

    # Extract 2-bit fields
    c0 = p_flat & 0x03
    c1 = (p_flat >> 2) & 0x03
    c2 = (p_flat >> 4) & 0x03
    c3 = (p_flat >> 6) & 0x03

    codes = torch.stack([c0, c1, c2, c3], dim=1).reshape(-1)

    # Decode: {0 -> 0.0, 1 -> 1.0, 2 -> -1.0}
    # Vectorized arithmetic: (code == 1) - (code == 2)
    unpacked_flat = (codes == 1).to(dtype) - (codes == 2).to(dtype)
    return unpacked_flat.reshape(orig_shape)

def benchmark_packing_pipeline():
    print("=" * 80)
    print("PHASE 6: PACKED TERNARY WEIGHT REPRESENTATION BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate 606M weights (or a large linear layer: 1024 x 2048)
    M, K = 1024, 2048
    num_weights = M * K
    raw_ternary = torch.randint(-1, 2, (M, K), dtype=torch.float32)

    # 1. Size Comparison
    fp32_bytes = raw_ternary.numel() * 4
    bf16_bytes = raw_ternary.numel() * 2
    packed, orig_shape = pack_ternary_tensor(raw_ternary)
    packed_bytes = packed.numel()

    print(f"Matrix Dimension: {M} x {K} ({num_weights:,} parameters)")
    print(f"  FP32 Footprint:   {fp32_bytes / 1024:.2f} KB ({fp32_bytes / (1024**2):.2f} MB)")
    print(f"  BF16 Footprint:   {bf16_bytes / 1024:.2f} KB ({bf16_bytes / (1024**2):.2f} MB)")
    print(f"  Packed Ternary:   {packed_bytes / 1024:.2f} KB ({packed_bytes / (1024**2):.2f} MB)")
    print(f"  Compression Ratio vs FP32: {fp32_bytes / packed_bytes:.1f}x")
    print(f"  Compression Ratio vs BF16: {bf16_bytes / packed_bytes:.1f}x")

    # 2. Correctness Gate
    unpacked = unpack_ternary_tensor(packed, orig_shape, dtype=torch.float32, device=device)
    max_err = (raw_ternary.to(device) - unpacked).abs().max().item()
    print(f"\nCorrectness Check:")
    print(f"  Max Absolute Reconstruction Error: {max_err:.8f}")
    assert max_err == 0.0, f"Unpacking error detected: {max_err}"
    print("  Reconstruction: 100% BIT-EXACT MATCH.")

    # 3. GPU Unpacking Throughput
    for _ in range(20):
        _ = unpack_ternary_tensor(packed, orig_shape, device=device)
    torch.cuda.synchronize()

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = unpack_ternary_tensor(packed, orig_shape, device=device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_unpack_ms = (sum(times) / len(times)) * 1000.0
    unpack_gb_s = (packed_bytes / 1e9) / (sum(times) / len(times))
    print(f"\nGPU Unpack Latency:    {avg_unpack_ms:.3f} ms ({avg_unpack_ms * 1000:.1f} us)")
    print(f"GPU Unpack Throughput: {unpack_gb_s:.2f} GB/s")

    # 4. Multi-Billion Scale Projection
    print("\n--- 3.2 BILLION PARAMETER MODEL SCALE PROJECTION ---")
    p32_fp32_gb = (3.2e9 * 4) / (1024**3)
    p32_bf16_gb = (3.2e9 * 2) / (1024**3)
    p32_packed_gb = (3.2e9 * 0.25) / (1024**3)
    pcie_bw_gb_s = 27.56  # Measured earlier on this hardware
    transfer_ms = (p32_packed_gb / pcie_bw_gb_s) * 1000.0

    print(f"Total Parameters: 3,200,000,000 (3.2B)")
    print(f"  FP32 Master Weight Storage:    {p32_fp32_gb:.2f} GB (Exceeds RTX 5070 & RTX 5050 VRAM)")
    print(f"  BF16 Weight Storage:           {p32_bf16_gb:.2f} GB (Exceeds RTX 5050 VRAM)")
    print(f"  Packed 2-bit Ternary Storage:  {p32_packed_gb:.2f} GB (Fits easily in 8GB RTX 5050 VRAM!)")
    print(f"  PCIe Gen 4 Transfer Time:      {transfer_ms:.2f} ms ({transfer_ms/1000:.3f} seconds)")
    print("=" * 80)

if __name__ == '__main__':
    benchmark_packing_pipeline()
