import os
import sys
import glob
import json
import torch

# Ensure CUDA bin is added to DLL search path on Windows
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

# Add isolated LSF vectorized path
OPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "lsf_vectorized"))
ROOT_LSF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "liquid_fusion_cuda"))

sys.path.insert(0, OPT_DIR)
import liquid_state_fusion_cuda_opt as lsf_opt
sys.path.pop(0)

sys.path.insert(0, ROOT_LSF_DIR)
import liquid_state_fusion_cuda as lsf_orig
sys.path.pop(0)

print(f"[OK] Control A (Original) Loaded: {lsf_orig}")
print(f"[OK] Candidate B (Vectorized Backward) Loaded: {lsf_opt}")

def benchmark_kernel(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    timings = []
    for _ in range(iters):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event)) # ms

    timings.sort()
    mean_ms = sum(timings) / len(timings)
    median_ms = timings[len(timings) // 2]
    min_ms = timings[0]
    p95_ms = timings[int(len(timings) * 0.95)]
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "p95_ms": p95_ms,
    }

def run_correctness(dtype=torch.bfloat16, B=2, T=512, D=1024):
    print(f"\n{'='*70}\nNUMERICAL CORRECTNESS VERIFICATION (dtype={dtype}, B={B}, T={T}, D={D})\n{'='*70}")
    torch.manual_seed(42)
    device = "cuda"

    alpha = 0.01 + 0.98 * torch.rand(B, T, D, dtype=dtype, device=device)
    M = torch.randn(B, T, D, dtype=dtype, device=device)
    h0 = torch.randn(B, D, dtype=dtype, device=device)
    grad_H = torch.randn(B, T, D, dtype=dtype, device=device)

    # Forward
    H_orig = lsf_orig.forward(alpha, M, h0, 64)
    H_opt = lsf_opt.forward(alpha, M, h0, 64)

    diff_fwd = (H_orig - H_opt).float().abs()
    fwd_max_abs = diff_fwd.max().item()
    fwd_mean_abs = diff_fwd.mean().item()
    fwd_max_rel = (diff_fwd / (H_orig.float().abs() + 1e-7)).max().item()

    # Backward
    g_a_orig, g_M_orig, g_h0_orig = lsf_orig.backward(alpha, M, h0, H_orig, grad_H, 64)
    g_a_opt, g_M_opt, g_h0_opt = lsf_opt.backward(alpha, M, h0, H_opt, grad_H, 64)

    diff_ga = (g_a_orig - g_a_opt).float().abs()
    diff_gm = (g_M_orig - g_M_opt).float().abs()
    diff_gh0 = (g_h0_orig - g_h0_opt).float().abs()

    ga_max = diff_ga.max().item()
    gm_max = diff_gm.max().item()
    gh0_max = diff_gh0.max().item()

    ga_rel = (diff_ga / (g_a_orig.float().abs() + 1e-7)).max().item()
    gm_rel = (diff_gm / (g_M_orig.float().abs() + 1e-7)).max().item()
    gh0_rel = (diff_gh0 / (g_h0_orig.float().abs() + 1e-7)).max().item()

    print(f"Forward Output : MaxAbsErr = {fwd_max_abs:.6e}, MeanAbsErr = {fwd_mean_abs:.6e}, MaxRelErr = {fwd_max_rel:.6e}")
    print(f"grad_alpha     : MaxAbsErr = {ga_max:.6e}, MaxRelErr = {ga_rel:.6e}")
    print(f"grad_M         : MaxAbsErr = {gm_max:.6e}, MaxRelErr = {gm_rel:.6e}")
    print(f"grad_h0        : MaxAbsErr = {gh0_max:.6e}, MaxRelErr = {gh0_rel:.6e}")

    tol = 1e-4 if dtype == torch.float32 else 5e-2
    passed = max(fwd_max_abs, ga_max, gm_max, gh0_max) < tol
    print(f"Correctness Result: {'PASSED [OK]' if passed else 'FAILED [REGRESSION]'}")
    return {
        "dtype": str(dtype),
        "fwd_max_abs": fwd_max_abs,
        "fwd_mean_abs": fwd_mean_abs,
        "ga_max_abs": ga_max,
        "gm_max_abs": gm_max,
        "gh0_max_abs": gh0_max,
        "passed": passed
    }

def run_benchmarks(B=2, T=512, D=1024, dtype=torch.bfloat16):
    print(f"\n{'='*70}\nLSF KERNEL BENCHMARK (B={B}, T={T}, D={D}, dtype={dtype})\n{'='*70}")
    device = "cuda"
    alpha = 0.01 + 0.98 * torch.rand(B, T, D, dtype=dtype, device=device)
    M = torch.randn(B, T, D, dtype=dtype, device=device)
    h0 = torch.randn(B, D, dtype=dtype, device=device)
    grad_H = torch.randn(B, T, D, dtype=dtype, device=device)

    H = lsf_orig.forward(alpha, M, h0, 64)

    block_sizes = [64, 128, 256, 512]
    results = {"forward": {}, "backward_orig": {}, "backward_opt": {}}

    print("\n--- FORWARD PASS (V=2 Vectorized) across Block Sizes ---")
    for bs in block_sizes:
        res = benchmark_kernel(lambda: lsf_opt.forward(alpha, M, h0, bs))
        results["forward"][bs] = res
        print(f"BlockSize={bs:3d}: Mean={res['mean_ms']:.4f} ms | Median={res['median_ms']:.4f} ms | Min={res['min_ms']:.4f} ms")

    print("\n--- BACKWARD PASS: Control A (Coalesced) vs Candidate B (V=2 Vectorized) ---")
    for bs in block_sizes:
        res_orig = benchmark_kernel(lambda: lsf_orig.backward(alpha, M, h0, H, grad_H, bs))
        res_opt  = benchmark_kernel(lambda: lsf_opt.backward(alpha, M, h0, H, grad_H, bs))
        results["backward_orig"][bs] = res_orig
        results["backward_opt"][bs] = res_opt
        speedup = (res_orig['median_ms'] - res_opt['median_ms']) / res_orig['median_ms'] * 100
        print(f"BlockSize={bs:3d} | Orig={res_orig['median_ms']:.4f} ms | Opt={res_opt['median_ms']:.4f} ms | Speedup={speedup:+.1f}%")

    return results

if __name__ == "__main__":
    corr_bf16 = run_correctness(torch.bfloat16)
    corr_fp32 = run_correctness(torch.float32)
    bench_results = run_benchmarks()

    out_file = r"G:\Jarvis_Training\run_50m_baseline\profiling\lsf_benchmark_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"correctness_bf16": corr_bf16, "correctness_fp32": corr_fp32, "benchmarks": bench_results}, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")
