# validate_liquid_state_fusion.py
import torch
import sys
from liquid_state_fusion import (
    LiquidStateFusion,
    _CUDA_AVAILABLE,
    reset_diagnostics,
    get_diagnostics,
)

def sequential_baseline(alpha, M, h0=None):
    B, T, D = alpha.shape
    H = torch.zeros_like(M)
    h_prev = h0 if h0 is not None else torch.zeros((B, D), dtype=alpha.dtype, device=alpha.device)
    for t in range(T):
        h_t = alpha[:, t, :] * h_prev + (1.0 - alpha[:, t, :]) * M[:, t, :]
        H[:, t, :] = h_t
        h_prev = h_t
    return H, h_prev

def run_correctness_tests(dtype, dtype_name, device, require_custom_cuda=False):
    print(f"\n{'='*80}\n  Testing {dtype_name} on {str(device).upper()}\n{'='*80}")
    test_lengths = [1, 15, 65, 127, 129, 256, 512]
    B, D = 2, 8
    tol = {torch.float64: 1e-12, torch.float32: 1e-5, torch.bfloat16: 8e-2}.get(dtype, 1e-3)
    torch.manual_seed(42)

    for T in test_lengths:
        reset_diagnostics()

        alpha_seq = torch.rand(B, T, D, dtype=dtype, device=device, requires_grad=True)
        with torch.no_grad():
            alpha_seq.copy_(0.01 + 0.98 * alpha_seq)
        M_seq = torch.rand(B, T, D, dtype=dtype, device=device, requires_grad=True)
        h0_seq = torch.rand(B, D, dtype=dtype, device=device, requires_grad=True)

        alpha_par = alpha_seq.detach().clone().requires_grad_(True)
        M_par = M_seq.detach().clone().requires_grad_(True)
        h0_par = h0_seq.detach().clone().requires_grad_(True)

        H_seq, _ = sequential_baseline(alpha_seq, M_seq, h0_seq)
        fusion = LiquidStateFusion(persistent=False)
        H_par = fusion(alpha_par, M_par, h0_par)

        diff_forward = torch.max(torch.abs(H_seq - H_par)).item()
        grad_out = torch.randn_like(H_seq)
        H_seq.backward(grad_out)
        H_par.backward(grad_out)
        diff_ga = torch.max(torch.abs(alpha_seq.grad - alpha_par.grad)).item()
        diff_gm = torch.max(torch.abs(M_seq.grad - M_par.grad)).item()
        diff_gh0 = torch.max(torch.abs(h0_seq.grad - h0_par.grad)).item()

        diag = get_diagnostics()
        passed_tol = max(diff_forward, diff_ga, diff_gm, diff_gh0) < tol

        if require_custom_cuda and not diag["custom_kernel_executed"]:
            status = "FAILED (Custom CUDA kernel NOT executed)"
        elif require_custom_cuda and diag["fallback_used"]:
            status = "FAILED (Fallback was used instead of custom CUDA)"
        elif passed_tol:
            status = "PASSED"
        else:
            status = f"FAILED (tol {tol:.1e})"

        print(f"T={T:<4} fwd={diff_forward:.2e} g_a={diff_ga:.2e} g_M={diff_gm:.2e} g_h0={diff_gh0:.2e} | {status}")
        if "FAILED" in status:
            sys.exit(1)

if __name__ == "__main__":
    print(f"CUDA Backend Compiled/Loaded: {_CUDA_AVAILABLE}")
    diag = get_diagnostics()
    print(f"Custom CUDA extension imported: {diag['extension_imported']}")
    
    device = torch.device("cuda" if (torch.cuda.is_available() and _CUDA_AVAILABLE) else "cpu")
    print(f"Selected test device: {device}")
    
    run_correctness_tests(torch.float32, "Float32", device, require_custom_cuda=_CUDA_AVAILABLE)
    run_correctness_tests(torch.bfloat16, "BFloat16", device, require_custom_cuda=_CUDA_AVAILABLE)
    
    diag_final = get_diagnostics()
    print("\n--- Diagnostic Summary ---")
    print(f"Custom CUDA extension imported: {diag_final['extension_imported']}")
    print(f"Custom CUDA kernel executed: {diag_final['custom_kernel_executed']}")
    print(f"Fallback used: {diag_final['fallback_used']}")
    print("\nDone. Read the PASSED/FAILED lines above — do not trust anything not shown here.")
