import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl
    _TRITON_STE_AVAILABLE = True
except ImportError:
    _TRITON_STE_AVAILABLE = False

if _TRITON_STE_AVAILABLE:
    @triton.jit
    def _ternary_quantize_fwd_kernel(
        W_ptr, Wq_ptr, Alpha_ptr,
        N_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements
        
        alpha = tl.load(Alpha_ptr)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w_norm = w / alpha
        w_clamped = tl.maximum(tl.minimum(w_norm, 1.0), -1.0)
        w_round = tl.extra.cuda.libdevice.nearbyint(w_clamped)
        w_q = w_round * alpha
        
        tl.store(Wq_ptr + offsets, w_q.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _ternary_quantize_bwd_kernel(
        GradOut_ptr, W_ptr, GradW_ptr,
        N_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements
        
        go = tl.load(GradOut_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_valid = tl.abs(w) <= 1.0
        gw = tl.where(is_valid, go, 0.0)
        
        tl.store(GradW_ptr + offsets, gw.to(tl.bfloat16), mask=mask)

    class FusedTernaryQuantizeSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w):
            alpha = w.abs().mean().clamp(min=1e-8)
            wq = torch.empty_like(w)
            N = w.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            _ternary_quantize_fwd_kernel[grid](
                w, wq, alpha,
                N, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4
            )
            ctx.save_for_backward(w)
            return wq

        @staticmethod
        def backward(ctx, grad_output):
            w, = ctx.saved_tensors
            grad_w = torch.empty_like(w)
            N = w.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            _ternary_quantize_bwd_kernel[grid](
                grad_output, w, grad_w,
                N, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4
            )
            return grad_w


if _TRITON_STE_AVAILABLE:
    TernaryQuantizeSTE = FusedTernaryQuantizeSTE
else:
    class TernaryQuantizeSTE(torch.autograd.Function):
        """
        Ternary weight quantization with AbsMean scaling.

        Forward (Eq. 3):
            α   = mean(|W_FP32|)          — AbsMean scale (BitNet convention)
            W̃  = round(clamp(W / α, -1, 1)) · α   — maps to {-1, 0, +1} in original scale

        Backward (Eq. 5 — Straight-Through Estimator):
            ∂L/∂W_FP32 ≈ ∂L/∂W̃ · 1{|W / α| ≤ 1}
        """

        @staticmethod
        def forward(ctx, w):
            alpha = w.abs().mean().clamp(min=1e-8)      # AbsMean scale (BitNet convention)
            w_norm = w / alpha                           # normalise to ≈ unit scale
            w_q = torch.round(torch.clamp(w_norm, -1.0, 1.0))
            ctx.save_for_backward(w)                     # save ORIGINAL W_FP32 for Eq. 5 mask
            return w_q * alpha                           # rescale back to original magnitude

        @staticmethod
        def backward(ctx, grad_output):
            w, = ctx.saved_tensors
            mask = (w.abs() <= 1.0).float()
            return grad_output * mask


class TernaryLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary weights via STE."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w_q = TernaryQuantizeSTE.apply(self.weight)
        return F.linear(x, w_q, self.bias)