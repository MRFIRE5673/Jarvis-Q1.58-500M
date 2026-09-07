import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Ternary weight quantization with AbsMean scaling.

    Forward (Eq. 3):
        α   = mean(|W_FP32|)          — AbsMean scale (BitNet convention)
        W̃  = round(clamp(W / α, -1, 1)) · α   — maps to {-1, 0, +1} in original scale

    Backward (Eq. 5 — Straight-Through Estimator):
        ∂L/∂W_FP32 ≈ ∂L/∂W̃ · 1{|W / α| ≤ 1}

    Note: the paper writes Eq. 3 as round(clamp(W_FP32, -1, 1)) without an explicit
    scale factor.  AbsMean scaling is the standard implementation practice (BitNet b1.58)
    — without it, randomly-initialised weights (σ ≈ 0.02) all collapse to 0 and the
    network cannot learn.  The STE mask is applied on the *normalised* weight W/α so it
    matches the paper's boundary exactly at ±1 after rescaling.
    """

    @staticmethod
    def forward(ctx, w):
        alpha = w.abs().mean().clamp(min=1e-8)      # AbsMean scale (BitNet convention)
        w_norm = w / alpha                           # normalise to ≈ unit scale
        # Eq. 3: clamp FIRST to [-1, 1], THEN round → {-1, 0, +1}
        w_q = torch.round(torch.clamp(w_norm, -1.0, 1.0))
        ctx.save_for_backward(w)                     # save ORIGINAL W_FP32 for Eq. 5 mask
        return w_q * alpha                           # rescale back to original magnitude

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        # Eq. 5 (paper exact): ∂L/∂W_FP32 ≈ ∂L/∂W̃ · 1{|W_FP32| ≤ 1}
        # Mask is on the ORIGINAL unscaled weight, not the normalised one.
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