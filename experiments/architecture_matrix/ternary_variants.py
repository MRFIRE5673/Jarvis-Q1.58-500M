# experiments/architecture_matrix/ternary_variants.py
"""
Architectural Evolution — Modern Ternary & BitLinear Variants (Track B)
========================================================================
1. PerChannelTernaryLinear (Track B3):
   - Row-wise (per output neuron) AbsMean scaling: alpha_i = mean_j(|W_ij|)
   - Normalized Straight-Through Estimator mask: (|W / alpha| <= 1.0)
   - Preserves 2-bit packing compatibility while expanding representation capacity.

2. BitLinear (Track B2 & B10):
   - Sublayer RMSNorm preceding projection
   - AbsMax activation quantization to INT8 range [-128, 127] with STE
   - Per-channel ternary weight quantization {-1, 0, +1}
   - Fused scaled linear execution.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import RMSNorm


class PerChannelTernaryQuantizeSTE(torch.autograd.Function):
    """
    Per-channel (row-wise) ternary weight quantization with Normalized STE.
    Forward:
        alpha_i = mean_j(|W_ij|)  (out_features, 1)
        W_norm = W / alpha
        W_q = round(clamp(W_norm, -1.0, 1.0)) * alpha
    Backward:
        grad_w = grad_output * 1{|W_norm| <= 1.0}
    """
    @staticmethod
    def forward(ctx, w):
        alpha = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)  # (out_features, 1)
        w_norm = w / alpha
        w_q = torch.round(torch.clamp(w_norm, -1.0, 1.0))
        ctx.save_for_backward(w_norm)
        return w_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        w_norm, = ctx.saved_tensors
        mask = (w_norm.abs() <= 1.0).float()
        return grad_output * mask


class PerChannelTernaryLinear(nn.Module):
    """Drop-in ternary linear replacement with per-output-channel scaling."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w_q = PerChannelTernaryQuantizeSTE.apply(self.weight)
        return F.linear(x, w_q, self.bias)


class ActivationQuantizeSTE(torch.autograd.Function):
    """
    BitNet 8-bit activation quantization with Straight-Through Estimator.
    Scales activations to [-127, 127] based on dynamic AbsMax.
    """
    @staticmethod
    def forward(ctx, x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = torch.clamp(torch.round(x * scale), -128, 127) / scale
        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitLinear(nn.Module):
    """
    Modern BitNet-style BitLinear layer.
    Combines sublayer RMSNorm + 8-bit activation quantization + per-channel ternary weights.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.norm = RMSNorm(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        x_norm = self.norm(x)
        x_quant = ActivationQuantizeSTE.apply(x_norm)
        w_q = PerChannelTernaryQuantizeSTE.apply(self.weight)
        return F.linear(x_quant, w_q, self.bias)
