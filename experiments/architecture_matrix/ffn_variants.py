# experiments/architecture_matrix/ffn_variants.py
"""
Architectural Evolution — Modern FFN Activations (Track E)
==========================================================
All variants are strictly calibrated to the EXACT SAME parameter budget:
~4,194,304 parameters per expert (24 layers x 4 experts).

1. Baseline GELU Expert:
   - y = W2(GELU(W1(x))), d_hidden = 2048 (2 x d_model)
   - Parameters: 2 x (1024 x 2048) = 4,194,304

2. Squared-ReLU Expert (Track E5 - Modern BitNet standard):
   - y = W2(ReLU(W1(x))^2), d_hidden = 2048 (2 x d_model)
   - Emphasizes sparse high-confidence activations in low-precision networks.
   - Parameters: 2 x (1024 x 2048) = 4,194,304 (Exact parity)

3. SwiGLU Expert (Track E2):
   - y = W2(SiLU(W_gate(x)) * W1(x))
   - d_hidden = 1368 (~4/3 x d_model) to match parameter budget
   - Parameters: 3 x (1024 x 1368) = 4,202,496 (0.19% parity difference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ternary_ops import TernaryLinear


class BaselineGELUExpert(nn.Module):
    def __init__(self, d_model=1024, hidden_dim=2048):
        super().__init__()
        self.w1 = TernaryLinear(d_model, hidden_dim)
        self.w2 = TernaryLinear(hidden_dim, d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class SquaredReLUExpert(nn.Module):
    """Modern BitNet-style squared-ReLU activation."""
    def __init__(self, d_model=1024, hidden_dim=2048):
        super().__init__()
        self.w1 = TernaryLinear(d_model, hidden_dim)
        self.w2 = TernaryLinear(hidden_dim, d_model)

    def forward(self, x):
        # ReLU(x)^2
        h = F.relu(self.w1(x))
        return self.w2(h * h)


class SwiGLUExpert(nn.Module):
    """SwiGLU feed-forward expert with budget-calibrated hidden dimension."""
    def __init__(self, d_model=1024, hidden_dim=1368):
        super().__init__()
        self.w_gate = TernaryLinear(d_model, hidden_dim)
        self.w1 = TernaryLinear(d_model, hidden_dim)
        self.w2 = TernaryLinear(hidden_dim, d_model)

    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


class MoELayerWithActivation(nn.Module):
    """
    MoE Layer with customizable FFN activation function:
    - 'squared_relu' (Modern BitNet standard: ReLU(x)^2)
    - 'swiglu' (SwiGLU with d_hidden=1368 for exact budget parity)
    - 'gelu' (Baseline)
    """
    def __init__(self, d_model=1024, num_experts=4, top_k=2, activation="squared_relu", noise_std=0.1, balance_alpha=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.balance_alpha = balance_alpha
        self.activation = activation
        self.router = nn.Linear(d_model, num_experts, bias=False)

        if activation == "swiglu":
            hidden = 1368
            self.w_gate = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
            self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
            self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])
        else:
            hidden = d_model * 2
            self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
            self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        logits = self.router(x_flat)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_gates = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        flat_idx = topk_idx.reshape(-1)
        flat_gate = topk_gates.reshape(-1)
        flat_x = x_flat.repeat_interleave(self.top_k, dim=0)
        flat_out = torch.zeros_like(flat_x)

        for e in range(self.num_experts):
            mask = (flat_idx == e)
            if mask.any():
                xe = flat_x[mask]
                if self.activation == "squared_relu":
                    h = F.relu(self.w1[e](xe))
                    ye = self.w2[e](h * h)
                elif self.activation == "swiglu":
                    ye = self.w2[e](F.silu(self.w_gate[e](xe)) * self.w1[e](xe))
                else:
                    ye = self.w2[e](F.gelu(self.w1[e](xe)))
                flat_out[mask] = flat_gate[mask].unsqueeze(-1) * ye

        out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)

        top1_idx = topk_idx[:, 0]
        f = F.one_hot(top1_idx, num_classes=self.num_experts).float().mean(dim=0)
        P = probs.mean(dim=0)
        l_balance = self.balance_alpha * self.num_experts * (f * P).sum()

        act_mean = out.mean()
        act_var = out.var()
        return out, l_balance, act_mean, act_var

