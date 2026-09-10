# experiments/architecture_matrix/moe_variants.py
"""
Architectural Evolution — Modern MoE Routing Variants (Tracks F & G)
====================================================================
1. DeepSeekSharedMoELayer (Track F):
   - 1 Shared Expert (always active for all tokens)
   - 3 Routed Experts with Top-1 routing
   - Total active experts per token: exactly 1 + 1 = 2 (identical to baseline Top-2)
   - Total parameter count: exactly 4 experts per layer (exact parameter parity)
   - Prevents routed experts from wasting capacity on common syntax/boilerplate.

2. AuxFreeBiasMoELayer (Track G):
   - Eliminates load balance loss L_bal from the loss function entirely
   - Dynamic additive expert bias: s_i = router(x)_i + bias_i
   - Online load balancing update: bias_i -= eta * (load_i - target_load)
   - Language modeling cross-entropy receives 100% of gradient capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ternary_ops import TernaryLinear


class DeepSeekSharedMoELayer(nn.Module):
    """
    DeepSeekMoE-style Shared + Routed Expert architecture.
    1 Shared Expert + 3 Routed Experts (Top-1 routed).
    Active compute: exactly 2 experts per token.
    Total parameters: exactly 4 experts per layer (matches baseline).
    """
    def __init__(self, d_model=1024, hidden_mult=2, noise_std=0.1):
        super().__init__()
        self.d_model = d_model
        hidden = d_model * hidden_mult

        # 1 Always-active Shared Expert
        self.shared_w1 = TernaryLinear(d_model, hidden)
        self.shared_w2 = TernaryLinear(hidden, d_model)

        # 3 Routed Experts
        self.num_routed = 3
        self.routed_w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(self.num_routed)])
        self.routed_w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(self.num_routed)])

        # Router over the 3 routed experts
        self.router = nn.Linear(d_model, self.num_routed, bias=False)
        self.noise_std = noise_std

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        # 1. Shared Expert pathway (all tokens)
        shared_out = self.shared_w2(F.gelu(self.shared_w1(x_flat)))

        # 2. Routed pathway (Top-1 of 3 routed experts)
        logits = self.router(x_flat)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        gate, top1_idx = probs.max(dim=-1)  # (N,)

        routed_out = torch.zeros_like(x_flat)
        for e in range(self.num_routed):
            mask = (top1_idx == e)
            if mask.any():
                xe = x_flat[mask]
                ye = self.routed_w2[e](F.gelu(self.routed_w1[e](xe)))
                routed_out[mask] = gate[mask].unsqueeze(-1) * ye

        total_out = (shared_out + routed_out).view(B, T, C)

        # Mild load balance loss over the 3 routed experts
        f = F.one_hot(top1_idx, num_classes=self.num_routed).float().mean(dim=0)
        P = probs.mean(dim=0)
        l_balance = 0.01 * self.num_routed * (f * P).sum()

        act_mean = total_out.mean()
        act_var = total_out.var()

        return total_out, l_balance, act_mean, act_var


class AuxFreeBiasMoELayer(nn.Module):
    """
    Auxiliary-Loss-Free MoE with dynamic additive expert bias.
    Completely eliminates load balance loss L_bal.
    Online bias update separates routing balance from backpropagated LM gradients.
    """
    def __init__(self, d_model=1024, num_experts=4, top_k=2, hidden_mult=2, bias_lr=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_lr = bias_lr
        hidden = d_model * hidden_mult

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

        # Dynamic router bias buffer (not trained by autograd)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        # Raw scores from language model features
        raw_logits = self.router(x_flat)  # (N, E)

        # Routing scores augmented with dynamic bias
        biased_logits = raw_logits + self.expert_bias.unsqueeze(0)
        probs = F.softmax(biased_logits, dim=-1)

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
                ye = self.w2[e](F.gelu(self.w1[e](xe)))
                flat_out[mask] = flat_gate[mask].unsqueeze(-1) * ye

        out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)

        # Store routing statistics without in-place buffer modification during forward (checkpoint-safe)
        if self.training:
            with torch.no_grad():
                top1_idx = topk_idx[:, 0]
                counts = torch.bincount(top1_idx, minlength=self.num_experts).float()
                self.last_fractions = (counts / N).detach()

        act_mean = out.mean()
        act_var = out.var()
        # Zero auxiliary loss!
        zero_aux_loss = torch.tensor(0.0, device=x.device)

        return out, zero_aux_loss, act_mean, act_var

    def update_bias(self):
        """Update dynamic bias post-optimizer step to decouple from gradient checkpointing."""
        if hasattr(self, 'last_fractions') and self.last_fractions is not None:
            target = 1.0 / self.num_experts
            self.expert_bias.data.sub_(self.bias_lr * (self.last_fractions - target))

