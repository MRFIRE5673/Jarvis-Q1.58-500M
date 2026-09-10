# experiments/architecture_matrix/jarvis_v2_combined.py
"""
JARVIS-600M-V2 COMBINED ARCHITECTURE (Track Z Winner)
=====================================================
Unified Next-Generation Architecture synthesizing empirical winners:
1. Track A Winner: Multi-Timescale Attention with Local Sliding Buffer (W=16) and Gated Memory Writes
2. Track E Winner: Squared-ReLU (ReLU(x)^2) Feed-Forward Experts with exact budget parity
3. Track F & G Winner: Aux-Free Dynamic Online Expert Bias MoE (L_bal = 0)
4. Track B Winner: Normalized Straight-Through Estimator (|w/alpha| <= 1.0)

Parameter count: ~606.8M parameters (matches baseline 606.4M within 0.06%).
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import RMSNorm, RotaryEmbedding, LiquidStateFusion, ReflectivePenalty
from utils.ternary_ops import TernaryLinear


class JarvisV2Attention(nn.Module):
    """
    Jarvis-V2 Unified Attention:
    - Multi-Timescale Associative Decay (4 timescale bands: fast, medium, slow, persistent)
    - Exact Local Sliding Window Attention Buffer (W=16) for syntax preservation
    - Dynamic Input-Dependent Write Gate (w_t = sigmoid(W_write * x_t))
    """
    CHUNK_SIZE = 64
    WINDOW_SIZE = 16

    def __init__(self, d_model=1024, n_heads=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        # Write strength projection: token hidden state -> per-head write strength
        self.write_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.write_gate_proj.bias, 1.0)

        # Multi-timescale decay: 4 bands [0.85, 0.95, 0.99, 0.999]
        init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        # Learnable blend factor between recurrent global context and local sliding window
        self.blend_gate = nn.Parameter(torch.zeros(n_heads))  # init sigmoid(0) = 0.5

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        W = self.WINDOW_SIZE

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Gated write strength: scale values before memory accumulation
        w_t = torch.sigmoid(self.write_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        v_gated = v * w_t

        q_feat = (F.elu(q) + 1.0) / math.sqrt(D)
        k_feat = F.elu(k) + 1.0
        q_feat, k_feat = self.rotary(q_feat, k_feat, start_pos=start_pos)

        # 1. Exact Local Sliding-Window Attention (Width W=16)
        t_idx = torch.arange(T, device=x.device)
        diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)
        local_mask = (diff >= 0) & (diff < W)

        local_attn = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D)
        local_attn = local_attn.masked_fill(~local_mask.view(1, 1, T, T), -1e4)
        local_probs = F.softmax(local_attn, dim=-1)
        local_out = torch.einsum('bhij,bhjd->bhid', local_probs, v).transpose(1, 2).contiguous().view(B, T, C)

        # 2. Global Multi-Timescale Recurrent Associative Attention
        log_g = torch.log(torch.sigmoid(self.gamma_raw)).view(1, H, 1, 1)
        decay_diff = (diff.clamp(min=0)).view(1, 1, T, T)
        causal_mat = (diff >= 0).float().view(1, 1, T, T)
        decay_mat = torch.exp(log_g * decay_diff) * causal_mat

        recurrent_scores = torch.einsum('bhid,bhjd->bhij', q_feat, k_feat) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', recurrent_scores, v_gated).transpose(1, 2).contiguous().view(B, T, C)

        # 3. Dynamic Blend: local exact syntax + global associative memory
        gate = torch.sigmoid(self.blend_gate).view(1, 1, H, 1)
        l_out = local_out.view(B, T, H, D)
        r_out = recurrent_out.view(B, T, H, D)
        blended = (gate * l_out + (1.0 - gate) * r_out).contiguous().view(B, T, C)

        return self.out_proj(blended)


class JarvisV2MoE(nn.Module):
    """
    Jarvis-V2 Unified Feed-Forward / MoE Layer:
    - Squared-ReLU (ReLU(x)^2) activation on ternary experts
    - Dynamic Auxiliary-Loss-Free Expert Bias (L_bal = 0)
    - Online load-balancing decoupled from backprop gradient checkpointing
    """
    def __init__(self, d_model=1024, num_experts=4, top_k=2, hidden_mult=2, noise_std=0.1, bias_lr=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.bias_lr = bias_lr
        hidden = d_model * hidden_mult

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.last_fractions = None

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        raw_logits = self.router(x_flat)
        if self.training:
            raw_logits = raw_logits + torch.randn_like(raw_logits) * self.noise_std

        # Augmented with dynamic bias
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
                # Squared-ReLU activation
                h = F.relu(self.w1[e](xe))
                ye = self.w2[e](h * h)
                flat_out[mask] = flat_gate[mask].unsqueeze(-1) * ye

        out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)

        if self.training:
            with torch.no_grad():
                top1_idx = topk_idx[:, 0]
                counts = torch.bincount(top1_idx, minlength=self.num_experts).float()
                self.last_fractions = (counts / N).detach()

        act_mean = out.mean()
        act_var = out.var()
        zero_aux_loss = torch.tensor(0.0, device=x.device)

        return out, zero_aux_loss, act_mean, act_var

    def update_bias(self):
        """Update dynamic bias post-optimizer step."""
        if hasattr(self, 'last_fractions') and self.last_fractions is not None:
            target = 1.0 / self.num_experts
            self.expert_bias.data.sub_(self.bias_lr * (self.last_fractions - target))


class JarvisV2Block(nn.Module):
    """Transformer block with Jarvis-V2 Attention, Squared-ReLU Aux-Free MoE, and Liquid State Fusion."""
    def __init__(self, d_model=1024, n_heads=16, max_seq_len=2048):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = JarvisV2Attention(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len)
        self.moe_norm = RMSNorm(d_model)
        self.moe = JarvisV2MoE(d_model=d_model, num_experts=4, top_k=2)
        self.liquid = LiquidStateFusion(d_model)
        self.reflect = ReflectivePenalty()

    def forward(self, x, h_prev=None, start_pos=0):
        # 1. Attention with residual
        x = x + self.attn(self.attn_norm(x), start_pos=start_pos)
        # 2. MoE
        moe_out, aux_loss, mean_a, var_a = self.moe(self.moe_norm(x))
        # 3. Liquid State Fusion (LIF membrane dynamics)
        h_out, h_last = self.liquid(moe_out, var_a, h_prev)
        x = x + h_out
        # 4. Reflective Penalty
        l_reflect = self.reflect(mean_a, var_a)
        return x, h_last, aux_loss, l_reflect


class JarvisV2(nn.Module):
    """Complete ~606M Jarvis-V2 Model with full paper-faithful neuromorphic lineage."""
    def __init__(
        self,
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        max_seq_len=512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            JarvisV2Block(d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None, start_pos=0, persist_state=False, **kwargs):
        B, T = idx.shape
        x = self.tok_emb(idx)

        total_aux = torch.tensor(0.0, device=idx.device)
        total_reflect = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            if self.training:
                def make_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x, h_last, aux, l_ref = torch.utils.checkpoint.checkpoint(make_custom_forward(block), x, None, start_pos, use_reentrant=False)
            else:
                x, h_last, aux, l_ref = block(x, h_prev=None, start_pos=start_pos)
            total_aux = total_aux + aux
            total_reflect = total_reflect + l_ref

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            loss = ce_loss + total_aux + total_reflect

        return logits, loss

    def reset_state(self):
        pass

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total


def verify_model():
    model = JarvisV2(max_seq_len=512)
    total_params = model.count_parameters()
    print(f"Jarvis-V2 Total Parameters: {total_params:,}")
    return model


if __name__ == "__main__":
    verify_model()
