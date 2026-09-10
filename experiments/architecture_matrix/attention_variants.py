# experiments/architecture_matrix/attention_variants.py
"""
Architectural Evolution — Attention & Memory Variants (Tracks A, V, U, W)
==========================================================================
1. MultiTimescaleAttention (Track A3 & V):
   - 16 heads split across 4 logarithmic decay bands:
     * Band 0 (Heads 0-3):   gamma ≈ 0.85  (half-life ≈ 4.2 tokens - local syntax)
     * Band 1 (Heads 4-7):   gamma ≈ 0.95  (half-life ≈ 13.5 tokens - phrase level)
     * Band 2 (Heads 8-11):  gamma ≈ 0.99  (half-life ≈ 69 tokens - paragraph level)
     * Band 3 (Heads 12-15): gamma ≈ 0.999 (half-life ≈ 693 tokens - document memory)
   - State normalization on reads to eliminate gradient and magnitude explosion.

2. HybridRecurrentSlidingAttention (Track U):
   - Recurrent infinite associative memory state S_t
   - PLUS exact local sliding-window attention buffer of width W=16 tokens
   - O(N) complexity preserved (W is a small constant).

3. GatedWriteAssociativeAttention (Track W):
   - Input-dependent write strength: w_t = sigmoid(W_write * x_t)
   - S_t = gamma * S_{t-1} + w_t * (v_t (x) k_t^T)
   - Protects associative memory from low-entropy boilerplate pollution.
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

from jarvis_model import RotaryEmbedding, RMSNorm
from utils.ternary_ops import TernaryLinear


class MultiTimescaleAttention(nn.Module):
    """
    Research Track A3 & V: Multi-Timescale Associative Attention with State Normalization.
    Allocates heads to 4 distinct memory horizons: short, medium, long, persistent.
    """
    CHUNK_SIZE = 64

    def __init__(self, d_model=1024, n_heads=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        # 4 distinct timescale initialization bands
        # sigmoid(1.75) ≈ 0.85, sigmoid(2.94) ≈ 0.95, sigmoid(4.60) ≈ 0.99, sigmoid(6.90) ≈ 0.999
        init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        cs = self.CHUNK_SIZE
        i = torch.arange(cs, dtype=torch.float32)
        diff = i.unsqueeze(1) - i.unsqueeze(0)
        self.register_buffer('_diff_clamp', diff.clamp(min=0))
        self.register_buffer('_causal', (diff >= 0).float())
        self.register_buffer('_i_idx', i)
        self.register_buffer('_i_idx_p1', i + 1)
        self.register_buffer('_c_m1_m_i', (cs - 1) - i)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = (F.elu(q) + 1.0) / math.sqrt(D)
        k = F.elu(k) + 1.0
        q, k = self.rotary(q, k, start_pos=start_pos)

        # Multi-timescale decay
        log_g = torch.log(torch.sigmoid(self.gamma_raw))  # (H,)

        cs = self.CHUNK_SIZE
        dc = self._diff_clamp
        ca = self._causal
        ip1 = self._i_idx_p1
        cm1mi = self._c_m1_m_i

        decay_mat = (torch.exp(log_g.view(H, 1, 1) * dc) * ca).to(dtype=x.dtype)
        gamma_cross = torch.exp(log_g.view(H, 1) * ip1).to(dtype=x.dtype)
        gw = torch.exp(log_g.view(H, 1) * cm1mi).to(dtype=x.dtype)
        gamma_c = torch.exp(log_g * cs).view(1, H, 1, 1).to(dtype=x.dtype)

        state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        outputs = []

        for start in range(0, T, cs):
            end = min(start + cs, T)
            c = end - start
            q_c = q[:, :, start:end, :]
            k_c = k[:, :, start:end, :]
            v_c = v[:, :, start:end, :]

            if c == cs:
                dm = decay_mat
                gc_cross = gamma_cross
                gw_c = gw
                gc_state = gamma_c
            else:
                i_c = self._i_idx[:c]
                diff_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0)).clamp(min=0)
                causal_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0) >= 0).float()
                dm = (torch.exp(log_g.view(H, 1, 1) * diff_c) * causal_c).to(dtype=x.dtype)
                gc_cross = torch.exp(log_g.view(H, 1) * (i_c + 1)).to(dtype=x.dtype)
                gw_c = torch.exp(log_g.view(H, 1) * (c - 1 - i_c)).to(dtype=x.dtype)
                gc_state = torch.exp(log_g * c).view(1, H, 1, 1).to(dtype=x.dtype)

            # Intra-chunk decayed linear attention
            raw = torch.einsum('bhid,bhjd->bhij', q_c, k_c)
            scores = raw * dm.unsqueeze(0)
            intra_out = torch.einsum('bhij,bhjd->bhid', scores, v_c)

            # Cross-chunk read
            raw_cross = torch.einsum('bhde,bhie->bhid', state, q_c)
            cross_out = raw_cross * gc_cross.unsqueeze(0).unsqueeze(-1)

            chunk_out = intra_out + cross_out
            outputs.append(chunk_out)

            # State update
            v_w = v_c * gw_c.unsqueeze(0).unsqueeze(-1)
            chunk_upd = torch.einsum('bhid,bhie->bhde', v_w, k_c)
            state = gc_state * state + chunk_upd

        out = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class HybridRecurrentSlidingAttention(nn.Module):
    """
    Research Track U: Hybrid Recurrent Associative State + Local Sliding Window Buffer (W=16).
    Global long-range context is accumulated into S_t; exact local dependencies are
    computed in a high-precision sliding window buffer.
    """
    CHUNK_SIZE = 64

    def __init__(self, d_model=1024, n_heads=16, window_size=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        # Decay: 8 medium-range heads + 8 persistent heads
        init_gammas = [2.94] * 8 + [5.50] * 8
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        # Learnable blend factor between recurrent global context and local sliding window
        self.blend_gate = nn.Parameter(torch.zeros(n_heads))  # init sigmoid(0) = 0.5

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        W = self.window_size

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q_feat = (F.elu(q) + 1.0) / math.sqrt(D)
        k_feat = F.elu(k) + 1.0
        q_feat, k_feat = self.rotary(q_feat, k_feat, start_pos=start_pos)

        # 1. Exact Local Sliding-Window Attention (Width W=16)
        # Compute local attention within band 0 <= i - j < W
        t_idx = torch.arange(T, device=x.device)
        diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)  # i - j
        local_mask = (diff >= 0) & (diff < W)  # causal + within window

        # Scaled dot-product attention on local window
        local_attn = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D)
        local_attn = local_attn.masked_fill(~local_mask.view(1, 1, T, T), -1e4)
        local_probs = F.softmax(local_attn, dim=-1)
        local_out = torch.einsum('bhij,bhjd->bhid', local_probs, v).transpose(1, 2).contiguous().view(B, T, C)

        # 2. Global Recurrent Associative Attention (Per-token parallel scan)
        log_g = torch.log(torch.sigmoid(self.gamma_raw)).view(1, H, 1, 1)  # (1, H, 1, 1)
        decay_diff = (diff.clamp(min=0)).view(1, 1, T, T)
        causal_mat = (diff >= 0).float().view(1, 1, T, T)
        decay_mat = torch.exp(log_g * decay_diff) * causal_mat

        recurrent_scores = torch.einsum('bhid,bhjd->bhij', q_feat, k_feat) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', recurrent_scores, v).transpose(1, 2).contiguous().view(B, T, C)

        # 3. Blend: local exact syntax + global associative memory
        gate = torch.sigmoid(self.blend_gate).view(1, 1, H, 1)  # (1, 1, H, 1)
        l_out = local_out.view(B, T, H, D)
        r_out = recurrent_out.view(B, T, H, D)
        blended = (gate * l_out + (1.0 - gate) * r_out).contiguous().view(B, T, C)

        return self.out_proj(blended)


class GatedWriteAssociativeAttention(nn.Module):
    """
    Research Track W: Gated Write Associative Attention.
    Learnable input-dependent write gate w_t = sigmoid(W_write * x_t).
    Low-entropy tokens (e.g. whitespace, punctuation) write weakly into memory;
    high-entropy semantic tokens write strongly.
    """
    CHUNK_SIZE = 64

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
        # Initialize bias to 1.0 (default open write gate)
        nn.init.constant_(self.write_gate_proj.bias, 1.0)

        # Multi-timescale decay parameters
        init_gammas = [2.00] * 4 + [3.50] * 4 + [5.00] * 4 + [6.50] * 4
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = (F.elu(q) + 1.0) / math.sqrt(D)
        k = F.elu(k) + 1.0
        q, k = self.rotary(q, k, start_pos=start_pos)

        # Input-dependent write strength w_t in (0, 1) per token per head
        w_t = torch.sigmoid(self.write_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)  # (B, H, T, 1)
        v = v * w_t  # Scale values by write strength before memory accumulation

        # Recurrent scan with decay
        log_g = torch.log(torch.sigmoid(self.gamma_raw)).view(1, H, 1, 1)
        t_idx = torch.arange(T, device=x.device)
        diff = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).clamp(min=0).view(1, 1, T, T)
        causal = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0) >= 0).float().view(1, 1, T, T)
        decay_mat = torch.exp(log_g * diff) * causal

        scores = torch.einsum('bhid,bhjd->bhij', q, k) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', scores, v)
        out = recurrent_out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class DeltaAssociativeAttention(nn.Module):
    """
    Research Track A6 & 7: Delta-Rule Associative Memory.
    Computes an error-driven associative update rather than pure additive accumulation:
        v_pred = S_{t-1} @ k_t
        error = v_t - v_pred
        S_t = gamma S_{t-1} + beta * (error ⊗ k_t^T)
    Prevents associative memory saturation and eliminates stale associations.
    """
    def __init__(self, d_model=1024, n_heads=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        # Learning rate beta for delta update (sigmoid(beta_raw) in (0, 1))
        self.beta_raw = nn.Parameter(torch.full((n_heads,), 0.0))  # init sigmoid(0) = 0.5

        # 4 distinct timescale initialization bands
        init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Normalize keys so k_t^T k_t ≈ 1 for stable delta projection
        k = F.normalize(k, p=2, dim=-1)
        q = (F.elu(q) + 1.0) / math.sqrt(D)
        q, k = self.rotary(q, k, start_pos=start_pos)

        gamma = torch.sigmoid(self.gamma_raw).view(1, H, 1, 1)
        beta = torch.sigmoid(self.beta_raw).view(1, H, 1, 1)

        # Sequential scan with delta error update
        # S_t = gamma * S_{t-1} + beta * (v_t - S_{t-1} @ k_t) ⊗ k_t^T
        state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            q_t = q[:, :, t, :]    # (B, H, D)
            k_t = k[:, :, t, :]    # (B, H, D)
            v_t = v[:, :, t, :]    # (B, H, D)

            # Predict current association from carried memory: v_pred = S_{t-1} @ k_t
            v_pred = torch.einsum('bhde,bhe->bhd', state, k_t)
            error = v_t - v_pred

            # Output read: z_t = S_t @ q_t
            z_t = torch.einsum('bhde,bhe->bhd', state, q_t)
            outputs.append(z_t)

            # Delta update: S_t = gamma * S_{t-1} + beta * (error ⊗ k_t^T)
            upd = torch.einsum('bhd,bhe->bhde', error, k_t)
            state = gamma * state + beta * upd

        out = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class WriteEraseAssociativeAttention(nn.Module):
    """
    Phase 1A: Separate Write and Erase Gate Associative Attention.
    - Write gate: w_t = sigmoid(W_write * x_t + b_write) controls information insertion
    - Erase gate: e_t = sigmoid(W_erase * x_t + b_erase) controls selective state erasure
    - Effective retention factor: alpha_t = gamma * (1 - e_t)
    - Vectorized parallel cumsum scan computes exact recurrent state dynamics.
    """
    def __init__(self, d_model=1024, n_heads=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        self.write_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.write_gate_proj.bias, 1.0)

        self.erase_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.erase_gate_proj.bias, -2.0)

        init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = (F.elu(q) + 1.0) / math.sqrt(D)
        k = F.elu(k) + 1.0
        q, k = self.rotary(q, k, start_pos=start_pos)

        w_t = torch.sigmoid(self.write_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        e_t = torch.sigmoid(self.erase_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)

        gamma = torch.sigmoid(self.gamma_raw).view(1, H, 1, 1)
        alpha_t = (gamma * (1.0 - e_t)).clamp(min=1e-4, max=1.0 - 1e-4)

        log_alpha = torch.log(alpha_t.squeeze(-1).float())
        L = torch.cumsum(log_alpha, dim=-1)

        t_idx = torch.arange(T, device=x.device)
        causal_mask = (t_idx.unsqueeze(1) >= t_idx.unsqueeze(0)).view(1, 1, T, T)
        L_diff = L.unsqueeze(-1) - L.unsqueeze(-2)
        L_diff_clamped = torch.where(causal_mask, L_diff.clamp(min=-50.0, max=0.0), torch.tensor(-1e4, device=x.device, dtype=torch.float32))
        decay_mat = torch.where(causal_mask, torch.exp(L_diff_clamped), torch.zeros_like(L_diff_clamped)).to(x.dtype)

        v_weighted = v * w_t
        scores = torch.einsum('bhid,bhjd->bhij', q, k) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', scores, v_weighted)

        out = recurrent_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class HybridWriteEraseBufferAttention(nn.Module):
    """
    Phase 1A + Local Buffer:
    Combines:
    1. Separate input-dependent Write and Erase Gates
    2. Recurrent Multi-Timescale Associative Memory via parallel cumsum scan
    3. Exact Local Sliding-Window Attention Buffer (W=16)
    4. Learnable per-head blend gate between local buffer and recurrent memory
    """
    def __init__(self, d_model=1024, n_heads=16, window_size=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        self.write_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.write_gate_proj.bias, 1.0)

        self.erase_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.erase_gate_proj.bias, -2.0)

        init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        self.blend_gate = nn.Parameter(torch.zeros(n_heads))

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        W = self.window_size

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

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

        # 2. Recurrent Memory with Write and Erase Gating
        w_t = torch.sigmoid(self.write_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        e_t = torch.sigmoid(self.erase_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        gamma = torch.sigmoid(self.gamma_raw).view(1, H, 1, 1)
        alpha_t = (gamma * (1.0 - e_t)).clamp(min=1e-4, max=1.0 - 1e-4)

        log_alpha = torch.log(alpha_t.squeeze(-1).float())
        L = torch.cumsum(log_alpha, dim=-1)
        causal_mask = (diff >= 0).view(1, 1, T, T)
        L_diff = L.unsqueeze(-1) - L.unsqueeze(-2)
        L_diff_clamped = torch.where(causal_mask, L_diff.clamp(min=-50.0, max=0.0), torch.tensor(-1e4, device=x.device, dtype=torch.float32))
        decay_mat = torch.where(causal_mask, torch.exp(L_diff_clamped), torch.zeros_like(L_diff_clamped)).to(x.dtype)

        v_weighted = v * w_t
        recurrent_scores = torch.einsum('bhid,bhjd->bhij', q_feat, k_feat) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', recurrent_scores, v_weighted).transpose(1, 2).contiguous().view(B, T, C)

        # 3. Blend local exact syntax + long-range write/erase memory
        gate = torch.sigmoid(self.blend_gate).view(1, 1, H, 1)
        l_out = local_out.view(B, T, H, D)
        r_out = recurrent_out.view(B, T, H, D)
        blended = (gate * l_out + (1.0 - gate) * r_out).contiguous().view(B, T, C)

        return self.out_proj(blended)


class AdaptiveDecayAssociativeAttention(nn.Module):
    """
    Phase 1B: Input-dependent adaptive decay:
    gamma_t = gamma_min + (gamma_max - gamma_min) * sigmoid(W_gamma * x_t + b_gamma)
    """
    def __init__(self, d_model=1024, n_heads=16, gamma_min=0.80, gamma_max=0.999, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        self.gamma_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.gamma_proj.bias)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = (F.elu(q) + 1.0) / math.sqrt(D)
        k = F.elu(k) + 1.0
        q, k = self.rotary(q, k, start_pos=start_pos)

        sig_g = torch.sigmoid(self.gamma_proj(x)).view(B, T, H, 1).transpose(1, 2)
        gamma_t = (self.gamma_min + (self.gamma_max - self.gamma_min) * sig_g).clamp(min=1e-4, max=1.0 - 1e-4)

        log_gamma = torch.log(gamma_t.squeeze(-1).float())
        L = torch.cumsum(log_gamma, dim=-1)

        t_idx = torch.arange(T, device=x.device)
        causal_mask = (t_idx.unsqueeze(1) >= t_idx.unsqueeze(0)).view(1, 1, T, T)
        L_diff = L.unsqueeze(-1) - L.unsqueeze(-2)
        L_diff_clamped = torch.where(causal_mask, L_diff.clamp(min=-50.0, max=0.0), torch.tensor(-1e4, device=x.device, dtype=torch.float32))
        decay_mat = torch.where(causal_mask, torch.exp(L_diff_clamped), torch.zeros_like(L_diff_clamped)).to(x.dtype)

        scores = torch.einsum('bhid,bhjd->bhij', q, k) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', scores, v)

        out = recurrent_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class GatedReadAssociativeAttention(nn.Module):
    """
    Phase 1D: Gated Read Associative Attention:
    read_gate = sigmoid(W_read * x_t + b_read)
    O_t = read_gate * (S_t @ Q_t)
    """
    def __init__(self, d_model=1024, n_heads=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        self.read_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.read_gate_proj.bias, 1.0)

        init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = (F.elu(q) + 1.0) / math.sqrt(D)
        k = F.elu(k) + 1.0
        q, k = self.rotary(q, k, start_pos=start_pos)

        log_g = torch.log(torch.sigmoid(self.gamma_raw)).view(1, H, 1, 1)
        t_idx = torch.arange(T, device=x.device)
        diff = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).clamp(min=0).view(1, 1, T, T)
        causal = (t_idx.unsqueeze(1) >= t_idx.unsqueeze(0)).float().view(1, 1, T, T)
        decay_mat = torch.exp(log_g * diff) * causal

        scores = torch.einsum('bhid,bhjd->bhij', q, k) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', scores, v)

        r_t = torch.sigmoid(self.read_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        gated_out = recurrent_out * r_t

        out = gated_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class EraseGateAssociativeAttention(nn.Module):
    """
    Phase 2 Variant 4: Baseline + Erase Gate Associative Attention.
    - Erase gate: e_t = sigmoid(W_erase * x_t + b_erase) controls selective state erasure
    - Effective retention factor: alpha_t = gamma * (1 - e_t)
    - Normal writes (no write gate): isolates erase gate mechanism completely.
    """
    def __init__(self, d_model=1024, n_heads=16, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        self.erase_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.erase_gate_proj.bias, -2.0)

        init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = (F.elu(q) + 1.0) / math.sqrt(D)
        k = F.elu(k) + 1.0
        q, k = self.rotary(q, k, start_pos=start_pos)

        e_t = torch.sigmoid(self.erase_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        gamma = torch.sigmoid(self.gamma_raw).view(1, H, 1, 1)
        alpha_t = (gamma * (1.0 - e_t)).clamp(min=1e-4, max=1.0 - 1e-4)

        log_alpha = torch.log(alpha_t.squeeze(-1).float())
        L = torch.cumsum(log_alpha, dim=-1)

        t_idx = torch.arange(T, device=x.device)
        causal_mask = (t_idx.unsqueeze(1) >= t_idx.unsqueeze(0)).view(1, 1, T, T)
        L_diff = L.unsqueeze(-1) - L.unsqueeze(-2)
        L_diff_clamped = torch.where(causal_mask, L_diff.clamp(min=-50.0, max=0.0), torch.tensor(-1e4, device=x.device, dtype=torch.float32))
        decay_mat = torch.where(causal_mask, torch.exp(L_diff_clamped), torch.zeros_like(L_diff_clamped)).to(x.dtype)

        scores = torch.einsum('bhid,bhjd->bhij', q, k) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', scores, v)

        out = recurrent_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class AdaptiveWriteEraseBufferAttention(nn.Module):
    """
    Phase 2 Variant 10 (Best Memory Combination):
    Combines all empirically validated memory mechanisms into a unified hierarchy:
    1. Local High-Precision Sliding Buffer (W=16) with Rotary Positional Embeddings
    2. Input-Dependent Adaptive Decay: gamma_t = gamma_min + (gamma_max - gamma_min) * sigmoid(W_gamma * x_t + b_gamma)
    3. Input-Dependent Write Gating: w_t = sigmoid(W_write * x_t + b_write)
    4. Input-Dependent Erase Gating: e_t = sigmoid(W_erase * x_t + b_erase)
    5. Input-Dependent Gated Read: r_t = sigmoid(W_read * x_t + b_read)
    6. Effective retention: alpha_t = gamma_t * (1 - e_t) via numerically stable parallel scan
    7. Per-head learnable local-global blending gate
    """
    def __init__(self, d_model=1024, n_heads=16, window_size=16, gamma_min=0.80, gamma_max=0.999, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        self.write_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.write_gate_proj.bias, 1.0)

        self.erase_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.erase_gate_proj.bias, -2.0)

        self.gamma_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.gamma_proj.bias)

        self.read_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.read_gate_proj.bias, 1.0)

        self.blend_gate = nn.Parameter(torch.zeros(n_heads))
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        W = self.window_size

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q_feat = (F.elu(q) + 1.0) / math.sqrt(D)
        k_feat = F.elu(k) + 1.0
        q_feat, k_feat = self.rotary(q_feat, k_feat, start_pos=start_pos)

        # 1. Local Sliding-Window Attention (W=16)
        t_idx = torch.arange(T, device=x.device)
        diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)
        local_mask = (diff >= 0) & (diff < W)

        local_attn = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D)
        local_attn = local_attn.masked_fill(~local_mask.view(1, 1, T, T), -1e4)
        local_probs = F.softmax(local_attn, dim=-1)
        local_out = torch.einsum('bhij,bhjd->bhid', local_probs, v).transpose(1, 2).contiguous().view(B, T, C)

        # 2. Dynamic Memory Gating: Write, Erase, Adaptive Decay, Gated Read
        w_t = torch.sigmoid(self.write_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        e_t = torch.sigmoid(self.erase_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
        r_t = torch.sigmoid(self.read_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)

        sig_g = torch.sigmoid(self.gamma_proj(x)).view(B, T, H, 1).transpose(1, 2)
        gamma_t = (self.gamma_min + (self.gamma_max - self.gamma_min) * sig_g).clamp(min=1e-4, max=1.0 - 1e-4)

        alpha_t = (gamma_t * (1.0 - e_t)).clamp(min=1e-4, max=1.0 - 1e-4)
        log_alpha = torch.log(alpha_t.squeeze(-1).float())
        L = torch.cumsum(log_alpha, dim=-1)

        causal_mask = (diff >= 0).view(1, 1, T, T)
        L_diff = L.unsqueeze(-1) - L.unsqueeze(-2)
        L_diff_clamped = torch.where(causal_mask, L_diff.clamp(min=-50.0, max=0.0), torch.tensor(-1e4, device=x.device, dtype=torch.float32))
        decay_mat = torch.where(causal_mask, torch.exp(L_diff_clamped), torch.zeros_like(L_diff_clamped)).to(x.dtype)

        v_weighted = v * w_t
        recurrent_scores = torch.einsum('bhid,bhjd->bhij', q_feat, k_feat) * decay_mat
        raw_recurrent = torch.einsum('bhij,bhjd->bhid', recurrent_scores, v_weighted)
        recurrent_out = (raw_recurrent * r_t).transpose(1, 2).contiguous().view(B, T, C)

        # 3. Blend local exact syntax + long-range adaptive memory
        gate = torch.sigmoid(self.blend_gate).view(1, 1, H, 1)
        l_out = local_out.view(B, T, H, D)
        r_out = recurrent_out.view(B, T, H, D)
        blended = (gate * l_out + (1.0 - gate) * r_out).contiguous().view(B, T, C)

        return self.out_proj(blended)




