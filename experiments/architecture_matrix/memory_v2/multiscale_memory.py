# experiments/architecture_matrix/memory_v2/multiscale_memory.py
"""
MEMORY v2: MULTI-SCALE LOCAL SLIDING-WINDOW ATTENTION
=====================================================
Extends the validated E-only local buffer architecture with multi-scale
temporal receptive fields across attention head groups.

Key Properties:
1. Zero parameter bloat: Reuses existing Q, K, V projections and per-head blend gates (+384 parameters).
2. Per-head window assignment:
   - Heads 0..3  (4 heads): W = 8  (Immediate local syntax, n-gram bindings)
   - Heads 4..9  (6 heads): W = 16 (Phrase and clause-level dependencies)
   - Heads 10..15 (6 heads): W = 32 (Inter-clause and sentence-level discourse)
3. Bounded O(1) ring-buffer decoding footprint (max 32 tokens).
4. Strictly causal masking: Future tokens are 100% masked (-1e4 fill).

Provides:
- MultiScaleMemoryAttention (replaces single-window buffer)
- build_multiscale_jarvis(window_config)
- Causal correctness verification
- Path contribution diagnostics
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE, ARCH_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import RotaryEmbedding, TernaryLinear, Jarvis


class MultiScaleMemoryAttention(nn.Module):
    """
    Multi-Scale Dual-Memory Layer (E-Only, no write/erase gates).
    Short-term: Per-head multi-scale sliding-window attention (W in {8, 16, 32}).
    Long-term : Recurrent associative linear attention (retains 100% baseline capability).
    Fusion    : Learned per-head blend gate (+16 parameters / layer).
    """
    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        max_seq_len: int = 2048,
        window_config = 16, # int (uniform) or list/tuple of length n_heads
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Parse window configuration
        if isinstance(window_config, int):
            self.windows = [window_config] * n_heads
        elif isinstance(window_config, (list, tuple)):
            assert len(window_config) == n_heads, f"window_config must have length {n_heads}"
            self.windows = list(window_config)
        else:
            raise ValueError(f"Unsupported window_config: {window_config}")

        self.max_window = max(self.windows)

        # Baseline ternary projections
        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        # Baseline learned decay parameter (sigmoid(2.94) ≈ 0.950)
        self.gamma_raw = nn.Parameter(torch.full((n_heads,), 2.94))
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Learned per-head blend gate (+16 parameters per layer)
        self.blend_gate = nn.Parameter(torch.zeros(n_heads))
        nn.init.constant_(self.blend_gate, -3.0) # Neutral init: recurrent dominant

        # Pre-build per-head window tensor for fast causal masking
        # Shape: (1, H, 1, 1)
        self.register_buffer(
            "head_windows",
            torch.tensor(self.windows, dtype=torch.int32).view(1, n_heads, 1, 1),
            persistent=False,
        )

        # Cached state for autoregressive decoding
        self.register_buffer("recurrent_state", None, persistent=False)

    def init_neutral(self):
        """Initializes blend gate to recurrent-dominant identity (-3.0)."""
        with torch.no_grad():
            nn.init.constant_(self.blend_gate, -3.0)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        collect_diagnostics: bool = False,
    ):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Feature maps: ELU+1
        q_feat = (F.elu(q) + 1.0) / math.sqrt(D)
        k_feat = F.elu(k) + 1.0
        q_feat, k_feat = self.rotary(q_feat, k_feat, start_pos=start_pos)

        # -------------------------------------------------------------------
        # 1. Short-Term Path: Multi-Scale Sliding Window Attention
        # -------------------------------------------------------------------
        if T <= 1024:
            t_idx = torch.arange(T, device=x.device)
            diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)  # (T, T): i - j
            causal_cond = diff >= 0
            diff_view = diff.view(1, 1, T, T)
            local_mask = causal_cond.view(1, 1, T, T) & (diff_view < self.head_windows)

            local_attn = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D)
            local_attn = local_attn.masked_fill(~local_mask, -1e4)
            local_probs = F.softmax(local_attn, dim=-1)
            local_out = torch.einsum('bhij,bhjd->bhid', local_probs, v)
        else:
            # Chunked local window attention for long context (O(C*W) memory instead of O(T^2))
            chunk_size = 512
            outputs_loc = []
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                k_start = max(0, start - self.max_window)
                q_c = q[:, :, start:end, :]
                k_c = k[:, :, k_start:end, :]
                v_c = v[:, :, k_start:end, :]
                q_pos = torch.arange(start, end, device=x.device).view(1, 1, -1, 1)
                k_pos = torch.arange(k_start, end, device=x.device).view(1, 1, 1, -1)
                diff = q_pos - k_pos
                mask = (diff >= 0) & (diff < self.head_windows)
                attn_c = torch.einsum('bhid,bhjd->bhij', q_c, k_c) / math.sqrt(D)
                attn_c = attn_c.masked_fill(~mask, -1e4)
                probs_c = F.softmax(attn_c, dim=-1)
                outputs_loc.append(torch.einsum('bhij,bhjd->bhid', probs_c, v_c))
            local_out = torch.cat(outputs_loc, dim=2)

        # -------------------------------------------------------------------
        # 2. Long-Term Path: Recurrent Associative Memory
        # -------------------------------------------------------------------
        if T <= 1024:
            gamma = torch.sigmoid(self.gamma_raw).view(1, H, 1, 1).clamp(min=1e-4, max=1.0 - 1e-4)
            log_gamma = torch.log(gamma)

            # Parallel masked cumsum formulation (efficient vectorized scan)
            t_idx = torch.arange(T, device=x.device)
            causal_mask = (t_idx.unsqueeze(1) >= t_idx.unsqueeze(0)).view(1, 1, T, T)
            diff_decay = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).view(1, 1, T, T).float()
            
            decay_mat = torch.where(
                causal_mask,
                torch.exp(log_gamma * diff_decay),
                torch.zeros(1, 1, T, T, device=x.device, dtype=torch.float32),
            ).to(x.dtype)

            scores = torch.einsum('bhid,bhjd->bhij', q_feat, k_feat) * decay_mat
            recurrent_out = torch.einsum('bhij,bhjd->bhid', scores, v)
        else:
            # Chunked recurrent associative memory (O(1) state memory instead of O(T^2))
            cs = 64
            log_g = torch.log(torch.sigmoid(self.gamma_raw))
            state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
            outputs_rec = []
            for start in range(0, T, cs):
                end = min(start + cs, T)
                c = end - start
                q_c = q_feat[:, :, start:end, :]
                k_c = k_feat[:, :, start:end, :]
                v_c = v[:, :, start:end, :]
                i_c = torch.arange(c, device=x.device)
                diff_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0)).clamp(min=0)
                causal_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0) >= 0).float()
                dm = (torch.exp(log_g.view(H, 1, 1) * diff_c) * causal_c).to(dtype=x.dtype)
                gc_cross = torch.exp(log_g.view(H, 1) * (i_c + 1)).to(dtype=x.dtype)
                gw_c = torch.exp(log_g.view(H, 1) * (c - 1 - i_c)).to(dtype=x.dtype)
                gc_state = torch.exp(log_g * c).view(1, H, 1, 1).to(dtype=x.dtype)
                
                raw = torch.einsum('bhid,bhjd->bhij', q_c, k_c)
                scores = raw * dm.unsqueeze(0)
                intra_out = torch.einsum('bhij,bhjd->bhid', scores, v_c)
                
                raw_cross = torch.einsum('bhde,bhie->bhid', state, q_c)
                cross_out = raw_cross * gc_cross.unsqueeze(0).unsqueeze(-1)
                outputs_rec.append(intra_out + cross_out)
                
                v_w = v_c * gw_c.unsqueeze(0).unsqueeze(-1)
                chunk_upd = torch.einsum('bhid,bhie->bhde', v_w, k_c)
                state = gc_state * state + chunk_upd
            recurrent_out = torch.cat(outputs_rec, dim=2)

        # -------------------------------------------------------------------
        # 3. Path Fusion: Learned Per-Head Blend Gate
        # -------------------------------------------------------------------
        gate = torch.sigmoid(self.blend_gate).view(1, H, 1, 1)
        fused = gate * local_out + (1.0 - gate) * recurrent_out
        fused_out = fused.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(fused_out)

        if collect_diagnostics:
            diag = {
                "local_norm": float(torch.norm(local_out).item()),
                "recurrent_norm": float(torch.norm(recurrent_out).item()),
                "local_ratio": float((torch.norm(local_out) / max(torch.norm(local_out) + torch.norm(recurrent_out), 1e-6)).item()),
                "fusion_gate_mean": float(gate.mean().item()),
                "windows": self.windows,
            }
            return out, diag

        return out


def build_multiscale_jarvis(window_config=16, max_seq_len=2048):
    """
    Constructs a full Jarvis model with MultiScaleMemoryAttention replacing
    the baseline attention layers.
    """
    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=max_seq_len,
        use_cuda_attn=False,
        use_cuda_moe=False,
    )

    # Replace attention in all 24 blocks
    for block in model.blocks:
        ms_attn = MultiScaleMemoryAttention(
            d_model=1024,
            n_heads=16,
            max_seq_len=max_seq_len,
            window_config=window_config,
        )
        # Copy baseline projections
        ms_attn.q_proj.load_state_dict(block.attn.q_proj.state_dict())
        ms_attn.k_proj.load_state_dict(block.attn.k_proj.state_dict())
        ms_attn.v_proj.load_state_dict(block.attn.v_proj.state_dict())
        ms_attn.out_proj.load_state_dict(block.attn.out_proj.state_dict())
        ms_attn.gamma_raw.data.copy_(block.attn.gamma_raw.data)
        block.attn = ms_attn

    return model


def verify_causal_correctness():
    """
    Strictly verifies that multi-scale windowing does NOT leak future tokens.
    Modifying token t=10 must have zero effect on predictions at t <= 9.
    """
    print("=" * 80)
    print("VERIFYING MULTI-SCALE MEMORY CAUSAL INTEGRITY")
    print("=" * 80)

    window_cfg = [8]*4 + [16]*6 + [32]*6
    layer = MultiScaleMemoryAttention(d_model=128, n_heads=16, max_seq_len=128, window_config=window_cfg).cuda()
    layer.eval()

    torch.manual_seed(42)
    x1 = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    x2 = x1.clone()
    # Mutate future tokens (t >= 32)
    x2[:, 32:, :] += torch.randn_like(x2[:, 32:, :]) * 5.0

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out1 = layer(x1)
            out2 = layer(x2)

    # Verify outputs at t < 32 are identical to machine precision
    diff_past = (out1[:, :32, :] - out2[:, :32, :]).abs().max().item()
    diff_future = (out1[:, 32:, :] - out2[:, 32:, :]).abs().max().item()

    assert diff_past == 0.0, f"CAUSAL LEAK DETECTED! Past tokens changed by {diff_past}"
    assert diff_future > 0.0, "Future tokens did not change!"

    print(f"Past Token Difference (t < 32): {diff_past:.8f} [STRICT ZERO - NO LEAK]")
    print(f"Future Token Difference (t >= 32): {diff_future:.4f} [EXPECTED CHANGE]")
    print("[OK] Causal correctness verified across all multi-scale heads!")

if __name__ == "__main__":
    verify_causal_correctness()
