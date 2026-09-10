# experiments/architecture_matrix/modular_memory.py
"""
MODULAR MEMORY ARCHITECTURE FOR JARVIS-600M
============================================
Refactored, highly configurable dual-path memory system with independent
toggles and neutral initialization for controlled factorial testing:

Components:
A. LOCAL BUFFER: Causal sliding-window high-precision local memory (W=16)
B. ADAPTIVE DECAY: Input-dependent decay gamma_t = gamma_min + (gamma_max - gamma_min)*sigmoid(W_gamma*x + b_gamma)
C. WRITE GATE: Input-dependent write strength w_t = sigmoid(W_write*x + b_write)
D. ERASE GATE: Selective memory clearing e_t = sigmoid(W_erase*x + b_erase)
E. GATED READ: Output readout filter r_t = sigmoid(W_read*x + b_read)
F. MULTI-TIMESCALE: Separately selectable fixed multi-timescale decay bands

Fusion Options (Short-Term Local vs Long-Term Recurrent):
1. Additive: output = local + recurrent
2. Learned Gate: output = gate * local + (1 - gate) * recurrent
3. Hierarchical: local -> recurrent write controller; recurrent -> local residual
"""

import math
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

from jarvis_model import TernaryLinear, RotaryEmbedding, RMSNorm


class ModularMemoryAttention(nn.Module):
    """
    Unified Modular Memory Block.
    Any subset of mechanisms can be enabled via configuration flags.
    """
    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        max_seq_len: int = 2048,
        use_local_buffer: bool = False,
        local_window_size: int = 16,
        use_adaptive_decay: bool = False,
        use_write_gate: bool = False,
        use_erase_gate: bool = False,
        use_gated_read: bool = False,
        use_multi_timescale: bool = False,
        fusion_option: int = 2,
        gamma_min: float = 0.80,
        gamma_max: float = 0.999,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Flags
        self.use_local_buffer = use_local_buffer
        self.local_window_size = local_window_size
        self.use_adaptive_decay = use_adaptive_decay
        self.use_write_gate = use_write_gate
        self.use_erase_gate = use_erase_gate
        self.use_gated_read = use_gated_read
        self.use_multi_timescale = use_multi_timescale
        self.fusion_option = fusion_option
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        # Core Ternary Projections (Inherited from baseline)
        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)

        # Base recurrent decay parameter (16 heads)
        if use_multi_timescale:
            init_gammas = [1.75] * 4 + [2.94] * 4 + [4.60] * 4 + [6.90] * 4
        else:
            init_gammas = [2.94] * n_heads  # baseline: sigmoid(2.94) ≈ 0.950
        self.gamma_raw = nn.Parameter(torch.tensor(init_gammas, dtype=torch.float32))

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Mechanism B: Adaptive Decay Projection
        if self.use_adaptive_decay:
            self.gamma_proj = nn.Linear(d_model, n_heads, bias=True)
        else:
            self.gamma_proj = None

        # Mechanism C: Write Gate Projection
        if self.use_write_gate:
            self.write_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        else:
            self.write_gate_proj = None

        # Mechanism D: Erase Gate Projection
        if self.use_erase_gate:
            self.erase_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        else:
            self.erase_gate_proj = None

        # Mechanism E: Gated Read Projection
        if self.use_gated_read:
            self.read_gate_proj = nn.Linear(d_model, n_heads, bias=True)
        else:
            self.read_gate_proj = None

        # Short-term / Long-term Fusion
        if self.use_local_buffer:
            if self.fusion_option == 2:
                # Per-head blend gate
                self.blend_gate = nn.Parameter(torch.zeros(n_heads))
            elif self.fusion_option == 3:
                # Hierarchical: local modulates recurrent write, recurrent injects into local
                self.local_to_write = nn.Linear(d_model, n_heads, bias=False)
                self.rec_inject_gate = nn.Linear(d_model, n_heads, bias=True)

        # Persistent Recurrent State for step-by-step decode
        self.register_buffer("recurrent_state", None, persistent=False)

        # Initialize neutral / identity defaults
        self.init_neutral()

    def init_neutral(self):
        """
        Mathematically initializes newly introduced gates to baseline identity behavior:
        - Write gate: open (w_t ≈ 0.982)
        - Erase gate: closed (e_t ≈ 0.018, 1 - e_t ≈ 0.982)
        - Read gate: open (r_t ≈ 0.982)
        - Adaptive decay: exactly reproduces baseline gamma (~0.950)
        - Local buffer blend: recurrent-dominant (gate ≈ 0.047)
        """
        with torch.no_grad():
            if self.write_gate_proj is not None:
                nn.init.zeros_(self.write_gate_proj.weight)
                nn.init.constant_(self.write_gate_proj.bias, 4.0)

            if self.erase_gate_proj is not None:
                nn.init.zeros_(self.erase_gate_proj.weight)
                nn.init.constant_(self.erase_gate_proj.bias, -4.0)

            if self.read_gate_proj is not None:
                nn.init.zeros_(self.read_gate_proj.weight)
                nn.init.constant_(self.read_gate_proj.bias, 4.0)

            if self.gamma_proj is not None:
                nn.init.zeros_(self.gamma_proj.weight)
                # Baseline gamma is sigmoid(2.94) = 0.950
                target_sig = (0.950 - self.gamma_min) / max(self.gamma_max - self.gamma_min, 1e-6)
                target_sig = max(min(target_sig, 0.999), 0.001)
                target_b = math.log(target_sig / (1.0 - target_sig))
                nn.init.constant_(self.gamma_proj.bias, target_b)

            if hasattr(self, "blend_gate") and self.blend_gate is not None:
                nn.init.constant_(self.blend_gate, -3.0)

            if hasattr(self, "rec_inject_gate") and self.rec_inject_gate is not None:
                nn.init.zeros_(self.rec_inject_gate.weight)
                nn.init.constant_(self.rec_inject_gate.bias, -2.0)

    def reset_state(self):
        """Resets cached recurrent associative memory state."""
        self.recurrent_state = None

    def detach_state(self):
        """Detaches cached state from autograd graph for TBPTT."""
        if self.recurrent_state is not None:
            self.recurrent_state = self.recurrent_state.detach()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        collect_diagnostics: bool = False,
    ):
        """
        Forward pass with dual-path memory and optional diagnostic instrumentation.
        """
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        W = self.local_window_size

        # Linear projections
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Feature maps: ELU+1
        q_feat = (F.elu(q) + 1.0) / math.sqrt(D)
        k_feat = F.elu(k) + 1.0
        q_feat, k_feat = self.rotary(q_feat, k_feat, start_pos=start_pos)

        # -------------------------------------------------------------------
        # 1. Short-Term Path: Local Sliding-Window Attention (W=16)
        # -------------------------------------------------------------------
        if self.use_local_buffer:
            t_idx = torch.arange(T, device=x.device)
            diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)  # i - j
            local_mask = (diff >= 0) & (diff < W)

            local_attn = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D)
            local_attn = local_attn.masked_fill(~local_mask.view(1, 1, T, T), -1e4)
            local_probs = F.softmax(local_attn, dim=-1)
            local_out = torch.einsum('bhij,bhjd->bhid', local_probs, v).transpose(1, 2).contiguous().view(B, T, C)
        else:
            local_out = None

        # -------------------------------------------------------------------
        # 2. Long-Term Path: Recurrent Associative Memory
        # -------------------------------------------------------------------
        # Compute Write Gate
        if self.use_write_gate:
            if self.fusion_option == 3 and local_out is not None:
                # Hierarchical: local representation modulates write gate
                w_raw = self.write_gate_proj(x) + self.local_to_write(local_out)
            else:
                w_raw = self.write_gate_proj(x)
            w_t = torch.sigmoid(w_raw).view(B, T, H, 1).transpose(1, 2)
        else:
            w_t = 1.0

        # Compute Decay Factor gamma_t
        if self.use_adaptive_decay:
            sig_g = torch.sigmoid(self.gamma_proj(x)).view(B, T, H, 1).transpose(1, 2)
            gamma_t = self.gamma_min + (self.gamma_max - self.gamma_min) * sig_g
        else:
            gamma_t = torch.sigmoid(self.gamma_raw).view(1, H, 1, 1).expand(B, H, T, 1)

        # Compute Erase Gate and Effective Retention alpha_t
        if self.use_erase_gate:
            e_t = torch.sigmoid(self.erase_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
            alpha_t = (gamma_t * (1.0 - e_t)).clamp(min=1e-4, max=1.0 - 1e-4)
        else:
            e_t = None
            alpha_t = gamma_t.clamp(min=1e-4, max=1.0 - 1e-4)

        # Numerically Stable Masked Cumsum Recurrent Scan
        log_alpha = torch.log(alpha_t.squeeze(-1).float())  # shape is (B, H, T)
        L = torch.cumsum(log_alpha, dim=-1)

        t_idx = torch.arange(T, device=x.device)
        causal_mask = (t_idx.unsqueeze(1) >= t_idx.unsqueeze(0)).view(1, 1, T, T)
        L_diff = L.unsqueeze(-1) - L.unsqueeze(-2)
        L_diff_clamped = torch.where(
            causal_mask,
            L_diff.clamp(min=-50.0, max=0.0),
            torch.tensor(-1e4, device=x.device, dtype=torch.float32),
        )
        decay_mat = torch.where(
            causal_mask,
            torch.exp(L_diff_clamped),
            torch.zeros_like(L_diff_clamped),
        ).to(x.dtype)

        v_weighted = v * w_t if isinstance(w_t, torch.Tensor) else v
        scores = torch.einsum('bhid,bhjd->bhij', q_feat, k_feat) * decay_mat
        recurrent_out = torch.einsum('bhij,bhjd->bhid', scores, v_weighted)

        # Compute Gated Read
        if self.use_gated_read:
            r_t = torch.sigmoid(self.read_gate_proj(x)).view(B, T, H, 1).transpose(1, 2)
            recurrent_out = recurrent_out * r_t
        else:
            r_t = None

        recurrent_out = recurrent_out.transpose(1, 2).contiguous().view(B, T, C)

        # -------------------------------------------------------------------
        # 3. Path Fusion: Short-Term + Long-Term Integration
        # -------------------------------------------------------------------
        if not self.use_local_buffer:
            fused_out = recurrent_out
            fusion_gate_val = 0.0
        elif self.fusion_option == 1:
            # Option 1: Direct Additive
            fused_out = local_out + recurrent_out
            fusion_gate_val = 0.5
        elif self.fusion_option == 2:
            # Option 2: Learned Per-Head Blend Gate
            gate = torch.sigmoid(self.blend_gate).view(1, 1, H, 1)
            l_view = local_out.view(B, T, H, D)
            r_view = recurrent_out.view(B, T, H, D)
            fused_out = (gate * l_view + (1.0 - gate) * r_view).contiguous().view(B, T, C)
            fusion_gate_val = gate.mean().item()
        elif self.fusion_option == 3:
            # Option 3: Hierarchical Injection
            inj_gate = torch.sigmoid(self.rec_inject_gate(x)).view(B, T, H, 1)
            l_view = local_out.view(B, T, H, D)
            r_view = recurrent_out.view(B, T, H, D)
            fused_out = (l_view + inj_gate * r_view).contiguous().view(B, T, C)
            fusion_gate_val = inj_gate.mean().item()
        else:
            raise ValueError(f"Unknown fusion_option: {self.fusion_option}")

        out = self.out_proj(fused_out)

        # Optional Diagnostic Collection
        if collect_diagnostics:
            diag = {
                "gamma_mean": gamma_t.mean().item() if isinstance(gamma_t, torch.Tensor) else float(gamma_t),
                "gamma_std": gamma_t.std().item() if isinstance(gamma_t, torch.Tensor) and gamma_t.numel() > 1 else 0.0,
                "write_gate_mean": w_t.mean().item() if isinstance(w_t, torch.Tensor) else 1.0,
                "write_gate_std": w_t.std().item() if isinstance(w_t, torch.Tensor) else 0.0,
                "erase_gate_mean": e_t.mean().item() if isinstance(e_t, torch.Tensor) else 0.0,
                "erase_gate_std": e_t.std().item() if isinstance(e_t, torch.Tensor) else 0.0,
                "read_gate_mean": r_t.mean().item() if isinstance(r_t, torch.Tensor) else 1.0,
                "read_gate_std": r_t.std().item() if isinstance(r_t, torch.Tensor) else 0.0,
                "local_norm": local_out.norm().item() / max(local_out.numel(), 1) if local_out is not None else 0.0,
                "recurrent_norm": recurrent_out.norm().item() / max(recurrent_out.numel(), 1),
                "fusion_ratio": fusion_gate_val,
            }
            return out, diag

        return out


def build_modular_jarvis(
    config_dict: dict,
    vocab_size: int = 50257,
    d_model: int = 1024,
    n_layers: int = 24,
    n_heads: int = 16,
    num_experts: int = 4,
    top_k: int = 2,
    max_seq_len: int = 512,
):
    """
    Constructs a complete 24-layer Jarvis-600M model with ModularMemoryAttention.
    """
    from jarvis_model import Jarvis
    model = Jarvis(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        num_experts=num_experts,
        top_k=top_k,
        max_seq_len=max_seq_len,
        use_cuda_attn=False,
        use_cuda_moe=False,
    )

    # Replace attention blocks with ModularMemoryAttention
    for block in model.blocks:
        block.attn = ModularMemoryAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            use_local_buffer=config_dict.get("use_local_buffer", False),
            local_window_size=config_dict.get("local_window_size", 16),
            use_adaptive_decay=config_dict.get("use_adaptive_decay", False),
            use_write_gate=config_dict.get("use_write_gate", False),
            use_erase_gate=config_dict.get("use_erase_gate", False),
            use_gated_read=config_dict.get("use_gated_read", False),
            use_multi_timescale=config_dict.get("use_multi_timescale", False),
            fusion_option=config_dict.get("fusion_option", 2),
            gamma_min=config_dict.get("gamma_min", 0.80),
            gamma_max=config_dict.get("gamma_max", 0.999),
        )

    return model
