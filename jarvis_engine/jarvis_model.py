"""
Jarvis Neuromorphic Transformer - fixed to match research paper equations.

Bugs fixed vs. original:
  1. AssociativeLinearAttention: replaced chunked intra-chunk softmax attention
     with the correct per-token O(N) linear recurrence (Eq. 1-2, Algo 1 L4-6).
  2. SparseMoELayer: added Gaussian noise injection before Top-K (Algo 1 L11).
  3. SparseMoELayer: fixed load-balance loss to use f_i * P_i (Eq. 6) not P_i^2.
  4. Reflective Penalty (Eq. 7) implemented and wired into the loss.
  5. LiquidStateFusion: changed to EMA H_t = alpha*H_{t-1} + (1-alpha)*M_t
     with alpha computed dynamically from expert output variance (Algo 1 L15).
  6. LiquidStateFusion: membrane state now persists across the full sequence.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from utils.ternary_ops import TernaryLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE, Su et al. 2021).
    Encodes relative position directly into Q and K in AssociativeLinearAttention,
    eliminating the need for fixed-size absolute pos_emb tables and enabling
    unbounded / infinite sequence lengths.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self._build_cache(max_seq_len, device=inv_freq.device)

    def _build_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = max(seq_len, 256)
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq.to(device))  # (max_seq_len_cached, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)           # (max_seq_len_cached, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        # q, k: (B, H, T, D)
        T = q.shape[2]
        end_pos = start_pos + T
        if end_pos > self.max_seq_len_cached or self.cos_cached.device != q.device:
            new_len = max(end_pos, self.max_seq_len_cached * 2)
            self._build_cache(new_len, device=q.device)

        cos = self.cos_cached[start_pos:end_pos].to(dtype=q.dtype)[None, None, :, :]  # (1, 1, T, D)
        sin = self.sin_cached[start_pos:end_pos].to(dtype=q.dtype)[None, None, :, :]  # (1, 1, T, D)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Vectorized Chunked O(N) Associative Linear Attention (Eq. 1-2, Algo 1 L4-6)
# with RoPE (Rotary Position Embeddings)
#
# Paper recurrence (per token):
#   S_t = γ * S_{t-1} + v_t ⊗ k_t^T
#   z_t = S_t @ q_t
#
# Optimization: instead of a T-step Python loop (256 serial kernel launches),
# we process CHUNK_SIZE=64-token chunks with fully vectorized matmuls (4 iters).
# Each chunk decomposes z_i (local index i within chunk) into:
#   Cross-chunk : γ^{i+1} * (S_prev @ q_i)               from carried state
#   Intra-chunk : Σ_{j≤i} γ^{i-j} * (k_j · q_i) * v_j   decayed attn within chunk
# State update  : S_new = γ^c * S_prev + Σ_j γ^{c-1-j} * v_j ⊗ k_j^T
# ---------------------------------------------------------------------------
class AssociativeLinearAttention(nn.Module):
    CHUNK_SIZE = 64

    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)
        # γ: per-head learnable decay, init sigmoid(2.94) ≈ 0.95
        self.gamma_raw = nn.Parameter(torch.full((n_heads,), 2.94))
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Pre-cache chunk buffers — allocated ONCE at init, reused every forward.
        # Eliminates torch.arange + outer-product diff + clamp + comparison per step.
        cs = self.CHUNK_SIZE
        i = torch.arange(cs, dtype=torch.float32)           # (cs,)
        diff = i.unsqueeze(1) - i.unsqueeze(0)              # (cs, cs)
        self.register_buffer('_diff_clamp', diff.clamp(min=0))          # (cs, cs)
        self.register_buffer('_causal',     (diff >= 0).float())         # (cs, cs)
        self.register_buffer('_i_idx',      i)                           # (cs,)
        self.register_buffer('_i_idx_p1',   i + 1)                       # (cs,) i+1
        self.register_buffer('_c_m1_m_i',   (cs - 1) - i)               # (cs,) c-1-i

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # ELU+1 keeps feature values positive (Katharopoulos 2020)
        q = (F.elu(q) + 1.0) / math.sqrt(D)
        k = F.elu(k) + 1.0

        # RoPE (Su et al. 2021): inject relative position information
        q, k = self.rotary(q, k, start_pos=start_pos)

        # Compute log(gamma) once — exp(log_g*diff) is faster than gamma**diff.
        log_g = torch.log(torch.sigmoid(self.gamma_raw))  # (H,) float32

        cs = self.CHUNK_SIZE
        # Use pre-cached float32 buffers directly — no .to() cast inside forward,
        # which would cause torch.compile to retrace on every step (graph break).
        # autocast handles mixed float32/bfloat16 ops correctly.
        dc    = self._diff_clamp   # (cs, cs) float32
        ca    = self._causal       # (cs, cs) float32
        ip1   = self._i_idx_p1     # (cs,)    float32
        cm1mi = self._c_m1_m_i     # (cs,)    float32

        # Per-chunk decay quantities — computed ONCE, reused for all 4 chunks
        decay_mat   = torch.exp(log_g.view(H,1,1) * dc) * ca   # (H,cs,cs) float32
        gamma_cross = torch.exp(log_g.view(H,1) * ip1)          # (H,cs)
        gw          = torch.exp(log_g.view(H,1) * cm1mi)        # (H,cs)
        gamma_c     = torch.exp(log_g * cs).view(1,H,1,1)       # (1,H,1,1)

        state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        outputs = []

        for start in range(0, T, cs):
            end   = min(start + cs, T)
            c     = end - start          # actual chunk length (may be < cs for last chunk)
            q_c   = q[:, :, start:end, :]   # (B, H, c, D)
            k_c   = k[:, :, start:end, :]
            v_c   = v[:, :, start:end, :]

            if c == cs:
                # Full chunk — use pre-cached decay tensors (fast path)
                dm       = decay_mat             # (H, cs, cs)
                gc_cross = gamma_cross           # (H, cs)
                gw_c     = gw                    # (H, cs)
                gc_state = gamma_c               # (1, H, 1, 1)
            else:
                # Partial chunk (last chunk when T % cs != 0) — compute on-the-fly
                i_c      = self._i_idx[:c]                                    # (c,)
                diff_c   = (i_c.unsqueeze(1) - i_c.unsqueeze(0)).clamp(min=0)  # (c,c)
                causal_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0) >= 0).float()  # (c,c)
                dm       = torch.exp(log_g.view(H,1,1) * diff_c) * causal_c   # (H,c,c)
                gc_cross = torch.exp(log_g.view(H,1) * (i_c + 1))             # (H,c)
                gw_c     = torch.exp(log_g.view(H,1) * (c - 1 - i_c))         # (H,c)
                gc_state = torch.exp(log_g * c).view(1,H,1,1)

            # Intra-chunk decayed linear attention
            raw       = torch.einsum('bhid,bhjd->bhij', q_c, k_c)        # (B,H,c,c)
            scores    = raw * dm.unsqueeze(0)                             # (B,H,c,c)
            intra_out = torch.einsum('bhij,bhjd->bhid', scores, v_c)     # (B,H,c,D)

            # Cross-chunk: exp(log_g*(i+1)) * S_prev @ q_i
            raw_cross = torch.einsum('bhde,bhie->bhid', state, q_c)      # (B,H,c,D)
            cross_out = raw_cross * gc_cross.unsqueeze(0).unsqueeze(-1)

            outputs.append(intra_out + cross_out)

            # State update: S_new = gamma^c * S + Σ_j exp(log_g*(c-1-j)) * v_j⊗k_j^T
            v_w       = v_c * gw_c.unsqueeze(0).unsqueeze(-1)            # (B,H,c,D)
            chunk_upd = torch.einsum('bhid,bhie->bhde', v_w, k_c)        # (B,H,D,D)
            state     = gc_state * state + chunk_upd

        out = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Bug 2 FIX: Gaussian noise added to router logits (Algo 1 L11)
# Bug 3 FIX: Load-balance loss uses f_i * P_i (Eq. 6), not P_i^2
# ---------------------------------------------------------------------------
class SparseMoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=2, hidden_mult=2,
                 noise_std=0.1, balance_alpha=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std        # σ for Gaussian routing noise
        self.balance_alpha = balance_alpha  # α in Eq. 6
        self.router = nn.Linear(d_model, num_experts, bias=False)
        hidden = d_model * hidden_mult
        # Paper Section 3.2: ALL Feed-Forward pathways are ternary
        self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        # Router logits + Gaussian noise (Algo 1 L11: G(A_spike) + N(0, σ²))
        logits = self.router(x_flat)  # (N, E)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)                        # (N, E)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)   # (N, top_k)
        topk_gates = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Vectorized MoE: flatten top-k dimension → one pass per expert (not k×E).
        # flat_idx/flat_gate: (N*top_k,)  flat_x: (N*top_k, C)
        flat_idx  = topk_idx.reshape(-1)                         # (N*K,)
        flat_gate = topk_gates.reshape(-1)                       # (N*K,)
        flat_x    = x_flat.repeat_interleave(self.top_k, dim=0)  # (N*K, C)
        flat_out  = torch.zeros_like(flat_x)                     # (N*K, C)

        for e in range(self.num_experts):
            mask = (flat_idx == e)
            if mask.any():
                xe = flat_x[mask]
                ye = self.w2[e](F.gelu(self.w1[e](xe)))
                flat_out[mask] = flat_gate[mask].unsqueeze(-1) * ye

        # Sum both top-k contributions per original token
        out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)

        # Load-balance loss: α * N_experts * Σ f_i * P_i  (Eq. 6)
        top1_idx = topk_idx[:, 0]
        f = F.one_hot(top1_idx, num_classes=self.num_experts).float().mean(dim=0)
        P = probs.mean(dim=0)
        l_balance = self.balance_alpha * self.num_experts * (f * P).sum()

        act_mean = out.mean()
        act_var  = out.var()
        return out, l_balance, act_mean, act_var


# ---------------------------------------------------------------------------
# Bug 5 & 6 FIX: Liquid State Fusion with correct EMA + dynamic α + state persistence
#
# Paper (Algo 1 L15):  H_t = α * H_{t-1} + (1 - α) * M_t
# α is dynamically derived from the variance of the expert output subset.
# The membrane potential H must persist across ALL tokens in the sequence.
# ---------------------------------------------------------------------------
class LiquidStateFusion(nn.Module):
    """
    Leaky Integrate-and-Fire membrane state, token-sequential EMA (Algo 1 L15).

    Paper (per-token):  H_t = α · H_{t-1} + (1 − α) · M_t

    α is computed dynamically from expert output variance:
      high variance (strong signal) → lower α (fast adaptation / less leakage)
      low  variance (weak  signal ) → higher α (more memory / slow decay)

    The EMA scan runs along the T dimension so that H_t truly depends on all
    previous tokens H_0 … H_{t-1}, matching the paper's sequential LIF update.
    Returns both h_out (B,T,C) for the residual connection and h_last (B,C)
    which can be persisted across sequences for infinite-context inference.
    """
    def __init__(self, d_model, alpha_min=0.1, alpha_max=0.99):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.var_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, act_var, h_prev=None):
        """
        Args:
            x:       MoE output          (B, T, C)
            act_var: scalar MoE variance  (from SparseMoELayer)
            h_prev:  last-token membrane  (B, C) or None — carry from previous sequence
        Returns:
            h_out:   per-token states     (B, T, C)
            h_last:  last-token state     (B, C)  — persist for next sequence if desired
        """
        B, T, C = x.shape
        # Initialise membrane to zero if no prior state (start of sequence)
        if h_prev is None:
            h_prev = x.new_zeros(B, C)

        # Dynamic α from variance (Eq. 8 / Algo 1 L15)
        alpha_raw = torch.sigmoid(-self.var_scale * act_var)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw

        # Token-sequential EMA scan: H_t = α·H_{t-1} + (1-α)·M_t  (Algo 1 L15)
        # A Python loop over T is fast here (T=256, each iter is cheap elementwise op).
        h_states = []
        for t in range(T):
            h_prev = alpha * h_prev + (1.0 - alpha) * x[:, t, :]   # (B, C)
            h_states.append(h_prev)

        h_out  = torch.stack(h_states, dim=1)   # (B, T, C)
        h_last = h_prev                          # (B, C) — last token membrane state
        return h_out, h_last


# ---------------------------------------------------------------------------
# Bug 4 FIX: Reflective Penalty (Eq. 7)
#
#   L_reflect = λ * [(µ_t - µ_batch)² + max(0, σ² - τ_max)]
#
# µ_t  = exponentially tracked running mean (updated each step)
# µ_batch = current batch activation mean
# σ²   = current batch activation variance
# τ_max = maximum allowable variance threshold
# ---------------------------------------------------------------------------
class ReflectivePenalty(nn.Module):
    """
    Tracks a running mean µ_t and penalises deviations from it and
    excess variance in activations (Eq. 7).
    """
    def __init__(self, lam=0.01, tau_max=1.0, ema_decay=0.99):
        super().__init__()
        self.lam = lam
        self.tau_max = tau_max
        self.ema_decay = ema_decay
        # Running mean µ_t — not a parameter, just a buffer
        self.register_buffer('mu_t', torch.tensor(0.0))

    def forward(self, act_mean, act_var):
        mu_batch = act_mean
        # L_reflect = λ * [(µ_t - µ_batch)² + max(0, σ² - τ_max)]
        mean_penalty = (self.mu_t.detach() - mu_batch) ** 2
        var_penalty = F.relu(act_var - self.tau_max)
        l_reflect = self.lam * (mean_penalty + var_penalty)

        # Update running mean (EMA)
        if self.training:
            self.mu_t = self.ema_decay * self.mu_t + (1.0 - self.ema_decay) * mu_batch.detach()

        return l_reflect


class JarvisBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_experts=4, top_k=2, max_seq_len=2048):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = AssociativeLinearAttention(d_model, n_heads, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.moe = SparseMoELayer(d_model, num_experts=num_experts, top_k=top_k)
        self.liquid = LiquidStateFusion(d_model)
        self.reflect = ReflectivePenalty()

    def forward(self, x, h_prev=None, start_pos: int = 0):
        """
        Args:
            x:         input hidden states  (B, T, C)
            h_prev:    last-token membrane  (B, C) or None  [carry from previous sequence]
            start_pos: token position offset for RoPE
        Returns:
            x:       updated hidden states  (B, T, C)
            h_last:  last-token membrane    (B, C)        [pass to next sequence]
            l_balance, l_reflect: aux losses
        """
        # Step 1: Associative Attention (Eq. 1-2) with RoPE
        x = x + self.attn(self.norm1(x), start_pos=start_pos)

        # Step 2: Sparse MoE (Eq. 6 + Algo 1 L11-13)
        moe_out, l_balance, act_mean, act_var = self.moe(self.norm2(x))

        # Step 3: Liquid State — token-sequential EMA scan (Algo 1 L15 / Eq. 8)
        # h_prev: (B,C) last membrane from prior sequence; None resets to zero
        h_out, h_last = self.liquid(moe_out, act_var, h_prev)
        x = x + h_out                      # residual add full (B,T,C) output

        # Step 4: Reflective Penalty (Eq. 7)
        l_reflect = self.reflect(act_mean, act_var)

        return x, h_last, l_balance, l_reflect


class Jarvis(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
                 num_experts=4, top_k=2, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Note: pos_emb removed — replaced by RoPE in AssociativeLinearAttention
        self.blocks = nn.ModuleList([
            JarvisBlock(d_model, n_heads, num_experts, top_k, max_seq_len=max_seq_len) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Persistent cross-sequence liquid membrane states & RoPE position offset
        # None = not yet initialised. Populated by forward(), cleared by reset_state().
        # shape per entry: (B, C) — the last-token membrane of each block.
        self._h_states: list = [None] * n_layers
        self._token_pos: int = 0

    def reset_state(self):
        """Clear persisted liquid membrane states and position (call between unrelated sequences)."""
        self._h_states = [None] * len(self.blocks)
        self._token_pos = 0

    def forward(self, idx, targets=None, persist_state=False):
        """
        Args:
            idx:           token indices (B, T)
            targets:       optional targets for loss (B, T)
            persist_state: if True, liquid membrane states and RoPE position offsets
                           are carried across forward calls (infinite-context inference mode).
                           If False (training default), states reset each call.
        """
        B, T = idx.shape
        x = self.tok_emb(idx)

        start_pos = self._token_pos if persist_state else 0
        if persist_state:
            self._token_pos += T

        l_bal_total = torch.tensor(0.0, device=idx.device)
        l_ref_total = torch.tensor(0.0, device=idx.device)

        # Liquid state persistence (paper Algo 1: H_t survives across sequence boundaries).
        # Training: persist_state=False  — random batches are independent sequences.
        # Inference: persist_state=True  — carry H_last across forward calls for infinite context.
        if persist_state:
            h_prevs = self._h_states   # carry from previous call
        else:
            h_prevs = [None] * len(self.blocks)   # reset per call (training)

        new_h_states = []
        for i, block in enumerate(self.blocks):
            # h_prev: (B, C) or None.  If batch size changed, reset silently.
            h_prev = h_prevs[i]
            if h_prev is not None and h_prev.shape[0] != B:
                h_prev = None   # batch size mismatch — reset gracefully

            if self.training:
                x, h_last, l_bal, l_ref = grad_ckpt(block, x, h_prev, start_pos, use_reentrant=False)
            else:
                x, h_last, l_bal, l_ref = block(x, h_prev, start_pos=start_pos)

            new_h_states.append(h_last.detach())   # (B, C), detached
            l_bal_total = l_bal_total + l_bal
            l_ref_total = l_ref_total + l_ref

        # Store updated states for next call (used when persist_state=True)
        self._h_states = new_h_states

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce + l_bal_total + l_ref_total

        return logits, loss

    def param_count(self):
        n = sum(p.numel() for p in self.parameters())
        return n, f"{n/1e6:.1f}M parameters"