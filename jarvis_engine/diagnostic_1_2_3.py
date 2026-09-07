import sys
sys.path.insert(0, ".")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ternary_ops import TernaryQuantizeSTE, TernaryLinear
from jarvis_model import AssociativeLinearAttention, RotaryEmbedding, rotate_half

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("=" * 60)
print("TEST 1: TERNARY QUANTIZATION")
print("=" * 60)

# 1. Distribution of FP32 weights before quantization
layer = TernaryLinear(1024, 1024).to(device)
w = layer.weight.data
w_mean = w.mean().item()
w_std = w.std().item()
w_min = w.min().item()
w_max = w.max().item()
alpha = w.abs().mean().clamp(min=1e-8).item()

print(f"FP32 Weights: mean={w_mean:.6f}, std={w_std:.6f}, min={w_min:.6f}, max={w_max:.6f}")
print(f"AbsMean alpha: {alpha:.6f}")

# Quantize
w_norm = w / alpha
w_clamped = torch.clamp(w_norm, -1.0, 1.0)
w_q = torch.round(w_clamped)

num_neg = (w_q == -1.0).sum().item()
num_zero = (w_q == 0.0).sum().item()
num_pos = (w_q == 1.0).sum().item()
total = w_q.numel()
pct_zero = (num_zero / total) * 100

print(f"Counts: -1: {num_neg} ({num_neg/total*100:.2f}%), 0: {num_zero} ({pct_zero:.2f}%), +1: {num_pos} ({num_pos/total*100:.2f}%)")
print(f"Any layer effectively all-zero: {num_zero == total}")

# Forward + Backward
x = torch.randn(2, 64, 1024, device=device, requires_grad=True)
out = layer(x)
loss = out.sum()
loss.backward()

grad = layer.weight.grad
nonzero_grad = (grad != 0).sum().item()
grad_norm = grad.norm().item()
grad_mean = grad.mean().item()
grad_std = grad.std().item()
print(f"Forward output shape: {out.shape}, mean={out.mean().item():.4f}, std={out.std().item():.4f}")
print(f"Gradient through FP32 weights: non-zero count={nonzero_grad}/{grad.numel()} ({nonzero_grad/grad.numel()*100:.2f}%)")
print(f"Grad norm: {grad_norm:.6f}, mean: {grad_mean:.6e}, std: {grad_std:.6e}")
test1_pass = (num_zero != total) and (nonzero_grad > 0)
print(f"TEST 1 VERDICT: {'PASS' if test1_pass else 'FAIL'}")

print("\n" + "=" * 60)
print("TEST 2: ASSOCIATIVE ATTENTION MATHEMATICAL EQUIVALENCE (NO RoPE)")
print("=" * 60)

def literal_associative_attention(q, k, v, gamma):
    # q, k, v: (B, H, T, D)
    # gamma: (H,)
    B, H, T, D = q.shape
    Z = torch.zeros_like(q)
    log_g = torch.log(torch.sigmoid(gamma)) # (H,)
    g = torch.sigmoid(gamma) # (H,)
    
    # Per-head loop
    for b in range(B):
        for h in range(H):
            gh = g[h].item()
            S = torch.zeros(D, D, device=q.device, dtype=q.dtype)
            for t in range(T):
                vt = v[b, h, t, :] # (D,)
                kt = k[b, h, t, :] # (D,)
                qt = q[b, h, t, :] # (D,)
                # S_t = gamma * S_{t-1} + v_t outer k_t^T
                S = gh * S + torch.outer(vt, kt)
                # Z_t = S_t @ q_t
                Z[b, h, t, :] = S @ qt
    return Z

test2_lengths = [20, 64, 65, 127, 128, 129, 256]
test2_results = []
B, H, D = 1, 2, 8
d_model = H * D

for T in test2_lengths:
    attn = AssociativeLinearAttention(d_model=d_model, n_heads=H).to(device)
    # Detach rotary for Test 2 (testing raw associative recurrence as defined in Eq. 1-2)
    with torch.no_grad():
        x = torch.randn(B, T, d_model, device=device)
        # Project using internal projections
        q_raw = attn.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k_raw = attn.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v_raw = attn.v_proj(x).view(B, T, H, D).transpose(1, 2)
        
        q = (F.elu(q_raw) + 1.0) / math.sqrt(D)
        k = F.elu(k_raw) + 1.0
        v = v_raw
        
        # Reference literal
        z_literal = literal_associative_attention(q, k, v, attn.gamma_raw)
        
        # Now run chunked loop exactly as in AssociativeLinearAttention.forward
        log_g = torch.log(torch.sigmoid(attn.gamma_raw))
        cs = attn.CHUNK_SIZE
        dc = attn._diff_clamp
        ca = attn._causal
        ip1 = attn._i_idx_p1
        cm1mi = attn._c_m1_m_i
        decay_mat = torch.exp(log_g.view(H,1,1) * dc) * ca
        gamma_cross = torch.exp(log_g.view(H,1) * ip1)
        gw = torch.exp(log_g.view(H,1) * cm1mi)
        gamma_c = torch.exp(log_g * cs).view(1,H,1,1)
        
        state = torch.zeros(B, H, D, D, device=device, dtype=x.dtype)
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
                i_c = attn._i_idx[:c]
                diff_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0)).clamp(min=0)
                causal_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0) >= 0).float()
                dm = torch.exp(log_g.view(H,1,1) * diff_c) * causal_c
                gc_cross = torch.exp(log_g.view(H,1) * (i_c + 1))
                gw_c = torch.exp(log_g.view(H,1) * (c - 1 - i_c))
                gc_state = torch.exp(log_g * c).view(1,H,1,1)
            raw = torch.einsum('bhid,bhjd->bhij', q_c, k_c)
            scores = raw * dm.unsqueeze(0)
            intra_out = torch.einsum('bhij,bhjd->bhid', scores, v_c)
            raw_cross = torch.einsum('bhde,bhie->bhid', state, q_c)
            cross_out = raw_cross * gc_cross.unsqueeze(0).unsqueeze(-1)
            outputs.append(intra_out + cross_out)
            v_w = v_c * gw_c.unsqueeze(0).unsqueeze(-1)
            chunk_upd = torch.einsum('bhid,bhie->bhde', v_w, k_c)
            state = gc_state * state + chunk_upd
            
        z_chunked = torch.cat(outputs, dim=2)
        
        max_abs_err = (z_literal - z_chunked).abs().max().item()
        mean_abs_err = (z_literal - z_chunked).abs().mean().item()
        rel_err = (max_abs_err / (z_literal.abs().max().item() + 1e-8))
        is_close = torch.allclose(z_literal, z_chunked, atol=1e-4, rtol=1e-4)
        print(f"T={T:3d} | max_abs_err={max_abs_err:.2e} | mean_abs_err={mean_abs_err:.2e} | rel_err={rel_err:.2e} | allclose={is_close}")
        test2_results.append(is_close)

print(f"TEST 2 VERDICT: {'PASS' if all(test2_results) else 'FAIL'}")

print("\n" + "=" * 60)
print("TEST 3: RoPE + RECURRENCE EQUIVALENCE")
print("=" * 60)

test3_results = []
for T in test2_lengths:
    attn = AssociativeLinearAttention(d_model=d_model, n_heads=H, max_seq_len=max(T, 256)).to(device)
    with torch.no_grad():
        x = torch.randn(B, T, d_model, device=device)
        q_raw = attn.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k_raw = attn.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v_raw = attn.v_proj(x).view(B, T, H, D).transpose(1, 2)
        
        q = (F.elu(q_raw) + 1.0) / math.sqrt(D)
        k = F.elu(k_raw) + 1.0
        v = v_raw
        
        # Apply RoPE directly as the model does
        q_rot, k_rot = attn.rotary(q, k, start_pos=0)
        
        # Literal with rotated Q & K
        z_literal_rope = literal_associative_attention(q_rot, k_rot, v, attn.gamma_raw)
        
        # Chunked forward as in AssociativeLinearAttention.forward
        # (which takes q_rot and k_rot)
        log_g = torch.log(torch.sigmoid(attn.gamma_raw))
        cs = attn.CHUNK_SIZE
        decay_mat = torch.exp(log_g.view(H,1,1) * attn._diff_clamp) * attn._causal
        gamma_cross = torch.exp(log_g.view(H,1) * attn._i_idx_p1)
        gw = torch.exp(log_g.view(H,1) * attn._c_m1_m_i)
        gamma_c = torch.exp(log_g * cs).view(1,H,1,1)
        
        state = torch.zeros(B, H, D, D, device=device, dtype=x.dtype)
        outputs = []
        for start in range(0, T, cs):
            end = min(start + cs, T)
            c = end - start
            q_c = q_rot[:, :, start:end, :]
            k_c = k_rot[:, :, start:end, :]
            v_c = v[:, :, start:end, :]
            if c == cs:
                dm = decay_mat
                gc_cross = gamma_cross
                gw_c = gw
                gc_state = gamma_c
            else:
                i_c = attn._i_idx[:c]
                diff_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0)).clamp(min=0)
                causal_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0) >= 0).float()
                dm = torch.exp(log_g.view(H,1,1) * diff_c) * causal_c
                gc_cross = torch.exp(log_g.view(H,1) * (i_c + 1))
                gw_c = torch.exp(log_g.view(H,1) * (c - 1 - i_c))
                gc_state = torch.exp(log_g * c).view(1,H,1,1)
            raw = torch.einsum('bhid,bhjd->bhij', q_c, k_c)
            scores = raw * dm.unsqueeze(0)
            intra_out = torch.einsum('bhij,bhjd->bhid', scores, v_c)
            raw_cross = torch.einsum('bhde,bhie->bhid', state, q_c)
            cross_out = raw_cross * gc_cross.unsqueeze(0).unsqueeze(-1)
            outputs.append(intra_out + cross_out)
            v_w = v_c * gw_c.unsqueeze(0).unsqueeze(-1)
            chunk_upd = torch.einsum('bhid,bhie->bhde', v_w, k_c)
            state = gc_state * state + chunk_upd
            
        z_chunked_rope = torch.cat(outputs, dim=2)
        
        max_abs_err = (z_literal_rope - z_chunked_rope).abs().max().item()
        mean_abs_err = (z_literal_rope - z_chunked_rope).abs().mean().item()
        rel_err = (max_abs_err / (z_literal_rope.abs().max().item() + 1e-8))
        is_close = torch.allclose(z_literal_rope, z_chunked_rope, atol=1e-4, rtol=1e-4)
        print(f"T={T:3d} | max_abs_err={max_abs_err:.2e} | mean_abs_err={mean_abs_err:.2e} | rel_err={rel_err:.2e} | allclose={is_close}")
        test3_results.append(is_close)

print(f"TEST 3 VERDICT: {'PASS' if all(test3_results) else 'FAIL'}")
