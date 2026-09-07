import sys
sys.path.insert(0, ".")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from jarvis_model import Jarvis, LiquidStateFusion, SparseMoELayer

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TEST 4: RoPE STATE CONTINUITY (FULL VS CHUNKED INFERENCE)")
print("=" * 60)

# Instantiate a small Jarvis model for fast, exact testing
model = Jarvis(vocab_size=1000, d_model=64, n_layers=2, n_heads=4,
               num_experts=4, top_k=2, max_seq_len=1024).to(device)
model.eval()

tokens_512 = torch.randint(0, 1000, (1, 512), device=device)

# Method A: One continuous call of 512 tokens
model.reset_state()
with torch.no_grad():
    out_continuous, _ = model(tokens_512, persist_state=True)

# Method B: Split into two chunks of 256 tokens with persist_state=True
model.reset_state()
with torch.no_grad():
    chunk1 = tokens_512[:, :256]
    chunk2 = tokens_512[:, 256:]
    pos_before_1 = model._token_pos
    out_chunk1, _ = model(chunk1, persist_state=True)
    pos_after_1 = model._token_pos
    out_chunk2, _ = model(chunk2, persist_state=True)
    pos_after_2 = model._token_pos

out_split = torch.cat([out_chunk1, out_chunk2], dim=1)

print(f"Token pos progression: start={pos_before_1}, after_chunk1={pos_after_1}, after_chunk2={pos_after_2}")

# First 256 tokens comparison (should match closely since start is identical)
diff_first_256 = (out_continuous[:, :256] - out_split[:, :256]).abs()
print(f"Tokens 0-255: max_abs_diff = {diff_first_256.max().item():.2e}, mean_abs_diff = {diff_first_256.mean().item():.2e}")

# Second 256 tokens comparison
diff_second_256 = (out_continuous[:, 256:] - out_split[:, 256:]).abs()
max_diff_2 = diff_second_256.max().item()
mean_diff_2 = diff_second_256.mean().item()
rel_diff_2 = max_diff_2 / (out_continuous[:, 256:].abs().max().item() + 1e-8)
print(f"Tokens 256-511: max_abs_diff = {max_diff_2:.2e}, mean_abs_diff = {mean_diff_2:.2e}, rel_diff = {rel_diff_2:.2e}")
print(f"Are outputs numerically consistent (allclose at 1e-3)? {torch.allclose(out_continuous, out_split, atol=1e-3, rtol=1e-3)}")

print("\n" + "=" * 60)
print("TEST 5: RESET VS PERSIST")
print("=" * 60)

# Run chunk1 -> chunk2 with persist_state=True
model.reset_state()
with torch.no_grad():
    _ = model(chunk1, persist_state=True)
    out_persisted_chunk2, _ = model(chunk2, persist_state=True)

# Run chunk2 independently with fresh reset
model.reset_state()
with torch.no_grad():
    out_fresh_chunk2, _ = model(chunk2, persist_state=False)

diff_persist_vs_fresh = (out_persisted_chunk2 - out_fresh_chunk2).abs()
max_p_diff = diff_persist_vs_fresh.max().item()
mean_p_diff = diff_persist_vs_fresh.mean().item()
print(f"Max difference between persisted chunk2 and fresh chunk2: {max_p_diff:.4f}")
print(f"Mean difference between persisted chunk2 and fresh chunk2: {mean_p_diff:.4f}")
test5_pass = max_p_diff > 1e-3
print(f"Do outputs differ as expected? {test5_pass}")
print(f"TEST 5 VERDICT: {'PASS' if test5_pass else 'FAIL'}")

print("\n" + "=" * 60)
print("TEST 6: LIQUID STATE DYNAMICS")
print("=" * 60)

liquid = LiquidStateFusion(d_model=64).to(device)
liquid.eval()

# Check temporal dependency: perturb input at t=10 and see if t>10 changes, while t<10 remains identical
seq_len = 32
x_orig = torch.randn(1, seq_len, 64, device=device)
act_var = torch.tensor(0.5, device=device)

with torch.no_grad():
    h_orig, h_last_orig = liquid(x_orig, act_var)

x_pert = x_orig.clone()
x_pert[0, 10, :] += 5.0 # Perturb token 10

with torch.no_grad():
    h_pert, h_last_pert = liquid(x_pert, act_var)

diff_before = (h_orig[:, :10, :] - h_pert[:, :10, :]).abs().max().item()
diff_at_t = (h_orig[:, 10, :] - h_pert[:, 10, :]).abs().max().item()
diff_after = (h_orig[:, 11:, :] - h_pert[:, 11:, :]).abs().max().item()
decay_steps = [(h_orig[:, t, :] - h_pert[:, t, :]).abs().max().item() for t in range(10, 20)]

print(f"Diff before perturbation (t < 10): {diff_before:.2e} (must be 0.0)")
print(f"Diff at perturbation (t = 10): {diff_at_t:.4f}")
print(f"Diff after perturbation (t > 10): {diff_after:.4f}")
print("Decay profile over next 10 tokens:")
for i, d in enumerate(decay_steps):
    print(f"  t={10+i}: diff={d:.4f}")

# Inspect alpha values, variance response, and h norm
vars_to_test = [0.01, 0.1, 1.0, 5.0, 10.0]
print("\nDynamic alpha response to expert variance:")
for v in vars_to_test:
    v_tensor = torch.tensor(v, device=device)
    alpha_raw = torch.sigmoid(-liquid.var_scale * v_tensor)
    alpha = liquid.alpha_min + (liquid.alpha_max - liquid.alpha_min) * alpha_raw
    print(f"  var={v:5.2f} -> alpha={alpha.item():.4f}")

h_norm = h_orig.norm(dim=-1).mean().item()
print(f"Average h state norm: {h_norm:.4f}")
test6_pass = (diff_before == 0.0) and (diff_after > 0.0) and (decay_steps[1] < decay_steps[0])
print(f"TEST 6 VERDICT: {'PASS' if test6_pass else 'FAIL'}")

print("\n" + "=" * 60)
print("TEST 7: MOE ROUTING OVER MULTIPLE BATCHES")
print("=" * 60)

moe = SparseMoELayer(d_model=64, num_experts=4, top_k=2).to(device)
moe.train()

num_batches = 20
expert_counts = torch.zeros(4, device=device)
gate_probs_sum = torch.zeros(4, device=device)
balance_losses = []

for _ in range(num_batches):
    bx = torch.randn(2, 64, 64, device=device)
    out, l_balance, act_mean, act_var = moe(bx)
    balance_losses.append(l_balance.item())
    
    # Analyze routing in this forward
    N = bx.shape[0] * bx.shape[1]
    logits = moe.router(bx.view(N, 64)) + torch.randn(N, 4, device=device) * moe.noise_std
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_idx = probs.topk(2, dim=-1)
    
    for e in range(4):
        expert_counts[e] += (topk_idx == e).sum().item()
    gate_probs_sum += probs.sum(dim=0)

total_routed = expert_counts.sum().item()
print(f"Total token-expert assignments: {int(total_routed)} across {num_batches} batches")
print("Expert Distribution:")
for e in range(4):
    pct = (expert_counts[e].item() / total_routed) * 100
    avg_p = (gate_probs_sum[e].item() / (num_batches * N))
    print(f"  Expert {e}: count={int(expert_counts[e].item()):5d} ({pct:5.2f}%), avg_prob={avg_p:.4f}")

any_zero = (expert_counts == 0).any().item()
mean_l_bal = sum(balance_losses) / len(balance_losses)
print(f"Any expert receives zero traffic: {any_zero}")
print(f"Mean load-balance loss: {mean_l_bal:.6f}")
test7_pass = (not any_zero) and (expert_counts.min() / expert_counts.max() > 0.5)
print(f"TEST 7 VERDICT: {'PASS' if test7_pass else 'FAIL'}")
