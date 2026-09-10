# experiments/architecture_matrix/verify_baseline_config.py
import os
import sys
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import Jarvis

ckpt_path = os.path.join(WORKSPACE_ROOT, "experiments", "extended_train", "ckpt_step_0004284_best.pt")
print(f"Inspecting Checkpoint: {ckpt_path}")
print(f"Exists: {os.path.exists(ckpt_path)}")
size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
print(f"File Size: {size_mb:.2f} MB")

ckpt = torch.load(ckpt_path, map_location="cpu")
print("Checkpoint Keys:", list(ckpt.keys()))
print(f"Saved Step: {ckpt.get('step')}")
if "config" in ckpt:
    print("Saved Config:", ckpt["config"])
if "metrics" in ckpt:
    print("Saved Metrics:", ckpt["metrics"])

sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
clean_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
total_params_ckpt = sum(v.numel() for v in clean_sd.values())
print(f"Total Tensors in Checkpoint: {len(clean_sd)}")
print(f"Total Parameters in Checkpoint: {total_params_ckpt:,}")

# Check baseline gamma in checkpoint
gamma_keys = [k for k in clean_sd.keys() if "gamma" in k]
print(f"Gamma keys found: {gamma_keys}")
if gamma_keys:
    first_gamma = clean_sd[gamma_keys[0]]
    print(f"First gamma tensor ({gamma_keys[0]}): shape={first_gamma.shape}, dtype={first_gamma.dtype}")
    print(f"Raw gamma values:\n{first_gamma}")
    print(f"Sigmoid(gamma) values:\n{torch.sigmoid(first_gamma.float())}")

# Instantiate model
model = Jarvis(
    vocab_size=50257,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    num_experts=4,
    top_k=2,
    max_seq_len=2048,
    use_cuda_attn=False,
    use_cuda_moe=False,
)

model_sd = model.state_dict()
matching = {k: v for k, v in clean_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
missing = [k for k in model_sd.keys() if k not in matching]
unexpected = [k for k in clean_sd.keys() if k not in model_sd]
print(f"Model state dict match: {len(matching)} / {len(model_sd)} keys")
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

total_params = sum(p.numel() for p in model.parameters())
# Active parameters per token accounting:
# Embedding: 50257 * 1024 = 51,463,168
# Per layer:
#   Attn: Q, K, V, Out = 4 * (1024 * 1024) = 4,194,304
#   Attn norm: 1024
#   MoE Router: 1024 * 4 = 4096
#   MoE Active Experts (Top-2 of 4): 2 * (2 * 1024 * 2048) = 2 * 4,194,304 = 8,388,608
#   MoE norm: 1024
# Final norm: 1024
# LM Head: 1024 * 50257 = 51,463,168
active_per_layer = 4194304 + 1024 + 4096 + 8388608 + 1024
active_params = 51463168 + (24 * active_per_layer) + 1024 + 51463168
print(f"Exact Total Parameters : {total_params:,}")
print(f"Exact Active Parameters: {active_params:,} ({active_params/total_params*100:.2f}%)")
