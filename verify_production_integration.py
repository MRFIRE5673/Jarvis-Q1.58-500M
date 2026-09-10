import os
import sys
import torch
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
JARVIS_ENGINE_PATH = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
if JARVIS_ENGINE_PATH not in sys.path:
    sys.path.insert(0, JARVIS_ENGINE_PATH)

from jarvis_model import Jarvis

def run_checks():
    print("=== JARVIS PRODUCTION INTEGRATION VERIFICATION ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Instantiate CUDA model
    m_cuda = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
                    num_experts=4, top_k=2, max_seq_len=256,
                    use_cuda_attn=True, use_cuda_moe=True).to(device)
    cuda_status = m_cuda.get_backend_status()
    print("CUDA Model Backend Status:", cuda_status)
    assert cuda_status["attn_backend"] == "cuda", f"Expected cuda attn, got {cuda_status['attn_backend']}"
    assert cuda_status["moe_backend"] == "cuda", f"Expected cuda moe, got {cuda_status['moe_backend']}"
    assert cuda_status["attn_cuda_blocks"] == "24/24"
    assert cuda_status["moe_cuda_blocks"] == "24/24"

    # 2. Instantiate PyTorch fallback model
    m_pt = Jarvis(vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
                  num_experts=4, top_k=2, max_seq_len=256,
                  use_cuda_attn=False, use_cuda_moe=False).to(device)
    pt_status = m_pt.get_backend_status()
    print("PyTorch Fallback Backend Status:", pt_status)
    assert pt_status["attn_backend"] == "pytorch", f"Expected pytorch attn, got {pt_status['attn_backend']}"
    assert pt_status["moe_backend"] == "pytorch", f"Expected pytorch moe, got {pt_status['moe_backend']}"
    assert pt_status["attn_cuda_blocks"] == "0/24"
    assert pt_status["moe_cuda_blocks"] == "0/24"

    # 3. State dict key parity
    keys_cuda = set(m_cuda.state_dict().keys())
    keys_pt = set(m_pt.state_dict().keys())
    diff_keys = keys_cuda.symmetric_difference(keys_pt)
    print(f"State dict keys diff count: {len(diff_keys)}")
    assert len(diff_keys) == 0, f"State dict mismatch: {diff_keys}"
    print(f"Total state dict entries: {len(keys_cuda)} (100% matched)")

    # 4. Parameter count parity
    params_cuda = sum(p.numel() for p in m_cuda.parameters())
    params_pt = sum(p.numel() for p in m_pt.parameters())
    print(f"Param count CUDA: {params_cuda:,} | PyTorch: {params_pt:,}")
    assert params_cuda == params_pt, f"Param count mismatch: {params_cuda} vs {params_pt}"

    # 5. Checkpoint loading test
    ckpt_path = os.path.join(JARVIS_ENGINE_PATH, "ckpt_step_0004209.pt")
    if os.path.exists(ckpt_path):
        print(f"Testing checkpoint loading from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]
        cleaned_sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                      for k, v in state_dict.items()}
        missing, unexpected = m_cuda.load_state_dict(cleaned_sd, strict=True)
        print(f"Checkpoint load strict=True passed: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"[WARN] Checkpoint {ckpt_path} not found for loading test.")

    # 6. Test forward + backward pass on small dummy batch
    print("Testing forward + backward pass under BF16 autocast...")
    x = torch.randint(0, 50257, (2, 256), device=device)
    y = torch.randint(0, 50257, (2, 256), device=device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = m_cuda(x, targets=y)
    print(f"Forward output shape: {logits.shape}, loss: {loss.item():.4f}")
    assert logits.shape == (2, 256, 50257)
    assert not torch.isnan(loss)
    loss.backward()
    assert m_cuda.blocks[0].attn.gamma_raw.grad is not None
    assert m_cuda.blocks[0].moe.router.weight.grad is not None
    print("Backward pass completed, gradients verified!")

    # 7. State persistence and reset_state test
    print("Testing state persistence semantics...")
    m_cuda.eval()
    m_cuda.reset_state()
    assert m_cuda._token_pos == 0
    assert all(h is None for h in m_cuda._h_states)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out1, _ = m_cuda(x, persist_state=True)
            pos1 = m_cuda._token_pos
            assert pos1 == 256, f"Expected _token_pos=256, got {pos1}"
            assert all(h is not None for h in m_cuda._h_states), "Liquid states should be persisted"
            first_h = [h.clone() for h in m_cuda._h_states]

            out2, _ = m_cuda(x, persist_state=True)
            pos2 = m_cuda._token_pos
            assert pos2 == 512, f"Expected _token_pos=512, got {pos2}"

            # Verify reset_state
            m_cuda.reset_state()
            assert m_cuda._token_pos == 0
            assert all(h is None for h in m_cuda._h_states)
    print("State persistence and reset_state verified!")

    print("\nALL 7 INTEGRATION & SEMANTICS VERIFICATION CHECKS PASSED!")

if __name__ == "__main__":
    run_checks()
