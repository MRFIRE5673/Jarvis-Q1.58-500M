# Jarvis Architectural Decisions & Evolution Log

This document records key architectural decisions, empirical rationale, and the explicit distinction between the **Paper-Faithful Baseline** and subsequent **Research Variants**.

---

## 1. Architectural Taxonomy & Terminology

- **Paper-Faithful Baseline:** The original, exact implementation of the Jarvis Q1.58-500M architecture described in the foundational specifications (`jarvis_engine/jarvis_model.py`, initialized from `ckpt_step_0004209.pt`). It uses chunked associative linear attention with scalar decay, Liquid State Fusion (LSF), ternary quantized weights ({-1, 0, +1}), and top-2 Sparse MoE.
- **Research Variant:** Controlled architectural modifications developed in `experiments/architecture_matrix/` to systematically test specific mechanisms (e.g. adaptive decay, write gating, erase gating, local buffer).
- **Ablation:** Staged removal or isolation of an individual component to quantify its marginal contribution (e.g., V2 without buffer, V2 without write gate).
- **Experimental Result:** Empirical observations measured under fixed compute/token budgets. Never conflated with claims of paper-baseline modifications.

---

## 2. Core Model Dimensions & Capacity (606M Parameter Scale)

Verified via model diagnostics:
- **Total Parameters:** $606,391,728$ (~606M)
- **Active Parameters / Token:** $405,014,528$ (~66.8% active compute per forward pass)
- **Layers:** 24 Transformer blocks
- **Hidden Dimension ($d_{\text{model}}$):** 1,024
- **Attention Heads ($n_{\text{heads}}$):** 16
- **Head Dimension ($d_{\text{head}}$):** 64
- **Vocabulary Size:** 50,257 (GPT-2 tokenizer encoding)
- **Experts per MoE Layer:** 4 experts (top-2 routed)
- **Rotary Position Embeddings (RoPE):** Applied to Query and Key representations after ELU+1 feature mapping.

---

## 3. Key Architectural Decisions

### ADR-01: Associative Linear Attention vs Softmax Attention
- **Decision:** Use recurrent associative attention with decaying state $S_t = \gamma S_{t-1} + k_t \otimes v_t$.
- **Rationale:** Standard softmax attention scales $O(T^2)$ in time and memory. Associative attention enables exact $O(1)$ memory state decoding during inference and $O(T)$ parallel training scans via chunked cumsum.

### ADR-02: Ternary Quantization with Straight-Through Estimator (STE)
- **Decision:** Restrict weight tensors to $\{-1, 0, +1\}$ during forward computation while maintaining latent FP32/BF16 master weights for gradient updates.
- **Rationale:** Ternary weights replace dense floating-point matrix multiplications with addition/subtraction accumulations, dramatically reducing inference energy and memory bandwidth.

### ADR-03: Sparse Mixture-of-Experts with Aux-Free Bias Routing
- **Decision:** Route tokens to top-2 experts out of 4 per layer, utilizing auxiliary-free bias balancing.
- **Rationale:** Enables scaling total capacity to 606M while evaluating only 405M parameters per token, maximizing parameter efficiency within consumer GPU VRAM limits.

### ADR-04: Audited CUDA Acceleration Backends
- **Decision:** Implement and independently audit custom CUDA C++ extensions for Associative Attention, Sparse MoE, and Liquid State Fusion.
- **Rationale:** PyTorch sequential loops in recurrent attention cause kernel launch overhead. The audited CUDA kernels fuse scan operations, yielding verified end-to-end training throughput of ~5,000+ tok/s while maintaining strict PyTorch fallback parity.

### ADR-05: Factorial Memory Research Over Monolithic Combination
- **Decision:** Reject the immediate adoption of "Memory Prime" (blindly enabling all proposed memory extensions). Instead, execute a full factorial elimination study across the 5 core mechanisms:
  - **A:** Adaptive Decay ($\gamma_t$)
  - **B:** Write Gate ($w_t$)
  - **C:** Erase Gate ($e_t$)
  - **D:** Gated Read ($r_t$)
  - **E:** Local Sliding Buffer ($W=16$)
- **Rationale:** Empirical data proved that combining all mechanisms simultaneously does not automatically yield superior performance; individual mechanisms can interfere or introduce redundant parameter overhead. Controlled interaction analysis ensures only mutually reinforcing mechanisms enter production.
