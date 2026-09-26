# JARVIS-Q1.58-500M: PERSISTENT MEMORY & SYSTEM DIRECTIVES

This document serves as the persistent memory store for all agent and engineering sessions on the Jarvis-Q1.58-500M project. It records non-negotiable architectural rules, hardware authority, verified baselines, and critical findings.

---

## 1. ABSOLUTE ARCHITECTURAL FIREWALL (`architecture.md` IS IMMUTABLE)

`architecture.md` is **STRICTLY NON-MUTABLE AND PERMANENTLY LOCKED**.
* **Do NOT alter model architecture, dimensions, or hyperparameters.**
* **Layer count is strictly 24.**
* **Hidden dimension $d_{\text{model}}$ is strictly 1024.**
* **Attention heads: 16 (head_dim = 64).**
* **MoE configuration: 4 experts per layer, Top-2 routing strictly locked.**
* **Quantization scheme: Spiking Ternary $Q_{1.58}$ with AbsMean scaling and STE.**
* **Attention mechanism: $O(N)$ Infinite Associative Linear Attention with per-head learnable decay $\gamma$.**
* **Membrane dynamics: Liquid State Fusion (LSF) SNN dynamics.**
* **Loss formulation: Cross-Entropy + Load-Balance Loss (Eq. 6) + Reflective Penalty (Eq. 7).**

---

## 2. HARDWARE ENVIRONMENT & OPERATING CEILING

* **GPU:** NVIDIA GeForce RTX 5070 12GB Gaming Trio (Blackwell SM120, Compute Capability 12.0)
* **Memory Bus:** 192-bit GDDR7 @ 16,001 MHz (28 Gbps) $\implies$ **672.0 GB/s physical rated bandwidth**
* **Clocks & Power:** 3,330–3,375 MHz core, 237–240 W load draw, 66–69°C operating thermals
* **Toolchain:** Windows 11 Pro, CUDA 13.3, MSVC 14.44, Python 3.14, PyTorch 2.12.0
* **Compilation Target:** `-gencode=arch=compute_120,code=sm_120` exclusively (100% native SM120 ISA)

---

## 3. VERIFIED PERFORMANCE & GOLDEN BASELINE AUTHORITY

* **Git Golden Tag:** `JARVIS_PHASE24_GOLDEN` (Commit `f6e92d5c363f79b4f8549f1273fabe32c3aea695`)
* **Sustained Training Throughput:** **51,733.7 tok/s** (79.175 ms/update over 1,000 updates / 4,096,000 tokens)
* **Peak Burst Throughput:** **52,070.1 tok/s** (78.663 ms)
* **Jitter:** 0.27% sustained
* **VRAM Footprint:** 1,166.07 MiB Allocated | 1,186.00 MiB Reserved (0 bytes leaked across 4M tokens)
* **Numerical Parity Invariant:** Deterministic Seed 42 loss progression:
  - Step 1: 10.987913
  - Step 2: 10.672153
  - Step 3: 10.300819 ($L_\infty = 0.0000000\text{e}+00$ bitwise match, 0 NaNs, 0 Infs)
* **Hardware Roofline Fact:** AdamW in serial execution achieves **673.8 GB/s (100.27% of rated GDDR7 bus)** across 606.4M parameters (13.342 GB DRAM traffic in 19.80 ms). AdamW is at the speed-of-light ceiling.

---

## 4. CRITICAL DISCOVERIES & LESSONS LEARNED

1. **Multi-Stream Event Overhead Regresses Performance:** In Phase 25, fine-grained cross-stream CUDA events in CUDA Graphs added driver/scheduling overhead that caused a +5.3 ms regression (79.9 ms $\to$ 85.2 ms). Single-stream linear DAG execution is superior for this model shape.
2. **L2 Cache Pinning Horizon:** Active single-layer forward activations (46.11 MB) fit completely within the 48 MB L2 cache only when staged through a single pinned buffer (`layer_x2`, 4.19 MB). Staging across multi-buffer arrays (100 MB) destroys cache pinning and degrades throughput by >3 ms.
3. **Compute-Bound Vectorization Trap:** Vectorizing transcendental kernels (`fused_gelu_bwd` with tanh + sech²) across 8 elements serially destroyed SM120 Instruction-Level Parallelism (ILP), regressing step time by +2.7 ms. Scalar thread distribution maximizes transcendental throughput.
4. **Quantization Collapse Under Paper Eq. 3:** Quantizing without AbsMean scaling ($W_q = \text{round}(\text{clamp}(W, -1, 1))$) causes **100.00% zero-weight collapse** because initial weights have $\sigma \approx 0.025 \ll 0.5$. AbsMean scaling ($\alpha = \text{mean}(|W|)$) is mathematically essential.
5. **Inference vs Training Mismatch:** Training processes 4,096 tokens in parallel via locked Tensor Core cuBLASLt GEMMs under a CUDA Graph. Eager PyTorch token-by-token generation has no state caching, causing it to run at 1.5–2.6 tok/s and recompute all 24 layers from scratch per token.
6. **Dataset & Checkpoint Behavioral Drift:**
   - Checkpoint 4,209 (~10M–17M tokens) was trained on raw Django Python code (`jarvis_engine/data.txt`), biasing it toward Python syntax.
   - Checkpoint 4,284 (~17.5M tokens) underwent extended training with patched STE on `data_clean.txt`, causing behavioral shifts without full loss convergence.

---

## 5. ACTIVE DIRECTIVES & WORKFLOW RULES

1. **Section 32 Regression Firewall:** Every optimization must be benchmarked against `JARVIS_PHASE24_GOLDEN`. If throughput drops below 51,733 tok/s or parity is broken ($L_\infty \neq 0$), revert immediately.
2. **Never stack unproven modifications.**
3. **Always separate Pre-training, SFT (Instruction Tuning), and Inference optimizations.**
