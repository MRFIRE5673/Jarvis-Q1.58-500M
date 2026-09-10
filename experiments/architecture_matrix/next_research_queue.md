# Jarvis Architecture Matrix — Future Research Experiment Queue
================================================================

This document establishes the scientific research roadmap following **Sprint #2**, ranked by expected value per FLOP, per parameter, and per GB of VRAM.

Every experiment has a concrete hypothesis, measurable acceptance criteria, complexity assessment, and risk profile.

---

## Priority Classification Summary

- **P0 (Critical Path):** Mandatory experiments required for core model capabilities, hardware-level acceleration, and full convergence.
- **P1 (High Value Architecture Refinements):** Controlled architectural improvements that increase efficiency or throughput without introducing instability.
- **P2 (Exploratory / Downstream):** Post-convergence capabilities, edge deployment, and reasoning extensions.

---

## Ranked Experiment Queue

### 1. [P0] Full 1.0B Token Pre-Training of Jarvis-vNext (W16 Gated Memory + MoE)
- **Hypothesis:** Pre-training Jarvis-vNext from scratch (or continuation from calibrated baseline) on 1,000,000,000 tokens of high-quality educational text (FineWeb-Edu curated subset) with BFloat16 master weights and AbsMean ternary STE will drive validation CE from $3.28$ to $<2.75$ ($\text{PPL} < 15.6$) without gradient divergence or loss spikes.
- **Expected Benefit:** Produces the first fully-trained, production-grade 1.58-bit associative model with long-context memory retention.
- **Implementation Complexity:** Low (Pipeline, sharded dataset, and `ShardedTokenDataset` memory-mapped dataloader are already completed and validated).
- **Compute Cost:** High (~138.2 wall-clock hours on RTX 5070 12GB, ~5.76 days).
- **VRAM Risk:** Very Low (modeled and verified at 6.10 GB / 12.0 GB headroom).
- **Scientific Value:** Extreme (milestone achievement for open-weights sub-bit ternary research).
- **Priority:** **P0**

---

### 2. [P0] Fused 2-Bit Packed Ternary Tensor Core CUDA Kernel (W2A16 GEMV / GEMM)
- **Hypothesis:** By implementing a custom CUDA/CUTLASS kernel that loads 2-bit packed ternary weights directly into warp registers, unpacks them via bitwise shift-and-mask (`SHR`, `AND`, `SUB`) into signed integers, and computes dot-products with BF16 activation vectors directly in shared memory/registers, inference speed will scale linearly with memory bandwidth savings, achieving $2.5\times - 3.5\times$ wall-clock speedup over standard PyTorch `nn.Linear`.
- **Expected Benefit:** Bridges the gap between theoretical VRAM compression ($7.2\times$) and actual execution speedup on consumer GPUs.
- **Implementation Complexity:** High (requires PTX assembly or CUTLASS tensor core template specialization).
- **Compute Cost:** Low (benchmarking and micro-kernels).
- **VRAM Risk:** None (reduces memory consumption).
- **Scientific Value:** Very High.
- **Priority:** **P0**

---

### 3. [P0] Multi-Scale Associative Decay Layer Ablation in Scaled Pre-Training
- **Hypothesis:** Assigning learned or geometrically-spaced decay factors ($\gamma_h \in [0.90, 0.999]$) across attention/memory heads combined with the W=16 chunk buffer will improve deep needle retrieval rank at $T=8,192$ by $>15\%$ relative to single-scalar $\gamma$, without causing recurrent state explosion.
- **Expected Benefit:** Enables Jarvis to track both short-term syntactic structures and long-horizon document facts simultaneously.
- **Implementation Complexity:** Medium (per-head decay vector integration into recurrent forward pass).
- **Compute Cost:** Medium (20M-token controlled comparison).
- **VRAM Risk:** Very Low (O(1) state per head).
- **Scientific Value:** Very High.
- **Priority:** **P0**

---

### 4. [P1] MoE Variant A (8 Experts, Top-1 Routing) Efficiency Validation
- **Hypothesis:** Scaling expert count from 4 to 8 with Top-1 routing increases total parameter capacity to 957.7M (+72.6%) while simultaneously decreasing active compute to 253.1M active parameters/token (-28.4% FLOPs), yielding lower perplexity per FLOP than the 4-expert Top-2 baseline.
- **Expected Benefit:** Maximizes model capacity on disk while increasing forward-pass throughput on bandwidth-constrained GPUs.
- **Implementation Complexity:** Medium (requires tuning router auxiliary loss or bias to prevent Top-1 expert starvation).
- **Compute Cost:** Medium (50M-token comparison).
- **VRAM Risk:** Low (+192 MB packed expert weights).
- **Scientific Value:** High.
- **Priority:** **P1**

---

### 5. [P1] Auxiliary-Free Dynamic Router Bias Balancing
- **Hypothesis:** Replacing static auxiliary balance loss ($\lambda_{\text{bal}} \mathcal{L}_{\text{bal}}$) with dynamic router bias adjustments ($s_i = \text{softmax}(W_g x + b_i)$ where $b_i \leftarrow b_i + \eta (\bar{f} - f_i)$) achieves perfect expert utilization ($CV < 0.05$) without corrupting the task representation gradient.
- **Expected Benefit:** Completely eliminates router load imbalance while eliminating the ~1.5% CE degradation caused by auxiliary penalty gradients.
- **Implementation Complexity:** Low (straightforward bias buffer update during training loop).
- **Compute Cost:** Very Low.
- **VRAM Risk:** None.
- **Scientific Value:** High.
- **Priority:** **P1**

---

### 6. [P1] Recurrent State Compaction via Head-Grouped State Projection
- **Hypothesis:** Grouping the 16 recurrent memory heads into 4 shared associative state matrices ($4\times$ reduction in state size) preserves $\ge 98\%$ of long-context needle retrieval while reducing recurrent state memory footprint to $<1.0$ MB per sequence.
- **Expected Benefit:** Allows scaling inference batch size $4\times$ at context lengths $T \ge 8,192$ on 12GB VRAM.
- **Implementation Complexity:** Medium (grouped query projection and state broadcasting).
- **Compute Cost:** Low.
- **VRAM Risk:** Positive (VRAM reduction).
- **Scientific Value:** Medium.
- **Priority:** **P1**

---

### 7. [P1] Multi-Stage Curriculum Context Expansion ($512 \to 1024 \to 2048$)
- **Hypothesis:** Staging the 1.0B token run as 800M tokens at $T=512$, 150M tokens at $T=1024$, and 50M tokens at $T=2048$ achieves parity with full-context pre-training while saving ~40 GPU hours of attention compute.
- **Expected Benefit:** Accelerates total training schedule from 5.76 days to 4.2 days.
- **Implementation Complexity:** Low (handled via dataloader sequence batcher).
- **Compute Cost:** High (part of the 1B run).
- **VRAM Risk:** Low (peak VRAM at T=2048 with gradient checkpointing is ~8.2 GB).
- **Scientific Value:** High.
- **Priority:** **P1**

---

### 8. [P2] Full-Integer 1.58-Bit Pipeline (W1.58A8 / W1.58A4 Activation Quantization)
- **Hypothesis:** Dynamic 8-bit per-token activation quantization paired with packed 1.58-bit ternary weights allows computing matrix multiplication purely through integer additions and subtractions (`IDP4A` or `MMA` instructions), removing FP16/BF16 ALUs from the inner loop.
- **Expected Benefit:** Extreme energy efficiency (lowest Watt/token) and latency on embedded/mobile GPUs.
- **Implementation Complexity:** High (requires calibration and smooth-quantization techniques).
- **Compute Cost:** Medium.
- **VRAM Risk:** None.
- **Scientific Value:** High.
- **Priority:** **P2**

---

### 9. [P2] Speculative Decoding with Lightweight 1.58-Bit Draft Model
- **Hypothesis:** Pairing Jarvis-607M with an ultra-compact 80M 1.58-bit draft model sharing the same tokenizer achieves speculative verification acceptance rates $>75\%$, increasing generation throughput from ~35 tok/s to $>90$ tok/s on single GPU.
- **Expected Benefit:** High-speed real-time interactive generation.
- **Implementation Complexity:** Medium (draft token tree verification).
- **Compute Cost:** Low (draft model training: ~10 hours).
- **VRAM Risk:** Low (+60 MB packed draft weights).
- **Scientific Value:** Medium.
- **Priority:** **P2**

---

### 10. [P2] Reasoning & Agentic Step-by-Step Supervised Fine-Tuning (SFT)
- **Hypothesis:** Fine-tuning the converged 1B-token Jarvis-vNext base checkpoint on 50,000 structured chain-of-thought trajectories (e.g. Open-R1, GSM8K, Code reasoning) enables the recurrent associative memory to hold intermediate working hypotheses across extended multi-step deductions.
- **Expected Benefit:** Activates reasoning and tool-use intelligence inside a sub-bit quantized parameter budget.
- **Implementation Complexity:** Medium.
- **Compute Cost:** Medium (~24 GPU hours).
- **VRAM Risk:** Low.
- **Scientific Value:** Very High.
- **Priority:** **P2**
