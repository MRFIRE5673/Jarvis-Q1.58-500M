# Jarvis Q1.58-500M: Sub-Quadratic Ternary Language Model with Recurrent Memory & Sparse MoE

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Jarvis Q1.58-500M** is a high-efficiency sub-quadratic language model combining:
1. **Recurrent Associative Linear Attention:** $O(1)$ constant-memory autoregressive decoding and $O(T)$ parallel training via causal cumsum scans.
2. **Ternary Quantized Weights (1.58-bit):** Ternary parameter representation (\{-1, 0, +1\}) evaluated via Straight-Through Estimators (STE).
3. **Sparse Mixture-of-Experts (MoE):** 4 experts per layer with top-2 auxiliary-free routing, expanding parameter capacity while bounding active compute per token.
4. **Liquid State Fusion (LSF):** Continuous-time decaying recurrence capturing multi-timescale sequential dynamics.

---

## Model Capacity & Architecture Specifications

Diagnostics verified on the 606M parameter scale:

| Metric | Verified Value | Description |
| :--- | :---: | :--- |
| **Total Parameters** | **606,391,728** | Total model weights (embeddings + 24 blocks + LM head) |
| **Active Parameters / Token** | **405,014,528** | Active compute per token forward pass (~66.8% active) |
| **Layer Count** | 24 | Transformer blocks with residual normalization |
| **Hidden Dimension ($d_{\text{model}}$)** | 1,024 | Internal representation width |
| **Attention Heads ($n_{\text{heads}}$)** | 16 | Number of associative attention heads |
| **Head Dimension ($d_{\text{head}}$)** | 64 | Dimension per attention head |
| **Vocabulary Size** | 50,257 | GPT-2 BPE tokenizer vocabulary |
| **MoE Experts** | 4 | 4 routed experts per layer (top-2 active per token) |
| **Position Encoding** | RoPE | Rotary Position Embeddings applied post-feature map |
| **Normalization** | RMSNorm | Root-mean-square layer normalization |

---

## Paper-Faithful Baseline vs. Research Variants

To maintain scientific integrity, this repository maintains an explicit distinction:

- **Paper-Faithful Baseline:** The original foundational architecture (`jarvis_engine/jarvis_model.py`, checkpoint `ckpt_step_0004209.pt`). Uses scalar decay associative linear attention, Liquid State Fusion, 1.58-bit ternary quantization, and top-2 MoE routing. The paper baseline remains permanently immutable.
- **Research Variants:** Controlled architectural extensions developed in `experiments/architecture_matrix/` exploring adaptive input-dependent retention, selective write/erase gates, local sliding-window buffers, and activation variants.
- **Factorial Memory Study:** Staged factorial investigation determining which recurrent memory mechanisms cooperate versus interfere under fixed token and compute budgets.

---

## Audited CUDA Acceleration Backends

Jarvis includes custom, audited CUDA C++ extensions with seamless PyTorch CPU/CUDA fallbacks:
- **`associative_attention_cuda/`:** Fused associative attention forward and backward kernel with exact synchronization and state persistence.
- **`sparse_model_cuda/`:** High-throughput sparse MoE router and scatter-add expert dispatcher.
- **`liquid_fusion_cuda/`:** Vectorized continuous-time liquid state causal decay scan.

### Benchmark Throughput:
- **Baseline PyTorch Fallback:** ~504 tok/s
- **Audited CUDA Acceleration:** ~5,160+ tok/s prefill throughput on an NVIDIA RTX 5070 (12GB sm_120).

---

## Interactive Terminal Inference & Evaluation

Jarvis provides a streaming command-line interface for interactive experimentation:

```bash
# Launch interactive CMD terminal session
python jarvis_engine/evaluate_jarvis.py --interactive

# Autoregressively stream a single prompt
python jarvis_engine/evaluate_jarvis.py --prompt "def quicksort(arr):" --max_new_tokens 150

# Quantitative evaluation on genuine holdout dataset
python jarvis_engine/evaluate_jarvis.py --eval_file fresh_holdout.txt --num_windows 50 --seq_len 512
```

---

## Research Program & Factorial Matrix

For comprehensive details on our ongoing research into advanced recurrent memory mechanisms:
- **[Research Overview](research/README.md)**
- **[Pairwise Interaction Matrix](research/interaction_matrix.md)**
- **[Experimental Methodology & Controls](research/methodology.md)**
- **[Architectural Decision Records](research/architecture_decisions.md)**

### Completed Factorial Memory Snapshot (Controlled 100-step test):
- **Baseline Holdout Loss:** CE **3.2858** | PPL **26.73**
- **Top Pair (`B+C`, Write + Erase Gate):** Final CE **3.2882** | PPL **26.79** | Interaction **-0.0109** (Synergistic)
- **Second Pair (`A+C`, Adaptive Decay + Erase Gate):** Final CE **3.2895** | PPL **26.83** | Interaction **-0.0118** (Synergistic)
- *Note:* All seven completed pairs exhibit negative interaction coefficients, showing that dynamic gating mechanisms cooperate effectively under neutral initialization.

---

## Repository Structure

```
Jarvis-Q1.58-500M/
├── jarvis_engine/                  # Core production model and runtime
│   ├── jarvis_model.py             # Paper-faithful 606M architecture
│   ├── train.py                    # Production training and fine-tuning engine
│   └── evaluate_jarvis.py          # Interactive terminal and evaluation CLI
├── associative_attention_cuda/     # Custom CUDA Associative Attention backend
├── sparse_model_cuda/              # Custom CUDA Sparse MoE backend
├── liquid_fusion_cuda/             # Custom CUDA Liquid State Fusion backend
├── runtime/                        # Optimized streaming runtime & scheduler
├── experiments/                    # Research infrastructure & ablation matrix
│   ├── architecture_matrix/        # Factorial memory experiments & modular code
│   │   ├── modular_memory.py       # Modular attention block (A, B, C, D, E)
│   │   ├── factorial_research_runner.py  # Controlled experiment driver
│   │   ├── batch_factorial_matrix.py     # Automated factorial queue
│   │   └── interaction_analyzer.py       # Synergy/antagonism calculator
│   └── leaderboard.md              # Live experiment leaderboard
├── research/                       # Scientific documentation & reports
│   ├── README.md                   # Research hub index
│   ├── methodology.md              # Controlled experimental protocols
│   ├── interaction_matrix.md       # Pairwise memory interaction results
│   └── architecture_decisions.md   # Architectural Decision Records (ADRs)
├── benchmarks/                     # Hardware and scaling benchmark suites
└── tests/                          # Rigorous numerical parity & correctness tests
```

---

## Verification & Correctness Testing

Run the full correctness test suite covering forward/backward parity, causal invariance, state persistence, and NaN/Inf stability:

```bash
python tests/test_correctness_gate.py
python verify_production_integration.py
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
