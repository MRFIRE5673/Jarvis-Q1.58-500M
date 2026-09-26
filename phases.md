# JARVIS-Q1.58-500M: STRATEGIC PROJECT PHASES & ROADMAP

**Operating Constraint:** `architecture.md` is strictly **NON-MUTABLE AND FROZEN**. All phases must adhere to the 24-layer, $d_{\text{model}}=1024$, 16-head, 4-expert Top-2 locked neuromorphic specification.

---

## Phase Overview & Goals

| Phase | Title | Primary Objective | Key Success Metric | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Phase 1** | **Architecture Lock & Training Audit** | Freeze architecture; audit all training scripts; identify dataset/checkpoint divergence | Audit completed; `architecture.md` locked; baseline documented | **ACTIVE** |
| **Phase 2** | **Dataset & Tokenization Pipeline** | Curate high-quality pre-training & code corpus; establish clean document & chat boundaries | Unified tokenized dataset ready; zero contamination; proper `<|endoftext|>` | **PENDING** |
| **Phase 3** | **Native High-Throughput Pre-Training** | Train model at scale using native 51,700+ tok/s CUDA engine to 100M+ and 1B tokens | Cross-Entropy Loss drops $< 2.2$; PPL $< 15.0$; zero NaNs | **PENDING** |
| **Phase 4** | **Instruction Fine-Tuning (SFT)** | Fine-tune model on conversational/coding instruction format (`User:` / `Assistant:`) | Coherent responses to "hi" & "write hello world code"; zero `` | **PENDING** |
| **Phase 5** | **Fast Native Inference Engine** | Implement recurrent linear attention state caching ($S_t$) for autoregressive generation | Inference throughput surges from 2.6 tok/s $\to$ **50–100+ tok/s** streaming | **PENDING** |
| **Phase 6** | **Throughput Optimization (100K+ tok/s)** | cuBLASLt epilogue fusion, persistent GEMM scheduling, and native integer ALU exploration | Sustained training throughput advances toward **100,000 tok/s** ceiling | **PENDING** |

---

## Detailed Phase Breakdown

### Phase 1: Architecture Lock & Training File System Audit
* **Goal:** Establish unambiguous ground truth.
* **Deliverables:**
  - Freeze `architecture.md` permanently.
  - Document all active and legacy training scripts in the repository.
  - Diagnose why earlier checkpoints (e.g. 4,209 / ~10M tokens) generated valid Python code while later ones (e.g. 4,284 / ~17.5M tokens) degraded.
* **Exit Gate:** Complete audit report delivered and agreed upon.

---

### Phase 2: Training Data & Tokenization Pipeline
* **Goal:** Clean, balance, and format data to ensure language fluency and code generation capability.
* **Deliverables:**
  - Audit raw sources: `jarvis_engine/data.txt` (Django source code) vs `data_clean.txt` vs `data/shards/` (FineWeb-Edu).
  - Create a balanced pre-training mixture: **50% High-Quality English (FineWeb-Edu) + 50% Verified Python/C++ Code**.
  - Fix boundary handling: ensure proper `<|endoftext|>` token insertion to prevent cross-document bleed and runaway whitespace generation.
* **Exit Gate:** Verified binary shards with deterministic dataloader streaming.

---

### Phase 3: Native High-Throughput Pre-Training Run
* **Goal:** Train the 606M model at high throughput to achieve linguistic and syntactic competence.
* **Deliverables:**
  - Harness the native CUDA engine running under CUDA Graphs at **51,733.7 tok/s**.
  - Train continuously through **100,000,000+ tokens** (or full 1.0B token budget).
  - Target training trajectory:
    - 10M tokens: Loss $\approx 3.2$
    - 50M tokens: Loss $\approx 2.4$
    - 100M tokens: Loss $\approx 1.9$
    - 500M tokens: Loss $\approx 1.5$
* **Exit Gate:** Sustained loss convergence $< 2.2$ on holdout test set with stable ternary weights.

---

### Phase 4: Supervised Instruction Fine-Tuning (SFT)
* **Goal:** Convert raw base next-token predictor into an interactive assistant that follows terminal prompts.
* **Deliverables:**
  - Curate a high-quality coding & conversational SFT dataset (e.g. Python instruction pairs, technical Q&A).
  - Define formal chat template:
    ```text
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>
    ```
  - Fine-tune base model for 500–1,000 steps with masked user loss.
* **Exit Gate:** Model answers greetings ("hi") with conversational greetings and prompts ("write hello world code") with valid Python scripts.

---

### Phase 5: High-Speed Native Inference & Recurrent State Caching
* **Goal:** Fix the 2.6 tok/s inference bottleneck to enable instant terminal chat.
* **Deliverables:**
  - Exploit linear attention recurrence: carry forward recurrent state $S_t \in \mathbb{R}^{B \times H \times D \times D}$ from step to step ($O(1)$ per token compute).
  - Eliminate the full-sequence context recomputation loop in `evaluate_jarvis.py`.
  - Stream tokens with sub-20ms latency per token (**50–100+ tokens/sec** interactive throughput).
* **Exit Gate:** Instant streaming terminal conversation with negligible VRAM overhead.

---

### Phase 6: Hardware Ceiling Breaking Toward 100,000+ Training tok/s
* **Goal:** Push native CUDA training throughput from 51.7K toward the theoretical 100K+ hardware floor.
* **Deliverables:**
  - **EXP-27-004:** cuBLASLt MoE W1 + GELU epilogue fusion (saving 0.58 ms and 1.6 GB DRAM).
  - **EXP-27-007:** Re-autotune backward GEMMs (MoE W1/W2 bwd dX/dW, Attn Out bwd).
  - **RMSNorm Register Caching:** Eliminate Phase 2 DRAM re-read in fused Add+RMSNorm.
  - **Kernel Pipelining:** Evaluate persistent kernel scheduling across Blackwell SMs.
* **Exit Gate:** New verified throughput milestone logged under `JARVIS_PHASE_XX` git tag.
