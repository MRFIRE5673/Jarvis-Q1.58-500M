# JARVIS ULTRA — PHASE 11B: CHECKPOINTING STRATEGY SWEEP REPORT
**Author:** Antigravity AI Engine Forensics  
**Date:** September 12, 2026  
**Hardware:** NVIDIA GeForce RTX 5070 12GB (Blackwell SM120, Physical VRAM Ceiling: 12,226.5 MiB)  
**Configuration Matrix:** $B=4, T=512, \text{accum}=2$, BF16, CUDA Graph ON (4,096 tokens/update)  

---

## 1. Executive Summary

This benchmark conducted an exhaustive, isolated sweep of **13 checkpointing topologies** to identify the Pareto frontier between recomputation FLOP reduction and VRAM allocation on the RTX 5070 12GB.

### **Key Findings & Physical Boundaries**:
1. **The 12GB Physical VRAM Cliff**:
   - Disabling gradient checkpointing completely (`NONE`) requires **14,014.0 MiB** reserved VRAM, triggering immediate OOM / fatal eviction beyond the 12,226.5 MiB physical limit.
   - Any coarse block-skip strategy below 50% checkpointing (`EVERY_3`, `EVERY_4`, `EVERY_6`, `EVERY_8`) demands between **12,386 MiB and 13,402 MiB**, failing the zero-paging physical contract.
2. **Selective Checkpointing Winner: `MOE_ONLY`**:
   - Checkpointing only the MoE feedforward blocks (`moe_only`) while storing attention activations achieves **14,009.8 tok/s (292.37 ms/update)**.
   - Peak Reserved VRAM: **10,774.0 MiB**, maintaining a safe **+1,452.5 MiB physical headroom** with **zero WDDM paging**.
   - This delivers a **+9.5% verified speedup (+1,216.0 tok/s)** over the full checkpointing production baseline.
3. **`ATTENTION_ONLY` Paging Degeneration**:
   - In contrast, checkpointing only attention while storing full MoE intermediate states reaches **12,166.0 MiB**, triggering Windows WDDM driver eviction churn and collapsing throughput to **2,447.3 tok/s** (1,673.71 ms).

---

## 2. Complete Sweep Results (Ranked by Verified Optimizer-Step Tok/s)

| Rank | Strategy ID & Description | Wall Step (ms) | GPU Time (ms) | True Tok/s | Peak Alloc (MiB) | Peak Reserved (MiB) | Headroom (MiB) | Paging | Loss | Grad Norm | Status | Forensic Reason |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **S04: MOE_ONLY** | **291.51** | **291.43** | **14,051.1** | 7,450.0 | **10,774.0** | **+1,452.5** | **None** | 11.2186 | 0.9975 | **PASS** | **Verified Pareto Optimal** |
| **2** | **S01: FULL Checkpointing (Baseline)** | **320.86** | **320.78** | **12,765.6** | 6,871.3 | **8,774.0** | **+3,452.5** | **None** | 11.2541 | 0.9961 | **PASS** | Production Baseline |
| **3** | **S05: EVERY_2 (Alternating Blocks)** | **469.43** | **469.31** | **8,725.5** | 8,629.5 | **11,582.0** | **+644.5** | **None** | 11.2542 | 0.9982 | **PASS** | Suboptimal Recomputation |
| 4 | S03: ATTENTION_ONLY | 1,673.71 | 1,673.55 | 2,447.3 | 8,790.4 | 12,166.0 | +60.5 | **WDDM Eviction** | 11.2164 | 0.9970 | **REJECTED** | Driver thrashing near VRAM limit |
| — | S06: EVERY_3 (1/3rd Blocks) | — | — | 0.0 | — | 12,386.0 | -159.5 | OOM / Paging | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |
| — | S11: ALT_MOE (MoE Every 2nd Block) | — | — | 0.0 | — | 12,574.0 | -347.5 | OOM / Paging | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |
| — | S07: EVERY_4 (1/4th Blocks) | — | — | 0.0 | — | 12,858.0 | -631.5 | OOM / Paging | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |
| — | S10: ALT_ATTN (Attn Every 2nd Block)| — | — | 0.0 | — | 13,098.0 | -871.5 | OOM / Paging | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |
| — | S12: ATTN_EVERY_3 (Attn 1/3rd) | — | — | 0.0 | — | 13,168.0 | -941.5 | OOM / Paging | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |
| — | S08: EVERY_6 (1/6th Blocks) | — | — | 0.0 | — | 13,238.0 | -1,011.5 | OOM / Paging | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |
| — | S09: EVERY_8 (1/8th Blocks) | — | — | 0.0 | — | 13,402.0 | -1,175.5 | OOM / Paging | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |
| — | S13: ATTN_EVERY_4 (Attn 1/4th) | — | — | 0.0 | — | 13,630.0 | -1,403.5 | OOM / Paging | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |
| — | S02: NONE (Zero Checkpointing) | — | — | 0.0 | — | 14,014.0 | -1,787.5 | Fatal OOM | — | — | **REJECTED** | Exceeds 12,226.5 MiB ceiling |

---

## 3. Finalist Rigorous Evaluation (25 Steady-State Iterations)

Evaluated in clean subprocesses across 25 consecutive optimizer updates following 3 warmup updates:

### **Rank 1 Winner: `S04: MOE_ONLY`**
- **Mean Latency:** $292.37\text{ ms} \pm 0.90\text{ ms}$
- **Median Latency:** $292.44\text{ ms}$
- **P95 Latency:** $293.57\text{ ms}$
- **Mean Optimizer Throughput:** **$14,009.8\text{ tok/s}$** ($+9.5\%$ vs baseline)
- **Peak Reserved VRAM:** $10,774.0\text{ MiB}$ (Headroom: $+1,452.5\text{ MiB}$)
- **Loss / Grad Norm:** $11.1871$ / $0.9975$ (Numerically stable, no NaN/Inf)
- **Parameter Max Delta:** $0.001709$, RMSE: $0.000373$

### **Rank 2: `S01: FULL Checkpointing` (Production Baseline)**
- **Mean Latency:** $320.15\text{ ms} \pm 0.73\text{ ms}$
- **Median Latency:** $320.16\text{ ms}$
- **P95 Latency:** $321.13\text{ ms}$
- **Mean Optimizer Throughput:** **$12,793.8\text{ tok/s}$**
- **Peak Reserved VRAM:** $8,774.0\text{ MiB}$ (Headroom: $+3,452.5\text{ MiB}$)
- **Loss / Grad Norm:** $11.1852$ / $0.9961$

### **Rank 3: `S05: EVERY_2 (Alternating Blocks)`**
- **Mean Latency:** $376.24\text{ ms} \pm 1.06\text{ ms}$
- **Mean Optimizer Throughput:** **$10,886.8\text{ tok/s}$**
- **Peak Reserved VRAM:** $11,582.0\text{ MiB}$ (Headroom: $+644.5\text{ MiB}$)

---

## 4. Phase B Verdict

`MOE_ONLY` is established as the **optimal safe checkpointing policy** for $B=4, \text{accum}=2$. It unlocks $14,010\text{ tok/s}$ while safely fitting inside physical memory with zero paging.
