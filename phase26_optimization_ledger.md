# Jarvis Phase 26 Optimization Ledger: Golden Baseline Restoration & Ceiling Breaker

**Hardware Authority**: NVIDIA GeForce RTX 5070 Gaming Trio (Blackwell SM120, 46 SMs, 48 MB L2 cache, 12 GB GDDR7 @ 16,001 MHz, CC 12.0)  
**Golden Tag**: `JARVIS_PHASE24_GOLDEN` (`f6e92d5c363f79b4f8549f1273fabe32c3aea695`)  
**Golden Baseline Authority**:
- **Mean Step Latency**: 79.175 ms (1,000 updates sustained) / 79.048 ms (500 updates) / 79.265 ms (3 independent fresh-process sessions)
- **Sustained Throughput**: **51,733.7 tok/s** (1,000 updates) / **51,816.4 tok/s** (500 updates) / **51,675.8 tok/s** (3-session mean)
- **Peak Burst**: **78.663 ms / 52,070.1 tok/s**
- **Clocks**: Core ~3,330–3,337 MHz, Memory 16,001 MHz
- **Thermals & Power**: 66.0–69.0°C, 237.7–240.5 W (250 W cap), VRAM 1,166.07 MiB
- **Bitwise Parity**: Loss Step 1 = 10.987913, Step 2 = 10.672153, Step 3 = 10.300819 ($L_\infty = 0.0$, 0 NaNs, 0 Infs)

---

## Acceptance Rule (Section 32)
- **KEEP**: Only if it improves full end-to-end throughput ($Y > X$ statistically validated against the Golden Baseline).
- **REJECT**: If it only improves isolated kernel timing, increases variance/jitter, thrashes L2 cache, causes thermal throttling, or alters locked architecture semantics.
- **REGRESSION FIREWALL**: Every single experiment branches strictly from `JARVIS_PHASE24_GOLDEN`. Never stack unproven changes.

---

## Master Ledger

| Exp ID | Base Commit | Change Description | Expected Mechanism | Latency (ms) | Throughput (tok/s) | Clocks (Core/Mem) | Temp / Power | Numerical Parity | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-26-000** | `JARVIS_PHASE24_GOLDEN` | Restored Phase 24 Golden Baseline (Single stream, L2 layer_x2 pin, 6-kernel AdamW, locked algos `[1,0,0,0,3,0,0,1,2]`) | Clean restoration eliminating Phase 25 multi-stream graph overhead and L2 thrashing | **79.175 ms** (1k) <br> **79.048 ms** (500) <br> **79.265 ms** (3-sess) | **51,733.7** (1k) <br> **51,816.4** (500) <br> **51,675.8** (3-sess) | 3330 / 16001 MHz | 69.0°C / 240.5 W | Loss: 10.987913, 10.672153, 10.300819 ($L_\infty = 0$) | **FROZEN GOLDEN BASELINE** |

---

## Detailed Experiment Logs

### EXP-26-000: Golden Baseline Restoration & Lock
- **Base State**: `JARVIS_PHASE24_GOLDEN`
- **Root Cause Forensics of Phase 25 Regression**:
  1. `stream_norm` pipelining: Introduced 26 `cudaEventRecord` and `cudaStreamWaitEvent` calls per backward pass into the CUDA Graph.
  2. Memory Contention: Concurrent execution of norm reduction kernels alongside heavy backward GEMMs caused L2 cache line collisions on the 192-bit GDDR7 bus.
  3. cuBLASLt Drift: LM Head Bwd dW drifted from Candidate 1 to Candidate 0.
- **Restoration**:
  - Reverted `full_engine.h` and `runtime.cu`: Removed `stream_norm` and all fork/join event handles.
  - Reverted `full_engine.cu`: Restored single-stream sequential backward pass and `ws.layer_act` MoE forward staging.
  - Reverted `cublaslt_engine.cu`: Locked all 9 algorithm IDs to `[1, 0, 0, 0, 3, 0, 0, 1, 2]`.
- **Validation**:
  - 3 Independent fresh-process sessions: 79.757 ms, 78.999 ms, 79.039 ms $\implies$ Mean = 79.265 ms / 51,675.8 tok/s (0.44% variation).
  - 500 Updates: 79.048 ms / 51,816.4 tok/s sustained (Jitter 0.21%).
  - 1,000 Updates: 79.175 ms / 51,733.7 tok/s sustained (Jitter 0.27%, Peak burst: 78.676 ms / 52,061.5 tok/s).
  - Numerical parity: Step 1 = 10.987913, Step 2 = 10.672153, Step 3 = 10.300819 ($L_\infty = 0.0$, 0 NaNs).
- **Decision**: **FROZEN AS GOLDEN BASELINE AUTHORITY**.
