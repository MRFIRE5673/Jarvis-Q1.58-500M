# Jarvis Reproducibility & RNG State Management: Future Patch Plan

**Status:** Documented Architecture Plan (Post-50M Baseline Execution)  
**Safety Constraint:** Live baseline run PID 28148 is active. **DO NOT alter the live checkpoint or model code while PID 28148 runs.**

---

## 1. Context & Motivation

During the optimizer static code audit, two critical reproducibility gaps were identified in the active pretraining stack:
1. **Unsaved RNG States:** Checkpoints currently serialize `model_state_dict`, `optimizer_state_dict`, and `dataloader_state`, but omit CPU, CUDA, and Python RNG states. A resumed run does not continue the identical stochastic trajectory.
2. **MoE Router Noise (`torch.randn_like`):** In `jarvis_engine/jarvis_model.py` (line 220):
   ```python
   if self.training and self.noise_std > 0:
       logits = logits + torch.randn_like(logits) * self.noise_std
   ```
   Router exploration noise draws directly from PyTorch's global CUDA random number generator. If checkpoint resumption alters the CUDA RNG seed, token-to-expert assignment deviates immediately.

For scientific comparisons between optimizers (e.g., AdamW Control vs. Hybrid Muon), identical stochastic sequences (data order, router perturbations, dropout if any) must be guaranteed.

---

## 2. Isolated Future Patch Specification

The following modifications are planned for deployment **after the 50M AdamW baseline completes**:

### A. Full RNG State Serialization in Checkpoints

In `save_checkpoint`:
```python
# Future additions to checkpoint dictionary
state["cpu_rng_state"] = torch.get_rng_state()
if torch.cuda.is_available():
    state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
state["python_rng_state"] = random.getstate()
state["numpy_rng_state"] = np.random.get_state()
```

In `load_checkpoint`:
```python
if "cpu_rng_state" in state:
    torch.set_rng_state(state["cpu_rng_state"])
if "cuda_rng_state" in state and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
if "python_rng_state" in state:
    random.setstate(state["python_rng_state"])
if "numpy_rng_state" in state:
    np.random.set_state(state["numpy_rng_state"])
```

Backward Compatibility Guarantee:
`state.get("cuda_rng_state")` will gracefully ignore missing keys when loading older checkpoints (such as `ckpt_step_0004284_best.pt` or `ckpt_step_002500.pt`).

---

### B. MoE Router Determinism & Independent Generator

To isolate router exploration noise from global CUDA RNG perturbations (e.g., data pipeline jitter or auxiliary operations), the router will support an isolated deterministic RNG generator:

```python
class MoERouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, noise_std: float = 0.1):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
        self.noise_std = noise_std
        self._rng_generator = None

    def set_generator(self, generator: torch.Generator):
        self._rng_generator = generator

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        if self.training and self.noise_std > 0:
            if self._rng_generator is not None:
                noise = torch.randn(logits.shape, device=logits.device, dtype=logits.dtype, generator=self._rng_generator)
            else:
                noise = torch.randn_like(logits)
            logits = logits + noise * self.noise_std
        ...
```

---

### C. Dataset Streaming RNG Alignment

In `data/streaming_dataloader.py`:
- Verify that shard transition shuffling or offset reading is strictly keyed on `(seed, current_shard_idx, current_offset)`.
- When initializing comparative optimizer runs via `--init-baseline`, ensure all candidates instantiate `ShardedTokenDataset(seed=42)` and read identical token streams.

---

## 3. Implementation Checklist for Post-50M Activation

- [ ] Add `cpu_rng_state`, `cuda_rng_state`, `python_rng_state` to `save_checkpoint` in `train_1b_production.py`.
- [ ] Add conditional restoration in `load_checkpoint`.
- [ ] Add dedicated `torch.Generator` initialization for MoE routers during bakeoff runs.
- [ ] Run a 50-step determinism regression test: verify that two independent runs initialized from the same checkpoint and seed yield bit-exact identical loss trajectories.
- [ ] Do NOT apply changes to PID 28148 production files until the baseline run finishes.
