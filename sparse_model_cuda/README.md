# Sparse MoE Custom CUDA Backend (`sparse_model_cuda`)

## Architecture & Engineering Overview
This is an isolated, experimental CUDA extension for accelerating the Sparse Mixture-of-Experts (MoE) dispatch, gather, and scatter-combining pathways in the Jarvis 606M architecture on NVIDIA GeForce RTX 5070 (Blackwell `sm_120`).

### 1. Motivation & Bottlenecks in Standard PyTorch MoE
The standard PyTorch MoE implementation (`SparseMoELayer`) in `jarvis_model.py` relies on:
1. `x_flat.repeat_interleave(self.top_k, dim=0)`: Allocates and copies $N \times K \times C$ intermediate hidden states into global memory.
2. `torch.zeros_like(flat_x)`: Allocates and zeroes another $N \times K \times C$ intermediate buffer.
3. Dynamic Boolean Indexing (`mask = flat_idx == e`): Triggers 4 separate non-zero index searches (`nonzero()`), 4 dynamic tensor allocations, and 4 gather passes.
4. Scattered Output Combination: Requires indexed assignments `flat_out[mask] = flat_gate[mask] * ye` followed by `.sum(dim=1)` reductions.
5. Backward Path: Traces through dynamic-sized graphs, creating over 27 separate kernel launches per layer (over 5,000 launches per optimizer update).

### 2. Custom CUDA Design Principles
1. **GEMMs Remain with cuBLAS**: Large matrix multiplications ($W_1$ and $W_2$) remain with highly-optimized cuBLAS/cuBLASLt tensor core engines.
2. **Fused Dispatch & Permutation Gathering (`moe_dispatch_gather_cuda`)**:
   - Computes deterministic token histogram and partition offsets across experts.
   - Gathers input tokens directly into a contiguous $(N \cdot K, C)$ buffer grouped by expert in a single coalesced memory pass.
   - Zero `repeat_interleave`, zero boolean masks, zero dynamic shape allocations.
3. **Fused Scatter & Gating Accumulation (`moe_scatter_combine_cuda`)**:
   - Directly reads expert outputs $(N \cdot K, C)$ and gate weights $(N, K)$, writing final combined output $(N, C)$ in a single launch.
   - Accumulates in FP32 registers for numerical stability before storing in BF16 or FP32.
   - Zero intermediate `flat_out`, zero atomic conflicts, zero race conditions.
4. **Coalesced Backward Passes (`moe_scatter_backward_cuda`, `moe_gather_backward_cuda`)**:
   - Reverses the scatter and gather operations with exact gradient accumulation for input $x$, expert outputs $Y$, and router gates.

### 3. File Structure
- `setup.py`: Build configuration supporting MSVC 2022 + CUDA 13.3 + `sm_120`.
- `sparse_model.cu`: CUDA kernels for dispatch, gather, scatter, and backward.
- `sparse_model_cpp.cpp`: C++ Pybind11 host bindings.
- `sparse_model.py`: Python wrapper and `CUDASparseMoELayer` module.
- `validate_sparse_model.py`: Numerical correctness and equivalence test suite across shapes and edge cases.
- `benchmark_sparse_model.py`: Isolated component sweeps and full-model A/B benchmark.
