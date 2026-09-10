# CUDA-Accelerated Associative Linear Attention Backend

## Architectural Overview
This directory contains the isolated, experimental CUDA-accelerated Associative Linear Attention backend for the Jarvis 606M neuromorphic language model on NVIDIA RTX 5070 (Blackwell `sm_120`).

### Key Optimizations
1. **Fused RoPE + ELU(x)+1 + Q-Scaling CUDA Kernel (`associative_attention_cuda.cu`)**:
   - Fuses feature activation ($\text{ELU}(x) + 1.0$), dimension scaling ($1 / \sqrt{D}$), and Rotary Position Embedding application into a single coalesced memory pass.
   - Eliminates intermediate tensor copies and allocations from `rotate_half` (`torch.cat`), cutting memory traffic and kernel launch overhead.
   - Exact mathematical forward and backward implementations matching PyTorch reference down to double-precision tolerances.
2. **High-Throughput Batched Tensor Core Recurrence (`torch.bmm`)**:
   - Replaces the serialized Python chunk loop with batched matrix multiplications across all $N_c = \lceil T / C \rceil$ chunks simultaneously.
   - Increases hardware occupancy on Blackwell SMs by grouping $B \times H \times N_c$ matrix operations into unified tensor core calls.
3. **Fused Recurrent Chunk State Scan Kernel (`recurrent_chunk_state_scan_forward_kernel`)**:
   - Maintains the $64 \times 64$ carried state matrices across chunks entirely within threadblock registers and shared memory.
   - Computes $S_{k+1} = \gamma^C S_k + \Delta S_k$ with zero intermediate global memory writes.
   - Implements reverse autograd backward pass accumulating exact $\nabla \gamma$ and propagating state gradients.

## Files
- `setup.py`: Build configuration targeting CUDA 13.3 and `sm_120`.
- `associative_attention_cuda.cu`: CUDA kernels and C++ host dispatchers.
- `associative_attention_cpp.cpp`: PyBind11 Python bindings.
- `associative_attention.py`: Python module providing `CUDAAssociativeLinearAttention` as a drop-in replacement.
- `validate_attention.py`: 12-test validation suite checking FP32/BF16 numerical precision, autograd gradients, edge cases, and inference state persistence.
- `benchmark_attention.py`: Microbenchmark measuring scaling across $B \in \{1, 2, 4\}$ and $T \in [64, 4096]$.
