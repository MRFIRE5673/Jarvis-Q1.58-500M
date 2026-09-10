// sparse_model.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <string>
#include <stdexcept>

#define CUDA_CHECK_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// 1. Metadata Generation Kernel (Deterministic Count, Offset & Maps)
// ---------------------------------------------------------------------------
// Computes expert counts, prefix offsets, scatter_map, gather_map, and gate_idx_map.
// Single-block implementation for M <= 8192 guarantees 100% deterministic FIFO ordering.
__global__ void moe_compute_metadata_kernel(
    const int64_t* __restrict__ topk_idx, // (N, K)
    int32_t* __restrict__ expert_counts,  // (E,)
    int32_t* __restrict__ expert_offsets, // (E + 1,)
    int32_t* __restrict__ scatter_map,   // (N, K)
    int32_t* __restrict__ gather_map,    // (M,)
    int32_t* __restrict__ gate_idx_map,  // (M,)
    int N, int K, int E
) {
    extern __shared__ int shared_mem[];
    int* s_counts = shared_mem;               // E elements
    int* s_offsets = s_counts + E;            // (E + 1) elements
    int* s_cur = s_offsets + (E + 1);         // E elements

    int tid = threadIdx.x;
    int M = N * K;

    // 1. Initialize shared memory
    if (tid < E) {
        s_counts[tid] = 0;
        s_cur[tid] = 0;
    }
    if (tid <= E) {
        s_offsets[tid] = 0;
    }
    __syncthreads();

    // 2. Count tokens per expert
    for (int m = tid; m < M; m += blockDim.x) {
        int e = (int)topk_idx[m];
        if (e >= 0 && e < E) {
            atomicAdd(&s_counts[e], 1);
        }
    }
    __syncthreads();

    // 3. Compute exclusive prefix sum on Thread 0
    if (tid == 0) {
        s_offsets[0] = 0;
        for (int e = 0; e < E; e++) {
            s_offsets[e + 1] = s_offsets[e] + s_counts[e];
            s_cur[e] = s_offsets[e]; // Initialize current pointer to start of each expert's partition
            expert_counts[e] = s_counts[e];
        }
        expert_offsets[E] = s_offsets[E];
    }
    __syncthreads();

    // Copy offsets to global memory
    if (tid <= E) {
        expert_offsets[tid] = s_offsets[tid];
    }
    __syncthreads();

    // 4. Deterministic sequential assignment of slots
    // Running on thread 0 guarantees 100% deterministic token ordering within each expert partition.
    if (tid == 0) {
        for (int m = 0; m < M; m++) {
            int e = (int)topk_idx[m];
            int slot = s_cur[e]++;
            int token_i = m / K;
            int rank_k = m % K;
            scatter_map[m] = slot;
            gather_map[slot] = token_i;
            gate_idx_map[slot] = rank_k;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Dispatch Gather Kernel (x -> dispatched_x)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void moe_dispatch_gather_kernel(
    const T* __restrict__ x,               // (N, C)
    const int32_t* __restrict__ gather_map,// (M,)
    T* __restrict__ dispatched_x,          // (M, C)
    int M, int C
) {
    int m = blockIdx.x; // Each block processes one row in dispatched_x
    if (m >= M) return;

    int src_token = gather_map[m];
    const T* src_row = x + (size_t)src_token * C;
    T* dst_row = dispatched_x + (size_t)m * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        dst_row[c] = src_row[c];
    }
}

// ---------------------------------------------------------------------------
// 3. Scatter & Combine Kernel (dispatched_y + gates -> out)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void moe_scatter_combine_kernel(
    const T* __restrict__ dispatched_y,    // (M, C)
    const T* __restrict__ topk_gates,      // (N, K)
    const int32_t* __restrict__ scatter_map,// (N, K)
    T* __restrict__ out,                   // (N, C)
    int N, int K, int C
) {
    int i = blockIdx.x; // Each block processes one output token i in [0, N-1]
    if (i >= N) return;

    // Load the K gate values and slots for this token into registers
    // Since K <= 8 (here K=2), small static arrays are optimal
    float g_val[8];
    int slot_val[8];

    for (int k = 0; k < K; k++) {
        int idx = i * K + k;
        slot_val[k] = scatter_map[idx];
        if constexpr (std::is_same_v<T, at::BFloat16>) {
            g_val[k] = static_cast<float>(reinterpret_cast<const c10::BFloat16*>(topk_gates)[idx]);
        } else {
            g_val[k] = static_cast<float>(topk_gates[idx]);
        }
    }

    T* out_row = out + (size_t)i * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            int slot = slot_val[k];
            const T* y_row = dispatched_y + (size_t)slot * C;
            float val;
            if constexpr (std::is_same_v<T, at::BFloat16>) {
                val = static_cast<float>(reinterpret_cast<const c10::BFloat16*>(y_row)[c]);
            } else {
                val = static_cast<float>(y_row[c]);
            }
            acc += g_val[k] * val;
        }

        if constexpr (std::is_same_v<T, at::BFloat16>) {
            reinterpret_cast<c10::BFloat16*>(out_row)[c] = c10::BFloat16(acc);
        } else {
            out_row[c] = static_cast<T>(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Backward: Scatter Backward Kernel (grad_out -> grad_dispatched_y & grad_gates)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void moe_scatter_backward_y_kernel(
    const T* __restrict__ grad_out,        // (N, C)
    const T* __restrict__ topk_gates,      // (N, K)
    const int32_t* __restrict__ gather_map,// (M,)
    const int32_t* __restrict__ gate_idx_map, // (M,)
    T* __restrict__ grad_dispatched_y,     // (M, C)
    int M, int K, int C
) {
    int m = blockIdx.x; // Each block processes row m of grad_dispatched_y
    if (m >= M) return;

    int src_token = gather_map[m];
    int rank_k = gate_idx_map[m];

    float gate;
    int g_idx = src_token * K + rank_k;
    if constexpr (std::is_same_v<T, at::BFloat16>) {
        gate = static_cast<float>(reinterpret_cast<const c10::BFloat16*>(topk_gates)[g_idx]);
    } else {
        gate = static_cast<float>(topk_gates[g_idx]);
    }

    const T* g_out_row = grad_out + (size_t)src_token * C;
    T* g_y_row = grad_dispatched_y + (size_t)m * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float go;
        if constexpr (std::is_same_v<T, at::BFloat16>) {
            go = static_cast<float>(reinterpret_cast<const c10::BFloat16*>(g_out_row)[c]);
            reinterpret_cast<c10::BFloat16*>(g_y_row)[c] = c10::BFloat16(gate * go);
        } else {
            go = static_cast<float>(g_out_row[c]);
            g_y_row[c] = static_cast<T>(gate * go);
        }
    }
}

template <typename T>
__global__ void moe_scatter_backward_gates_kernel(
    const T* __restrict__ grad_out,        // (N, C)
    const T* __restrict__ dispatched_y,    // (M, C)
    const int32_t* __restrict__ scatter_map,// (N, K)
    T* __restrict__ grad_topk_gates,       // (N, K)
    int N, int K, int C
) {
    int i = blockIdx.x;
    int k = blockIdx.y;
    if (i >= N || k >= K) return;

    int slot = scatter_map[i * K + k];
    const T* g_out_row = grad_out + (size_t)i * C;
    const T* y_row = dispatched_y + (size_t)slot * C;

    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float go, y;
        if constexpr (std::is_same_v<T, at::BFloat16>) {
            go = static_cast<float>(reinterpret_cast<const c10::BFloat16*>(g_out_row)[c]);
            y  = static_cast<float>(reinterpret_cast<const c10::BFloat16*>(y_row)[c]);
        } else {
            go = static_cast<float>(g_out_row[c]);
            y  = static_cast<float>(y_row[c]);
        }
        thread_sum += go * y;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        s_warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        float block_sum = (lane < num_warps) ? s_warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            int out_idx = i * K + k;
            if constexpr (std::is_same_v<T, at::BFloat16>) {
                reinterpret_cast<c10::BFloat16*>(grad_topk_gates)[out_idx] = c10::BFloat16(block_sum);
            } else {
                grad_topk_gates[out_idx] = static_cast<T>(block_sum);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Backward: Gather Backward Kernel (grad_dispatched_x -> grad_x)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void moe_gather_backward_x_kernel(
    const T* __restrict__ grad_dispatched_x, // (M, C)
    const int32_t* __restrict__ scatter_map, // (N, K)
    T* __restrict__ grad_x,                  // (N, C)
    int N, int K, int C
) {
    int i = blockIdx.x; // Each block processes output token i in [0, N-1]
    if (i >= N) return;

    int slot_val[8];
    for (int k = 0; k < K; k++) {
        slot_val[k] = scatter_map[i * K + k];
    }

    T* gx_row = grad_x + (size_t)i * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            int slot = slot_val[k];
            const T* gdx_row = grad_dispatched_x + (size_t)slot * C;
            float val;
            if constexpr (std::is_same_v<T, at::BFloat16>) {
                val = static_cast<float>(reinterpret_cast<const c10::BFloat16*>(gdx_row)[c]);
            } else {
                val = static_cast<float>(gdx_row[c]);
            }
            acc += val;
        }

        if constexpr (std::is_same_v<T, at::BFloat16>) {
            reinterpret_cast<c10::BFloat16*>(gx_row)[c] = c10::BFloat16(acc);
        } else {
            gx_row[c] = static_cast<T>(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// C++ Host Invocation Wrappers
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> moe_compute_metadata_cuda(
    torch::Tensor topk_idx, int num_experts
) {
    int N = topk_idx.size(0);
    int K = topk_idx.size(1);
    int M = N * K;
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_idx.device());

    auto expert_counts = torch::empty({num_experts}, options_i32);
    auto expert_offsets = torch::empty({num_experts + 1}, options_i32);
    auto scatter_map = torch::empty({N, K}, options_i32);
    auto gather_map = torch::empty({M}, options_i32);
    auto gate_idx_map = torch::empty({M}, options_i32);

    int shared_bytes = (num_experts + (num_experts + 1) + num_experts) * sizeof(int);

    moe_compute_metadata_kernel<<<1, 256, shared_bytes>>>(
        topk_idx.data_ptr<int64_t>(),
        expert_counts.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        scatter_map.data_ptr<int32_t>(),
        gather_map.data_ptr<int32_t>(),
        gate_idx_map.data_ptr<int32_t>(),
        N, K, num_experts
    );
    CUDA_CHECK_LAUNCH();

    return {expert_counts, expert_offsets, scatter_map, gather_map, gate_idx_map};
}

torch::Tensor moe_dispatch_gather_cuda(
    torch::Tensor x, torch::Tensor gather_map
) {
    int N = x.size(0);
    int C = x.size(1);
    int M = gather_map.size(0);

    auto dispatched_x = torch::empty({M, C}, x.options());
    int block_size = 256;
    int grid_size = M;

    if (x.dtype() == torch::kFloat32) {
        moe_dispatch_gather_kernel<float><<<grid_size, block_size>>>(
            x.data_ptr<float>(),
            gather_map.data_ptr<int32_t>(),
            dispatched_x.data_ptr<float>(),
            M, C
        );
    } else if (x.dtype() == torch::kBFloat16) {
        moe_dispatch_gather_kernel<at::BFloat16><<<grid_size, block_size>>>(
            x.data_ptr<at::BFloat16>(),
            gather_map.data_ptr<int32_t>(),
            dispatched_x.data_ptr<at::BFloat16>(),
            M, C
        );
    }
    CUDA_CHECK_LAUNCH();

    return dispatched_x;
}

torch::Tensor moe_scatter_combine_cuda(
    torch::Tensor dispatched_y, torch::Tensor topk_gates, torch::Tensor scatter_map
) {
    int N = topk_gates.size(0);
    int K = topk_gates.size(1);
    int M = dispatched_y.size(0);
    int C = dispatched_y.size(1);

    auto out = torch::empty({N, C}, dispatched_y.options());
    int block_size = 256;
    int grid_size = N;

    if (dispatched_y.dtype() == torch::kFloat32) {
        moe_scatter_combine_kernel<float><<<grid_size, block_size>>>(
            dispatched_y.data_ptr<float>(),
            topk_gates.data_ptr<float>(),
            scatter_map.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            N, K, C
        );
    } else if (dispatched_y.dtype() == torch::kBFloat16) {
        moe_scatter_combine_kernel<at::BFloat16><<<grid_size, block_size>>>(
            dispatched_y.data_ptr<at::BFloat16>(),
            topk_gates.data_ptr<at::BFloat16>(),
            scatter_map.data_ptr<int32_t>(),
            out.data_ptr<at::BFloat16>(),
            N, K, C
        );
    }
    CUDA_CHECK_LAUNCH();

    return out;
}

std::vector<torch::Tensor> moe_scatter_backward_cuda(
    torch::Tensor grad_out, torch::Tensor dispatched_y, torch::Tensor topk_gates,
    torch::Tensor gather_map, torch::Tensor gate_idx_map, torch::Tensor scatter_map
) {
    int N = topk_gates.size(0);
    int K = topk_gates.size(1);
    int M = dispatched_y.size(0);
    int C = dispatched_y.size(1);

    auto grad_dispatched_y = torch::empty({M, C}, dispatched_y.options());
    auto grad_topk_gates = torch::empty({N, K}, topk_gates.options());

    int block_size = 256;

    // 1. grad_dispatched_y
    if (grad_out.dtype() == torch::kFloat32) {
        moe_scatter_backward_y_kernel<float><<<M, block_size>>>(
            grad_out.data_ptr<float>(),
            topk_gates.data_ptr<float>(),
            gather_map.data_ptr<int32_t>(),
            gate_idx_map.data_ptr<int32_t>(),
            grad_dispatched_y.data_ptr<float>(),
            M, K, C
        );
    } else if (grad_out.dtype() == torch::kBFloat16) {
        moe_scatter_backward_y_kernel<at::BFloat16><<<M, block_size>>>(
            grad_out.data_ptr<at::BFloat16>(),
            topk_gates.data_ptr<at::BFloat16>(),
            gather_map.data_ptr<int32_t>(),
            gate_idx_map.data_ptr<int32_t>(),
            grad_dispatched_y.data_ptr<at::BFloat16>(),
            M, K, C
        );
    }
    CUDA_CHECK_LAUNCH();

    // 2. grad_topk_gates
    dim3 grid_gates(N, K);
    if (grad_out.dtype() == torch::kFloat32) {
        moe_scatter_backward_gates_kernel<float><<<grid_gates, block_size>>>(
            grad_out.data_ptr<float>(),
            dispatched_y.data_ptr<float>(),
            scatter_map.data_ptr<int32_t>(),
            grad_topk_gates.data_ptr<float>(),
            N, K, C
        );
    } else if (grad_out.dtype() == torch::kBFloat16) {
        moe_scatter_backward_gates_kernel<at::BFloat16><<<grid_gates, block_size>>>(
            grad_out.data_ptr<at::BFloat16>(),
            dispatched_y.data_ptr<at::BFloat16>(),
            scatter_map.data_ptr<int32_t>(),
            grad_topk_gates.data_ptr<at::BFloat16>(),
            N, K, C
        );
    }
    CUDA_CHECK_LAUNCH();

    return {grad_dispatched_y, grad_topk_gates};
}

torch::Tensor moe_gather_backward_cuda(
    torch::Tensor grad_dispatched_x, torch::Tensor scatter_map, int N
) {
    int K = scatter_map.size(1);
    int C = grad_dispatched_x.size(1);

    auto grad_x = torch::empty({N, C}, grad_dispatched_x.options());
    int block_size = 256;
    int grid_size = N;

    if (grad_dispatched_x.dtype() == torch::kFloat32) {
        moe_gather_backward_x_kernel<float><<<grid_size, block_size>>>(
            grad_dispatched_x.data_ptr<float>(),
            scatter_map.data_ptr<int32_t>(),
            grad_x.data_ptr<float>(),
            N, K, C
        );
    } else if (grad_dispatched_x.dtype() == torch::kBFloat16) {
        moe_gather_backward_x_kernel<at::BFloat16><<<grid_size, block_size>>>(
            grad_dispatched_x.data_ptr<at::BFloat16>(),
            scatter_map.data_ptr<int32_t>(),
            grad_x.data_ptr<at::BFloat16>(),
            N, K, C
        );
    }
    CUDA_CHECK_LAUNCH();

    return grad_x;
}
