// liquid_state_fusion_cuda.cu
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>

#define BLOCK_SIZE 128

#define CUDA_CHECK_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

__global__ void liquid_state_fusion_forward_local_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ M,
    const float* __restrict__ h0,
    float* __restrict__ temp_a,
    float* __restrict__ temp_b,
    float* __restrict__ block_sums_a,
    float* __restrict__ block_sums_b,
    int B, int T, int D, int num_chunks
) {
    int channel = blockIdx.x;
    int b = channel / D;
    int d = channel % D;
    int chunk_idx = blockIdx.y;

    int start = chunk_idx * BLOCK_SIZE;
    int end = start + BLOCK_SIZE;
    int t = start + threadIdx.x;

    __shared__ float sh_a[BLOCK_SIZE];
    __shared__ float sh_b[BLOCK_SIZE];

    float a_val = 1.0f;
    float b_val = 0.0f;

    if (t < T) {
        int offset = b * T * D + t * D + d;
        a_val = alpha[offset];
        b_val = (1.0f - a_val) * M[offset];
        if (t == 0 && h0 != nullptr) {
            b_val += a_val * h0[b * D + d];
        }
    }

    sh_a[threadIdx.x] = a_val;
    sh_b[threadIdx.x] = b_val;

    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        __syncthreads();
        float prev_a = 1.0f;
        float prev_b = 0.0f;
        if (threadIdx.x >= offset) {
            prev_a = sh_a[threadIdx.x - offset];
            prev_b = sh_b[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x >= offset) {
            sh_b[threadIdx.x] = sh_a[threadIdx.x] * prev_b + sh_b[threadIdx.x];
            sh_a[threadIdx.x] = sh_a[threadIdx.x] * prev_a;
        }
    }

    __syncthreads();

    if (t < T) {
        int offset = b * T * D + t * D + d;
        temp_a[offset] = sh_a[threadIdx.x];
        temp_b[offset] = sh_b[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        int last_valid = (end > T) ? (T - start - 1) : (BLOCK_SIZE - 1);
        int out_idx = channel * num_chunks + chunk_idx;
        block_sums_a[out_idx] = sh_a[last_valid];
        block_sums_b[out_idx] = sh_b[last_valid];
    }
}

__global__ void liquid_state_fusion_scan_block_sums_kernel(
    float* __restrict__ block_sums_a,
    float* __restrict__ block_sums_b,
    int B, int D, int num_chunks, int num_chunks_pow2
) {
    int channel = blockIdx.x;
    int t = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* sh_a = shared_mem;
    float* sh_b = &shared_mem[num_chunks_pow2];

    float a_val = 1.0f;
    float b_val = 0.0f;

    if (t < num_chunks) {
        int idx = channel * num_chunks + t;
        a_val = block_sums_a[idx];
        b_val = block_sums_b[idx];
    }

    sh_a[t] = a_val;
    sh_b[t] = b_val;

    for (int offset = 1; offset < num_chunks_pow2; offset *= 2) {
        __syncthreads();
        float prev_a = 1.0f;
        float prev_b = 0.0f;
        if (t >= offset) {
            prev_a = sh_a[t - offset];
            prev_b = sh_b[t - offset];
        }
        __syncthreads();
        if (t >= offset) {
            sh_b[t] = sh_a[t] * prev_b + sh_b[t];
            sh_a[t] = sh_a[t] * prev_a;
        }
    }

    __syncthreads();

    if (t < num_chunks) {
        int idx = channel * num_chunks + t;
        block_sums_a[idx] = sh_a[t];
        block_sums_b[idx] = sh_b[t];
    }
}

__global__ void liquid_state_fusion_forward_apply_block_sums_kernel(
    const float* __restrict__ temp_a,
    const float* __restrict__ temp_b,
    const float* __restrict__ block_sums_a,
    const float* __restrict__ block_sums_b,
    float* __restrict__ H,
    int B, int T, int D, int num_chunks
) {
    int channel = blockIdx.x;
    int b = channel / D;
    int d = channel % D;
    int chunk_idx = blockIdx.y;
    int t = chunk_idx * BLOCK_SIZE + threadIdx.x;

    if (t >= T) return;

    int offset = b * T * D + t * D + d;
    float local_a = temp_a[offset];
    float local_b = temp_b[offset];

    if (chunk_idx == 0) {
        H[offset] = local_b;
    } else {
        int pref_idx = channel * num_chunks + (chunk_idx - 1);
        float pref_b = block_sums_b[pref_idx];
        H[offset] = local_a * pref_b + local_b;
    }
}

__global__ void liquid_state_fusion_backward_local_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ grad_H,
    float* __restrict__ temp_a_rev,
    float* __restrict__ temp_b_rev,
    float* __restrict__ block_sums_a_rev,
    float* __restrict__ block_sums_b_rev,
    int B, int T, int D, int num_chunks
) {
    int channel = blockIdx.x;
    int b = channel / D;
    int d = channel % D;
    int chunk_idx = blockIdx.y;

    int start = chunk_idx * BLOCK_SIZE;
    int end = start + BLOCK_SIZE;

    int t_rev = start + threadIdx.x;
    int t = T - 1 - t_rev;

    __shared__ float sh_a[BLOCK_SIZE];
    __shared__ float sh_b[BLOCK_SIZE];

    float a_val = 1.0f;
    float b_val = 0.0f;

    if (t >= 0 && t < T) {
        int offset = b * T * D + t * D + d;
        a_val = (t < T - 1) ? alpha[b * T * D + (t + 1) * D + d] : 0.0f;
        b_val = grad_H[offset];
    }

    sh_a[threadIdx.x] = a_val;
    sh_b[threadIdx.x] = b_val;

    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        __syncthreads();
        float prev_a = 1.0f;
        float prev_b = 0.0f;
        if (threadIdx.x >= offset) {
            prev_a = sh_a[threadIdx.x - offset];
            prev_b = sh_b[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x >= offset) {
            sh_b[threadIdx.x] = sh_a[threadIdx.x] * prev_b + sh_b[threadIdx.x];
            sh_a[threadIdx.x] = sh_a[threadIdx.x] * prev_a;
        }
    }

    __syncthreads();

    if (t >= 0 && t < T) {
        int offset = b * T * D + t * D + d;
        temp_a_rev[offset] = sh_a[threadIdx.x];
        temp_b_rev[offset] = sh_b[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        int last_valid = (end > T) ? (T - start - 1) : (BLOCK_SIZE - 1);
        int out_idx = channel * num_chunks + chunk_idx;
        block_sums_a_rev[out_idx] = sh_a[last_valid];
        block_sums_b_rev[out_idx] = sh_b[last_valid];
    }
}

__global__ void liquid_state_fusion_backward_apply_block_sums_kernel(
    const float* __restrict__ temp_a_rev,
    const float* __restrict__ temp_b_rev,
    const float* __restrict__ block_sums_a_rev,
    const float* __restrict__ block_sums_b_rev,
    float* __restrict__ dH,
    int B, int T, int D, int num_chunks
) {
    int channel = blockIdx.x;
    int b = channel / D;
    int d = channel % D;
    int chunk_idx = blockIdx.y;

    int t_rev = chunk_idx * BLOCK_SIZE + threadIdx.x;
    int t = T - 1 - t_rev;

    if (t < 0 || t >= T) return;

    int offset = b * T * D + t * D + d;
    float local_a = temp_a_rev[offset];
    float local_b = temp_b_rev[offset];

    if (chunk_idx == 0) {
        dH[offset] = local_b;
    } else {
        int pref_idx = channel * num_chunks + (chunk_idx - 1);
        float pref_b = block_sums_b_rev[pref_idx];
        dH[offset] = local_a * pref_b + local_b;
    }
}

__global__ void liquid_state_fusion_backward_grads_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ M,
    const float* __restrict__ h0,
    const float* __restrict__ H,
    const float* __restrict__ dH,
    float* __restrict__ grad_alpha,
    float* __restrict__ grad_M,
    float* __restrict__ grad_h0,
    int B, int T, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * D;
    if (idx >= total_elements) return;

    int b = idx / (T * D);
    int t = (idx % (T * D)) / D;
    int d = idx % D;

    float dH_val = dH[idx];
    float alpha_val = alpha[idx];
    float M_val = M[idx];

    grad_M[idx] = dH_val * (1.0f - alpha_val);

    float h_prev = 0.0f;
    if (t == 0) {
        if (h0 != nullptr) h_prev = h0[b * D + d];
    } else {
        h_prev = H[b * T * D + (t - 1) * D + d];
    }

    grad_alpha[idx] = dH_val * (h_prev - M_val);

    if (t == 0 && grad_h0 != nullptr) {
        grad_h0[b * D + d] = dH_val * alpha_val;
    }
}

template <typename scalar_t>
__global__ void liquid_state_fusion_forward_coalesced_kernel(
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ M,
    const scalar_t* __restrict__ h0,
    scalar_t* __restrict__ H,
    int B, int T, int D
) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = B * D;
    if (channel >= total_channels) return;

    int b = channel / D;
    int d = channel % D;

    float h_val = 0.0f;
    if (h0 != nullptr) {
        h_val = static_cast<float>(h0[b * D + d]);
    }

    int base_offset = b * T * D + d;

    #pragma unroll 4
    for (int t = 0; t < T; ++t) {
        int idx = base_offset + t * D;
        float a_val = static_cast<float>(alpha[idx]);
        float m_val = static_cast<float>(M[idx]);
        h_val = a_val * h_val + (1.0f - a_val) * m_val;
        H[idx] = static_cast<scalar_t>(h_val);
    }
}

template <typename scalar_t>
__global__ void liquid_state_fusion_backward_coalesced_kernel(
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ M,
    const scalar_t* __restrict__ h0,
    const scalar_t* __restrict__ H,
    const scalar_t* __restrict__ grad_H,
    scalar_t* __restrict__ grad_alpha,
    scalar_t* __restrict__ grad_M,
    scalar_t* __restrict__ grad_h0,
    int B, int T, int D
) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = B * D;
    if (channel >= total_channels) return;

    int b = channel / D;
    int d = channel % D;
    int base_offset = b * T * D + d;

    if (T == 1) {
        int idx = base_offset;
        float gH = static_cast<float>(grad_H[idx]);
        float a_val = static_cast<float>(alpha[idx]);
        float m_val = static_cast<float>(M[idx]);
        float h_prev = (h0 != nullptr) ? static_cast<float>(h0[b * D + d]) : 0.0f;
        grad_M[idx] = static_cast<scalar_t>(gH * (1.0f - a_val));
        grad_alpha[idx] = static_cast<scalar_t>(gH * (h_prev - m_val));
        if (grad_h0 != nullptr) {
            grad_h0[b * D + d] = static_cast<scalar_t>(gH * a_val);
        }
        return;
    }

    // Peeling iteration t = T - 1 (first iteration of reverse pass)
    int idx_last = base_offset + (T - 1) * D;
    float dH_val = static_cast<float>(grad_H[idx_last]);
    float a_val = static_cast<float>(alpha[idx_last]);
    float m_val = static_cast<float>(M[idx_last]);
    float next_alpha = a_val;
    grad_M[idx_last] = static_cast<scalar_t>(dH_val * (1.0f - a_val));
    float h_prev = static_cast<float>(H[base_offset + (T - 2) * D]);
    grad_alpha[idx_last] = static_cast<scalar_t>(dH_val * (h_prev - m_val));

    // Branch-free main loop: t = T - 2 down to 1
    #pragma unroll 2
    for (int t = T - 2; t >= 1; --t) {
        int idx = base_offset + t * D;
        float gH = static_cast<float>(grad_H[idx]);
        dH_val = gH + next_alpha * dH_val;

        a_val = static_cast<float>(alpha[idx]);
        m_val = static_cast<float>(M[idx]);
        next_alpha = a_val;

        grad_M[idx] = static_cast<scalar_t>(dH_val * (1.0f - a_val));
        h_prev = static_cast<float>(H[base_offset + (t - 1) * D]);
        grad_alpha[idx] = static_cast<scalar_t>(dH_val * (h_prev - m_val));
    }

    // Peeling iteration t = 0 (last iteration of reverse pass)
    int idx_0 = base_offset;
    float gH_0 = static_cast<float>(grad_H[idx_0]);
    dH_val = gH_0 + next_alpha * dH_val;
    a_val = static_cast<float>(alpha[idx_0]);
    m_val = static_cast<float>(M[idx_0]);
    grad_M[idx_0] = static_cast<scalar_t>(dH_val * (1.0f - a_val));
    h_prev = (h0 != nullptr) ? static_cast<float>(h0[b * D + d]) : 0.0f;
    grad_alpha[idx_0] = static_cast<scalar_t>(dH_val * (h_prev - m_val));
    if (grad_h0 != nullptr) {
        grad_h0[b * D + d] = static_cast<scalar_t>(dH_val * a_val);
    }
}

template <typename scalar_t, int VecSize>
struct alignas(sizeof(scalar_t) * VecSize) AlignedVec {
    scalar_t val[VecSize];
};

template <typename scalar_t, int V>
__global__ void liquid_state_fusion_forward_vec_kernel(
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ M,
    const scalar_t* __restrict__ h0,
    scalar_t* __restrict__ H,
    int B, int T, int D
) {
    int total_vec_channels = (B * D) / V;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vec_channels) return;

    int D_vec = D / V;
    int b = idx / D_vec;
    int d_v = idx % D_vec;
    int d_start = d_v * V;

    float h_val[V];
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        h_val[v] = (h0 != nullptr) ? static_cast<float>(h0[b * D + d_start + v]) : 0.0f;
    }

    using Vec_t = AlignedVec<scalar_t, V>;

    for (int t = 0; t < T; ++t) {
        int offset = b * T * D + t * D + d_start;
        Vec_t a_vec = *reinterpret_cast<const Vec_t*>(&alpha[offset]);
        Vec_t m_vec = *reinterpret_cast<const Vec_t*>(&M[offset]);
        Vec_t out_vec;

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            float a = static_cast<float>(a_vec.val[v]);
            float m = static_cast<float>(m_vec.val[v]);
            h_val[v] = a * h_val[v] + (1.0f - a) * m;
            out_vec.val[v] = static_cast<scalar_t>(h_val[v]);
        }

        *reinterpret_cast<Vec_t*>(&H[offset]) = out_vec;
    }
}

template <typename scalar_t, int V>
__global__ void liquid_state_fusion_backward_vec_kernel(
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ M,
    const scalar_t* __restrict__ h0,
    const scalar_t* __restrict__ H,
    const scalar_t* __restrict__ grad_H,
    scalar_t* __restrict__ grad_alpha,
    scalar_t* __restrict__ grad_M,
    scalar_t* __restrict__ grad_h0,
    int B, int T, int D
) {
    int total_vec_channels = (B * D) / V;
    int channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel_idx >= total_vec_channels) return;

    int D_vec = D / V;
    int b = channel_idx / D_vec;
    int d_v = channel_idx % D_vec;
    int d_start = d_v * V;

    using Vec_t = AlignedVec<scalar_t, V>;

    float dH_val[V];
    float next_alpha[V];
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        dH_val[v] = 0.0f;
        next_alpha[v] = 0.0f;
    }

    for (int t = T - 1; t >= 0; --t) {
        int offset = b * T * D + t * D + d_start;
        Vec_t gH_vec = *reinterpret_cast<const Vec_t*>(&grad_H[offset]);
        Vec_t a_vec = *reinterpret_cast<const Vec_t*>(&alpha[offset]);
        Vec_t m_vec = *reinterpret_cast<const Vec_t*>(&M[offset]);
        Vec_t h_prev_vec;
        if (t > 0) {
            h_prev_vec = *reinterpret_cast<const Vec_t*>(&H[b * T * D + (t - 1) * D + d_start]);
        }

        Vec_t grad_M_vec;
        Vec_t grad_a_vec;

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            float gH = static_cast<float>(gH_vec.val[v]);
            if (t == T - 1) {
                dH_val[v] = gH;
            } else {
                dH_val[v] = gH + next_alpha[v] * dH_val[v];
            }

            float a_val = static_cast<float>(a_vec.val[v]);
            float m_val = static_cast<float>(m_vec.val[v]);
            next_alpha[v] = a_val; // Cached in register for timestep t - 1

            grad_M_vec.val[v] = static_cast<scalar_t>(dH_val[v] * (1.0f - a_val));

            float h_prev = 0.0f;
            if (t == 0) {
                if (h0 != nullptr) h_prev = static_cast<float>(h0[b * D + d_start + v]);
            } else {
                h_prev = static_cast<float>(h_prev_vec.val[v]);
            }

            grad_a_vec.val[v] = static_cast<scalar_t>(dH_val[v] * (h_prev - m_val));

            if (t == 0 && grad_h0 != nullptr) {
                grad_h0[b * D + d_start + v] = static_cast<scalar_t>(dH_val[v] * a_val);
            }
        }

        *reinterpret_cast<Vec_t*>(&grad_M[offset]) = grad_M_vec;
        *reinterpret_cast<Vec_t*>(&grad_alpha[offset]) = grad_a_vec;
    }
}

torch::Tensor liquid_state_fusion_forward_cuda(torch::Tensor alpha, torch::Tensor M, torch::Tensor h0, int threads_per_block = 64) {
    int B = alpha.size(0); int T = alpha.size(1); int D = alpha.size(2);
    auto H = torch::empty_like(M);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, alpha.scalar_type(), "liquid_state_fusion_forward_cuda", ([&] {
        if (D % 2 == 0) {
            int total_vec_channels = (B * D) / 2;
            int blocks = (total_vec_channels + threads_per_block - 1) / threads_per_block;
            liquid_state_fusion_forward_vec_kernel<scalar_t, 2><<<blocks, threads_per_block, 0, stream>>>(
                alpha.data_ptr<scalar_t>(), M.data_ptr<scalar_t>(),
                h0.defined() ? h0.data_ptr<scalar_t>() : nullptr,
                H.data_ptr<scalar_t>(),
                B, T, D
            );
        } else {
            int total_channels = B * D;
            int blocks = (total_channels + threads_per_block - 1) / threads_per_block;
            liquid_state_fusion_forward_coalesced_kernel<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
                alpha.data_ptr<scalar_t>(), M.data_ptr<scalar_t>(),
                h0.defined() ? h0.data_ptr<scalar_t>() : nullptr,
                H.data_ptr<scalar_t>(),
                B, T, D
            );
        }
    }));
    CUDA_CHECK_LAUNCH();

    return H;
}

std::vector<torch::Tensor> liquid_state_fusion_backward_cuda(
    torch::Tensor alpha, torch::Tensor M, torch::Tensor h0, torch::Tensor H, torch::Tensor grad_H, int threads_per_block = 64
) {
    int B = alpha.size(0); int T = alpha.size(1); int D = alpha.size(2);
    auto grad_alpha = torch::empty_like(alpha);
    auto grad_M = torch::empty_like(M);
    auto grad_h0 = h0.defined() ? torch::empty_like(h0) : torch::Tensor();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, alpha.scalar_type(), "liquid_state_fusion_backward_cuda", ([&] {
        int total_channels = B * D;
        int blocks = (total_channels + threads_per_block - 1) / threads_per_block;
        liquid_state_fusion_backward_coalesced_kernel<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
            alpha.data_ptr<scalar_t>(), M.data_ptr<scalar_t>(),
            h0.defined() ? h0.data_ptr<scalar_t>() : nullptr,
            H.data_ptr<scalar_t>(), grad_H.data_ptr<scalar_t>(),
            grad_alpha.data_ptr<scalar_t>(), grad_M.data_ptr<scalar_t>(),
            h0.defined() ? grad_h0.data_ptr<scalar_t>() : nullptr,
            B, T, D
        );
    }));
    CUDA_CHECK_LAUNCH();

    return {grad_alpha, grad_M, grad_h0};
}

py::dict get_device_occupancy_cuda(int block_size) {
    py::dict res;
    int blocks_fwd_bf16 = 0;
    int blocks_bwd_bf16 = 0;
    int blocks_fwd_fp32 = 0;
    int blocks_bwd_fp32 = 0;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_fwd_bf16,
        liquid_state_fusion_forward_vec_kernel<c10::BFloat16, 2>,
        block_size,
        0
    );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_bwd_bf16,
        liquid_state_fusion_backward_vec_kernel<c10::BFloat16, 2>,
        block_size,
        0
    );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_fwd_fp32,
        liquid_state_fusion_forward_vec_kernel<float, 2>,
        block_size,
        0
    );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_bwd_fp32,
        liquid_state_fusion_backward_vec_kernel<float, 2>,
        block_size,
        0
    );

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    res["device_name"] = std::string(prop.name);
    res["sm_count"] = prop.multiProcessorCount;
    res["max_threads_per_sm"] = prop.maxThreadsPerMultiProcessor;
    res["fwd_bf16_blocks_per_sm"] = blocks_fwd_bf16;
    res["bwd_bf16_blocks_per_sm"] = blocks_bwd_bf16;
    res["fwd_fp32_blocks_per_sm"] = blocks_fwd_fp32;
    res["bwd_fp32_blocks_per_sm"] = blocks_bwd_fp32;
    res["fwd_bf16_occupancy_pct"] = (double)(blocks_fwd_bf16 * block_size) / prop.maxThreadsPerMultiProcessor * 100.0;
    res["bwd_bf16_occupancy_pct"] = (double)(blocks_bwd_bf16 * block_size) / prop.maxThreadsPerMultiProcessor * 100.0;
    res["fwd_fp32_occupancy_pct"] = (double)(blocks_fwd_fp32 * block_size) / prop.maxThreadsPerMultiProcessor * 100.0;
    res["bwd_fp32_occupancy_pct"] = (double)(blocks_bwd_fp32 * block_size) / prop.maxThreadsPerMultiProcessor * 100.0;

    return res;
}
