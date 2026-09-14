#include "liquid_state_fusion.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

#define LSF_BLOCK_D 64

// ---------------------------------------------------------------------------
// 1. Compute Mean and Variance of MoE Output (M, C)
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void compute_lsf_mean_var_kernel(
    const __nv_bfloat16* __restrict__ x,
    int N,
    float* __restrict__ mean_var_out // [0]: sum, [1]: sum_sq
) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 8;
    int stride = gridDim.x * BLOCK_SIZE * 8;
    
    for (int i = idx; i + 7 < N; i += stride) {
        uint4 raw = *reinterpret_cast<const uint4*>(x + i);
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(p[k]);
            sum += val;
            sum_sq += val * val;
        }
    }
    
    int rem_base = (N / 8) * 8;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (rem_base + tid < N) {
        float val = __bfloat162float(x[rem_base + tid]);
        sum += val;
        sum_sq += val * val;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    __shared__ float s_sum[32];
    __shared__ float s_sum_sq[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) {
        s_sum[wid] = sum;
        s_sum_sq[wid] = sum_sq;
    }
    __syncthreads();
    
    if (wid == 0) {
        float b_sum = (lane < (BLOCK_SIZE / 32)) ? s_sum[lane] : 0.0f;
        float b_sum_sq = (lane < (BLOCK_SIZE / 32)) ? s_sum_sq[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            b_sum += __shfl_down_sync(0xffffffff, b_sum, offset);
            b_sum_sq += __shfl_down_sync(0xffffffff, b_sum_sq, offset);
        }
        if (lane == 0) {
            atomicAdd(&mean_var_out[0], b_sum);
            atomicAdd(&mean_var_out[1], b_sum_sq);
        }
    }
}

__global__ void finalize_lsf_mean_var_kernel(float* mean_var_out, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = mean_var_out[0];
        float sum_sq = mean_var_out[1];
        float mean = sum / (float)N;
        float var = (sum_sq / (float)N) - (mean * mean);
        if (var < 0.0f) var = 0.0f;
        mean_var_out[0] = mean;
        mean_var_out[1] = var;
    }
}

// ---------------------------------------------------------------------------
// 2. Liquid State Fusion Forward Kernel:
// H_t = \alpha \cdot H_{t-1} + (1 - \alpha) \cdot X_t
// ---------------------------------------------------------------------------
__global__ void lsf_fwd_kernel(
    const __nv_bfloat16* __restrict__ X,     // (B, T, D)
    __nv_bfloat16* __restrict__ H,           // (B, T, D)
    __nv_bfloat16* __restrict__ H_last,      // (B, D) or nullptr
    const __nv_bfloat16* __restrict__ H0,    // (B, D) or nullptr
    const float* __restrict__ var_scale,     // (1)
    const float* __restrict__ mean_var_in,   // [1] = act_var
    float* __restrict__ alpha_out,           // (1) computed alpha
    int B, int T, int D
) {
    int pid_d = blockIdx.x;
    int pid_b = blockIdx.y;
    int tid = threadIdx.x;
    int col = pid_d * LSF_BLOCK_D + tid;
    
    float act_var = mean_var_in[1];
    float s = *var_scale;
    float alpha_raw = 1.0f / (1.0f + expf(s * act_var));
    float alpha = 0.1f + (0.99f - 0.1f) * alpha_raw;
    float one_minus_alpha = 1.0f - alpha;
    
    if (tid == 0 && pid_d == 0 && pid_b == 0 && alpha_out != nullptr) {
        *alpha_out = alpha;
    }
    
    if (col >= D) return;
    
    float h = 0.0f;
    if (H0 != nullptr) {
        h = __bfloat162float(H0[(size_t)pid_b * D + col]);
    }
    
    size_t base_b = (size_t)pid_b * T * D + col;
    for (int t = 0; t < T; ++t) {
        size_t offset = base_b + (size_t)t * D;
        float x = __bfloat162float(X[offset]);
        h = alpha * h + one_minus_alpha * x;
        H[offset] = __float2bfloat16(h);
    }
    
    if (H_last != nullptr) {
        H_last[(size_t)pid_b * D + col] = __float2bfloat16(h);
    }
}

void launch_liquid_state_fusion_fwd(
    const __nv_bfloat16* X,
    const float* var_scale,
    __nv_bfloat16* H,
    __nv_bfloat16* H_last,
    const __nv_bfloat16* H0,
    float* alpha_buf,
    float* mean_var_buf,
    int B, int T, int C,
    cudaStream_t stream
) {
    int N = B * T * C;
    cudaMemsetAsync(mean_var_buf, 0, 2 * sizeof(float), stream);
    
    const int BLOCK = 256;
    int grid_stat = 64;
    compute_lsf_mean_var_kernel<BLOCK><<<grid_stat, BLOCK, 0, stream>>>(X, N, mean_var_buf);
    finalize_lsf_mean_var_kernel<<<1, 1, 0, stream>>>(mean_var_buf, N);
    
    int num_blocks_d = (C + LSF_BLOCK_D - 1) / LSF_BLOCK_D;
    dim3 grid_lsf(num_blocks_d, B);
    lsf_fwd_kernel<<<grid_lsf, LSF_BLOCK_D, 0, stream>>>(
        X, H, H_last, H0, var_scale, mean_var_buf, alpha_buf, B, T, C
    );
}

// ---------------------------------------------------------------------------
// 3. Liquid State Fusion Backward Kernel:
// \lambda_t = grad_H_t + \alpha \cdot \lambda_{t+1}
// grad_X_t = (1 - \alpha) \cdot \lambda_t
// d\alpha += \lambda_t \cdot (H_{t-1} - X_t)
// ---------------------------------------------------------------------------
__global__ void lsf_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_H, // (B, T, D)
    const __nv_bfloat16* __restrict__ X,      // (B, T, D)
    const __nv_bfloat16* __restrict__ H,      // (B, T, D)
    const __nv_bfloat16* __restrict__ H0,     // (B, D) or nullptr
    __nv_bfloat16* __restrict__ grad_X,       // (B, T, D)
    float* __restrict__ grad_alpha_blocks,    // (num_blocks)
    const float* __restrict__ alpha_ptr,      // (1)
    int B, int T, int D
) {
    int pid_d = blockIdx.x;
    int pid_b = blockIdx.y;
    int tid = threadIdx.x;
    int col = pid_d * LSF_BLOCK_D + tid;
    
    float alpha = *alpha_ptr;
    float one_minus_alpha = 1.0f - alpha;
    
    float lambda_val = 0.0f;
    float d_alpha_acc = 0.0f;
    
    if (col < D) {
        size_t base_b = (size_t)pid_b * T * D + col;
        for (int t = T - 1; t >= 0; --t) {
            size_t offset = base_b + (size_t)t * D;
            float gh = __bfloat162float(grad_H[offset]);
            lambda_val = lambda_val + gh;
            
            float gx = one_minus_alpha * lambda_val;
            grad_X[offset] = __float2bfloat16(gx);
            
            float h_prev = 0.0f;
            if (t > 0) {
                h_prev = __bfloat162float(H[base_b + (size_t)(t - 1) * D]);
            } else if (H0 != nullptr) {
                h_prev = __bfloat162float(H0[(size_t)pid_b * D + col]);
            }
            
            float x = __bfloat162float(X[offset]);
            d_alpha_acc += lambda_val * (h_prev - x);
            lambda_val = lambda_val * alpha;
        }
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        d_alpha_acc += __shfl_down_sync(0xffffffff, d_alpha_acc, offset);
    }
    
    __shared__ float s_dalpha[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_dalpha[wid] = d_alpha_acc;
    __syncthreads();
    
    if (wid == 0) {
        float b_dalpha = (lane < (LSF_BLOCK_D / 32)) ? s_dalpha[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            b_dalpha += __shfl_down_sync(0xffffffff, b_dalpha, offset);
        }
        if (lane == 0) {
            int num_blocks_d = (D + LSF_BLOCK_D - 1) / LSF_BLOCK_D;
            int block_id = pid_b * num_blocks_d + pid_d;
            grad_alpha_blocks[block_id] = b_dalpha;
        }
    }
}

__global__ void finalize_lsf_d_var_scale_kernel(
    const float* __restrict__ grad_alpha_blocks,
    int num_blocks,
    const float* __restrict__ var_scale,
    const float* __restrict__ mean_var_in,
    float* __restrict__ d_var_scale,
    float beta
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum_dalpha = 0.0f;
        for (int i = 0; i < num_blocks; ++i) {
            sum_dalpha += grad_alpha_blocks[i];
        }
        float act_var = mean_var_in[1];
        float s = *var_scale;
        float alpha_raw = 1.0f / (1.0f + expf(s * act_var));
        float d_alpha_ds = (0.99f - 0.1f) * alpha_raw * (1.0f - alpha_raw) * (-act_var);
        float grad = sum_dalpha * d_alpha_ds;
        
        if (beta == 0.0f) {
            *d_var_scale = grad;
        } else {
            *d_var_scale += grad;
        }
    }
}

void launch_liquid_state_fusion_bwd(
    const __nv_bfloat16* grad_H,
    const __nv_bfloat16* X,
    const __nv_bfloat16* H,
    const __nv_bfloat16* H0,
    const float* var_scale,
    const float* mean_var_buf,
    const float* alpha_buf,
    __nv_bfloat16* grad_X,
    float* d_var_scale,
    float* grad_alpha_buf,
    int B, int T, int C,
    float beta,
    cudaStream_t stream
) {
    int num_blocks_d = (C + LSF_BLOCK_D - 1) / LSF_BLOCK_D;
    int num_blocks = B * num_blocks_d;
    dim3 grid_lsf(num_blocks_d, B);
    
    lsf_bwd_kernel<<<grid_lsf, LSF_BLOCK_D, 0, stream>>>(
        grad_H, X, H, H0, grad_X, grad_alpha_buf, alpha_buf, B, T, C
    );
    
    finalize_lsf_d_var_scale_kernel<<<1, 1, 0, stream>>>(
        grad_alpha_buf, num_blocks, var_scale, mean_var_buf, d_var_scale, beta
    );
}
