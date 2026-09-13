#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// ---------------------------------------------------------------------------
// 1. Ternary AbsMean Reduction Kernel
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void ternary_abs_mean_kernel(
    const __nv_bfloat16* __restrict__ w,
    int N,
    float* __restrict__ alpha_out
) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;
    float sum_abs = 0.0f;
    
    for (int i = idx; i < N; i += stride) {
        sum_abs += fabsf(__bfloat162float(w[i]));
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_abs += __shfl_down_sync(0xffffffff, sum_abs, offset);
    }
    
    __shared__ float s_sum[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) s_sum[wid] = sum_abs;
    __syncthreads();
    
    float block_sum = 0.0f;
    if (wid == 0) {
        block_sum = (lane < (BLOCK_SIZE / 32)) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(alpha_out, block_sum / (float)N);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Fused Ternary Quantize Forward Kernel: W -> W_q
// ---------------------------------------------------------------------------
__global__ void ternary_quantize_fwd_kernel(
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ w_q,
    const float* __restrict__ alpha_ptr,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float alpha = fmaxf(*alpha_ptr, 1e-8f);
    float val = __bfloat162float(w[idx]);
    float val_norm = val / alpha;
    float clamped = fminf(fmaxf(val_norm, -1.0f), 1.0f);
    float rounded = nearbyintf(clamped);
    w_q[idx] = __float2bfloat16(rounded * alpha);
}

// ---------------------------------------------------------------------------
// 3. Fused Ternary Quantize Backward Kernel (STE)
// ---------------------------------------------------------------------------
__global__ void ternary_quantize_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ grad_w,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float val = __bfloat162float(w[idx]);
    float go = __bfloat162float(grad_out[idx]);
    float gw = (fabsf(val) <= 1.0f) ? go : 0.0f;
    grad_w[idx] = __float2bfloat16(gw);
}

void launch_ternary_quantize_fwd(
    const __nv_bfloat16* w,
    __nv_bfloat16* w_q,
    const float* alpha_ptr,
    int N,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    ternary_quantize_fwd_kernel<<<grid, BLOCK, 0, stream>>>(w, w_q, alpha_ptr, N);
}

void launch_ternary_quantize_bwd(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* w,
    __nv_bfloat16* grad_w,
    int N,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    ternary_quantize_bwd_kernel<<<grid, BLOCK, 0, stream>>>(grad_out, w, grad_w, N);
}
