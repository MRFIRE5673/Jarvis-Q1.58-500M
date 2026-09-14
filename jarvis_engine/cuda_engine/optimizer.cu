#include "optimizer.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void zero_grad_norm_kernel(float* grad_norm_sq) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *grad_norm_sq = 0.0f;
    }
}

void launch_zero_grad_norm(float* grad_norm_sq, cudaStream_t stream) {
    zero_grad_norm_kernel<<<1, 1, 0, stream>>>(grad_norm_sq);
}

// ---------------------------------------------------------------------------
// 1. Accumulate ||g||^2 across all parameters (128-bit Vec8 Vectorized)
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void accumulate_grad_norm_sq_kernel(
    const __nv_bfloat16* __restrict__ grad,
    int N,
    float* __restrict__ grad_norm_sq
) {
    int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 8;
    int stride = gridDim.x * BLOCK_SIZE * 8;
    float sum_sq = 0.0f;
    
    for (int i = idx; i + 7 < N; i += stride) {
        uint4 raw = *reinterpret_cast<const uint4*>(grad + i);
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(p[k]);
            sum_sq += val * val;
        }
    }
    
    // Remainder
    int rem_base = (N / 8) * 8;
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (rem_base + tid < N) {
        float val = __bfloat162float(grad[rem_base + tid]);
        sum_sq += val * val;
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    __shared__ float s_sum[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) s_sum[wid] = sum_sq;
    __syncthreads();
    
    float block_sum = 0.0f;
    if (wid == 0) {
        block_sum = (lane < (BLOCK_SIZE / 32)) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(grad_norm_sq, block_sum);
        }
    }
}

void launch_accumulate_grad_norm_sq(
    const __nv_bfloat16* grad,
    int num_elements,
    float* grad_norm_sq,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    int num_vecs = (num_elements + 7) / 8;
    int grid = (num_vecs + BLOCK - 1) / BLOCK;
    if (grid > 1024) grid = 1024;
    accumulate_grad_norm_sq_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(grad, num_elements, grad_norm_sq);
}

struct MultiGradRefs {
    const __nv_bfloat16* ptrs[16];
    int sizes[16];
    int num_refs;
};

template<int BLOCK_SIZE>
__global__ void accumulate_multi_grad_norm_sq_kernel(
    MultiGradRefs refs,
    float* __restrict__ grad_norm_sq
) {
    float sum_sq = 0.0f;
    int tid = threadIdx.x;
    int stride = BLOCK_SIZE * gridDim.x * 8;
    
    #pragma unroll 1
    for (int r = 0; r < refs.num_refs; ++r) {
        const __nv_bfloat16* grad = refs.ptrs[r];
        int N = refs.sizes[r];
        int idx = (blockIdx.x * BLOCK_SIZE + tid) * 8;
        
        for (int i = idx; i + 7 < N; i += stride) {
            uint4 raw = *reinterpret_cast<const uint4*>(grad + i);
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float val = __bfloat162float(p[k]);
                sum_sq += val * val;
            }
        }
        
        int rem_base = (N / 8) * 8;
        int rem_tid = blockIdx.x * BLOCK_SIZE + tid;
        if (rem_base + rem_tid < N) {
            float val = __bfloat162float(grad[rem_base + rem_tid]);
            sum_sq += val * val;
        }
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    __shared__ float s_sum[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) s_sum[wid] = sum_sq;
    __syncthreads();
    
    if (wid == 0) {
        float block_sum = (lane < (BLOCK_SIZE / 32)) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(grad_norm_sq, block_sum);
        }
    }
}

void launch_accumulate_layer_grad_norm_sq(
    const LayerWeights& lay,
    float* grad_norm_sq,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
) {
    // EXP-27-009: Only d_norm1_weight and d_qkv_weight receive actual BPTT gradients.
    // d_out_proj_weight, d_norm2_weight, d_router_weight, d_w1_weights, d_w2_weights
    // are ternary/frozen weights whose gradient buffers are always zero — including
    // them wastes ~408M element reads + atomicAdds per training step with zero benefit.
    MultiGradRefs refs;
    refs.ptrs[0] = lay.d_norm1_weight; refs.sizes[0] = cfg.C;
    refs.ptrs[1] = lay.d_qkv_weight;   refs.sizes[1] = 3 * cfg.C * cfg.C;
    refs.num_refs = 2;
    
    const int BLOCK = 256;
    const int GRID = 256;
    accumulate_multi_grad_norm_sq_kernel<BLOCK><<<GRID, BLOCK, 0, stream>>>(refs, grad_norm_sq);
}

void launch_accumulate_global_grad_norm_sq(
    const FullModelParameters& params,
    float* grad_norm_sq,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
) {
    MultiGradRefs refs;
    refs.ptrs[0] = params.d_final_norm_weight; refs.sizes[0] = cfg.C;
    refs.ptrs[1] = params.d_lm_head_weight;    refs.sizes[1] = cfg.vocab_pad * cfg.C;
    refs.num_refs = 2;
    
    const int BLOCK = 256;
    const int GRID = 256;
    accumulate_multi_grad_norm_sq_kernel<BLOCK><<<GRID, BLOCK, 0, stream>>>(refs, grad_norm_sq);
}


// ---------------------------------------------------------------------------
// 2. Compute Clipping Coefficient Once (Scalar Broadcast Kernel)
// ---------------------------------------------------------------------------
__global__ void compute_clip_coef_kernel(
    const float* __restrict__ grad_norm_sq,
    float* __restrict__ clip_coef,
    float max_norm
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total_norm = sqrtf(*grad_norm_sq);
        float coef = 1.0f;
        if (total_norm > max_norm) {
            coef = max_norm / (total_norm + 1e-6f);
        }
        *clip_coef = coef;
    }
}

void launch_compute_clip_coef(
    const float* grad_norm_sq,
    float* clip_coef,
    float max_norm,
    cudaStream_t stream
) {
    compute_clip_coef_kernel<<<1, 1, 0, stream>>>(grad_norm_sq, clip_coef, max_norm);
}

// ---------------------------------------------------------------------------
// 3. Fused AdamW Update Kernel (BF16 Parameters, 128-bit / 4-way Vectorized)
// Fuses:
// - Direct broadcasted clip_coef
// - First moment: m = beta1 * m + (1 - beta1) * (g * scale)
// ---------------------------------------------------------------------------
// 3. Fused AdamW BF16 Kernel (EXP-24-003: 128-bit Vec8 Vectorized)
// - Processes 8 bfloat16 elements per thread using pure 128-bit memory instructions
// - Reads: 128-bit param (uint4), 128-bit grad (uint4), 2x 128-bit m, 2x 128-bit v
// - Writes: 128-bit param, 128-bit zeroed grad, 2x 128-bit m, 2x 128-bit v
// ---------------------------------------------------------------------------
__global__ void fused_adamw_update_bf16_vec8_kernel(
    __nv_bfloat16* __restrict__ param,
    __nv_bfloat16* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int N
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    
    if (idx + 7 < N) {
        uint4* p_u4 = reinterpret_cast<uint4*>(param + idx);
        uint4* g_u4 = reinterpret_cast<uint4*>(grad + idx);
        float4* m_f4_0 = reinterpret_cast<float4*>(m + idx);
        float4* m_f4_1 = reinterpret_cast<float4*>(m + idx + 4);
        float4* v_f4_0 = reinterpret_cast<float4*>(v + idx);
        float4* v_f4_1 = reinterpret_cast<float4*>(v + idx + 4);
        
        uint4 raw_p = *p_u4;
        uint4 raw_g = *g_u4;
        float4 mi0 = *m_f4_0;
        float4 mi1 = *m_f4_1;
        float4 vi0 = *v_f4_0;
        float4 vi1 = *v_f4_1;
        
        const __nv_bfloat16* p_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_p);
        const __nv_bfloat16* g_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_g);
        
        float m_vals[8], v_vals[8];
        *reinterpret_cast<float4*>(&m_vals[0]) = mi0;
        *reinterpret_cast<float4*>(&m_vals[4]) = mi1;
        *reinterpret_cast<float4*>(&v_vals[0]) = vi0;
        *reinterpret_cast<float4*>(&v_vals[4]) = vi1;
        
        __nv_bfloat16 new_p[8];
        float new_m[8];
        float new_v[8];
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float m_val = m_vals[k];
            float v_val = v_vals[k];
            float g_val = __bfloat162float(g_bf16[k]) * clip_coef;
            float p_val = __bfloat162float(p_bf16[k]);
            
            m_val = beta1 * m_val + (1.0f - beta1) * g_val;
            v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;
            p_val = p_val - lr * (m_val / (sqrtf(v_val) + eps) + weight_decay * p_val);
            
            new_p[k] = __float2bfloat16(p_val);
            new_m[k] = m_val;
            new_v[k] = v_val;
        }
        
        *p_u4 = *reinterpret_cast<uint4*>(new_p);
        *m_f4_0 = *reinterpret_cast<float4*>(&new_m[0]);
        *m_f4_1 = *reinterpret_cast<float4*>(&new_m[4]);
        *v_f4_0 = *reinterpret_cast<float4*>(&new_v[0]);
        *v_f4_1 = *reinterpret_cast<float4*>(&new_v[4]);
    } else {
        for (int k = 0; k < 8 && idx + k < N; ++k) {
            int i = idx + k;
            float g = __bfloat162float(grad[i]) * clip_coef;
            float p = __bfloat162float(param[i]);
            float mi = m[i];
            float vi = v[i];
            
            mi = beta1 * mi + (1.0f - beta1) * g;
            vi = beta2 * vi + (1.0f - beta2) * g * g;
            p = p - lr * (mi / (sqrtf(vi) + eps) + weight_decay * p);
            
            param[i] = __float2bfloat16(p);
            m[i] = mi;
            v[i] = vi;
        }
    }
}

void launch_fused_adamw_update_bf16(
    __nv_bfloat16* param,
    __nv_bfloat16* grad,
    float* m,
    float* v,
    const float* clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int num_elements,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    int grid = ((num_elements + 7) / 8 + BLOCK - 1) / BLOCK;
    fused_adamw_update_bf16_vec8_kernel<<<grid, BLOCK, 0, stream>>>(
        param, grad, m, v, clip_coef, lr, beta1, beta2, eps, weight_decay, num_elements
    );
}

// ---------------------------------------------------------------------------
// Phase 29: Vectorized 8-way BF16 Moments Single Parameter Update (14 B/elem)
// ---------------------------------------------------------------------------
__global__ void fused_adamw_update_bf16_moments_vec8_kernel(
    __nv_bfloat16* __restrict__ param,
    __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ m,
    __nv_bfloat16* __restrict__ v,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int N,
    __nv_fp8_e4m3* __restrict__ param_fp8 = nullptr,
    float scale_fp8 = 64.0f
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    
    if (idx + 7 < N) {
        uint4* p_u4 = reinterpret_cast<uint4*>(param + idx);
        uint4* g_u4 = reinterpret_cast<uint4*>(grad + idx);
        uint4* m_u4 = reinterpret_cast<uint4*>(m + idx);
        uint4* v_u4 = reinterpret_cast<uint4*>(v + idx);
        
        uint4 raw_p = *p_u4;
        uint4 raw_g = *g_u4;
        uint4 raw_m = *m_u4;
        uint4 raw_v = *v_u4;
        
        const __nv_bfloat16* p_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_p);
        const __nv_bfloat16* g_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_g);
        const __nv_bfloat16* m_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_m);
        const __nv_bfloat16* v_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_v);
        
        __nv_bfloat16 new_p[8];
        __nv_bfloat16 new_m[8];
        __nv_bfloat16 new_v[8];
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float m_val = __bfloat162float(m_bf16[k]);
            float v_val = __bfloat162float(v_bf16[k]);
            float g_val = __bfloat162float(g_bf16[k]) * clip_coef;
            float p_val = __bfloat162float(p_bf16[k]);
            
            m_val = beta1 * m_val + (1.0f - beta1) * g_val;
            v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;
            p_val = p_val - lr * (m_val / (sqrtf(v_val) + eps) + weight_decay * p_val);
            
            new_p[k] = __float2bfloat16(p_val);
            new_m[k] = __float2bfloat16(m_val);
            new_v[k] = __float2bfloat16(v_val);
        }
        
        *p_u4 = *reinterpret_cast<uint4*>(new_p);
        *m_u4 = *reinterpret_cast<uint4*>(new_m);
        *v_u4 = *reinterpret_cast<uint4*>(new_v);
        if (param_fp8) {
            __nv_fp8_e4m3 res_fp8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                res_fp8[k] = __nv_fp8_e4m3(__bfloat162float(new_p[k]) * scale_fp8);
            }
            *reinterpret_cast<uint2*>(param_fp8 + idx) = *reinterpret_cast<uint2*>(res_fp8);
        }
    } else {
        for (int k = 0; k < 8 && idx + k < N; ++k) {
            int i = idx + k;
            float g = __bfloat162float(grad[i]) * clip_coef;
            float p = __bfloat162float(param[i]);
            float mi = __bfloat162float(m[i]);
            float vi = __bfloat162float(v[i]);
            
            mi = beta1 * mi + (1.0f - beta1) * g;
            vi = beta2 * vi + (1.0f - beta2) * g * g;
            p = p - lr * (mi / (sqrtf(vi) + eps) + weight_decay * p);
            
            param[i] = __float2bfloat16(p);
            m[i] = __float2bfloat16(mi);
            v[i] = __float2bfloat16(vi);
            if (param_fp8) {
                param_fp8[i] = __nv_fp8_e4m3(p * scale_fp8);
            }
        }
    }
}

void launch_fused_adamw_update_bf16_moments(
    __nv_bfloat16* param,
    __nv_bfloat16* grad,
    __nv_bfloat16* m,
    __nv_bfloat16* v,
    const float* clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int num_elements,
    cudaStream_t stream,
    __nv_fp8_e4m3* param_fp8,
    float scale_fp8
) {
    const int BLOCK = 256;
    int grid = ((num_elements + 7) / 8 + BLOCK - 1) / BLOCK;
    fused_adamw_update_bf16_moments_vec8_kernel<<<grid, BLOCK, 0, stream>>>(
        param, grad, m, v, clip_coef, lr, beta1, beta2, eps, weight_decay, num_elements,
        param_fp8, scale_fp8
    );
}

// ---------------------------------------------------------------------------
// 3. FP8 Moments Fused AdamW Kernel (10 Bytes/Element: m E4M3, v E5M2)
// ---------------------------------------------------------------------------
__global__ void fused_adamw_update_fp8_moments_vec8_kernel(
    __nv_bfloat16* __restrict__ param,
    __nv_bfloat16* __restrict__ grad,
    __nv_fp8_e4m3* __restrict__ m_fp8,
    __nv_fp8_e5m2* __restrict__ v_fp8,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int N,
    __nv_fp8_e4m3* __restrict__ param_fp8 = nullptr,
    float scale_fp8 = 64.0f,
    float scale_m = 256.0f,
    float inv_scale_m = 1.0f / 256.0f,
    float scale_s = 512.0f,
    float inv_scale_s = 1.0f / 512.0f
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    
    if (idx + 7 < N) {
        uint4* p_u4 = reinterpret_cast<uint4*>(param + idx);
        const uint4* g_u4 = reinterpret_cast<const uint4*>(grad + idx);
        uint2* m_u2 = reinterpret_cast<uint2*>(m_fp8 + idx);
        uint2* v_u2 = reinterpret_cast<uint2*>(v_fp8 + idx);
        
        uint4 raw_p = *p_u4;
        uint4 raw_g = *g_u4;
        uint2 raw_m = *m_u2;
        uint2 raw_v = *v_u2;
        
        const __nv_bfloat16* p_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_p);
        const __nv_bfloat16* g_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_g);
        const __nv_fp8_e4m3* m_in = reinterpret_cast<const __nv_fp8_e4m3*>(&raw_m);
        const __nv_fp8_e5m2* s_in = reinterpret_cast<const __nv_fp8_e5m2*>(&raw_v);
        
        __nv_bfloat16 new_p[8];
        __nv_fp8_e4m3 new_m[8];
        __nv_fp8_e5m2 new_v[8];
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float m_val = float(m_in[k]) * inv_scale_m;
            float s_val = float(s_in[k]) * inv_scale_s;
            float v_prev = s_val * s_val;
            float g_val = __bfloat162float(g_bf16[k]) * clip_coef;
            float p_val = __bfloat162float(p_bf16[k]);
            
            m_val = beta1 * m_val + (1.0f - beta1) * g_val;
            float v_val = beta2 * v_prev + (1.0f - beta2) * g_val * g_val;
            float s_val_new = sqrtf(v_val);
            p_val = p_val - lr * (m_val / (s_val_new + eps) + weight_decay * p_val);
            
            new_p[k] = __float2bfloat16(p_val);
            new_m[k] = __nv_fp8_e4m3(m_val * scale_m);
            new_v[k] = __nv_fp8_e5m2(s_val_new * scale_s);
        }
        
        *p_u4 = *reinterpret_cast<uint4*>(new_p);
        *m_u2 = *reinterpret_cast<uint2*>(new_m);
        *v_u2 = *reinterpret_cast<uint2*>(new_v);
        if (param_fp8) {
            __nv_fp8_e4m3 res_fp8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                res_fp8[k] = __nv_fp8_e4m3(__bfloat162float(new_p[k]) * scale_fp8);
            }
            *reinterpret_cast<uint2*>(param_fp8 + idx) = *reinterpret_cast<uint2*>(res_fp8);
        }
    } else {
        for (int k = 0; k < 8 && idx + k < N; ++k) {
            int i = idx + k;
            float g_val = __bfloat162float(grad[i]) * clip_coef;
            float p_val = __bfloat162float(param[i]);
            float m_val = float(m_fp8[i]) * inv_scale_m;
            float s_val = float(v_fp8[i]) * inv_scale_s;
            float v_prev = s_val * s_val;
            
            m_val = beta1 * m_val + (1.0f - beta1) * g_val;
            float v_val = beta2 * v_prev + (1.0f - beta2) * g_val * g_val;
            float s_val_new = sqrtf(v_val);
            p_val = p_val - lr * (m_val / (s_val_new + eps) + weight_decay * p_val);
            
            param[i] = __float2bfloat16(p_val);
            m_fp8[i] = __nv_fp8_e4m3(m_val * scale_m);
            v_fp8[i] = __nv_fp8_e5m2(s_val_new * scale_s);
            if (param_fp8) {
                param_fp8[i] = __nv_fp8_e4m3(p_val * scale_fp8);
            }
        }
    }
}

void launch_fused_adamw_update_fp8_moments(
    __nv_bfloat16* param,
    __nv_bfloat16* grad,
    __nv_fp8_e4m3* m,
    __nv_fp8_e5m2* v,
    const float* clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int num_elements,
    cudaStream_t stream,
    __nv_fp8_e4m3* param_fp8,
    float scale_fp8,
    float scale_m,
    float scale_v
) {
    const int BLOCK = 256;
    int grid = ((num_elements + 7) / 8 + BLOCK - 1) / BLOCK;
    fused_adamw_update_fp8_moments_vec8_kernel<<<grid, BLOCK, 0, stream>>>(
        param, grad, m, v, clip_coef, lr, beta1, beta2, eps, weight_decay, num_elements,
        param_fp8, scale_fp8, scale_m, 1.0f / scale_m, scale_v, 1.0f / scale_v
    );
}

__global__ void fused_adamw_update_f32_kernel(
    float* __restrict__ param,
    float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float clip_coef = *clip_coef_ptr;
    float g = grad[idx] * clip_coef;
    float p = param[idx];
    float mi = m[idx];
    float vi = v[idx];
    
    mi = beta1 * mi + (1.0f - beta1) * g;
    vi = beta2 * vi + (1.0f - beta2) * g * g;
    p = p - lr * (mi / (sqrtf(vi) + eps) + weight_decay * p);
    
    param[idx] = p;
    m[idx] = mi;
    v[idx] = vi;
    grad[idx] = 0.0f;
}

void launch_fused_adamw_update_f32(
    float* param,
    float* grad,
    float* m,
    float* v,
    const float* clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int num_elements,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    int grid = (num_elements + BLOCK - 1) / BLOCK;
    fused_adamw_update_f32_kernel<<<grid, BLOCK, 0, stream>>>(
        param, grad, m, v, clip_coef, lr, beta1, beta2, eps, weight_decay, num_elements
    );
}

// ---------------------------------------------------------------------------
// 4. Multi-Tensor Consolidated Grouped AdamW Kernels (EXP-24-010)
// - Fuses ALL 192 MoE expert matrices across all 24 layers into a single launch: dim3(1024, 192)
// - Fuses ALL 24 QKV matrices across all 24 layers into a single launch: dim3(1536, 24)
// - Fuses ALL 24 Out Proj matrices across all 24 layers into a single launch: dim3(512, 24)
// - Fuses ALL small layer parameters (norm1, norm2, router, gamma, var) + final_norm into 1 launch: dim3(25)
// - Reduces total optimizer launches from 99 launches down to 6 launches!
// - Eliminates 23 SM tail remainder waves per update step.
// ---------------------------------------------------------------------------
struct AllMoEExperts {
    __nv_bfloat16* p[192];
    __nv_bfloat16* g[192];
    float* m[192];
    float* v[192];
};

struct AllQKV {
    __nv_bfloat16* p[24];
    __nv_bfloat16* g[24];
    float* m[24];
    float* v[24];
};

struct AllOutProj {
    __nv_bfloat16* p[24];
    __nv_bfloat16* g[24];
    float* m[24];
    float* v[24];
};

struct AllSmallParams {
    __nv_bfloat16* norm1[24];
    __nv_bfloat16* d_norm1[24];
    float* m_norm1[24];
    float* v_norm1[24];
    
    __nv_bfloat16* norm2[24];
    __nv_bfloat16* d_norm2[24];
    float* m_norm2[24];
    float* v_norm2[24];
    
    __nv_bfloat16* router[24];
    __nv_bfloat16* d_router[24];
    float* m_router[24];
    float* v_router[24];
    
    float* gamma[24];
    float* d_gamma[24];
    float* m_gamma[24];
    float* v_gamma[24];
    
    float* var[24];
    float* d_var[24];
    float* m_var[24];
    float* v_var[24];
    
    __nv_bfloat16* final_norm;
    __nv_bfloat16* d_final_norm;
    float* m_final_norm;
    float* v_final_norm;
};

__device__ __forceinline__ void update_bf16_vec8(
    __nv_bfloat16* param,
    __nv_bfloat16* grad,
    float* m,
    float* v,
    int idx,
    float clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    uint4* p_u4 = reinterpret_cast<uint4*>(param + idx);
    uint4* g_u4 = reinterpret_cast<uint4*>(grad + idx);
    float4* m_f4_0 = reinterpret_cast<float4*>(m + idx);
    float4* m_f4_1 = reinterpret_cast<float4*>(m + idx + 4);
    float4* v_f4_0 = reinterpret_cast<float4*>(v + idx);
    float4* v_f4_1 = reinterpret_cast<float4*>(v + idx + 4);
    
    uint4 raw_p = *p_u4;
    uint4 raw_g = *g_u4;
    float4 mi0 = *m_f4_0;
    float4 mi1 = *m_f4_1;
    float4 vi0 = *v_f4_0;
    float4 vi1 = *v_f4_1;
    
    const __nv_bfloat16* p_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_p);
    const __nv_bfloat16* g_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_g);
    
    float m_vals[8], v_vals[8];
    *reinterpret_cast<float4*>(&m_vals[0]) = mi0;
    *reinterpret_cast<float4*>(&m_vals[4]) = mi1;
    *reinterpret_cast<float4*>(&v_vals[0]) = vi0;
    *reinterpret_cast<float4*>(&v_vals[4]) = vi1;
    
    __nv_bfloat16 new_p[8];
    float new_m[8];
    float new_v[8];
    
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float m_val = m_vals[k];
        float v_val = v_vals[k];
        float g_val = __bfloat162float(g_bf16[k]) * clip_coef;
        float p_val = __bfloat162float(p_bf16[k]);
        
        m_val = beta1 * m_val + (1.0f - beta1) * g_val;
        v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;
        p_val = p_val - lr * (m_val / (sqrtf(v_val) + eps) + weight_decay * p_val);
        
        new_p[k] = __float2bfloat16(p_val);
        new_m[k] = m_val;
        new_v[k] = v_val;
    }
    
    *p_u4 = *reinterpret_cast<uint4*>(new_p);
    *m_f4_0 = *reinterpret_cast<float4*>(&new_m[0]);
    *m_f4_1 = *reinterpret_cast<float4*>(&new_m[4]);
    *v_f4_0 = *reinterpret_cast<float4*>(&new_v[0]);
    *v_f4_1 = *reinterpret_cast<float4*>(&new_v[4]);
}

// ---------------------------------------------------------------------------
// 1. Consolidated All MoE Experts (192 experts in 1 launch: dim3(1024, 192))
// ---------------------------------------------------------------------------
__global__ void fused_adamw_all_moe_experts_vec8_kernel(
    const AllMoEExperts* __restrict__ experts,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ float* s_m;
    __shared__ float* s_v;
    
    if (threadIdx.x == 0) {
        int expert_id = blockIdx.y;
        s_param = experts->p[expert_id];
        s_grad = experts->g[expert_id];
        s_m = experts->m[expert_id];
        s_v = experts->v[expert_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_bf16_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
}

// ---------------------------------------------------------------------------
// 2. Consolidated All QKV (24 layers in 1 launch: dim3(1536, 24))
// ---------------------------------------------------------------------------
__global__ void fused_adamw_all_qkv_vec8_kernel(
    const AllQKV* __restrict__ qkv,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ float* s_m;
    __shared__ float* s_v;
    
    if (threadIdx.x == 0) {
        int layer_id = blockIdx.y;
        s_param = qkv->p[layer_id];
        s_grad = qkv->g[layer_id];
        s_m = qkv->m[layer_id];
        s_v = qkv->v[layer_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_bf16_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
}

// ---------------------------------------------------------------------------
// 3. Consolidated All Out Proj (24 layers in 1 launch: dim3(512, 24))
// ---------------------------------------------------------------------------
__global__ void fused_adamw_all_out_proj_vec8_kernel(
    const AllOutProj* __restrict__ out_proj,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ float* s_m;
    __shared__ float* s_v;
    
    if (threadIdx.x == 0) {
        int layer_id = blockIdx.y;
        s_param = out_proj->p[layer_id];
        s_grad = out_proj->g[layer_id];
        s_m = out_proj->m[layer_id];
        s_v = out_proj->v[layer_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_bf16_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
}

// ---------------------------------------------------------------------------
// 4. Consolidated All Small Params + Final Norm (25 blocks in 1 launch: dim3(25))
// ---------------------------------------------------------------------------
__global__ void fused_adamw_all_small_params_kernel(
    const AllSmallParams* __restrict__ sp,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int C,
    int E,
    int H
) {
    int block_id = blockIdx.x; // 0..23: layers 0..23, 24: final_norm
    int tid = threadIdx.x;
    float clip_coef = *clip_coef_ptr;
    
    if (block_id < 24) {
        int l = block_id;
        // 1. norm1 (C elements)
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_bf16_vec8(sp->norm1[l], sp->d_norm1[l], sp->m_norm1[l], sp->v_norm1[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        // 2. norm2 (C elements)
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_bf16_vec8(sp->norm2[l], sp->d_norm2[l], sp->m_norm2[l], sp->v_norm2[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        // 3. router (E * C elements)
        int EC = E * C;
        for (int idx = tid * 8; idx < EC; idx += blockDim.x * 8) {
            update_bf16_vec8(sp->router[l], sp->d_router[l], sp->m_router[l], sp->v_router[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        // 4. gamma (H elements)
        if (tid < H) {
            float g = sp->d_gamma[l][tid] * clip_coef;
            float param = sp->gamma[l][tid];
            float mi = sp->m_gamma[l][tid];
            float vi = sp->v_gamma[l][tid];
            mi = beta1 * mi + (1.0f - beta1) * g;
            vi = beta2 * vi + (1.0f - beta2) * g * g;
            param = param - lr * (mi / (sqrtf(vi) + eps));
            sp->gamma[l][tid] = param;
            sp->m_gamma[l][tid] = mi;
            sp->v_gamma[l][tid] = vi;
            sp->d_gamma[l][tid] = 0.0f;
        }
        // 5. var (1 element)
        if (tid == 0) {
            float g = sp->d_var[l][0] * clip_coef;
            float param = sp->var[l][0];
            float mi = sp->m_var[l][0];
            float vi = sp->v_var[l][0];
            mi = beta1 * mi + (1.0f - beta1) * g;
            vi = beta2 * vi + (1.0f - beta2) * g * g;
            param = param - lr * (mi / (sqrtf(vi) + eps));
            sp->var[l][0] = param;
            sp->m_var[l][0] = mi;
            sp->v_var[l][0] = vi;
            sp->d_var[l][0] = 0.0f;
        }
    } else if (block_id == 24) {
        // Block 24: final_norm (C elements)
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_bf16_vec8(sp->final_norm, sp->d_final_norm, sp->m_final_norm, sp->v_final_norm, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// BF16 Moments Consolidated Structs & Kernels (Phase 29: 14 Bytes/Element)
// ---------------------------------------------------------------------------
struct AllMoEExpertsBF16 {
    __nv_bfloat16* p[192];
    __nv_bfloat16* g[192];
    __nv_bfloat16* m[192];
    __nv_bfloat16* v[192];
};

struct AllQKVBF16 {
    __nv_bfloat16* p[24];
    __nv_bfloat16* g[24];
    __nv_bfloat16* m[24];
    __nv_bfloat16* v[24];
    __nv_fp8_e4m3* p_fp8[24];
};

struct AllOutProjBF16 {
    __nv_bfloat16* p[24];
    __nv_bfloat16* g[24];
    __nv_bfloat16* m[24];
    __nv_bfloat16* v[24];
};

struct AllSmallParamsBF16 {
    __nv_bfloat16* norm1[24];
    __nv_bfloat16* d_norm1[24];
    __nv_bfloat16* m_norm1[24];
    __nv_bfloat16* v_norm1[24];
    
    __nv_bfloat16* norm2[24];
    __nv_bfloat16* d_norm2[24];
    __nv_bfloat16* m_norm2[24];
    __nv_bfloat16* v_norm2[24];
    
    __nv_bfloat16* router[24];
    __nv_bfloat16* d_router[24];
    __nv_bfloat16* m_router[24];
    __nv_bfloat16* v_router[24];
    
    float* gamma[24];
    float* d_gamma[24];
    float* m_gamma[24];
    float* v_gamma[24];
    
    float* var[24];
    float* d_var[24];
    float* m_var[24];
    float* v_var[24];
    
    __nv_bfloat16* final_norm;
    __nv_bfloat16* d_final_norm;
    __nv_bfloat16* m_final_norm;
    __nv_bfloat16* v_final_norm;
};

__device__ __forceinline__ void update_bf16_moments_vec8(
    __nv_bfloat16* param,
    __nv_bfloat16* grad,
    __nv_bfloat16* m,
    __nv_bfloat16* v,
    int idx,
    float clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    __nv_fp8_e4m3* param_fp8 = nullptr,
    float scale_fp8 = 64.0f
) {
    uint4* p_u4 = reinterpret_cast<uint4*>(param + idx);
    const uint4* g_u4 = reinterpret_cast<const uint4*>(grad + idx);
    uint4* m_u4 = reinterpret_cast<uint4*>(m + idx);
    uint4* v_u4 = reinterpret_cast<uint4*>(v + idx);
    
    uint4 raw_p = *p_u4;
    uint4 raw_g = *g_u4;
    uint4 raw_m = *m_u4;
    uint4 raw_v = *v_u4;
    
    const __nv_bfloat16* p_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_p);
    const __nv_bfloat16* g_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_g);
    const __nv_bfloat16* m_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_m);
    const __nv_bfloat16* v_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_v);
    
    __nv_bfloat16 new_p[8];
    __nv_bfloat16 new_m[8];
    __nv_bfloat16 new_v[8];
    
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float m_val = __bfloat162float(m_bf16[k]);
        float v_val = __bfloat162float(v_bf16[k]);
        float g_val = __bfloat162float(g_bf16[k]) * clip_coef;
        float p_val = __bfloat162float(p_bf16[k]);
        
        m_val = beta1 * m_val + (1.0f - beta1) * g_val;
        v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;
        p_val = p_val - lr * (m_val / (sqrtf(v_val) + eps) + weight_decay * p_val);
        
        new_p[k] = __float2bfloat16(p_val);
        new_m[k] = __float2bfloat16(m_val);
        new_v[k] = __float2bfloat16(v_val);
    }
    
    *p_u4 = *reinterpret_cast<uint4*>(new_p);
    *m_u4 = *reinterpret_cast<uint4*>(new_m);
    *v_u4 = *reinterpret_cast<uint4*>(new_v);
    if (param_fp8) {
        __nv_fp8_e4m3 res_fp8[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            res_fp8[k] = __nv_fp8_e4m3(__bfloat162float(new_p[k]) * scale_fp8);
        }
        *reinterpret_cast<uint2*>(param_fp8 + idx) = *reinterpret_cast<uint2*>(res_fp8);
    }
}

__global__ void fused_adamw_all_moe_experts_bf16_moments_vec8_kernel(
    const AllMoEExpertsBF16* __restrict__ experts,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ __nv_bfloat16* s_m;
    __shared__ __nv_bfloat16* s_v;
    
    if (threadIdx.x == 0) {
        int expert_id = blockIdx.y;
        s_param = experts->p[expert_id];
        s_grad = experts->g[expert_id];
        s_m = experts->m[expert_id];
        s_v = experts->v[expert_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_bf16_moments_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
}

__global__ void fused_adamw_all_qkv_bf16_moments_vec8_kernel(
    const AllQKVBF16* __restrict__ qkv,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ __nv_bfloat16* s_m;
    __shared__ __nv_bfloat16* s_v;
    __shared__ __nv_fp8_e4m3* s_param_fp8;
    
    if (threadIdx.x == 0) {
        int layer_id = blockIdx.y;
        s_param = qkv->p[layer_id];
        s_grad = qkv->g[layer_id];
        s_m = qkv->m[layer_id];
        s_v = qkv->v[layer_id];
        s_param_fp8 = qkv->p_fp8[layer_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_bf16_moments_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay, s_param_fp8, 64.0f);
}

__global__ void fused_adamw_all_out_proj_bf16_moments_vec8_kernel(
    const AllOutProjBF16* __restrict__ out_proj,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ __nv_bfloat16* s_m;
    __shared__ __nv_bfloat16* s_v;
    
    if (threadIdx.x == 0) {
        int layer_id = blockIdx.y;
        s_param = out_proj->p[layer_id];
        s_grad = out_proj->g[layer_id];
        s_m = out_proj->m[layer_id];
        s_v = out_proj->v[layer_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_bf16_moments_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
}

__global__ void fused_adamw_all_small_params_bf16_moments_kernel(
    const AllSmallParamsBF16* __restrict__ sp,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int C,
    int E,
    int H
) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    float clip_coef = *clip_coef_ptr;
    
    if (block_id < 24) {
        int l = block_id;
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_bf16_moments_vec8(sp->norm1[l], sp->d_norm1[l], sp->m_norm1[l], sp->v_norm1[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_bf16_moments_vec8(sp->norm2[l], sp->d_norm2[l], sp->m_norm2[l], sp->v_norm2[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        int EC = E * C;
        for (int idx = tid * 8; idx < EC; idx += blockDim.x * 8) {
            update_bf16_moments_vec8(sp->router[l], sp->d_router[l], sp->m_router[l], sp->v_router[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        if (tid < H) {
            float g = sp->d_gamma[l][tid] * clip_coef;
            float param = sp->gamma[l][tid];
            float mi = sp->m_gamma[l][tid];
            float vi = sp->v_gamma[l][tid];
            mi = beta1 * mi + (1.0f - beta1) * g;
            vi = beta2 * vi + (1.0f - beta2) * g * g;
            param = param - lr * (mi / (sqrtf(vi) + eps));
            sp->gamma[l][tid] = param;
            sp->m_gamma[l][tid] = mi;
            sp->v_gamma[l][tid] = vi;
            sp->d_gamma[l][tid] = 0.0f;
        }
        if (tid == 0) {
            float g = sp->d_var[l][0] * clip_coef;
            float param = sp->var[l][0];
            float mi = sp->m_var[l][0];
            float vi = sp->v_var[l][0];
            mi = beta1 * mi + (1.0f - beta1) * g;
            vi = beta2 * vi + (1.0f - beta2) * g * g;
            param = param - lr * (mi / (sqrtf(vi) + eps));
            sp->var[l][0] = param;
            sp->m_var[l][0] = mi;
            sp->v_var[l][0] = vi;
            sp->d_var[l][0] = 0.0f;
        }
    } else if (block_id == 24) {
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_bf16_moments_vec8(sp->final_norm, sp->d_final_norm, sp->m_final_norm, sp->v_final_norm, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 Moments Consolidated Structs & Kernels (Phase 31: 10 Bytes/Element)
// ---------------------------------------------------------------------------
struct AllMoEExpertsFP8 {
    __nv_bfloat16* p[192];
    __nv_bfloat16* g[192];
    __nv_fp8_e4m3* m[192];
    __nv_fp8_e5m2* v[192];
};

struct AllQKVFP8 {
    __nv_bfloat16* p[24];
    __nv_bfloat16* g[24];
    __nv_fp8_e4m3* m[24];
    __nv_fp8_e5m2* v[24];
    __nv_fp8_e4m3* p_fp8[24];
};

struct AllOutProjFP8 {
    __nv_bfloat16* p[24];
    __nv_bfloat16* g[24];
    __nv_fp8_e4m3* m[24];
    __nv_fp8_e5m2* v[24];
};

struct AllSmallParamsFP8 {
    __nv_bfloat16* norm1[24];
    __nv_bfloat16* d_norm1[24];
    __nv_fp8_e4m3* m_norm1[24];
    __nv_fp8_e5m2* v_norm1[24];
    
    __nv_bfloat16* norm2[24];
    __nv_bfloat16* d_norm2[24];
    __nv_fp8_e4m3* m_norm2[24];
    __nv_fp8_e5m2* v_norm2[24];
    
    __nv_bfloat16* router[24];
    __nv_bfloat16* d_router[24];
    __nv_fp8_e4m3* m_router[24];
    __nv_fp8_e5m2* v_router[24];
    
    float* gamma[24];
    float* d_gamma[24];
    float* m_gamma[24];
    float* v_gamma[24];
    
    float* var[24];
    float* d_var[24];
    float* m_var[24];
    float* v_var[24];
    
    __nv_bfloat16* final_norm;
    __nv_bfloat16* d_final_norm;
    __nv_fp8_e4m3* m_final_norm;
    __nv_fp8_e5m2* v_final_norm;
};

__device__ __forceinline__ void update_fp8_moments_vec8(
    __nv_bfloat16* param,
    const __nv_bfloat16* grad,
    __nv_fp8_e4m3* m,
    __nv_fp8_e5m2* v, // stores s = sqrt(v)
    int idx,
    float clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    __nv_fp8_e4m3* param_fp8 = nullptr,
    float scale_fp8 = 64.0f,
    float scale_m = 256.0f,
    float inv_scale_m = 1.0f / 256.0f,
    float scale_s = 512.0f,
    float inv_scale_s = 1.0f / 512.0f
) {
    uint4* p_u4 = reinterpret_cast<uint4*>(param + idx);
    const uint4* g_u4 = reinterpret_cast<const uint4*>(grad + idx);
    uint2* m_u2 = reinterpret_cast<uint2*>(m + idx);
    uint2* v_u2 = reinterpret_cast<uint2*>(v + idx);
    
    uint4 raw_p = *p_u4;
    uint4 raw_g = *g_u4;
    uint2 raw_m = *m_u2;
    uint2 raw_v = *v_u2;
    
    const __nv_bfloat16* p_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_p);
    const __nv_bfloat16* g_bf16 = reinterpret_cast<const __nv_bfloat16*>(&raw_g);
    const __nv_fp8_e4m3* m_in = reinterpret_cast<const __nv_fp8_e4m3*>(&raw_m);
    const __nv_fp8_e5m2* s_in = reinterpret_cast<const __nv_fp8_e5m2*>(&raw_v);
    
    __nv_bfloat16 new_p[8];
    __nv_fp8_e4m3 new_m[8];
    __nv_fp8_e5m2 new_v[8];
    
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float m_val = float(m_in[k]) * inv_scale_m;
        float s_val = float(s_in[k]) * inv_scale_s;
        float v_prev = s_val * s_val;
        float g_val = __bfloat162float(g_bf16[k]) * clip_coef;
        float p_val = __bfloat162float(p_bf16[k]);
        
        m_val = beta1 * m_val + (1.0f - beta1) * g_val;
        float v_val = beta2 * v_prev + (1.0f - beta2) * g_val * g_val;
        float s_val_new = sqrtf(v_val);
        p_val = p_val - lr * (m_val / (s_val_new + eps) + weight_decay * p_val);
        
        new_p[k] = __float2bfloat16(p_val);
        new_m[k] = __nv_fp8_e4m3(m_val * scale_m);
        new_v[k] = __nv_fp8_e5m2(s_val_new * scale_s);
    }
    
    *p_u4 = *reinterpret_cast<uint4*>(new_p);
    *m_u2 = *reinterpret_cast<uint2*>(new_m);
    *v_u2 = *reinterpret_cast<uint2*>(new_v);
    if (param_fp8) {
        __nv_fp8_e4m3 res_fp8[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            res_fp8[k] = __nv_fp8_e4m3(__bfloat162float(new_p[k]) * scale_fp8);
        }
        *reinterpret_cast<uint2*>(param_fp8 + idx) = *reinterpret_cast<uint2*>(res_fp8);
    }
}

__global__ void fused_adamw_all_moe_experts_fp8_moments_vec8_kernel(
    const AllMoEExpertsFP8* __restrict__ experts,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ __nv_fp8_e4m3* s_m;
    __shared__ __nv_fp8_e5m2* s_v;
    
    if (threadIdx.x == 0) {
        int expert_id = blockIdx.y;
        s_param = experts->p[expert_id];
        s_grad = experts->g[expert_id];
        s_m = experts->m[expert_id];
        s_v = experts->v[expert_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_fp8_moments_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
}

__global__ void fused_adamw_all_qkv_fp8_moments_vec8_kernel(
    const AllQKVFP8* __restrict__ qkv,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ __nv_fp8_e4m3* s_m;
    __shared__ __nv_fp8_e5m2* s_v;
    __shared__ __nv_fp8_e4m3* s_param_fp8;
    
    if (threadIdx.x == 0) {
        int layer_id = blockIdx.y;
        s_param = qkv->p[layer_id];
        s_grad = qkv->g[layer_id];
        s_m = qkv->m[layer_id];
        s_v = qkv->v[layer_id];
        s_param_fp8 = qkv->p_fp8[layer_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_fp8_moments_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay, s_param_fp8, 64.0f);
}

__global__ void fused_adamw_all_out_proj_fp8_moments_vec8_kernel(
    const AllOutProjFP8* __restrict__ out_proj,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    __shared__ __nv_bfloat16* s_param;
    __shared__ __nv_bfloat16* s_grad;
    __shared__ __nv_fp8_e4m3* s_m;
    __shared__ __nv_fp8_e5m2* s_v;
    
    if (threadIdx.x == 0) {
        int layer_id = blockIdx.y;
        s_param = out_proj->p[layer_id];
        s_grad = out_proj->g[layer_id];
        s_m = out_proj->m[layer_id];
        s_v = out_proj->v[layer_id];
    }
    __syncthreads();
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float clip_coef = *clip_coef_ptr;
    update_fp8_moments_vec8(s_param, s_grad, s_m, s_v, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
}

__global__ void fused_adamw_all_small_params_fp8_moments_kernel(
    const AllSmallParamsFP8* __restrict__ sp,
    const float* __restrict__ clip_coef_ptr,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int C,
    int E,
    int H
) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    float clip_coef = *clip_coef_ptr;
    
    if (block_id < 24) {
        int l = block_id;
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_fp8_moments_vec8(sp->norm1[l], sp->d_norm1[l], sp->m_norm1[l], sp->v_norm1[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_fp8_moments_vec8(sp->norm2[l], sp->d_norm2[l], sp->m_norm2[l], sp->v_norm2[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        int EC = E * C;
        for (int idx = tid * 8; idx < EC; idx += blockDim.x * 8) {
            update_fp8_moments_vec8(sp->router[l], sp->d_router[l], sp->m_router[l], sp->v_router[l], idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
        if (tid < H) {
            float g = sp->d_gamma[l][tid] * clip_coef;
            float param = sp->gamma[l][tid];
            float mi = sp->m_gamma[l][tid];
            float vi = sp->v_gamma[l][tid];
            mi = beta1 * mi + (1.0f - beta1) * g;
            vi = beta2 * vi + (1.0f - beta2) * g * g;
            param = param - lr * (mi / (sqrtf(vi) + eps));
            sp->gamma[l][tid] = param;
            sp->m_gamma[l][tid] = mi;
            sp->v_gamma[l][tid] = vi;
            sp->d_gamma[l][tid] = 0.0f;
        }
        if (tid == 0) {
            float g = sp->d_var[l][0] * clip_coef;
            float param = sp->var[l][0];
            float mi = sp->m_var[l][0];
            float vi = sp->v_var[l][0];
            mi = beta1 * mi + (1.0f - beta1) * g;
            vi = beta2 * vi + (1.0f - beta2) * g * g;
            param = param - lr * (mi / (sqrtf(vi) + eps));
            sp->var[l][0] = param;
            sp->m_var[l][0] = mi;
            sp->v_var[l][0] = vi;
            sp->d_var[l][0] = 0.0f;
        }
    } else if (block_id == 24) {
        for (int idx = tid * 8; idx < C; idx += blockDim.x * 8) {
            update_fp8_moments_vec8(sp->final_norm, sp->d_final_norm, sp->m_final_norm, sp->v_final_norm, idx, clip_coef, lr, beta1, beta2, eps, weight_decay);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Synchronize Consolidated Pointer Tables to GPU Workspace (One-time)
// ---------------------------------------------------------------------------
void sync_optimizer_tables(
    const FullModelParameters& params,
    FullModelWorkspace& ws,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
) {
    if (ws.opt_tables_synced) return;
    
    if (cfg.use_fp8_moments) {
        AllMoEExpertsFP8 host_moe;
        AllQKVFP8 host_qkv;
        AllOutProjFP8 host_out_proj;
        AllSmallParamsFP8 host_small;
        
        for (int l = 0; l < cfg.num_layers; ++l) {
            const auto& lay = params.layers[l];
            for (int e = 0; e < 4; ++e) {
                int idx_w1 = l * 8 + e;
                host_moe.p[idx_w1] = lay.w1_weights[e];
                host_moe.g[idx_w1] = lay.d_w1_weights[e];
                host_moe.m[idx_w1] = lay.m_w1_fp8[e];
                host_moe.v[idx_w1] = lay.v_w1_fp8[e];
                
                int idx_w2 = l * 8 + 4 + e;
                host_moe.p[idx_w2] = lay.w2_weights[e];
                host_moe.g[idx_w2] = lay.d_w2_weights[e];
                host_moe.m[idx_w2] = lay.m_w2_fp8[e];
                host_moe.v[idx_w2] = lay.v_w2_fp8[e];
            }
            
            host_qkv.p[l] = lay.qkv_weight;
            host_qkv.g[l] = lay.d_qkv_weight;
            host_qkv.m[l] = lay.m_qkv_fp8;
            host_qkv.v[l] = lay.v_qkv_fp8;
            host_qkv.p_fp8[l] = lay.qkv_weight_fp8;
            
            host_out_proj.p[l] = lay.out_proj_weight;
            host_out_proj.g[l] = lay.d_out_proj_weight;
            host_out_proj.m[l] = lay.m_out_fp8;
            host_out_proj.v[l] = lay.v_out_fp8;
            
            host_small.norm1[l] = lay.norm1_weight; host_small.d_norm1[l] = lay.d_norm1_weight; host_small.m_norm1[l] = lay.m_norm1_fp8; host_small.v_norm1[l] = lay.v_norm1_fp8;
            host_small.norm2[l] = lay.norm2_weight; host_small.d_norm2[l] = lay.d_norm2_weight; host_small.m_norm2[l] = lay.m_norm2_fp8; host_small.v_norm2[l] = lay.v_norm2_fp8;
            host_small.router[l] = lay.router_weight; host_small.d_router[l] = lay.d_router_weight; host_small.m_router[l] = lay.m_router_fp8; host_small.v_router[l] = lay.v_router_fp8;
            host_small.gamma[l] = lay.gamma_raw; host_small.d_gamma[l] = lay.d_gamma_raw; host_small.m_gamma[l] = lay.m_gamma; host_small.v_gamma[l] = lay.v_gamma;
            host_small.var[l] = lay.var_scale; host_small.d_var[l] = lay.d_var_scale; host_small.m_var[l] = lay.m_var; host_small.v_var[l] = lay.v_var;
        }
        host_small.final_norm = params.final_norm_weight;
        host_small.d_final_norm = params.d_final_norm_weight;
        host_small.m_final_norm = params.m_final_norm_fp8;
        host_small.v_final_norm = params.v_final_norm_fp8;
        
        cudaMemcpyAsync(ws.d_all_moe_experts, &host_moe, sizeof(AllMoEExpertsFP8), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_qkv, &host_qkv, sizeof(AllQKVFP8), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_out_proj, &host_out_proj, sizeof(AllOutProjFP8), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_small_params, &host_small, sizeof(AllSmallParamsFP8), cudaMemcpyHostToDevice, stream);
    } else if (cfg.use_bf16_moments) {
        AllMoEExpertsBF16 host_moe;
        AllQKVBF16 host_qkv;
        AllOutProjBF16 host_out_proj;
        AllSmallParamsBF16 host_small;
        
        for (int l = 0; l < cfg.num_layers; ++l) {
            const auto& lay = params.layers[l];
            for (int e = 0; e < 4; ++e) {
                int idx_w1 = l * 8 + e;
                host_moe.p[idx_w1] = lay.w1_weights[e];
                host_moe.g[idx_w1] = lay.d_w1_weights[e];
                host_moe.m[idx_w1] = lay.m_w1_bf16[e];
                host_moe.v[idx_w1] = lay.v_w1_bf16[e];
                
                int idx_w2 = l * 8 + 4 + e;
                host_moe.p[idx_w2] = lay.w2_weights[e];
                host_moe.g[idx_w2] = lay.d_w2_weights[e];
                host_moe.m[idx_w2] = lay.m_w2_bf16[e];
                host_moe.v[idx_w2] = lay.v_w2_bf16[e];
            }
            
            host_qkv.p[l] = lay.qkv_weight;
            host_qkv.g[l] = lay.d_qkv_weight;
            host_qkv.m[l] = lay.m_qkv_bf16;
            host_qkv.v[l] = lay.v_qkv_bf16;
            host_qkv.p_fp8[l] = lay.qkv_weight_fp8;
            
            host_out_proj.p[l] = lay.out_proj_weight;
            host_out_proj.g[l] = lay.d_out_proj_weight;
            host_out_proj.m[l] = lay.m_out_bf16;
            host_out_proj.v[l] = lay.v_out_bf16;
            
            host_small.norm1[l] = lay.norm1_weight; host_small.d_norm1[l] = lay.d_norm1_weight; host_small.m_norm1[l] = lay.m_norm1_bf16; host_small.v_norm1[l] = lay.v_norm1_bf16;
            host_small.norm2[l] = lay.norm2_weight; host_small.d_norm2[l] = lay.d_norm2_weight; host_small.m_norm2[l] = lay.m_norm2_bf16; host_small.v_norm2[l] = lay.v_norm2_bf16;
            host_small.router[l] = lay.router_weight; host_small.d_router[l] = lay.d_router_weight; host_small.m_router[l] = lay.m_router_bf16; host_small.v_router[l] = lay.v_router_bf16;
            host_small.gamma[l] = lay.gamma_raw; host_small.d_gamma[l] = lay.d_gamma_raw; host_small.m_gamma[l] = lay.m_gamma; host_small.v_gamma[l] = lay.v_gamma;
            host_small.var[l] = lay.var_scale; host_small.d_var[l] = lay.d_var_scale; host_small.m_var[l] = lay.m_var; host_small.v_var[l] = lay.v_var;
        }
        host_small.final_norm = params.final_norm_weight;
        host_small.d_final_norm = params.d_final_norm_weight;
        host_small.m_final_norm = params.m_final_norm_bf16;
        host_small.v_final_norm = params.v_final_norm_bf16;
        
        cudaMemcpyAsync(ws.d_all_moe_experts, &host_moe, sizeof(AllMoEExpertsBF16), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_qkv, &host_qkv, sizeof(AllQKVBF16), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_out_proj, &host_out_proj, sizeof(AllOutProjBF16), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_small_params, &host_small, sizeof(AllSmallParamsBF16), cudaMemcpyHostToDevice, stream);
    } else {
        AllMoEExperts host_moe;
        AllQKV host_qkv;
        AllOutProj host_out_proj;
        AllSmallParams host_small;
        
        for (int l = 0; l < cfg.num_layers; ++l) {
            const auto& lay = params.layers[l];
            for (int e = 0; e < 4; ++e) {
                int idx_w1 = l * 8 + e;
                host_moe.p[idx_w1] = lay.w1_weights[e];
                host_moe.g[idx_w1] = lay.d_w1_weights[e];
                host_moe.m[idx_w1] = lay.m_w1[e];
                host_moe.v[idx_w1] = lay.v_w1[e];
                
                int idx_w2 = l * 8 + 4 + e;
                host_moe.p[idx_w2] = lay.w2_weights[e];
                host_moe.g[idx_w2] = lay.d_w2_weights[e];
                host_moe.m[idx_w2] = lay.m_w2[e];
                host_moe.v[idx_w2] = lay.v_w2[e];
            }
            
            host_qkv.p[l] = lay.qkv_weight;
            host_qkv.g[l] = lay.d_qkv_weight;
            host_qkv.m[l] = lay.m_qkv;
            host_qkv.v[l] = lay.v_qkv;
            
            host_out_proj.p[l] = lay.out_proj_weight;
            host_out_proj.g[l] = lay.d_out_proj_weight;
            host_out_proj.m[l] = lay.m_out;
            host_out_proj.v[l] = lay.v_out;
            
            host_small.norm1[l] = lay.norm1_weight; host_small.d_norm1[l] = lay.d_norm1_weight; host_small.m_norm1[l] = lay.m_norm1; host_small.v_norm1[l] = lay.v_norm1;
            host_small.norm2[l] = lay.norm2_weight; host_small.d_norm2[l] = lay.d_norm2_weight; host_small.m_norm2[l] = lay.m_norm2; host_small.v_norm2[l] = lay.v_norm2;
            host_small.router[l] = lay.router_weight; host_small.d_router[l] = lay.d_router_weight; host_small.m_router[l] = lay.m_router; host_small.v_router[l] = lay.v_router;
            host_small.gamma[l] = lay.gamma_raw; host_small.d_gamma[l] = lay.d_gamma_raw; host_small.m_gamma[l] = lay.m_gamma; host_small.v_gamma[l] = lay.v_gamma;
            host_small.var[l] = lay.var_scale; host_small.d_var[l] = lay.d_var_scale; host_small.m_var[l] = lay.m_var; host_small.v_var[l] = lay.v_var;
        }
        host_small.final_norm = params.final_norm_weight;
        host_small.d_final_norm = params.d_final_norm_weight;
        host_small.m_final_norm = params.m_final_norm;
        host_small.v_final_norm = params.v_final_norm;
        
        cudaMemcpyAsync(ws.d_all_moe_experts, &host_moe, sizeof(AllMoEExperts), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_qkv, &host_qkv, sizeof(AllQKV), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_out_proj, &host_out_proj, sizeof(AllOutProj), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ws.d_all_small_params, &host_small, sizeof(AllSmallParams), cudaMemcpyHostToDevice, stream);
    }
    
    ws.opt_tables_synced = true;
}

// ---------------------------------------------------------------------------
// 6. Orchestrate Fused Optimizer Step across All Model Parameters
// Reduced from 99 kernel launches down to 6 launches!
// ---------------------------------------------------------------------------
void run_fused_optimizer_step(
    FullModelParameters& params,
    FullModelWorkspace& ws,
    const FullJarvisConfig& cfg,
    float lr,
    cudaStream_t stream
) {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    
    sync_optimizer_tables(params, ws, cfg, stream);
    
    const int BLOCK = 256;
    
    if (cfg.use_fp8_moments) {
        // Phase 31: FP8 Moments Path (10 B/elem)
        // 1. Token Embeddings (50257 * 1024) - Launch 1
        launch_fused_adamw_update_fp8_moments(
            params.tok_emb_weight, params.d_tok_emb_weight, params.m_tok_emb_fp8, params.v_tok_emb_fp8,
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay, cfg.vocab_size * cfg.C, stream
        );
        
        // 2. LM Head (50304 * 1024) - Launch 2
        launch_fused_adamw_update_fp8_moments(
            params.lm_head_weight, params.d_lm_head_weight, params.m_lm_head_fp8, params.v_lm_head_fp8,
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay, cfg.vocab_pad * cfg.C, stream,
            (cfg.use_fp8_lm_head || cfg.use_fp8_lm_head_backward) ? params.lm_head_weight_fp8 : nullptr, 64.0f
        );
        
        // 3. Consolidated All MoE Experts (192 experts = 24 layers * 8 experts) - Launch 3
        int expert_elements = cfg.hidden_dim * cfg.C; // 2,097,152
        dim3 grid_moe((expert_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers * 8); // dim3(1024, 192)
        fused_adamw_all_moe_experts_fp8_moments_vec8_kernel<<<grid_moe, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllMoEExpertsFP8*>(ws.d_all_moe_experts),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 4. Consolidated All QKV (24 layers * 3 * C * C) - Launch 4
        int qkv_elements = 3 * cfg.C * cfg.C; // 3,145,728
        dim3 grid_qkv((qkv_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers); // dim3(1536, 24)
        fused_adamw_all_qkv_fp8_moments_vec8_kernel<<<grid_qkv, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllQKVFP8*>(ws.d_all_qkv),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 5. Consolidated All Out Proj (24 layers * C * C) - Launch 5
        int out_elements = cfg.C * cfg.C; // 1,048,576
        dim3 grid_out((out_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers); // dim3(512, 24)
        fused_adamw_all_out_proj_fp8_moments_vec8_kernel<<<grid_out, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllOutProjFP8*>(ws.d_all_out_proj),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 6. Consolidated All Small Params + Final Norm (25 blocks) - Launch 6
        fused_adamw_all_small_params_fp8_moments_kernel<<<cfg.num_layers + 1, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllSmallParamsFP8*>(ws.d_all_small_params),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay,
            cfg.C, cfg.E, cfg.H
        );
    } else if (cfg.use_bf16_moments) {
        // Phase 29: BF16 Moments Path (14 B/elem)
        // 1. Token Embeddings (50257 * 1024) - Launch 1
        launch_fused_adamw_update_bf16_moments(
            params.tok_emb_weight, params.d_tok_emb_weight, params.m_tok_emb_bf16, params.v_tok_emb_bf16,
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay, cfg.vocab_size * cfg.C, stream
        );
        
        // 2. LM Head (50304 * 1024) - Launch 2
        launch_fused_adamw_update_bf16_moments(
            params.lm_head_weight, params.d_lm_head_weight, params.m_lm_head_bf16, params.v_lm_head_bf16,
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay, cfg.vocab_pad * cfg.C, stream,
            (cfg.use_fp8_lm_head || cfg.use_fp8_lm_head_backward) ? params.lm_head_weight_fp8 : nullptr, 64.0f
        );
        
        // 3. Consolidated All MoE Experts (192 experts = 24 layers * 8 experts) - Launch 3
        int expert_elements = cfg.hidden_dim * cfg.C; // 2,097,152
        dim3 grid_moe((expert_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers * 8); // dim3(1024, 192)
        fused_adamw_all_moe_experts_bf16_moments_vec8_kernel<<<grid_moe, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllMoEExpertsBF16*>(ws.d_all_moe_experts),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 4. Consolidated All QKV (24 layers * 3 * C * C) - Launch 4
        int qkv_elements = 3 * cfg.C * cfg.C; // 3,145,728
        dim3 grid_qkv((qkv_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers); // dim3(1536, 24)
        fused_adamw_all_qkv_bf16_moments_vec8_kernel<<<grid_qkv, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllQKVBF16*>(ws.d_all_qkv),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 5. Consolidated All Out Proj (24 layers * C * C) - Launch 5
        int out_elements = cfg.C * cfg.C; // 1,048,576
        dim3 grid_out((out_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers); // dim3(512, 24)
        fused_adamw_all_out_proj_bf16_moments_vec8_kernel<<<grid_out, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllOutProjBF16*>(ws.d_all_out_proj),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 6. Consolidated All Small Params + Final Norm (25 blocks) - Launch 6
        fused_adamw_all_small_params_bf16_moments_kernel<<<cfg.num_layers + 1, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllSmallParamsBF16*>(ws.d_all_small_params),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay,
            cfg.C, cfg.E, cfg.H
        );
    } else {
        // Pure Reference FP32 Moments Path (22 B/elem)
        // 1. Token Embeddings (50257 * 1024) - Launch 1
        launch_fused_adamw_update_bf16(
            params.tok_emb_weight, params.d_tok_emb_weight, params.m_tok_emb, params.v_tok_emb,
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay, cfg.vocab_size * cfg.C, stream
        );
        
        // 2. LM Head (50304 * 1024) - Launch 2
        launch_fused_adamw_update_bf16(
            params.lm_head_weight, params.d_lm_head_weight, params.m_lm_head, params.v_lm_head,
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay, cfg.vocab_pad * cfg.C, stream
        );
        
        // 3. Consolidated All MoE Experts (192 experts = 24 layers * 8 experts) - Launch 3
        int expert_elements = cfg.hidden_dim * cfg.C; // 2,097,152
        dim3 grid_moe((expert_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers * 8); // dim3(1024, 192)
        fused_adamw_all_moe_experts_vec8_kernel<<<grid_moe, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllMoEExperts*>(ws.d_all_moe_experts),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 4. Consolidated All QKV (24 layers * 3 * C * C) - Launch 4
        int qkv_elements = 3 * cfg.C * cfg.C; // 3,145,728
        dim3 grid_qkv((qkv_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers); // dim3(1536, 24)
        fused_adamw_all_qkv_vec8_kernel<<<grid_qkv, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllQKV*>(ws.d_all_qkv),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 5. Consolidated All Out Proj (24 layers * C * C) - Launch 5
        int out_elements = cfg.C * cfg.C; // 1,048,576
        dim3 grid_out((out_elements / 8 + BLOCK - 1) / BLOCK, cfg.num_layers); // dim3(512, 24)
        fused_adamw_all_out_proj_vec8_kernel<<<grid_out, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllOutProj*>(ws.d_all_out_proj),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay
        );
        
        // 6. Consolidated All Small Params + Final Norm (25 blocks) - Launch 6
        fused_adamw_all_small_params_kernel<<<cfg.num_layers + 1, BLOCK, 0, stream>>>(
            reinterpret_cast<const AllSmallParams*>(ws.d_all_small_params),
            ws.clip_coef, lr, beta1, beta2, eps, weight_decay,
            cfg.C, cfg.E, cfg.H
        );
    }
}
