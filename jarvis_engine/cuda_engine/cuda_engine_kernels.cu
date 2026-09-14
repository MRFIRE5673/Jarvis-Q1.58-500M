#include "cuda_engine.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdio.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// 1. Memory Allocation Helpers
// ---------------------------------------------------------------------------
JarvisLayerWorkspace allocate_layer_workspace(const JarvisLayerConfig& cfg) {
    JarvisLayerWorkspace ws;
    memset(&ws, 0, sizeof(ws));
    
    int M = cfg.M();
    int C = cfg.C;
    int H = cfg.H;
    int D = cfg.D;
    int top_k = cfg.top_k;
    int hidden_dim = cfg.hidden_dim;
    int E = cfg.E;
    
    size_t total = 0;
    
    // Helper to allocate buffer
    auto alloc_bf16 = [&](__nv_bfloat16*& ptr, size_t count) {
        size_t bytes = count * sizeof(__nv_bfloat16);
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        total += bytes;
    };
    auto alloc_f32 = [&](float*& ptr, size_t count) {
        size_t bytes = count * sizeof(float);
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        total += bytes;
    };
    auto alloc_i32 = [&](int32_t*& ptr, size_t count) {
        size_t bytes = count * sizeof(int32_t);
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        total += bytes;
    };
    
    // 1. Forward Activations
    alloc_bf16(ws.x_norm1, M * C);
    alloc_f32(ws.rsqrt1, M);
    alloc_bf16(ws.qkv, M * 3 * C);
    alloc_bf16(ws.q_rot, M * C);
    alloc_bf16(ws.k_rot, M * C);
    alloc_bf16(ws.attn_out, M * C);
    alloc_bf16(ws.x1, M * C);
    alloc_bf16(ws.x_norm2, M * C);
    alloc_f32(ws.rsqrt2, M);
    alloc_f32(ws.router_logits, M * E);
    alloc_f32(ws.topk_gates, M * top_k);
    alloc_i32(ws.topk_idx, M * top_k);
    alloc_i32(ws.expert_counts, E);
    alloc_i32(ws.expert_offsets, E + 1);
    alloc_i32(ws.scatter_map, M * top_k);
    alloc_i32(ws.gather_map, M * top_k);
    alloc_i32(ws.gate_idx_map, M * top_k);
    alloc_bf16(ws.dispatched_x, M * top_k * C);
    alloc_bf16(ws.h1, M * top_k * hidden_dim);
    alloc_bf16(ws.act, M * top_k * hidden_dim);
    alloc_bf16(ws.dispatched_y, M * top_k * C);
    alloc_bf16(ws.moe_out, M * C);
    alloc_bf16(ws.h_out, M * C);
    alloc_bf16(ws.h_last, cfg.B * C);
    alloc_bf16(ws.x2, M * C);
    
    // 2. Backward Gradients
    alloc_bf16(ws.grad_x2, M * C);
    alloc_bf16(ws.grad_h_out, M * C);
    alloc_bf16(ws.grad_moe_out, M * C);
    alloc_bf16(ws.grad_dispatched_y, M * top_k * C);
    alloc_f32(ws.grad_topk_gates, M * top_k);
    alloc_bf16(ws.grad_act, M * top_k * hidden_dim);
    alloc_bf16(ws.grad_h1, M * top_k * hidden_dim);
    alloc_bf16(ws.grad_dispatched_x, M * top_k * C);
    alloc_bf16(ws.grad_x_norm2, M * C);
    alloc_bf16(ws.grad_x1, M * C);
    alloc_bf16(ws.grad_attn_out, M * C);
    alloc_bf16(ws.grad_x_norm1, M * C);
    alloc_bf16(ws.grad_x, M * C);
    
    ws.total_bytes = total;
    ws.is_initialized = true;
    return ws;
}

void free_layer_workspace(JarvisLayerWorkspace& ws) {
    if (!ws.is_initialized) return;
    
    auto free_p = [](void*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    
    free_p((void*&)ws.x_norm1);
    free_p((void*&)ws.rsqrt1);
    free_p((void*&)ws.qkv);
    free_p((void*&)ws.q_rot);
    free_p((void*&)ws.k_rot);
    free_p((void*&)ws.attn_out);
    free_p((void*&)ws.x1);
    free_p((void*&)ws.x_norm2);
    free_p((void*&)ws.rsqrt2);
    free_p((void*&)ws.router_logits);
    free_p((void*&)ws.topk_gates);
    free_p((void*&)ws.topk_idx);
    free_p((void*&)ws.expert_counts);
    free_p((void*&)ws.expert_offsets);
    free_p((void*&)ws.scatter_map);
    free_p((void*&)ws.gather_map);
    free_p((void*&)ws.gate_idx_map);
    free_p((void*&)ws.dispatched_x);
    free_p((void*&)ws.h1);
    free_p((void*&)ws.act);
    free_p((void*&)ws.dispatched_y);
    free_p((void*&)ws.moe_out);
    free_p((void*&)ws.h_out);
    free_p((void*&)ws.h_last);
    free_p((void*&)ws.x2);
    
    free_p((void*&)ws.grad_x2);
    free_p((void*&)ws.grad_h_out);
    free_p((void*&)ws.grad_moe_out);
    free_p((void*&)ws.grad_dispatched_y);
    free_p((void*&)ws.grad_topk_gates);
    free_p((void*&)ws.grad_act);
    free_p((void*&)ws.grad_h1);
    free_p((void*&)ws.grad_dispatched_x);
    free_p((void*&)ws.grad_x_norm2);
    free_p((void*&)ws.grad_x1);
    free_p((void*&)ws.grad_attn_out);
    free_p((void*&)ws.grad_x_norm1);
    free_p((void*&)ws.grad_x);
    
    ws.total_bytes = 0;
    ws.is_initialized = false;
}

// ---------------------------------------------------------------------------
// 2. Fused RMSNorm & Residual Addition Kernels
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out_add,
    __nv_bfloat16* __restrict__ out_norm,
    float* __restrict__ rsqrt_out,
    int M, int C, float eps,
    __nv_fp8_e4m3* __restrict__ out_norm_fp8 = nullptr,
    float fp8_scale = 1.0f
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    
    const __nv_bfloat16* row_x = x + row * C;
    const __nv_bfloat16* row_res = res ? (res + row * C) : nullptr;
    __nv_bfloat16* row_add = out_add ? (out_add + row * C) : nullptr;
    __nv_bfloat16* row_norm = out_norm ? (out_norm + row * C) : nullptr;
    __nv_fp8_e4m3* row_norm_fp8 = out_norm_fp8 ? (out_norm_fp8 + row * C) : nullptr;
    
    // Phase 1: Sum of squares + Register Caching (EXP-28-002 / BUG-009)
    __nv_bfloat16 cached_val[8];
    int item_idx = 0;
    
    for (int col = tid; col < C; col += BLOCK_SIZE) {
        float val = __bfloat162float(row_x[col]);
        if (row_res) {
            val += __bfloat162float(row_res[col]);
        }
        __nv_bfloat16 val_bf16 = __float2bfloat16(val);
        if (row_add) {
            row_add[col] = val_bf16;
        }
        sum_sq += val * val;
        if (item_idx < 8) {
            cached_val[item_idx++] = row_add ? val_bf16 : row_x[col];
        }
    }
    
    // Block-level reduction
    __shared__ float s_sum[32];
    int lane = tid % 32;
    int wid = tid / 32;
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    if (lane == 0) {
        s_sum[wid] = sum_sq;
    }
    __syncthreads();
    
    float block_sum = 0.0f;
    if (wid == 0) {
        block_sum = (lane < (BLOCK_SIZE / 32)) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            s_sum[0] = block_sum;
        }
    }
    __syncthreads();
    
    float mean_sq = s_sum[0] / (float)C;
    float rsqrt_val = rsqrtf(mean_sq + eps);
    if (tid == 0 && rsqrt_out) {
        rsqrt_out[row] = rsqrt_val;
    }
    
    // Phase 2: Normalize and scale using cached register values (zero DRAM re-read)
    item_idx = 0;
    for (int col = tid; col < C; col += BLOCK_SIZE) {
        float val = (item_idx < 8) ? __bfloat162float(cached_val[item_idx++])
                                   : (row_add ? __bfloat162float(row_add[col]) : __bfloat162float(row_x[col]));
        float w = __bfloat162float(weight[col]);
        float norm_val = val * rsqrt_val * w;
        if (row_norm) {
            row_norm[col] = __float2bfloat16(norm_val);
        }
        if (row_norm_fp8) {
            row_norm_fp8[col] = __nv_fp8_e4m3(norm_val * fp8_scale);
        }
    }
}

void launch_fused_add_rmsnorm_fwd(
    const __nv_bfloat16* x,
    const __nv_bfloat16* res,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out_add,
    __nv_bfloat16* out_norm,
    float* rsqrt,
    int M, int C, float eps,
    cudaStream_t stream,
    __nv_fp8_e4m3* out_norm_fp8,
    float fp8_scale
) {
    const int BLOCK_SIZE = 256;
    fused_add_rmsnorm_fwd_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        x, res, weight, out_add, out_norm, rsqrt, M, C, eps, out_norm_fp8, fp8_scale
    );
}

void launch_fused_rmsnorm_fwd(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out_norm,
    float* rsqrt,
    int M, int C, float eps,
    cudaStream_t stream,
    __nv_fp8_e4m3* out_norm_fp8,
    float fp8_scale
) {
    launch_fused_add_rmsnorm_fwd(x, nullptr, weight, nullptr, out_norm, rsqrt, M, C, eps, stream, out_norm_fp8, fp8_scale);
}

template<int BLOCK_SIZE>
__global__ void fused_rmsnorm_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ rsqrt_in,
    __nv_bfloat16* __restrict__ grad_x,
    __nv_bfloat16* __restrict__ grad_weight,
    int M, int C,
    __nv_fp8_e4m3* __restrict__ grad_x_fp8 = nullptr,
    float scale_fp8 = 1.0f
) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    
    const __nv_bfloat16* row_gy = grad_out + row * C;
    const __nv_bfloat16* row_x = x + row * C;
    __nv_bfloat16* row_gx = grad_x + row * C;
    __nv_fp8_e4m3* row_gx_fp8 = grad_x_fp8 ? (grad_x_fp8 + row * C) : nullptr;
    float rsqrt_val = rsqrt_in[row];
    
    float sum_x_w_gy = 0.0f;
    for (int col = tid; col < C; col += BLOCK_SIZE) {
        float gy = __bfloat162float(row_gy[col]);
        float xi = __bfloat162float(row_x[col]);
        float wi = __bfloat162float(weight[col]);
        sum_x_w_gy += xi * wi * gy;
    }
    
    // Warp and block reduction
    __shared__ float s_sum[32];
    int lane = tid % 32;
    int wid = tid / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_x_w_gy += __shfl_down_sync(0xffffffff, sum_x_w_gy, offset);
    }
    if (lane == 0) s_sum[wid] = sum_x_w_gy;
    __syncthreads();
    
    float block_sum = 0.0f;
    if (wid == 0) {
        block_sum = (lane < (BLOCK_SIZE / 32)) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) s_sum[0] = block_sum;
    }
    __syncthreads();
    
    float coeff = (rsqrt_val * rsqrt_val * rsqrt_val / (float)C) * s_sum[0];
    
    for (int col = tid; col < C; col += BLOCK_SIZE) {
        float gy = __bfloat162float(row_gy[col]);
        float xi = __bfloat162float(row_x[col]);
        float wi = __bfloat162float(weight[col]);
        float dxi = rsqrt_val * wi * gy - coeff * xi;
        row_gx[col] = __float2bfloat16(dxi);
        if (row_gx_fp8) {
            row_gx_fp8[col] = __nv_fp8_e4m3(dxi * scale_fp8);
        }
    }
}

void launch_fused_rmsnorm_bwd(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const float* rsqrt,
    __nv_bfloat16* grad_x,
    __nv_bfloat16* grad_weight,
    int M, int C,
    cudaStream_t stream,
    __nv_fp8_e4m3* grad_x_fp8,
    float scale_fp8
) {
    const int BLOCK_SIZE = 256;
    fused_rmsnorm_bwd_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        grad_out, x, weight, rsqrt, grad_x, grad_weight, M, C, grad_x_fp8, scale_fp8
    );
}

// ---------------------------------------------------------------------------
// 3. Fused GELU Kernels
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gelu_fwd_val(float x) {
    // 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_bwd_val(float dy, float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = k0 * (x + k1 * x3);
    float tanh_val = tanhf(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float d_inner = k0 * (1.0f + 3.0f * k1 * x2);
    float dgelu = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
    return dy * dgelu;
}

__global__ void fused_gelu_fwd_vec8_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < N) {
        uint4 raw_in = *reinterpret_cast<const uint4*>(in + idx);
        const __nv_bfloat16* p_in = reinterpret_cast<const __nv_bfloat16*>(&raw_in);
        __nv_bfloat16 res[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x = __bfloat162float(p_in[k]);
            res[k] = __float2bfloat16(gelu_fwd_val(x));
        }
        *reinterpret_cast<uint4*>(out + idx) = *reinterpret_cast<uint4*>(res);
    } else {
        for (int k = 0; k < 8 && idx + k < N; ++k) {
            float x = __bfloat162float(in[idx + k]);
            out[idx + k] = __float2bfloat16(gelu_fwd_val(x));
        }
    }
}

void launch_fused_gelu_fwd(const __nv_bfloat16* in, __nv_bfloat16* out, int num_elements, cudaStream_t stream) {
    const int BLOCK = 256;
    int grid = ((num_elements + 7) / 8 + BLOCK - 1) / BLOCK;
    fused_gelu_fwd_vec8_kernel<<<grid, BLOCK, 0, stream>>>(in, out, num_elements);
}

__global__ void fused_gelu_bwd_kernel(const __nv_bfloat16* grad_out, const __nv_bfloat16* in, __nv_bfloat16* grad_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float dy = __bfloat162float(grad_out[idx]);
        float x = __bfloat162float(in[idx]);
        grad_in[idx] = __float2bfloat16(gelu_bwd_val(dy, x));
    }
}

void launch_fused_gelu_bwd(const __nv_bfloat16* grad_out, const __nv_bfloat16* in, __nv_bfloat16* grad_in, int num_elements, cudaStream_t stream) {
    const int BLOCK = 256;
    int grid = (num_elements + BLOCK - 1) / BLOCK;
    fused_gelu_bwd_kernel<<<grid, BLOCK, 0, stream>>>(grad_out, in, grad_in, num_elements);
}

// ---------------------------------------------------------------------------
// 4. Vectorized Residual Addition
// ---------------------------------------------------------------------------
__global__ void fused_add_residual_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(va + vb);
    }
}

void launch_fused_add_residual(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int num_elements, cudaStream_t stream) {
    const int BLOCK = 256;
    int grid = (num_elements + BLOCK - 1) / BLOCK;
    fused_add_residual_kernel<<<grid, BLOCK, 0, stream>>>(a, b, out, num_elements);
}
