#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// ---------------------------------------------------------------------------
// 1. Fused ELU + 1 & Scaling Kernel
// ---------------------------------------------------------------------------
__global__ void fused_elu_rope_prep_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    float inv_sqrt_d,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float q = __bfloat162float(q_in[idx]);
    float k = __bfloat162float(k_in[idx]);
    
    float q_elu = (q > 0.0f) ? (q + 1.0f) : (expf(q));
    float k_elu = (k > 0.0f) ? (k + 1.0f) : (expf(k));
    
    q_out[idx] = __float2bfloat16(q_elu * inv_sqrt_d);
    k_out[idx] = __float2bfloat16(k_elu);
}

void launch_fused_elu_prep(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    float inv_sqrt_d,
    int total_elements,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    int grid = (total_elements + BLOCK - 1) / BLOCK;
    fused_elu_rope_prep_kernel<<<grid, BLOCK, 0, stream>>>(
        q_in, k_in, q_out, k_out, inv_sqrt_d, total_elements
    );
}

// ---------------------------------------------------------------------------
// 2. Fused RoPE (Rotary Position Embedding) Kernel
// For B=4, T=512, H=16, D=64: encodes relative token position
// ---------------------------------------------------------------------------
__global__ void fused_rope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int B, int T, int H, int D
) {
    // Each thread processes one pair (dim, dim + D/2) for a given (b, h, t)
    int t = blockIdx.x; // [0, T)
    int bh = blockIdx.y; // [0, B * H)
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x; // [0, D/2)
    
    if (tid >= D / 2) return;
    
    int half_d = D / 2;
    int base_offset = ((b * H + h) * T + t) * D;
    int idx1 = base_offset + tid;
    int idx2 = base_offset + tid + half_d;
    
    float c = cos_cache[t * D + tid];
    float s = sin_cache[t * D + tid];
    
    // Q rotation
    float q1 = __bfloat162float(q[idx1]);
    float q2 = __bfloat162float(q[idx2]);
    float q_rot1 = q1 * c - q2 * s;
    float q_rot2 = q2 * c + q1 * s;
    q[idx1] = __float2bfloat16(q_rot1);
    q[idx2] = __float2bfloat16(q_rot2);
    
    // K rotation
    float k1 = __bfloat162float(k[idx1]);
    float k2 = __bfloat162float(k[idx2]);
    float k_rot1 = k1 * c - k2 * s;
    float k_rot2 = k2 * c + k1 * s;
    k[idx1] = __float2bfloat16(k_rot1);
    k[idx2] = __float2bfloat16(k_rot2);
}

void launch_fused_rope(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const float* cos_cache,
    const float* sin_cache,
    int B, int T, int H, int D,
    cudaStream_t stream
) {
    dim3 grid(T, B * H);
    int block = D / 2;
    fused_rope_kernel<<<grid, block, 0, stream>>>(q, k, cos_cache, sin_cache, B, T, H, D);
}
