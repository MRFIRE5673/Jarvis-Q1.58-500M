#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

// ---------------------------------------------------------------------------
// 1. Fused Cross-Entropy Loss & Analytical dLogits Kernel (Online Softmax)
// Single-pass reduction for global max and sum_exp eliminates redundant DRAM reads
// Phase 31: Direct FP8 dLogits output eliminates standalone quantization pass
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void fused_cross_entropy_bwd_kernel(
    const __nv_bfloat16* __restrict__ logits, // (M, vocab_pad)
    const int32_t* __restrict__ targets,       // (M,)
    __nv_bfloat16* __restrict__ d_logits,     // (M, vocab_pad)
    float* __restrict__ loss_out,              // (1)
    int M, int vocab_size, int vocab_pad, float loss_scale,
    __nv_fp8_e4m3* __restrict__ d_logits_fp8 = nullptr,
    float scale_fp8 = 1048576.0f
) {
    int row = blockIdx.x; // Token index in [0, M)
    if (row >= M) return;
    
    int tid = threadIdx.x;
    int target = targets[row];
    const __nv_bfloat16* row_logits = logits + (size_t)row * vocab_pad;
    __nv_bfloat16* row_dlogits = d_logits + (size_t)row * vocab_pad;
    __nv_fp8_e4m3* row_dlogits_fp8 = d_logits_fp8 ? (d_logits_fp8 + (size_t)row * vocab_pad) : nullptr;
    
    // Pass 1: Online Softmax (Simultaneous global max and sum of exp in a single read pass)
    float m_i = -1e30f;
    float d_i = 0.0f;
    for (int col = tid; col < vocab_size; col += BLOCK_SIZE) {
        float val = __bfloat162float(row_logits[col]);
        float m_new = fmaxf(m_i, val);
        d_i = d_i * expf(m_i - m_new) + expf(val - m_new);
        m_i = m_new;
    }
    
    // Warp-level online softmax reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_m = __shfl_down_sync(0xffffffff, m_i, offset);
        float other_d = __shfl_down_sync(0xffffffff, d_i, offset);
        float m_new = fmaxf(m_i, other_m);
        d_i = d_i * expf(m_i - m_new) + other_d * expf(other_m - m_new);
        m_i = m_new;
    }
    
    __shared__ float s_m[32];
    __shared__ float s_d[32];
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) {
        s_m[wid] = m_i;
        s_d[wid] = d_i;
    }
    __syncthreads();
    
    // Block-level online reduction
    if (wid == 0) {
        float block_m = (lane < (BLOCK_SIZE / 32)) ? s_m[lane] : -1e30f;
        float block_d = (lane < (BLOCK_SIZE / 32)) ? s_d[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_m = __shfl_down_sync(0xffffffff, block_m, offset);
            float other_d = __shfl_down_sync(0xffffffff, block_d, offset);
            float m_new = fmaxf(block_m, other_m);
            block_d = block_d * expf(block_m - m_new) + other_d * expf(other_m - m_new);
            block_m = m_new;
        }
        if (lane == 0) {
            s_m[0] = block_m;
            s_d[0] = block_d;
        }
    }
    __syncthreads();
    
    float global_max = s_m[0];
    float global_sum_exp = s_d[0];
    float inv_sum_exp = 1.0f / (global_sum_exp + 1e-12f);
    
    // Target loss contribution
    if (tid == 0 && target >= 0 && target < vocab_size) {
        float target_val = __bfloat162float(row_logits[target]);
        float ce_loss = -(target_val - global_max - logf(global_sum_exp));
        atomicAdd(loss_out, (ce_loss * loss_scale) / (float)M);
    }
    
    // Pass 2: Analytical dLogits = (prob - 1(col == target)) * loss_scale / M
    float norm_factor = loss_scale / (float)M;
    for (int col = tid; col < vocab_pad; col += BLOCK_SIZE) {
        if (col < vocab_size) {
            float val = __bfloat162float(row_logits[col]);
            float prob = expf(val - global_max) * inv_sum_exp;
            float grad = (col == target) ? (prob - 1.0f) : prob;
            float scaled_grad = grad * norm_factor;
            row_dlogits[col] = __float2bfloat16(scaled_grad);
            if (row_dlogits_fp8) {
                row_dlogits_fp8[col] = __nv_fp8_e4m3(scaled_grad * scale_fp8);
            }
        } else {
            row_dlogits[col] = __float2bfloat16(0.0f); // Padded columns have zero gradient
            if (row_dlogits_fp8) {
                row_dlogits_fp8[col] = __nv_fp8_e4m3(0.0f);
            }
        }
    }
}

void launch_fused_cross_entropy_bwd(
    const __nv_bfloat16* logits,
    const int32_t* targets,
    __nv_bfloat16* d_logits,
    float* loss_out,
    int M, int vocab_size, int vocab_pad, float loss_scale,
    cudaStream_t stream,
    __nv_fp8_e4m3* d_logits_fp8,
    float scale_fp8
) {
    const int BLOCK = 256;
    fused_cross_entropy_bwd_kernel<BLOCK><<<M, BLOCK, 0, stream>>>(
        logits, targets, d_logits, loss_out, M, vocab_size, vocab_pad, loss_scale,
        d_logits_fp8, scale_fp8
    );
}

// ---------------------------------------------------------------------------
// 2. Token Embedding Forward (Gather) & Backward (Scatter Add)
// ---------------------------------------------------------------------------
__global__ void tok_emb_fwd_kernel(
    const int32_t* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ emb_weight,
    __nv_bfloat16* __restrict__ out,
    int M, int C
) {
    int token_idx = blockIdx.x;
    if (token_idx >= M) return;
    
    int id = input_ids[token_idx];
    const __nv_bfloat16* src = emb_weight + (size_t)id * C;
    __nv_bfloat16* dst = out + (size_t)token_idx * C;
    
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        dst[c] = src[c];
    }
}

void launch_tok_emb_fwd(
    const int32_t* input_ids,
    const __nv_bfloat16* emb_weight,
    __nv_bfloat16* out,
    int M, int C,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    tok_emb_fwd_kernel<<<M, BLOCK, 0, stream>>>(input_ids, emb_weight, out, M, C);
}

__global__ void tok_emb_bwd_kernel(
    const __nv_bfloat16* __restrict__ d_out,
    const int32_t* __restrict__ input_ids,
    __nv_bfloat16* __restrict__ d_emb_weight,
    int M, int C
) {
    int token_idx = blockIdx.x;
    if (token_idx >= M) return;
    
    int id = input_ids[token_idx];
    const __nv_bfloat16* src = d_out + (size_t)token_idx * C;
    __nv_bfloat16* dst = d_emb_weight + (size_t)id * C;
    
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = __bfloat162float(src[c]);
        // Simple atomic accumulation for word embeddings
        #if __CUDA_ARCH__ >= 800
        atomicAdd(reinterpret_cast<__nv_bfloat16*>(dst + c), src[c]);
        #else
        // Software atomic fallback
        atomicAdd((float*)dst + c, val);
        #endif
    }
}

void launch_tok_emb_bwd(
    const __nv_bfloat16* d_out,
    const int32_t* input_ids,
    __nv_bfloat16* d_emb_weight,
    int M, int C,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    tok_emb_bwd_kernel<<<M, BLOCK, 0, stream>>>(d_out, input_ids, d_emb_weight, M, C);
}
