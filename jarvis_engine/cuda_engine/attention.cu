#include "attention.h"
#include "cublaslt_engine.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>

// ===========================================================================
// Native CUDA Associative Linear Attention Forward Implementation
// Target: NVIDIA RTX 5070 (Blackwell SM120)
// Exact Parity with Reference PyTorch AssociativeLinearAttention
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// 1. Static Rotary Tables Initialization Kernel (RoPE)
// ---------------------------------------------------------------------------
__global__ void init_rotary_tables_kernel(
    __nv_bfloat16* __restrict__ cos_tab, // (T, D) = (512, 64)
    __nv_bfloat16* __restrict__ sin_tab, // (T, D) = (512, 64)
    int T, int D, float base
) {
    int t = blockIdx.x; // [0, T - 1]
    int j = threadIdx.x; // [0, D / 2 - 1]
    if (t >= T || j >= D / 2) return;
    
    float inv_freq = 1.0f / powf(base, (float)(2 * j) / (float)D);
    float freq = (float)t * inv_freq;
    
    float c = cosf(freq);
    float s = sinf(freq);
    
    // freqs duplicated across both halves: [freqs, freqs]
    cos_tab[t * D + j] = __float2bfloat16(c);
    cos_tab[t * D + j + D / 2] = __float2bfloat16(c);
    
    sin_tab[t * D + j] = __float2bfloat16(s);
    sin_tab[t * D + j + D / 2] = __float2bfloat16(s);
}

// ---------------------------------------------------------------------------
// 2. Per-Layer Causal Decay Tables Kernel
// ---------------------------------------------------------------------------
__global__ void compute_decay_tables_kernel(
    const float* __restrict__ gamma_raw,      // (H,) = (16,)
    __nv_bfloat16* __restrict__ decay_mat,    // (H, cs, cs) = (16, 64, 64)
    __nv_bfloat16* __restrict__ gamma_cross,  // (H, cs) = (16, 64)
    __nv_bfloat16* __restrict__ gw,           // (H, cs) = (16, 64)
    float* __restrict__ gamma_c               // (H,) = (16,)
) {
    int h = blockIdx.x;  // head index [0, 15]
    int i = threadIdx.x; // chunk token index [0, 63]
    if (h >= 16 || i >= 64) return;
    
    float g_raw = gamma_raw[h];
    float sig = 1.0f / (1.0f + expf(-g_raw));
    float log_g = logf(sig);
    
    // 1. decay_mat row i for head h: exp(log_g * (i - j)) for j <= i, else 0
    int base_h = h * 4096;
    for (int j = 0; j < 64; ++j) {
        float val = 0.0f;
        if (i >= j) {
            val = expf(log_g * (float)(i - j));
        }
        decay_mat[base_h + i * 64 + j] = __float2bfloat16(val);
    }
    
    // 2. gamma_cross: exp(log_g * (i + 1))
    gamma_cross[h * 64 + i] = __float2bfloat16(expf(log_g * (float)(i + 1)));
    
    // 3. gw: exp(log_g * (63 - i))
    gw[h * 64 + i] = __float2bfloat16(expf(log_g * (float)(63 - i)));
    
    // 4. gamma_c: exp(log_g * 64) (written by thread 0)
    if (i == 0) {
        gamma_c[h] = expf(log_g * 64.0f);
    }
}

// ---------------------------------------------------------------------------
// 3. Fused Unpack + RoPE + ELU+1 + Q-scale + Vw Kernel
// ---------------------------------------------------------------------------
__global__ void fused_unpack_rope_elu_kernel(
    const __nv_bfloat16* __restrict__ layer_qkv,     // (M, 3 * C) = (2048, 3072)
    const __nv_bfloat16* __restrict__ cos_tab,       // (T, D) = (512, 64)
    const __nv_bfloat16* __restrict__ sin_tab,       // (T, D) = (512, 64)
    const __nv_bfloat16* __restrict__ gw_tab,        // (H, cs) = (16, 64)
    __nv_bfloat16* __restrict__ q_chunks,            // (512, 64, 64)
    __nv_bfloat16* __restrict__ k_chunks,            // (512, 64, 64)
    __nv_bfloat16* __restrict__ v_chunks,            // (512, 64, 64)
    __nv_bfloat16* __restrict__ v_w_chunks,          // (512, 64, 64)
    int total_pairs                                  // B * H * T * (D / 2) = 1,048,576
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    
    int d = idx % 32;
    int rem1 = idx / 32;
    int t = rem1 % 512;
    int rem2 = rem1 / 512;
    int h = rem2 % 16;
    int b = rem2 / 16;
    
    int k = t / 64; // chunk index [0, 7]
    int i = t % 64; // local index in chunk [0, 63]
    
    // Position in row of layer_qkv (2048, 3072)
    int m = b * 512 + t;
    int base_q1 = m * 3072 + h * 64 + d;
    int base_q2 = base_q1 + 32;
    int base_k1 = base_q1 + 1024;
    int base_k2 = base_q2 + 1024;
    int base_v1 = base_q1 + 2048;
    int base_v2 = base_q2 + 2048;
    
    float q1_in = __bfloat162float(layer_qkv[base_q1]);
    float q2_in = __bfloat162float(layer_qkv[base_q2]);
    float k1_in = __bfloat162float(layer_qkv[base_k1]);
    float k2_in = __bfloat162float(layer_qkv[base_k2]);
    float v1_in = __bfloat162float(layer_qkv[base_v1]);
    float v2_in = __bfloat162float(layer_qkv[base_v2]);
    
    // Rotary tables at (t, d) and (t, d+32)
    float c1 = __bfloat162float(cos_tab[t * 64 + d]);
    float s1 = __bfloat162float(sin_tab[t * 64 + d]);
    float c2 = __bfloat162float(cos_tab[t * 64 + d + 32]);
    float s2 = __bfloat162float(sin_tab[t * 64 + d + 32]);
    
    // 1. ELU + 1 & Q scaling (inv_sqrt_d = 0.125f for D=64)
    float q1 = ((q1_in > 0.0f) ? (q1_in + 1.0f) : expf(q1_in)) * 0.125f;
    float q2 = ((q2_in > 0.0f) ? (q2_in + 1.0f) : expf(q2_in)) * 0.125f;
    float k1 = (k1_in > 0.0f) ? (k1_in + 1.0f) : expf(k1_in);
    float k2 = (k2_in > 0.0f) ? (k2_in + 1.0f) : expf(k2_in);
    
    // 2. RoPE rotation
    float q1_rot = q1 * c1 - q2 * s1;
    float q2_rot = q2 * c2 + q1 * s2;
    
    float k1_rot = k1 * c1 - k2 * s1;
    float k2_rot = k2 * c2 + k1 * s2;
    
    // 3. Write outputs to chunk layout (512, 64, 64)
    int chunk_idx = (b * 16 + h) * 8 + k;
    int out_offset1 = (chunk_idx * 64 + i) * 64 + d;
    int out_offset2 = out_offset1 + 32;
    
    q_chunks[out_offset1] = __float2bfloat16(q1_rot);
    q_chunks[out_offset2] = __float2bfloat16(q2_rot);
    
    k_chunks[out_offset1] = __float2bfloat16(k1_rot);
    k_chunks[out_offset2] = __float2bfloat16(k2_rot);
    
    v_chunks[out_offset1] = __float2bfloat16(v1_in);
    v_chunks[out_offset2] = __float2bfloat16(v2_in);
    
    // 4. V * gw for delta_S computation
    float gw = __bfloat162float(gw_tab[h * 64 + i]);
    v_w_chunks[out_offset1] = __float2bfloat16(v1_in * gw);
    v_w_chunks[out_offset2] = __float2bfloat16(v2_in * gw);
}

// ---------------------------------------------------------------------------
// 4. Causal Decay Masking Kernel
// ---------------------------------------------------------------------------
__global__ void apply_causal_decay_kernel(
    __nv_bfloat16* __restrict__ scores,          // (B, H, Nc, cs, cs) = (512, 64, 64)
    const __nv_bfloat16* __restrict__ decay_mat, // (H, cs, cs) = (16, 64, 64)
    int total_elements                           // 512 * 64 * 64 = 2,097,152
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int flat_chunk = idx / 4096;
    int elem = idx % 4096;
    int h = (flat_chunk / 8) % 16;
    
    float s = __bfloat162float(scores[idx]);
    float d = __bfloat162float(decay_mat[h * 4096 + elem]);
    scores[idx] = __float2bfloat16(s * d);
}

// ---------------------------------------------------------------------------
// 5. Recurrent Chunk State Scan Forward Kernel (FP32 Accumulation)
// ---------------------------------------------------------------------------
__global__ void recurrent_chunk_state_scan_forward_kernel(
    const __nv_bfloat16* __restrict__ delta_S,    // (B, H, Nc, D, D)
    const float* __restrict__ gamma_c_tab,        // (H,)
    const __nv_bfloat16* __restrict__ h_prev,     // (B, H, D, D) or nullptr
    __nv_bfloat16* __restrict__ all_states,       // (B, H, Nc, D, D) carried forward
    __nv_bfloat16* __restrict__ h_last,           // (B, H, D, D) final state
    int B, int H, int Nc, int D
) {
    int bh_idx = blockIdx.x; // [0, B * H - 1]
    if (bh_idx >= B * H) return;
    
    int b = bh_idx / H;
    int h = bh_idx % H;
    float gamma_c = gamma_c_tab[h];
    
    int state_size = D * D; // 4096
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    for (int elem_idx = tid; elem_idx < state_size; elem_idx += num_threads) {
        float cur_s = 0.0f;
        if (h_prev != nullptr) {
            int prev_offset = (b * H + h) * state_size + elem_idx;
            cur_s = __bfloat162float(h_prev[prev_offset]);
        }
        
        for (int k = 0; k < Nc; ++k) {
            int out_offset = ((b * H + h) * Nc + k) * state_size + elem_idx;
            all_states[out_offset] = __float2bfloat16(cur_s);
            
            float ds_val = __bfloat162float(delta_S[out_offset]);
            cur_s = gamma_c * cur_s + ds_val;
        }
        
        if (h_last != nullptr) {
            int last_offset = (b * H + h) * state_size + elem_idx;
            h_last[last_offset] = __float2bfloat16(cur_s);
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Context Combination Kernel (Intra + Cross -> Layer Context (M, C))
// ---------------------------------------------------------------------------
__global__ void combine_intra_cross_kernel(
    const __nv_bfloat16* __restrict__ intra_out,   // (B, H, Nc, cs, D) = (512, 64, 64)
    const __nv_bfloat16* __restrict__ raw_cross,   // (B, H, Nc, cs, D) = (512, 64, 64)
    const __nv_bfloat16* __restrict__ gamma_cross, // (H, cs) = (16, 64)
    __nv_bfloat16* __restrict__ attn_context,      // (M, C) = (2048, 1024)
    int total_elements                             // 512 * 64 * 64 = 2,097,152
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    int d = idx % 64;
    int rem1 = idx / 64;
    int i = rem1 % 64;
    int rem2 = rem1 / 64;
    int k = rem2 % 8;
    int rem3 = rem2 / 8;
    int h = rem3 % 16;
    int b = rem3 / 16;
    
    float intra = __bfloat162float(intra_out[idx]);
    float cross_raw = __bfloat162float(raw_cross[idx]);
    float gc = __bfloat162float(gamma_cross[h * 64 + i]);
    float total = intra + cross_raw * gc;
    
    int m = (b * 8 + k) * 64 + i;
    int c = h * 64 + d;
    attn_context[m * 1024 + c] = __float2bfloat16(total);
}

} // anonymous namespace

// ===========================================================================
// Public Interface Implementations
// ===========================================================================

void init_attention_rotary_tables(
    FullModelWorkspace& ws,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
) {
    dim3 grid(cfg.T);
    dim3 block(cfg.D / 2);
    init_rotary_tables_kernel<<<grid, block, 0, stream>>>(
        ws.cos_tab, ws.sin_tab, cfg.T, cfg.D, 10000.0f
    );
}

void compute_attention_decay_tables(
    const float* gamma_raw,
    FullModelWorkspace& ws,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
) {
    compute_decay_tables_kernel<<<cfg.H, cfg.chunk_size, 0, stream>>>(
        gamma_raw, ws.decay_mat_tab, ws.gamma_cross_tab, ws.gw_tab, ws.gamma_c_tab
    );
}

void run_native_associative_attention_forward(
    FullModelWorkspace& ws,
    const LayerWeights& lay,
    int layer_idx,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
) {
    // 1. Compute layer causal decay tables from gamma_raw
    compute_attention_decay_tables(lay.gamma_raw, ws, cfg, stream);
    
    // 2. Fused Unpack + RoPE + ELU+1 + Q-scale + Vw
    int total_pairs = cfg.B * cfg.H * cfg.T * (cfg.D / 2); // 1,048,576
    int block_size = 256;
    int grid_size = (total_pairs + block_size - 1) / block_size;
    fused_unpack_rope_elu_kernel<<<grid_size, block_size, 0, stream>>>(
        ws.layer_qkv, ws.cos_tab, ws.sin_tab, ws.gw_tab,
        ws.attn_q_chunks, ws.attn_k_chunks, ws.attn_v_chunks, ws.attn_v_w_chunks,
        total_pairs
    );
    
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);
    
    float alpha = 1.0f, beta = 0.0f;
    int batch_count = cfg.B * cfg.H * (cfg.T / cfg.chunk_size); // 512
    int cs = cfg.chunk_size; // 64
    int D = cfg.D;           // 64
    long long int stride_mat = cs * D; // 4096
    
    // 3. Batched GEMM 1: raw = Q @ K^T: (512, cs, D) @ (512, D, cs) -> (512, cs, cs)
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        cs, cs, D,
        &alpha,
        ws.attn_k_chunks, CUDA_R_16BF, D, stride_mat,
        ws.attn_q_chunks, CUDA_R_16BF, D, stride_mat,
        &beta,
        ws.attn_scores, CUDA_R_16BF, cs, stride_mat,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // 4. scores = raw * decay_mat
    int total_scores = batch_count * stride_mat;
    apply_causal_decay_kernel<<<(total_scores + 255) / 256, 256, 0, stream>>>(
        ws.attn_scores, ws.decay_mat_tab, total_scores
    );
    
    // 5. Batched GEMM 2: intra_out = scores @ V: (512, cs, cs) @ (512, cs, D) -> (512, cs, D)
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, cs, cs,
        &alpha,
        ws.attn_v_chunks, CUDA_R_16BF, D, stride_mat,
        ws.attn_scores, CUDA_R_16BF, cs, stride_mat,
        &beta,
        ws.attn_intra_out, CUDA_R_16BF, D, stride_mat,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // 6. Batched GEMM 3: delta_S = V_w^T @ K: (512, D, cs) @ (512, cs, D) -> (512, D, D)
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        D, D, cs,
        &alpha,
        ws.attn_k_chunks, CUDA_R_16BF, D, stride_mat,
        ws.attn_v_w_chunks, CUDA_R_16BF, D, stride_mat,
        &beta,
        ws.attn_delta_s, CUDA_R_16BF, D, stride_mat,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // 7. Recurrent chunk state scan: S_{k+1} = gamma_c * S_k + delta_S_k
    recurrent_chunk_state_scan_forward_kernel<<<cfg.B * cfg.H, 256, 0, stream>>>(
        ws.attn_delta_s, ws.gamma_c_tab, ws.layer_attn_state[layer_idx],
        ws.attn_all_states, ws.layer_attn_state[layer_idx],
        cfg.B, cfg.H, cfg.T / cfg.chunk_size, cfg.D
    );
    
    // 8. Batched GEMM 4: raw_cross = Q @ state^T: (512, cs, D) @ (512, D, D) -> (512, cs, D)
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        D, cs, D,
        &alpha,
        ws.attn_all_states, CUDA_R_16BF, D, stride_mat,
        ws.attn_q_chunks, CUDA_R_16BF, D, stride_mat,
        &beta,
        ws.attn_cross_out, CUDA_R_16BF, D, stride_mat,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    
    // 9. Combine intra_out + raw_cross * gamma_cross -> ws.layer_attn_context (M, C)
    combine_intra_cross_kernel<<<(total_scores + 255) / 256, 256, 0, stream>>>(
        ws.attn_intra_out, ws.attn_cross_out, ws.gamma_cross_tab,
        ws.layer_attn_context, total_scores
    );
    
    // 10. Attention Out Projection GEMM: (M, C) @ (C, C).T -> ws.layer_attn_out (M, C)
    cublaslt_gemm_attn_out_fwd(
        ws.layer_attn_context, lay.out_proj_weight, ws.layer_attn_out,
        cfg.M(), cfg.C, stream
    );
}
