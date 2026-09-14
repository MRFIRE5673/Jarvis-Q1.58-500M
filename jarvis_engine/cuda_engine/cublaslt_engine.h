#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstddef>

// Lifecycle
void init_cublaslt_engine(size_t workspace_bytes = 64 * 1024 * 1024);
void cleanup_cublaslt_engine();

// Forward GEMMs (Pure native cuBLASLt execution on Tensor Cores)
void cublaslt_gemm_qkv_fwd(
    const __nv_bfloat16* x, const __nv_bfloat16* w_qkv, __nv_bfloat16* out_qkv,
    int M, int C, cudaStream_t stream
);

void cublaslt_gemm_attn_out_fwd(
    const __nv_bfloat16* q_chunk, const __nv_bfloat16* w_out, __nv_bfloat16* out_attn,
    int M, int C, cudaStream_t stream
);

void cublaslt_gemm_router_fwd(
    const __nv_bfloat16* x, const __nv_bfloat16* w_router, __nv_bfloat16* router_logits,
    int M, int C, int E, cudaStream_t stream
);

void cublaslt_gemm_moe_w1_fwd(
    const __nv_bfloat16* disp_x, const __nv_bfloat16* w1, __nv_bfloat16* h1,
    int total_dispatched, int C, int hidden_dim, cudaStream_t stream
);

void cublaslt_gemm_moe_w2_fwd(
    const __nv_bfloat16* act, const __nv_bfloat16* w2, __nv_bfloat16* disp_y,
    int total_dispatched, int hidden_dim, int C, cudaStream_t stream
);

// Phase 28: FP8 MoE GEMMs (CUDA_R_8F_E4M3 inputs, FP32 accumulator, BF16 output)
void cublaslt_gemm_moe_w1_fp8(
    const __nv_fp8_e4m3* disp_x, const __nv_fp8_e4m3* w1, __nv_bfloat16* h1,
    int total_dispatched, int C, int hidden_dim, float alpha, cudaStream_t stream
);

void cublaslt_gemm_moe_w2_fp8(
    const __nv_fp8_e4m3* act, const __nv_fp8_e4m3* w2, __nv_bfloat16* disp_y,
    int total_dispatched, int hidden_dim, int C, float alpha, cudaStream_t stream
);

void cublaslt_gemm_lm_head_fwd(
    const __nv_bfloat16* final_norm, const __nv_bfloat16* lm_head_w, __nv_bfloat16* logits,
    int M, int C, int vocab_pad, cudaStream_t stream
);

// Phase 28: FP8 LM Head Forward GEMM (CUDA_R_8F_E4M3 inputs, FP32 accumulator, BF16 output)
void cublaslt_gemm_lm_head_fwd_fp8(
    const __nv_fp8_e4m3* final_norm, const __nv_fp8_e4m3* lm_head_w, __nv_bfloat16* logits,
    int M, int C, int vocab_pad, float alpha, cudaStream_t stream
);

// Backward GEMMs
void cublaslt_gemm_lm_head_bwd_dx(
    const __nv_bfloat16* d_logits, const __nv_bfloat16* lm_head_w, __nv_bfloat16* d_final_norm,
    int M, int vocab_pad, int C, cudaStream_t stream
);

void cublaslt_gemm_lm_head_bwd_dx_fp8(
    const __nv_fp8_e4m3* d_logits, const __nv_fp8_e4m3* lm_head_w, __nv_bfloat16* d_final_norm,
    int M, int vocab_pad, int C, float alpha, cudaStream_t stream
);

void cublaslt_gemm_lm_head_bwd_dw(
    const __nv_bfloat16* d_logits, const __nv_bfloat16* final_norm, __nv_bfloat16* d_lm_head_w,
    int M, int vocab_pad, int C, cudaStream_t stream, float beta = 1.0f
);

void cublaslt_gemm_lm_head_bwd_dw_fp8(
    const __nv_fp8_e4m3* d_logits, const __nv_fp8_e4m3* final_norm, __nv_bfloat16* d_lm_head_w,
    int M, int vocab_pad, int C, float alpha, float beta, cudaStream_t stream
);

void cublaslt_gemm_qkv_bwd_dw_slice(
    const __nv_bfloat16* d_x_norm, const __nv_bfloat16* stashed_x, __nv_bfloat16* d_slice,
    int M, int C, cudaStream_t stream, float beta = 1.0f
);

// 128-bit vectorized replication of QKV dW slice 0 to slice 1 and slice 2
void replicate_qkv_dw_slices(__nv_bfloat16* d_qkv_weight, int C, cudaStream_t stream);
