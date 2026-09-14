#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Forward Gating & Maps
void launch_moe_top2_gating(
    const __nv_bfloat16* logits,
    float* topk_gates,
    int32_t* topk_idx,
    float* l_bal,
    int M, int E, float noise_std, bool training,
    cudaStream_t stream
);

void launch_moe_compute_maps(
    const int32_t* topk_idx,
    int32_t* scatter_map,
    int32_t* gather_map,
    int32_t* gate_idx_map,
    int32_t* expert_offsets,
    int M, int E,
    cudaStream_t stream
);

void launch_moe_dispatch_gather(
    const __nv_bfloat16* x,
    const int32_t* gather_map,
    __nv_bfloat16* dispatched_x,
    int total_dispatched, int C,
    cudaStream_t stream
);

void launch_moe_scatter_combine(
    const __nv_bfloat16* dispatched_y,
    const float* topk_gates,
    const int32_t* scatter_map,
    __nv_bfloat16* out,
    int M, int C,
    cudaStream_t stream
);

void launch_moe_scatter_combine_add_residual(
    const __nv_bfloat16* dispatched_y,
    const float* topk_gates,
    const int32_t* scatter_map,
    const __nv_bfloat16* x1,
    __nv_bfloat16* x2,
    int M, int C,
    cudaStream_t stream,
    __nv_fp8_e4m3* x2_fp8 = nullptr,
    float scale_fp8 = 16.0f
);

// Grouped Expert Forward GEMMs
void launch_moe_grouped_gemm_fwd_w1(
    const __nv_bfloat16* dispatched_x,
    __nv_bfloat16* const* w1_weights,
    const int32_t* expert_offsets,
    __nv_bfloat16* h1,
    int total_tokens, int C, int hidden_dim, int E,
    cudaStream_t stream
);

void launch_moe_grouped_gemm_fwd_w2(
    const __nv_bfloat16* act,
    __nv_bfloat16* const* w2_weights,
    const int32_t* expert_offsets,
    __nv_bfloat16* dispatched_y,
    int total_tokens, int hidden_dim, int C, int E,
    cudaStream_t stream
);

// Analytical Backward Kernels
void launch_moe_scatter_backward(
    const __nv_bfloat16* grad_moe_out,
    const __nv_bfloat16* dispatched_y,
    const float* topk_gates,
    const int32_t* gather_map,
    const int32_t* gate_idx_map,
    const int32_t* scatter_map,
    __nv_bfloat16* grad_dispatched_y,
    float* grad_topk_gates,
    int M, int total_dispatched, int C,
    cudaStream_t stream
);

void launch_moe_grouped_gemm_w2_bwd(
    const __nv_bfloat16* grad_dispatched_y,
    const __nv_bfloat16* act,
    __nv_bfloat16* const* w2_weights,
    const int32_t* expert_offsets,
    __nv_bfloat16* const* d_w2_weights,
    __nv_bfloat16* grad_act,
    int total_tokens, int hidden_dim, int C, int E, float beta,
    cudaStream_t stream
);

void launch_moe_grouped_gemm_w1_bwd(
    const __nv_bfloat16* grad_h1,
    const __nv_bfloat16* dispatched_x,
    __nv_bfloat16* const* w1_weights,
    const int32_t* expert_offsets,
    __nv_bfloat16* const* d_w1_weights,
    __nv_bfloat16* grad_dispatched_x,
    int total_tokens, int C, int hidden_dim, int E, float beta,
    cudaStream_t stream
);

void launch_moe_gather_backward(
    const __nv_bfloat16* grad_dispatched_x,
    const int32_t* scatter_map,
    __nv_bfloat16* grad_x_expert,
    int M, int top_k, int C,
    cudaStream_t stream
);

void launch_moe_router_backward(
    const float* grad_topk_gates,
    const __nv_bfloat16* router_logits,
    const int32_t* topk_idx,
    __nv_bfloat16* grad_router_logits,
    int M, int E,
    cudaStream_t stream
);

void launch_moe_router_gemms_bwd(
    const __nv_bfloat16* grad_router_logits,
    const __nv_bfloat16* layer_x_norm2,
    const __nv_bfloat16* router_weight,
    const __nv_bfloat16* grad_x_expert,
    __nv_bfloat16* d_router_weight,
    __nv_bfloat16* grad_x_norm2,
    int M, int C, int E, float beta,
    cudaStream_t stream
);
