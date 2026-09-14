#include "full_engine.h"
#include "optimizer.h"
#include "cublaslt_engine.h"
#include "attention.h"
#include "moe.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <utility>

// External CUDA kernel declarations
void launch_fused_rmsnorm_fwd(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out_norm,
    float* rsqrt, int M, int C, float eps, cudaStream_t stream,
    __nv_fp8_e4m3* out_norm_fp8 = nullptr, float fp8_scale = 1.0f
);

void launch_fused_add_rmsnorm_fwd(
    const __nv_bfloat16* x, const __nv_bfloat16* res, const __nv_bfloat16* weight,
    __nv_bfloat16* out_add, __nv_bfloat16* out_norm, float* rsqrt,
    int M, int C, float eps, cudaStream_t stream,
    __nv_fp8_e4m3* out_norm_fp8 = nullptr, float fp8_scale = 1.0f
);

void launch_fused_rmsnorm_bwd(
    const __nv_bfloat16* grad_out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const float* rsqrt, __nv_bfloat16* grad_x, __nv_bfloat16* grad_weight,
    int M, int C, cudaStream_t stream,
    __nv_fp8_e4m3* grad_x_fp8 = nullptr, float scale_fp8 = 1.0f,
    float beta = 0.0f
);

void launch_fused_gelu_fwd(const __nv_bfloat16* in, __nv_bfloat16* out, int num_elements, cudaStream_t stream);
void launch_fused_gelu_bwd(const __nv_bfloat16* grad_out, const __nv_bfloat16* in, __nv_bfloat16* grad_in, int num_elements, cudaStream_t stream);
void launch_fused_add_residual(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int num_elements, cudaStream_t stream);

void launch_fused_cross_entropy_bwd(
    const __nv_bfloat16* logits, const int32_t* targets, __nv_bfloat16* d_logits,
    float* loss_out, int M, int vocab_size, int vocab_pad, float loss_scale, cudaStream_t stream,
    __nv_fp8_e4m3* d_logits_fp8 = nullptr, float scale_fp8 = 1048576.0f
);

void launch_tok_emb_fwd(
    const int32_t* input_ids, const __nv_bfloat16* emb_weight,
    __nv_bfloat16* out, int M, int C, cudaStream_t stream,
    __nv_fp8_e4m3* out_fp8 = nullptr, float scale_fp8 = 16.0f
);

void launch_tok_emb_bwd(
    const __nv_bfloat16* d_out, const int32_t* input_ids, __nv_bfloat16* d_emb_weight,
    int M, int C, cudaStream_t stream
);

void launch_moe_dispatch_gather_fp8(
    const __nv_fp8_e4m3* x_fp8, const int32_t* gather_map, __nv_fp8_e4m3* dispatched_x_fp8,
    int total_dispatched, int C, cudaStream_t stream
);

// Phase 28: FP8 MoE Quantization & Fused GELU
void launch_quantize_bf16_to_fp8(
    const __nv_bfloat16* in, __nv_fp8_e4m3* out, float scale, int N, cudaStream_t stream
);

void launch_fused_gelu_bf16_to_fp8(
    const __nv_bfloat16* in, __nv_fp8_e4m3* out, float scale, int N, cudaStream_t stream
);

// ---------------------------------------------------------------------------
// Full Model Forward Execution (24 Layers)
// Zero dynamic allocations: all operations use pre-allocated static workspace
// ---------------------------------------------------------------------------
void run_full_model_forward(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    const FullModelParameters& params,
    cudaStream_t stream
) {
    int M = cfg.M();
    int C = cfg.C;
    float eps = 1e-6f;
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    
    // 1. Token Embeddings: input_ids -> ws.stashed_x[0] (and direct FP8 stashing)
    launch_tok_emb_fwd(
        ws.input_ids, params.tok_emb_weight, ws.stashed_x[0], M, C, stream,
        cfg.use_fp8_qkv_backward ? ws.stashed_x_fp8[0] : nullptr, 16.0f
    );
    __nv_bfloat16* cur_x = ws.stashed_x[0];
    
    // 2. Loop through all 24 layers sequentially (L2-pinned layer_x2 active staging)
    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lay = params.layers[l];
        
        // Step 2a: Fused RMSNorm 1: cur_x -> ws.layer_x_norm1 (or direct FP8)
        if (cfg.use_fused_rmsnorm_quant && cfg.use_fp8_qkv) {
            launch_fused_rmsnorm_fwd(
                cur_x, lay.norm1_weight, nullptr, ws.layer_rsqrt1, M, C, eps, stream,
                ws.layer_x_norm1_fp8, 16.0f
            );
            cublaslt_gemm_qkv_fwd_fp8(ws.layer_x_norm1_fp8, lay.qkv_weight_fp8, ws.layer_qkv, M, C, 3 * C, 1.0f / (16.0f * 64.0f), stream);
        } else if (cfg.use_fp8_qkv) {
            launch_fused_rmsnorm_fwd(
                cur_x, lay.norm1_weight, ws.layer_x_norm1, ws.layer_rsqrt1, M, C, eps, stream
            );
            launch_quantize_bf16_to_fp8(ws.layer_x_norm1, ws.layer_x_norm1_fp8, 16.0f, M * C, stream);
            cublaslt_gemm_qkv_fwd_fp8(ws.layer_x_norm1_fp8, lay.qkv_weight_fp8, ws.layer_qkv, M, C, 3 * C, 1.0f / (16.0f * 64.0f), stream);
        } else {
            launch_fused_rmsnorm_fwd(
                cur_x, lay.norm1_weight, ws.layer_x_norm1, ws.layer_rsqrt1, M, C, eps, stream
            );
            cublaslt_gemm_qkv_fwd(ws.layer_x_norm1, lay.qkv_weight, ws.layer_qkv, M, C, stream);
        }
        
        // Step 2c: Native Associative Linear Attention Forward Pipeline
        run_native_associative_attention_forward(ws, lay, l, cfg, stream);
        
        // Step 2d: Fused Residual 1 + RMSNorm 2 (Phase 16 Single-Pass Breakthrough)
        // Computes x1 = cur_x + attn_out AND x_norm2 = RMSNorm(x1) in ONE memory pass!
        if (cfg.use_fused_rmsnorm_quant && cfg.use_fp8_moe) {
            launch_fused_add_rmsnorm_fwd(
                cur_x, ws.layer_attn_out, lay.norm2_weight,
                ws.layer_x1, ws.layer_x_norm2, ws.layer_rsqrt2,
                M, C, eps, stream,
                ws.layer_x_norm2_fp8, 16.0f
            );
        } else {
            launch_fused_add_rmsnorm_fwd(
                cur_x, ws.layer_attn_out, lay.norm2_weight,
                ws.layer_x1, ws.layer_x_norm2, ws.layer_rsqrt2,
                M, C, eps, stream
            );
        }
        
        // Step 2e: MoE Router Linear Projection: (M, C) @ (E, C).T -> (M, E)
        cublaslt_gemm_router_fwd(ws.layer_x_norm2, lay.router_weight, ws.layer_router_logits, M, C, cfg.E, stream);
        
        // Step 2f: MoE Top-2 Gating & Dispatch Maps
        launch_moe_top2_gating(
            ws.layer_router_logits, ws.layer_topk_gates, ws.layer_topk_idx, ws.l_bal_total,
            M, cfg.E, 0.1f, true, stream
        );
        launch_moe_compute_maps(
            ws.layer_topk_idx, ws.layer_scatter_map, ws.layer_gather_map, ws.layer_gate_idx_map, ws.layer_expert_offsets,
            M, cfg.E, stream
        );
        
        // Step 2g: Expert Computation (Grouped MoE W1 + Fused GELU + W2)
        if (cfg.use_fused_rmsnorm_quant && cfg.use_fp8_moe) {
            launch_moe_dispatch_gather_fp8(
                ws.layer_x_norm2_fp8, ws.layer_gather_map, ws.layer_dispatched_x_fp8,
                M * cfg.top_k, C, stream
            );
            cublaslt_gemm_moe_w1_fp8(ws.layer_dispatched_x_fp8, lay.w1_weights_fp8[0], ws.layer_h1, M * cfg.top_k, C, cfg.hidden_dim, 1.0f / (16.0f * 64.0f), stream);
            launch_fused_gelu_bf16_to_fp8(ws.layer_h1, ws.layer_act_fp8, 16.0f, M * cfg.top_k * cfg.hidden_dim, stream);
            cublaslt_gemm_moe_w2_fp8(ws.layer_act_fp8, lay.w2_weights_fp8[0], ws.layer_dispatched_y, M * cfg.top_k, cfg.hidden_dim, C, 1.0f / (16.0f * 64.0f), stream);
        } else if (cfg.use_fp8_moe) {
            launch_moe_dispatch_gather(
                ws.layer_x_norm2, ws.layer_gather_map, ws.layer_dispatched_x,
                M * cfg.top_k, C, stream
            );
            launch_quantize_bf16_to_fp8(ws.layer_dispatched_x, ws.layer_dispatched_x_fp8, 16.0f, M * cfg.top_k * C, stream);
            cublaslt_gemm_moe_w1_fp8(ws.layer_dispatched_x_fp8, lay.w1_weights_fp8[0], ws.layer_h1, M * cfg.top_k, C, cfg.hidden_dim, 1.0f / (16.0f * 64.0f), stream);
            launch_fused_gelu_bf16_to_fp8(ws.layer_h1, ws.layer_act_fp8, 16.0f, M * cfg.top_k * cfg.hidden_dim, stream);
            cublaslt_gemm_moe_w2_fp8(ws.layer_act_fp8, lay.w2_weights_fp8[0], ws.layer_dispatched_y, M * cfg.top_k, cfg.hidden_dim, C, 1.0f / (16.0f * 64.0f), stream);
        } else {
            launch_moe_dispatch_gather(
                ws.layer_x_norm2, ws.layer_gather_map, ws.layer_dispatched_x,
                M * cfg.top_k, C, stream
            );
            launch_moe_grouped_gemm_fwd_w1(
                ws.layer_dispatched_x, lay.w1_weights, ws.layer_expert_offsets, ws.layer_h1,
                M * cfg.top_k, C, cfg.hidden_dim, cfg.E, stream
            );
            launch_fused_gelu_fwd(ws.layer_h1, ws.layer_act, M * cfg.top_k * cfg.hidden_dim, stream);
            launch_moe_grouped_gemm_fwd_w2(
                ws.layer_act, lay.w2_weights, ws.layer_expert_offsets, ws.layer_dispatched_y,
                M * cfg.top_k, cfg.hidden_dim, C, cfg.E, stream
            );
        }
        
        // Step 2h+2i: Fused MoE Scatter Combine + Residual 2 Addition (Direct next-layer stashing!)
        if (l < cfg.num_layers - 1) {
            launch_moe_scatter_combine_add_residual(
                ws.layer_dispatched_y, ws.layer_topk_gates, ws.layer_scatter_map,
                ws.layer_x1, ws.stashed_x[l + 1], M, C, stream,
                cfg.use_fp8_qkv_backward ? ws.stashed_x_fp8[l + 1] : nullptr, 16.0f
            );
            cur_x = ws.stashed_x[l + 1];
        } else {
            launch_moe_scatter_combine_add_residual(
                ws.layer_dispatched_y, ws.layer_topk_gates, ws.layer_scatter_map,
                ws.layer_x1, ws.layer_x2, M, C, stream
            );
            cur_x = ws.layer_x2;
        }
    }
    
    // 3. Final RMSNorm: cur_x -> ws.final_norm_out
    if (cfg.use_fused_rmsnorm_quant && cfg.use_fp8_lm_head) {
        launch_fused_rmsnorm_fwd(
            cur_x, params.final_norm_weight, ws.final_norm_out, ws.final_rsqrt, M, C, eps, stream,
            ws.final_norm_out_fp8, 16.0f
        );
        cublaslt_gemm_lm_head_fwd_fp8(ws.final_norm_out_fp8, params.lm_head_weight_fp8, ws.logits, M, C, cfg.vocab_pad, 1.0f / (16.0f * 64.0f), stream);
    } else if (cfg.use_fp8_lm_head) {
        launch_fused_rmsnorm_fwd(
            cur_x, params.final_norm_weight, ws.final_norm_out, ws.final_rsqrt, M, C, eps, stream
        );
        launch_quantize_bf16_to_fp8(ws.final_norm_out, ws.final_norm_out_fp8, 16.0f, M * C, stream);
        cublaslt_gemm_lm_head_fwd_fp8(ws.final_norm_out_fp8, params.lm_head_weight_fp8, ws.logits, M, C, cfg.vocab_pad, 1.0f / (16.0f * 64.0f), stream);
    } else {
        launch_fused_rmsnorm_fwd(
            cur_x, params.final_norm_weight, ws.final_norm_out, ws.final_rsqrt, M, C, eps, stream
        );
        cublaslt_gemm_lm_head_fwd(ws.final_norm_out, params.lm_head_weight, ws.logits, M, C, cfg.vocab_pad, stream);
    }
    
    // 5. Fused Cross-Entropy Loss and Analytical dLogits (Phase 31: writes d_logits_fp8 directly from registers)
    launch_fused_cross_entropy_bwd(
        ws.logits, ws.targets,
        cfg.use_fp8_lm_head_backward ? nullptr : ws.d_logits,
        ws.loss_buffer,
        M, cfg.vocab_size, cfg.vocab_pad, 1.0f / (float)cfg.accum_steps, stream,
        (cfg.use_fp8_lm_head || cfg.use_fp8_lm_head_backward) ? ws.d_logits_fp8 : nullptr,
        1048576.0f
    );
}

// ---------------------------------------------------------------------------
// Full Model Analytical Backward Execution (24 Layers)
// Zero dynamic allocations: accumulates gradients directly into parameter buffers
// ---------------------------------------------------------------------------
void run_full_model_backward(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    FullModelParameters& params,
    bool is_first_step,
    bool is_final_step,
    cudaStream_t stream
) {
    int M = cfg.M();
    int C = cfg.C;
    
    if (is_first_step) {
        // Zero token embedding gradient for the new update step
        cudaMemsetAsync(params.d_tok_emb_weight, 0, (size_t)cfg.vocab_size * cfg.C * sizeof(__nv_bfloat16), stream);
    }
    
    if (is_final_step) {
        launch_zero_grad_norm(ws.grad_norm_sq, stream);
    }
    
    float beta = is_first_step ? 0.0f : 1.0f;
    
    // 1. LM Head Backward:
    // d_final_norm = d_logits @ W_lm_head
    // d_W_lm_head = d_logits.T @ final_norm_out (overwrites on step 0, accumulates on step 1)
    if (cfg.use_fp8_lm_head_backward) {
        float scale_dlog = 1048576.0f; // 2^20
        float scale_w = 64.0f;
        float scale_fn = 16.0f;
        // Phase 31: d_logits_fp8 is populated directly from registers during CE; standalone quantization eliminated!
        if (!cfg.use_fp8_lm_head) {
            launch_quantize_bf16_to_fp8(ws.final_norm_out, ws.final_norm_out_fp8, scale_fn, M * C, stream);
        }
        float alpha_dx = 1.0f / (scale_dlog * scale_w);
        cublaslt_gemm_lm_head_bwd_dx_fp8(ws.d_logits_fp8, params.lm_head_weight_fp8, ws.d_final_norm_out, M, cfg.vocab_pad, C, alpha_dx, stream);
        float alpha_dw = 1.0f / (scale_dlog * scale_fn);
        cublaslt_gemm_lm_head_bwd_dw_fp8(ws.d_logits_fp8, ws.final_norm_out_fp8, params.d_lm_head_weight, M, cfg.vocab_pad, C, alpha_dw, beta, stream);
    } else {
        cublaslt_gemm_lm_head_bwd_dx(ws.d_logits, params.lm_head_weight, ws.d_final_norm_out, M, cfg.vocab_pad, C, stream);
        cublaslt_gemm_lm_head_bwd_dw(ws.d_logits, ws.final_norm_out, params.d_lm_head_weight, M, cfg.vocab_pad, C, stream, beta);
    }
    
    // 2. Final RMSNorm Backward: d_final_norm -> ws.d_layer_x
    launch_fused_rmsnorm_bwd(
        ws.d_final_norm_out, ws.stashed_x[cfg.num_layers - 1], params.final_norm_weight,
        ws.final_rsqrt, ws.d_layer_x, params.d_final_norm_weight, M, C, stream
    );
    
    if (is_final_step) {
        launch_accumulate_global_grad_norm_sq(params, ws.grad_norm_sq, cfg, stream);
    }
    
    // 3. Backward Pass through 24 layers in reverse order (23 down to 0)
    __nv_bfloat16* cur_dx = ws.d_layer_x;
    __nv_bfloat16* next_dx = ws.d_layer_x_prev;
    
    for (int l = cfg.num_layers - 1; l >= 0; --l) {
        auto& lay = params.layers[l];
        
        // 1. Recompute forward activations for Layer l
        launch_fused_rmsnorm_fwd(
            ws.stashed_x[l], lay.norm1_weight, ws.layer_x_norm1, ws.layer_rsqrt1,
            M, C, 1e-6f, stream
        );
        cublaslt_gemm_qkv_fwd(ws.layer_x_norm1, lay.qkv_weight, ws.layer_qkv, M, C, stream);
        run_native_associative_attention_forward(ws, lay, l, cfg, stream);
        
        launch_fused_add_rmsnorm_fwd(
            ws.stashed_x[l], ws.layer_attn_out, lay.norm2_weight,
            ws.layer_x1, ws.layer_x_norm2, ws.layer_rsqrt2,
            M, C, 1e-6f, stream
        );
        cublaslt_gemm_router_fwd(ws.layer_x_norm2, lay.router_weight, ws.layer_router_logits, M, C, cfg.E, stream);
        launch_moe_top2_gating(
            ws.layer_router_logits, ws.layer_topk_gates, ws.layer_topk_idx, ws.l_bal_total,
            M, cfg.E, 0.1f, true, stream
        );
        launch_moe_compute_maps(
            ws.layer_topk_idx, ws.layer_scatter_map, ws.layer_gather_map, ws.layer_gate_idx_map, ws.layer_expert_offsets,
            M, cfg.E, stream
        );
        launch_moe_dispatch_gather(
            ws.layer_x_norm2, ws.layer_gather_map, ws.layer_dispatched_x,
            M * cfg.top_k, C, stream
        );
        launch_moe_grouped_gemm_fwd_w1(
            ws.layer_dispatched_x, lay.w1_weights, ws.layer_expert_offsets, ws.layer_h1,
            M * cfg.top_k, C, cfg.hidden_dim, cfg.E, stream
        );
        launch_fused_gelu_fwd(ws.layer_h1, ws.layer_act, M * cfg.top_k * cfg.hidden_dim, stream);
        launch_moe_grouped_gemm_fwd_w2(
            ws.layer_act, lay.w2_weights, ws.layer_expert_offsets, ws.layer_dispatched_y,
            M * cfg.top_k, cfg.hidden_dim, C, cfg.E, stream
        );
        
        // 2a. MoE Scatter Backward:
        // cur_dx (dL/dx2) -> ws.d_dispatched_y, ws.d_topk_gates
        launch_moe_scatter_backward(
            cur_dx, ws.layer_dispatched_y, ws.layer_topk_gates,
            ws.layer_gather_map, ws.layer_gate_idx_map, ws.layer_scatter_map,
            ws.d_dispatched_y, ws.d_topk_gates,
            M, M * cfg.top_k, C, stream
        );
        
        // 2b. MoE Grouped W2 Backward:
        // ws.d_dispatched_y, ws.layer_act -> lay.d_w2_weights (with beta), ws.d_act
        launch_moe_grouped_gemm_w2_bwd(
            ws.d_dispatched_y, ws.layer_act, lay.w2_weights,
            ws.layer_expert_offsets, lay.d_w2_weights, ws.d_act,
            M * cfg.top_k, cfg.hidden_dim, C, cfg.E, beta, stream
        );
        
        // 2c. Fused GELU Backward:
        // ws.d_act, ws.layer_h1 -> ws.d_h1
        launch_fused_gelu_bwd(ws.d_act, ws.layer_h1, ws.d_h1, M * cfg.top_k * cfg.hidden_dim, stream);
        
        // 2d. MoE Grouped W1 Backward:
        // ws.d_h1, ws.layer_dispatched_x -> lay.d_w1_weights (with beta), ws.d_dispatched_x
        launch_moe_grouped_gemm_w1_bwd(
            ws.d_h1, ws.layer_dispatched_x, lay.w1_weights,
            ws.layer_expert_offsets, lay.d_w1_weights, ws.d_dispatched_x,
            M * cfg.top_k, C, cfg.hidden_dim, cfg.E, beta, stream
        );
        
        // 2e. MoE Gather Backward:
        // ws.d_dispatched_x -> ws.dx_norm2_expert
        launch_moe_gather_backward(
            ws.d_dispatched_x, ws.layer_scatter_map, ws.dx_norm2_expert,
            M, cfg.top_k, C, stream
        );
        
        // 2f. MoE Router Backward:
        // ws.d_topk_gates, ws.layer_router_logits, ws.layer_topk_idx -> ws.d_router_logits
        launch_moe_router_backward(
            ws.d_topk_gates, ws.layer_router_logits, ws.layer_topk_idx,
            ws.d_router_logits, M, cfg.E, stream
        );
        
        // 2g. MoE Router GEMMs Backward & Norm2 Input Combination:
        // ws.d_router_logits, ws.layer_x_norm2 -> lay.d_router_weight (with beta)
        // ws.d_router_logits @ router_weight + ws.dx_norm2_expert -> ws.dx_norm2
        launch_moe_router_gemms_bwd(
            ws.d_router_logits, ws.layer_x_norm2, lay.router_weight,
            ws.dx_norm2_expert, lay.d_router_weight, ws.dx_norm2,
            M, C, cfg.E, beta, stream
        );
        
        // 2h. RMSNorm 2 Backward:
        // ws.dx_norm2 -> ws.dx_norm2_in, lay.d_norm2_weight
        launch_fused_rmsnorm_bwd(
            ws.dx_norm2, ws.layer_x1, lay.norm2_weight,
            ws.layer_rsqrt2, ws.dx_norm2_in, lay.d_norm2_weight, M, C, stream
        );
        
        // 2i. Residual 2 Addition:
        // ws.dx1 = cur_dx + ws.dx_norm2_in
        launch_fused_add_residual(cur_dx, ws.dx_norm2_in, ws.dx1, M * C, stream);
        
        // 3. Analytical Associative Attention Backward:
        // Computes dW_out (accumulating into lay.d_out_proj_weight with beta)
        // and d_layer_qkv (M, 3 * C) with distinct dQ, dK, dV
        run_native_associative_attention_backward(
            ws, lay, l, cfg, ws.dx1, beta, stream
        );
        
        // 4. QKV Projection Backward GEMMs:
        // 4a. dW_qkv += (d_layer_qkv)^T @ layer_x_norm1 (3 * C, C)
        cublasHandle_t handle = get_cublas_handle();
        cublasSetStream(handle, stream);
        float alpha = 1.0f, beta_zero = 0.0f;
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            C, 3 * C, M,
            &alpha,
            ws.layer_x_norm1, CUDA_R_16BF, C,
            ws.d_layer_qkv, CUDA_R_16BF, 3 * C,
            &beta,
            lay.d_qkv_weight, CUDA_R_16BF, C,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
        
        // 4b. dx_norm1 = d_layer_qkv @ W_qkv (M, C)
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            C, M, 3 * C,
            &alpha,
            lay.qkv_weight, CUDA_R_16BF, C,
            ws.d_layer_qkv, CUDA_R_16BF, 3 * C,
            &beta_zero,
            ws.dx_norm1, CUDA_R_16BF, C,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
        
        // 5. RMSNorm 1 Backward:
        // Propagate dx_norm1 through RMSNorm 1 -> dx_norm1_in and accumulate lay.d_norm1_weight
        launch_fused_rmsnorm_bwd(
            ws.dx_norm1, ws.stashed_x[l], lay.norm1_weight,
            ws.layer_rsqrt1, ws.dx_norm1_in, lay.d_norm1_weight, M, C, stream
        );
        
        // 6. Residual 1 Addition:
        // next_dx = ws.dx1 + dx_norm1_in
        launch_fused_add_residual(ws.dx1, ws.dx_norm1_in, next_dx, M * C, stream);
        
        // Zero-overhead ping-pong gradient pointer swap (eliminates cudaMemcpyAsync)
        std::swap(cur_dx, next_dx);
        
        if (is_final_step) {
            launch_accumulate_layer_grad_norm_sq(lay, ws.grad_norm_sq, cfg, stream);
        }
    }
    
    // 4. Token Embedding Backward: cur_dx -> params.d_tok_emb_weight (cur_dx is ws.d_layer_x after 24 swaps)
    launch_tok_emb_bwd(cur_dx, ws.input_ids, params.d_tok_emb_weight, M, C, stream);
    
    if (is_final_step) {
        launch_accumulate_grad_norm_sq(params.d_tok_emb_weight, cfg.vocab_size * cfg.C, ws.grad_norm_sq, stream);
        launch_compute_clip_coef(ws.grad_norm_sq, ws.clip_coef, 1.0f, stream);
    }
}


// ---------------------------------------------------------------------------
// Complete Training Step Coordinator
// Executes 2 microsteps (gradient accumulation) followed by fused AdamW optimizer
// ---------------------------------------------------------------------------
void run_native_training_step(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    FullModelParameters& params,
    float lr,
    cudaStream_t stream
) {
    // Accumulate across 2 microsteps (4096 tokens total)
    for (int step = 0; step < cfg.accum_steps; ++step) {
        run_full_model_forward(cfg, ws, params, stream);
        bool is_first_step = (step == 0);
        bool is_final_step = (step == cfg.accum_steps - 1);
        run_full_model_backward(cfg, ws, params, is_first_step, is_final_step, stream);
    }
    
    // Fused AdamW optimizer update with in-kernel norm clipping and zero_grad
    run_fused_optimizer_step(params, ws, cfg, lr, stream);
    
    // Phase 31: If BF16/FP8 moments AdamW is active, FP8 weights are written directly by AdamW kernel; standalone quantize eliminated!
    if (!cfg.use_bf16_moments && !cfg.use_fp8_moments) {
        if (cfg.use_fp8_lm_head || cfg.use_fp8_lm_head_backward) {
            launch_quantize_bf16_to_fp8(params.lm_head_weight, params.lm_head_weight_fp8, 64.0f, cfg.vocab_pad * cfg.C, stream);
        }
        if (cfg.use_fp8_qkv) {
            for (int l = 0; l < cfg.num_layers; ++l) {
                launch_quantize_bf16_to_fp8(params.layers[l].qkv_weight, params.layers[l].qkv_weight_fp8, 64.0f, 3 * cfg.C * cfg.C, stream);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Phase 22: Layer-Wise Microstep Interleaved Forward Pass
// Eliminates 2.42 GB of redundant DRAM weight streaming by keeping Layer l
// weights hot in the 48 MB L2 cache across Microstep 0 and Microstep 1.
// ---------------------------------------------------------------------------
void run_interleaved_forward(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    const FullModelParameters& params,
    cudaStream_t stream
) {
    int M = cfg.M();
    int C = cfg.C;
    float eps = 1e-6f;
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    
    // 1. Token Embeddings for both microsteps (directly stashed to layer 0 BF16 & FP8)
    for (int ms = 0; ms < 2; ++ms) {
        launch_tok_emb_fwd(
            ws.input_ids_ms[ms], params.tok_emb_weight, ws.stashed_x_ms[ms][0], M, C, stream,
            cfg.use_fp8_qkv_backward ? ws.stashed_x_fp8_ms[ms][0] : nullptr, 16.0f
        );
    }
    
    __nv_bfloat16* cur_x[2] = { ws.stashed_x_ms[0][0], ws.stashed_x_ms[1][0] };
    
    // Static tensor wrappers for shared active workspace
    auto qkv_t = torch::from_blob(ws.layer_qkv, {M, 3 * C}, options_bf16);
    auto q_chunk = qkv_t.slice(1, 0, C);
    auto attn_out_t = torch::from_blob(ws.layer_attn_out, {M, C}, options_bf16);
    auto x_norm1_t = torch::from_blob(ws.layer_x_norm1, {M, C}, options_bf16);
    auto x_norm2_t = torch::from_blob(ws.layer_x_norm2, {M, C}, options_bf16);
    auto router_logits_t = torch::from_blob(ws.layer_router_logits, {M, cfg.E}, options_bf16);
    auto disp_x_t = torch::from_blob(ws.layer_dispatched_x, {M * cfg.top_k, C}, options_bf16);
    auto disp_y_t = torch::from_blob(ws.layer_dispatched_y, {M * cfg.top_k, C}, options_bf16);
    auto h1_t = torch::from_blob(ws.layer_h1, {M * cfg.top_k, cfg.hidden_dim}, options_bf16);
    auto act_t = torch::from_blob(ws.layer_act, {M * cfg.top_k, cfg.hidden_dim}, options_bf16);
    
    // 2. Loop through all 24 layers with Layer-Wise Microstep Interleaving!
    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lay = params.layers[l];
        auto lay_qkv_w_t = torch::from_blob(lay.qkv_weight, {3 * C, C}, options_bf16);
        auto lay_out_w_t = torch::from_blob(lay.out_proj_weight, {C, C}, options_bf16);
        auto lay_router_w_t = torch::from_blob(lay.router_weight, {cfg.E, C}, options_bf16);
        auto lay_w1_t = torch::from_blob(lay.w1_weights[0], {cfg.hidden_dim, C}, options_bf16);
        auto lay_w2_t = torch::from_blob(lay.w2_weights[0], {C, cfg.hidden_dim}, options_bf16);
        
        // Microstep 0 followed immediately by Microstep 1
        for (int ms = 0; ms < 2; ++ms) {
            // cur_x[ms] is already ws.stashed_x_ms[ms][l]!
            // ws.stashed_x_fp8_ms[ms][l] is already pre-quantized from previous residual pass!
            
            // Step 2a: Fused RMSNorm 1
            if (cfg.use_fused_rmsnorm_quant && cfg.use_fp8_qkv) {
                launch_fused_rmsnorm_fwd(
                    cur_x[ms], lay.norm1_weight, nullptr, ws.layer_rsqrt1, M, C, eps, stream,
                    ws.layer_x_norm1_fp8, 16.0f
                );
                cublaslt_gemm_qkv_fwd_fp8(ws.layer_x_norm1_fp8, lay.qkv_weight_fp8, ws.layer_qkv, M, C, 3 * C, 1.0f / (16.0f * 64.0f), stream);
            } else if (cfg.use_fp8_qkv) {
                launch_fused_rmsnorm_fwd(cur_x[ms], lay.norm1_weight, ws.layer_x_norm1, ws.layer_rsqrt1, M, C, eps, stream);
                launch_quantize_bf16_to_fp8(ws.layer_x_norm1, ws.layer_x_norm1_fp8, 16.0f, M * C, stream);
                cublaslt_gemm_qkv_fwd_fp8(ws.layer_x_norm1_fp8, lay.qkv_weight_fp8, ws.layer_qkv, M, C, 3 * C, 1.0f / (16.0f * 64.0f), stream);
            } else {
                launch_fused_rmsnorm_fwd(cur_x[ms], lay.norm1_weight, ws.layer_x_norm1, ws.layer_rsqrt1, M, C, eps, stream);
                cublaslt_gemm_qkv_fwd(ws.layer_x_norm1, lay.qkv_weight, ws.layer_qkv, M, C, stream);
            }
            
            // Step 2c: Native Associative Linear Attention Forward Pipeline
            run_native_associative_attention_forward(ws, lay, l, cfg, stream);
            
            // Step 2d: Fused Residual 1 + RMSNorm 2
            if (cfg.use_fused_rmsnorm_quant && cfg.use_fp8_moe) {
                launch_fused_add_rmsnorm_fwd(
                    cur_x[ms], ws.layer_attn_out, lay.norm2_weight,
                    ws.layer_x1, ws.layer_x_norm2, ws.layer_rsqrt2,
                    M, C, eps, stream,
                    ws.layer_x_norm2_fp8, 16.0f
                );
            } else {
                launch_fused_add_rmsnorm_fwd(
                    cur_x[ms], ws.layer_attn_out, lay.norm2_weight,
                    ws.layer_x1, ws.layer_x_norm2, ws.layer_rsqrt2,
                    M, C, eps, stream
                );
            }
            
            // Step 2e: Router GEMM
            cublaslt_gemm_router_fwd(ws.layer_x_norm2, lay.router_weight, ws.layer_router_logits, M, C, cfg.E, stream);
            
            // Step 2f: MoE Gating & Dispatch
            launch_moe_top2_gating(ws.layer_router_logits, ws.layer_topk_gates, ws.layer_topk_idx, ws.l_bal_total, M, cfg.E, 0.1f, true, stream);
            launch_moe_compute_maps(ws.layer_topk_idx, ws.layer_scatter_map, ws.layer_gather_map, ws.layer_gate_idx_map, ws.layer_expert_offsets, M, cfg.E, stream);
            
            // Step 2g: Expert Grouped GEMMs (W1 and W2)
            if (cfg.use_fused_rmsnorm_quant && cfg.use_fp8_moe) {
                launch_moe_dispatch_gather_fp8(ws.layer_x_norm2_fp8, ws.layer_gather_map, ws.layer_dispatched_x_fp8, M * cfg.top_k, C, stream);
                cublaslt_gemm_moe_w1_fp8(ws.layer_dispatched_x_fp8, lay.w1_weights_fp8[0], ws.layer_h1, M * cfg.top_k, C, cfg.hidden_dim, 1.0f / (16.0f * 64.0f), stream);
                launch_fused_gelu_bf16_to_fp8(ws.layer_h1, ws.layer_act_fp8, 16.0f, M * cfg.top_k * cfg.hidden_dim, stream);
                cublaslt_gemm_moe_w2_fp8(ws.layer_act_fp8, lay.w2_weights_fp8[0], ws.layer_dispatched_y, M * cfg.top_k, cfg.hidden_dim, C, 1.0f / (16.0f * 64.0f), stream);
            } else if (cfg.use_fp8_moe) {
                launch_moe_dispatch_gather(ws.layer_x_norm2, ws.layer_gather_map, ws.layer_dispatched_x, M * cfg.top_k, C, stream);
                launch_quantize_bf16_to_fp8(ws.layer_dispatched_x, ws.layer_dispatched_x_fp8, 16.0f, M * cfg.top_k * C, stream);
                cublaslt_gemm_moe_w1_fp8(ws.layer_dispatched_x_fp8, lay.w1_weights_fp8[0], ws.layer_h1, M * cfg.top_k, C, cfg.hidden_dim, 1.0f / (16.0f * 64.0f), stream);
                launch_fused_gelu_bf16_to_fp8(ws.layer_h1, ws.layer_act_fp8, 16.0f, M * cfg.top_k * cfg.hidden_dim, stream);
                cublaslt_gemm_moe_w2_fp8(ws.layer_act_fp8, lay.w2_weights_fp8[0], ws.layer_dispatched_y, M * cfg.top_k, cfg.hidden_dim, C, 1.0f / (16.0f * 64.0f), stream);
            } else {
                launch_moe_dispatch_gather(ws.layer_x_norm2, ws.layer_gather_map, ws.layer_dispatched_x, M * cfg.top_k, C, stream);
                launch_moe_grouped_gemm_fwd_w1(ws.layer_dispatched_x, lay.w1_weights, ws.layer_expert_offsets, ws.layer_h1, M * cfg.top_k, C, cfg.hidden_dim, cfg.E, stream);
                launch_fused_gelu_fwd(ws.layer_h1, ws.layer_act, M * cfg.top_k * cfg.hidden_dim, stream);
                launch_moe_grouped_gemm_fwd_w2(ws.layer_act, lay.w2_weights, ws.layer_expert_offsets, ws.layer_dispatched_y, M * cfg.top_k, cfg.hidden_dim, C, cfg.E, stream);
            }
            
            // Step 2h+2i: Fused MoE Scatter Combine + Residual 2 Addition (Direct next-layer stashing!)
            if (l < cfg.num_layers - 1) {
                launch_moe_scatter_combine_add_residual(
                    ws.layer_dispatched_y, ws.layer_topk_gates, ws.layer_scatter_map,
                    ws.layer_x1, ws.stashed_x_ms[ms][l + 1], M, C, stream,
                    cfg.use_fp8_qkv_backward ? ws.stashed_x_fp8_ms[ms][l + 1] : nullptr, 16.0f
                );
                cur_x[ms] = ws.stashed_x_ms[ms][l + 1];
            } else {
                launch_moe_scatter_combine_add_residual(
                    ws.layer_dispatched_y, ws.layer_topk_gates, ws.layer_scatter_map,
                    ws.layer_x1, ws.layer_x2_ms[ms], M, C, stream
                );
                cur_x[ms] = ws.layer_x2_ms[ms];
            }
        }
    }
    
    // 3. Final RMSNorm & LM Head for both microsteps
    for (int ms = 0; ms < 2; ++ms) {
        if (cfg.use_fused_rmsnorm_quant && cfg.use_fp8_lm_head) {
            launch_fused_rmsnorm_fwd(
                cur_x[ms], params.final_norm_weight, ws.final_norm_out_ms[ms], ws.final_rsqrt_ms[ms], M, C, eps, stream,
                ws.final_norm_out_fp8_ms[ms], 16.0f
            );
            cublaslt_gemm_lm_head_fwd_fp8(ws.final_norm_out_fp8_ms[ms], params.lm_head_weight_fp8, ws.logits_ms[ms], M, C, cfg.vocab_pad, 1.0f / (16.0f * 64.0f), stream);
        } else if (cfg.use_fp8_lm_head) {
            launch_fused_rmsnorm_fwd(cur_x[ms], params.final_norm_weight, ws.final_norm_out_ms[ms], ws.final_rsqrt_ms[ms], M, C, eps, stream);
            launch_quantize_bf16_to_fp8(ws.final_norm_out_ms[ms], ws.final_norm_out_fp8_ms[ms], 16.0f, M * C, stream);
            cublaslt_gemm_lm_head_fwd_fp8(ws.final_norm_out_fp8_ms[ms], params.lm_head_weight_fp8, ws.logits_ms[ms], M, C, cfg.vocab_pad, 1.0f / (16.0f * 64.0f), stream);
        } else {
            launch_fused_rmsnorm_fwd(cur_x[ms], params.final_norm_weight, ws.final_norm_out_ms[ms], ws.final_rsqrt_ms[ms], M, C, eps, stream);
            cublaslt_gemm_lm_head_fwd(ws.final_norm_out_ms[ms], params.lm_head_weight, ws.logits_ms[ms], M, C, cfg.vocab_pad, stream);
        }
        
        // Phase 31: writes d_logits_fp8_ms[ms] directly from registers
        launch_fused_cross_entropy_bwd(
            ws.logits_ms[ms], ws.targets_ms[ms],
            cfg.use_fp8_lm_head_backward ? nullptr : ws.d_logits_ms[ms],
            ws.loss_buffer,
            M, cfg.vocab_size, cfg.vocab_pad, 1.0f / (float)cfg.accum_steps, stream,
            (cfg.use_fp8_lm_head || cfg.use_fp8_lm_head_backward) ? ws.d_logits_fp8_ms[ms] : nullptr,
            1048576.0f
        );
    }
}

// ---------------------------------------------------------------------------
// Phase 22: Layer-Wise Microstep Interleaved Backward Pass
// Eliminates 2.42 GB of redundant DRAM weight streaming by keeping Layer l
// weights and dW buffers hot in L2 cache across Microsteps 0 and 1.
// ---------------------------------------------------------------------------
void run_interleaved_backward(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    FullModelParameters& params,
    cudaStream_t stream
) {
    int M = cfg.M();
    int C = cfg.C;
    
    // Zero grad norm accumulator at the beginning of backward
    launch_zero_grad_norm(ws.grad_norm_sq, stream);
    
    // 1. LM Head Backward & Final RMSNorm Backward for both microsteps
    for (int ms = 0; ms < 2; ++ms) {
        float beta = (ms == 0 ? 0.0f : 1.0f);
        if (cfg.use_fp8_lm_head_backward) {
            float scale_dlog = 1048576.0f; // 2^20
            float scale_w = 64.0f;
            float scale_fn = 16.0f;
            // Phase 31: d_logits_fp8_ms[ms] populated directly from registers during CE; standalone quantization eliminated!
            if (!cfg.use_fp8_lm_head) {
                launch_quantize_bf16_to_fp8(ws.final_norm_out_ms[ms], ws.final_norm_out_fp8_ms[ms], scale_fn, M * C, stream);
            }
            float alpha_dx = 1.0f / (scale_dlog * scale_w);
            cublaslt_gemm_lm_head_bwd_dx_fp8(ws.d_logits_fp8_ms[ms], params.lm_head_weight_fp8, ws.d_final_norm_out_ms[ms], M, cfg.vocab_pad, C, alpha_dx, stream);
            float alpha_dw = 1.0f / (scale_dlog * scale_fn);
            cublaslt_gemm_lm_head_bwd_dw_fp8(ws.d_logits_fp8_ms[ms], ws.final_norm_out_fp8_ms[ms], params.d_lm_head_weight, M, cfg.vocab_pad, C, alpha_dw, beta, stream);
        } else {
            cublaslt_gemm_lm_head_bwd_dx(ws.d_logits_ms[ms], params.lm_head_weight, ws.d_final_norm_out_ms[ms], M, cfg.vocab_pad, C, stream);
            cublaslt_gemm_lm_head_bwd_dw(ws.d_logits_ms[ms], ws.final_norm_out_ms[ms], params.d_lm_head_weight, M, cfg.vocab_pad, C, stream, beta);
        }
        
        launch_fused_rmsnorm_bwd(
            ws.d_final_norm_out_ms[ms], ws.stashed_x_ms[ms][cfg.num_layers - 1], params.final_norm_weight,
            ws.final_rsqrt_ms[ms], ws.d_layer_x_ms[ms], params.d_final_norm_weight, M, C, stream
        );
    }
    
    // Asynchronously accumulate LM Head & Final Norm gradients while layer loop begins
    launch_accumulate_global_grad_norm_sq(params, ws.grad_norm_sq, cfg, stream);
    
    // 2. Loop through all 24 layers in reverse order with Layer-Wise Microstep Interleaving!
    __nv_bfloat16* cur_dx[2] = { ws.d_layer_x_ms[0], ws.d_layer_x_ms[1] };
    __nv_bfloat16* next_dx[2] = { ws.d_layer_x_prev_ms[0], ws.d_layer_x_prev_ms[1] };
    
    for (int l = cfg.num_layers - 1; l >= 0; --l) {
        auto& lay = params.layers[l];
        
        // Interleave MS0 then MS1: both accumulate into the same layer parameter buffers
        for (int ms = 0; ms < 2; ++ms) {
            float beta_layer = (ms == 0 ? 0.0f : 1.0f);
            
            // 1. Recompute forward activations for Layer l, Microstep ms
            launch_fused_rmsnorm_fwd(
                ws.stashed_x_ms[ms][l], lay.norm1_weight, ws.layer_x_norm1, ws.layer_rsqrt1,
                M, C, 1e-6f, stream
            );
            cublaslt_gemm_qkv_fwd(ws.layer_x_norm1, lay.qkv_weight, ws.layer_qkv, M, C, stream);
            run_native_associative_attention_forward(ws, lay, l, cfg, stream);
            
            launch_fused_add_rmsnorm_fwd(
                ws.stashed_x_ms[ms][l], ws.layer_attn_out, lay.norm2_weight,
                ws.layer_x1, ws.layer_x_norm2, ws.layer_rsqrt2,
                M, C, 1e-6f, stream
            );
            cublaslt_gemm_router_fwd(ws.layer_x_norm2, lay.router_weight, ws.layer_router_logits, M, C, cfg.E, stream);
            launch_moe_top2_gating(
                ws.layer_router_logits, ws.layer_topk_gates, ws.layer_topk_idx, ws.l_bal_total,
                M, cfg.E, 0.1f, true, stream
            );
            launch_moe_compute_maps(
                ws.layer_topk_idx, ws.layer_scatter_map, ws.layer_gather_map, ws.layer_gate_idx_map, ws.layer_expert_offsets,
                M, cfg.E, stream
            );
            launch_moe_dispatch_gather(
                ws.layer_x_norm2, ws.layer_gather_map, ws.layer_dispatched_x,
                M * cfg.top_k, C, stream
            );
            launch_moe_grouped_gemm_fwd_w1(
                ws.layer_dispatched_x, lay.w1_weights, ws.layer_expert_offsets, ws.layer_h1,
                M * cfg.top_k, C, cfg.hidden_dim, cfg.E, stream
            );
            launch_fused_gelu_fwd(ws.layer_h1, ws.layer_act, M * cfg.top_k * cfg.hidden_dim, stream);
            launch_moe_grouped_gemm_fwd_w2(
                ws.layer_act, lay.w2_weights, ws.layer_expert_offsets, ws.layer_dispatched_y,
                M * cfg.top_k, cfg.hidden_dim, C, cfg.E, stream
            );
            
            // 2a. MoE Scatter Backward:
            // cur_dx[ms] (dL/dx2) -> ws.d_dispatched_y, ws.d_topk_gates
            launch_moe_scatter_backward(
                cur_dx[ms], ws.layer_dispatched_y, ws.layer_topk_gates,
                ws.layer_gather_map, ws.layer_gate_idx_map, ws.layer_scatter_map,
                ws.d_dispatched_y, ws.d_topk_gates,
                M, M * cfg.top_k, C, stream
            );
            
            // 2b. MoE Grouped W2 Backward:
            // ws.d_dispatched_y, ws.layer_act -> lay.d_w2_weights (with beta_layer), ws.d_act
            launch_moe_grouped_gemm_w2_bwd(
                ws.d_dispatched_y, ws.layer_act, lay.w2_weights,
                ws.layer_expert_offsets, lay.d_w2_weights, ws.d_act,
                M * cfg.top_k, cfg.hidden_dim, C, cfg.E, beta_layer, stream
            );
            
            // 2c. Fused GELU Backward:
            // ws.d_act, ws.layer_h1 -> ws.d_h1
            launch_fused_gelu_bwd(ws.d_act, ws.layer_h1, ws.d_h1, M * cfg.top_k * cfg.hidden_dim, stream);
            
            // 2d. MoE Grouped W1 Backward:
            // ws.d_h1, ws.layer_dispatched_x -> lay.d_w1_weights (with beta_layer), ws.d_dispatched_x
            launch_moe_grouped_gemm_w1_bwd(
                ws.d_h1, ws.layer_dispatched_x, lay.w1_weights,
                ws.layer_expert_offsets, lay.d_w1_weights, ws.d_dispatched_x,
                M * cfg.top_k, C, cfg.hidden_dim, cfg.E, beta_layer, stream
            );
            
            // 2e. MoE Gather Backward:
            // ws.d_dispatched_x -> ws.dx_norm2_expert
            launch_moe_gather_backward(
                ws.d_dispatched_x, ws.layer_scatter_map, ws.dx_norm2_expert,
                M, cfg.top_k, C, stream
            );
            
            // 2f. MoE Router Backward:
            // ws.d_topk_gates, ws.layer_router_logits, ws.layer_topk_idx -> ws.d_router_logits
            launch_moe_router_backward(
                ws.d_topk_gates, ws.layer_router_logits, ws.layer_topk_idx,
                ws.d_router_logits, M, cfg.E, stream
            );
            
            // 2g. MoE Router GEMMs Backward & Norm2 Input Combination:
            // ws.d_router_logits, ws.layer_x_norm2 -> lay.d_router_weight (with beta_layer)
            // ws.d_router_logits @ router_weight + ws.dx_norm2_expert -> ws.dx_norm2
            launch_moe_router_gemms_bwd(
                ws.d_router_logits, ws.layer_x_norm2, lay.router_weight,
                ws.dx_norm2_expert, lay.d_router_weight, ws.dx_norm2,
                M, C, cfg.E, beta_layer, stream
            );
            
            // 2h. RMSNorm 2 Backward:
            // ws.dx_norm2 -> ws.dx_norm2_in, lay.d_norm2_weight
            launch_fused_rmsnorm_bwd(
                ws.dx_norm2, ws.layer_x1, lay.norm2_weight,
                ws.layer_rsqrt2, ws.dx_norm2_in, lay.d_norm2_weight, M, C, stream,
                nullptr, 1.0f, beta_layer
            );
            
            // 2i. Residual 2 Addition:
            // ws.dx1 = cur_dx[ms] + ws.dx_norm2_in
            launch_fused_add_residual(cur_dx[ms], ws.dx_norm2_in, ws.dx1, M * C, stream);
            
            // 3. Analytical Associative Attention Backward:
            // Computes dW_out (accumulating into lay.d_out_proj_weight with beta_layer)
            // and d_layer_qkv (M, 3 * C) with distinct dQ, dK, dV
            run_native_associative_attention_backward(
                ws, lay, l, cfg, ws.dx1, beta_layer, stream
            );
            
            // 4. QKV Projection Backward GEMMs:
            // 4a. dW_qkv += (d_layer_qkv)^T @ layer_x_norm1 (3 * C, C)
            cublasHandle_t handle = get_cublas_handle();
            cublasSetStream(handle, stream);
            float alpha = 1.0f, beta_zero = 0.0f;
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                C, 3 * C, M,
                &alpha,
                ws.layer_x_norm1, CUDA_R_16BF, C,
                ws.d_layer_qkv, CUDA_R_16BF, 3 * C,
                &beta_layer,
                lay.d_qkv_weight, CUDA_R_16BF, C,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            );
            
            // 4b. dx_norm1 = d_layer_qkv @ W_qkv (M, C)
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                C, M, 3 * C,
                &alpha,
                lay.qkv_weight, CUDA_R_16BF, C,
                ws.d_layer_qkv, CUDA_R_16BF, 3 * C,
                &beta_zero,
                ws.dx_norm1, CUDA_R_16BF, C,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            );
            
            // 5. RMSNorm 1 Backward:
            // Propagate dx_norm1 through RMSNorm 1 -> dx_norm1_in and accumulate lay.d_norm1_weight
            launch_fused_rmsnorm_bwd(
                ws.dx_norm1, ws.stashed_x_ms[ms][l], lay.norm1_weight,
                ws.layer_rsqrt1, ws.dx_norm1_in, lay.d_norm1_weight, M, C, stream,
                nullptr, 1.0f, beta_layer
            );
            
            // 6. Residual 1 Addition:
            // next_dx[ms] = ws.dx1 + dx_norm1_in
            launch_fused_add_residual(ws.dx1, ws.dx_norm1_in, next_dx[ms], M * C, stream);
            
            // Zero-overhead ping-pong gradient pointer swap (eliminates cudaMemcpyAsync)
            std::swap(cur_dx[ms], next_dx[ms]);
        }
        
        // Asynchronously accumulate layer l's completed gradients into ||g||^2
        launch_accumulate_layer_grad_norm_sq(lay, ws.grad_norm_sq, cfg, stream);
    }
    
    // 3. Token Embedding Backward for both microsteps (cur_dx[ms] is ws.d_layer_x_ms[ms] after 24 swaps)
    for (int ms = 0; ms < 2; ++ms) {
        launch_tok_emb_bwd(cur_dx[ms], ws.input_ids_ms[ms], params.d_tok_emb_weight, M, C, stream);
    }
    
    // Accumulate token embedding and compute clipping scale factor on GPU
    launch_accumulate_grad_norm_sq(params.d_tok_emb_weight, cfg.vocab_size * cfg.C, ws.grad_norm_sq, stream);
    launch_compute_clip_coef(ws.grad_norm_sq, ws.clip_coef, 1.0f, stream);
}



// ---------------------------------------------------------------------------
// Phase 22 Interleaved Step Coordinator
// ---------------------------------------------------------------------------
void run_native_training_step_interleaved(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    FullModelParameters& params,
    float lr,
    cudaStream_t stream
) {
    run_interleaved_forward(cfg, ws, params, stream);
    run_interleaved_backward(cfg, ws, params, stream);
    run_fused_optimizer_step(params, ws, cfg, lr, stream);
    
    // Phase 31: If BF16/FP8 moments AdamW is active, FP8 weights are written directly by AdamW kernel; standalone quantize eliminated!
    if (!cfg.use_bf16_moments && !cfg.use_fp8_moments) {
        if (cfg.use_fp8_lm_head || cfg.use_fp8_lm_head_backward) {
            launch_quantize_bf16_to_fp8(params.lm_head_weight, params.lm_head_weight_fp8, 64.0f, cfg.vocab_pad * cfg.C, stream);
        }
        if (cfg.use_fp8_qkv) {
            for (int l = 0; l < cfg.num_layers; ++l) {
                launch_quantize_bf16_to_fp8(params.layers[l].qkv_weight, params.layers[l].qkv_weight_fp8, 64.0f, 3 * cfg.C * cfg.C, stream);
            }
        }
    }
}

