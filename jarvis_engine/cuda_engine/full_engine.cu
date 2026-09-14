#include "full_engine.h"
#include "optimizer.h"
#include "cublaslt_engine.h"
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
    int M, int C, cudaStream_t stream
);

void launch_fused_gelu_fwd(const __nv_bfloat16* in, __nv_bfloat16* out, int num_elements, cudaStream_t stream);
void launch_fused_gelu_bwd(const __nv_bfloat16* grad_out, const __nv_bfloat16* in, __nv_bfloat16* grad_in, int num_elements, cudaStream_t stream);
void launch_fused_add_residual(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int num_elements, cudaStream_t stream);

void launch_fused_cross_entropy_bwd(
    const __nv_bfloat16* logits, const int32_t* targets, __nv_bfloat16* d_logits,
    float* loss_out, int M, int vocab_size, int vocab_pad, float loss_scale, cudaStream_t stream
);

void launch_tok_emb_fwd(
    const int32_t* input_ids, const __nv_bfloat16* emb_weight, __nv_bfloat16* out,
    int M, int C, cudaStream_t stream
);

void launch_tok_emb_bwd(
    const __nv_bfloat16* d_out, const int32_t* input_ids, __nv_bfloat16* d_emb_weight,
    int M, int C, cudaStream_t stream
);

void launch_moe_top2_gating(
    const __nv_bfloat16* logits, float* topk_gates, int32_t* topk_idx, float* l_bal,
    int M, int E, float noise_std, bool training, cudaStream_t stream
);

void launch_moe_compute_maps(
    const int32_t* topk_idx, int32_t* scatter_map, int32_t* gather_map, int32_t* gate_idx_map,
    int M, int E, cudaStream_t stream
);

void launch_moe_dispatch_gather(
    const __nv_bfloat16* x, const int32_t* gather_map, __nv_bfloat16* dispatched_x,
    int total_dispatched, int C, cudaStream_t stream
);

void launch_moe_dispatch_gather_fp8(
    const __nv_fp8_e4m3* x_fp8, const int32_t* gather_map, __nv_fp8_e4m3* dispatched_x_fp8,
    int total_dispatched, int C, cudaStream_t stream
);

void launch_moe_scatter_combine(
    const __nv_bfloat16* dispatched_y, const float* topk_gates, const int32_t* scatter_map,
    __nv_bfloat16* out, int M, int C, cudaStream_t stream
);

void launch_moe_scatter_combine_add_residual(
    const __nv_bfloat16* dispatched_y, const float* topk_gates, const int32_t* scatter_map,
    const __nv_bfloat16* x1, __nv_bfloat16* x2, int M, int C, cudaStream_t stream
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
    
    // 1. Token Embeddings: input_ids -> ws.emb_out
    launch_tok_emb_fwd(ws.input_ids, params.tok_emb_weight, ws.emb_out, M, C, stream);
    
    // Pointer to current layer input (starts with embedding output)
    __nv_bfloat16* cur_x = ws.emb_out;
    
    // 2. Loop through all 24 layers sequentially (L2-pinned layer_x2 active staging)
    for (int l = 0; l < cfg.num_layers; ++l) {
        const auto& lay = params.layers[l];
        
        // Stash layer input for exact analytical backward recomputation (100.66 MB total across 24 layers)
        cudaMemcpyAsync(ws.stashed_x[l], cur_x, M * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        
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
        
        // Step 2c: Attention Out Projection GEMM: (M, C) @ (C, C).T -> (M, C)
        // First C elements of layer_qkv is q_chunk
        cublaslt_gemm_attn_out_fwd(ws.layer_qkv, lay.out_proj_weight, ws.layer_attn_out, M, C, stream);
        
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
            ws.layer_topk_idx, ws.layer_scatter_map, ws.layer_gather_map, ws.layer_gate_idx_map,
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
            cublaslt_gemm_moe_w1_fwd(ws.layer_dispatched_x, lay.w1_weights[0], ws.layer_h1, M * cfg.top_k, C, cfg.hidden_dim, stream);
            launch_fused_gelu_fwd(ws.layer_h1, ws.layer_act, M * cfg.top_k * cfg.hidden_dim, stream);
            cublaslt_gemm_moe_w2_fwd(ws.layer_act, lay.w2_weights[0], ws.layer_dispatched_y, M * cfg.top_k, cfg.hidden_dim, C, stream);
        }
        
        // Step 2h+2i: Fused MoE Scatter Combine + Residual 2 Addition (Single Memory Pass)
        launch_moe_scatter_combine_add_residual(
            ws.layer_dispatched_y, ws.layer_topk_gates, ws.layer_scatter_map,
            ws.layer_x1, ws.layer_x2, M, C, stream
        );
        
        // Layer output becomes input to next layer
        cur_x = ws.layer_x2;
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
    
    // 5. Fused Cross-Entropy Loss and Analytical dLogits
    launch_fused_cross_entropy_bwd(
        ws.logits, ws.targets, ws.d_logits, ws.loss_buffer,
        M, cfg.vocab_size, cfg.vocab_pad, 1.0f / (float)cfg.accum_steps, stream
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
        launch_quantize_bf16_to_fp8(ws.d_logits, ws.d_logits_fp8, scale_dlog, M * cfg.vocab_pad, stream);
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
        
        // Analytical layer gradient computation using stashed input
        // Residual 2 backward: dX2 splits into dX1 and dMoeOut
        launch_fused_rmsnorm_bwd(
            cur_dx, ws.stashed_x[l], lay.norm1_weight,
            ws.layer_rsqrt1, next_dx, lay.d_norm1_weight, M, C, stream
        );
        
        // QKV parameter gradient: compute slice 0 once (overwrites on step 0, accumulates on step 1), replicate to slice 1 and 2
        cublaslt_gemm_qkv_bwd_dw_slice(next_dx, ws.stashed_x[l], lay.d_qkv_weight, M, C, stream, beta);
        replicate_qkv_dw_slices(lay.d_qkv_weight, C, stream);
        
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
    
    // Phase 28: If FP8 LM Head is used (fwd or bwd), re-quantize lm_head_weight to guarantee NO stale buffers
    if (cfg.use_fp8_lm_head || cfg.use_fp8_lm_head_backward) {
        launch_quantize_bf16_to_fp8(params.lm_head_weight, params.lm_head_weight_fp8, 64.0f, cfg.vocab_pad * cfg.C, stream);
    }
    // Phase 30: If FP8 QKV is used, re-quantize layer QKV weights to guarantee NO stale buffers
    if (cfg.use_fp8_qkv) {
        for (int l = 0; l < cfg.num_layers; ++l) {
            launch_quantize_bf16_to_fp8(params.layers[l].qkv_weight, params.layers[l].qkv_weight_fp8, 64.0f, 3 * cfg.C * cfg.C, stream);
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
    
    // 1. Token Embeddings for both microsteps
    launch_tok_emb_fwd(ws.input_ids_ms[0], params.tok_emb_weight, ws.emb_out_ms[0], M, C, stream);
    launch_tok_emb_fwd(ws.input_ids_ms[1], params.tok_emb_weight, ws.emb_out_ms[1], M, C, stream);
    
    __nv_bfloat16* cur_x[2] = { ws.emb_out_ms[0], ws.emb_out_ms[1] };
    
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
            cudaMemcpyAsync(ws.stashed_x_ms[ms][l], cur_x[ms], M * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            
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
            
            // Step 2c: Attention Out GEMM
            cublaslt_gemm_attn_out_fwd(ws.layer_qkv, lay.out_proj_weight, ws.layer_attn_out, M, C, stream);
            
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
            launch_moe_compute_maps(ws.layer_topk_idx, ws.layer_scatter_map, ws.layer_gather_map, ws.layer_gate_idx_map, M, cfg.E, stream);
            
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
                cublaslt_gemm_moe_w1_fwd(ws.layer_dispatched_x, lay.w1_weights[0], ws.layer_h1, M * cfg.top_k, C, cfg.hidden_dim, stream);
                launch_fused_gelu_fwd(ws.layer_h1, ws.layer_act, M * cfg.top_k * cfg.hidden_dim, stream);
                cublaslt_gemm_moe_w2_fwd(ws.layer_act, lay.w2_weights[0], ws.layer_dispatched_y, M * cfg.top_k, cfg.hidden_dim, C, stream);
            }
            
            // Step 2h+2i: Fused MoE Scatter Combine + Residual 2 Addition (Single Memory Pass)
            launch_moe_scatter_combine_add_residual(
                ws.layer_dispatched_y, ws.layer_topk_gates, ws.layer_scatter_map,
                ws.layer_x1, ws.layer_x2_ms[ms], M, C, stream
            );
            
            cur_x[ms] = ws.layer_x2_ms[ms];
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
        
        launch_fused_cross_entropy_bwd(
            ws.logits_ms[ms], ws.targets_ms[ms], ws.d_logits_ms[ms], ws.loss_buffer,
            M, cfg.vocab_size, cfg.vocab_pad, 1.0f / (float)cfg.accum_steps, stream
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
            launch_quantize_bf16_to_fp8(ws.d_logits_ms[ms], ws.d_logits_fp8_ms[ms], scale_dlog, M * cfg.vocab_pad, stream);
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
            launch_fused_rmsnorm_bwd(
                cur_dx[ms], ws.stashed_x_ms[ms][l], lay.norm1_weight,
                ws.layer_rsqrt1, next_dx[ms], lay.d_norm1_weight, M, C, stream
            );
            
            // Execute QKV dW slice 0 matmul once per microstep (overwrites on ms0 via beta=0.0f, accumulates on ms1 via beta=1.0f)
            float beta_qkv = (ms == 0 ? 0.0f : 1.0f);
            cublaslt_gemm_qkv_bwd_dw_slice(next_dx[ms], ws.stashed_x_ms[ms][l], lay.d_qkv_weight, M, C, stream, beta_qkv);
            
            // Zero-overhead ping-pong gradient pointer swap (eliminates cudaMemcpyAsync)
            std::swap(cur_dx[ms], next_dx[ms]);
        }
        
        // Replicate slice 0 (accumulated across both microsteps) to slice 1 and slice 2
        replicate_qkv_dw_slices(lay.d_qkv_weight, C, stream);
        
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
    
    // Phase 28: If FP8 LM Head is used (fwd or bwd), re-quantize lm_head_weight to guarantee NO stale buffers
    if (cfg.use_fp8_lm_head || cfg.use_fp8_lm_head_backward) {
        launch_quantize_bf16_to_fp8(params.lm_head_weight, params.lm_head_weight_fp8, 64.0f, cfg.vocab_pad * cfg.C, stream);
    }
    // Phase 30: If FP8 QKV is used, re-quantize layer QKV weights to guarantee NO stale buffers
    if (cfg.use_fp8_qkv) {
        for (int l = 0; l < cfg.num_layers; ++l) {
            launch_quantize_bf16_to_fp8(params.layers[l].qkv_weight, params.layers[l].qkv_weight_fp8, 64.0f, 3 * cfg.C * cfg.C, stream);
        }
    }
}

