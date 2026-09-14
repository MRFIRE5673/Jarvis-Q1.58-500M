#include "runtime.h"
#include "attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <string.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        } \
    } while (0)

FullModelWorkspace allocate_full_workspace(const FullJarvisConfig& cfg) {
    FullModelWorkspace ws;
    memset(&ws, 0, sizeof(ws));
    
    int M = cfg.M();
    int C = cfg.C;
    int top_k = cfg.top_k;
    int hidden_dim = cfg.hidden_dim;
    int E = cfg.E;
    int vocab_pad = cfg.vocab_pad;
    
    size_t total = 0;
    
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
    auto alloc_fp8 = [&](__nv_fp8_e4m3*& ptr, size_t count) {
        size_t bytes = count * sizeof(__nv_fp8_e4m3);
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        total += bytes;
    };
    
    // 1. Input & Embedding Buffers (Phase 22 Dual Microstep Allocation)
    for (int ms = 0; ms < 2; ++ms) {
        alloc_i32(ws.input_ids_ms[ms], cfg.B * cfg.T);
        alloc_i32(ws.targets_ms[ms], cfg.B * cfg.T);
        alloc_bf16(ws.emb_out_ms[ms], M * C);
        alloc_bf16(ws.layer_x2_ms[ms], M * C);
        alloc_bf16(ws.final_norm_out_ms[ms], M * C);
        alloc_fp8(ws.final_norm_out_fp8_ms[ms], M * C);
        alloc_f32(ws.final_rsqrt_ms[ms], M);
        alloc_bf16(ws.logits_ms[ms], M * vocab_pad);
        alloc_bf16(ws.d_logits_ms[ms], M * vocab_pad);
        alloc_fp8(ws.d_logits_fp8_ms[ms], (size_t)M * vocab_pad);
        alloc_bf16(ws.d_final_norm_out_ms[ms], M * C);
        alloc_bf16(ws.d_layer_x_ms[ms], M * C);
        alloc_fp8(ws.d_layer_x_fp8_ms[ms], M * C);
        alloc_bf16(ws.d_layer_x_prev_ms[ms], M * C);
    }
    // Backward compatibility aliasing
    ws.input_ids = ws.input_ids_ms[0];
    ws.targets = ws.targets_ms[0];
    ws.emb_out = ws.emb_out_ms[0];
    ws.final_norm_out = ws.final_norm_out_ms[0];
    ws.final_norm_out_fp8 = ws.final_norm_out_fp8_ms[0];
    ws.final_rsqrt = ws.final_rsqrt_ms[0];
    ws.logits = ws.logits_ms[0];
    ws.d_logits = ws.d_logits_ms[0];
    ws.d_logits_fp8 = ws.d_logits_fp8_ms[0];
    ws.d_final_norm_out = ws.d_final_norm_out_ms[0];
    ws.d_layer_x = ws.d_layer_x_ms[0];
    ws.d_layer_x_fp8 = ws.d_layer_x_fp8_ms[0];
    ws.d_layer_x_prev = ws.d_layer_x_prev_ms[0];
    
    // 2. Stashed Layer Inputs for Zero-Overhead Activation Recomputation
    // 2 microsteps * 24 layers * (M, C) BF16 + FP8
    for (int ms = 0; ms < 2; ++ms) {
        for (int l = 0; l < cfg.num_layers; ++l) {
            alloc_bf16(ws.stashed_x_ms[ms][l], M * C);
            alloc_fp8(ws.stashed_x_fp8_ms[ms][l], M * C);
        }
    }
    for (int l = 0; l < cfg.num_layers; ++l) {
        ws.stashed_x[l] = ws.stashed_x_ms[0][l];
        ws.stashed_x_fp8[l] = ws.stashed_x_fp8_ms[0][l];
        alloc_bf16(ws.layer_h_last[l], cfg.B * C);
    }
    
    // 3. Reusable Active Layer Workspace (Shared across all 24 layers sequentially)
    alloc_bf16(ws.layer_x_norm1, M * C);
    alloc_fp8(ws.layer_x_norm1_fp8, M * C);
    alloc_f32(ws.layer_rsqrt1, M);
    alloc_bf16(ws.layer_qkv, M * 3 * C);
    alloc_bf16(ws.layer_attn_out, M * C);
    alloc_bf16(ws.layer_x1, M * C);
    alloc_bf16(ws.layer_x_norm2, M * C);
    alloc_fp8(ws.layer_x_norm2_fp8, M * C);
    alloc_f32(ws.layer_rsqrt2, M);
    alloc_bf16(ws.layer_router_logits, M * E);
    alloc_f32(ws.layer_topk_gates, M * top_k);
    alloc_i32(ws.layer_topk_idx, M * top_k);
    alloc_i32(ws.layer_gather_map, M * top_k);
    alloc_i32(ws.layer_scatter_map, M * top_k);
    alloc_i32(ws.layer_gate_idx_map, M * top_k);
    alloc_bf16(ws.layer_dispatched_x, M * top_k * C);
    alloc_fp8(ws.layer_dispatched_x_fp8, M * top_k * C);
    alloc_bf16(ws.layer_h1, M * top_k * hidden_dim);
    alloc_bf16(ws.layer_act, M * top_k * hidden_dim);
    alloc_fp8(ws.layer_act_fp8, M * top_k * hidden_dim);
    alloc_bf16(ws.layer_dispatched_y, M * top_k * C);
    alloc_bf16(ws.layer_moe_out, M * C);
    alloc_bf16(ws.layer_h_out, M * C);
    alloc_bf16(ws.layer_x2, M * C);
    
    // 3b. Native Associative Linear Attention Buffers
    const size_t chunk_mat_elements = (size_t)cfg.B * cfg.H * (cfg.T / cfg.chunk_size) * cfg.chunk_size * cfg.D;
    alloc_bf16(ws.attn_q_chunks, chunk_mat_elements);
    alloc_bf16(ws.attn_k_chunks, chunk_mat_elements);
    alloc_bf16(ws.attn_v_chunks, chunk_mat_elements);
    alloc_bf16(ws.attn_v_w_chunks, chunk_mat_elements);
    alloc_bf16(ws.attn_scores, chunk_mat_elements);
    alloc_bf16(ws.attn_intra_out, chunk_mat_elements);
    alloc_bf16(ws.attn_delta_s, chunk_mat_elements);
    alloc_bf16(ws.attn_all_states, chunk_mat_elements);
    alloc_bf16(ws.attn_cross_out, chunk_mat_elements);
    alloc_bf16(ws.layer_attn_context, M * C);
    
    alloc_bf16(ws.cos_tab, cfg.T * cfg.D);
    alloc_bf16(ws.sin_tab, cfg.T * cfg.D);
    alloc_bf16(ws.decay_mat_tab, cfg.H * cfg.chunk_size * cfg.chunk_size);
    alloc_bf16(ws.gamma_cross_tab, cfg.H * cfg.chunk_size);
    alloc_bf16(ws.gw_tab, cfg.H * cfg.chunk_size);
    alloc_f32(ws.gamma_c_tab, cfg.H);
    
    const size_t attn_state_elements = (size_t)cfg.B * cfg.H * cfg.D * cfg.D;
    for (int l = 0; l < cfg.num_layers; ++l) {
        alloc_bf16(ws.layer_attn_state[l], attn_state_elements);
        cudaMemset(ws.layer_attn_state[l], 0, attn_state_elements * sizeof(__nv_bfloat16));
    }
    
    // 3c. Native Associative Linear Attention Backward Buffers
    alloc_bf16(ws.attn_d_raw_cross, chunk_mat_elements);
    alloc_bf16(ws.attn_dq_cross, chunk_mat_elements);
    alloc_bf16(ws.attn_ds_all, chunk_mat_elements);
    alloc_bf16(ws.attn_d_delta_s, chunk_mat_elements);
    alloc_bf16(ws.attn_dk_delta, chunk_mat_elements);
    alloc_bf16(ws.attn_dv_w, chunk_mat_elements);
    alloc_bf16(ws.attn_d_scores, chunk_mat_elements);
    alloc_bf16(ws.attn_dv_intra, chunk_mat_elements);
    alloc_bf16(ws.attn_dq_intra, chunk_mat_elements);
    alloc_bf16(ws.attn_dk_intra, chunk_mat_elements);
    alloc_bf16(ws.attn_d_context, M * C);
    alloc_bf16(ws.d_layer_qkv, M * 3 * C);
    alloc_bf16(ws.dx_norm1, M * C);
    alloc_bf16(ws.dx_norm1_in, M * C);
    
    // Initialize static rotary position embedding tables
    init_attention_rotary_tables(ws, cfg, 0);
    cudaDeviceSynchronize();
    
    // 4. Output & Loss Buffers
    alloc_f32(ws.loss_buffer, 1);
    alloc_f32(ws.l_bal_total, 1);
    alloc_f32(ws.l_ref_total, 1);
    
    // 5. Backward Gradient Buffers (Scratch buffers)
    alloc_bf16(ws.d_layer_x1, M * C);
    alloc_bf16(ws.d_layer_attn_out, M * C);
    alloc_bf16(ws.d_layer_moe_out, M * C);
    alloc_bf16(ws.d_dispatched_y, M * top_k * C);
    alloc_bf16(ws.d_act, M * top_k * hidden_dim);
    alloc_bf16(ws.d_h1, M * top_k * hidden_dim);
    alloc_bf16(ws.d_dispatched_x, M * top_k * C);
    alloc_f32(ws.d_topk_gates, M * top_k);
    alloc_i32(ws.layer_expert_offsets, E + 1);
    alloc_bf16(ws.d_router_logits, M * E);
    alloc_bf16(ws.dx_norm2_expert, M * C);
    alloc_bf16(ws.dx_norm2, M * C);
    alloc_bf16(ws.dx_norm2_in, M * C);
    alloc_bf16(ws.dx1, M * C);
    
    // Optimizer reduction buffer
    alloc_f32(ws.grad_norm_sq, 1);
    alloc_f32(ws.clip_coef, 1);
    alloc_f32(ws.lsf_alpha_buf, 1);
    alloc_f32(ws.lsf_mean_var_buf, 2);
    alloc_f32(ws.lsf_grad_alpha_buf, 64);
    alloc_f32(ws.d_gamma_c, 16);
    ws.step_count = 0;
    
    // EXP-24-010 Consolidated Grouped AdamW Pointer Tables
    CHECK_CUDA(cudaMalloc(&ws.d_all_moe_experts, 192 * 4 * sizeof(void*)));
    total += 192 * 4 * sizeof(void*);
    CHECK_CUDA(cudaMalloc(&ws.d_all_qkv, 24 * 5 * sizeof(void*)));
    total += 24 * 5 * sizeof(void*);
    CHECK_CUDA(cudaMalloc(&ws.d_all_out_proj, 24 * 4 * sizeof(void*)));
    total += 24 * 4 * sizeof(void*);
    CHECK_CUDA(cudaMalloc(&ws.d_all_small_params, (24 * 20 + 4) * sizeof(void*)));
    total += (24 * 20 + 4) * sizeof(void*);
    ws.opt_tables_synced = false;
    
    ws.total_workspace_bytes = total;
    ws.is_initialized = true;
    return ws;
}

void free_full_workspace(FullModelWorkspace& ws) {
    if (!ws.is_initialized) return;
    
    auto free_p = [](void*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    
    for (int ms = 0; ms < 2; ++ms) {
        free_p((void*&)ws.input_ids_ms[ms]);
        free_p((void*&)ws.targets_ms[ms]);
        free_p((void*&)ws.emb_out_ms[ms]);
        free_p((void*&)ws.layer_x2_ms[ms]);
        free_p((void*&)ws.final_norm_out_ms[ms]);
        free_p((void*&)ws.final_norm_out_fp8_ms[ms]);
        free_p((void*&)ws.final_rsqrt_ms[ms]);
        free_p((void*&)ws.logits_ms[ms]);
        free_p((void*&)ws.d_logits_ms[ms]);
        free_p((void*&)ws.d_logits_fp8_ms[ms]);
        free_p((void*&)ws.d_final_norm_out_ms[ms]);
        free_p((void*&)ws.d_layer_x_ms[ms]);
        free_p((void*&)ws.d_layer_x_fp8_ms[ms]);
        free_p((void*&)ws.d_layer_x_prev_ms[ms]);
        for (int l = 0; l < 24; ++l) {
            free_p((void*&)ws.stashed_x_ms[ms][l]);
            free_p((void*&)ws.stashed_x_fp8_ms[ms][l]);
        }
    }
    
    for (int l = 0; l < 24; ++l) {
        free_p((void*&)ws.layer_h_last[l]);
    }
    
    free_p((void*&)ws.layer_x_norm1);
    free_p((void*&)ws.layer_x_norm1_fp8);
    free_p((void*&)ws.layer_rsqrt1);
    free_p((void*&)ws.layer_qkv);
    free_p((void*&)ws.layer_attn_out);
    free_p((void*&)ws.layer_x1);
    free_p((void*&)ws.layer_x_norm2);
    free_p((void*&)ws.layer_x_norm2_fp8);
    free_p((void*&)ws.layer_rsqrt2);
    free_p((void*&)ws.layer_router_logits);
    free_p((void*&)ws.layer_topk_gates);
    free_p((void*&)ws.layer_topk_idx);
    free_p((void*&)ws.layer_gather_map);
    free_p((void*&)ws.layer_scatter_map);
    free_p((void*&)ws.layer_gate_idx_map);
    free_p((void*&)ws.layer_dispatched_x);
    free_p((void*&)ws.layer_dispatched_x_fp8);
    free_p((void*&)ws.layer_h1);
    free_p((void*&)ws.layer_act);
    free_p((void*&)ws.layer_act_fp8);
    free_p((void*&)ws.layer_dispatched_y);
    free_p((void*&)ws.layer_moe_out);
    free_p((void*&)ws.layer_h_out);
    free_p((void*&)ws.layer_x2);
    
    free_p((void*&)ws.attn_q_chunks);
    free_p((void*&)ws.attn_k_chunks);
    free_p((void*&)ws.attn_v_chunks);
    free_p((void*&)ws.attn_v_w_chunks);
    free_p((void*&)ws.attn_scores);
    free_p((void*&)ws.attn_intra_out);
    free_p((void*&)ws.attn_delta_s);
    free_p((void*&)ws.attn_all_states);
    free_p((void*&)ws.attn_cross_out);
    free_p((void*&)ws.layer_attn_context);
    free_p((void*&)ws.cos_tab);
    free_p((void*&)ws.sin_tab);
    free_p((void*&)ws.decay_mat_tab);
    free_p((void*&)ws.gamma_cross_tab);
    free_p((void*&)ws.gw_tab);
    free_p((void*&)ws.gamma_c_tab);
    for (int l = 0; l < 24; ++l) {
        free_p((void*&)ws.layer_attn_state[l]);
    }
    
    free_p((void*&)ws.attn_d_raw_cross);
    free_p((void*&)ws.attn_dq_cross);
    free_p((void*&)ws.attn_ds_all);
    free_p((void*&)ws.attn_d_delta_s);
    free_p((void*&)ws.attn_dk_delta);
    free_p((void*&)ws.attn_dv_w);
    free_p((void*&)ws.attn_d_scores);
    free_p((void*&)ws.attn_dv_intra);
    free_p((void*&)ws.attn_dq_intra);
    free_p((void*&)ws.attn_dk_intra);
    free_p((void*&)ws.attn_d_context);
    free_p((void*&)ws.d_layer_qkv);
    free_p((void*&)ws.dx_norm1);
    free_p((void*&)ws.dx_norm1_in);
    
    free_p((void*&)ws.loss_buffer);
    free_p((void*&)ws.l_bal_total);
    free_p((void*&)ws.l_ref_total);
    
    free_p((void*&)ws.d_layer_x1);
    free_p((void*&)ws.d_layer_attn_out);
    free_p((void*&)ws.d_layer_moe_out);
    free_p((void*&)ws.d_dispatched_y);
    free_p((void*&)ws.d_act);
    free_p((void*&)ws.d_h1);
    free_p((void*&)ws.d_dispatched_x);

    free_p((void*&)ws.d_topk_gates);
    free_p((void*&)ws.layer_expert_offsets);
    free_p((void*&)ws.d_router_logits);
    free_p((void*&)ws.dx_norm2_expert);
    free_p((void*&)ws.dx_norm2);
    free_p((void*&)ws.dx_norm2_in);
    free_p((void*&)ws.dx1);
    
    free_p((void*&)ws.grad_norm_sq);
    free_p((void*&)ws.clip_coef);
    free_p((void*&)ws.lsf_alpha_buf);
    free_p((void*&)ws.lsf_mean_var_buf);
    free_p((void*&)ws.lsf_grad_alpha_buf);
    free_p((void*&)ws.d_gamma_c);
    
    free_p(ws.d_all_moe_experts);
    free_p(ws.d_all_qkv);
    free_p(ws.d_all_out_proj);
    free_p(ws.d_all_small_params);
    ws.opt_tables_synced = false;
    
    ws.total_workspace_bytes = 0;
    ws.is_initialized = false;
}

void replay_graph_step(CUDAGraphContext& ctx, cudaStream_t stream) {
    if (ctx.is_captured && ctx.instance) {
        CHECK_CUDA(cudaGraphLaunch(ctx.instance, stream));
    }
}

void destroy_graph(CUDAGraphContext& ctx) {
    if (ctx.instance) {
        cudaGraphExecDestroy(ctx.instance);
        ctx.instance = nullptr;
    }
    if (ctx.graph) {
        cudaGraphDestroy(ctx.graph);
        ctx.graph = nullptr;
    }
    ctx.is_captured = false;
}
