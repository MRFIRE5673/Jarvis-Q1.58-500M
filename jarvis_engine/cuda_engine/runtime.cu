#include "runtime.h"
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
    
    // Optimizer reduction buffer
    alloc_f32(ws.grad_norm_sq, 1);
    alloc_f32(ws.clip_coef, 1);
    
    // EXP-24-010 Consolidated Grouped AdamW Pointer Tables
    CHECK_CUDA(cudaMalloc(&ws.d_all_moe_experts, 192 * 4 * sizeof(void*)));
    total += 192 * 4 * sizeof(void*);
    CHECK_CUDA(cudaMalloc(&ws.d_all_qkv, 24 * 4 * sizeof(void*)));
    total += 24 * 4 * sizeof(void*);
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
    
    free_p((void*&)ws.grad_norm_sq);
    free_p((void*&)ws.clip_coef);
    
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
