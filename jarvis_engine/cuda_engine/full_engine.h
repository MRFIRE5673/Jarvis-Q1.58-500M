#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <vector>

// Full Jarvis-Q1.58-500M Architecture Configuration
struct FullJarvisConfig {
    int B = 4;              // Microbatch size (B=4 for optimal 12GB VRAM margin)
    int T = 512;            // Sequence length
    int C = 1024;           // Hidden dimension d_model
    int H = 16;             // Number of attention heads
    int D = 64;             // Head dimension (C / H)
    int E = 4;              // Number of MoE experts
    int top_k = 2;          // Top-K experts
    int hidden_dim = 2048;  // MoE hidden dimension (2 * d_model)
    int chunk_size = 64;    // Associative attention chunk size
    int num_layers = 24;    // Number of transformer layers
    int vocab_size = 50257; // GPT-2 vocabulary size
    int vocab_pad = 50304;  // 64-aligned padded vocabulary for Tensor Cores
    int accum_steps = 2;    // Gradient accumulation steps (2 * 4 * 512 = 4096 tokens/update)
    bool use_fp8_moe = false; // Phase 28: Native FP8 MoE forward toggle
    bool use_fp8_lm_head = false; // Phase 28: Native FP8 LM Head forward toggle
    bool use_fp8_lm_head_backward = false; // Phase 28: Native FP8 LM Head backward toggle
    bool use_bf16_moments = false;        // Phase 29: Native BF16 moments optimizer state toggle (14 B/elem)
    bool use_fp8_moments = false;         // Phase 31: Native FP8 moments optimizer state toggle (10 B/elem)
    bool use_fp8_qkv = false;             // Phase 30: Native FP8 QKV forward toggle
    bool use_fused_rmsnorm_quant = true;  // Phase 31: Native Fused RMSNorm + FP8 Activation Quantization toggle
    bool use_fp8_qkv_backward = true;     // Phase 31: Native FP8 QKV backward toggle
    
    int M() const { return B * T; }                     // 2048 tokens per microstep
    int total_tokens_per_step() const { return B * T * accum_steps; } // 4096 tokens
};

// Parameter pointers for ONE Jarvis Block (Layer l)
struct LayerWeights {
    __nv_bfloat16* norm1_weight;     // (C)
    __nv_bfloat16* qkv_weight;       // (3 * C, C)
    __nv_fp8_e4m3* qkv_weight_fp8;   // (3 * C, C) Phase 30: FP8 E4M3 pre-scaled
    __nv_bfloat16* out_proj_weight;  // (C, C)
    __nv_bfloat16* norm2_weight;     // (C)
    __nv_bfloat16* router_weight;    // (E, C)
    __nv_bfloat16* w1_weights[4];    // 4 experts, each (hidden_dim, C)
    __nv_bfloat16* w2_weights[4];    // 4 experts, each (C, hidden_dim)
    __nv_fp8_e4m3* w1_weights_fp8[4];// 4 experts, FP8 E4M3 pre-scaled
    __nv_fp8_e4m3* w2_weights_fp8[4];// 4 experts, FP8 E4M3 pre-scaled
    float*         gamma_raw;        // (H)
    float*         var_scale;        // (1)
    
    // Gradient pointers (accumulated in BF16 or FP32)
    __nv_bfloat16* d_norm1_weight;
    __nv_bfloat16* d_qkv_weight;
    __nv_bfloat16* d_out_proj_weight;
    __nv_bfloat16* d_norm2_weight;
    __nv_bfloat16* d_router_weight;
    __nv_bfloat16* d_w1_weights[4];
    __nv_bfloat16* d_w2_weights[4];
    float*         d_gamma_raw;
    float*         d_var_scale;
    
    // AdamW momentum buffers (tri-precision: FP32, BF16, or FP8)
    union { float* m_norm1; __nv_bfloat16* m_norm1_bf16; __nv_fp8_e4m3* m_norm1_fp8; };
    union { float* v_norm1; __nv_bfloat16* v_norm1_bf16; __nv_fp8_e5m2* v_norm1_fp8; };
    union { float* m_qkv;   __nv_bfloat16* m_qkv_bf16;   __nv_fp8_e4m3* m_qkv_fp8; };
    union { float* v_qkv;   __nv_bfloat16* v_qkv_bf16;   __nv_fp8_e5m2* v_qkv_fp8; };
    union { float* m_out;   __nv_bfloat16* m_out_bf16;   __nv_fp8_e4m3* m_out_fp8; };
    union { float* v_out;   __nv_bfloat16* v_out_bf16;   __nv_fp8_e5m2* v_out_fp8; };
    union { float* m_norm2; __nv_bfloat16* m_norm2_bf16; __nv_fp8_e4m3* m_norm2_fp8; };
    union { float* v_norm2; __nv_bfloat16* v_norm2_bf16; __nv_fp8_e5m2* v_norm2_fp8; };
    union { float* m_router;__nv_bfloat16* m_router_bf16;__nv_fp8_e4m3* m_router_fp8; };
    union { float* v_router;__nv_bfloat16* v_router_bf16;__nv_fp8_e5m2* v_router_fp8; };
    union { float* m_w1[4]; __nv_bfloat16* m_w1_bf16[4]; __nv_fp8_e4m3* m_w1_fp8[4]; };
    union { float* v_w1[4]; __nv_bfloat16* v_w1_bf16[4]; __nv_fp8_e5m2* v_w1_fp8[4]; };
    union { float* m_w2[4]; __nv_bfloat16* m_w2_bf16[4]; __nv_fp8_e4m3* m_w2_fp8[4]; };
    union { float* v_w2[4]; __nv_bfloat16* v_w2_bf16[4]; __nv_fp8_e5m2* v_w2_fp8[4]; };
    float* m_gamma; float* v_gamma;
    float* m_var;   float* v_var;
};

// Global Model Parameters & Buffers
struct FullModelParameters {
    __nv_bfloat16* tok_emb_weight;   // (vocab_size, C)
    __nv_bfloat16* d_tok_emb_weight; // (vocab_size, C)
    union { float* m_tok_emb; __nv_bfloat16* m_tok_emb_bf16; __nv_fp8_e4m3* m_tok_emb_fp8; };
    union { float* v_tok_emb; __nv_bfloat16* v_tok_emb_bf16; __nv_fp8_e5m2* v_tok_emb_fp8; };
    
    __nv_bfloat16* final_norm_weight;   // (C)
    __nv_bfloat16* d_final_norm_weight; // (C)
    union { float* m_final_norm; __nv_bfloat16* m_final_norm_bf16; __nv_fp8_e4m3* m_final_norm_fp8; };
    union { float* v_final_norm; __nv_bfloat16* v_final_norm_bf16; __nv_fp8_e5m2* v_final_norm_fp8; };
    
    __nv_bfloat16* lm_head_weight;   // (vocab_pad, C)
    __nv_fp8_e4m3* lm_head_weight_fp8; // (vocab_pad, C) FP8 E4M3
    __nv_bfloat16* d_lm_head_weight; // (vocab_pad, C)
    union { float* m_lm_head; __nv_bfloat16* m_lm_head_bf16; __nv_fp8_e4m3* m_lm_head_fp8; };
    union { float* v_lm_head; __nv_bfloat16* v_lm_head_bf16; __nv_fp8_e5m2* v_lm_head_fp8; };
    
    LayerWeights layers[24];
};

// Static Persistent Workspace for Full Engine Execution
struct FullModelWorkspace {
    // 1. Input & Embedding Buffers
    int32_t*       input_ids;        // (B, T)
    int32_t*       targets;          // (B, T)
    __nv_bfloat16* emb_out;          // (M, C)
    
    // 2. Stashed Layer Inputs for Zero-Overhead Activation Recomputation
    // Stashing only 24 layer input hidden states requires only:
    // 24 * 2048 * 1024 * 2 bytes = 100.66 MiB!
    __nv_bfloat16* stashed_x[24];    // 24 pointers to (M, C) buffers
    __nv_fp8_e4m3* stashed_x_fp8[24];// 24 pointers to (M, C) FP8 buffers
    
    // 3. Reusable Active Layer Workspace (Shared across all 24 layers sequentially)
    __nv_bfloat16* layer_x_norm1;    // (M, C)
    __nv_fp8_e4m3* layer_x_norm1_fp8;// (M, C) Phase 30: FP8 E4M3
    float*         layer_rsqrt1;     // (M)
    __nv_bfloat16* layer_qkv;        // (M, 3 * C)
    __nv_bfloat16* layer_attn_out;   // (M, C)
    __nv_bfloat16* layer_x1;         // (M, C)
    __nv_bfloat16* layer_x_norm2;    // (M, C)
    __nv_fp8_e4m3* layer_x_norm2_fp8;// (M, C) Phase 31: FP8 E4M3 for MoE dispatch
    float*         layer_rsqrt2;     // (M)
    __nv_bfloat16* layer_router_logits; // (M, E)
    float*         layer_topk_gates;    // (M, top_k)
    int32_t*       layer_topk_idx;      // (M, top_k)
    int32_t*       layer_gather_map;    // (M * top_k)
    int32_t*       layer_scatter_map;   // (M * top_k)
    int32_t*       layer_gate_idx_map;  // (M * top_k)
    __nv_bfloat16* layer_dispatched_x;  // (M * top_k, C)
    __nv_fp8_e4m3* layer_dispatched_x_fp8; // (M * top_k, C) FP8 E4M3
    __nv_bfloat16* layer_h1;            // (M * top_k, hidden_dim)
    __nv_bfloat16* layer_act;           // (M * top_k, hidden_dim)
    __nv_fp8_e4m3* layer_act_fp8;       // (M * top_k, hidden_dim) FP8 E4M3
    __nv_bfloat16* layer_dispatched_y;  // (M * top_k, C)
    __nv_bfloat16* layer_moe_out;       // (M, C)
    __nv_bfloat16* layer_h_out;         // (M, C) LSF state
    __nv_bfloat16* layer_h_last[24];    // (B, C) per layer
    __nv_bfloat16* layer_x2;            // (M, C)
    
    // 4. Output & Loss Buffers
    __nv_bfloat16* final_norm_out;   // (M, C)
    __nv_fp8_e4m3* final_norm_out_fp8; // (M, C) FP8 E4M3
    float*         final_rsqrt;      // (M)
    __nv_bfloat16* logits;           // (M, vocab_pad)
    float*         loss_buffer;      // (1)
    float*         l_bal_total;      // (1)
    float*         l_ref_total;      // (1)
    
    // 5. Backward Gradient Buffers (Reusable layer buffers)
    __nv_bfloat16* d_logits;         // (M, vocab_pad)
    __nv_fp8_e4m3* d_logits_fp8;     // (M, vocab_pad) FP8 E4M3
    __nv_bfloat16* d_final_norm_out; // (M, C)
    __nv_bfloat16* d_layer_x;        // (M, C) ping-pong buffer A
    __nv_fp8_e4m3* d_layer_x_fp8;    // (M, C) FP8 buffer for QKV backward
    __nv_bfloat16* d_layer_x_prev;   // (M, C) ping-pong buffer B
    __nv_bfloat16* d_layer_x1;       // (M, C)
    __nv_bfloat16* d_layer_attn_out; // (M, C)
    __nv_bfloat16* d_layer_moe_out;  // (M, C)
    __nv_bfloat16* d_dispatched_y;   // (M * top_k, C)
    __nv_bfloat16* d_act;            // (M * top_k, hidden_dim)
    __nv_bfloat16* d_h1;             // (M * top_k, hidden_dim)
    __nv_bfloat16* d_dispatched_x;   // (M * top_k, C)
    float*         d_topk_gates;     // (M, top_k)
    
    // Optimizer reduction workspace
    float*         grad_norm_sq;     // (1) Device reduction buffer for ||g||^2
    float*         clip_coef;        // (1) Device clipping coefficient buffer
    
    // 6. Phase 22: Layer-Wise Interleaved Microstep Buffers (2 Microsteps)
    int32_t*       input_ids_ms[2];
    int32_t*       targets_ms[2];
    __nv_bfloat16* emb_out_ms[2];
    __nv_bfloat16* stashed_x_ms[2][24];
    __nv_fp8_e4m3* stashed_x_fp8_ms[2][24];
    __nv_bfloat16* layer_x2_ms[2];
    __nv_bfloat16* final_norm_out_ms[2];
    __nv_fp8_e4m3* final_norm_out_fp8_ms[2];
    float*         final_rsqrt_ms[2];
    __nv_bfloat16* logits_ms[2];
    __nv_bfloat16* d_logits_ms[2];
    __nv_fp8_e4m3* d_logits_fp8_ms[2];
    __nv_bfloat16* d_final_norm_out_ms[2];
    __nv_bfloat16* d_layer_x_ms[2];
    __nv_fp8_e4m3* d_layer_x_fp8_ms[2];
    __nv_bfloat16* d_layer_x_prev_ms[2];
    
    // 7. EXP-24-010 Consolidated Grouped AdamW Pointer Tables
    void*          d_all_moe_experts;
    void*          d_all_qkv;
    void*          d_all_out_proj;
    void*          d_all_small_params;
    bool           opt_tables_synced;
    
    // Memory accounting
    size_t total_workspace_bytes;
    bool is_initialized;
};

// Phase 22 Interleaved Step Declarations
void run_native_training_step_interleaved(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    FullModelParameters& params,
    float lr,
    cudaStream_t stream
);
