#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_engine.h"
#include "full_engine.h"
#include "runtime.h"
#include "optimizer.h"
#include "cublaslt_engine.h"
#include "attention.h"
#include "moe.h"
#include <cuda_fp8.h>

void launch_quantize_bf16_to_fp8(
    const __nv_bfloat16* in, __nv_fp8_e4m3* out, float scale, int N, cudaStream_t stream
);

// Forward declaration from full_engine.cu
void run_full_model_forward(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    const FullModelParameters& params,
    cudaStream_t stream
);

void run_full_model_backward(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    FullModelParameters& params,
    cudaStream_t stream
);

void run_native_training_step(
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    FullModelParameters& params,
    float lr,
    cudaStream_t stream
);

// Global static instances
static FullJarvisConfig g_cfg;
static FullModelWorkspace g_ws;
static FullModelParameters g_params;
static CUDAGraphContext g_graph_ctx;
static bool g_engine_initialized = false;

void cleanup_full_engine();

// ---------------------------------------------------------------------------
// Engine Lifecycle
// ---------------------------------------------------------------------------
void init_full_engine(
    int B, int T, int C, int H, int E, int top_k, int hidden_dim,
    int chunk_size, int num_layers, int vocab_size, int vocab_pad, int accum_steps
) {
    if (g_engine_initialized) {
        cleanup_full_engine();
    }
    
    g_cfg.B = B;
    g_cfg.T = T;
    g_cfg.C = C;
    g_cfg.H = H;
    g_cfg.D = C / H;
    g_cfg.E = E;
    g_cfg.top_k = top_k;
    g_cfg.hidden_dim = hidden_dim;
    g_cfg.chunk_size = chunk_size;
    g_cfg.num_layers = num_layers;
    g_cfg.vocab_size = vocab_size;
    g_cfg.vocab_pad = vocab_pad;
    g_cfg.accum_steps = accum_steps;
    
    g_ws = allocate_full_workspace(g_cfg);
    init_cublaslt_engine();
    
    // Allocate AdamW momentum buffers (dual precision: FP32 or BF16) and gradient buffers
    // Global parameters
    auto alloc_f32 = [](float*& ptr, size_t count) {
        cudaMalloc(&ptr, count * sizeof(float));
        cudaMemset(ptr, 0, count * sizeof(float));
    };
    auto alloc_bf16 = [](__nv_bfloat16*& ptr, size_t count) {
        cudaMalloc(&ptr, count * sizeof(__nv_bfloat16));
        cudaMemset(ptr, 0, count * sizeof(__nv_bfloat16));
    };
    auto alloc_moments_m = [](auto& ptr, size_t count) {
        if (g_cfg.use_fp8_moments) {
            cudaMalloc((void**)&ptr, count * sizeof(__nv_fp8_e4m3));
            cudaMemset(ptr, 0, count * sizeof(__nv_fp8_e4m3));
        } else if (g_cfg.use_bf16_moments) {
            cudaMalloc((void**)&ptr, count * sizeof(__nv_bfloat16));
            cudaMemset(ptr, 0, count * sizeof(__nv_bfloat16));
        } else {
            cudaMalloc((void**)&ptr, count * sizeof(float));
            cudaMemset(ptr, 0, count * sizeof(float));
        }
    };
    auto alloc_moments_v = [](auto& ptr, size_t count) {
        if (g_cfg.use_fp8_moments) {
            cudaMalloc((void**)&ptr, count * sizeof(__nv_fp8_e5m2));
            cudaMemset(ptr, 0, count * sizeof(__nv_fp8_e5m2));
        } else if (g_cfg.use_bf16_moments) {
            cudaMalloc((void**)&ptr, count * sizeof(__nv_bfloat16));
            cudaMemset(ptr, 0, count * sizeof(__nv_bfloat16));
        } else {
            cudaMalloc((void**)&ptr, count * sizeof(float));
            cudaMemset(ptr, 0, count * sizeof(float));
        }
    };
    
    alloc_bf16(g_params.d_tok_emb_weight, g_cfg.vocab_size * g_cfg.C);
    alloc_moments_m(g_params.m_tok_emb, g_cfg.vocab_size * g_cfg.C);
    alloc_moments_v(g_params.v_tok_emb, g_cfg.vocab_size * g_cfg.C);
    
    alloc_bf16(g_params.d_final_norm_weight, g_cfg.C);
    alloc_moments_m(g_params.m_final_norm, g_cfg.C);
    alloc_moments_v(g_params.v_final_norm, g_cfg.C);
    
    alloc_bf16(g_params.d_lm_head_weight, g_cfg.vocab_pad * g_cfg.C);
    alloc_moments_m(g_params.m_lm_head, g_cfg.vocab_pad * g_cfg.C);
    alloc_moments_v(g_params.v_lm_head, g_cfg.vocab_pad * g_cfg.C);
    
    for (int l = 0; l < g_cfg.num_layers; ++l) {
        auto& lay = g_params.layers[l];
        alloc_bf16(lay.d_norm1_weight, g_cfg.C);
        alloc_moments_m(lay.m_norm1, g_cfg.C);
        alloc_moments_v(lay.v_norm1, g_cfg.C);
        
        lay.qkv_weight_fp8 = nullptr;
        alloc_bf16(lay.d_qkv_weight, 3 * g_cfg.C * g_cfg.C);
        alloc_moments_m(lay.m_qkv, 3 * g_cfg.C * g_cfg.C);
        alloc_moments_v(lay.v_qkv, 3 * g_cfg.C * g_cfg.C);
        
        alloc_bf16(lay.d_out_proj_weight, g_cfg.C * g_cfg.C);
        alloc_moments_m(lay.m_out, g_cfg.C * g_cfg.C);
        alloc_moments_v(lay.v_out, g_cfg.C * g_cfg.C);
        
        alloc_bf16(lay.d_norm2_weight, g_cfg.C);
        alloc_moments_m(lay.m_norm2, g_cfg.C);
        alloc_moments_v(lay.v_norm2, g_cfg.C);
        
        alloc_bf16(lay.d_router_weight, g_cfg.E * g_cfg.C);
        alloc_moments_m(lay.m_router, g_cfg.E * g_cfg.C);
        alloc_moments_v(lay.v_router, g_cfg.E * g_cfg.C);
        
        for (int e = 0; e < g_cfg.E; ++e) {
            alloc_bf16(lay.d_w1_weights[e], g_cfg.hidden_dim * g_cfg.C);
            alloc_moments_m(lay.m_w1[e], g_cfg.hidden_dim * g_cfg.C);
            alloc_moments_v(lay.v_w1[e], g_cfg.hidden_dim * g_cfg.C);
            
            alloc_bf16(lay.d_w2_weights[e], g_cfg.C * g_cfg.hidden_dim);
            alloc_moments_m(lay.m_w2[e], g_cfg.C * g_cfg.hidden_dim);
            alloc_moments_v(lay.v_w2[e], g_cfg.C * g_cfg.hidden_dim);
        }
        
        alloc_f32(lay.d_gamma_raw, g_cfg.H);
        alloc_f32(lay.m_gamma, g_cfg.H);
        alloc_f32(lay.v_gamma, g_cfg.H);
        
        alloc_f32(lay.d_var_scale, 1);
        alloc_f32(lay.m_var, 1);
        alloc_f32(lay.v_var, 1);
    }
    
    g_engine_initialized = true;
}

void cleanup_full_engine() {
    if (!g_engine_initialized) return;
    auto free_p = [](void*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    for (int l = 0; l < g_cfg.num_layers; ++l) {
        auto& lay = g_params.layers[l];
        for (int e = 0; e < 4; ++e) {
            if (lay.w1_weights_fp8[e]) {
                cudaFree(lay.w1_weights_fp8[e]);
                lay.w1_weights_fp8[e] = nullptr;
            }
            if (lay.w2_weights_fp8[e]) {
                cudaFree(lay.w2_weights_fp8[e]);
                lay.w2_weights_fp8[e] = nullptr;
            }
            free_p((void*&)lay.d_w1_weights[e]);
            free_p((void*&)lay.m_w1[e]);
            free_p((void*&)lay.v_w1[e]);
            free_p((void*&)lay.d_w2_weights[e]);
            free_p((void*&)lay.m_w2[e]);
            free_p((void*&)lay.v_w2[e]);
        }
        if (lay.qkv_weight_fp8) {
            cudaFree(lay.qkv_weight_fp8);
            lay.qkv_weight_fp8 = nullptr;
        }
        free_p((void*&)lay.d_norm1_weight);
        free_p((void*&)lay.m_norm1);
        free_p((void*&)lay.v_norm1);
        free_p((void*&)lay.d_qkv_weight);
        free_p((void*&)lay.m_qkv);
        free_p((void*&)lay.v_qkv);
        free_p((void*&)lay.d_out_proj_weight);
        free_p((void*&)lay.m_out);
        free_p((void*&)lay.v_out);
        free_p((void*&)lay.d_norm2_weight);
        free_p((void*&)lay.m_norm2);
        free_p((void*&)lay.v_norm2);
        free_p((void*&)lay.d_router_weight);
        free_p((void*&)lay.m_router);
        free_p((void*&)lay.v_router);
        free_p((void*&)lay.d_gamma_raw);
        free_p((void*&)lay.m_gamma);
        free_p((void*&)lay.v_gamma);
        free_p((void*&)lay.d_var_scale);
        free_p((void*&)lay.m_var);
        free_p((void*&)lay.v_var);
    }
    if (g_params.lm_head_weight_fp8) {
        cudaFree(g_params.lm_head_weight_fp8);
        g_params.lm_head_weight_fp8 = nullptr;
    }
    free_p((void*&)g_params.d_tok_emb_weight);
    free_p((void*&)g_params.m_tok_emb);
    free_p((void*&)g_params.v_tok_emb);
    free_p((void*&)g_params.d_final_norm_weight);
    free_p((void*&)g_params.m_final_norm);
    free_p((void*&)g_params.v_final_norm);
    free_p((void*&)g_params.d_lm_head_weight);
    free_p((void*&)g_params.m_lm_head);
    free_p((void*&)g_params.v_lm_head);

    cleanup_cublaslt_engine();
    free_full_workspace(g_ws);
    destroy_graph(g_graph_ctx);
    g_engine_initialized = false;
}

size_t get_workspace_memory_bytes() {
    return g_engine_initialized ? g_ws.total_workspace_bytes : 0;
}

// ---------------------------------------------------------------------------
// Zero-Copy Parameter Binding (Points directly to model parameters)
// ---------------------------------------------------------------------------
void bind_global_parameters(
    torch::Tensor tok_emb,
    torch::Tensor final_norm,
    torch::Tensor lm_head
) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    g_params.tok_emb_weight = reinterpret_cast<__nv_bfloat16*>(tok_emb.data_ptr<at::BFloat16>());
    g_params.final_norm_weight = reinterpret_cast<__nv_bfloat16*>(final_norm.data_ptr<at::BFloat16>());
    g_params.lm_head_weight = reinterpret_cast<__nv_bfloat16*>(lm_head.data_ptr<at::BFloat16>());
    
    // Phase 28: Quantize LM Head weight to FP8 E4M3 with scale 64.0f
    if (!g_params.lm_head_weight_fp8) {
        cudaMalloc(&g_params.lm_head_weight_fp8, (size_t)g_cfg.vocab_pad * g_cfg.C * sizeof(__nv_fp8_e4m3));
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_quantize_bf16_to_fp8(g_params.lm_head_weight, g_params.lm_head_weight_fp8, 64.0f, g_cfg.vocab_pad * g_cfg.C, stream);
    
    g_ws.opt_tables_synced = false;
}

void bind_layer_parameters(
    int layer_idx,
    torch::Tensor norm1_w,
    torch::Tensor qkv_w,
    torch::Tensor out_proj_w,
    torch::Tensor norm2_w,
    torch::Tensor router_w,
    torch::Tensor w1_0, torch::Tensor w1_1, torch::Tensor w1_2, torch::Tensor w1_3,
    torch::Tensor w2_0, torch::Tensor w2_1, torch::Tensor w2_2, torch::Tensor w2_3,
    torch::Tensor gamma_raw,
    torch::Tensor var_scale
) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < g_cfg.num_layers, "Invalid layer index");
    
    auto& lay = g_params.layers[layer_idx];
    lay.norm1_weight = reinterpret_cast<__nv_bfloat16*>(norm1_w.data_ptr<at::BFloat16>());
    lay.qkv_weight = reinterpret_cast<__nv_bfloat16*>(qkv_w.data_ptr<at::BFloat16>());
    lay.out_proj_weight = reinterpret_cast<__nv_bfloat16*>(out_proj_w.data_ptr<at::BFloat16>());
    lay.norm2_weight = reinterpret_cast<__nv_bfloat16*>(norm2_w.data_ptr<at::BFloat16>());
    lay.router_weight = reinterpret_cast<__nv_bfloat16*>(router_w.data_ptr<at::BFloat16>());
    
    lay.w1_weights[0] = reinterpret_cast<__nv_bfloat16*>(w1_0.data_ptr<at::BFloat16>());
    lay.w1_weights[1] = reinterpret_cast<__nv_bfloat16*>(w1_1.data_ptr<at::BFloat16>());
    lay.w1_weights[2] = reinterpret_cast<__nv_bfloat16*>(w1_2.data_ptr<at::BFloat16>());
    lay.w1_weights[3] = reinterpret_cast<__nv_bfloat16*>(w1_3.data_ptr<at::BFloat16>());
    
    lay.w2_weights[0] = reinterpret_cast<__nv_bfloat16*>(w2_0.data_ptr<at::BFloat16>());
    lay.w2_weights[1] = reinterpret_cast<__nv_bfloat16*>(w2_1.data_ptr<at::BFloat16>());
    lay.w2_weights[2] = reinterpret_cast<__nv_bfloat16*>(w2_2.data_ptr<at::BFloat16>());
    lay.w2_weights[3] = reinterpret_cast<__nv_bfloat16*>(w2_3.data_ptr<at::BFloat16>());
    
    lay.gamma_raw = gamma_raw.data_ptr<float>();
    lay.var_scale = var_scale.data_ptr<float>();
    
    // Phase 28: Quantize MoE weights w1 and w2 to FP8 E4M3 with scale 64.0f
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    for (int e = 0; e < 4; ++e) {
        if (!lay.w1_weights_fp8[e]) {
            cudaMalloc(&lay.w1_weights_fp8[e], (size_t)g_cfg.hidden_dim * g_cfg.C * sizeof(__nv_fp8_e4m3));
        }
        if (!lay.w2_weights_fp8[e]) {
            cudaMalloc(&lay.w2_weights_fp8[e], (size_t)g_cfg.C * g_cfg.hidden_dim * sizeof(__nv_fp8_e4m3));
        }
        launch_quantize_bf16_to_fp8(lay.w1_weights[e], lay.w1_weights_fp8[e], 64.0f, g_cfg.hidden_dim * g_cfg.C, stream);
        launch_quantize_bf16_to_fp8(lay.w2_weights[e], lay.w2_weights_fp8[e], 64.0f, g_cfg.C * g_cfg.hidden_dim, stream);
    }
    
    // Phase 30: Quantize QKV weight to FP8 E4M3 with scale 64.0f
    if (!lay.qkv_weight_fp8) {
        cudaMalloc(&lay.qkv_weight_fp8, (size_t)3 * g_cfg.C * g_cfg.C * sizeof(__nv_fp8_e4m3));
    }
    launch_quantize_bf16_to_fp8(lay.qkv_weight, lay.qkv_weight_fp8, 64.0f, 3 * g_cfg.C * g_cfg.C, stream);
    
    g_ws.opt_tables_synced = false;
}

// ---------------------------------------------------------------------------
// Step Execution & Loss Retrieval
// ---------------------------------------------------------------------------
void set_inputs(torch::Tensor input_ids, torch::Tensor targets) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(g_ws.input_ids, input_ids.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_ws.targets, targets.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    // Mirror to both microsteps for interleaved execution
    cudaMemcpyAsync(g_ws.input_ids_ms[0], input_ids.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_ws.input_ids_ms[1], input_ids.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_ws.targets_ms[0], targets.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_ws.targets_ms[1], targets.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
}

void set_inputs_interleaved(torch::Tensor input_ids_0, torch::Tensor targets_0, torch::Tensor input_ids_1, torch::Tensor targets_1) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(g_ws.input_ids_ms[0], input_ids_0.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_ws.targets_ms[0], targets_0.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_ws.input_ids_ms[1], input_ids_1.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_ws.targets_ms[1], targets_1.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    // Alias to legacy pointer
    cudaMemcpyAsync(g_ws.input_ids, input_ids_0.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_ws.targets, targets_0.data_ptr<int32_t>(), g_cfg.B * g_cfg.T * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
}

void train_step(float lr) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(g_ws.loss_buffer, 0, sizeof(float), stream);
    run_native_training_step(g_cfg, g_ws, g_params, lr, stream);
}

void train_step_interleaved(float lr) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(g_ws.loss_buffer, 0, sizeof(float), stream);
    run_native_training_step_interleaved(g_cfg, g_ws, g_params, lr, stream);
}

float get_loss() {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    float host_loss = 0.0f;
    cudaMemcpyAsync(&host_loss, g_ws.loss_buffer, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return host_loss;
}

float train_step_eager(float lr) {
    train_step(lr);
    return get_loss();
}

float train_step_interleaved_eager(float lr) {
    train_step_interleaved(lr);
    return get_loss();
}

void capture_full_graph(float lr) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    if (g_graph_ctx.is_captured) {
        destroy_graph(g_graph_ctx);
    }
    
    // Warmup prior to capture
    for (int i = 0; i < 3; ++i) {
        train_step(lr);
    }
    cudaStreamSynchronize(stream);
    
    cudaGraphCreate(&g_graph_ctx.graph, 0);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    train_step(lr);
    
    cudaStreamEndCapture(stream, &g_graph_ctx.graph);
    cudaGraphInstantiate(&g_graph_ctx.instance, g_graph_ctx.graph, NULL, NULL, 0);
    g_graph_ctx.is_captured = true;
}

void replay_graph() {
    TORCH_CHECK(g_graph_ctx.is_captured, "CUDA Graph not captured. Call capture_full_graph first.");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    replay_graph_step(g_graph_ctx, stream);
}

float train_step_graph() {
    replay_graph();
    return get_loss();
}

// Component 3: Isolated Attention Forward Test Binding
torch::Tensor test_attention_forward(torch::Tensor layer_qkv, int layer_idx) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < g_cfg.num_layers, "Invalid layer index");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto& lay = g_params.layers[layer_idx];
    
    cudaMemcpyAsync(g_ws.layer_qkv, layer_qkv.data_ptr(), g_cfg.M() * 3 * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    
    const size_t attn_state_elements = (size_t)g_cfg.B * g_cfg.H * g_cfg.D * g_cfg.D;
    cudaMemsetAsync(g_ws.layer_attn_state[layer_idx], 0, attn_state_elements * sizeof(__nv_bfloat16), stream);
    
    run_native_associative_attention_forward(g_ws, lay, layer_idx, g_cfg, stream);
    cudaStreamSynchronize(stream);
    
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    torch::Tensor out = torch::empty({g_cfg.M(), g_cfg.C}, opts);
    cudaMemcpy(out.data_ptr(), g_ws.layer_attn_out, g_cfg.M() * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    return out;
}

// Component 3: Isolated Attention Backward Test Binding
std::vector<torch::Tensor> test_attention_backward(torch::Tensor d_attn_out, int layer_idx) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < g_cfg.num_layers, "Invalid layer index");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    auto& lay = g_params.layers[layer_idx];
    const __nv_bfloat16* d_out_ptr = reinterpret_cast<const __nv_bfloat16*>(d_attn_out.data_ptr<at::BFloat16>());
    
    run_native_associative_attention_backward(g_ws, lay, layer_idx, g_cfg, d_out_ptr, 0.0f, stream);
    cudaStreamSynchronize(stream);
    
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    
    // Copy d_layer_qkv (2048, 3072)
    torch::Tensor d_qkv = torch::empty({g_cfg.M(), 3 * g_cfg.C}, opts);
    cudaMemcpyAsync(d_qkv.data_ptr(), g_ws.d_layer_qkv, g_cfg.M() * 3 * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    
    // dW_out (C, C)
    torch::Tensor dW_out = torch::empty({g_cfg.C, g_cfg.C}, opts);
    cudaMemcpyAsync(dW_out.data_ptr(), lay.d_out_proj_weight, g_cfg.C * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    
    // dS_all (512, 64, 64)
    torch::Tensor dS_all = torch::empty({g_cfg.B, g_cfg.H, g_cfg.T / g_cfg.chunk_size, g_cfg.D, g_cfg.D}, opts);
    cudaMemcpyAsync(dS_all.data_ptr(), g_ws.attn_ds_all, g_cfg.B * g_cfg.H * (g_cfg.T / g_cfg.chunk_size) * g_cfg.D * g_cfg.D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    
    // dDeltaS (512, 64, 64)
    torch::Tensor dDeltaS = torch::empty({g_cfg.B, g_cfg.H, g_cfg.T / g_cfg.chunk_size, g_cfg.D, g_cfg.D}, opts);
    cudaMemcpyAsync(dDeltaS.data_ptr(), g_ws.attn_d_delta_s, g_cfg.B * g_cfg.H * (g_cfg.T / g_cfg.chunk_size) * g_cfg.D * g_cfg.D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    
    cudaStreamSynchronize(stream);
    return {d_qkv, dW_out, dS_all, dDeltaS};
}

std::vector<torch::Tensor> get_layer_gradients(int layer_idx) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < g_cfg.num_layers, "Invalid layer index");
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto& lay = g_params.layers[layer_idx];
    
    torch::Tensor d_qkv = torch::empty({3 * g_cfg.C, g_cfg.C}, opts);
    cudaMemcpy(d_qkv.data_ptr(), lay.d_qkv_weight, 3 * g_cfg.C * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    
    torch::Tensor d_out = torch::empty({g_cfg.C, g_cfg.C}, opts);
    cudaMemcpy(d_out.data_ptr(), lay.d_out_proj_weight, g_cfg.C * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    
    torch::Tensor d_norm1 = torch::empty({g_cfg.C}, opts);
    cudaMemcpy(d_norm1.data_ptr(), lay.d_norm1_weight, g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    
    return {d_qkv, d_out, d_norm1};
}

std::vector<torch::Tensor> get_all_layer_gradients(int layer_idx) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < g_cfg.num_layers, "Invalid layer index");
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto& lay = g_params.layers[layer_idx];
    
    std::vector<torch::Tensor> grads;
    // 0: d_qkv (3 * C, C)
    torch::Tensor d_qkv = torch::empty({3 * g_cfg.C, g_cfg.C}, opts);
    cudaMemcpy(d_qkv.data_ptr(), lay.d_qkv_weight, 3 * g_cfg.C * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    grads.push_back(d_qkv);
    
    // 1: d_out (C, C)
    torch::Tensor d_out = torch::empty({g_cfg.C, g_cfg.C}, opts);
    cudaMemcpy(d_out.data_ptr(), lay.d_out_proj_weight, g_cfg.C * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    grads.push_back(d_out);
    
    // 2: d_norm1 (C)
    torch::Tensor d_norm1 = torch::empty({g_cfg.C}, opts);
    cudaMemcpy(d_norm1.data_ptr(), lay.d_norm1_weight, g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    grads.push_back(d_norm1);
    
    // 3: d_norm2 (C)
    torch::Tensor d_norm2 = torch::empty({g_cfg.C}, opts);
    cudaMemcpy(d_norm2.data_ptr(), lay.d_norm2_weight, g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    grads.push_back(d_norm2);
    
    // 4: d_router (E, C)
    torch::Tensor d_router = torch::empty({g_cfg.E, g_cfg.C}, opts);
    cudaMemcpy(d_router.data_ptr(), lay.d_router_weight, g_cfg.E * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    grads.push_back(d_router);
    
    // 5..8: d_w1_0..3 (hidden_dim, C)
    for (int e = 0; e < g_cfg.E; ++e) {
        torch::Tensor dw1 = torch::empty({g_cfg.hidden_dim, g_cfg.C}, opts);
        cudaMemcpy(dw1.data_ptr(), lay.d_w1_weights[e], g_cfg.hidden_dim * g_cfg.C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        grads.push_back(dw1);
    }
    
    // 9..12: d_w2_0..3 (C, hidden_dim)
    for (int e = 0; e < g_cfg.E; ++e) {
        torch::Tensor dw2 = torch::empty({g_cfg.C, g_cfg.hidden_dim}, opts);
        cudaMemcpy(dw2.data_ptr(), lay.d_w2_weights[e], g_cfg.C * g_cfg.hidden_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        grads.push_back(dw2);
    }
    
    return grads;
}

// Component 4: Isolated MoE Forward Test Binding
std::vector<torch::Tensor> test_moe_forward(torch::Tensor x1, int layer_idx, float noise_std) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < g_cfg.num_layers, "Invalid layer index");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto& lay = g_params.layers[layer_idx];
    int M = g_cfg.M();
    int C = g_cfg.C;
    int top_k = g_cfg.top_k;
    int hidden_dim = g_cfg.hidden_dim;
    int E = g_cfg.E;
    
    cudaMemcpyAsync(g_ws.layer_x1, x1.data_ptr(), (size_t)M * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    launch_fused_rmsnorm_fwd(g_ws.layer_x1, lay.norm2_weight, g_ws.layer_x_norm2, g_ws.layer_rsqrt2, M, C, 1e-6f, stream);
    
    cublaslt_gemm_router_fwd(g_ws.layer_x_norm2, lay.router_weight, g_ws.layer_router_logits, M, C, E, stream);
    launch_moe_top2_gating(g_ws.layer_router_logits, g_ws.layer_topk_gates, g_ws.layer_topk_idx, g_ws.l_bal_total, M, E, noise_std, noise_std > 0.0f, stream);
    launch_moe_compute_maps(g_ws.layer_topk_idx, g_ws.layer_scatter_map, g_ws.layer_gather_map, g_ws.layer_gate_idx_map, g_ws.layer_expert_offsets, M, E, stream);
    launch_moe_dispatch_gather(g_ws.layer_x_norm2, g_ws.layer_gather_map, g_ws.layer_dispatched_x, M * top_k, C, stream);
    launch_moe_grouped_gemm_fwd_w1(g_ws.layer_dispatched_x, lay.w1_weights, g_ws.layer_expert_offsets, g_ws.layer_h1, M * top_k, C, hidden_dim, E, stream);
    launch_fused_gelu_fwd(g_ws.layer_h1, g_ws.layer_act, M * top_k * hidden_dim, stream);
    launch_moe_grouped_gemm_fwd_w2(g_ws.layer_act, lay.w2_weights, g_ws.layer_expert_offsets, g_ws.layer_dispatched_y, M * top_k, hidden_dim, C, E, stream);
    launch_moe_scatter_combine_add_residual(g_ws.layer_dispatched_y, g_ws.layer_topk_gates, g_ws.layer_scatter_map, g_ws.layer_x1, g_ws.layer_moe_out, M, C, stream);
    cudaStreamSynchronize(stream);
    
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    
    torch::Tensor out = torch::empty({M, C}, opts_bf16);
    cudaMemcpy(out.data_ptr(), g_ws.layer_moe_out, (size_t)M * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    
    torch::Tensor gates = torch::empty({M, top_k}, opts_f32);
    cudaMemcpy(gates.data_ptr(), g_ws.layer_topk_gates, (size_t)M * top_k * sizeof(float), cudaMemcpyDeviceToDevice);
    
    torch::Tensor top_idx = torch::empty({M, top_k}, opts_i32);
    cudaMemcpy(top_idx.data_ptr(), g_ws.layer_topk_idx, (size_t)M * top_k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    
    torch::Tensor offsets = torch::empty({E + 1}, opts_i32);
    cudaMemcpy(offsets.data_ptr(), g_ws.layer_expert_offsets, (size_t)(E + 1) * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    
    torch::Tensor logits = torch::empty({M, E}, opts_bf16);
    cudaMemcpy(logits.data_ptr(), g_ws.layer_router_logits, (size_t)M * E * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    
    torch::Tensor x_norm2 = torch::empty({M, C}, opts_bf16);
    cudaMemcpy(x_norm2.data_ptr(), g_ws.layer_x_norm2, (size_t)M * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    
    return {out, gates, top_idx, offsets, logits, x_norm2};
}

// Component 4: Isolated MoE Backward Test Binding
std::vector<torch::Tensor> test_moe_backward(torch::Tensor grad_out, int layer_idx) {
    TORCH_CHECK(g_engine_initialized, "Engine not initialized");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < g_cfg.num_layers, "Invalid layer index");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto& lay = g_params.layers[layer_idx];
    int M = g_cfg.M();
    int C = g_cfg.C;
    int top_k = g_cfg.top_k;
    int hidden_dim = g_cfg.hidden_dim;
    int E = g_cfg.E;
    
    // Zero gradients before backward for clean measurement
    cudaMemsetAsync(lay.d_norm2_weight, 0, C * sizeof(__nv_bfloat16), stream);
    cudaMemsetAsync(lay.d_router_weight, 0, E * C * sizeof(__nv_bfloat16), stream);
    for (int e = 0; e < E; ++e) {
        cudaMemsetAsync(lay.d_w1_weights[e], 0, hidden_dim * C * sizeof(__nv_bfloat16), stream);
        cudaMemsetAsync(lay.d_w2_weights[e], 0, C * hidden_dim * sizeof(__nv_bfloat16), stream);
    }
    
    const __nv_bfloat16* grad_out_ptr = reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>());
    
    // 1. Scatter backward: grad_out -> d_dispatched_y, d_topk_gates
    launch_moe_scatter_backward(
        grad_out_ptr, g_ws.layer_dispatched_y, g_ws.layer_topk_gates,
        g_ws.layer_gather_map, g_ws.layer_gate_idx_map, g_ws.layer_scatter_map,
        g_ws.d_dispatched_y, g_ws.d_topk_gates,
        M, M * top_k, C, stream
    );
    
    // 2. Grouped W2 backward: d_dispatched_y, act -> dW2, d_act
    launch_moe_grouped_gemm_w2_bwd(
        g_ws.d_dispatched_y, g_ws.layer_act, lay.w2_weights,
        g_ws.layer_expert_offsets, lay.d_w2_weights, g_ws.d_act,
        M * top_k, hidden_dim, C, E, 0.0f, stream
    );
    
    // 3. Fused GELU backward: d_act, h1 -> d_h1
    launch_fused_gelu_bwd(g_ws.d_act, g_ws.layer_h1, g_ws.d_h1, M * top_k * hidden_dim, stream);
    
    // 4. Grouped W1 backward: d_h1, dispatched_x -> dW1, d_dispatched_x
    launch_moe_grouped_gemm_w1_bwd(
        g_ws.d_h1, g_ws.layer_dispatched_x, lay.w1_weights,
        g_ws.layer_expert_offsets, lay.d_w1_weights, g_ws.d_dispatched_x,
        M * top_k, C, hidden_dim, E, 0.0f, stream
    );
    
    // 5. Gather backward: d_dispatched_x -> dx_norm2_expert
    launch_moe_gather_backward(
        g_ws.d_dispatched_x, g_ws.layer_scatter_map, g_ws.dx_norm2_expert,
        M, top_k, C, stream
    );
    
    // 6. Router backward: d_topk_gates -> d_router_logits
    launch_moe_router_backward(
        g_ws.d_topk_gates, g_ws.layer_router_logits, g_ws.layer_topk_idx,
        g_ws.d_router_logits, M, E, stream
    );
    
    // 7. Router GEMMs backward: d_router_logits -> d_router_weight, dx_norm2
    launch_moe_router_gemms_bwd(
        g_ws.d_router_logits, g_ws.layer_x_norm2, lay.router_weight,
        g_ws.dx_norm2_expert, lay.d_router_weight, g_ws.dx_norm2,
        M, C, E, 0.0f, stream
    );
    
    // 8. RMSNorm 2 backward: dx_norm2 -> dx_norm2_in, d_norm2_weight
    launch_fused_rmsnorm_bwd(
        g_ws.dx_norm2, g_ws.layer_x1, lay.norm2_weight,
        g_ws.layer_rsqrt2, g_ws.dx_norm2_in, lay.d_norm2_weight, M, C, stream
    );
    
    // 9. Residual 2 addition: cur_dx + dx_norm2_in -> dx1
    launch_fused_add_residual(grad_out_ptr, g_ws.dx_norm2_in, g_ws.dx1, M * C, stream);
    
    cudaStreamSynchronize(stream);
    
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    
    std::vector<torch::Tensor> res;
    // 0..3: expert dW1 [0..3]
    for (int e = 0; e < E; ++e) {
        torch::Tensor dw1 = torch::empty({hidden_dim, C}, opts_bf16);
        cudaMemcpy(dw1.data_ptr(), lay.d_w1_weights[e], (size_t)hidden_dim * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        res.push_back(dw1);
    }
    // 4..7: expert dW2 [0..3]
    for (int e = 0; e < E; ++e) {
        torch::Tensor dw2 = torch::empty({C, hidden_dim}, opts_bf16);
        cudaMemcpy(dw2.data_ptr(), lay.d_w2_weights[e], (size_t)C * hidden_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        res.push_back(dw2);
    }
    // 8: d_router_weight (E, C)
    torch::Tensor d_rw = torch::empty({E, C}, opts_bf16);
    cudaMemcpy(d_rw.data_ptr(), lay.d_router_weight, (size_t)E * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    res.push_back(d_rw);
    
    // 9: dx_norm2 (M, C)
    torch::Tensor dx_n2 = torch::empty({M, C}, opts_bf16);
    cudaMemcpy(dx_n2.data_ptr(), g_ws.dx_norm2, (size_t)M * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    res.push_back(dx_n2);
    
    // 10: dx_norm2_in (M, C)
    torch::Tensor dx_n2_in = torch::empty({M, C}, opts_bf16);
    cudaMemcpy(dx_n2_in.data_ptr(), g_ws.dx_norm2_in, (size_t)M * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    res.push_back(dx_n2_in);
    
    // 11: dx1 (M, C)
    torch::Tensor dx1 = torch::empty({M, C}, opts_bf16);
    cudaMemcpy(dx1.data_ptr(), g_ws.dx1, (size_t)M * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    res.push_back(dx1);
    
    // 12: d_router_logits (M, E)
    torch::Tensor d_rl = torch::empty({M, E}, opts_bf16);
    cudaMemcpy(d_rl.data_ptr(), g_ws.d_router_logits, (size_t)M * E * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    res.push_back(d_rl);
    
    // 13: d_topk_gates (M, top_k)
    torch::Tensor d_tg = torch::empty({M, top_k}, opts_f32);
    cudaMemcpy(d_tg.data_ptr(), g_ws.d_topk_gates, (size_t)M * top_k * sizeof(float), cudaMemcpyDeviceToDevice);
    res.push_back(d_tg);
    
    // 14: d_norm2_weight (C)
    torch::Tensor d_n2_w = torch::empty({C}, opts_bf16);
    cudaMemcpy(d_n2_w.data_ptr(), lay.d_norm2_weight, (size_t)C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    res.push_back(d_n2_w);
    
    return res;
}

// ---------------------------------------------------------------------------
// PyBind11 Module Exports
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Phase 16 single-layer exports
    m.def("init_workspace", &init_workspace, "Allocate static layer workspace");
    m.def("cleanup_workspace", &cleanup_workspace, "Free static layer workspace");
    m.def("get_workspace_bytes", &get_workspace_bytes, "Get total allocated workspace bytes");
    m.def("native_fused_forward", &native_fused_forward, "Native fused forward layer");
    m.def("native_fused_backward", &native_fused_backward, "Native fused backward layer");

    // Phase 17 full-model exports
    m.def("init_full_engine", &init_full_engine, "Initialize full 24-layer engine workspace");
    m.def("cleanup_full_engine", &cleanup_full_engine, "Free full engine workspace");
    m.def("get_workspace_memory_bytes", &get_workspace_memory_bytes, "Get static workspace bytes");
    m.def("bind_global_parameters", &bind_global_parameters, "Bind global model parameters");
    m.def("bind_layer_parameters", &bind_layer_parameters, "Bind layer parameters for layer l");
    m.def("set_inputs", &set_inputs, "Set input tokens and targets");
    m.def("set_inputs_interleaved", &set_inputs_interleaved, "Phase 22: Set distinct inputs for microstep 0 and 1");
    m.def("train_step", &train_step, "Execute GPU training step with zero host transfers");
    m.def("train_step_interleaved", &train_step_interleaved, "Phase 22: Execute Layer-Wise Interleaved training step");
    m.def("get_loss", &get_loss, "Retrieve last scalar training loss");
    m.def("train_step_eager", &train_step_eager, "Execute eager training step and return loss");
    m.def("train_step_interleaved_eager", &train_step_interleaved_eager, "Phase 22: Execute eager interleaved step and return loss");
    m.def("capture_full_graph", &capture_full_graph, "Capture complete training step into CUDA Graph");
    m.def("replay_graph", &replay_graph, "Replay captured CUDA Graph (device only)");
    m.def("train_step_graph", &train_step_graph, "Replay captured CUDA Graph and return loss");
    m.def("set_fp8_moe", [](bool enabled) { g_cfg.use_fp8_moe = enabled; }, "Phase 28: Toggle FP8 MoE forward");
    m.def("get_fp8_moe", []() { return g_cfg.use_fp8_moe; }, "Phase 28: Get FP8 MoE toggle state");
    m.def("set_fp8_lm_head", [](bool enabled) { g_cfg.use_fp8_lm_head = enabled; }, "Phase 28: Toggle FP8 LM Head forward");
    m.def("get_fp8_lm_head", []() { return g_cfg.use_fp8_lm_head; }, "Phase 28: Get FP8 LM Head toggle state");
    m.def("set_fp8_lm_head_backward", [](bool enabled) { g_cfg.use_fp8_lm_head_backward = enabled; }, "Phase 28: Toggle FP8 LM Head backward");
    m.def("get_fp8_lm_head_backward", []() { return g_cfg.use_fp8_lm_head_backward; }, "Phase 28: Get FP8 LM Head backward toggle state");
    m.def("set_use_bf16_moments", [](bool enabled) { g_cfg.use_bf16_moments = enabled; }, "Phase 29: Toggle BF16 moments in AdamW");
    m.def("get_use_bf16_moments", []() { return g_cfg.use_bf16_moments; }, "Phase 29: Get BF16 moments toggle state");
    m.def("set_use_fp8_moments", [](bool enabled) { g_cfg.use_fp8_moments = enabled; }, "Phase 31: Toggle FP8 moments in AdamW");
    m.def("get_use_fp8_moments", []() { return g_cfg.use_fp8_moments; }, "Phase 31: Get FP8 moments toggle state");
    m.def("set_fp8_qkv", [](bool enabled) { g_cfg.use_fp8_qkv = enabled; }, "Phase 30: Toggle native FP8 QKV forward execution");
    m.def("get_fp8_qkv", []() { return g_cfg.use_fp8_qkv; }, "Phase 30: Query native FP8 QKV forward status");
    m.def("set_fused_rmsnorm_quant", [](bool enabled) { g_cfg.use_fused_rmsnorm_quant = enabled; }, "Phase 31: Toggle native fused RMSNorm + FP8 quant");
    m.def("get_fused_rmsnorm_quant", []() { return g_cfg.use_fused_rmsnorm_quant; }, "Phase 31: Query native fused RMSNorm + FP8 quant status");
    m.def("set_fp8_qkv_backward", [](bool enabled) { g_cfg.use_fp8_qkv_backward = enabled; }, "Phase 31: Toggle native FP8 QKV backward execution");
    m.def("get_fp8_qkv_backward", []() { return g_cfg.use_fp8_qkv_backward; }, "Phase 31: Query native FP8 QKV backward status");
    m.def("test_attention_backward", &test_attention_backward, "Component 3: Test native attention backward");
    m.def("test_attention_forward", &test_attention_forward, "Component 3: Test native attention forward");
    m.def("get_layer_gradients", &get_layer_gradients, "Component 3: Get layer parameter gradients");
    m.def("get_all_layer_gradients", &get_all_layer_gradients, "Component 4: Get all layer parameter gradients including MoE");
    m.def("test_moe_forward", &test_moe_forward, "Component 4: Test native MoE forward");
    m.def("test_moe_backward", &test_moe_backward, "Component 4: Test native MoE backward");
}

