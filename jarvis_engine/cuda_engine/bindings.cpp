#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_engine.h"
#include "full_engine.h"
#include "runtime.h"
#include "optimizer.h"
#include "cublaslt_engine.h"

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

// ---------------------------------------------------------------------------
// Engine Lifecycle
// ---------------------------------------------------------------------------
void init_full_engine(
    int B, int T, int C, int H, int E, int top_k, int hidden_dim,
    int chunk_size, int num_layers, int vocab_size, int vocab_pad, int accum_steps
) {
    if (g_engine_initialized) {
        free_full_workspace(g_ws);
        destroy_graph(g_graph_ctx);
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
    
    // Allocate AdamW momentum buffers (m, v in FP32) and gradient buffers
    // Global parameters
    auto alloc_f32 = [](float*& ptr, size_t count) {
        cudaMalloc(&ptr, count * sizeof(float));
        cudaMemset(ptr, 0, count * sizeof(float));
    };
    auto alloc_bf16 = [](__nv_bfloat16*& ptr, size_t count) {
        cudaMalloc(&ptr, count * sizeof(__nv_bfloat16));
        cudaMemset(ptr, 0, count * sizeof(__nv_bfloat16));
    };
    
    alloc_bf16(g_params.d_tok_emb_weight, g_cfg.vocab_size * g_cfg.C);
    alloc_f32(g_params.m_tok_emb, g_cfg.vocab_size * g_cfg.C);
    alloc_f32(g_params.v_tok_emb, g_cfg.vocab_size * g_cfg.C);
    
    alloc_bf16(g_params.d_final_norm_weight, g_cfg.C);
    alloc_f32(g_params.m_final_norm, g_cfg.C);
    alloc_f32(g_params.v_final_norm, g_cfg.C);
    
    alloc_bf16(g_params.d_lm_head_weight, g_cfg.vocab_pad * g_cfg.C);
    alloc_f32(g_params.m_lm_head, g_cfg.vocab_pad * g_cfg.C);
    alloc_f32(g_params.v_lm_head, g_cfg.vocab_pad * g_cfg.C);
    
    for (int l = 0; l < g_cfg.num_layers; ++l) {
        auto& lay = g_params.layers[l];
        alloc_bf16(lay.d_norm1_weight, g_cfg.C);
        alloc_f32(lay.m_norm1, g_cfg.C);
        alloc_f32(lay.v_norm1, g_cfg.C);
        
        alloc_bf16(lay.d_qkv_weight, 3 * g_cfg.C * g_cfg.C);
        alloc_f32(lay.m_qkv, 3 * g_cfg.C * g_cfg.C);
        alloc_f32(lay.v_qkv, 3 * g_cfg.C * g_cfg.C);
        
        alloc_bf16(lay.d_out_proj_weight, g_cfg.C * g_cfg.C);
        alloc_f32(lay.m_out, g_cfg.C * g_cfg.C);
        alloc_f32(lay.v_out, g_cfg.C * g_cfg.C);
        
        alloc_bf16(lay.d_norm2_weight, g_cfg.C);
        alloc_f32(lay.m_norm2, g_cfg.C);
        alloc_f32(lay.v_norm2, g_cfg.C);
        
        alloc_bf16(lay.d_router_weight, g_cfg.E * g_cfg.C);
        alloc_f32(lay.m_router, g_cfg.E * g_cfg.C);
        alloc_f32(lay.v_router, g_cfg.E * g_cfg.C);
        
        for (int e = 0; e < g_cfg.E; ++e) {
            alloc_bf16(lay.d_w1_weights[e], g_cfg.hidden_dim * g_cfg.C);
            alloc_f32(lay.m_w1[e], g_cfg.hidden_dim * g_cfg.C);
            alloc_f32(lay.v_w1[e], g_cfg.hidden_dim * g_cfg.C);
            
            alloc_bf16(lay.d_w2_weights[e], g_cfg.C * g_cfg.hidden_dim);
            alloc_f32(lay.m_w2[e], g_cfg.C * g_cfg.hidden_dim);
            alloc_f32(lay.v_w2[e], g_cfg.C * g_cfg.hidden_dim);
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
}

