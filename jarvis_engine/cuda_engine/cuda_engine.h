#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <torch/extension.h>

// Layer dimensions for Jarvis-Q1.58-500M
struct JarvisLayerConfig {
    int B;          // Batch size (e.g. 8)
    int T;          // Sequence length (e.g. 512)
    int C;          // Hidden dimension d_model (1024)
    int H;          // Number of heads (16)
    int D;          // Head dimension (64)
    int E;          // Number of experts (4)
    int top_k;      // Top-K experts (2)
    int hidden_dim; // MoE hidden dimension (2048)
    int chunk_size; // Attention chunk size (64)
    
    int M() const { return B * T; } // Total tokens (4096)
};

// Static workspace containing all pre-allocated device buffers for ONE Jarvis layer
struct JarvisLayerWorkspace {
    // 1. Forward intermediate activations
    __nv_bfloat16* x_norm1;         // (M, C)
    float*         rsqrt1;          // (M)
    __nv_bfloat16* qkv;             // (M, 3 * C)
    __nv_bfloat16* q_rot;           // (B, H, T, D)
    __nv_bfloat16* k_rot;           // (B, H, T, D)
    __nv_bfloat16* attn_out;        // (M, C)
    __nv_bfloat16* x1;              // (M, C) residual 1
    __nv_bfloat16* x_norm2;         // (M, C)
    float*         rsqrt2;          // (M)
    float*         router_logits;   // (M, E)
    float*         topk_gates;      // (M, top_k)
    int32_t*       topk_idx;        // (M, top_k)
    int32_t*       expert_counts;   // (E)
    int32_t*       expert_offsets;  // (E + 1)
    int32_t*       scatter_map;     // (M * top_k)
    int32_t*       gather_map;      // (M * top_k)
    int32_t*       gate_idx_map;    // (M * top_k)
    __nv_bfloat16* dispatched_x;    // (M * top_k, C)
    __nv_bfloat16* h1;              // (M * top_k, hidden_dim)
    __nv_bfloat16* act;             // (M * top_k, hidden_dim)
    __nv_bfloat16* dispatched_y;    // (M * top_k, C)
    __nv_bfloat16* moe_out;         // (M, C)
    __nv_bfloat16* h_out;           // (M, C) liquid state
    __nv_bfloat16* h_last;          // (B, C) last membrane
    __nv_bfloat16* x2;              // (M, C) final layer output
    
    // 2. Backward gradients
    __nv_bfloat16* grad_x2;         // (M, C)
    __nv_bfloat16* grad_h_out;      // (M, C)
    __nv_bfloat16* grad_moe_out;    // (M, C)
    __nv_bfloat16* grad_dispatched_y;//(M * top_k, C)
    float*         grad_topk_gates; // (M, top_k)
    __nv_bfloat16* grad_act;        // (M * top_k, hidden_dim)
    __nv_bfloat16* grad_h1;         // (M * top_k, hidden_dim)
    __nv_bfloat16* grad_dispatched_x;//(M * top_k, C)
    __nv_bfloat16* grad_x_norm2;    // (M, C)
    __nv_bfloat16* grad_x1;         // (M, C)
    __nv_bfloat16* grad_attn_out;   // (M, C)
    __nv_bfloat16* grad_x_norm1;    // (M, C)
    __nv_bfloat16* grad_x;          // (M, C) input gradient dX
    
    // Total workspace size in bytes
    size_t total_bytes;
    bool is_initialized;
};

// Function prototypes
JarvisLayerWorkspace allocate_layer_workspace(const JarvisLayerConfig& cfg);
void free_layer_workspace(JarvisLayerWorkspace& ws);

// Native fused kernels
void launch_fused_add_rmsnorm_fwd(
    const __nv_bfloat16* x,
    const __nv_bfloat16* res,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out_add,
    __nv_bfloat16* out_norm,
    float* rsqrt,
    int M, int C, float eps,
    cudaStream_t stream,
    __nv_fp8_e4m3* out_norm_fp8 = nullptr,
    float fp8_scale = 1.0f
);

void launch_fused_rmsnorm_fwd(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out_norm,
    float* rsqrt,
    int M, int C, float eps,
    cudaStream_t stream,
    __nv_fp8_e4m3* out_norm_fp8 = nullptr,
    float fp8_scale = 1.0f
);

void launch_fused_rmsnorm_bwd(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const float* rsqrt,
    __nv_bfloat16* grad_x,
    __nv_bfloat16* grad_weight,
    int M, int C,
    cudaStream_t stream,
    __nv_fp8_e4m3* grad_x_fp8 = nullptr,
    float scale_fp8 = 1.0f
);

void launch_fused_gelu_fwd(
    const __nv_bfloat16* in,
    __nv_bfloat16* out,
    int num_elements,
    cudaStream_t stream
);

void launch_fused_gelu_bwd(
    const __nv_bfloat16* grad_out,
    const __nv_bfloat16* in,
    __nv_bfloat16* grad_in,
    int num_elements,
    cudaStream_t stream
);

// Single-layer C++ API prototypes
void init_workspace(int B, int T, int C, int H, int E, int top_k, int hidden_dim, int chunk_size);
void cleanup_workspace();
size_t get_workspace_bytes();
torch::Tensor native_fused_forward(
    torch::Tensor x,
    torch::Tensor norm1_weight,
    torch::Tensor qkv_weight,
    torch::Tensor out_proj_weight,
    torch::Tensor norm2_weight,
    torch::Tensor router_weight
);
torch::Tensor native_fused_backward(
    torch::Tensor grad_x1,
    torch::Tensor x,
    torch::Tensor norm1_weight,
    torch::Tensor qkv_weight,
    torch::Tensor out_proj_weight
);
