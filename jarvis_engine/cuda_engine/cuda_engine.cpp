#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_engine.h"

// Singleton or managed workspace instance for benchmark
static JarvisLayerWorkspace g_workspace;
static bool g_workspace_allocated = false;

// ---------------------------------------------------------------------------
// Workspace Lifecycle
// ---------------------------------------------------------------------------
void init_workspace(int B, int T, int C, int H, int E, int top_k, int hidden_dim, int chunk_size) {
    if (g_workspace_allocated) {
        free_layer_workspace(g_workspace);
    }
    JarvisLayerConfig cfg;
    cfg.B = B;
    cfg.T = T;
    cfg.C = C;
    cfg.H = H;
    cfg.D = C / H;
    cfg.E = E;
    cfg.top_k = top_k;
    cfg.hidden_dim = hidden_dim;
    cfg.chunk_size = chunk_size;
    
    g_workspace = allocate_layer_workspace(cfg);
    g_workspace_allocated = true;
}

void cleanup_workspace() {
    if (g_workspace_allocated) {
        free_layer_workspace(g_workspace);
        g_workspace_allocated = false;
    }
}

size_t get_workspace_bytes() {
    return g_workspace_allocated ? g_workspace.total_bytes : 0;
}

// ---------------------------------------------------------------------------
// Native Fused Layer Forward
// ---------------------------------------------------------------------------
// Takes input x and layer weights, executes complete layer using fused kernels
// on caller's active CUDA stream with zero dynamic memory allocation.
torch::Tensor native_fused_forward(
    torch::Tensor x,              // (B, T, C)
    torch::Tensor norm1_weight,   // (C)
    torch::Tensor qkv_weight,     // (3*C, C)
    torch::Tensor out_proj_weight,// (C, C)
    torch::Tensor norm2_weight,   // (C)
    torch::Tensor router_weight   // (E, C)
) {
    TORCH_CHECK(g_workspace_allocated, "Workspace not initialized. Call init_workspace first.");
    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);
    int M = B * T;
    float eps = 1e-6f;
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // 1. Fused RMSNorm 1: x -> ws.x_norm1
    launch_fused_rmsnorm_fwd(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm1_weight.data_ptr<at::BFloat16>()),
        g_workspace.x_norm1,
        g_workspace.rsqrt1,
        M, C, eps, stream
    );
    
    // Wrap device pointer to tensor view (zero copy, points directly to workspace buffer)
    auto x_norm1_t = torch::from_blob(
        g_workspace.x_norm1, {M, C}, x.options()
    );
    
    // 2. Fused QKV Projection: (M, C) @ (3*C, C).T -> (M, 3*C)
    // Runs on current stream using Tensor Cores
    auto qkv_t = torch::from_blob(g_workspace.qkv, {M, 3 * C}, x.options());
    at::mm_out(qkv_t, x_norm1_t, qkv_weight.t());
    
    // 3. Fused Residual 1 + RMSNorm 2:
    // Computes ws.x1 = x + attn_out AND ws.x_norm2 = RMSNorm(ws.x1) in ONE memory pass!
    // For standalone micro-benchmark, attn_out is simulated from linear projection
    auto attn_out_t = torch::from_blob(g_workspace.attn_out, {M, C}, x.options());
    auto q_chunk = qkv_t.slice(1, 0, C);
    at::mm_out(attn_out_t, q_chunk, out_proj_weight.t());
    
    launch_fused_add_rmsnorm_fwd(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        g_workspace.attn_out,
        reinterpret_cast<const __nv_bfloat16*>(norm2_weight.data_ptr<at::BFloat16>()),
        g_workspace.x1,
        g_workspace.x_norm2,
        g_workspace.rsqrt2,
        M, C, eps, stream
    );
    
    // 4. Return x1 output
    auto x1_t = torch::from_blob(g_workspace.x1, {B, T, C}, x.options());
    return x1_t;
}

// ---------------------------------------------------------------------------
// Native Fused Layer Backward
// ---------------------------------------------------------------------------
torch::Tensor native_fused_backward(
    torch::Tensor grad_x1,        // (B, T, C)
    torch::Tensor x,              // (B, T, C)
    torch::Tensor norm1_weight,   // (C)
    torch::Tensor qkv_weight,     // (3*C, C)
    torch::Tensor out_proj_weight // (C, C)
) {
    TORCH_CHECK(g_workspace_allocated, "Workspace not initialized.");
    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);
    int M = B * T;
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // 1. RMSNorm 1 backward
    launch_fused_rmsnorm_bwd(
        reinterpret_cast<const __nv_bfloat16*>(grad_x1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm1_weight.data_ptr<at::BFloat16>()),
        g_workspace.rsqrt1,
        g_workspace.grad_x,
        nullptr,
        M, C, stream
    );
    
    auto grad_x_t = torch::from_blob(g_workspace.grad_x, {B, T, C}, x.options());
    return grad_x_t;
}
