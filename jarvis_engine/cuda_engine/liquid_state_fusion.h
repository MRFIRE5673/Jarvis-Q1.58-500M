#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Liquid State Fusion (LSF) CUDA Engine
// Equations:
//   H_t = \alpha \cdot H_{t-1} + (1 - \alpha) \cdot M_t
//   \alpha = \alpha_{min} + (\alpha_{max} - \alpha_{min}) \cdot \sigma(-var\_scale \cdot act\_var)
// ---------------------------------------------------------------------------

void launch_liquid_state_fusion_fwd(
    const __nv_bfloat16* X,           // (M, C) MoE output
    const float* var_scale,           // (1)
    __nv_bfloat16* H,                 // (M, C) LSF state out
    __nv_bfloat16* H_last,            // (B, C) or nullptr
    const __nv_bfloat16* H0,          // (B, C) or nullptr
    float* alpha_buf,                 // (1) intermediate scalar alpha
    float* mean_var_buf,              // (2) intermediate mean and var
    int B, int T, int C,
    cudaStream_t stream
);

void launch_liquid_state_fusion_bwd(
    const __nv_bfloat16* grad_H,      // (M, C) gradient from residual x2
    const __nv_bfloat16* X,           // (M, C) forward MoE output
    const __nv_bfloat16* H,           // (M, C) forward LSF state
    const __nv_bfloat16* H0,          // (B, C) or nullptr
    const float* var_scale,           // (1)
    const float* mean_var_buf,        // (2) forward mean and var
    const float* alpha_buf,           // (1) forward alpha
    __nv_bfloat16* grad_X,            // (M, C) gradient into MoE scatter backward
    float* d_var_scale,               // (1) gradient accumulator for var_scale
    float* grad_alpha_buf,            // (num_blocks) scratch
    int B, int T, int C,
    float beta,
    cudaStream_t stream
);
