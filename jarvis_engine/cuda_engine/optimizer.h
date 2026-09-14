#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "full_engine.h"

// Fused AdamW Optimizer Engine
void launch_zero_grad_norm(float* grad_norm_sq, cudaStream_t stream);

void launch_accumulate_grad_norm_sq(
    const __nv_bfloat16* grad,
    int num_elements,
    float* grad_norm_sq,
    cudaStream_t stream
);

void launch_accumulate_layer_grad_norm_sq(
    const LayerWeights& lay,
    float* grad_norm_sq,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
);

void launch_accumulate_global_grad_norm_sq(
    const FullModelParameters& params,
    float* grad_norm_sq,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
);

void launch_compute_clip_coef(
    const float* grad_norm_sq,
    float* clip_coef,
    float max_norm,
    cudaStream_t stream
);

void launch_fused_adamw_update_bf16(
    __nv_bfloat16* param,
    __nv_bfloat16* grad,
    float* m,
    float* v,
    const float* clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int num_elements,
    cudaStream_t stream,
    float step_size = 0.0f,
    float eps_corrected = 0.0f
);

void launch_fused_adamw_update_bf16_moments(
    __nv_bfloat16* param,
    __nv_bfloat16* grad,
    __nv_bfloat16* m,
    __nv_bfloat16* v,
    const float* clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int num_elements,
    cudaStream_t stream,
    __nv_fp8_e4m3* param_fp8 = nullptr,
    float scale_fp8 = 64.0f,
    float step_size = 0.0f,
    float eps_corrected = 0.0f
);

void launch_fused_adamw_update_fp8_moments(
    __nv_bfloat16* param,
    __nv_bfloat16* grad,
    __nv_fp8_e4m3* m,
    __nv_fp8_e5m2* v,
    const float* clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int num_elements,
    cudaStream_t stream,
    __nv_fp8_e4m3* param_fp8 = nullptr,
    float scale_fp8 = 64.0f,
    float scale_m = 256.0f,
    float scale_v = 512.0f,
    float step_size = 0.0f,
    float eps_corrected = 0.0f
);

void launch_fused_adamw_update_f32(
    float* param,
    float* grad,
    float* m,
    float* v,
    const float* clip_coef,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int num_elements,
    cudaStream_t stream,
    float step_size = 0.0f,
    float eps_corrected = 0.0f
);

void run_fused_optimizer_step(
    FullModelParameters& params,
    FullModelWorkspace& ws,
    const FullJarvisConfig& cfg,
    float lr,
    cudaStream_t stream
);
