// experiments/throughput_optimization/ternary_cuda/ternary_gemm.h
#pragma once
#include <torch/extension.h>

at::Tensor pack_ternary_weights(
    const at::Tensor& w_fp32,
    double alpha
);

at::Tensor packed_ternary_gemm_forward(
    const at::Tensor& x,            // (M, K) bfloat16
    const at::Tensor& w_packed,     // (N, K/4) uint8
    double alpha,                   // scalar scale
    const c10::optional<at::Tensor>& bias // optional (N,) bfloat16
);

std::vector<at::Tensor> packed_ternary_linear_forward(
    const at::Tensor& x,            // (M, K) bfloat16
    const at::Tensor& w_fp32,       // (N, K) float32/bfloat16 master
    const c10::optional<at::Tensor>& bias
);

at::Tensor packed_ternary_linear_backward(
    const at::Tensor& grad_out,     // (M, N) bfloat16
    const at::Tensor& x,            // (M, K) bfloat16
    const at::Tensor& w_packed,     // (N, K/4) uint8
    const at::Tensor& w_fp32,       // (N, K) float32 master (for STE mask)
    double alpha
);
