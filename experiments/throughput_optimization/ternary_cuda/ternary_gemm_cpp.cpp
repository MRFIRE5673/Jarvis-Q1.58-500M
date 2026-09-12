#include <torch/extension.h>
#include "ternary_gemm.h"

// ---------------------------------------------------------------------------
// Combined Forward: compute alpha, pack, and run GEMM
// ---------------------------------------------------------------------------
std::vector<at::Tensor> packed_ternary_linear_forward(
    const at::Tensor& x,
    const at::Tensor& w_fp32,
    const c10::optional<at::Tensor>& bias
) {
    // 1. Compute AbsMean scale alpha
    at::Tensor alpha_t = w_fp32.abs().mean().clamp_min(1e-8);
    double alpha = alpha_t.item<double>();

    // 2. Pack weights directly on CUDA (4 weights/byte)
    at::Tensor w_packed = pack_ternary_weights(w_fp32, alpha);

    // 3. Flatten x if 3D (B, T, K) -> (B*T, K)
    at::Tensor x_2d = x;
    bool is_3d = (x.dim() == 3);
    if (is_3d) {
        x_2d = x.reshape({-1, x.size(-1)});
    }

    // 4. Run Packed Ternary GEMM
    at::Tensor y_2d = packed_ternary_gemm_forward(x_2d, w_packed, alpha, bias);

    at::Tensor y = y_2d;
    if (is_3d) {
        y = y_2d.reshape({x.size(0), x.size(1), -1});
    }

    return {y, w_packed, alpha_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_ternary_weights", &pack_ternary_weights, "Pack ternary weights into uint8");
    m.def("packed_ternary_gemm_forward", &packed_ternary_gemm_forward, "Packed ternary GEMM forward");
    m.def("packed_ternary_linear_forward", &packed_ternary_linear_forward, "Fused packed ternary linear forward");
}
