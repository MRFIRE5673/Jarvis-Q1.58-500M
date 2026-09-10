// experiments/architecture_matrix/ternary_packed/packed_kernel_cpp.cpp
#include <torch/extension.h>

torch::Tensor packed_ternary_matmul_cuda(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor alpha
);

torch::Tensor packed_ternary_matmul(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor alpha
) {
    return packed_ternary_matmul_cuda(X, W_packed, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_ternary_matmul", &packed_ternary_matmul, "Packed 1.58-bit Ternary Matmul (CUDA)");
}
