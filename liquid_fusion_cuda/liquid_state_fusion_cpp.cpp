// liquid_state_fusion_cpp.cpp
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

torch::Tensor liquid_state_fusion_forward_cuda(torch::Tensor alpha, torch::Tensor M, torch::Tensor h0, int threads_per_block);
std::vector<torch::Tensor> liquid_state_fusion_backward_cuda(
    torch::Tensor alpha, torch::Tensor M, torch::Tensor h0, torch::Tensor H, torch::Tensor grad_H, int threads_per_block
);
py::dict get_device_occupancy_cuda(int block_size);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x) AT_ASSERTM(x.dtype() == torch::kFloat32 || x.dtype() == torch::kBFloat16, #x " must be Float32 or BFloat16")

torch::Tensor liquid_state_fusion_forward(torch::Tensor alpha, torch::Tensor M, torch::Tensor h0, int threads_per_block = 64) {
    CHECK_CUDA(alpha);
    CHECK_CUDA(M);
    if (h0.defined()) {
        CHECK_CUDA(h0);
        if (alpha.device() != h0.device())
            throw std::runtime_error("Device mismatch: alpha and h0 must be on the same GPU device.");
        CHECK_DTYPE(h0);
        CHECK_CONTIGUOUS(h0);
        if (alpha.dtype() != h0.dtype())
            throw std::runtime_error("Dtype mismatch: alpha and h0 must have the same dtype.");
    }
    if (alpha.device() != M.device())
        throw std::runtime_error("Device mismatch: alpha and M must be on the same GPU device.");

    CHECK_DTYPE(alpha);
    CHECK_DTYPE(M);
    if (alpha.dtype() != M.dtype())
        throw std::runtime_error("Dtype mismatch: alpha and M must have the same dtype.");

    CHECK_CONTIGUOUS(alpha);
    CHECK_CONTIGUOUS(M);

    if (alpha.dim() != 3 || M.dim() != 3)
        throw std::runtime_error("Shape mismatch: alpha and M must have 3 dimensions (B, T, D).");
    if (alpha.sizes() != M.sizes())
        throw std::runtime_error("Shape mismatch: alpha and M must have identical dimensions.");
    if (h0.defined()) {
        if (h0.dim() != 2)
            throw std::runtime_error("Shape mismatch: h0 must have 2 dimensions (B, D).");
        if (h0.size(0) != alpha.size(0) || h0.size(1) != alpha.size(2))
            throw std::runtime_error("Shape mismatch: h0 dimensions must match (B, D) from alpha.");
    }
    return liquid_state_fusion_forward_cuda(alpha, M, h0, threads_per_block);
}

std::vector<torch::Tensor> liquid_state_fusion_backward(
    torch::Tensor alpha, torch::Tensor M, torch::Tensor h0, torch::Tensor H, torch::Tensor grad_H, int threads_per_block = 64
) {
    CHECK_CUDA(alpha); CHECK_CUDA(M); CHECK_CUDA(H); CHECK_CUDA(grad_H);
    if (h0.defined()) CHECK_CUDA(h0);

    CHECK_DTYPE(alpha); CHECK_DTYPE(M); CHECK_DTYPE(H); CHECK_DTYPE(grad_H);
    if (h0.defined()) CHECK_DTYPE(h0);

    CHECK_CONTIGUOUS(alpha); CHECK_CONTIGUOUS(M); CHECK_CONTIGUOUS(H); CHECK_CONTIGUOUS(grad_H);
    if (h0.defined()) CHECK_CONTIGUOUS(h0);

    return liquid_state_fusion_backward_cuda(alpha, M, h0, H, grad_H, threads_per_block);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &liquid_state_fusion_forward, "Liquid State Fusion Forward (CUDA FP32/BF16)",
          py::arg("alpha"), py::arg("M"), py::arg("h0") = torch::Tensor(), py::arg("threads_per_block") = 64);
    m.def("backward", &liquid_state_fusion_backward, "Liquid State Fusion Backward (CUDA FP32/BF16)",
          py::arg("alpha"), py::arg("M"), py::arg("h0"), py::arg("H"), py::arg("grad_H"), py::arg("threads_per_block") = 64);
    m.def("get_device_occupancy", &get_device_occupancy_cuda, "Query CUDA hardware occupancy info",
          py::arg("block_size") = 64);
}
