// sparse_model_cpp.cpp
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

// Declarations of CUDA functions implemented in sparse_model.cu
std::vector<torch::Tensor> moe_compute_metadata_cuda(
    torch::Tensor topk_idx, int num_experts
);

torch::Tensor moe_dispatch_gather_cuda(
    torch::Tensor x, torch::Tensor gather_map
);

torch::Tensor moe_scatter_combine_cuda(
    torch::Tensor dispatched_y, torch::Tensor topk_gates, torch::Tensor scatter_map
);

std::vector<torch::Tensor> moe_scatter_backward_cuda(
    torch::Tensor grad_out, torch::Tensor dispatched_y, torch::Tensor topk_gates,
    torch::Tensor gather_map, torch::Tensor gate_idx_map, torch::Tensor scatter_map
);

torch::Tensor moe_gather_backward_cuda(
    torch::Tensor grad_dispatched_x, torch::Tensor scatter_map, int N
);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT_OR_BF16(x) AT_ASSERTM(x.dtype() == torch::kFloat32 || x.dtype() == torch::kBFloat16, #x " must be Float32 or BFloat16")

std::vector<torch::Tensor> compute_metadata(torch::Tensor topk_idx, int num_experts) {
    CHECK_CUDA(topk_idx);
    CHECK_CONTIGUOUS(topk_idx);
    AT_ASSERTM(topk_idx.dtype() == torch::kInt64, "topk_idx must be int64");
    AT_ASSERTM(topk_idx.dim() == 2, "topk_idx must be (N, K)");
    return moe_compute_metadata_cuda(topk_idx, num_experts);
}

torch::Tensor dispatch_gather(torch::Tensor x, torch::Tensor gather_map) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT_OR_BF16(x);
    CHECK_CUDA(gather_map);
    CHECK_CONTIGUOUS(gather_map);
    AT_ASSERTM(gather_map.dtype() == torch::kInt32, "gather_map must be int32");
    AT_ASSERTM(x.dim() == 2, "x must be (N, C)");
    return moe_dispatch_gather_cuda(x, gather_map);
}

torch::Tensor scatter_combine(torch::Tensor dispatched_y, torch::Tensor topk_gates, torch::Tensor scatter_map) {
    CHECK_CUDA(dispatched_y);
    CHECK_CONTIGUOUS(dispatched_y);
    CHECK_FLOAT_OR_BF16(dispatched_y);
    CHECK_CUDA(topk_gates);
    CHECK_CONTIGUOUS(topk_gates);
    CHECK_CUDA(scatter_map);
    CHECK_CONTIGUOUS(scatter_map);
    AT_ASSERTM(topk_gates.dtype() == dispatched_y.dtype(), "topk_gates and dispatched_y must have identical dtype");
    AT_ASSERTM(scatter_map.dtype() == torch::kInt32, "scatter_map must be int32");
    return moe_scatter_combine_cuda(dispatched_y, topk_gates, scatter_map);
}

std::vector<torch::Tensor> scatter_backward(
    torch::Tensor grad_out, torch::Tensor dispatched_y, torch::Tensor topk_gates,
    torch::Tensor gather_map, torch::Tensor gate_idx_map, torch::Tensor scatter_map
) {
    CHECK_CUDA(grad_out); CHECK_CONTIGUOUS(grad_out); CHECK_FLOAT_OR_BF16(grad_out);
    CHECK_CUDA(dispatched_y); CHECK_CONTIGUOUS(dispatched_y);
    CHECK_CUDA(topk_gates); CHECK_CONTIGUOUS(topk_gates);
    CHECK_CUDA(gather_map); CHECK_CONTIGUOUS(gather_map);
    CHECK_CUDA(gate_idx_map); CHECK_CONTIGUOUS(gate_idx_map);
    CHECK_CUDA(scatter_map); CHECK_CONTIGUOUS(scatter_map);
    return moe_scatter_backward_cuda(grad_out, dispatched_y, topk_gates, gather_map, gate_idx_map, scatter_map);
}

torch::Tensor gather_backward(torch::Tensor grad_dispatched_x, torch::Tensor scatter_map, int N) {
    CHECK_CUDA(grad_dispatched_x); CHECK_CONTIGUOUS(grad_dispatched_x); CHECK_FLOAT_OR_BF16(grad_dispatched_x);
    CHECK_CUDA(scatter_map); CHECK_CONTIGUOUS(scatter_map);
    return moe_gather_backward_cuda(grad_dispatched_x, scatter_map, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-Performance Sparse MoE Dispatch, Gather, and Scatter Extension for Jarvis sm_120";
    m.def("compute_metadata", &compute_metadata, "Compute expert counts, prefix offsets, scatter, and gather maps");
    m.def("dispatch_gather", &dispatch_gather, "Permute and gather input tokens into contiguous expert partitions");
    m.def("scatter_combine", &scatter_combine, "Scatter expert outputs and accumulate with gate weights");
    m.def("scatter_backward", &scatter_backward, "Backward pass through scatter and gating combination");
    m.def("gather_backward", &gather_backward, "Backward pass through dispatch gather permutation");
}
