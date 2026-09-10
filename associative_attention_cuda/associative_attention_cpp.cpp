#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA wrappers
std::vector<at::Tensor> fused_rope_elu_forward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos_tab,
    const at::Tensor& sin_tab
);

std::vector<at::Tensor> fused_rope_elu_backward_cuda(
    const at::Tensor& grad_q_out,
    const at::Tensor& grad_k_out,
    const at::Tensor& q_in,
    const at::Tensor& k_in,
    const at::Tensor& cos_tab,
    const at::Tensor& sin_tab
);

std::vector<at::Tensor> recurrent_chunk_state_scan_forward_cuda(
    const at::Tensor& delta_S,
    const at::Tensor& gamma_c,
    const at::Tensor& h_prev_opt
);

std::vector<at::Tensor> recurrent_chunk_state_scan_backward_cuda(
    const at::Tensor& grad_all_states,
    const at::Tensor& grad_h_last_opt,
    const at::Tensor& all_states,
    const at::Tensor& gamma_c,
    bool return_grad_h_prev
);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_rope_elu_forward",
        &fused_rope_elu_forward_cuda,
        "Fused RoPE + ELU(x)+1 + Q-scaling forward pass (CUDA)"
    );
    m.def(
        "fused_rope_elu_backward",
        &fused_rope_elu_backward_cuda,
        "Fused RoPE + ELU(x)+1 + Q-scaling backward pass (CUDA)"
    );
    m.def(
        "recurrent_chunk_state_scan_forward",
        &recurrent_chunk_state_scan_forward_cuda,
        "Recurrent chunk state scan forward pass (CUDA)"
    );
    m.def(
        "recurrent_chunk_state_scan_backward",
        &recurrent_chunk_state_scan_backward_cuda,
        "Recurrent chunk state scan backward pass (CUDA)"
    );
}
