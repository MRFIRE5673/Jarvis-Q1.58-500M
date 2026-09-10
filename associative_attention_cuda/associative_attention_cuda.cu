#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cmath>

// ===========================================================================
// CUDA Kernels for Jarvis Associative Linear Attention
// Target Architecture: NVIDIA RTX 5070 / sm_120 (Blackwell)
// ===========================================================================

namespace {

// Inline helper for ELU+1
__device__ __forceinline__ float elu_plus_one(float x) {
    return (x > 0.0f) ? (x + 1.0f) : expf(x);
}

// Inline helper for d(ELU+1)/dx
__device__ __forceinline__ float d_elu_plus_one(float x) {
    return (x > 0.0f) ? 1.0f : expf(x);
}

// ---------------------------------------------------------------------------
// 1. FUSED ROPE + ELU+1 + SCALING FORWARD KERNEL (BF16 & FP32)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void fused_rope_elu_forward_kernel(
    const scalar_t* __restrict__ q,       // (B, H, T, D)
    const scalar_t* __restrict__ k,       // (B, H, T, D)
    const scalar_t* __restrict__ cos_tab, // (T, D)
    const scalar_t* __restrict__ sin_tab, // (T, D)
    scalar_t* __restrict__ q_out,         // (B, H, T, D)
    scalar_t* __restrict__ k_out,         // (B, H, T, D)
    const float inv_scale,                // 1.0 / sqrt(D)
    const int B,
    const int H,
    const int T,
    const int D,
    const int half_D
) {
    // Total tokens: B * H * T
    const int total_tokens = B * H * T;
    const int token_idx = blockIdx.x; // token index [0, total_tokens - 1]
    const int d = threadIdx.x;        // feature pair index [0, half_D - 1]

    if (token_idx >= total_tokens || d >= half_D) return;

    const int t = token_idx % T;
    const int offset_1 = token_idx * D + d;
    const int offset_2 = offset_1 + half_D;

    const int tab_offset_1 = t * D + d;
    const int tab_offset_2 = tab_offset_1 + half_D;

    // Read inputs
    float q1_in, q2_in, k1_in, k2_in;
    float c1, s1, c2, s2;

    if constexpr (std::is_same<scalar_t, at::BFloat16>::value) {
        q1_in = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&q[offset_1]));
        q2_in = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&q[offset_2]));
        k1_in = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&k[offset_1]));
        k2_in = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&k[offset_2]));

        c1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&cos_tab[tab_offset_1]));
        s1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sin_tab[tab_offset_1]));
        c2 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&cos_tab[tab_offset_2]));
        s2 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sin_tab[tab_offset_2]));
    } else {
        q1_in = static_cast<float>(q[offset_1]);
        q2_in = static_cast<float>(q[offset_2]);
        k1_in = static_cast<float>(k[offset_1]);
        k2_in = static_cast<float>(k[offset_2]);

        c1 = static_cast<float>(cos_tab[tab_offset_1]);
        s1 = static_cast<float>(sin_tab[tab_offset_1]);
        c2 = static_cast<float>(cos_tab[tab_offset_2]);
        s2 = static_cast<float>(sin_tab[tab_offset_2]);
    }

    // 1. ELU+1 and Q-scaling
    const float q1 = elu_plus_one(q1_in) * inv_scale;
    const float q2 = elu_plus_one(q2_in) * inv_scale;
    const float k1 = elu_plus_one(k1_in);
    const float k2 = elu_plus_one(k2_in);

    // 2. Rotary Position Embedding:
    // rotate_half(x) = [-x2, x1]
    // x_rot = x * cos + rotate_half(x) * sin
    // First half (d < half_D):   x1 * c1 - x2 * s1
    // Second half (d >= half_D): x2 * c2 + x1 * s2
    const float q1_rot = q1 * c1 - q2 * s1;
    const float q2_rot = q2 * c2 + q1 * s2;

    const float k1_rot = k1 * c1 - k2 * s1;
    const float k2_rot = k2 * c2 + k1 * s2;

    // Write outputs
    if constexpr (std::is_same<scalar_t, at::BFloat16>::value) {
        *reinterpret_cast<__nv_bfloat16*>(&q_out[offset_1]) = __float2bfloat16(q1_rot);
        *reinterpret_cast<__nv_bfloat16*>(&q_out[offset_2]) = __float2bfloat16(q2_rot);
        *reinterpret_cast<__nv_bfloat16*>(&k_out[offset_1]) = __float2bfloat16(k1_rot);
        *reinterpret_cast<__nv_bfloat16*>(&k_out[offset_2]) = __float2bfloat16(k2_rot);
    } else {
        q_out[offset_1] = static_cast<scalar_t>(q1_rot);
        q_out[offset_2] = static_cast<scalar_t>(q2_rot);
        k_out[offset_1] = static_cast<scalar_t>(k1_rot);
        k_out[offset_2] = static_cast<scalar_t>(k2_rot);
    }
}

// ---------------------------------------------------------------------------
// 2. FUSED ROPE + ELU+1 + SCALING BACKWARD KERNEL (BF16 & FP32)
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void fused_rope_elu_backward_kernel(
    const scalar_t* __restrict__ grad_q_out, // (B, H, T, D)
    const scalar_t* __restrict__ grad_k_out, // (B, H, T, D)
    const scalar_t* __restrict__ q_in,       // (B, H, T, D) original inputs
    const scalar_t* __restrict__ k_in,       // (B, H, T, D) original inputs
    const scalar_t* __restrict__ cos_tab,    // (T, D)
    const scalar_t* __restrict__ sin_tab,    // (T, D)
    scalar_t* __restrict__ grad_q_in,        // (B, H, T, D) output grads
    scalar_t* __restrict__ grad_k_in,        // (B, H, T, D) output grads
    const float inv_scale,
    const int B,
    const int H,
    const int T,
    const int D,
    const int half_D
) {
    const int total_tokens = B * H * T;
    const int token_idx = blockIdx.x;
    const int d = threadIdx.x;

    if (token_idx >= total_tokens || d >= half_D) return;

    const int t = token_idx % T;
    const int offset_1 = token_idx * D + d;
    const int offset_2 = offset_1 + half_D;

    const int tab_offset_1 = t * D + d;
    const int tab_offset_2 = tab_offset_1 + half_D;

    float gq1_out, gq2_out, gk1_out, gk2_out;
    float q1_raw, q2_raw, k1_raw, k2_raw;
    float c1, s1, c2, s2;

    if constexpr (std::is_same<scalar_t, at::BFloat16>::value) {
        gq1_out = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&grad_q_out[offset_1]));
        gq2_out = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&grad_q_out[offset_2]));
        gk1_out = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&grad_k_out[offset_1]));
        gk2_out = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&grad_k_out[offset_2]));

        q1_raw = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&q_in[offset_1]));
        q2_raw = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&q_in[offset_2]));
        k1_raw = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&k_in[offset_1]));
        k2_raw = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&k_in[offset_2]));

        c1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&cos_tab[tab_offset_1]));
        s1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sin_tab[tab_offset_1]));
        c2 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&cos_tab[tab_offset_2]));
        s2 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sin_tab[tab_offset_2]));
    } else {
        gq1_out = static_cast<float>(grad_q_out[offset_1]);
        gq2_out = static_cast<float>(grad_q_out[offset_2]);
        gk1_out = static_cast<float>(grad_k_out[offset_1]);
        gk2_out = static_cast<float>(grad_k_out[offset_2]);

        q1_raw = static_cast<float>(q_in[offset_1]);
        q2_raw = static_cast<float>(q_in[offset_2]);
        k1_raw = static_cast<float>(k_in[offset_1]);
        k2_raw = static_cast<float>(k_in[offset_2]);

        c1 = static_cast<float>(cos_tab[tab_offset_1]);
        s1 = static_cast<float>(sin_tab[tab_offset_1]);
        c2 = static_cast<float>(cos_tab[tab_offset_2]);
        s2 = static_cast<float>(sin_tab[tab_offset_2]);
    }

    // Gradients through RoPE rotation:
    // q_rot1 = q1 * c1 - q2 * s1
    // q_rot2 = q2 * c2 + q1 * s2
    // dL/dq1 = gq1 * c1 + gq2 * s2
    // dL/dq2 = -gq1 * s1 + gq2 * c2
    const float dL_dq1 = gq1_out * c1 + gq2_out * s2;
    const float dL_dq2 = -gq1_out * s1 + gq2_out * c2;

    const float dL_dk1 = gk1_out * c1 + gk2_out * s2;
    const float dL_dk2 = -gk1_out * s1 + gk2_out * c2;

    // Gradients through ELU+1 & scaling:
    const float dq1 = dL_dq1 * inv_scale * d_elu_plus_one(q1_raw);
    const float dq2 = dL_dq2 * inv_scale * d_elu_plus_one(q2_raw);

    const float dk1 = dL_dk1 * d_elu_plus_one(k1_raw);
    const float dk2 = dL_dk2 * d_elu_plus_one(k2_raw);

    if constexpr (std::is_same<scalar_t, at::BFloat16>::value) {
        *reinterpret_cast<__nv_bfloat16*>(&grad_q_in[offset_1]) = __float2bfloat16(dq1);
        *reinterpret_cast<__nv_bfloat16*>(&grad_q_in[offset_2]) = __float2bfloat16(dq2);
        *reinterpret_cast<__nv_bfloat16*>(&grad_k_in[offset_1]) = __float2bfloat16(dk1);
        *reinterpret_cast<__nv_bfloat16*>(&grad_k_in[offset_2]) = __float2bfloat16(dk2);
    } else {
        grad_q_in[offset_1] = static_cast<scalar_t>(dq1);
        grad_q_in[offset_2] = static_cast<scalar_t>(dq2);
        grad_k_in[offset_1] = static_cast<scalar_t>(dk1);
        grad_k_in[offset_2] = static_cast<scalar_t>(dk2);
    }
}

// ---------------------------------------------------------------------------
// 3. RECURRENT CHUNK STATE SCAN FORWARD KERNEL (FP32 Accumulation)
// S_{k+1} = gamma_c * S_k + delta_S_k
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void recurrent_chunk_state_scan_forward_kernel(
    const scalar_t* __restrict__ delta_S,    // (B, H, Nc, D, D)
    const float* __restrict__ gamma_c_tab,   // (H,)
    const scalar_t* __restrict__ h_prev,     // (B, H, D, D) or nullptr
    scalar_t* __restrict__ all_states,       // (B, H, Nc, D, D) output carried states
    scalar_t* __restrict__ h_last,           // (B, H, D, D) final output state
    const int B,
    const int H,
    const int Nc,
    const int D
) {
    // Each threadblock handles one (batch, head) pair
    const int bh_idx = blockIdx.x; // [0, B * H - 1]
    if (bh_idx >= B * H) return;

    const int b = bh_idx / H;
    const int h = bh_idx % H;
    const float gamma_c = gamma_c_tab[h];

    const int state_size = D * D; // 64 * 64 = 4096
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Stride loop: each thread updates multiple matrix elements
    for (int elem_idx = tid; elem_idx < state_size; elem_idx += num_threads) {
        float cur_s = 0.0f;
        if (h_prev != nullptr) {
            const int prev_offset = (b * H + h) * state_size + elem_idx;
            cur_s = static_cast<float>(h_prev[prev_offset]);
        }

        for (int k = 0; k < Nc; ++k) {
            // 1. Record carried state for chunk k: all_states[b, h, k, elem_idx] = cur_s
            const int out_offset = ((b * H + h) * Nc + k) * state_size + elem_idx;
            all_states[out_offset] = static_cast<scalar_t>(cur_s);

            // 2. Accumulate delta_S: cur_s = gamma_c * cur_s + delta_S[b, h, k, elem_idx]
            const int ds_offset = out_offset;
            const float ds_val = static_cast<float>(delta_S[ds_offset]);
            cur_s = gamma_c * cur_s + ds_val;
        }

        // 3. Write final state to h_last
        const int last_offset = (b * H + h) * state_size + elem_idx;
        h_last[last_offset] = static_cast<scalar_t>(cur_s);
    }
}

// ---------------------------------------------------------------------------
// 4. RECURRENT CHUNK STATE SCAN BACKWARD KERNEL
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void recurrent_chunk_state_scan_backward_kernel(
    const scalar_t* __restrict__ grad_all_states, // (B, H, Nc, D, D)
    const scalar_t* __restrict__ grad_h_last,     // (B, H, D, D) or nullptr
    const scalar_t* __restrict__ all_states,      // (B, H, Nc, D, D) saved forward states
    const float* __restrict__ gamma_c_tab,        // (H,)
    scalar_t* __restrict__ grad_delta_S,          // (B, H, Nc, D, D)
    scalar_t* __restrict__ grad_h_prev,           // (B, H, D, D) or nullptr
    float* __restrict__ grad_gamma_c,             // (H,) partial grads
    const int B,
    const int H,
    const int Nc,
    const int D
) {
    const int bh_idx = blockIdx.x; // [0, B * H - 1]
    if (bh_idx >= B * H) return;

    const int b = bh_idx / H;
    const int h = bh_idx % H;
    const float gamma_c = gamma_c_tab[h];

    const int state_size = D * D;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float local_grad_gamma = 0.0f;

    for (int elem_idx = tid; elem_idx < state_size; elem_idx += num_threads) {
        float grad_s = 0.0f;
        if (grad_h_last != nullptr) {
            const int last_offset = (b * H + h) * state_size + elem_idx;
            grad_s = static_cast<float>(grad_h_last[last_offset]);
        }

        // Reverse scan from chunk Nc - 1 down to 0
        for (int k = Nc - 1; k >= 0; --k) {
            const int offset = ((b * H + h) * Nc + k) * state_size + elem_idx;
            
            // grad_delta_S[b, h, k] = grad_s (since S_{k+1} = gamma * S_k + delta_S_k)
            grad_delta_S[offset] = static_cast<scalar_t>(grad_s);

            // grad_gamma contribution: grad_s * S_k
            const float s_k = static_cast<float>(all_states[offset]);
            local_grad_gamma += grad_s * s_k;

            // Backward propagation to S_k:
            // S_k contributes to:
            // 1. S_{k+1} via (gamma_c * S_k)
            // 2. all_states[k] directly (grad_all_states[k])
            const float direct_grad = static_cast<float>(grad_all_states[offset]);
            grad_s = direct_grad + gamma_c * grad_s;
        }

        if (grad_h_prev != nullptr) {
            const int prev_offset = (b * H + h) * state_size + elem_idx;
            grad_h_prev[prev_offset] = static_cast<scalar_t>(grad_s);
        }
    }

    // Warp reduce grad_gamma and atomic add to grad_gamma_c[h]
    for (int offset = 16; offset > 0; offset /= 2) {
        local_grad_gamma += __shfl_down_sync(0xffffffff, local_grad_gamma, offset);
    }
    if ((tid % 32) == 0) {
        atomicAdd(&grad_gamma_c[h], local_grad_gamma);
    }
}

} // namespace

// ===========================================================================
// C++ Wrapper Interface
// ===========================================================================

std::vector<at::Tensor> fused_rope_elu_forward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos_tab,
    const at::Tensor& sin_tab
) {
    const int B = q.size(0);
    const int H = q.size(1);
    const int T = q.size(2);
    const int D = q.size(3);
    const int half_D = D / 2;
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto q_out = at::empty_like(q);
    auto k_out = at::empty_like(k);

    const int total_tokens = B * H * T;
    const int threads_per_block = half_D; // 32 threads per token (1 warp)
    const int blocks = total_tokens;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "fused_rope_elu_forward_kernel", ([&] {
            fused_rope_elu_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                cos_tab.data_ptr<scalar_t>(),
                sin_tab.data_ptr<scalar_t>(),
                q_out.data_ptr<scalar_t>(),
                k_out.data_ptr<scalar_t>(),
                inv_scale,
                B, H, T, D, half_D
            );
        })
    );

    return {q_out, k_out};
}

std::vector<at::Tensor> fused_rope_elu_backward_cuda(
    const at::Tensor& grad_q_out,
    const at::Tensor& grad_k_out,
    const at::Tensor& q_in,
    const at::Tensor& k_in,
    const at::Tensor& cos_tab,
    const at::Tensor& sin_tab
) {
    const int B = q_in.size(0);
    const int H = q_in.size(1);
    const int T = q_in.size(2);
    const int D = q_in.size(3);
    const int half_D = D / 2;
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto grad_q_in = at::empty_like(q_in);
    auto grad_k_in = at::empty_like(k_in);

    const int total_tokens = B * H * T;
    const int threads_per_block = half_D;
    const int blocks = total_tokens;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_in.scalar_type(), "fused_rope_elu_backward_kernel", ([&] {
            fused_rope_elu_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                grad_q_out.data_ptr<scalar_t>(),
                grad_k_out.data_ptr<scalar_t>(),
                q_in.data_ptr<scalar_t>(),
                k_in.data_ptr<scalar_t>(),
                cos_tab.data_ptr<scalar_t>(),
                sin_tab.data_ptr<scalar_t>(),
                grad_q_in.data_ptr<scalar_t>(),
                grad_k_in.data_ptr<scalar_t>(),
                inv_scale,
                B, H, T, D, half_D
            );
        })
    );

    return {grad_q_in, grad_k_in};
}

std::vector<at::Tensor> recurrent_chunk_state_scan_forward_cuda(
    const at::Tensor& delta_S,
    const at::Tensor& gamma_c,
    const at::Tensor& h_prev_opt
) {
    const int B = delta_S.size(0);
    const int H = delta_S.size(1);
    const int Nc = delta_S.size(2);
    const int D = delta_S.size(3);

    auto all_states = at::empty_like(delta_S);
    auto h_last = at::empty({B, H, D, D}, delta_S.options());

    const int blocks = B * H;
    const int threads_per_block = 256;

    auto gamma_c_float = gamma_c.contiguous().to(at::kFloat);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        delta_S.scalar_type(), "recurrent_chunk_state_scan_forward_kernel", ([&] {
            const scalar_t* h_prev_ptr = h_prev_opt.defined() && h_prev_opt.numel() > 0
                ? h_prev_opt.data_ptr<scalar_t>() : nullptr;

            recurrent_chunk_state_scan_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                delta_S.data_ptr<scalar_t>(),
                gamma_c_float.data_ptr<float>(),
                h_prev_ptr,
                all_states.data_ptr<scalar_t>(),
                h_last.data_ptr<scalar_t>(),
                B, H, Nc, D
            );
        })
    );

    return {all_states, h_last};
}

std::vector<at::Tensor> recurrent_chunk_state_scan_backward_cuda(
    const at::Tensor& grad_all_states,
    const at::Tensor& grad_h_last_opt,
    const at::Tensor& all_states,
    const at::Tensor& gamma_c,
    bool return_grad_h_prev
) {
    const int B = all_states.size(0);
    const int H = all_states.size(1);
    const int Nc = all_states.size(2);
    const int D = all_states.size(3);

    auto grad_delta_S = at::empty_like(all_states);
    at::Tensor grad_h_prev = return_grad_h_prev ? at::empty({B, H, D, D}, all_states.options()) : at::Tensor();
    auto grad_gamma_c = at::zeros({H}, gamma_c.options().dtype(at::kFloat));

    const int blocks = B * H;
    const int threads_per_block = 256;

    auto gamma_c_float = gamma_c.contiguous().to(at::kFloat);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        all_states.scalar_type(), "recurrent_chunk_state_scan_backward_kernel", ([&] {
            const scalar_t* grad_last_ptr = grad_h_last_opt.defined() && grad_h_last_opt.numel() > 0
                ? grad_h_last_opt.data_ptr<scalar_t>() : nullptr;
            scalar_t* grad_prev_ptr = return_grad_h_prev ? grad_h_prev.data_ptr<scalar_t>() : nullptr;

            recurrent_chunk_state_scan_backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                grad_all_states.data_ptr<scalar_t>(),
                grad_last_ptr,
                all_states.data_ptr<scalar_t>(),
                gamma_c_float.data_ptr<float>(),
                grad_delta_S.data_ptr<scalar_t>(),
                grad_prev_ptr,
                grad_gamma_c.data_ptr<float>(),
                B, H, Nc, D
            );
        })
    );

    return {grad_delta_S, grad_h_prev, grad_gamma_c.to(gamma_c.dtype())};
}
