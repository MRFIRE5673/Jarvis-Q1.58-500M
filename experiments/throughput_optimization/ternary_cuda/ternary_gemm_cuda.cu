#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "ternary_gemm.h"

// ---------------------------------------------------------------------------
// Kernel 1: High-throughput Fused Quantize & Pack (4 weights per byte)
// ---------------------------------------------------------------------------
__global__ void quantize_and_pack_kernel(
    const float* __restrict__ w_fp32,
    uint8_t* __restrict__ w_packed,
    float inv_alpha,
    int N,
    int K
) {
    int total_bytes = N * (K / 4);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bytes) return;

    int row = idx / (K / 4);
    int col_byte = idx % (K / 4);
    int k_base = col_byte * 4;
    int src_offset = row * K + k_base;

    // Load 4 contiguous float elements
    float w0 = w_fp32[src_offset + 0] * inv_alpha;
    float w1 = w_fp32[src_offset + 1] * inv_alpha;
    float w2 = w_fp32[src_offset + 2] * inv_alpha;
    float w3 = w_fp32[src_offset + 3] * inv_alpha;

    // Fast ternary encoding:
    // val > 0.5 -> 1 (01), val < -0.5 -> 2 (10), else 0 (00)
    uint8_t c0 = (w0 > 0.5f) ? 1 : ((w0 < -0.5f) ? 2 : 0);
    uint8_t c1 = (w1 > 0.5f) ? 1 : ((w1 < -0.5f) ? 2 : 0);
    uint8_t c2 = (w2 > 0.5f) ? 1 : ((w2 < -0.5f) ? 2 : 0);
    uint8_t c3 = (w3 > 0.5f) ? 1 : ((w3 < -0.5f) ? 2 : 0);

    uint8_t packed = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
    w_packed[idx] = packed;
}

// ---------------------------------------------------------------------------
// Kernel 2: Packed Ternary GEMM Forward: Y = alpha * (X @ W_q^T) + bias
// X: (M, K) __nv_bfloat16, W_packed: (N, K/4) uint8_t, Y: (M, N) __nv_bfloat16
// Tiled with Shared Memory & Register Unpacking
// ---------------------------------------------------------------------------
#define BM 64
#define BN 64
#define BK 64 // reduction chunk (64 elements = 16 bytes of packed weights)

__global__ void packed_ternary_gemm_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ W_packed,
    __nv_bfloat16* __restrict__ Y,
    const __nv_bfloat16* __restrict__ bias,
    float alpha,
    int M,
    int N,
    int K
) {
    // 2D thread block: 16 x 16 = 256 threads
    // Each thread computes a 4x4 sub-tile of outputs: BM/16 = 4, BN/16 = 4
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    int tid = ty * 16 + tx; // 0..255

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Shared memory buffers:
    // s_X: 64 x 64 bfloat16 = 8,192 bytes
    // s_W: 64 x 16 uint8 = 1,024 bytes (8x smaller than dense BF16!)
    __shared__ __nv_bfloat16 s_X[BM][BK + 1]; // +1 to avoid bank conflicts
    __shared__ uint8_t s_W[BN][BK / 4];

    // Thread register accumulators (4x4 tile per thread = 16 values)
    float accum[4][4] = {0.0f};

    int num_k_tiles = K / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_offset = kt * BK;

        // 1. Cooperative load of X into s_X (64 x 64 = 4096 elements, 256 threads -> 16 elements/thread)
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int load_idx = tid * 16 + i;
            int r = load_idx / BK;
            int c = load_idx % BK;
            int global_m = block_row + r;
            int global_k = k_offset + c;
            if (global_m < M && global_k < K) {
                s_X[r][c] = X[global_m * K + global_k];
            } else {
                s_X[r][c] = __float2bfloat16(0.0f);
            }
        }

        // 2. Cooperative load of W_packed into s_W (64 x 16 = 1024 bytes, 256 threads -> 4 bytes/thread)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid * 4 + i;
            int r = load_idx / (BK / 4);
            int c = load_idx % (BK / 4);
            int global_n = block_col + r;
            int global_k_byte = (k_offset / 4) + c;
            if (global_n < N && global_k_byte < (K / 4)) {
                s_W[r][c] = W_packed[global_n * (K / 4) + global_k_byte];
            } else {
                s_W[r][c] = 0;
            }
        }

        __syncthreads();

        // 3. Compute 4x4 sub-tile accumulation over BK
        #pragma unroll
        for (int kb = 0; kb < BK / 4; ++kb) {
            // Load 4 weight bytes for the 4 rows of N computed by this thread
            uint8_t w_bytes[4];
            #pragma unroll
            for (int n_sub = 0; n_sub < 4; ++n_sub) {
                w_bytes[n_sub] = s_W[ty * 4 + n_sub][kb];
            }

            // Unroll 4 ternary weights per byte
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                int k_elem = kb * 4 + p;

                // Load 4 elements of X for the 4 rows of M computed by this thread
                float x_vals[4];
                #pragma unroll
                for (int m_sub = 0; m_sub < 4; ++m_sub) {
                    x_vals[m_sub] = __bfloat162float(s_X[tx * 4 + m_sub][k_elem]);
                }

                #pragma unroll
                for (int n_sub = 0; n_sub < 4; ++n_sub) {
                    uint8_t code = (w_bytes[n_sub] >> (2 * p)) & 0x03;
                    // Arithmetic decode: (code & 1) - (code >> 1)
                    int w = (code & 1) - (code >> 1);
                    if (w != 0) {
                        float wf = (float)w;
                        #pragma unroll
                        for (int m_sub = 0; m_sub < 4; ++m_sub) {
                            accum[m_sub][n_sub] += x_vals[m_sub] * wf;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // 4. Epilogue: Apply scale alpha, add bias, write out in BF16
    #pragma unroll
    for (int m_sub = 0; m_sub < 4; ++m_sub) {
        int global_m = block_row + tx * 4 + m_sub;
        if (global_m < M) {
            #pragma unroll
            for (int n_sub = 0; n_sub < 4; ++n_sub) {
                int global_n = block_col + ty * 4 + n_sub;
                if (global_n < N) {
                    float val = accum[m_sub][n_sub] * alpha;
                    if (bias != nullptr) {
                        val += __bfloat162float(bias[global_n]);
                    }
                    Y[global_m * N + global_n] = __float2bfloat16(val);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C++ Host Invocation Wrappers
// ---------------------------------------------------------------------------
at::Tensor pack_ternary_weights(
    const at::Tensor& w_fp32,
    double alpha
) {
    TORCH_CHECK(w_fp32.is_cuda(), "w_fp32 must be a CUDA tensor");
    TORCH_CHECK(w_fp32.dim() == 2, "w_fp32 must be 2D (N, K)");
    int N = w_fp32.size(0);
    int K = w_fp32.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for 2-bit packing");

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(w_fp32.device());
    at::Tensor w_packed = torch::empty({N, K / 4}, options);

    int total_bytes = N * (K / 4);
    int threads = 256;
    int blocks = (total_bytes + threads - 1) / threads;
    float inv_alpha = 1.0f / (float)alpha;

    auto stream = c10::cuda::getCurrentCUDAStream();
    quantize_and_pack_kernel<<<blocks, threads, 0, stream>>>(
        w_fp32.data_ptr<float>(),
        w_packed.data_ptr<uint8_t>(),
        inv_alpha,
        N,
        K
    );
    return w_packed;
}

at::Tensor packed_ternary_gemm_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    double alpha,
    const c10::optional<at::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(w_packed.dtype() == torch::kUInt8, "w_packed must be uint8");

    int M = x.size(0);
    int K = x.size(1);
    int N = w_packed.size(0);
    TORCH_CHECK(w_packed.size(1) == K / 4, "w_packed second dimension must be K/4");

    auto y = torch::empty({M, N}, x.options());

    const __nv_bfloat16* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.value().data_ptr<at::BFloat16>());
    }

    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    auto stream = c10::cuda::getCurrentCUDAStream();
    packed_ternary_gemm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        w_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        bias_ptr,
        (float)alpha,
        M,
        N,
        K
    );

    return y;
}
