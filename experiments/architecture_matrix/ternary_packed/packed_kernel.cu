// experiments/architecture_matrix/ternary_packed/packed_kernel.cu
/*
 * PACKED TERNARY MATRIX-MULTIPLICATION CUDA KERNEL PROTOTYPE
 * ===========================================================
 * Computes:
 *   Y = alpha * (X @ W_unpacked^T)
 *
 * Where:
 *   X             : (M, K) in __nv_bfloat16
 *   packed_weight : (N, K/4) in uint8_t (4 ternary values {-1, 0, +1} per byte)
 *   alpha         : (N,) or (1,) scale factor in float
 *   Y             : (M, N) in __nv_bfloat16
 *
 * Arithmetic property:
 *   Multiplications are eliminated during accumulation.
 *   Inner loop uses only ADD, SUB, and conditional/table accumulation.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void packed_ternary_gemm_kernel(
    const __nv_bfloat16* __restrict__ X,          // (M, K)
    const uint8_t* __restrict__ W_packed,        // (N, K/4)
    const float* __restrict__ alpha,             // (N,) or (1,)
    __nv_bfloat16* __restrict__ Y,               // (M, N)
    int M, int N, int K,
    bool per_channel_alpha
) {
    // 2D grid: blockIdx.x -> N (output channels), blockIdx.y -> M (tokens/batch)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // index in M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // index in N

    if (row >= M || col >= N) return;

    int K_packed = K / 4;
    float accum = 0.0f;

    const __nv_bfloat16* x_row = X + row * K;
    const uint8_t* w_col = W_packed + col * K_packed;

    // Loop over packed bytes
    for (int k = 0; k < K_packed; ++k) {
        uint8_t byte_val = w_col[k];
        int k_base = k * 4;

        // Unpack 4 trits (2 bits each)
        // 00 (0) = 0, 01 (1) = +1, 10 (2) = -1, 11 (3) = 0
        uint8_t t0 = byte_val & 0x03;
        uint8_t t1 = (byte_val >> 2) & 0x03;
        uint8_t t2 = (byte_val >> 4) & 0x03;
        uint8_t t3 = (byte_val >> 6) & 0x03;

        float x0 = __bfloat162float(x_row[k_base]);
        float x1 = __bfloat162float(x_row[k_base + 1]);
        float x2 = __bfloat162float(x_row[k_base + 2]);
        float x3 = __bfloat162float(x_row[k_base + 3]);

        // Branchless ternary accumulation:
        // if code == 1 -> +x
        // if code == 2 -> -x
        if (t0 == 1) accum += x0; else if (t0 == 2) accum -= x0;
        if (t1 == 1) accum += x1; else if (t1 == 2) accum -= x1;
        if (t2 == 1) accum += x2; else if (t2 == 2) accum -= x2;
        if (t3 == 1) accum += x3; else if (t3 == 2) accum -= x3;
    }

    float scale = per_channel_alpha ? alpha[col] : alpha[0];
    Y[row * N + col] = __float2bfloat16(accum * scale);
}

torch::Tensor packed_ternary_matmul_cuda(
    torch::Tensor X,             // (M, K) BF16
    torch::Tensor W_packed,      // (N, K/4) UINT8
    torch::Tensor alpha          // (N,) or (1,) FLOAT32
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(X.dtype() == torch::kBFloat16, "X must be BFloat16");
    TORCH_CHECK(W_packed.dtype() == torch::kUInt8, "W_packed must be UInt8");

    int M = X.size(0);
    int K = X.size(1);
    int N = W_packed.size(0);
    int K_packed = W_packed.size(1);

    TORCH_CHECK(K == K_packed * 4, "K must equal K_packed * 4");

    auto Y = torch::empty({M, N}, X.options());
    bool per_channel = (alpha.numel() == N);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    packed_ternary_gemm_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        W_packed.data_ptr<uint8_t>(),
        alpha.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        M, N, K,
        per_channel
    );

    return Y;
}
