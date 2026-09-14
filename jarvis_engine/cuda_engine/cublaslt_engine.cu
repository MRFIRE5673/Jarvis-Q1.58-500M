#include "cublaslt_engine.h"
#include <cublasLt.h>
#include <stdio.h>
#include <stdlib.h>

static cublasLtHandle_t g_lt = nullptr;
static void* g_ws = nullptr;
static size_t g_ws_size = 32 * 1024 * 1024; // 32 MiB static workspace
static bool g_initialized = false;

// Pre-configured operation descriptors
static cublasLtMatmulDesc_t g_desc_nt = nullptr;
static cublasLtMatmulDesc_t g_desc_nn = nullptr;
static cublasLtMatmulDesc_t g_desc_tn = nullptr;

// Pre-configured layout descriptors
static cublasLtMatrixLayout_t g_layout_x_2048_1024 = nullptr;
static cublasLtMatrixLayout_t g_layout_w_qkv = nullptr;
static cublasLtMatrixLayout_t g_layout_w_qkv_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_out_qkv = nullptr;
static cublasLtMatmulAlgo_t g_algo_qkv_fwd;
static cublasLtMatmulAlgo_t g_algo_qkv_fwd_fp8;

static cublasLtMatrixLayout_t g_layout_w_attn_out = nullptr;
static cublasLtMatrixLayout_t g_layout_q_chunk = nullptr;
static cublasLtMatrixLayout_t g_layout_out_attn = nullptr;
static cublasLtMatmulAlgo_t g_algo_attn_out_fwd;

static cublasLtMatrixLayout_t g_layout_w_router = nullptr;
static cublasLtMatrixLayout_t g_layout_out_router = nullptr;
static cublasLtMatmulAlgo_t g_algo_router_fwd;

static cublasLtMatrixLayout_t g_layout_x_disp = nullptr;
static cublasLtMatrixLayout_t g_layout_w1 = nullptr;
static cublasLtMatrixLayout_t g_layout_h1 = nullptr;
static cublasLtMatmulAlgo_t g_algo_w1_fwd;

static cublasLtMatrixLayout_t g_layout_act = nullptr;
static cublasLtMatrixLayout_t g_layout_w2 = nullptr;
static cublasLtMatrixLayout_t g_layout_disp_y = nullptr;
static cublasLtMatmulAlgo_t g_algo_w2_fwd;

// Phase 28: FP8 MoE Operation Descriptors, Layouts, and Algorithms
static cublasLtMatmulDesc_t g_desc_nt_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_w1_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_x_disp_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_w2_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_act_fp8 = nullptr;
static cublasLtMatmulAlgo_t g_algo_w1_fp8;
static cublasLtMatmulAlgo_t g_algo_w2_fp8;

static cublasLtMatrixLayout_t g_layout_w_lm_head = nullptr;
static cublasLtMatrixLayout_t g_layout_w_lm_head_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_x_2048_1024_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_logits = nullptr;
static cublasLtMatmulAlgo_t g_algo_lm_head_fwd;
static cublasLtMatmulAlgo_t g_algo_lm_head_fwd_fp8;

static cublasLtMatrixLayout_t g_layout_d_logits_nn = nullptr;
static cublasLtMatrixLayout_t g_layout_w_lm_head_nn = nullptr;
static cublasLtMatrixLayout_t g_layout_d_final_norm_nn = nullptr;
static cublasLtMatmulAlgo_t g_algo_lm_head_bwd_dx;

static cublasLtMatrixLayout_t g_layout_d_logits_tn = nullptr;
static cublasLtMatrixLayout_t g_layout_final_norm_tn = nullptr;
static cublasLtMatrixLayout_t g_layout_d_lm_head_tn = nullptr;
static cublasLtMatmulAlgo_t g_algo_lm_head_bwd_dw;

// Phase 28: FP8 LM Head Backward
static cublasLtMatmulDesc_t g_desc_nn_fp8 = nullptr;
static cublasLtMatmulDesc_t g_desc_tn_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_d_logits_nn_fp8 = nullptr;
static cublasLtMatrixLayout_t g_layout_d_logits_tn_fp8 = nullptr;
static cublasLtMatmulAlgo_t g_algo_lm_head_bwd_dx_fp8;
static cublasLtMatmulAlgo_t g_algo_lm_head_bwd_dw_fp8;

static cublasLtMatrixLayout_t g_layout_d_x_tn = nullptr;
static cublasLtMatrixLayout_t g_layout_stashed_x_tn = nullptr;
static cublasLtMatrixLayout_t g_layout_d_qkv_slice_tn = nullptr;
static cublasLtMatmulAlgo_t g_algo_qkv_bwd_slice;
static cublasLtMatmulAlgo_t g_algo_qkv_bwd_slice_fp8;

static cublasLtMatmulAlgo_t autotune_matmul_algo(
    cublasLtHandle_t lt,
    cublasLtMatmulDesc_t desc,
    cublasLtMatrixLayout_t layoutA,
    cublasLtMatrixLayout_t layoutB,
    cublasLtMatrixLayout_t layoutC,
    cublasLtMatrixLayout_t layoutD,
    cublasLtMatmulPreference_t pref,
    const void* A,
    const void* B,
    void* C,
    void* D,
    float alpha,
    float beta,
    void* ws,
    size_t ws_size,
    const char* shape_name,
    int locked_candidate_idx = -1
) {
    cublasLtMatmulHeuristicResult_t res_list[32];
    int returned = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        lt, desc, layoutA, layoutB, layoutC, layoutD, pref, 32, res_list, &returned
    );
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        printf("[cuBLASLt] Heuristic query fallback for %s (status=%d, returned=%d)\n", shape_name, (int)status, returned);
        return res_list[0].algo;
    }
    
    // Fast path: If a verified winning candidate is locked for SM120, lock it permanently
    if (locked_candidate_idx >= 0 && locked_candidate_idx < returned && res_list[locked_candidate_idx].state == CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLASLt SM120 Locked] %-18s: Winner [%2d/%2d]\n",
               shape_name, locked_candidate_idx, returned);
        return res_list[locked_candidate_idx].algo;
    }
    
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);
    
    float best_time_ms = 1e9f;
    int best_idx = 0;
    
    for (int i = 0; i < returned; ++i) {
        if (res_list[i].state != CUBLAS_STATUS_SUCCESS) continue;
        
        // Warmup 10 iterations to stabilize SM clock
        for (int w = 0; w < 10; ++w) {
            cublasLtMatmul(
                lt, desc, &alpha, A, layoutA, B, layoutB, &beta, C, layoutC, D, layoutD,
                &res_list[i].algo, ws, ws_size, 0
            );
        }
        cudaDeviceSynchronize();
        
        // Timed 25 iterations
        cudaEventRecord(ev_start, 0);
        for (int it = 0; it < 25; ++it) {
            cublasLtMatmul(
                lt, desc, &alpha, A, layoutA, B, layoutB, &beta, C, layoutC, D, layoutD,
                &res_list[i].algo, ws, ws_size, 0
            );
        }
        cudaEventRecord(ev_end, 0);
        cudaEventSynchronize(ev_end);
        
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_end);
        float avg_ms = elapsed_ms / 25.0f;
        
        if (avg_ms < best_time_ms) {
            best_time_ms = avg_ms;
            best_idx = i;
        }
    }
    
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    
    printf("[cuBLASLt Autotuner] %-18s: Winner [%2d/%2d] -> %6.3f ms\n",
           shape_name, best_idx, returned, best_time_ms);
           
    return res_list[best_idx].algo;
}

void init_cublaslt_engine(size_t workspace_bytes) {
    if (g_initialized) return;
    
    g_ws_size = workspace_bytes;
    cublasLtCreate(&g_lt);
    cudaMalloc(&g_ws, g_ws_size);
    
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    
    // 1. Operation Descriptors
    cublasLtMatmulDescCreate(&g_desc_nt, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(g_desc_nt, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(g_desc_nt, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    
    cublasLtMatmulDescCreate(&g_desc_nn, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(g_desc_nn, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(g_desc_nn, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    
    cublasLtMatmulDescCreate(&g_desc_tn, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(g_desc_tn, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(g_desc_tn, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT));
    
    // Preference
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws_size, sizeof(g_ws_size));
    
    // Layouts & Algorithms (Targeting measured optimal Blackwell SM120 algorithms)
    // Temporary scratch buffers for empirical autotuning
    void *d_scratchA = nullptr, *d_scratchB = nullptr, *d_scratchC = nullptr;
    size_t max_bytes = 50304ULL * 2048ULL * sizeof(__nv_bfloat16);
    cudaMalloc(&d_scratchA, max_bytes);
    cudaMalloc(&d_scratchB, max_bytes);
    cudaMalloc(&d_scratchC, max_bytes);
    cudaMemset(d_scratchA, 0, max_bytes);
    cudaMemset(d_scratchB, 0, max_bytes);
    cudaMemset(d_scratchC, 0, max_bytes);
    
    // A. QKV Forward: C (2048 x 3072) = A (2048 x 1024) * B^T (3072 x 1024).T
    // col-major: m=3072, n=2048, k=1024
    cublasLtMatrixLayoutCreate(&g_layout_w_qkv, CUDA_R_16BF, 1024, 3072, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_x_2048_1024, CUDA_R_16BF, 1024, 2048, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_out_qkv, CUDA_R_16BF, 3072, 2048, 3072);
    
    g_algo_qkv_fwd = autotune_matmul_algo(g_lt, g_desc_nt, g_layout_w_qkv, g_layout_x_2048_1024, g_layout_out_qkv, g_layout_out_qkv, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "QKV Fwd", 1);
    
    // B. Attn Out Forward: C (2048 x 1024) = A (2048 x 1024) * B^T (1024 x 1024).T
    // Q chunk is sliced from (2048, 3072), so its leading dimension is 3072!
    cublasLtMatrixLayoutCreate(&g_layout_w_attn_out, CUDA_R_16BF, 1024, 1024, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_q_chunk, CUDA_R_16BF, 1024, 2048, 3072);
    cublasLtMatrixLayoutCreate(&g_layout_out_attn, CUDA_R_16BF, 1024, 2048, 1024);
    g_algo_attn_out_fwd = autotune_matmul_algo(g_lt, g_desc_nt, g_layout_w_attn_out, g_layout_q_chunk, g_layout_out_attn, g_layout_out_attn, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "Attn Out Fwd", 0);
    
    // C. Router Forward: C (2048 x 4) = A (2048 x 1024) * B^T (4 x 1024).T
    cublasLtMatrixLayoutCreate(&g_layout_w_router, CUDA_R_16BF, 1024, 4, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_out_router, CUDA_R_16BF, 4, 2048, 4);
    g_algo_router_fwd = autotune_matmul_algo(g_lt, g_desc_nt, g_layout_w_router, g_layout_x_2048_1024, g_layout_out_router, g_layout_out_router, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "Router Fwd", 0);
    
    // D. MoE W1 Forward: C (4096 x 2048) = A (4096 x 1024) * B^T (2048 x 1024).T
    cublasLtMatrixLayoutCreate(&g_layout_w1, CUDA_R_16BF, 1024, 2048, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_x_disp, CUDA_R_16BF, 1024, 4096, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_h1, CUDA_R_16BF, 2048, 4096, 2048);
    g_algo_w1_fwd = autotune_matmul_algo(g_lt, g_desc_nt, g_layout_w1, g_layout_x_disp, g_layout_h1, g_layout_h1, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "MoE W1 Fwd", 0);
    
    // E. MoE W2 Forward: C (4096 x 1024) = A (4096 x 2048) * B^T (1024 x 2048).T
    cublasLtMatrixLayoutCreate(&g_layout_w2, CUDA_R_16BF, 2048, 1024, 2048);
    cublasLtMatrixLayoutCreate(&g_layout_act, CUDA_R_16BF, 2048, 4096, 2048);
    cublasLtMatrixLayoutCreate(&g_layout_disp_y, CUDA_R_16BF, 1024, 4096, 1024);
    g_algo_w2_fwd = autotune_matmul_algo(g_lt, g_desc_nt, g_layout_w2, g_layout_act, g_layout_disp_y, g_layout_disp_y, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "MoE W2 Fwd", 3);
    
    // E2. Phase 28: FP8 MoE W1 & W2 Layouts and Algorithms
    cublasLtMatmulDescCreate(&g_desc_nt_fp8, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(g_desc_nt_fp8, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(g_desc_nt_fp8, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    
    cublasLtMatrixLayoutCreate(&g_layout_w1_fp8, CUDA_R_8F_E4M3, 1024, 2048, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_x_disp_fp8, CUDA_R_8F_E4M3, 1024, 4096, 1024);
    g_algo_w1_fp8 = autotune_matmul_algo(g_lt, g_desc_nt_fp8, g_layout_w1_fp8, g_layout_x_disp_fp8, g_layout_h1, g_layout_h1, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "MoE W1 FP8", 1);
    
    cublasLtMatrixLayoutCreate(&g_layout_w2_fp8, CUDA_R_8F_E4M3, 2048, 1024, 2048);
    cublasLtMatrixLayoutCreate(&g_layout_act_fp8, CUDA_R_8F_E4M3, 2048, 4096, 2048);
    g_algo_w2_fp8 = autotune_matmul_algo(g_lt, g_desc_nt_fp8, g_layout_w2_fp8, g_layout_act_fp8, g_layout_disp_y, g_layout_disp_y, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "MoE W2 FP8", 0);
    
    // F. LM Head Forward: C (2048 x 50304) = A (2048 x 1024) * B^T (50304 x 1024).T
    cublasLtMatrixLayoutCreate(&g_layout_w_lm_head, CUDA_R_16BF, 1024, 50304, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_logits, CUDA_R_16BF, 50304, 2048, 50304);
    g_algo_lm_head_fwd = autotune_matmul_algo(g_lt, g_desc_nt, g_layout_w_lm_head, g_layout_x_2048_1024, g_layout_logits, g_layout_logits, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "LM Head Fwd", 0);
    
    // F2. Phase 28: FP8 LM Head Forward
    cublasLtMatrixLayoutCreate(&g_layout_w_lm_head_fp8, CUDA_R_8F_E4M3, 1024, 50304, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_x_2048_1024_fp8, CUDA_R_8F_E4M3, 1024, 2048, 1024);
    g_algo_lm_head_fwd_fp8 = autotune_matmul_algo(g_lt, g_desc_nt_fp8, g_layout_w_lm_head_fp8, g_layout_x_2048_1024_fp8, g_layout_logits, g_layout_logits, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "LM Head FP8 Fwd", 1);
    
    // F3. Phase 30: FP8 QKV Forward: C (2048 x 3072) = A (2048 x 1024) * B^T (3072 x 1024).T
    cublasLtMatrixLayoutCreate(&g_layout_w_qkv_fp8, CUDA_R_8F_E4M3, 1024, 3072, 1024);
    g_algo_qkv_fwd_fp8 = autotune_matmul_algo(g_lt, g_desc_nt_fp8, g_layout_w_qkv_fp8, g_layout_x_2048_1024_fp8, g_layout_out_qkv, g_layout_out_qkv, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "QKV Fwd FP8", 0);
    
    // G. LM Head Bwd dX: C (2048 x 1024) = A (2048 x 50304) * B (50304 x 1024)
    cublasLtMatrixLayoutCreate(&g_layout_w_lm_head_nn, CUDA_R_16BF, 1024, 50304, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_d_logits_nn, CUDA_R_16BF, 50304, 2048, 50304);
    cublasLtMatrixLayoutCreate(&g_layout_d_final_norm_nn, CUDA_R_16BF, 1024, 2048, 1024);
    g_algo_lm_head_bwd_dx = autotune_matmul_algo(g_lt, g_desc_nn, g_layout_w_lm_head_nn, g_layout_d_logits_nn, g_layout_d_final_norm_nn, g_layout_d_final_norm_nn, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "LM Head Bwd dX", 0);
    
    // H. LM Head Bwd dW: C (50304 x 1024) += A^T (2048 x 50304).T * B (2048 x 1024)
    cublasLtMatrixLayoutCreate(&g_layout_final_norm_tn, CUDA_R_16BF, 1024, 2048, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_d_logits_tn, CUDA_R_16BF, 50304, 2048, 50304);
    cublasLtMatrixLayoutCreate(&g_layout_d_lm_head_tn, CUDA_R_16BF, 1024, 50304, 1024);
    g_algo_lm_head_bwd_dw = autotune_matmul_algo(g_lt, g_desc_tn, g_layout_final_norm_tn, g_layout_d_logits_tn, g_layout_d_lm_head_tn, g_layout_d_lm_head_tn, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 1.0f, g_ws, g_ws_size, "LM Head Bwd dW", 1);
    
    // H2. Phase 28: FP8 LM Head Backward Layouts & Algorithms
    cublasLtMatmulDescCreate(&g_desc_nn_fp8, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(g_desc_nn_fp8, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(g_desc_nn_fp8, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatmulDescCreate(&g_desc_tn_fp8, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(g_desc_tn_fp8, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(g_desc_tn_fp8, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT));

    cublasLtMatrixLayoutCreate(&g_layout_d_logits_nn_fp8, CUDA_R_8F_E4M3, 50304, 2048, 50304);
    cublasLtMatrixLayoutCreate(&g_layout_d_logits_tn_fp8, CUDA_R_8F_E4M3, 50304, 2048, 50304);

    g_algo_lm_head_bwd_dx_fp8 = autotune_matmul_algo(
        g_lt, g_desc_nn_fp8, g_layout_w_lm_head_fp8, g_layout_d_logits_nn_fp8,
        g_layout_d_final_norm_nn, g_layout_d_final_norm_nn, pref,
        d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 0.0f, g_ws, g_ws_size, "LM Head Bwd dX FP8", 1
    );

    g_algo_lm_head_bwd_dw_fp8 = autotune_matmul_algo(
        g_lt, g_desc_tn_fp8, g_layout_x_2048_1024_fp8, g_layout_d_logits_tn_fp8,
        g_layout_d_lm_head_tn, g_layout_d_lm_head_tn, pref,
        d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 1.0f, g_ws, g_ws_size, "LM Head Bwd dW FP8", 0
    );
    
    // I. QKV Bwd dSlice: C (1024 x 1024) += A^T (2048 x 1024).T * B (2048 x 1024)
    cublasLtMatrixLayoutCreate(&g_layout_stashed_x_tn, CUDA_R_16BF, 1024, 2048, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_d_x_tn, CUDA_R_16BF, 1024, 2048, 1024);
    cublasLtMatrixLayoutCreate(&g_layout_d_qkv_slice_tn, CUDA_R_16BF, 1024, 1024, 1024);
    g_algo_qkv_bwd_slice = autotune_matmul_algo(g_lt, g_desc_tn, g_layout_stashed_x_tn, g_layout_d_x_tn, g_layout_d_qkv_slice_tn, g_layout_d_qkv_slice_tn, pref, d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 1.0f, g_ws, g_ws_size, "QKV Bwd dSlice", 2);
    
    // I2. Phase 31: FP8 QKV Bwd dSlice: C (1024 x 1024) += A^T (2048 x 1024).T * B (2048 x 1024) in FP8
    g_algo_qkv_bwd_slice_fp8 = autotune_matmul_algo(
        g_lt, g_desc_tn_fp8, g_layout_x_2048_1024_fp8, g_layout_x_2048_1024_fp8,
        g_layout_d_qkv_slice_tn, g_layout_d_qkv_slice_tn, pref,
        d_scratchA, d_scratchB, d_scratchC, d_scratchC, 1.0f, 1.0f, g_ws, g_ws_size, "QKV Bwd dSlice FP8", 1
    );
    
    cudaFree(d_scratchA);
    cudaFree(d_scratchB);
    cudaFree(d_scratchC);
    
    cublasLtMatmulPreferenceDestroy(pref);
    g_initialized = true;
}

void cleanup_cublaslt_engine() {
    if (!g_initialized) return;
    
    cublasLtMatrixLayoutDestroy(g_layout_d_qkv_slice_tn);
    cublasLtMatrixLayoutDestroy(g_layout_d_x_tn);
    cublasLtMatrixLayoutDestroy(g_layout_stashed_x_tn);
    
    cublasLtMatrixLayoutDestroy(g_layout_d_lm_head_tn);
    cublasLtMatrixLayoutDestroy(g_layout_d_logits_tn);
    cublasLtMatrixLayoutDestroy(g_layout_final_norm_tn);
    
    cublasLtMatrixLayoutDestroy(g_layout_d_logits_tn_fp8);
    cublasLtMatrixLayoutDestroy(g_layout_d_logits_nn_fp8);
    cublasLtMatmulDescDestroy(g_desc_tn_fp8);
    cublasLtMatmulDescDestroy(g_desc_nn_fp8);
    
    cublasLtMatrixLayoutDestroy(g_layout_d_final_norm_nn);
    cublasLtMatrixLayoutDestroy(g_layout_d_logits_nn);
    cublasLtMatrixLayoutDestroy(g_layout_w_lm_head_nn);
    
    cublasLtMatrixLayoutDestroy(g_layout_logits);
    cublasLtMatrixLayoutDestroy(g_layout_w_lm_head);
    cublasLtMatrixLayoutDestroy(g_layout_x_2048_1024_fp8);
    cublasLtMatrixLayoutDestroy(g_layout_w_lm_head_fp8);
    
    cublasLtMatrixLayoutDestroy(g_layout_disp_y);
    cublasLtMatrixLayoutDestroy(g_layout_w2);
    cublasLtMatrixLayoutDestroy(g_layout_act);
    
    cublasLtMatrixLayoutDestroy(g_layout_act_fp8);
    cublasLtMatrixLayoutDestroy(g_layout_w2_fp8);
    cublasLtMatrixLayoutDestroy(g_layout_x_disp_fp8);
    cublasLtMatrixLayoutDestroy(g_layout_w1_fp8);
    cublasLtMatmulDescDestroy(g_desc_nt_fp8);
    
    cublasLtMatrixLayoutDestroy(g_layout_h1);
    cublasLtMatrixLayoutDestroy(g_layout_x_disp);
    cublasLtMatrixLayoutDestroy(g_layout_w1);
    
    cublasLtMatrixLayoutDestroy(g_layout_out_router);
    cublasLtMatrixLayoutDestroy(g_layout_w_router);
    
    cublasLtMatrixLayoutDestroy(g_layout_out_attn);
    cublasLtMatrixLayoutDestroy(g_layout_q_chunk);
    cublasLtMatrixLayoutDestroy(g_layout_w_attn_out);
    
    cublasLtMatrixLayoutDestroy(g_layout_out_qkv);
    cublasLtMatrixLayoutDestroy(g_layout_x_2048_1024);
    cublasLtMatrixLayoutDestroy(g_layout_w_qkv);
    if (g_layout_w_qkv_fp8) { cublasLtMatrixLayoutDestroy(g_layout_w_qkv_fp8); g_layout_w_qkv_fp8 = nullptr; }
    
    cublasLtMatmulDescDestroy(g_desc_tn);
    cublasLtMatmulDescDestroy(g_desc_nn);
    cublasLtMatmulDescDestroy(g_desc_nt);
    
    if (g_ws) {
        cudaFree(g_ws);
        g_ws = nullptr;
    }
    if (g_lt) {
        cublasLtDestroy(g_lt);
        g_lt = nullptr;
    }
    g_initialized = false;
}

// 1. QKV Forward
void cublaslt_gemm_qkv_fwd(
    const __nv_bfloat16* x, const __nv_bfloat16* w_qkv, __nv_bfloat16* out_qkv,
    int M, int C, cudaStream_t stream
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt, &alpha, w_qkv, g_layout_w_qkv, x, g_layout_x_2048_1024,
        &beta, out_qkv, g_layout_out_qkv, out_qkv, g_layout_out_qkv,
        &g_algo_qkv_fwd, g_ws, g_ws_size, stream
    );
}

// Phase 30: FP8 QKV Forward GEMM (CUDA_R_8F_E4M3 inputs, FP32 accumulator, BF16 output)
void cublaslt_gemm_qkv_fwd_fp8(
    const __nv_fp8_e4m3* x, const __nv_fp8_e4m3* w_qkv, __nv_bfloat16* out_qkv,
    int M, int C, int N, float alpha, cudaStream_t stream
) {
    float beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt_fp8, &alpha, w_qkv, g_layout_w_qkv_fp8, x, g_layout_x_2048_1024_fp8,
        &beta, out_qkv, g_layout_out_qkv, out_qkv, g_layout_out_qkv,
        &g_algo_qkv_fwd_fp8, g_ws, g_ws_size, stream
    );
}

// 2. Attn Out Forward
void cublaslt_gemm_attn_out_fwd(
    const __nv_bfloat16* q_chunk, const __nv_bfloat16* w_out, __nv_bfloat16* out_attn,
    int M, int C, cudaStream_t stream
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt, &alpha, w_out, g_layout_w_attn_out, q_chunk, g_layout_q_chunk,
        &beta, out_attn, g_layout_out_attn, out_attn, g_layout_out_attn,
        &g_algo_attn_out_fwd, g_ws, g_ws_size, stream
    );
}

// 3. Router Forward
void cublaslt_gemm_router_fwd(
    const __nv_bfloat16* x, const __nv_bfloat16* w_router, __nv_bfloat16* router_logits,
    int M, int C, int E, cudaStream_t stream
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt, &alpha, w_router, g_layout_w_router, x, g_layout_x_2048_1024,
        &beta, router_logits, g_layout_out_router, router_logits, g_layout_out_router,
        &g_algo_router_fwd, g_ws, g_ws_size, stream
    );
}

// 4. MoE W1 Forward
void cublaslt_gemm_moe_w1_fwd(
    const __nv_bfloat16* disp_x, const __nv_bfloat16* w1, __nv_bfloat16* h1,
    int total_dispatched, int C, int hidden_dim, cudaStream_t stream
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt, &alpha, w1, g_layout_w1, disp_x, g_layout_x_disp,
        &beta, h1, g_layout_h1, h1, g_layout_h1,
        &g_algo_w1_fwd, g_ws, g_ws_size, stream
    );
}

// 5. MoE W2 Forward
void cublaslt_gemm_moe_w2_fwd(
    const __nv_bfloat16* act, const __nv_bfloat16* w2, __nv_bfloat16* disp_y,
    int total_dispatched, int hidden_dim, int C, cudaStream_t stream
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt, &alpha, w2, g_layout_w2, act, g_layout_act,
        &beta, disp_y, g_layout_disp_y, disp_y, g_layout_disp_y,
        &g_algo_w2_fwd, g_ws, g_ws_size, stream
    );
}

// 5b. Phase 28: MoE W1 FP8 Forward
void cublaslt_gemm_moe_w1_fp8(
    const __nv_fp8_e4m3* disp_x, const __nv_fp8_e4m3* w1, __nv_bfloat16* h1,
    int total_dispatched, int C, int hidden_dim, float alpha, cudaStream_t stream
) {
    float beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt_fp8, &alpha, w1, g_layout_w1_fp8, disp_x, g_layout_x_disp_fp8,
        &beta, h1, g_layout_h1, h1, g_layout_h1,
        &g_algo_w1_fp8, g_ws, g_ws_size, stream
    );
}

// 5c. Phase 28: MoE W2 FP8 Forward
void cublaslt_gemm_moe_w2_fp8(
    const __nv_fp8_e4m3* act, const __nv_fp8_e4m3* w2, __nv_bfloat16* disp_y,
    int total_dispatched, int hidden_dim, int C, float alpha, cudaStream_t stream
) {
    float beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt_fp8, &alpha, w2, g_layout_w2_fp8, act, g_layout_act_fp8,
        &beta, disp_y, g_layout_disp_y, disp_y, g_layout_disp_y,
        &g_algo_w2_fp8, g_ws, g_ws_size, stream
    );
}

// 6. LM Head Forward
void cublaslt_gemm_lm_head_fwd(
    const __nv_bfloat16* final_norm, const __nv_bfloat16* lm_head_w, __nv_bfloat16* logits,
    int M, int C, int vocab_pad, cudaStream_t stream
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt, &alpha, lm_head_w, g_layout_w_lm_head, final_norm, g_layout_x_2048_1024,
        &beta, logits, g_layout_logits, logits, g_layout_logits,
        &g_algo_lm_head_fwd, g_ws, g_ws_size, stream
    );
}

// 6b. Phase 28: FP8 LM Head Forward
void cublaslt_gemm_lm_head_fwd_fp8(
    const __nv_fp8_e4m3* final_norm, const __nv_fp8_e4m3* lm_head_w, __nv_bfloat16* logits,
    int M, int C, int vocab_pad, float alpha, cudaStream_t stream
) {
    float beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nt_fp8, &alpha, lm_head_w, g_layout_w_lm_head_fp8, final_norm, g_layout_x_2048_1024_fp8,
        &beta, logits, g_layout_logits, logits, g_layout_logits,
        &g_algo_lm_head_fwd_fp8, g_ws, g_ws_size, stream
    );
}

// 7. LM Head Backward dX
void cublaslt_gemm_lm_head_bwd_dx(
    const __nv_bfloat16* d_logits, const __nv_bfloat16* lm_head_w, __nv_bfloat16* d_final_norm,
    int M, int vocab_pad, int C, cudaStream_t stream
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nn, &alpha, lm_head_w, g_layout_w_lm_head_nn, d_logits, g_layout_d_logits_nn,
        &beta, d_final_norm, g_layout_d_final_norm_nn, d_final_norm, g_layout_d_final_norm_nn,
        &g_algo_lm_head_bwd_dx, g_ws, g_ws_size, stream
    );
}

// 8. LM Head Backward dW
void cublaslt_gemm_lm_head_bwd_dw(
    const __nv_bfloat16* d_logits, const __nv_bfloat16* final_norm, __nv_bfloat16* d_lm_head_w,
    int M, int vocab_pad, int C, cudaStream_t stream, float beta
) {
    float alpha = 1.0f;
    cublasLtMatmul(
        g_lt, g_desc_tn, &alpha, final_norm, g_layout_final_norm_tn, d_logits, g_layout_d_logits_tn,
        &beta, d_lm_head_w, g_layout_d_lm_head_tn, d_lm_head_w, g_layout_d_lm_head_tn,
        &g_algo_lm_head_bwd_dw, g_ws, g_ws_size, stream
    );
}

// 7b. Phase 28: FP8 LM Head Backward dX
void cublaslt_gemm_lm_head_bwd_dx_fp8(
    const __nv_fp8_e4m3* d_logits, const __nv_fp8_e4m3* lm_head_w, __nv_bfloat16* d_final_norm,
    int M, int vocab_pad, int C, float alpha, cudaStream_t stream
) {
    float beta = 0.0f;
    cublasLtMatmul(
        g_lt, g_desc_nn_fp8, &alpha, lm_head_w, g_layout_w_lm_head_fp8, d_logits, g_layout_d_logits_nn_fp8,
        &beta, d_final_norm, g_layout_d_final_norm_nn, d_final_norm, g_layout_d_final_norm_nn,
        &g_algo_lm_head_bwd_dx_fp8, g_ws, g_ws_size, stream
    );
}

// 8b. Phase 28: FP8 LM Head Backward dW
void cublaslt_gemm_lm_head_bwd_dw_fp8(
    const __nv_fp8_e4m3* d_logits, const __nv_fp8_e4m3* final_norm, __nv_bfloat16* d_lm_head_w,
    int M, int vocab_pad, int C, float alpha, float beta, cudaStream_t stream
) {
    cublasLtMatmul(
        g_lt, g_desc_tn_fp8, &alpha, final_norm, g_layout_x_2048_1024_fp8, d_logits, g_layout_d_logits_tn_fp8,
        &beta, d_lm_head_w, g_layout_d_lm_head_tn, d_lm_head_w, g_layout_d_lm_head_tn,
        &g_algo_lm_head_bwd_dw_fp8, g_ws, g_ws_size, stream
    );
}

// 9. QKV Backward dSlice
void cublaslt_gemm_qkv_bwd_dw_slice(
    const __nv_bfloat16* d_x_norm, const __nv_bfloat16* stashed_x, __nv_bfloat16* d_slice,
    int M, int C, cudaStream_t stream, float beta
) {
    float alpha = 1.0f;
    cublasLtMatmul(
        g_lt, g_desc_tn, &alpha, stashed_x, g_layout_stashed_x_tn, d_x_norm, g_layout_d_x_tn,
        &beta, d_slice, g_layout_d_qkv_slice_tn, d_slice, g_layout_d_qkv_slice_tn,
        &g_algo_qkv_bwd_slice, g_ws, g_ws_size, stream
    );
}

void cublaslt_gemm_qkv_bwd_dw_slice_fp8(
    const __nv_fp8_e4m3* d_x_norm_fp8, const __nv_fp8_e4m3* stashed_x_fp8, __nv_bfloat16* d_slice,
    int M, int C, float alpha, float beta, cudaStream_t stream
) {
    cublasLtMatmul(
        g_lt, g_desc_tn_fp8, &alpha, stashed_x_fp8, g_layout_x_2048_1024_fp8,
        d_x_norm_fp8, g_layout_x_2048_1024_fp8, &beta,
        d_slice, g_layout_d_qkv_slice_tn, d_slice, g_layout_d_qkv_slice_tn,
        &g_algo_qkv_bwd_slice_fp8, g_ws, g_ws_size, stream
    );
}

// 10. Replicate QKV dW Slice 0 to Slices 1 and 2 (128-bit Vectorized)
__global__ void replicate_qkv_dw_slices_vec8_kernel(
    __nv_bfloat16* __restrict__ d_qkv_weight,
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_vecs = (C * C) / 8;
    if (idx < num_vecs) {
        const int4* src = reinterpret_cast<const int4*>(d_qkv_weight);
        int4* dst1 = reinterpret_cast<int4*>(d_qkv_weight + C * C);
        int4* dst2 = reinterpret_cast<int4*>(d_qkv_weight + 2 * C * C);
        
        int4 val = src[idx];
        dst1[idx] = val;
        dst2[idx] = val;
    }
}

void replicate_qkv_dw_slices(__nv_bfloat16* d_qkv_weight, int C, cudaStream_t stream) {
    const int BLOCK = 256;
    int num_vecs = (C * C) / 8;
    int grid = (num_vecs + BLOCK - 1) / BLOCK;
    replicate_qkv_dw_slices_vec8_kernel<<<grid, BLOCK, 0, stream>>>(d_qkv_weight, C);
}
