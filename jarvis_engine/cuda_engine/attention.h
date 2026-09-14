#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "full_engine.h"

// Initialize static rotary position embedding tables (cos_tab, sin_tab)
void init_attention_rotary_tables(
    FullModelWorkspace& ws,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
);

// Compute per-layer causal decay tables from gamma_raw
void compute_attention_decay_tables(
    const float* gamma_raw,
    FullModelWorkspace& ws,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
);

// Run the full native associative attention forward pipeline for one layer
void run_native_associative_attention_forward(
    FullModelWorkspace& ws,
    const LayerWeights& lay,
    int layer_idx,
    const FullJarvisConfig& cfg,
    cudaStream_t stream
);
