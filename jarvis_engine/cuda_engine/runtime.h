#pragma once

#include "full_engine.h"

// Memory lifecycle
FullModelWorkspace allocate_full_workspace(const FullJarvisConfig& cfg);
void free_full_workspace(FullModelWorkspace& ws);

FullModelParameters allocate_model_parameters(const FullJarvisConfig& cfg);
void free_model_parameters(FullModelParameters& params);

// CUDA Graph management
struct CUDAGraphContext {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    bool is_captured = false;
};

void capture_graph_step(
    CUDAGraphContext& ctx,
    const FullJarvisConfig& cfg,
    FullModelWorkspace& ws,
    FullModelParameters& params,
    float lr,
    cudaStream_t stream
);

void replay_graph_step(CUDAGraphContext& ctx, cudaStream_t stream);
void destroy_graph(CUDAGraphContext& ctx);
