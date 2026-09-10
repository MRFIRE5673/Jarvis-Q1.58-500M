# runtime/cuda_graph_runner.py
"""
CUDA Graph Execution Engine for Jarvis Fixed-Shape Passes
=========================================================
Eliminates Python/Host launch latency for:
1. Autoregressive single-token decode steps (T=1, fixed batch)
2. Fixed sequence-length prefill blocks (T=256)

Compares:
- Eager mode execution latency vs CUDA Graph replay latency
- Measures launch overhead reduction and wall-clock speedup
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLayerGraphHarness(nn.Module):
    """Encapsulates static-shape forward pass for CUDA Graph capture."""
    def __init__(self, d_model=1024):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ffn1 = nn.Linear(d_model, d_model * 2, bias=False)
        self.ffn2 = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x):
        # Attention
        norm_x = self.norm(x)
        q = self.q(norm_x)
        k = self.k(norm_x)
        v = self.v(norm_x)
        attn = torch.matmul(q, k.transpose(-1, -2)) * 0.03125
        attn_out = self.out(torch.matmul(attn, v))
        x = x + attn_out

        # FFN
        h = F.gelu(self.ffn1(self.norm(x)))
        x = x + self.ffn2(h)
        return x

class CUDAGraphRunner:
    def __init__(self, model_fn, static_input: torch.Tensor, device="cuda"):
        self.model_fn = model_fn
        self.device = torch.device(device)
        self.static_input = static_input.clone().detach()

        # Warmup on dedicated side stream before capture
        self.stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.static_output = self.model_fn(self.static_input)
        self.stream.synchronize()

        # Graph Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.static_output = self.model_fn(self.static_input)
        self.stream.synchronize()

    def replay(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Copies new input into static input buffer and replays the captured graph."""
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output

def benchmark_cuda_graphs():
    print("=" * 80)
    print("PHASE 9 & 10: CUDA GRAPH EXECUTION & LAUNCH OVERHEAD BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 1024

    # Test 1: Single-Token Autoregressive Decode (B=1, T=1) - Highly Launch Bound
    print("--- TEST 1: AUTOREGRESSIVE DECODE STEP (B=1, T=1) ---")
    harness_decode = SingleLayerGraphHarness(d_model=d_model).to(device).eval()
    x_decode = torch.randn(1, 1, d_model, device=device)

    # Eager Mode Timing
    for _ in range(20):
        with torch.no_grad():
            _ = harness_decode(x_decode)
    torch.cuda.synchronize()

    eager_times = []
    for _ in range(500):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = harness_decode(x_decode)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eager_times.append(t1 - t0)
    avg_eager_us = (sum(eager_times) / len(eager_times)) * 1e6

    # CUDA Graph Timing
    runner = CUDAGraphRunner(harness_decode, x_decode, device=device)
    graph_times = []
    for _ in range(500):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = runner.replay(x_decode)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        graph_times.append(t1 - t0)
    avg_graph_us = (sum(graph_times) / len(graph_times)) * 1e6

    speedup_decode = avg_eager_us / avg_graph_us
    print(f"Eager Mode Latency:      {avg_eager_us:8.2f} us ({1e6/avg_eager_us:,.0f} launches/s)")
    print(f"CUDA Graph Latency:      {avg_graph_us:8.2f} us ({1e6/avg_graph_us:,.0f} launches/s)")
    print(f"Launch Latency Speedup:  {speedup_decode:8.2f}x faster")

    # Test 2: Prefill Block (B=1, T=256) - Compute Bound
    print("\n--- TEST 2: PREFILL BLOCK (B=1, T=256) ---")
    x_prefill = torch.randn(1, 256, d_model, device=device)
    harness_prefill = SingleLayerGraphHarness(d_model=d_model).to(device).eval()

    for _ in range(10):
        with torch.no_grad():
            _ = harness_prefill(x_prefill)
    torch.cuda.synchronize()

    eager_p_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = harness_prefill(x_prefill)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eager_p_times.append(t1 - t0)
    avg_eager_p_us = (sum(eager_p_times) / len(eager_p_times)) * 1e6

    runner_prefill = CUDAGraphRunner(harness_prefill, x_prefill, device=device)
    graph_p_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = runner_prefill.replay(x_prefill)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        graph_p_times.append(t1 - t0)
    avg_graph_p_us = (sum(graph_p_times) / len(graph_p_times)) * 1e6

    speedup_prefill = avg_eager_p_us / avg_graph_p_us
    print(f"Eager Mode Latency:      {avg_eager_p_us:8.2f} us")
    print(f"CUDA Graph Latency:      {avg_graph_p_us:8.2f} us")
    print(f"Prefill Speedup:         {speedup_prefill:8.2f}x faster")
    print("=" * 80)

if __name__ == '__main__':
    benchmark_cuda_graphs()
