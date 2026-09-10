# sparse_model.py
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

# On Windows (Python 3.8+), native extensions require CUDA bin in DLL directory
if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        cuda_candidates = sorted(
            glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"),
            reverse=True
        )
        if cuda_candidates:
            cuda_home = cuda_candidates[0]
    if cuda_home:
        cuda_bin = os.path.join(cuda_home, "bin")
        if os.path.exists(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
            except Exception:
                pass

# Add current directory to path for local import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import sparse_model_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

_CUSTOM_CUDA_EXECUTED = False
_FALLBACK_USED = False

def reset_diagnostics():
    global _CUSTOM_CUDA_EXECUTED, _FALLBACK_USED
    _CUSTOM_CUDA_EXECUTED = False
    _FALLBACK_USED = False

def get_diagnostics():
    return {
        "extension_imported": _CUDA_AVAILABLE,
        "custom_kernel_executed": _CUSTOM_CUDA_EXECUTED,
        "fallback_used": _FALLBACK_USED,
    }


class MoeDispatchGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, gather_map, scatter_map, N):
        global _CUSTOM_CUDA_EXECUTED
        _CUSTOM_CUDA_EXECUTED = True
        dispatched_x = sparse_model_cuda.dispatch_gather(x_flat.contiguous(), gather_map.contiguous())
        ctx.save_for_backward(scatter_map)
        ctx.N = N
        return dispatched_x

    @staticmethod
    def backward(ctx, grad_dispatched_x):
        scatter_map, = ctx.saved_tensors
        grad_x = sparse_model_cuda.gather_backward(grad_dispatched_x.contiguous(), scatter_map, ctx.N)
        return grad_x, None, None, None


class MoeScatterCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dispatched_y, topk_gates, scatter_map, gather_map, gate_idx_map):
        global _CUSTOM_CUDA_EXECUTED
        _CUSTOM_CUDA_EXECUTED = True
        out = sparse_model_cuda.scatter_combine(dispatched_y.contiguous(), topk_gates.contiguous(), scatter_map.contiguous())
        ctx.save_for_backward(dispatched_y, topk_gates, gather_map, gate_idx_map, scatter_map)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        dispatched_y, topk_gates, gather_map, gate_idx_map, scatter_map = ctx.saved_tensors
        grad_dispatched_y, grad_topk_gates = sparse_model_cuda.scatter_backward(
            grad_out.contiguous(), dispatched_y, topk_gates, gather_map, gate_idx_map, scatter_map
        )
        return grad_dispatched_y, grad_topk_gates, None, None, None


class CUDASparseMoELayer(nn.Module):
    """
    Experimental Drop-in Replacement for SparseMoELayer using high-performance
    fused CUDA dispatch, gather, and scatter kernels on sm_120.

    Preserves 100% exact mathematical routing, gating, load-balancing loss,
    and parameter gradients.
    """
    def __init__(self, d_model, num_experts=4, top_k=2, hidden_mult=2,
                 noise_std=0.1, balance_alpha=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.balance_alpha = balance_alpha
        self.router = nn.Linear(d_model, num_experts, bias=False)
        hidden = d_model * hidden_mult

        try:
            from utils.ternary_ops import TernaryLinear
        except ImportError:
            try:
                from jarvis_engine.utils.ternary_ops import TernaryLinear
            except ImportError:
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jarvis_engine")))
                from utils.ternary_ops import TernaryLinear
        self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

    def forward(self, x):
        global _FALLBACK_USED
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        # 1. Router logits + Gaussian noise
        logits = self.router(x_flat)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_gates = (topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)).to(dtype=x_flat.dtype)

        # Fallback to pure PyTorch if CUDA extension is unavailable or tensor is on CPU
        if not _CUDA_AVAILABLE or not x.is_cuda:
            _FALLBACK_USED = True
            flat_idx  = topk_idx.reshape(-1)
            flat_gate = topk_gates.reshape(-1)
            flat_x    = x_flat.repeat_interleave(self.top_k, dim=0)
            flat_out  = torch.zeros_like(flat_x)

            for e in range(self.num_experts):
                mask = (flat_idx == e)
                xe = flat_x[mask]
                ye = self.w2[e](F.gelu(self.w1[e](xe)))
                flat_out[mask] = flat_gate[mask].unsqueeze(-1) * ye

            out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)
        else:
            # 2. Compute dispatch metadata (counts, offsets, scatter/gather maps)
            expert_counts, expert_offsets, scatter_map, gather_map, gate_idx_map = (
                sparse_model_cuda.compute_metadata(topk_idx, self.num_experts)
            )

            # 3. Fused dispatch gather: permutes x_flat into contiguous expert partitions
            dispatched_x = MoeDispatchGather.apply(x_flat, gather_map, scatter_map, N)

            # 4. Expert forward computation on contiguous slices
            dispatched_y = torch.empty_like(dispatched_x)
            offsets_cpu = expert_offsets.cpu().numpy()

            for e in range(self.num_experts):
                s = int(offsets_cpu[e])
                e_end = int(offsets_cpu[e + 1])
                if e_end > s:
                    xe = dispatched_x[s:e_end]
                    dispatched_y[s:e_end] = self.w2[e](F.gelu(self.w1[e](xe)))

            # 5. Fused scatter combine & gate weighting
            out_flat = MoeScatterCombine.apply(dispatched_y, topk_gates, scatter_map, gather_map, gate_idx_map)
            out = out_flat.view(B, T, C)

        # 6. Load-balancing loss (Eq. 6)
        top1_idx = topk_idx[:, 0]
        f = F.one_hot(top1_idx, num_classes=self.num_experts).float().mean(dim=0)
        P = probs.mean(dim=0)
        l_balance = self.balance_alpha * self.num_experts * (f * P).sum()

        act_mean = out.mean()
        act_var  = out.var()
        return out, l_balance, act_mean, act_var
