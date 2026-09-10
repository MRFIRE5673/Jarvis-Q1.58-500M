# experiments/architecture_matrix/optimizers.py
"""
Architectural Evolution — Modern Hybrid Optimizers (Track M: Muon + AdamW)
==========================================================================
Implements Muon (Momentum Orthogonalized by Newton-Schulz) for 2D matrix weights
and fused AdamW for 1D weights (RMSNorm, biases, router weights) and embeddings.

Reference: Keller Jordan et al. (2024) "Muon: An Optimizer for Hidden Layers in Neural Networks".
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration for matrix orthogonalization.
    Normalizes singular values to 1, producing isometric updates.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)  # spectral scaling
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D weight matrices (linear projections, ternary weights).
    Applies quintic Newton-Schulz orthogonalization to momentum buffers.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                if len(p.shape) == 2:
                    ortho_update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    p.data.add_(ortho_update, alpha=-lr)
                else:
                    p.data.add_(update, alpha=-lr)


def create_hybrid_optimizer(model: nn.Module, muon_lr: float = 0.02, adamw_lr: float = 5e-5) -> Tuple[Muon, torch.optim.AdamW]:
    """
    Constructs hybrid optimizer:
    - Muon handles all 2D internal weight matrices (Q, K, V, Out projections, FFN w1, w2).
    - AdamW handles embeddings, final LM head, RMSNorm weights, and router weights.
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if 2D internal matrix
        if param.ndim == 2 and "tok_emb" not in name and "lm_head" not in name and "router" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon_opt = Muon(muon_params, lr=muon_lr)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=adamw_lr, fused=True)

    return muon_opt, adamw_opt
