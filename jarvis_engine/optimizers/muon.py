"""
Muon Optimizer (Momentum Orthogonalized by Newton-Schulz)
=========================================================
Research implementation of the Muon optimizer designed for 2D hidden
linear transformations in neural networks (Keller Jordan 2024; Moonshot AI 2025).

Mathematical Formulation:
-------------------------
For a 2D weight matrix W in R^{M x N} with stochastic gradient G_t:
1. First-Moment Momentum Accumulation:
     M_t = beta * M_{t-1} + (1 - beta) * G_t
2. Polar Factor Orthogonalization via 5th-Order Newton-Schulz Iteration:
     O_t = zeropower_via_newtonschulz5(M_t, steps=K)
     where O_t approximates U @ V^T from the SVD of M_t = U @ Sigma @ V^T.
3. Aspect-Ratio Scaled Parameter Update:
     W_{t+1} = W_t * (1 - lr * weight_decay) - lr * scale(M, N) * O_t
     where scale(M, N) = 0.2 * max(1.0, sqrt(M / N))

Assumptions and Constraints:
----------------------------
- Applicable ONLY to 2D tensors (matrices). Vectors and scalars are not supported.
- Operates on continuous master parameters in FP32 or BF16.
- For non-square matrices (M != N), the smaller Gramian is computed to minimize FLOPs.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the polar factor O = U @ V^T of matrix G using a 5th-order Newton-Schulz iteration.
    
    The polynomial iteration maps singular values in (0, sqrt(3)] rapidly toward 1.0.
    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315.
    
    Args:
        G: 2D input tensor of shape (M, N).
        steps: Number of polynomial iterations (default: 5).
        eps: Epsilon for Frobenius norm normalization.
        
    Returns:
        Orthogonalized 2D tensor with singular values approximately equal to 1.0.
    """
    assert G.ndim == 2, f"Newton-Schulz requires a 2D matrix, but got tensor with shape {G.shape}"
    M, N = G.shape
    
    # Work in float32 for numerical stability during matrix multiplications
    orig_dtype = G.dtype
    X = G.float()
    
    # Normalize Frobenius norm so singular values satisfy sum(sigma_i^2) = 1.0 (sigma_max <= 1.0 < sqrt(3))
    norm = X.norm() + eps
    X = X / norm
    
    # Optimize compute by operating on the smaller Gramian matrix:
    # If M > N (tall matrix), X @ X^T is M x M (large), but X^T @ X is N x N (small).
    transposed = False
    if M > N:
        X = X.T
        transposed = True
        
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        # A is min(M, N) x min(M, N)
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if transposed:
        X = X.T
        
    return X.to(dtype=orig_dtype)


class Muon(Optimizer):
    """
    Muon Optimizer for 2D Weight Matrices.
    
    Args:
        params: Iterable of 2D parameters to optimize.
        lr: Learning rate (default: 2.0e-3 for continuous master weights).
        momentum: Momentum coefficient beta (default: 0.95).
        weight_decay: Decoupled weight decay coefficient (default: 0.05).
        ns_steps: Number of Newton-Schulz polynomial iterations (default: 5).
        aspect_scale: If True, scale updates by 0.2 * max(1.0, sqrt(M/N)) for non-square matrices.
    """
    def __init__(
        self,
        params,
        lr: float = 2.0e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.05,
        ns_steps: int = 5,
        aspect_scale: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            aspect_scale=aspect_scale,
        )
        super().__init__(params, defaults)
        
        # Verify all parameters are strictly 2-dimensional
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon strictly requires 2D matrix parameters, but found parameter with shape {p.shape} (ndim={p.ndim})"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            aspect_scale = group["aspect_scale"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                g = p.grad
                state = self.state[p]
                
                # State initialization: exactly 1 momentum buffer in FP32
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    
                state["step"] += 1
                buf = state["momentum"]
                
                # 1. Update momentum: M_t = momentum * M_{t-1} + (1 - momentum) * G_t
                buf.mul_(momentum).add_(g.float(), alpha=1.0 - momentum)
                
                # 2. Polar factor orthogonalization via Newton-Schulz
                update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                
                # 3. Calculate aspect ratio scale: alpha(M, N) = 0.2 * max(1.0, sqrt(M / N))
                scale = 1.0
                if aspect_scale:
                    M, N = p.shape
                    scale = 0.2 * max(1.0, math.sqrt(M / N))
                    
                # 4. Apply decoupled weight decay: W = W * (1 - lr * weight_decay)
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                    
                # 5. Apply orthogonal update: W = W - lr * scale * O_t
                p.add_(update.to(dtype=p.dtype), alpha=-lr * scale)
                
        return loss
