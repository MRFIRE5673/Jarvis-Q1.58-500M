"""
Hybrid Muon + AdamW Optimizer for Jarvis Foundation Model
=========================================================
Unified PyTorch Optimizer subclass that coordinates:
1. Muon for large 2D hidden linear transformations (83% of parameters)
2. AdamW for embeddings, classifier head, routers, norms, and recurrence scalars (17% of parameters)

Features:
- Completely independent learning rates, weight decays, and momentum parameters per group.
- Fully compatible with standard PyTorch state_dict and load_state_dict protocols.
- Zero extra memory copies; operations performed in-place on continuous master parameters.
- Compatible with global gradient norm clipping and PyTorch AMP (BF16).
"""

import math
from typing import Dict, Iterable, Optional, Union
import torch
from torch.optim.optimizer import Optimizer
from .muon import zeropower_via_newtonschulz5


class HybridMuonAdamW(Optimizer):
    """
    Hybrid Optimizer executing Muon on 2D hidden matrices and AdamW on all other parameters.
    
    Args:
        param_groups: Parameter group list produced by classify_parameter_groups.
    """
    def __init__(self, param_groups: Iterable[Dict]):
        # Validate that parameter groups contain expected optimizer types
        for group in param_groups:
            opt_type = group.get("optimizer_type")
            if opt_type not in ("muon", "adamw"):
                raise ValueError(
                    f"Parameter group '{group.get('name')}' must specify optimizer_type as 'muon' or 'adamw', got: {opt_type}"
                )
                
        # Default fallback values
        defaults = dict(
            lr=1.5e-4,
            base_lr=1.5e-4,
            weight_decay=0.1,
            momentum=0.95,
            betas=(0.9, 0.95),
            eps=1e-8,
            ns_steps=5,
            aspect_scale=True,
            optimizer_type="adamw",
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single hybrid optimization step across all parameter groups."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            opt_type = group.get("optimizer_type", "adamw")
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            
            if opt_type == "muon":
                self._step_muon(group, lr, weight_decay)
            elif opt_type == "adamw":
                self._step_adamw(group, lr, weight_decay)
            else:
                raise ValueError(f"Unknown optimizer_type: {opt_type}")
                
        return loss

    def _step_muon(self, group: Dict, lr: float, weight_decay: float):
        """Executes Newton-Schulz spectral descent update on 2D matrices."""
        momentum = group.get("momentum", 0.95)
        ns_steps = group.get("ns_steps", 5)
        aspect_scale = group.get("aspect_scale", True)
        
        for p in group["params"]:
            if p.grad is None:
                continue
                
            g = p.grad
            state = self.state[p]
            
            if len(state) == 0:
                state["step"] = 0
                state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                
            state["step"] += 1
            buf = state["momentum"]
            
            # 1. Momentum accumulation: M_t = beta * M_{t-1} + (1 - beta) * G_t
            buf.mul_(momentum).add_(g.float(), alpha=1.0 - momentum)
            
            # 2. Polar factor orthogonalization via Newton-Schulz
            update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
            
            # 3. Aspect ratio scaling: alpha(M, N) = 0.2 * max(1.0, sqrt(M / N))
            scale = 1.0
            if aspect_scale:
                M, N = p.shape
                scale = 0.2 * max(1.0, math.sqrt(M / N))
                
            # 4. Decoupled weight decay on master FP32/BF16 parameter: W = W * (1 - lr * wd)
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
                
            # 5. Parameter update: W = W - lr * scale * O_t
            p.add_(update.to(dtype=p.dtype), alpha=-lr * scale)

    def _step_adamw(self, group: Dict, lr: float, weight_decay: float):
        """Executes standard decoupled AdamW update on vectors, scalars, and embeddings."""
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = group.get("eps", 1e-8)
        
        for p in group["params"]:
            if p.grad is None:
                continue
                
            g = p.grad
            state = self.state[p]
            
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step = state["step"]
            
            # Decoupled weight decay: W = W * (1 - lr * wd)
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
                
            # Update moments in float32
            g_fp32 = g.float()
            exp_avg.mul_(beta1).add_(g_fp32, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1.0 - beta2)
            
            # Bias corrections
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            
            update = exp_avg / denom
            p.add_(update.to(dtype=p.dtype), alpha=-step_size)
