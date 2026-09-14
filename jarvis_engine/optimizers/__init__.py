"""
Jarvis Optimizers Package
=========================
Modular optimizer suite supporting AdamW control baseline and Hybrid Muon + AdamW
for post-50M foundation model pretraining experiments.
"""

import torch
import torch.nn as nn
from .muon import Muon, zeropower_via_newtonschulz5
from .parameter_groups import classify_parameter_groups
from .hybrid import HybridMuonAdamW


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """
    Factory function instantiating the requested optimizer according to CLI arguments.
    
    Default behavior is strictly AdamW with identical baseline hyperparameters.
    """
    opt_name = getattr(args, "optimizer", "adamw").lower()
    
    if opt_name == "adamw":
        max_lr = getattr(args, "max_lr", 1.5e-4)
        weight_decay = getattr(args, "weight_decay", 0.1)
        fused = getattr(args, "fused", True) and torch.cuda.is_available()
        return torch.optim.AdamW(
            model.parameters(),
            lr=max_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
            fused=fused,
        )
        
    elif opt_name in ("hybrid_muon", "muon"):
        muon_lr = getattr(args, "muon_lr", 2.0e-3)
        adamw_lr = getattr(args, "adamw_lr", 1.5e-4)
        muon_wd = getattr(args, "muon_wd", 0.05)
        adamw_wd = getattr(args, "adamw_wd", 0.10)
        muon_momentum = getattr(args, "muon_momentum", 0.95)
        
        param_groups = classify_parameter_groups(
            model=model,
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            muon_wd=muon_wd,
            adamw_wd=adamw_wd,
            muon_momentum=muon_momentum,
            adamw_betas=(0.9, 0.95),
            validate=True,
        )
        return HybridMuonAdamW(param_groups)
        
    else:
        raise ValueError(f"Unsupported optimizer: '{opt_name}'. Choose from ['adamw', 'hybrid_muon'].")


__all__ = [
    "Muon",
    "HybridMuonAdamW",
    "zeropower_via_newtonschulz5",
    "classify_parameter_groups",
    "build_optimizer",
]
