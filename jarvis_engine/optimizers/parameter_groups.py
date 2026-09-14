"""
Parameter Group Classifier for Jarvis Hybrid Optimization
==========================================================
Deterministically partitions Jarvis parameters into:
1. Muon Group: 2D hidden linear projections (attention Q,K,V,Out and MoE W1,W2).
2. AdamW Group: 1D vectors, scalars, vocabulary embeddings, LM head, and MoE routers.

Strict Architectural Exclusions from Muon:
------------------------------------------
- tok_emb.weight: Sparse row gradients; Newton-Schulz polynomial iteration would densify
  updates across all 50,257 rows, injecting noise into unobserved vocabulary tokens.
- lm_head.weight: Output classifier head directly sets logit scales; standard LLM practice
  preserves logit calibration and temperature dynamics under AdamW.
- moe.router.weight: Extreme non-square matrix (4 x 1024, rank <= 4). Polar factor
  orthogonalization forces 4 singular values to 1.0 and zeros out 1,020 dimensions,
  destroying softmax routing entropy calibration.
- norm1, norm2, final_norm: 1D RMSNorm scale vectors. Matrix orthogonalization is undefined.
- gamma_raw: 1D recurrence decay vector (16 heads). Governs exponential memory retention.
- var_scale: 0D scalar parameter (numel=1). Governs dynamic LSF membrane leakiness.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


def classify_parameter_groups(
    model: nn.Module,
    muon_lr: float = 2.0e-3,
    adamw_lr: float = 1.5e-4,
    muon_wd: float = 0.05,
    adamw_wd: float = 0.10,
    muon_momentum: float = 0.95,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    validate: bool = True,
) -> List[Dict]:
    """
    Classifies all trainable parameters of a Jarvis model into Muon and AdamW groups.
    
    Args:
        model: PyTorch model instance (Jarvis).
        muon_lr: Base learning rate for Muon group (default: 2.0e-3).
        adamw_lr: Base learning rate for AdamW group (default: 1.5e-4).
        muon_wd: Decoupled weight decay for Muon group (default: 0.05).
        adamw_wd: Decoupled weight decay for AdamW group (default: 0.10).
        muon_momentum: Momentum for Muon group (default: 0.95).
        adamw_betas: Betas tuple for AdamW group (default: (0.9, 0.95)).
        validate: If True, execute strict structural consistency checks.
        
    Returns:
        List of two parameter group dictionaries ready for HybridMuonAdamW.
    """
    muon_params = []
    adamw_params = []
    
    muon_names = []
    adamw_names = []
    
    # Exact name patterns that MUST remain on AdamW
    ADAMW_EXCLUSIONS = (
        "tok_emb",
        "lm_head",
        "router",
        "norm",
        "gamma_raw",
        "var_scale",
    )
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1. Any non-2D tensor (scalars, 1D vectors, 3D+ tensors) MUST be AdamW
        if param.ndim != 2:
            adamw_params.append(param)
            adamw_names.append(name)
            continue
            
        # 2. Check explicit architectural exclusions
        is_excluded = any(exc in name for exc in ADAMW_EXCLUSIONS)
        
        # 3. Check rank / dimension safety (avoid degenerate matrices where min(M, N) < 32)
        is_degenerate_2d = min(param.shape) < 32
        
        if is_excluded or is_degenerate_2d:
            adamw_params.append(param)
            adamw_names.append(name)
        else:
            # 4. Eligible 2D hidden linear transformations
            muon_params.append(param)
            muon_names.append(name)
            
    # Validation checks
    if validate:
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        total_trainable_count = len(all_trainable)
        assigned_count = len(muon_params) + len(adamw_params)
        
        if assigned_count != total_trainable_count:
            raise ValueError(
                f"Parameter classification mismatch: {assigned_count} parameters assigned, "
                f"but model has {total_trainable_count} trainable parameters."
            )
            
        # Check uniqueness (no parameter in both groups)
        muon_set = set(id(p) for p in muon_params)
        adamw_set = set(id(p) for p in adamw_params)
        overlap = muon_set.intersection(adamw_set)
        if overlap:
            raise ValueError(f"Found {len(overlap)} duplicate parameters assigned to both Muon and AdamW!")
            
        # Verify numel sum matches model exactly
        total_trainable_numel = sum(p.numel() for p in all_trainable)
        classified_numel = sum(p.numel() for p in muon_params) + sum(p.numel() for p in adamw_params)
        if total_trainable_numel != classified_numel:
            raise ValueError(
                f"Trainable numel mismatch: model has {total_trainable_numel:,} elements, "
                f"but classification produced {classified_numel:,} elements."
            )
            
    param_groups = [
        {
            "name": "muon_2d_hidden",
            "params": muon_params,
            "optimizer_type": "muon",
            "lr": muon_lr,
            "base_lr": muon_lr,
            "weight_decay": muon_wd,
            "momentum": muon_momentum,
            "ns_steps": 5,
            "aspect_scale": True,
        },
        {
            "name": "adamw_1d_and_special",
            "params": adamw_params,
            "optimizer_type": "adamw",
            "lr": adamw_lr,
            "base_lr": adamw_lr,
            "weight_decay": adamw_wd,
            "betas": adamw_betas,
            "eps": 1e-8,
        },
    ]
    return param_groups
