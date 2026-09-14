"""
Unit Tests for Jarvis Optimizer Infrastructure (CPU Only)
=========================================================
Tests Muon, Parameter Group Classification, HybridMuonAdamW, Ternary STE compatibility,
Scheduler scaling, and Checkpoint serialization strictly on CPU to guarantee zero
GPU memory allocation or interference with the live training run.
"""

import math
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root and jarvis_engine are in path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
for p in [WORKSPACE_ROOT, JARVIS_ENGINE]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jarvis_engine.optimizers import (
    Muon,
    HybridMuonAdamW,
    zeropower_via_newtonschulz5,
    classify_parameter_groups,
    build_optimizer,
)
from jarvis_engine.utils.ternary_ops import TernaryLinear, TernaryQuantizeSTE


class TestOptimizerInfrastructure(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_newton_schulz_shapes(self):
        """Test Newton-Schulz polar factor orthogonalization on square, tall, and wide matrices."""
        shapes = [(64, 64), (128, 64), (64, 128)]
        for M, N in shapes:
            G = torch.randn(M, N, dtype=torch.float32)
            O = zeropower_via_newtonschulz5(G, steps=5)
            
            # 1. Output shape must match input shape
            self.assertEqual(O.shape, (M, N))
            # 2. No NaNs or Infs
            self.assertFalse(torch.isnan(O).any())
            self.assertFalse(torch.isinf(O).any())
            
            # 3. Check polar factor property: singular values should contract tightly toward 1.0
            U, S, Vh = torch.linalg.svd(O, full_matrices=False)
            # Input random matrix typically has condition number > 30.
            # 5-step Newton-Schulz contracts singular values to [0.65, 1.20] with condition number < 2.0.
            self.assertGreater(S.min().item(), 0.60)
            self.assertLess(S.max().item(), 1.25)
            self.assertLess((S.max() / S.min()).item(), 2.0)

    def test_muon_step_cpu(self):
        """Test that pure Muon optimizer updates parameters and tracks momentum state."""
        w = nn.Parameter(torch.randn(32, 32, dtype=torch.float32))
        opt = Muon([w], lr=2.0e-3, momentum=0.95, weight_decay=0.05)
        
        w_orig = w.clone()
        x = torch.randn(4, 32)
        loss = (x @ w).sum()
        loss.backward()
        
        opt.step()
        
        # Verify parameter changed
        self.assertFalse(torch.equal(w, w_orig))
        self.assertFalse(torch.isnan(w).any())
        self.assertFalse(torch.isinf(w).any())
        
        # Verify state contains step and momentum
        state = opt.state[w]
        self.assertEqual(state["step"], 1)
        self.assertIn("momentum", state)
        self.assertEqual(state["momentum"].shape, (32, 32))

    def test_parameter_classification_synthetic(self):
        """Test deterministic parameter classification on a synthetic model with all parameter types."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb = nn.Embedding(100, 64)
                self.lm_head = nn.Linear(64, 100, bias=False)
                self.router = nn.Linear(64, 4, bias=False)
                self.norm = nn.LayerNorm(64)
                self.gamma_raw = nn.Parameter(torch.full((8,), 2.94))
                self.var_scale = nn.Parameter(torch.tensor(1.0))
                # Eligible Muon parameters:
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.w1 = nn.Linear(64, 128, bias=False)
                self.w2 = nn.Linear(128, 64, bias=False)

        model = MockModel()
        groups = classify_parameter_groups(
            model,
            muon_lr=2.0e-3,
            adamw_lr=1.5e-4,
            validate=True
        )
        
        self.assertEqual(len(groups), 2)
        muon_group = groups[0]
        adamw_group = groups[1]
        
        self.assertEqual(muon_group["name"], "muon_2d_hidden")
        self.assertEqual(adamw_group["name"], "adamw_1d_and_special")
        
        # Muon group must contain exactly 3 parameters: q_proj, w1, w2
        self.assertEqual(len(muon_group["params"]), 3)
        # AdamW group must contain exactly 7 parameters: tok_emb, lm_head, router, norm.weight, norm.bias, gamma_raw, var_scale
        self.assertEqual(len(adamw_group["params"]), 7)
        
        # Check independent base LRs
        self.assertEqual(muon_group["base_lr"], 2.0e-3)
        self.assertEqual(adamw_group["base_lr"], 1.5e-4)

    def test_ternary_ste_compatibility_cpu(self):
        """Verify that Muon operates on FP32 master weights while forward produces valid ternary values."""
        layer = TernaryLinear(64, 64, bias=False)
        
        # Master weights are continuous
        self.assertEqual(layer.weight.dtype, torch.float32)
        
        # Forward pass quantizes to {-alpha, 0, +alpha}
        x = torch.randn(2, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 64))
        
        # Calculate loss and backward
        loss = out.sum()
        loss.backward()
        
        # Check that gradient was computed
        self.assertIsNotNone(layer.weight.grad)
        
        # Run Muon update
        opt = Muon([layer.weight], lr=2.0e-3, momentum=0.95, weight_decay=0.05)
        w_before = layer.weight.clone()
        opt.step()
        w_after = layer.weight.clone()
        
        # Master weight must have updated continuously
        self.assertFalse(torch.equal(w_before, w_after))
        
        # Subsequent forward must re-quantize dynamically with valid output
        out2 = layer(x)
        self.assertFalse(torch.isnan(out2).any())
        self.assertFalse(torch.isinf(out2).any())

    def test_hybrid_muon_adamw_step_cpu(self):
        """Verify that HybridMuonAdamW executes updates across both groups with independent parameters."""
        w_2d = nn.Parameter(torch.randn(32, 32))
        w_1d = nn.Parameter(torch.randn(32))
        
        groups = [
            {
                "name": "muon_group",
                "params": [w_2d],
                "optimizer_type": "muon",
                "lr": 2.0e-3,
                "base_lr": 2.0e-3,
                "weight_decay": 0.05,
                "momentum": 0.95,
                "ns_steps": 5,
                "aspect_scale": True,
            },
            {
                "name": "adamw_group",
                "params": [w_1d],
                "optimizer_type": "adamw",
                "lr": 1.5e-4,
                "base_lr": 1.5e-4,
                "weight_decay": 0.10,
                "betas": (0.9, 0.95),
            }
        ]
        
        opt = HybridMuonAdamW(groups)
        
        loss = (w_2d.sum() + w_1d.sum())
        loss.backward()
        
        opt.step()
        
        # Verify states were initialized
        self.assertIn("momentum", opt.state[w_2d])
        self.assertIn("exp_avg", opt.state[w_1d])
        self.assertIn("exp_avg_sq", opt.state[w_1d])
        
        # Verify no NaNs
        self.assertFalse(torch.isnan(w_2d).any())
        self.assertFalse(torch.isnan(w_1d).any())

    def test_scheduler_proportional_scaling_cpu(self):
        """Verify that proportional learning rate scaling preserves distinct base LRs."""
        p1 = nn.Parameter(torch.randn(10, 10))
        p2 = nn.Parameter(torch.randn(10))
        
        groups = [
            {"params": [p1], "optimizer_type": "muon", "base_lr": 2.0e-3, "lr": 2.0e-3},
            {"params": [p2], "optimizer_type": "adamw", "base_lr": 1.5e-4, "lr": 1.5e-4},
        ]
        opt = HybridMuonAdamW(groups)
        
        # Simulate 50% decay (decay_mult = 0.5)
        decay_mult = 0.5
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * decay_mult
            
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1.0e-3)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 0.75e-4)

    def test_regression_adamw_default(self):
        """Verify that build_optimizer with default options returns exact standard AdamW."""
        class Args:
            optimizer = "adamw"
            max_lr = 1.5e-4
            weight_decay = 0.1
            fused = False
            
        m = nn.Linear(10, 10)
        opt = build_optimizer(m, Args())
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual(len(opt.param_groups), 1)
        self.assertEqual(opt.param_groups[0]["lr"], 1.5e-4)

    def test_checkpoint_serialization_and_model_only_init(self):
        """Verify state_dict serialization and model-only initialization semantics."""
        w1 = nn.Parameter(torch.randn(16, 16))
        w2 = nn.Parameter(torch.randn(16))
        
        groups = [
            {"params": [w1], "optimizer_type": "muon", "base_lr": 2.0e-3, "lr": 2.0e-3, "weight_decay": 0.05, "momentum": 0.95, "ns_steps": 5, "aspect_scale": True},
            {"params": [w2], "optimizer_type": "adamw", "base_lr": 1.5e-4, "lr": 1.5e-4, "weight_decay": 0.10, "betas": (0.9, 0.95), "eps": 1e-8},
        ]
        opt = HybridMuonAdamW(groups)
        
        # Step once to populate momentum states
        loss = (w1.sum() + w2.sum())
        loss.backward()
        opt.step()
        
        # Serialize state_dict
        sd = opt.state_dict()
        self.assertIn("state", sd)
        self.assertIn("param_groups", sd)
        self.assertEqual(len(sd["param_groups"]), 2)
        
        # Create fresh optimizer with different initial parameters
        w1_new = nn.Parameter(torch.randn(16, 16))
        w2_new = nn.Parameter(torch.randn(16))
        new_groups = [
            {"params": [w1_new], "optimizer_type": "muon", "base_lr": 2.0e-3, "lr": 2.0e-3, "weight_decay": 0.05, "momentum": 0.95, "ns_steps": 5, "aspect_scale": True},
            {"params": [w2_new], "optimizer_type": "adamw", "base_lr": 1.5e-4, "lr": 1.5e-4, "weight_decay": 0.10, "betas": (0.9, 0.95), "eps": 1e-8},
        ]
        opt_new = HybridMuonAdamW(new_groups)
        
        # Load state dict
        opt_new.load_state_dict(sd)
        self.assertEqual(len(opt_new.state[w1_new]), 2)  # step and momentum
        self.assertEqual(len(opt_new.state[w2_new]), 3)  # step, exp_avg, exp_avg_sq


if __name__ == "__main__":
    unittest.main(verbosity=2)
