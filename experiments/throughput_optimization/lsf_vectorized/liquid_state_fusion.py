# liquid_state_fusion.py
import os
import glob
import torch
import torch.nn as nn
import math

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

try:
    import liquid_state_fusion_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

# Runtime execution tracking
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


class LiquidStateFusionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, M, h0=None):
        global _CUSTOM_CUDA_EXECUTED, _FALLBACK_USED
        alpha = alpha.contiguous()
        M = M.contiguous()

        orig_dtype = alpha.dtype
        device = alpha.device

        if device.type == 'cuda' and orig_dtype == torch.float64:
            raise TypeError("Double precision (float64) is not supported on CUDA for LiquidStateFusion.")

        # Native CUDA fast path: supports both Float32 and BFloat16 without conversion
        native_cuda = _CUDA_AVAILABLE and alpha.is_cuda and orig_dtype in [torch.float32, torch.bfloat16]
        if native_cuda:
            _CUSTOM_CUDA_EXECUTED = True
            h0_in = h0.contiguous() if h0 is not None else None
            H = liquid_state_fusion_cuda.forward(alpha, M, h0_in)
            ctx.save_for_backward(alpha, M, h0_in, H)
            ctx.orig_dtype = orig_dtype
            return H

        # PyTorch associative scan fallback
        _FALLBACK_USED = True
        is_half = orig_dtype in [torch.bfloat16, torch.float16, torch.half]
        if device.type == 'cuda' and is_half:
            alpha_comp = alpha.to(torch.float32)
            M_comp = M.to(torch.float32)
            h0_comp = h0.to(torch.float32) if h0 is not None else None
        else:
            alpha_comp = alpha
            M_comp = M
            h0_comp = h0

        if h0_comp is not None:
            h0_comp = h0_comp.contiguous()

        B, T, D = alpha_comp.shape
        chunk_size = 128

        a = alpha_comp.clone()
        b = (1.0 - alpha_comp) * M_comp
        if h0_comp is not None:
            b[:, 0, :] = b[:, 0, :] + a[:, 0, :] * h0_comp

        num_chunks = int(math.ceil(T / chunk_size))

        a_chunks, b_chunks = [], []
        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, T)
            sz = end - start

            a_c = a[:, start:end, :].clone()
            b_c = b[:, start:end, :].clone()

            steps = int(math.ceil(math.log2(sz))) if sz > 1 else 0
            for step in range(steps):
                offset = 1 << step
                if offset >= sz:
                    break
                new_a = a_c[:, offset:, :] * a_c[:, :-offset, :]
                new_b = a_c[:, offset:, :] * b_c[:, :-offset, :] + b_c[:, offset:, :]
                a_c[:, offset:, :] = new_a
                b_c[:, offset:, :] = new_b

            a_chunks.append(a_c)
            b_chunks.append(b_c)

        a_scanned = torch.cat(a_chunks, dim=1)
        b_scanned = torch.cat(b_chunks, dim=1)

        block_sums_a, block_sums_b = [], []
        for c in range(num_chunks):
            end_idx = min((c + 1) * chunk_size, T) - 1
            block_sums_a.append(a_scanned[:, end_idx:end_idx + 1, :])
            block_sums_b.append(b_scanned[:, end_idx:end_idx + 1, :])

        bs_a = torch.cat(block_sums_a, dim=1)
        bs_b = torch.cat(block_sums_b, dim=1)

        steps = int(math.ceil(math.log2(num_chunks))) if num_chunks > 1 else 0
        for step in range(steps):
            offset = 1 << step
            if offset >= num_chunks:
                break
            new_bs_a = bs_a[:, offset:, :] * bs_a[:, :-offset, :]
            new_bs_b = bs_a[:, offset:, :] * bs_b[:, :-offset, :] + bs_b[:, offset:, :]
            bs_a[:, offset:, :] = new_bs_a
            bs_b[:, offset:, :] = new_bs_b

        H_comp = b_scanned.clone()
        for c in range(1, num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, T)

            pref_b = bs_b[:, c - 1:c, :]
            local_a = a_scanned[:, start:end, :]
            local_b = b_scanned[:, start:end, :]

            H_comp[:, start:end, :] = local_a * pref_b + local_b

        ctx.save_for_backward(alpha_comp, M_comp, h0_comp, H_comp)
        ctx.orig_dtype = orig_dtype
        return H_comp.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        global _CUSTOM_CUDA_EXECUTED, _FALLBACK_USED
        alpha_comp, M_comp, h0_comp, H_comp = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        orig_dtype = ctx.orig_dtype
        device = alpha_comp.device

        # Native CUDA fast path: supports both Float32 and BFloat16 directly
        native_cuda = _CUDA_AVAILABLE and alpha_comp.is_cuda and orig_dtype in [torch.float32, torch.bfloat16]
        if native_cuda:
            _CUSTOM_CUDA_EXECUTED = True
            grad_alpha, grad_M, grad_h0 = liquid_state_fusion_cuda.backward(
                alpha_comp, M_comp, h0_comp, H_comp, grad_output
            )
            return grad_alpha, grad_M, grad_h0

        # PyTorch associative scan fallback
        _FALLBACK_USED = True
        is_half = orig_dtype in [torch.bfloat16, torch.float16, torch.half]
        if device.type == 'cuda' and is_half:
            grad_output_comp = grad_output.to(torch.float32)
        else:
            grad_output_comp = grad_output

        B, T, D = alpha_comp.shape
        chunk_size = 128

        a_rev = torch.zeros_like(alpha_comp)
        a_rev[:, :-1, :] = alpha_comp[:, 1:, :]
        b_rev = grad_output_comp.clone()

        a_rev_flipped = torch.flip(a_rev, dims=[1])
        b_rev_flipped = torch.flip(b_rev, dims=[1])

        num_chunks = int(math.ceil(T / chunk_size))

        a_chunks, b_chunks = [], []
        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, T)
            sz = end - start

            a_c = a_rev_flipped[:, start:end, :].clone()
            b_c = b_rev_flipped[:, start:end, :].clone()

            steps = int(math.ceil(math.log2(sz))) if sz > 1 else 0
            for step in range(steps):
                offset = 1 << step
                if offset >= sz:
                    break
                new_a = a_c[:, offset:, :] * a_c[:, :-offset, :]
                new_b = a_c[:, offset:, :] * b_c[:, :-offset, :] + b_c[:, offset:, :]
                a_c[:, offset:, :] = new_a
                b_c[:, offset:, :] = new_b

            a_chunks.append(a_c)
            b_chunks.append(b_c)

        a_scanned = torch.cat(a_chunks, dim=1)
        b_scanned = torch.cat(b_chunks, dim=1)

        block_sums_a, block_sums_b = [], []
        for c in range(num_chunks):
            end_idx = min((c + 1) * chunk_size, T) - 1
            block_sums_a.append(a_scanned[:, end_idx:end_idx + 1, :])
            block_sums_b.append(b_scanned[:, end_idx:end_idx + 1, :])

        bs_a = torch.cat(block_sums_a, dim=1)
        bs_b = torch.cat(block_sums_b, dim=1)

        steps = int(math.ceil(math.log2(num_chunks))) if num_chunks > 1 else 0
        for step in range(steps):
            offset = 1 << step
            if offset >= num_chunks:
                break
            new_bs_a = bs_a[:, offset:, :] * bs_a[:, :-offset, :]
            new_bs_b = bs_a[:, offset:, :] * bs_b[:, :-offset, :] + bs_b[:, offset:, :]
            bs_a[:, offset:, :] = new_bs_a
            bs_b[:, offset:, :] = new_bs_b

        dH_flipped = b_scanned.clone()
        for c in range(1, num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, T)
            pref_b = bs_b[:, c - 1:c, :]
            local_a = a_scanned[:, start:end, :]
            local_b = b_scanned[:, start:end, :]
            dH_flipped[:, start:end, :] = local_a * pref_b + local_b

        dH = torch.flip(dH_flipped, dims=[1])
        grad_M_comp = dH * (1.0 - alpha_comp)

        h_prev = torch.zeros_like(dH)
        if h0_comp is not None:
            h_prev[:, 0, :] = h0_comp
        if T > 1:
            h_prev[:, 1:, :] = H_comp[:, :-1, :]

        grad_alpha_comp = dH * (h_prev - M_comp)

        grad_h0_comp = None
        if h0_comp is not None:
            grad_h0_comp = dH[:, 0, :] * alpha_comp[:, 0, :]

        grad_alpha = grad_alpha_comp.to(orig_dtype)
        grad_M = grad_M_comp.to(orig_dtype)
        grad_h0 = grad_h0_comp.to(orig_dtype) if grad_h0_comp is not None else None

        return grad_alpha, grad_M, grad_h0


class LiquidStateFusion(nn.Module):
    def __init__(self, persistent=False):
        super().__init__()
        self.persistent = persistent
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, alpha, M, h0=None):
        B, T, D = alpha.shape
        if h0 is not None:
            initial_state = h0
        elif self.persistent and self.h is not None and self.h.shape == (B, D) and self.h.device == alpha.device:
            initial_state = self.h
        else:
            initial_state = None

        H = LiquidStateFusionFunction.apply(alpha, M, initial_state)

        if self.persistent:
            self.h = H[:, -1, :].detach()
        else:
            self.h = None

        return H
