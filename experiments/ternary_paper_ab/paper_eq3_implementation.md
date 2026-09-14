# JARVIS — PAPER EQ. 3 IMPLEMENTATION DOCUMENTATION
## Mathematical Specification, Forward Quantization, & Straight-Through Estimators

**Target Workload**: Jarvis-Q1.58-500M ($d_{\text{model}}=1024$, 24 Layers, 4 MoE Experts, Top-2 Routing)  
**Comparison**: Current Production AbsMean Baseline vs Literal Paper Equations (Eq. 3 & Eq. 5)

---

## 1. Mathematical Specification

### A. Current Production Baseline (AbsMean / BitNet Convention)
The current production engine implements an adaptive AbsMean scale factor:

1. **Scale Factor Estimation**:
   $$\alpha = \frac{1}{N} \sum_{i=1}^N |W_i|$$
2. **Normalized Clamping & Rounding**:
   $$W_{\text{norm}} = \frac{W}{\alpha}$$
   $$\widetilde{W} = \text{round}\left(\text{clamp}(W_{\text{norm}}, -1.0, 1.0)\right) \times \alpha$$
   Resulting values: $\{-\alpha, 0, +\alpha\}$.
3. **Straight-Through Estimator (STE)**:
   $$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \widetilde{W}} \cdot \mathbf{1}_{\{|W / \alpha| \le 1.0\}}$$

---

### B. Paper-Faithful Implementation (Literal Paper Eq. 3 & Eq. 5)
The original Jarvis research paper specifies:

1. **Section 3.2, Equation (3)**:
   $$\widetilde{W} = \text{round}\left(\text{clamp}(W_{\text{FP32}}, -1, 1)\right)$$
   Values: $\{-1, 0, +1\}$. Strictly **no $\alpha$ multiplier**, no group scaling, no activation scaling.
2. **Section 4.1, Equation (5) — Straight-Through Estimator**:
   $$\frac{\partial \mathcal{L}}{\partial W_{\text{FP32}}} \approx \frac{\partial \mathcal{L}}{\partial \widetilde{W}} \cdot \mathbf{1}_{\{|W_{\text{FP32}}| \le 1\}}$$
   The gradient passes through unscaled if $|W_{\text{FP32}}| \le 1.0$ and is clipped to zero if $|W_{\text{FP32}}| > 1.0$.

---

## 2. PyTorch Autograd Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbsMeanTernarySTE(torch.autograd.Function):
    """Experiment A: Current production AbsMean baseline."""
    @staticmethod
    def forward(ctx, w):
        alpha = w.abs().mean().clamp(min=1e-8)
        w_norm = w / alpha
        w_q = torch.round(torch.clamp(w_norm, -1.0, 1.0)) * alpha
        ctx.save_for_backward(w, alpha)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        w, alpha = ctx.saved_tensors
        mask = ((w / alpha).abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask


class PaperEq3TernarySTE(torch.autograd.Function):
    """Experiment B: Literal Paper Eq. 3 & Eq. 5 implementation."""
    @staticmethod
    def forward(ctx, w):
        # Literal Eq. 3: W_f = round(clamp(W_FP32, -1, 1))
        # Strictly values in {-1, 0, +1}, NO alpha scaling
        w_q = torch.round(torch.clamp(w, -1.0, 1.0))
        ctx.save_for_backward(w)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        # Literal Eq. 5: dL/dW_FP32 = dL/dW_q * 1{|W_FP32| <= 1}
        mask = (w.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask
```

---

## 3. CUDA Kernel Realization

In native CUDA C++, the respective forward kernels are implemented as follows:

```cuda
// Kernel A: Current AbsMean
__global__ void ternary_absmean_fwd_kernel(
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ w_q,
    const float* __restrict__ alpha_ptr,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float alpha = fmaxf(*alpha_ptr, 1e-8f);
    float val = __bfloat162float(w[idx]);
    float clamped = fminf(fmaxf(val / alpha, -1.0f), 1.0f);
    w_q[idx] = __float2bfloat16(nearbyintf(clamped) * alpha);
}

// Kernel B: Literal Paper Eq. 3
__global__ void ternary_paper_eq3_fwd_kernel(
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ w_q,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = __bfloat162float(w[idx]);
    // Clamp directly to [-1, 1] without scaling
    float clamped = fminf(fmaxf(val, -1.0f), 1.0f);
    // Round to nearest integer {-1.0, 0.0, +1.0}
    w_q[idx] = __float2bfloat16(nearbyintf(clamped));
}
```

---

## 4. Key Implementation Differences

| Property | Current AbsMean (A) | Literal Paper Eq. 3 (B) |
| :--- | :---: | :---: |
| **Quantization Domain** | $\{-\alpha, 0, +\alpha\}$ | $\{-1, 0, +1\}$ |
| **Adaptive Scale $\alpha$** | Yes ($\alpha = \text{mean}(\|W\|)$) | **None** |
| **STE Mask Condition** | $\|W / \alpha\| \le 1.0$ | $\|W\| \le 1.0$ |
| **Active Range** | Effective for any scale $\sigma$ | Requires $\|W\| \ge 0.5$ for non-zero weights |
| **Paper Fidelity** | Engineering/BitNet-style variant | **Literal Paper Eq. 3 & Eq. 5** |
