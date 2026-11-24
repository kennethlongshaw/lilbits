"""DeepSeek-V4-style Muon.

Differences from torch.optim.Muon (PyTorch's version follows Keller Jordan's original):
- 10 Newton-Schulz steps in two phases: 8 with the fast quintic (3.4445, -4.7750, 2.0315),
  then 2 with (2, -1.5, 0.5), which pins singular values at 1 instead of leaving them ~0.7-1.2.
- Update RMS matched to AdamW at 0.18 (scale 0.18 * sqrt(max(n, m))), so AdamW lrs carry over.
- Nesterov momentum 0.95, decoupled weight decay W <- W (1 - lr*wd).
Newton-Schulz runs in bf16 on CUDA (as DeepSeek does) and fp32 elsewhere.
"""

from __future__ import annotations

import math

import torch

FAST = (3.4445, -4.7750, 2.0315)
STABLE = (2.0, -1.5, 0.5)


@torch.no_grad()
def hybrid_newton_schulz(
    G: torch.Tensor, fast_steps: int = 8, stable_steps: int = 2, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the orthogonal polar factor U V^T of G (2D)."""
    dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.t()
    X = X / (X.norm() + eps)
    for i in range(fast_steps + stable_steps):
        a, b, c = FAST if i < fast_steps else STABLE
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X.to(G.dtype)


class DeepSeekMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        update_rms: float = 0.18,
        nesterov: bool = True,
        fast_steps: int = 8,
        stable_steps: int = 2,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            update_rms=update_rms,
            nesterov=nesterov,
            fast_steps=fast_steps,
            stable_steps=stable_steps,
        )
        super().__init__(params, defaults)
        for g in self.param_groups:
            for p in g["params"]:
                if p.dim() != 2:
                    raise ValueError("DeepSeekMuon only handles 2D parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(g["momentum"]).add_(p.grad)
                update = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                O = hybrid_newton_schulz(update, g["fast_steps"], g["stable_steps"])
                scale = g["update_rms"] * math.sqrt(max(p.shape))
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(O, alpha=-g["lr"] * scale)
        return loss
