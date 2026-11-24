"""Shared weights with per-loop FP scales, for looped (recurrent-depth) cores.

A looped core reuses one set of weights N times. `LoopScaledLinear` keeps that one weight and adds a small FP
scale set per loop, so each pass can behave differently at almost no storage cost:

  * over a LittleBit layer, the per-loop scales are (h_i, g_i, l_i), applied on top of each path's own
    scales:  y = sum_p ((x * g_p * g_i) @ S_V,p * l_p * l_i) @ S_U,p^T * h_p * h_i
    The 1-bit signs S_U, S_V, the large part, are stored once.
  * over any other linear (FP, BitNet), the per-loop scales are (h_i, g_i):  y = base(x * g_i) * h_i

Scales start at 1, so a freshly wrapped layer computes exactly the base layer. Select the loop with
`set_loop(module, i)` before each pass.
"""

from __future__ import annotations

import torch
from torch import nn

from lilbits.layers.littlebit import LittleBitLinear, binarize, bits_per_weight


class LoopScaledLinear(nn.Module):
    def __init__(self, base: nn.Module, n_loops: int):
        super().__init__()
        self.base = base
        self.n_loops = n_loops
        self.in_features: int = getattr(base, "in_features")  # noqa: B009
        self.out_features: int = getattr(base, "out_features")  # noqa: B009
        self.loop = 0
        self.g = nn.Parameter(torch.ones(n_loops, self.in_features))
        self.h = nn.Parameter(torch.ones(n_loops, self.out_features))
        self.l = nn.Parameter(torch.ones(n_loops, base.rank)) if isinstance(base, LittleBitLinear) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i = self.loop
        base = self.base
        if self.l is None or not isinstance(base, LittleBitLinear):
            return base(x * self.g[i]) * self.h[i]
        x = x * self.g[i]
        y = None
        for p in base.paths:
            h, g, l = p.scales(base.scale_bits)
            z = ((x * g) @ binarize(p.V, base.ste, base.smooth_k)) * (l * self.l[i])
            z = (z @ binarize(p.U, base.ste, base.smooth_k).t()) * h
            y = z if y is None else y + z
        assert y is not None
        y = y * self.h[i]
        return y if base.bias is None else y + base.bias * self.h[i]

    def storage_bits(self) -> float:
        """Base storage + 16-bit per-loop scales."""
        base_bits = (
            bits_per_weight(
                self.in_features, self.out_features, self.base.rank, len(self.base.paths), self.base.scale_bits
            )
            * self.in_features
            * self.out_features
            if isinstance(self.base, LittleBitLinear)
            else 16 * self.in_features * self.out_features
        )
        return base_bits + 16 * sum(p.numel() for p in (self.g, self.h, self.l) if p is not None)


def set_loop(module: nn.Module, i: int) -> None:
    for m in module.modules():
        if isinstance(m, LoopScaledLinear):
            m.loop = i


def loop_linear_factory(base_factory, n_loops: int):
    """Wrap a linear factory so every linear it builds gets per-loop scales."""
    return lambda d_in, d_out: LoopScaledLinear(base_factory(d_in, d_out), n_loops)


@torch.no_grad()
def convert_loop_linears(module: nn.Module, make_littlebit, itq_iters: int = 0) -> None:
    """In place: each LoopScaledLinear over an FP nn.Linear gets a LittleBit base initialized from that weight
    (Dual-SVID). Per-loop g and h are kept; a per-loop l (ones) is added. `make_littlebit(d_in, d_out)` builds
    the empty LittleBitLinear."""
    for m in module.modules():
        if isinstance(m, LoopScaledLinear) and isinstance(m.base, nn.Linear):
            lb = make_littlebit(m.in_features, m.out_features).to(m.g.device)
            lb.init_from_weight(m.base.weight, itq_iters)
            if m.base.bias is not None and lb.bias is not None:
                lb.bias.copy_(m.base.bias)
            m.base = lb
            m.l = nn.Parameter(torch.ones(m.n_loops, lb.rank, device=m.g.device))
