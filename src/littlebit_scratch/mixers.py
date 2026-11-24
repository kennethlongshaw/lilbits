"""Linear-attention token mixers with pluggable (FP or LittleBit) projections.

Only the d_model x d_model projections go through the `linear` factory, so they are the
ones LittleBit compresses. The small, numerically sensitive pieces -- per-head decay and
write-strength gates, short convolutions, A_log / dt_bias -- stay full precision unless
`lb_gates=True`. That mirrors how the attention softmax itself is never quantized.

GatedDeltaNet follows Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule"
(ICLR 2025), the mixer used in the Qwen3-Next / Qwen3.5 / Qwen3.8-Flash-Next hybrids and the
`fla` library's GatedDeltaNet layer:

    S_t = exp(g_t) * S_{t-1}                       (gated decay, g_t <= 0)
    S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T   (delta-rule write, ||k_t|| = 1)
    o_t = S_t^T q_t

The recurrence here is the naive per-token loop: exact and easy to read, fine for the short
sequences in this repo, slow for long ones. `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
computes the same thing with chunked Triton kernels on GPU.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

LinearFactory = Callable[[int, int], nn.Module]


class ShortConv(nn.Module):
    """Causal depthwise conv + SiLU over the time axis, (B, T, C) -> (B, T, C)."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x.transpose(1, 2), (self.kernel - 1, 0)))
        return F.silu(y).transpose(1, 2)


def gated_delta_rule(q, k, v, g, beta, initial_state=None):
    """Naive recurrent gated delta rule.

    q, k: (B, T, H, Dk), unit-norm k; v: (B, T, H, Dv); g (log decay, <= 0), beta: (B, T, H).
    Returns o (B, T, H, Dv) and the final state (B, H, Dk, Dv).
    """
    B, T, H, Dk = k.shape
    S = q.new_zeros(B, H, Dk, v.shape[-1]) if initial_state is None else initial_state
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        delta = (v[:, t] - torch.einsum("bhkv,bhk->bhv", S, kt)) * beta[:, t, :, None]
        S = S + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(out, dim=1), S


class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int, n_heads: int, linear: LinearFactory, conv_size: int = 4,
                 lb_gates: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        # Large projections: compressed when linear is LittleBit.
        self.q = linear(d_model, d_model)
        self.k = linear(d_model, d_model)
        self.v = linear(d_model, d_model)
        self.gate = linear(d_model, d_model)  # output gate
        self.o = linear(d_model, d_model)
        # Per-head scalars: d_model -> n_heads, a negligible share of the weights.
        small = linear if lb_gates else partial(nn.Linear, bias=False)
        self.a = small(d_model, n_heads)  # decay
        self.b = small(d_model, n_heads)  # write strength
        self.conv_q = ShortConv(d_model, conv_size)
        self.conv_k = ShortConv(d_model, conv_size)
        self.conv_v = ShortConv(d_model, conv_size)
        # Mamba2-style decay init: A in [1, 16], dt in [1e-3, 1e-1].
        self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.o_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, D)

        q = F.normalize(heads(self.conv_q(self.q(x))), dim=-1) * D**-0.5
        k = F.normalize(heads(self.conv_k(self.k(x))), dim=-1)
        v = heads(self.conv_v(self.v(x)))
        beta = torch.sigmoid(self.b(x))
        g = -self.A_log.exp() * F.softplus(self.a(x) + self.dt_bias)
        o, _ = gated_delta_rule(q, k, v, g, beta)
        o = self.o_norm(o) * F.silu(heads(self.gate(x)))
        return self.o(o.reshape(B, T, C))
