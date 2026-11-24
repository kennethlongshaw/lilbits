"""Causal softmax attention with pluggable linears and optional QK-norm."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

LinearFactory = Callable[[int, int], nn.Module]


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, linear: LinearFactory, qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        # DeepSeek-V4-style: RMSNorm on per-head queries and keys keeps attention logits bounded
        self.q_norm = nn.RMSNorm(d_model // n_heads) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(d_model // n_heads) if qk_norm else nn.Identity()
        self.q = linear(d_model, d_model)
        self.k = linear(d_model, d_model)
        self.v = linear(d_model, d_model)
        self.o = linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_heads

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, C // H).transpose(1, 2)

        q, k = self.q_norm(heads(self.q(x))), self.k_norm(heads(self.k(x)))
        y = F.scaled_dot_product_attention(q, k, heads(self.v(x)), is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))
