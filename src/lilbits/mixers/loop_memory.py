"""Loop memory: inside a looped core, each pass attends over the earlier passes of the same token.

At loop i a position's query reads the core states it had after passes 1..i-1 plus its current input, so the core
can compare its current estimate with earlier ones. The memory is per token and is thrown away once the token's
loops finish: it never enters a cache. Cost is i vectors per position at loop i.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoopMemory(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        assert d_model % n_heads == 0
        self.H, self.D = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.o.weight)  # starts as a no-op residual

    def forward(self, x: torch.Tensor, memory: list[torch.Tensor]) -> torch.Tensor:
        """x: (B, L, d) current input; memory: earlier passes' states, each (B, L, d)."""
        B, L, d = x.shape
        H, D = self.H, self.D
        m = torch.stack([*memory, x], 2)  # B, L, i, d
        k, v = self.kv(m).view(B, L, m.shape[2], 2, H, D).unbind(3)
        q = self.q(x).view(B, L, H, D)
        att = torch.einsum("blhd,blihd->blhi", q, k).div(math.sqrt(D)).softmax(-1)
        return self.o(torch.einsum("blhi,blihd->blhd", att, v).reshape(B, L, d))
