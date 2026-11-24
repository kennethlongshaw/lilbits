"""Feed-forward blocks with pluggable linears."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

LinearFactory = Callable[[int, int], nn.Module]


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, linear: LinearFactory):
        super().__init__()
        self.gate = linear(d_model, hidden)
        self.up = linear(d_model, hidden)
        self.down = linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
