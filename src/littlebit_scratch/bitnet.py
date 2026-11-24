"""BitNet b1.58 linear ("The Era of 1-bit LLMs", Ma et al. 2024): ternary weights, 8-bit activations.

Training keeps a latent FP weight W. The forward pass uses
    W_q = round_clip(W / mean|W|, -1, 1) * mean|W|           (absmean, one scale per matrix)
    x_q = round_clip(x * 127 / max|x|, -128, 127) * max|x| / 127   (absmax, per token)
with straight-through gradients for both. As in the public b1.58 reproduction and Microsoft's
training notes, the input is RMS-normalized (no learned gain) inside the layer first.

Storage is log2(3) = 1.585 bits per weight plus one FP16 scale per matrix. That's the
information-theoretic floor; a practical packing (5 trits per byte) is 1.6 bits.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

TERNARY_BITS = math.log2(3)
FP_BITS = 16


def ternary(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s = w.abs().mean().clamp(min=eps)
    return (w / s).round().clamp(-1, 1) * s


def act_quant(x: torch.Tensor, bits: int = 8, eps: float = 1e-5) -> torch.Tensor:
    q = 2 ** (bits - 1)
    s = (q - 1) / x.abs().amax(-1, keepdim=True).clamp(min=eps)
    return (x * s).round().clamp(-q, q - 1) / s


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, act_bits: int | None = 8,
                 norm: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.act_bits = act_bits
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm:
            x = F.rms_norm(x, (x.shape[-1],))
        if self.act_bits:
            x = x + (act_quant(x, self.act_bits) - x).detach()
        w = self.weight + (ternary(self.weight) - self.weight).detach()
        return F.linear(x, w, self.bias)

    def storage_bits(self) -> float:
        return TERNARY_BITS * self.weight.numel() + FP_BITS

    def bpw(self) -> float:
        return self.storage_bits() / self.weight.numel()
