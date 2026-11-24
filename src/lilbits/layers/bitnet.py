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
    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, act_bits: int | None = 8, norm: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.act_bits = act_bits
        self.norm = norm
        self._infer_w: torch.Tensor | None = None

    def train(self, mode: bool = True):
        self._infer_w = None  # weights may change once training resumes
        return super().train(mode)

    def _quant_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm:
            x = F.rms_norm(x, (x.shape[-1],))
        if self.act_bits:
            x = x + (act_quant(x, self.act_bits) - x).detach()
        return x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._quant_input(input)
        if not self.training and not torch.is_grad_enabled():  # inference: ternarize once, reuse
            if self._infer_w is None:
                self._infer_w = ternary(self.weight).detach()
            return F.linear(x, self._infer_w, self.bias)
        w = self.weight + (ternary(self.weight) - self.weight).detach()
        return F.linear(x, w, self.bias)

    def storage_bits(self) -> float:
        return TERNARY_BITS * self.weight.numel() + FP_BITS

    def bpw(self) -> float:
        return self.storage_bits() / self.weight.numel()


class PackedBitLinear(nn.Module):
    """Inference-only BitLinear: 2-bit codes (4 weights per byte) + one FP16 scale, unpacked per call.

    2 bits/weight is the simple runtime layout (bitnet.cpp's I2_S uses the same); the 1.585-bit figure
    above is the storage floor.
    """

    codes: torch.Tensor
    scale: torch.Tensor
    lut: torch.Tensor

    def __init__(self, layer: BitLinear):
        super().__init__()
        self.in_features, self.out_features = layer.in_features, layer.out_features
        self.act_bits, self.norm = layer.act_bits, layer.norm
        w = layer.weight.detach()
        s = w.abs().mean().clamp(min=1e-5)
        codes = ((w / s).round().clamp(-1, 1) + 1).to(torch.uint8).flatten()  # 0, 1, 2
        codes = F.pad(codes, (0, (-codes.numel()) % 4)).view(-1, 4)
        self.register_buffer("codes", codes[:, 0] | codes[:, 1] << 2 | codes[:, 2] << 4 | codes[:, 3] << 6)
        self.register_buffer("scale", s.half())
        self.bias = layer.bias
        lut = torch.tensor([[((b >> (2 * i)) & 3) - 1.0 for i in range(4)] for b in range(256)])
        self.register_buffer("lut", lut, persistent=False)

    def weight(self) -> torch.Tensor:
        n = self.in_features * self.out_features
        w = self.lut[self.codes.long()].flatten()[:n].view(self.out_features, self.in_features)
        return w * self.scale.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(x, (x.shape[-1],)) if self.norm else x
        x = act_quant(x, self.act_bits) if self.act_bits else x
        return F.linear(x, self.weight(), self.bias)


def pack_bitnet(model: nn.Module) -> nn.Module:
    """In place: BitLinear -> PackedBitLinear (drops the FP32 latent weights)."""
    for name, child in list(model.named_children()):
        if isinstance(child, BitLinear):
            setattr(model, name, PackedBitLinear(child))
        else:
            pack_bitnet(child)
    return model.eval()
