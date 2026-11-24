"""Inference-only, bit-packed LittleBit layers.

`pack_model(model)` replaces every LittleBitLinear with a PackedLittleBitLinear that stores
- sign(U), sign(V) as 1 bit each, packed 8 per byte (uint8), per path
- h, g, l as FP16
and drops the FP32 latent factors. This matches the storage the paper (and our bpw numbers) assume.

The forward unpacks signs in rank chunks (default 128 columns), so the only transient memory is
(in + out) * chunk floats per layer, never a full FP matrix:

    y = sum_paths sum_chunks ((x * g) @ S_V[:, c] * l[c]) @ S_U[:, c]^T * h

There's no custom kernel: on CPU the unpacking costs time. A fused kernel (as in the paper) would
unpack in registers and cut weight memory traffic instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.littlebit import LittleBitLinear

_SHIFTS = torch.arange(8, dtype=torch.uint8)


def pack_signs(x: torch.Tensor) -> torch.Tensor:
    """(n, r) real -> (n, ceil(r/8)) uint8, bit j of byte k = [x[:, 8k+j] >= 0]."""
    n, r = x.shape
    bits = (x >= 0).to(torch.uint8)
    pad = (-r) % 8
    if pad:
        bits = F.pad(bits, (0, pad))
    bits = bits.view(n, -1, 8) << _SHIFTS.to(x.device)
    return bits.sum(dim=2, dtype=torch.uint8)


_LUT: dict = {}


def _lut(dtype, device) -> torch.Tensor:
    """(256, 8) table: byte value -> its 8 bits as +-1."""
    key = (dtype, device)
    if key not in _LUT:
        v = torch.arange(256, dtype=torch.uint8, device=device)
        _LUT[key] = (((v.unsqueeze(-1) >> _SHIFTS.to(device)) & 1).to(dtype) * 2 - 1).contiguous()
    return _LUT[key]


def unpack_signs(packed: torch.Tensor, r: int, dtype=torch.float32) -> torch.Tensor:
    """(n, nbytes) uint8 -> (n, r) of +-1 in `dtype` (table lookup, 8 values per byte)."""
    out = _lut(dtype, packed.device)[packed.to(torch.int32)]  # n, nbytes, 8
    return out.view(packed.shape[0], -1)[:, :r]


class PackedLittleBitLinear(nn.Module):
    """backend: "torch" (chunked unpack + matmul, any device) or "helion" (fused kernels,
    see helion_kernels.py; CUDA, or CPU with HELION_INTERPRET=1 for correctness checks)."""

    backend: str = "torch"
    U_bits: torch.Tensor
    V_bits: torch.Tensor
    h: torch.Tensor
    g: torch.Tensor
    l: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, layer: LittleBitLinear, chunk: int = 128, backend: str = "torch"):
        super().__init__()
        self.backend = backend
        assert chunk % 8 == 0
        self.in_features, self.out_features, self.rank = layer.in_features, layer.out_features, layer.rank
        self.paths = len(layer.paths)
        self.chunk = chunk
        with torch.no_grad():
            self.register_buffer("U_bits", torch.stack([pack_signs(p.U) for p in layer.paths]))  # P, out, r/8
            self.register_buffer("V_bits", torch.stack([pack_signs(p.V) for p in layer.paths]))  # P, in, r/8
            self.register_buffer("h", torch.stack([p.h for p in layer.paths]).half())
            self.register_buffer("g", torch.stack([p.g for p in layer.paths]).half())
            self.register_buffer("l", torch.stack([p.l for p in layer.paths]).half())
            self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().half())

    @classmethod
    def empty(cls, in_features: int, out_features: int, rank: int, paths: int, bias: bool = False, chunk: int = 128):
        """Allocate an unfilled packed layer (for load_state_dict) without building FP32 latents."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.backend = "torch"
        self.in_features, self.out_features, self.rank, self.paths, self.chunk = (
            in_features,
            out_features,
            rank,
            paths,
            chunk,
        )
        nb = (rank + 7) // 8
        self.register_buffer("U_bits", torch.zeros(paths, out_features, nb, dtype=torch.uint8))
        self.register_buffer("V_bits", torch.zeros(paths, in_features, nb, dtype=torch.uint8))
        self.register_buffer("h", torch.zeros(paths, out_features, dtype=torch.half))
        self.register_buffer("g", torch.zeros(paths, in_features, dtype=torch.half))
        self.register_buffer("l", torch.zeros(paths, rank, dtype=torch.half))
        self.register_buffer("bias", torch.zeros(out_features, dtype=torch.half) if bias else None)
        return self

    def _l8(self, i: int, nb: int) -> torch.Tensor:
        cache = self.__dict__.setdefault("_l8_cache", {})
        key = (i, self.l.device)
        if key not in cache:
            from lilbits.inference.helion_kernels import l_planes

            cache[key] = l_planes(self.l[i], nb)
        return cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "helion":
            from lilbits.inference.helion_kernels import packed_forward

            return packed_forward(self, x)
        # Scales are folded into each small unpacked chunk (never into activations), and the
        # output accumulates in place, so the only activation-sized tensor is y itself.
        dt = x.dtype
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        y = torch.zeros(x2.shape[0], self.out_features, dtype=dt, device=x.device)
        cb = self.chunk // 8
        for i in range(self.paths):
            g, h = self.g[i].to(dt), self.h[i].to(dt)
            for b0 in range(0, self.U_bits.shape[2], cb):
                r0 = b0 * 8
                r = min(self.chunk, self.rank - r0)
                Sv = unpack_signs(self.V_bits[i, :, b0 : b0 + cb], r, dt)  # in x r
                Sv = Sv * g[:, None] * self.l[i, r0 : r0 + r].to(dt)[None, :]
                Su = unpack_signs(self.U_bits[i, :, b0 : b0 + cb], r, dt) * h[:, None]  # out x r
                y.addmm_(x2 @ Sv, Su.t())
        if self.bias is not None:
            y += self.bias.to(dt)
        return y.view(*lead, self.out_features)

    def storage_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, paths={self.paths}, "
            f"packed={self.storage_bytes() / 1024:.1f} KiB"
        )


@torch.no_grad()
def pack_model(model: nn.Module, chunk: int = 128, backend: str = "torch") -> nn.Module:
    """In place: swap LittleBitLinear -> PackedLittleBitLinear (drops FP32 latents). Returns model in eval mode."""
    for name, child in list(model.named_children()):
        if isinstance(child, LittleBitLinear):
            setattr(model, name, PackedLittleBitLinear(child, chunk, backend))
        else:
            pack_model(child, chunk, backend)
    return model.eval()


def set_backend(model: nn.Module, backend: str) -> nn.Module:
    for m in model.modules():
        if isinstance(m, PackedLittleBitLinear):
            m.backend = backend
    return model


def model_bytes(model: nn.Module) -> int:
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    return total
