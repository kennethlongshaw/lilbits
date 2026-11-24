"""Helion kernels for bit-packed LittleBit inference.

A packed LittleBit path computes  y = h * ( ((x * g) @ S_V) * l ) @ S_U^T,  where S_V (in x r) and
S_U (out x r) are +-1 matrices stored 1 bit per entry, 8 rank columns per byte
(bit j of byte b = column 8b + j, see packed.pack_signs).

Two kernels, each unpacking bits inside the tile (the weights never exist in unpacked form in memory):

  down:  zp[j, m, b] = l[8b+j] * sum_k (x[m,k] * g[k]) * sign_bit_j(Vb[k, b])
  up:    y[m, n]     = y_in[m, n] + h[n] * sum_b sum_j zp[j, m, b] * sign_bit_j(Ub[n, b])

The latent activation z is kept in 8 "bit planes" (zp[j] holds rank columns j, 8+j, 16+j, ...) so
neither kernel needs to reshape a tile; each byte tile is loaded once per plane and shifted.
Padding bits (rank not a multiple of 8) get l = 0, so they contribute nothing.

Requires `helion` (uv sync --extra gpu). On a CUDA box the first call autotunes over CONFIGS_*
(set LB_HELION_AUTOTUNE=1 for Helion's full autotuner). On CPU, run with HELION_INTERPRET=1 to execute
the kernels in Helion's eager reference mode (correctness only, no speed).
"""

from __future__ import annotations

import os

import torch

import helion
import helion.language as hl

_FULL = os.environ.get("LB_HELION_AUTOTUNE") == "1"

# block_sizes order follows the hl.tile calls: down = [m, b(bytes), k], up = [m, n, b(bytes)]
CONFIGS_DOWN = [
    helion.Config(block_sizes=[16, 16, 128], num_warps=4, num_stages=3),
    helion.Config(block_sizes=[16, 32, 64], num_warps=4, num_stages=3),
    helion.Config(block_sizes=[64, 16, 64], num_warps=4, num_stages=3),
    helion.Config(block_sizes=[128, 32, 64], num_warps=8, num_stages=3),
]
CONFIGS_UP = [
    helion.Config(block_sizes=[16, 64, 16], num_warps=4, num_stages=3),
    helion.Config(block_sizes=[16, 128, 32], num_warps=4, num_stages=3),
    helion.Config(block_sizes=[64, 64, 16], num_warps=4, num_stages=3),
    helion.Config(block_sizes=[128, 128, 32], num_warps=8, num_stages=3),
]


def _kernel(configs):
    return helion.kernel() if _FULL else helion.kernel(configs=configs)


@_kernel(CONFIGS_DOWN)
def littlebit_down(x: torch.Tensor, g: torch.Tensor, vb: torch.Tensor, l8: torch.Tensor) -> torch.Tensor:
    """x [m, k], g [k], vb [k, nb] uint8, l8 [8, nb] -> zp [8, m, nb] fp32."""
    m, k = x.size()
    nb = vb.size(1)
    zp = torch.empty([8, m, nb], dtype=torch.float32, device=x.device)
    for tile_m, tile_b in hl.tile([m, nb]):
        for j in hl.static_range(8):
            acc = hl.zeros([tile_m, tile_b], dtype=torch.float32)
            for tile_k in hl.tile(k):
                s = ((vb[tile_k, tile_b].to(torch.int32) >> j) & 1).to(torch.float32) * 2.0 - 1.0
                xg = x[tile_m, tile_k].to(torch.float32) * g[tile_k].to(torch.float32)[None, :]
                acc = acc + torch.matmul(xg, s)
            zp[j, tile_m, tile_b] = acc * l8[j, tile_b].to(torch.float32)[None, :]
    return zp


@_kernel(CONFIGS_UP)
def littlebit_up(zp: torch.Tensor, ub: torch.Tensor, h: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
    """zp [8, m, nb] fp32, ub [n, nb] uint8, h [n], y_in [m, n] -> y [m, n] (dtype of y_in)."""
    _, m, nb = zp.size()
    n = ub.size(0)
    y = torch.empty_like(y_in)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_b in hl.tile(nb):
            u = ub[tile_n, tile_b].to(torch.int32)
            for j in hl.static_range(8):
                s = ((u >> j) & 1).to(torch.float32) * 2.0 - 1.0  # [tile_n, tile_b]
                acc = acc + torch.matmul(zp[j, tile_m, tile_b], s.t())
        out = acc * h[tile_n].to(torch.float32)[None, :] + y_in[tile_m, tile_n].to(torch.float32)
        y[tile_m, tile_n] = out.to(y.dtype)
    return y


def l_planes(l: torch.Tensor, nb: int) -> torch.Tensor:
    """l [r] -> l8 [8, nb] with l8[j, b] = l[8b+j] (0 for padding)."""
    lp = torch.zeros(nb * 8, dtype=torch.float32, device=l.device)
    lp[: l.numel()] = l.float()
    return lp.view(nb, 8).t().contiguous()


def packed_forward(layer, x: torch.Tensor) -> torch.Tensor:
    """PackedLittleBitLinear forward via the Helion kernels."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, layer.in_features).contiguous()
    nb = layer.U_bits.shape[2]
    y = torch.zeros(x2.shape[0], layer.out_features, dtype=x.dtype, device=x.device)
    for i in range(layer.paths):
        zp = littlebit_down(x2, layer.g[i], layer.V_bits[i], layer._l8(i, nb))
        y = littlebit_up(zp, layer.U_bits[i], layer.h[i], y)
    if layer.bias is not None:
        y = y + layer.bias.to(y.dtype)
    return y.view(*lead, layer.out_features)
