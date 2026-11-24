"""Pre-norm transformer block / tiny causal LM with pluggable linear layers.

Following LittleBit, only the linear projections inside the blocks
(q, k, v, o, gate, up, down) are compressed. Embeddings, norms and the
LM head stay full precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.bitnet import BitLinear
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.mixers.recurrent import MIXERS, GatedDeltaNet
from lilbits.layers.ffn import SwiGLU
from lilbits.mixers.attention import Attention


@dataclass
class Config:
    vocab_size: int = 32
    seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: float = 8 / 3
    # linear layer choice: "fp", "littlebit" or "bitnet" (BitNet b1.58 baseline)
    linear: str = "fp"
    bpw: float = 1.0
    residual: bool = True
    ste: str = "clipped"
    smooth_k: float = 100.0
    # latent init: "small" (N(0, 0.1)), "bimodal" (+-0.5 + noise), "svd" (Dual-SVID of a random dense init)
    init: str = "small"
    qk_norm: bool = False
    itq_iters: int = 0  # >0: Joint-ITQ rotation before binarizing (svd init / FP warm start)
    # token mixer: "softmax" (causal attention), "gdn" (Gated DeltaNet), "mamba2" (SSD-style SSM),
    # "gla" (gated linear attention / RWKV-6 style), "mlstm" (xLSTM matrix memory), "mingru" (element-wise RNN)
    mixer: str = "softmax"
    conv_size: int = 4  # gdn: short causal conv kernel on q/k/v
    lb_gates: bool = False  # gdn: also LittleBit the per-head decay/beta projections (default keeps them FP)


LinearFactory = Callable[[int, int], nn.Module]


def make_linear_factory(cfg: Config) -> LinearFactory:
    if cfg.linear == "fp":
        return partial(nn.Linear, bias=False)
    if cfg.linear == "littlebit":
        return partial(
            LittleBitLinear,
            bpw=cfg.bpw,
            residual=cfg.residual,
            ste=cfg.ste,
            smooth_k=cfg.smooth_k,
            init=cfg.init,
            itq_iters=cfg.itq_iters,
        )
    if cfg.linear == "bitnet":
        return BitLinear
    raise ValueError(f"unknown linear {cfg.linear!r}")


def make_mixer(d_model: int, n_heads: int, linear: LinearFactory, mixer: str = "softmax", qk_norm: bool = False,
               conv_size: int = 4, lb_gates: bool = False) -> nn.Module:
    if mixer == "softmax":
        return Attention(d_model, n_heads, linear, qk_norm)
    if mixer == "gdn":
        return GatedDeltaNet(d_model, n_heads, linear, conv_size, lb_gates)
    if mixer == "gla":
        return MIXERS[mixer](d_model, n_heads, linear, lb_gates=lb_gates)
    if mixer in MIXERS:
        return MIXERS[mixer](d_model, n_heads, linear, conv_size=conv_size, lb_gates=lb_gates)
    raise ValueError(f"unknown mixer {mixer!r}")


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, linear: LinearFactory, qk_norm: bool = False,
                 mixer: str = "softmax", conv_size: int = 4, lb_gates: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = make_mixer(d_model, n_heads, linear, mixer, qk_norm, conv_size, lb_gates)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden, linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        linear = make_linear_factory(cfg)
        hidden = int(round(cfg.d_model * cfg.mlp_ratio / 8)) * 8
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model, cfg.n_heads, hidden, linear, cfg.qk_norm, cfg.mixer, cfg.conv_size,
                             cfg.lb_gates)
            for _ in range(cfg.n_layers)
        )
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


@torch.no_grad()
def convert_fp_to_littlebit(fp_model: TinyLM, cfg: Config) -> TinyLM:
    """Build a LittleBit TinyLM whose block linears are SVD(+ITQ)-initialized from fp_model."""
    lb = TinyLM(cfg).to(next(fp_model.parameters()).device)
    fp_mods = dict(fp_model.named_modules())
    for name, m in lb.named_modules():
        if isinstance(m, LittleBitLinear):
            m.init_from_weight(fp_mods[name].weight, cfg.itq_iters)
    lin_keys = {k for k in lb.state_dict() if ".paths." in k}
    rest = {k: v for k, v in fp_model.state_dict().items() if k in lb.state_dict() and k not in lin_keys}
    lb.load_state_dict(rest, strict=False)
    return lb


def compression_report(model: nn.Module) -> dict:
    """Bits used by compressible block linears, FP16 vs actual."""
    dense = 0
    bits = 0
    for m in model.modules():
        if isinstance(m, (LittleBitLinear, BitLinear)):
            dense += m.in_features * m.out_features
            bits += m.storage_bits()
        elif isinstance(m, nn.Linear) and m is not getattr(model, "head", None):
            dense += m.in_features * m.out_features
            bits += 16 * m.in_features * m.out_features
    return {"block_linear_weights": dense, "bits": bits, "bpw": bits / max(dense, 1)}
