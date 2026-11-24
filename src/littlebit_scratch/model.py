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

from .littlebit import LittleBitLinear


@dataclass
class Config:
    vocab_size: int = 32
    seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: float = 8 / 3
    # linear layer choice: "fp" or "littlebit"
    linear: str = "fp"
    bpw: float = 1.0
    residual: bool = True
    ste: str = "clipped"
    smooth_k: float = 100.0
    # latent init: "small" (N(0, 0.1)), "bimodal" (+-0.5 + noise), "svd" (Dual-SVID of a random dense init)
    init: str = "small"
    qk_norm: bool = False
    itq_iters: int = 0  # >0: Joint-ITQ rotation before binarizing (svd init / FP warm start)


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
    raise ValueError(f"unknown linear {cfg.linear!r}")


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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, linear: LinearFactory):
        super().__init__()
        self.gate = linear(d_model, hidden)
        self.up = linear(d_model, hidden)
        self.down = linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, linear: LinearFactory, qk_norm: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, linear, qk_norm)
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
            TransformerBlock(cfg.d_model, cfg.n_heads, hidden, linear, cfg.qk_norm) for _ in range(cfg.n_layers)
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
        if isinstance(m, LittleBitLinear):
            dense += m.in_features * m.out_features
            bits += m.storage_bits()
        elif isinstance(m, nn.Linear) and m is not getattr(model, "head", None):
            dense += m.in_features * m.out_features
            bits += 16 * m.in_features * m.out_features
    return {"block_linear_weights": dense, "bits": bits, "bpw": bits / max(dense, 1)}
