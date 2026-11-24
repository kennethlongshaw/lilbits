"""Primitives for decision models that answer typed questions about a state instead of generating text
(the Jev / "System One" interface: Choice, Score, Noul).

- `StatefulBlock`: pre-norm recurrent mixer + SwiGLU that takes and returns its recurrent state, so a state
  can be prefilled once and many question suffixes forked from copies of it.
- `fork_states`: expand per-layer states to one copy per question.
- `TypedReadout`: logits from the hidden state at the readout position against tied embedding rows of the
  allowed answer codes, with a learned temperature per question type.
- `ordinal_stats`, `crps`: expected value and the CRPS proper score for ordinal (Score) answers.
- `ece`, `brier`: calibration metrics.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.ffn import SwiGLU
from lilbits.mixers.recurrent import MIXERS


class StatefulBlock(nn.Module):
    """x, state -> x, state. With conv_size=1 the mixer's recurrent state is the block's whole memory."""

    def __init__(self, d_model: int, n_heads: int, linear, mixer: str = "gdn", mlp_linear=None):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
        kw: dict[str, Any] = {"conv_size": 1} if mixer in ("gdn", "mamba2", "mlstm", "mingru") else {}
        self.mixer = MIXERS[mixer](d_model, n_heads, linear, **kw)
        self.mlp = SwiGLU(d_model, int(d_model * 8 / 3) // 8 * 8, mlp_linear or linear)

    def forward(self, x: torch.Tensor, state=None):
        y, state = self.mixer(self.n1(x), initial_state=state, return_state=True)
        x = x + y
        return x + self.mlp(self.n2(x)), state


def _map_state(s, fn):
    return tuple(fn(t) for t in s) if isinstance(s, tuple) else fn(s)


def fork_states(states: list, n: int) -> list:
    """Per-layer states for ONE prefilled state (batch 1) -> n independent copies (batch n)."""
    return [_map_state(s, lambda t: t.expand(n, *t.shape[1:]).contiguous()) for s in states]


def clone_states(states: list) -> list:
    return [_map_state(s, torch.clone) for s in states]


class TypedReadout(nn.Module):
    """log p over the allowed answer codes for each row, read from tied embedding rows."""

    def __init__(self, embed: nn.Embedding, types: tuple[str, ...]):
        super().__init__()
        self.embed = embed
        self.types = types
        self.log_temp = nn.Parameter(torch.zeros(len(types)))

    def forward(self, h: torch.Tensor, allowed: torch.Tensor, kind: str) -> torch.Tensor:
        """h: (B, d) hidden at the readout token; allowed: (B, K) token ids of the answer codes."""
        logits = torch.einsum("bd,bkd->bk", h, self.embed.weight[allowed])
        return F.log_softmax(logits / self.log_temp[self.types.index(kind)].exp(), -1)


def ordinal_stats(p: torch.Tensor) -> torch.Tensor:
    """E[x] for a distribution over levels 0..K-1."""
    return p @ torch.arange(p.shape[-1], dtype=p.dtype, device=p.device)


def crps(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Discrete CRPS (ranked probability score) per row: sum_k (CDF_pred(k) - 1[k >= target])^2."""
    levels = torch.arange(p.shape[-1], device=p.device)
    return (p.cumsum(-1) - (levels >= target[:, None]).to(p.dtype)).square().sum(-1)


def brier(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (p - F.one_hot(target, p.shape[-1]).to(p.dtype)).square().sum(-1).mean()


def ece(conf: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    """Expected calibration error over equal-width confidence bins."""
    edges = torch.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.float().mean() * (conf[m].mean() - correct[m].float().mean()).abs()
    return float(err)
