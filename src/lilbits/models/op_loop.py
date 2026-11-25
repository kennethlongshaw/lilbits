"""Operator-loop LM: a byte encoder scans the input once, then a shared `OperatorLoop` runs N times, each loop one
routed instruction over the context and a register workspace (mixers/operators.py). `ops=("attend",)` gives the
attention-only loop to compare against.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.mixers.operators import ALL_OPS, OperatorLoop
from lilbits.models.decision import StatefulBlock


class OpLoopLM(nn.Module):
    def __init__(self, d: int = 128, n_loops: int = 4, ops=ALL_OPS, heads: int = 2, n_set: int = 2, n_vec: int = 2):
        super().__init__()
        self.n_loops = n_loops
        self.embed = nn.Embedding(256, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.encoder = StatefulBlock(d, heads, partial(nn.Linear, bias=False))
        self.loop = OperatorLoop(d, n_loops, ops, n_set, n_vec)
        self.norm = nn.RMSNorm(d)
        self.tau, self.hard = 1.0, False  # router temperature / hard choices (set by the training schedule)

    @property
    def ops(self) -> tuple[str, ...]:
        return self.loop.ops

    def forward(self, x: torch.Tensor, n_loops: int | None = None, generator=None):
        """x (B, T) bytes -> (logits (B, T, 256), op probabilities per loop [(B, T, n_ops)])."""
        ctx, _ = self.encoder(self.embed(x))
        h = ctx
        regs = self.loop.new_registers(x.shape[0], x.shape[1], ctx)
        trace = []
        for i in range(n_loops or self.n_loops):
            h, regs, probs = self.loop(h, ctx, regs, i, self.tau, self.hard, generator)
            trace.append(probs["op"])
        return self.norm(h) @ self.embed.weight.t(), trace

    def loss(self, x: torch.Tensor, mask: torch.Tensor, lb_weight: float = 0.01, generator=None):
        logits, trace = self(x[:, :-1], generator=generator)
        ce = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="none")
        w = mask.flatten()
        loss = (ce * w).sum() / w.sum().clamp_min(1)
        lb = torch.stack([self.loop.load_balance(p) for p in trace]).mean()
        info = {"ce": loss.item(), "load_balance": lb.item(), "tau": self.tau}
        return loss + lb_weight * lb if len(self.ops) > 1 else loss, info

    @torch.no_grad()
    def complete(self, prompt: bytes, max_bytes: int = 40, n_loops: int | None = None) -> tuple[str, list[str]]:
        """Greedy answer up to a newline, with hard routing; returns (answer, op chosen per loop at the first
        answer position)."""
        hard, self.hard = self.hard, True
        x = torch.tensor(list(prompt))[None]
        out, ops_first = [], []
        for t in range(max_bytes):
            logits, trace = self(x, n_loops)
            if t == 0:
                ops_first = [self.ops[int(p[0, -1].argmax())] for p in trace]
            nxt = int(logits[0, -1].argmax())
            if nxt == 10:
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], 1)
        self.hard = hard
        return bytes(out).decode(errors="replace"), ops_first


__all__ = ["OpLoopLM"]
