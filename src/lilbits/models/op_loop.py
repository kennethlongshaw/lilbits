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

    def forward(self, x: torch.Tensor, n_loops: int | None = None, generator=None, full_trace: bool = False):
        """x (B, T) bytes -> (logits (B, T, 256), op probabilities per loop [(B, T, n_ops)]). full_trace=True
        returns every loop's router dict instead (op, set_in, vec_in, set_dst, vec_dst, op_logp)."""
        ctx, _ = self.encoder(self.embed(x))
        h = ctx
        regs = self.loop.new_registers(x.shape[0], x.shape[1], ctx)
        trace = []
        for i in range(n_loops or self.n_loops):
            h, regs, probs = self.loop(h, ctx, regs, i, self.tau, self.hard, generator)
            trace.append(probs if full_trace else probs["op"])
        return self.norm(h) @ self.embed.weight.t(), trace

    def op_targets(self, groups: list[tuple[set[str], ...] | None]) -> torch.Tensor:
        """Per example, the expected op group for each loop -> multi-hot (B, n_loops, n_ops). Group j supervises
        loop j; later loops and examples with None stay all-False (unsupervised)."""
        t = torch.zeros(len(groups), self.n_loops, len(self.ops), dtype=torch.bool)
        for b, gs in enumerate(groups):
            for j, g in enumerate((gs or ())[: self.n_loops]):
                for k, op in enumerate(self.ops):
                    t[b, j, k] = op in g
        return t

    def loss(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        lb_weight: float = 0.01,
        generator=None,
        op_targets: torch.Tensor | None = None,
        router_weight: float = 1.0,
    ):
        """Masked next-byte CE + load balance. With op_targets (B, n_loops, n_ops) from `op_targets`, adds a
        teacher-forcing term on the router: -log P(op in expected group) at the trained (answer) positions."""
        logits, trace = self(x[:, :-1], generator=generator, full_trace=True)
        ce = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="none")
        w = mask.flatten()
        loss = (ce * w).sum() / w.sum().clamp_min(1)
        info = {"ce": loss.item(), "tau": self.tau}
        if len(self.ops) == 1:
            return loss, info
        lb = torch.stack([self.loop.load_balance(p["op"]) for p in trace]).mean()
        info["load_balance"] = lb.item()
        total = loss + lb_weight * lb
        if op_targets is not None:
            terms = []
            for i, p in enumerate(trace):
                allowed = op_targets[:, i]  # (B, n_ops)
                sel = mask.bool() & allowed.any(-1, keepdim=True)  # (B, T) supervised positions
                if sel.any():
                    lp = p["op_logp"].masked_fill(~allowed[:, None, :], float("-inf")).logsumexp(-1)
                    terms.append(-lp[sel].mean())
            if terms:
                rl = torch.stack(terms).mean()
                info["router_ce"] = rl.item()
                total = total + router_weight * rl
        return total, info

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

    @torch.no_grad()
    def sample(self, prompt: bytes, group: int, max_bytes: int = 40, temperature: float = 1.0, generator=None):
        """`group` sampled completions of prompt (hard routing), each up to a newline. Returns list of bytes."""
        hard, self.hard = self.hard, True
        x = torch.tensor(list(prompt))[None].expand(group, -1)
        done = torch.zeros(group, dtype=torch.bool)
        outs: list[list[int]] = [[] for _ in range(group)]
        for _ in range(max_bytes):
            logits, _ = self(x)
            p = (logits[:, -1] / temperature).softmax(-1)
            tok = torch.multinomial(p, 1, generator=generator).squeeze(-1)
            tok = torch.where(done, torch.full_like(tok, 10), tok)
            for i, t in enumerate(tok.tolist()):
                if not done[i] and t != 10:
                    outs[i].append(t)
            done |= tok == 10
            if done.all():
                break
            x = torch.cat([x, tok[:, None]], 1)
        self.hard = hard
        return [bytes(o) for o in outs]


__all__ = ["OpLoopLM"]
