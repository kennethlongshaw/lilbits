"""Typed query operators over a scanned context, composed by a router across loops into small programs.

Attention is one fixed query: a soft filtered join (softmax over q.k) followed by a weighted-sum aggregate. A
looped model can instead pick, on every loop and at every position, which query to run, what it reads and where
the result goes, like one instruction of a program over a per-position workspace:

    registers   per position: n_set SET registers (a weight per context position: a mask or a pointer) and
                n_vec VEC registers (d-dim), plus the hidden state h as a read-only vector input. They persist
                across loops and are discarded after the forward, like loop memory.
    ops         filter(q)        SET  sigmoid(q.k): a soft predicate over context positions
                point(q)         SET  softmax(q.k): a pointer (one position, softly)
                topk(S)          SET  keep the k largest weights of S (hard mask, straight-through)
                sum(S) / mean(S) VEC  aggregate context values under S
                max(S)           VEC  per-channel max of values where S is on
                count(S)         VEC  the number of positions in S, as a vector (count * u)
                attend(q)        VEC  softmax attention (point + mean in one step): the transformer op
                mix(v)           VEC  an MLP of the chosen vector input and h
                noop                  nothing
    arguments   q is a projection of the chosen VEC input (h or a register); S is the chosen SET register
    router      from h and the loop index: probabilities over ops, the SET / VEC input slots and the SET / VEC
                destination slots. Training: Gumbel-softmax (soft, annealed temperature). Inference: argmax, one
                op per loop per position, recorded as a readable trace.
    write back  a gated projection of the VEC destination adds to h after every loop; a shared MLP follows.

Every op reads context positions <= its own (causal), so the model trains as a language model. A load-balance
loss (n_ops * sum_i mean_p_i^2, minimal when ops are used evenly) works against the router collapsing onto one op.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

SET_OPS = ("filter", "point", "topk")
VEC_OPS = ("sum", "mean", "max", "count", "attend", "mix")
ALL_OPS = (*SET_OPS, *VEC_OPS, "noop")


def gumbel_probs(logits: torch.Tensor, tau: float, hard: bool, generator=None) -> torch.Tensor:
    """Soft Gumbel-softmax probabilities (training) or a one-hot argmax (hard, inference)."""
    if hard:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits)
    u = torch.rand(logits.shape, generator=generator).to(logits).clamp(1e-6, 1 - 1e-6)
    return ((logits - (-u.log()).log()) / tau).softmax(-1)


class OperatorLoop(nn.Module):
    """One shared instruction step: route, run the typed ops, write registers and h."""

    def __init__(self, d: int, n_loops: int, ops=ALL_OPS, n_set: int = 2, n_vec: int = 2, k: int = 4):
        super().__init__()
        self.ops = tuple(ops)
        for o in self.ops:
            assert o in ALL_OPS, o
        self.d, self.n_set, self.n_vec, self.k = d, n_set, n_vec, k
        self.loop_emb = nn.Embedding(n_loops, d)
        nn.init.normal_(self.loop_emb.weight, std=0.02)
        self.norm = nn.RMSNorm(d)
        sizes = (len(self.ops), n_set, n_vec + 1, n_set, n_vec)  # op, set in, vec in (h + regs), set dst, vec dst
        self.sizes = sizes
        self.router = nn.Linear(d, sum(sizes))
        self.q = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.count_u = nn.Parameter(torch.randn(d) * 0.02)
        self.mix = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.filter_bias = nn.Parameter(torch.zeros(()))
        self.gate = nn.Linear(d, d)
        self.write = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.RMSNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def new_registers(self, B: int, T: int, ref: torch.Tensor):
        return [ref.new_zeros(B, T, T) for _ in range(self.n_set)], [
            ref.new_zeros(B, T, self.d) for _ in range(self.n_vec)
        ]

    def forward(self, h, ctx, regs, loop: int, tau: float = 1.0, hard: bool = False, generator=None):
        """h (B, T, d) hidden; ctx (B, T, d) scanned context; regs (sets, vecs). Returns h, regs, probs dict."""
        B, T, d = h.shape
        sets, vecs = regs
        z = self.norm(h) + self.loop_emb.weight[loop]
        parts = self.router(z).split(self.sizes, -1)
        p_op, p_sin, p_vin, p_sdst, p_vdst = (gumbel_probs(x, tau, hard, generator) for x in parts)
        causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
        S = sum(p_sin[..., j, None] * sets[j] for j in range(self.n_set))  # chosen SET input (B, T, T)
        vin_all = [h, *vecs]
        vin = sum(p_vin[..., j, None] * vin_all[j] for j in range(self.n_vec + 1))
        K, V = self.k_proj(ctx), self.v_proj(ctx)
        q = self.q(vin)
        score = torch.einsum("btd,bsd->bts", q, K) / math.sqrt(d)
        score = score.masked_fill(~causal, float("-inf"))
        out_set, out_vec = {}, {}
        for i_op, name in enumerate(self.ops):
            if hard and not bool(p_op[..., i_op].any()):
                continue  # hard routing: run only the ops some position chose
            if name == "filter":
                out_set[name] = torch.sigmoid(score + self.filter_bias).masked_fill(~causal, 0.0)
            elif name == "point":
                out_set[name] = score.softmax(-1)
            elif name == "topk":
                kk = min(self.k, T)
                idx = S.masked_fill(~causal, -1.0).topk(kk, -1).indices
                keep = torch.zeros_like(S).scatter(-1, idx, 1.0) * causal
                out_set[name] = S * keep + (S - S.detach()) * (1 - keep)  # forward: masked; grad: through S
            elif name == "sum":
                out_vec[name] = self.o(S @ V)
            elif name == "mean":
                out_vec[name] = self.o((S @ V) / S.sum(-1, keepdim=True).clamp_min(1e-6))
            elif name == "max":
                w = S.unsqueeze(-1)
                m = (V.unsqueeze(1) * w + (w - 1) * 1e4).amax(2)  # positions off in S cannot win
                out_vec[name] = self.o(m * (S.sum(-1, keepdim=True) > 0))
            elif name == "count":
                out_vec[name] = S.sum(-1, keepdim=True) * self.count_u
            elif name == "attend":
                out_vec[name] = self.o(score.softmax(-1) @ V)
            elif name == "mix":
                out_vec[name] = self.mix(torch.cat([vin, h], -1))
        idx = {n: i for i, n in enumerate(self.ops)}  # (ops skipped above have probability 0 everywhere)
        w_set = sum(p_op[..., idx[n]] for n in out_set) if out_set else h.new_zeros(B, T)
        w_vec = sum(p_op[..., idx[n]] for n in out_vec) if out_vec else h.new_zeros(B, T)
        new_set = sum(p_op[..., idx[n], None] * s for n, s in out_set.items()) if out_set else None
        new_vec = sum(p_op[..., idx[n], None] * v for n, v in out_vec.items()) if out_vec else None
        if new_set is not None:  # write into the chosen SET slot (weight = prob of a set op x prob of the slot)
            sets = [
                (1 - (w_set * p_sdst[..., j])[..., None]) * sets[j] + p_sdst[..., j, None] * new_set
                for j in range(self.n_set)
            ]
        if new_vec is not None:
            vecs = [
                (1 - (w_vec * p_vdst[..., j])[..., None]) * vecs[j] + p_vdst[..., j, None] * new_vec
                for j in range(self.n_vec)
            ]
            h = h + torch.sigmoid(self.gate(h)) * self.write(new_vec)
        h = h + self.mlp(h)
        probs = {"op": p_op, "set_in": p_sin, "vec_in": p_vin, "set_dst": p_sdst, "vec_dst": p_vdst}
        return h, (sets, vecs), probs

    def load_balance(self, p_op: torch.Tensor) -> torch.Tensor:
        mean = p_op.flatten(0, -2).mean(0)
        return len(self.ops) * (mean**2).sum()


__all__ = ["ALL_OPS", "SET_OPS", "VEC_OPS", "OperatorLoop", "gumbel_probs"]
