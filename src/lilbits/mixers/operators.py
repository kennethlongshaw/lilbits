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
                destination slots. Training: Gumbel-softmax (soft, annealed temperature; every op runs) or
                straight-through (one-hot forward, only chosen ops run). Inference: argmax, one op per loop per
                position, recorded as a readable trace.
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


def gumbel_probs(logits: torch.Tensor, tau: float, hard: bool, generator=None, st: bool = False) -> torch.Tensor:
    """Soft Gumbel-softmax probabilities (training), a one-hot argmax (hard, inference), or straight-through
    (st: one-hot of the noisy sample in the forward pass, gradient of the soft sample in the backward pass)."""
    if hard:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits)
    u = torch.rand(logits.shape, generator=generator).to(logits).clamp(1e-6, 1 - 1e-6)
    y = ((logits - (-u.log()).log()) / tau).softmax(-1)
    if st:
        return F.one_hot(y.argmax(-1), y.shape[-1]).to(y) - y.detach() + y
    return y


def is_word_byte(x: torch.Tensor) -> torch.Tensor:
    """ASCII letters and digits (names, numbers); everything else separates words."""
    return ((x >= 48) & (x <= 57)) | ((x >= 65) & (x <= 90)) | ((x >= 97) & (x <= 122))


def word_pool(ctx: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Word-level keys over bytes x (B, T) with context ctx (B, T, d). Returns (wctx, wend): wctx[s] is the mean of
    ctx over the word containing s up to s (ctx itself on separator bytes); wend[s] marks the last byte of a word,
    known only once byte s + 1 is seen (the final position is never marked). A query at t may use word ends s < t,
    so this stays causal."""
    w = is_word_byte(x)
    start = w & ~F.pad(w[:, :-1], (1, 0), value=False)  # first byte of each word
    seg = torch.cumsum(start.long() | (~w).long(), 1)  # a new segment at every word start and every separator
    csum = torch.cumsum(ctx, 1)
    cnt = torch.arange(1, x.shape[1] + 1, device=x.device).expand_as(seg)
    # segment start index for each position: last index where seg changed
    change = F.pad(seg[:, 1:] != seg[:, :-1], (1, 0), value=True)
    idx = torch.where(change, torch.arange(x.shape[1], device=x.device).expand_as(seg), torch.zeros_like(seg))
    first = torch.cummax(idx, 1).values
    before = torch.where(first[..., None] > 0, csum.gather(1, (first - 1).clamp_min(0)[..., None].expand_as(ctx)), 0)
    wctx = (csum - before) / (cnt - first)[..., None].to(ctx)
    wend = w & ~F.pad(w[:, 1:], (0, 1), value=True)  # next byte is a separator (unknown at the last position)
    return wctx, wend


class OperatorLoop(nn.Module):
    """One shared instruction step: route, run the typed ops, write registers and h."""

    def __init__(
        self, d: int, n_loops: int, ops=ALL_OPS, n_set: int = 2, n_vec: int = 2, k: int = 4, word_keys: bool = False
    ):
        super().__init__()
        self.word_keys = word_keys  # set ops (filter / point / topk and the aggregates over S) address whole words
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
        # task summary for the router: one learned query attends causally over the context, so a position far
        # from the instruction (e.g. the answer slot) can still route on it (its own hidden state may not carry it)
        self.summary_q = nn.Parameter(torch.randn(d) * 0.02)
        self.summary_k = nn.Linear(d, d, bias=False)
        self.summary_o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.summary_o.weight)
        self.summary_q_in = nn.Linear(d, d, bias=False)  # the task summary also feeds every op's query
        nn.init.zeros_(self.summary_q_in.weight)
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

    def new_registers(self, B: int, T: int, ref: torch.Tensor, Tk: int | None = None):
        """Zeroed registers for T query positions over Tk context positions (Tk = T unless decoding a suffix)."""
        return [ref.new_zeros(B, T, Tk or T) for _ in range(self.n_set)], [
            ref.new_zeros(B, T, self.d) for _ in range(self.n_vec)
        ]

    def task_summary(self, ctx: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        """(B, T, d): per position, a softmax-weighted causal pool of the context under one learned query."""
        a = (self.summary_k(ctx) @ self.summary_q) / math.sqrt(self.d)  # (B, T)
        a = a[:, None, :].expand(-1, causal.shape[0], -1).masked_fill(~causal, float("-inf"))
        return a.softmax(-1) @ ctx

    def forward(
        self, h, ctx, regs, loop: int, tau: float = 1.0, hard: bool = False, generator=None, st=False, words=None
    ):
        """h (B, T, d) hidden of the last T positions; ctx (B, Tk, d) scanned context (Tk >= T); regs (sets
        (B, T, Tk), vecs (B, T, d)). Returns h, regs, probs dict.
        st=True trains with straight-through one-hot routing: only ops some position picked are computed (the
        soft path computes every op at every loop), at the price of a biased router gradient.
        words: (wctx, wend) from `word_pool` over the full context; with word_keys, SET ops score and aggregate
        word-end positions (keys and values pooled over each word) instead of single bytes."""
        B, T, d = h.shape
        Tk = ctx.shape[1]  # context may be longer than the queries: h holds the last T positions (cached decode)
        sets, vecs = regs
        causal = (torch.arange(T, device=h.device)[:, None] + Tk - T) >= torch.arange(Tk, device=h.device)[None]
        summary = self.task_summary(ctx, causal)
        z = self.norm(h) + self.loop_emb.weight[loop] + self.summary_o(summary)
        parts = self.router(z).split(self.sizes, -1)
        # Gumbel noise explores the op choice only; the input / output slots use the plain tempered softmax, so
        # an op reads and writes a consistent register from step to step
        p_op = gumbel_probs(parts[0], tau, hard, generator, st)
        p_sin, p_vin, p_sdst, p_vdst = (
            F.one_hot(x.argmax(-1), x.shape[-1]).to(x) if hard else (x / tau).softmax(-1) for x in parts[1:]
        )
        chosen = p_op.detach().flatten(0, -2).amax(0) > 0.5 if (hard or st) else None
        S = sum(p_sin[..., j, None] * sets[j] for j in range(self.n_set))  # chosen SET input (B, T, T)
        vin_all = [h, *vecs]
        vin = sum(p_vin[..., j, None] * vin_all[j] for j in range(self.n_vec + 1))
        K, V = self.k_proj(ctx), self.v_proj(ctx)
        q = self.q(vin) + self.summary_q_in(summary)  # e.g. a filter at the answer slot can match the queried name
        score = torch.einsum("btd,bsd->bts", q, K) / math.sqrt(d)
        score = score.masked_fill(~causal, float("-inf"))
        set_mask, set_score, Vs = causal, score, V
        if self.word_keys and words is not None:
            wctx, wend = words
            strict = (torch.arange(T, device=h.device)[:, None] + Tk - T) > torch.arange(Tk, device=h.device)[None]
            set_mask = strict[None] & wend[:, None, :]  # (B, T, Tk): completed words before the query
            set_score = (torch.einsum("btd,bsd->bts", q, self.k_proj(wctx)) / math.sqrt(d)).masked_fill(
                ~set_mask, float("-inf")
            )
            Vs = self.v_proj(wctx)
        out_set, out_vec = {}, {}
        for i_op, name in enumerate(self.ops):
            if chosen is not None and not bool(chosen[i_op]):
                continue  # hard / straight-through routing: run only the ops some position chose
            if name == "filter":
                out_set[name] = torch.sigmoid(set_score + self.filter_bias).masked_fill(~set_mask, 0.0)
            elif name == "point":
                any_ok = set_mask.any(-1, keepdim=True)
                out_set[name] = torch.where(any_ok, set_score.masked_fill(~any_ok, 0.0).softmax(-1), 0.0)
            elif name == "topk":
                kk = min(self.k, Tk)
                idx = S.masked_fill(~set_mask, -1.0).topk(kk, -1).indices
                keep = torch.zeros_like(S).scatter(-1, idx, 1.0) * set_mask
                out_set[name] = S * keep + (S - S.detach()) * (1 - keep)  # forward: masked; grad: through S
            elif name == "sum":
                out_vec[name] = self.o(S @ Vs)
            elif name == "mean":
                out_vec[name] = self.o((S @ Vs) / S.sum(-1, keepdim=True).clamp_min(1e-6))
            elif name == "max":
                w = S.unsqueeze(-1)
                m = (Vs.unsqueeze(1) * w + (w - 1) * 1e4).amax(2)  # positions off in S cannot win
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
        probs["op_logp"] = parts[0].log_softmax(-1)  # noise-free router log-probs (for supervised warm-up)
        return h, (sets, vecs), probs

    def load_balance(self, p_op: torch.Tensor) -> torch.Tensor:
        mean = p_op.flatten(0, -2).mean(0)
        return len(self.ops) * (mean**2).sum()


__all__ = ["ALL_OPS", "SET_OPS", "VEC_OPS", "OperatorLoop", "gumbel_probs", "is_word_byte", "word_pool"]
