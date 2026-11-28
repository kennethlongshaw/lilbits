"""Looped depth over MiniMax Sparse Attention (MSA): a byte LM whose looped core reads a K/V cache sparsely.

    bytes -> embed -> P x GDN prelude block -> h0
             h0 -> K, V (GQA, RoPE) and index keys K_idx, written to the cache ONCE per byte
             x = h0; for l in 1..L:  x = core(x + loop_emb[l])    (one shared block, L times)
                 core: MSA(x -> Q, Q_idx over the shared cache) + SwiGLU, pre-norm residual
             -> RMSNorm -> tied next-byte logits

MSA (MiniMax Sparse Attention, arXiv 2606.13392), at byte scale:
- Index branch: one index query head per GQA group (Q_idx = x W_qi) and one index key head shared by all groups
  (K_idx = h0 W_ki). Token scores Q_idx K_idx^T / sqrt(d_idx) are max-pooled over each key block (causally visible
  tokens only), and every query keeps its top-k blocks per group. The query's own block is always kept and uses one
  of the k slots.
- Main branch: ordinary softmax attention, each head over the visible tokens of its group's selected blocks.
- Training: top-k is not differentiable, so the index branch learns from a KL loss towards the group-averaged,
  detached main-branch attention, with stop-gradient on the index inputs (the aux loss never reaches the backbone).
  The first `dense_frac` of training is dense attention + KL (the paper's warmup), then sparse.

Why the loop reads one cache: the K/V cache is written once per byte from the prelude, and every loop iteration
asks it a new question (its own Q and Q_idx from the current loop state). So a loop costs Wq, Wqi, Wo and the MLP
plus at most k x block cached tokens, never another K/V projection or another cache. Each loop can fetch different
blocks, which is the thing a looped GDN core cannot do: it only ever sees its own compressed state.

The paper's scale is block 128, k 16 (2,048 tokens per query at 1M context). Bytes at a 512-byte window need small
blocks: block 16, k 4 keeps at most 64 of up to 512 cached bytes per query and group.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.ffn import SwiGLU
from lilbits.models.decision import StatefulBlock


def rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x: (..., T, D), pos: (T,) absolute positions. Rotate-half RoPE."""
    D = x.shape[-1]
    inv = base ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    ang = pos.to(torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def select_blocks(scores: torch.Tensor, block: int, topk: int, q_pos: torch.Tensor) -> torch.Tensor:
    """MSA block selection. scores: (B, G, Tq, Tk) index scores with causally invisible keys already -inf;
    q_pos: (Tq,) absolute query positions. -> bool (B, G, Tq, Tk), True = key in a selected block (and visible)."""
    B, G, Tq, Tk = scores.shape
    nb = -(-Tk // block)
    pad = nb * block - Tk
    s = F.pad(scores, (0, pad), value=float("-inf")) if pad else scores
    bmax = s.view(B, G, Tq, nb, block).amax(-1)  # (B, G, Tq, nb), -inf for blocks with no visible key
    own = (q_pos // block).clamp_max(nb - 1)
    bmax = bmax.scatter(-1, own.view(1, 1, Tq, 1).expand(B, G, Tq, 1), float("inf"))  # own block always kept
    k = min(topk, nb)
    idx = bmax.topk(k, -1).indices
    keep = torch.zeros_like(bmax, dtype=torch.bool).scatter(-1, idx, True)
    keep &= bmax > float("-inf")  # a query with fewer than k visible blocks: don't select invisible ones
    tok = keep.repeat_interleave(block, -1)[..., :Tk]
    return tok & (scores > float("-inf"))


class MSACore(nn.Module):
    """The looped block: MSA over the shared cache + SwiGLU. The cache projections (K, V, K_idx) live here too but
    run once per byte on the prelude output, not once per loop."""

    def __init__(self, d: int, n_heads: int, n_groups: int, d_idx: int, block: int, topk: int):
        super().__init__()
        assert d % n_heads == 0 and n_heads % n_groups == 0
        self.H, self.G, self.D, self.d_idx = n_heads, n_groups, d // n_heads, d_idx
        self.block, self.topk = block, topk
        self.n_kv, self.n_idx = nn.RMSNorm(d), nn.RMSNorm(d)
        self.k = nn.Linear(d, n_groups * self.D, bias=False)
        self.v = nn.Linear(d, n_groups * self.D, bias=False)
        self.k_idx = nn.Linear(d, d_idx, bias=False)
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.q_idx = nn.Linear(d, n_groups * d_idx, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.q_norm, self.k_norm = nn.RMSNorm(self.D), nn.RMSNorm(self.D)
        self.mlp = SwiGLU(d, int(d * 8 / 3) // 8 * 8, lambda i, o: nn.Linear(i, o, bias=False))

    def cache_entries(self, h0: torch.Tensor, pos: torch.Tensor):
        """h0: (B, T, d) prelude output -> K, V (B, G, T, D), K_idx (B, T, d_idx). K_idx from detached h0."""
        B, T, _ = h0.shape
        hn = self.n_kv(h0)
        k = self.k_norm(self.k(hn).view(B, T, self.G, self.D).transpose(1, 2))
        v = self.v(hn).view(B, T, self.G, self.D).transpose(1, 2)
        k_idx = self.k_idx(self.n_idx(h0.detach()))
        return rope(k, pos), v, k_idx

    def index_scores(self, xn: torch.Tensor, k_idx: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        """-> (B, G, Tq, Tk) index scores, -inf where not causally visible. Index query from detached input."""
        B, Tq, _ = xn.shape
        qi = self.q_idx(xn.detach()).view(B, Tq, self.G, self.d_idx).transpose(1, 2)
        s = qi @ k_idx.unsqueeze(1).transpose(-1, -2) / math.sqrt(self.d_idx)
        return s.masked_fill(~causal, float("-inf"))

    def forward(self, x, cache, q_pos: torch.Tensor, sparse: bool, kl: bool = False):
        """Parallel form (training / prefill / eval). x: (B, Tq, d); cache = (K, V, K_idx) with Tk keys at positions
        0..Tk-1; q_pos: (Tq,) positions of the queries. -> (x, kl_loss or None, mean attended keys per query)."""
        k, v, k_idx = cache
        B, Tq, d = x.shape
        Tk = k.shape[2]
        xn = self.n1(x)
        causal = torch.arange(Tk, device=x.device)[None, :] <= q_pos[:, None]  # (Tq, Tk)
        q = rope(self.q_norm(self.q(xn).view(B, Tq, self.H, self.D).transpose(1, 2)), q_pos)
        rep = self.H // self.G
        kh, vh = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        need_idx = sparse or kl
        s_idx = self.index_scores(xn, k_idx, causal) if need_idx else None
        if sparse:
            assert s_idx is not None
            mask_g = select_blocks(s_idx, self.block, self.topk, q_pos)  # (B, G, Tq, Tk)
            mask = mask_g.repeat_interleave(rep, 1)
        else:
            mask_g = causal.expand(B, self.G, Tq, Tk)
            mask = causal
        att = (q @ kh.transpose(-1, -2)) / math.sqrt(self.D)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ vh).transpose(1, 2).reshape(B, Tq, d)
        x = x + self.o(y)
        x = x + self.mlp(self.n2(x))
        kl_loss = None
        if kl:
            assert s_idx is not None
            # target: group-averaged, detached main attention; the index distribution over the same keys
            p = att.detach().view(B, self.G, rep, Tq, Tk).mean(2)
            logq = s_idx.masked_fill(~mask_g, float("-inf")).log_softmax(-1)
            term = torch.where(p > 0, p * (p.clamp_min(1e-12).log() - logq), torch.zeros_like(p))
            kl_loss = term.sum(-1).mean()
        return x, kl_loss, mask_g.sum(-1).float().mean()

    @torch.no_grad()
    def step(self, x: torch.Tensor, cache, t: int, sparse: bool):
        """Decode one byte at position t (batch 1) from a cache holding positions 0..t: a real gather of the top-k
        blocks, not a mask. x: (1, 1, d). -> x, attended keys."""
        k, v, k_idx = cache  # (1, G, t+1, D), (1, G, t+1, D), (1, t+1, d_idx)
        n = t + 1
        xn = self.n1(x)
        pos = torch.tensor([t])
        q = rope(self.q_norm(self.q(xn).view(1, 1, self.H, self.D).transpose(1, 2)), pos)  # (1, H, 1, D)
        rep = self.H // self.G
        q = q.view(self.G, rep, self.D)
        if sparse and n > self.block * self.topk:
            qi = self.q_idx(xn).view(self.G, self.d_idx)
            s = qi @ k_idx[0, :n].t() / math.sqrt(self.d_idx)  # (G, n)
            nb = -(-n // self.block)
            pad = nb * self.block - n
            bmax = F.pad(s, (0, pad), value=float("-inf")).view(self.G, nb, self.block).amax(-1)
            bmax[:, t // self.block] = float("inf")
            blocks = bmax.topk(self.topk, -1).indices  # (G, k)
            idx = (blocks[:, :, None] * self.block + torch.arange(self.block)).flatten(1)  # (G, k*block)
            valid = idx < n
            idx = idx.clamp_max(n - 1)
            kg = k[0].gather(1, idx[:, :, None].expand(-1, -1, self.D))  # (G, m, D)
            vg = v[0].gather(1, idx[:, :, None].expand(-1, -1, self.D))
            att = (q @ kg.transpose(-1, -2)) / math.sqrt(self.D)  # (G, rep, m)
            att = att.masked_fill(~valid[:, None], float("-inf")).softmax(-1)
            y = att @ vg
            m = int(valid[0].sum())
        else:
            att = ((q @ k[0, :, :n].transpose(-1, -2)) / math.sqrt(self.D)).softmax(-1)
            y = att @ v[0, :, :n]
            m = n
        x = x + self.o(y.reshape(1, 1, -1))
        return x + self.mlp(self.n2(x)), m


class LoopedMSALM(nn.Module):
    def __init__(
        self,
        d: int = 256,
        gdn_heads: int = 4,
        n_prelude: int = 2,
        loops: int = 3,
        attn_heads: int = 4,
        groups: int = 2,
        d_idx: int = 32,
        block: int = 16,
        topk: int = 4,
        vocab: int = 256,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        lin = lambda i, o: nn.Linear(i, o, bias=False)
        self.blocks = nn.ModuleList(StatefulBlock(d, gdn_heads, lin, "gdn") for _ in range(n_prelude))
        self.core = MSACore(d, attn_heads, groups, d_idx, block, topk)
        self.loops = loops
        self.loop_emb = nn.Parameter(torch.zeros(loops, d))
        self.norm = nn.RMSNorm(d)
        self.sparse = True  # False: dense attention over the cache (the warmup phase, and the dense ablation)

    def prelude(self, tokens, states=None):
        states = states or [None] * len(self.blocks)
        x = self.embed(tokens)
        new = []
        for blk, s in zip(self.blocks, states, strict=True):
            x, s = blk(x, s)
            new.append(s)
        return x, new

    def forward(self, tokens: torch.Tensor, kl: bool = False, loops: int | None = None):
        """Parallel form over a whole window. -> logits, kl (mean over loops) or None, mean attended keys/query."""
        h0, _ = self.prelude(tokens)
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        cache = self.core.cache_entries(h0, pos)
        x, kls, att = h0, [], []
        for l in range(loops or self.loops):
            x, k, a = self.core(x + self.loop_emb[min(l, self.loops - 1)], cache, pos, self.sparse, kl)
            att.append(a)
            if k is not None:
                kls.append(k)
        logits = self.norm(x) @ self.embed.weight.t()
        return logits, (torch.stack(kls).mean() if kls else None), torch.stack(att).mean()

    def loss(self, x: torch.Tensor, kl_weight: float = 1.0):
        logits, kl, att = self(x[:, :-1], kl=kl_weight > 0)
        ce = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())
        info = {"ce": ce.item(), "attended": att.item()}
        if kl is None:
            return ce, info
        info["index_kl"] = kl.item()
        return ce + kl_weight * kl, info

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new: int, loops: int | None = None):
        """Greedy, batch 1, with a preallocated K/V + index-key cache. Prefill: the parallel form fills the cache;
        decode: one GDN prelude step, one cache write, then `loops` sparse core steps per byte."""
        loops = loops or self.loops
        T0 = prompt.shape[1]
        cap = T0 + n_new
        c = self.core
        K = torch.zeros(1, c.G, cap, c.D)
        V = torch.zeros(1, c.G, cap, c.D)
        KI = torch.zeros(1, cap, c.d_idx)
        h0, states = self.prelude(prompt)
        k, v, ki = c.cache_entries(h0, torch.arange(T0))
        K[:, :, :T0], V[:, :, :T0], KI[:, :T0] = k, v, ki
        x = h0
        pos = torch.arange(T0)
        cache = (K[:, :, :T0], V[:, :, :T0], KI[:, :T0])
        for l in range(loops):
            x, _, _ = c(x + self.loop_emb[min(l, self.loops - 1)], cache, pos, self.sparse)
        logits = self.norm(x[:, -1:]) @ self.embed.weight.t()
        out, attended = [], 0
        for i in range(n_new):
            tok = logits.argmax(-1)
            out.append(tok)
            if i == n_new - 1:
                break
            t = T0 + i
            h, states = self.prelude(tok, states)
            k, v, ki = c.cache_entries(h, torch.tensor([t]))
            K[:, :, t : t + 1], V[:, :, t : t + 1], KI[:, t : t + 1] = k, v, ki
            x = h
            for l in range(loops):
                x, m = c.step(x + self.loop_emb[min(l, self.loops - 1)], (K, V, KI), t, self.sparse)
                attended += m
            logits = self.norm(x) @ self.embed.weight.t()
        steps = max(1, (n_new - 1) * loops)
        return torch.cat(out, 1), {"attended_keys_per_loop": attended / steps, "cache_len": cap}

    def cache_bytes(self, T: int, dtype_bytes: int = 2) -> int:
        """K + V + index keys for T bytes (one cache for all loops)."""
        c = self.core
        return T * (2 * c.G * c.D + c.d_idx) * dtype_bytes
