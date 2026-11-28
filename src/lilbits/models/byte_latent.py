"""Tokenizer-free byte-level models with dynamic patching: no 200k-entry vocabulary, shorter sequences inside.

Two published designs, built from the repo's blocks (any linear can be FP / LittleBit / BitNet):

BLT, Byte Latent Transformer (Pagnoni et al., 2024, arXiv:2412.09871)
    bytes -> byte + hash n-gram embeddings -> LOCAL ENCODER (small causal transformer over bytes)
          -> patch pooling (cross-attention: one query per patch, attending to its own bytes)
          -> LATENT TRANSFORMER over patches (where most compute goes)
          -> LOCAL DECODER over bytes (each byte reads the latent output of the *previous* patch) -> next byte
    Patches come from a small byte LM: a new patch starts where next-byte entropy is high (global threshold)
    or jumps (relative threshold), so predictable runs become long patches.

H-Net (Hwang, Wang & Gu, 2025, "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")
    bytes -> ENCODER (Mamba2) -> ROUTER: p_t = (1 - cos(q_t, k_{t-1})) / 2, boundary where p_t >= 0.5
          -> keep boundary positions as chunks -> MAIN network over chunks
          -> DECHUNK: EMA smoothing with p, repeat each chunk over its bytes, straight-through confidence
          -> + encoder residual -> DECODER (Mamba2) -> next byte
    Chunking is learned end to end; a ratio loss pulls the boundary rate toward 1/N (target compression N).

Both are causal: a byte's prediction only depends on bytes at or before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.ffn import SwiGLU
from lilbits.mixers.recurrent import Mamba2
from lilbits.models.tinylm import Config, TransformerBlock, make_linear_factory

N_BYTES = 256


# ---------------------------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------------------------
class HashNgramEmbedding(nn.Module):
    """BLT byte embedding: e_i = E_byte(x_i) + sum_n E_n[hash(x_{i-n+1..i}) mod buckets], n in ngrams."""

    def __init__(self, d: int, ngrams=(3, 4, 5, 6), buckets: int = 4096):
        super().__init__()
        self.ngrams, self.buckets = ngrams, buckets
        self.byte = nn.Embedding(N_BYTES, d)
        self.tables = nn.ModuleList(nn.Embedding(buckets, d) for _ in ngrams)
        for t in [self.byte, *self.tables]:
            assert isinstance(t, nn.Embedding)
            nn.init.normal_(t.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.byte(x)
        xl = x.long()
        for n, table in zip(self.ngrams, self.tables, strict=False):
            h = torch.zeros_like(xl)
            for j in range(n):  # rolling polynomial hash of the n bytes ending at i (zero-padded on the left)
                h = h * 257 + F.pad(xl, (j, 0))[:, : xl.shape[1]] + 1
            out = out + table(h % self.buckets)
        return out


def block_stack(n: int, d: int, heads: int, linear, mixer: str = "softmax") -> nn.ModuleList:
    hidden = round(d * 8 / 3 / 8) * 8
    return nn.ModuleList(TransformerBlock(d, heads, hidden, linear, mixer=mixer) for _ in range(n))


def patch_ids(boundaries: torch.Tensor) -> torch.Tensor:
    """boundaries (B, T) bool, True where a new patch starts -> patch index of every byte (B, T).

    Byte 0 always starts a patch.
    """
    b = boundaries.clone()
    b[:, 0] = True
    return b.long().cumsum(1) - 1


# ---------------------------------------------------------------------------------------------
# BLT
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def entropy_boundaries(
    entropy: torch.Tensor,
    theta_global: float | None = None,
    theta_rel: float | None = None,
    max_patch: int | None = None,
) -> torch.Tensor:
    """BLT patching. entropy[:, t] = H(p(x_t | x_<t)) from a small byte LM. A patch starts at t when
    entropy[t] > theta_global, or (monotonic constraint) when entropy[t] - entropy[t-1] > theta_rel.
    max_patch forces a boundary after that many bytes."""
    b = torch.zeros_like(entropy, dtype=torch.bool)
    if theta_global is not None:
        b |= entropy > theta_global
    if theta_rel is not None:
        b[:, 1:] |= (entropy[:, 1:] - entropy[:, :-1]) > theta_rel
    b[:, 0] = True
    return cap_patches(b, max_patch)


def cap_patches(b: torch.Tensor, max_patch: int | None) -> torch.Tensor:
    """Force a patch start every max_patch bytes inside long patches, in one vectorized pass: the k-th byte after
    a natural start is a forced start iff k % max_patch == 0 (same result as a running counter, no Python loop)."""
    if not max_patch:
        return b
    t = torch.arange(b.shape[1], device=b.device)
    start = torch.where(b, t, torch.zeros_like(t)).cummax(1).values
    return b | ((t - start) % max_patch == 0)


def spacelike(x: torch.Tensor) -> torch.Tensor:
    """SpaceByte (Slagle 2024): a byte is space-like unless it is an ASCII letter / digit or a UTF-8 continuation."""
    x = x.long()
    alnum = ((x >= 48) & (x <= 57)) | ((x >= 65) & (x <= 90)) | ((x >= 97) & (x <= 122))
    return ~(alnum | ((x >= 0x80) & (x < 0xC0)))


@torch.no_grad()
def space_boundaries(x: torch.Tensor, max_patch: int | None = 16) -> torch.Tensor:
    """SpaceByte-style patching with no patcher model: a patch starts right after the first space-like byte of a run
    (so a word and the spaces / punctuation that end it share a patch). Start at s needs only x_{<s}, so decode
    knows the next patch start before it samples that byte. (B, T) -> (B, T) bool."""
    sp = spacelike(x)
    b = torch.zeros_like(sp)
    b[:, 1:] = sp[:, :-1]
    b[:, 2:] &= ~sp[:, :-2]
    b[:, 0] = True
    return cap_patches(b, max_patch)


@torch.no_grad()
def byte_entropy(entropy_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Entropy of the entropy model's prediction for each byte x_t given x_<t (position 0 gets max entropy)."""
    logp = F.log_softmax(entropy_model(x[:, :-1]), -1)
    h = -(logp.exp() * logp).sum(-1)
    return F.pad(h, (1, 0), value=math.log(N_BYTES))


@dataclass
class BLTConfig:
    seq_len: int = 256  # bytes
    d_local: int = 64
    d_latent: int = 128
    heads: int = 4
    n_enc: int = 1
    n_latent: int = 2
    n_dec: int = 1
    max_patches: int = 128
    linear: str = "fp"  # latent transformer linears: fp | littlebit | bitnet
    local_linear: str = "fp"
    bpw: float = 1.0


class PatchPooling(nn.Module):
    """One query per patch (initialised from the patch's max-pooled bytes) cross-attends to its own bytes."""

    def __init__(self, d_local: int, d_latent: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d_local, d_latent, bias=False)
        self.kv = nn.Linear(d_local, 2 * d_latent, bias=False)
        self.o = nn.Linear(d_latent, d_latent, bias=False)

    def forward(self, h: torch.Tensor, pid: torch.Tensor, n_patches: int) -> torch.Tensor:
        B, T, d = h.shape
        pooled = h.new_full((B, n_patches, d), float("-inf"))
        pooled = pooled.scatter_reduce(1, pid[..., None].expand_as(h), h, reduce="amax", include_self=True)
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        q = self.q(pooled)
        k, v = self.kv(h).chunk(2, -1)
        H, D = self.heads, q.shape[-1] // self.heads
        # segment softmax: each byte scores only against its own patch's query, O(T) instead of a (P, T) mask
        qb = q.gather(1, pid[..., None].expand(B, T, q.shape[-1])).view(B, T, H, D)
        s = (qb * k.view(B, T, H, D)).sum(-1) / math.sqrt(D)  # (B, T, H)
        idx = pid[..., None].expand(B, T, H)
        m = s.new_full((B, n_patches, H), float("-inf")).scatter_reduce(1, idx, s, reduce="amax", include_self=True)
        w = (s - m.gather(1, idx)).exp()
        den = s.new_zeros(B, n_patches, H).scatter_add(1, idx, w)
        wv = w[..., None] * v.view(B, T, H, D)
        num = v.new_zeros(B, n_patches, H, D).scatter_add(1, idx[..., None].expand(B, T, H, D), wv)
        y = torch.nan_to_num(num / den[..., None])  # empty (padding) patches
        return q + self.o(y.reshape(B, n_patches, -1))


class ByteLatentTransformer(nn.Module):
    def __init__(self, cfg: BLTConfig):
        super().__init__()
        self.cfg = cfg
        lat = make_linear_factory(Config(linear=cfg.linear, bpw=cfg.bpw, init="svd", itq_iters=20))
        loc = make_linear_factory(Config(linear=cfg.local_linear, bpw=cfg.bpw, init="svd", itq_iters=20))
        self.embed = HashNgramEmbedding(cfg.d_local)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_local)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.encoder = block_stack(cfg.n_enc, cfg.d_local, cfg.heads, loc)
        self.pool = PatchPooling(cfg.d_local, cfg.d_latent, cfg.heads)
        self.patch_pos = nn.Embedding(cfg.max_patches, cfg.d_latent)
        self.latent = block_stack(cfg.n_latent, cfg.d_latent, cfg.heads, lat)
        self.bos = nn.Parameter(torch.zeros(cfg.d_latent))  # what bytes of patch 0 read (no earlier patch exists)
        # decoder cross-attention with exactly one visible key (the previous patch) reduces to this projection
        self.from_latent = nn.Linear(cfg.d_latent, cfg.d_local, bias=False)
        self.decoder = block_stack(cfg.n_dec, cfg.d_local, cfg.heads, loc)
        self.norm = nn.RMSNorm(cfg.d_local)
        self.head = nn.Linear(cfg.d_local, N_BYTES, bias=False)

    def forward(self, x: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """x: (B, T) bytes, boundaries: (B, T) patch starts (extra columns are ignored). Returns next-byte logits
        (B, T, 256)."""
        B, T = x.shape
        pid = patch_ids(boundaries[:, :T]).clamp(max=self.cfg.max_patches - 1)
        P = int(pid.max()) + 1
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for blk in self.encoder:
            h = blk(h)
        z = self.pool(h, pid, P) + self.patch_pos(torch.arange(P, device=x.device))
        for blk in self.latent:  # causal over patches
            z = blk(z)
        # byte in patch j reads latent output j-1 (all of whose bytes precede it); patch 0 reads BOS.
        # (Tried: letting a patch's last byte read its own just-closed patch, so the next patch's first byte is
        # predicted with fresh context. It was 0.03-0.04 bits/byte worse for both patchers at 800 steps.)
        z_prev = torch.cat([self.bos.expand(B, 1, -1), z[:, :-1]], 1)
        y = h + self.from_latent(z_prev.gather(1, pid[..., None].expand(B, T, z.shape[-1])))
        for blk in self.decoder:
            y = blk(y)
        return self.head(self.norm(y))


# ---------------------------------------------------------------------------------------------
# H-Net
# ---------------------------------------------------------------------------------------------
class DynamicChunker(nn.Module):
    """H-Net routing + downsampling, and the matching smoothing + upsampling.

    p_t = (1 - cos(W_q x_t, W_k x_{t-1})) / 2 (p_0 = 1); boundary b_t = p_t >= 0.5.
    """

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        with torch.no_grad():  # start as identity, so boundaries initially mark representation changes
            self.q.weight.copy_(torch.eye(d))
            self.k.weight.copy_(torch.eye(d))

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = F.cosine_similarity(self.q(x[:, 1:]), self.k(x[:, :-1]), dim=-1)
        p = torch.cat([torch.ones_like(x[:, :1, 0]), (1 - cos) / 2], 1).clamp(0, 1)
        return p, p >= 0.5

    @staticmethod
    def downsample(x: torch.Tensor, b: torch.Tensor):
        """Gather boundary positions into a padded chunk sequence. Returns (chunks, chunk index per byte, n chunks)."""
        cid = patch_ids(b)
        n = int(cid.max()) + 1
        B, T, d = x.shape
        chunks = x.new_zeros(B, n, d)
        idx = torch.where(b, cid, torch.full_like(cid, n))  # non-boundary bytes go to a dropped slot
        chunks = torch.cat([chunks, x.new_zeros(B, 1, d)], 1).scatter(1, idx[..., None].expand(B, T, d), x)[:, :n]
        return chunks, cid, n

    @staticmethod
    def upsample(z: torch.Tensor, p: torch.Tensor, b: torch.Tensor, cid: torch.Tensor) -> torch.Tensor:
        """EMA-smooth chunk outputs with their boundary probs, repeat over bytes, scale by STE confidence."""
        B, T = p.shape
        n = z.shape[1]
        p_chunk = torch.zeros(B, n + 1, device=p.device, dtype=p.dtype)
        idx = torch.where(b, cid, torch.full_like(cid, n))
        p_chunk = p_chunk.scatter(1, idx, p)[:, :n]
        smoothed, prev = [], torch.zeros_like(z[:, 0])
        for c in range(n):  # z_bar_c = P_c z_c + (1 - P_c) z_bar_{c-1}
            prev = p_chunk[:, c, None] * z[:, c] + (1 - p_chunk[:, c, None]) * prev
            smoothed.append(prev)
        zb = torch.stack(smoothed, 1).gather(1, cid[..., None].expand(B, T, z.shape[-1]))
        c = torch.where(b, p, 1 - p)
        c = c + (1 - c).detach()  # straight-through: forward 1, gradient of the confidence
        return zb * c[..., None]

    @staticmethod
    def ratio_loss(p: torch.Tensor, b: torch.Tensor, target_ratio: float) -> torch.Tensor:
        """H-Net: L = N/(N-1) * ((N-1) F G + (1-F)(1-G)), F = boundary fraction, G = mean prob; min at F = G = 1/N."""
        N = target_ratio
        Fr, G = b.float().mean(), p.mean()
        return N / (N - 1) * ((N - 1) * Fr * G + (1 - Fr) * (1 - G))


@dataclass
class HNetConfig:
    seq_len: int = 256
    d_outer: int = 64
    d_main: int = 128
    heads: int = 4
    n_enc: int = 1
    n_main: int = 2
    n_dec: int = 1
    target_ratio: float = 4.0  # bytes per chunk the ratio loss aims for
    ratio_weight: float = 0.03
    linear: str = "fp"  # main network linears
    outer_linear: str = "fp"
    bpw: float = 1.0


class _MambaBlock(nn.Module):
    def __init__(self, d: int, heads: int, linear):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.mix = Mamba2(d, heads, linear)
        self.mlp = SwiGLU(d, int(d * 8 / 3) // 8 * 8, linear)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.mlp(self.n2(x))


class HNet(nn.Module):
    """One-stage H-Net: Mamba2 encoder/decoder over bytes, transformer main network over learned chunks."""

    def __init__(self, cfg: HNetConfig):
        super().__init__()
        self.cfg = cfg
        main = make_linear_factory(Config(linear=cfg.linear, bpw=cfg.bpw, init="svd", itq_iters=20))
        outer = make_linear_factory(Config(linear=cfg.outer_linear, bpw=cfg.bpw, init="svd", itq_iters=20))
        self.embed = nn.Embedding(N_BYTES, cfg.d_outer)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.encoder = nn.ModuleList(_MambaBlock(cfg.d_outer, cfg.heads, outer) for _ in range(cfg.n_enc))
        self.chunker = DynamicChunker(cfg.d_outer)
        self.up_in = nn.Linear(cfg.d_outer, cfg.d_main, bias=False)
        self.chunk_pos = nn.Embedding(cfg.seq_len, cfg.d_main)
        nn.init.normal_(self.chunk_pos.weight, std=0.02)
        self.main = block_stack(cfg.n_main, cfg.d_main, cfg.heads, main)
        self.down_out = nn.Linear(cfg.d_main, cfg.d_outer, bias=False)
        self.residual = nn.Linear(cfg.d_outer, cfg.d_outer, bias=False)
        self.decoder = nn.ModuleList(_MambaBlock(cfg.d_outer, cfg.heads, outer) for _ in range(cfg.n_dec))
        self.norm = nn.RMSNorm(cfg.d_outer)
        self.head = nn.Linear(cfg.d_outer, N_BYTES, bias=False)

    def forward(self, x: torch.Tensor):
        """Returns (next-byte logits (B, T, 256), ratio loss, stats)."""
        h = self.embed(x)
        for blk in self.encoder:
            h = blk(h)
        p, b = self.chunker.route(h)
        chunks, cid, n = self.chunker.downsample(h, b)
        z = self.up_in(chunks) + self.chunk_pos(torch.arange(n, device=x.device))
        for blk in self.main:  # causal over chunks; chunk c only sees bytes up to its start
            z = blk(z)
        y = self.chunker.upsample(self.down_out(z), p, b, cid) + self.residual(h)
        for blk in self.decoder:
            y = blk(y)
        ratio = self.chunker.ratio_loss(p, b, self.cfg.target_ratio)
        return self.head(self.norm(y)), ratio, {"bytes_per_chunk": b.numel() / b.sum().item(), "chunks": n}


# ---------------------------------------------------------------------------------------------
# streaming decode: every network keeps a cache, the inner network runs once per patch
# ---------------------------------------------------------------------------------------------
def block_step(blk: nn.Module, x: torch.Tensor, cache: list) -> torch.Tensor:
    """One new position through a softmax TransformerBlock. x (B, 1, d); cache = [K, V] (grown in place)."""
    a = blk.attn
    B, T, C = x.shape
    H = a.n_heads
    h = blk.norm1(x)
    sh = lambda t: t.view(B, T, H, C // H).transpose(1, 2)
    q, k, v = a.q_norm(sh(a.q(h))), a.k_norm(sh(a.k(h))), sh(a.v(h))
    if cache:
        k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
    cache[:] = [k, v]
    x = x + a.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, C))
    return x + blk.mlp(blk.norm2(x))


def mamba_block_step(blk: nn.Module, x: torch.Tensor, state: list) -> torch.Tensor:
    """One new position through an H-Net _MambaBlock. x (B, 1, d); state = [conv tail, SSM state] (in place)."""
    m = blk.mix
    B, _, C = x.shape
    H, P, N = m.n_heads, m.head_dim, m.d_state
    u = blk.n1(x)
    xp = m.x(u)  # (B, 1, C), before the short conv
    k = m.conv.kernel
    tail = state[0] if state else xp.new_zeros(B, k - 1, C)
    win = torch.cat([tail, xp], 1)  # last k inputs
    xs = F.silu((win * m.conv.conv.weight[:, 0, :].T[None]).sum(1)).view(B, H, P)
    b, c = m.bc(u).view(B, 2, H, N).unbind(1)
    dt = F.softplus(m.dt(u)[:, 0] + m.dt_bias)  # (B, H)
    S = state[1] if state else u.new_zeros(B, H, N, P)
    S = S * (-m.A_log.exp() * dt).exp()[..., None, None] + b[..., None] * (xs * dt[..., None])[:, :, None]
    y = torch.einsum("bhkv,bhk->bhv", S, c) + m.D[:, None] * xs
    y = m.norm(y) * F.silu(m.z(u).view(B, H, P))
    state[:] = [win[:, 1:], S]
    x = x + m.o(y.reshape(B, 1, C))
    return x + blk.mlp(blk.n2(x))


def lm_step(lm: nn.Module, byte: torch.Tensor, t: int, cache: list) -> torch.Tensor:
    """TinyLM (softmax blocks) next-byte logits after feeding one byte at position t. byte (B,); cache: per layer."""
    x = lm.tok(byte[:, None]) + lm.pos.weight[t]
    if not cache:
        cache.extend([] for _ in lm.blocks)
    for blk, c in zip(lm.blocks, cache, strict=True):
        x = block_step(blk, x, c)
    return lm.head(lm.norm(x))[:, 0]


_SPACELIKE = frozenset(i for i in range(256) if bool(spacelike(torch.tensor(i))))


class BLTStream:
    """Batch-1 streaming BLT. `feed(byte)` returns next-byte logits (256,). Per byte: hash embedding (hashes in
    plain ints), local encoder and local decoder steps, plus the entropy model's step when patching="entropy";
    once per patch: pooling and one latent step. Everything is KV-cached, so nothing is recomputed. Produces the
    logits of `model(x, boundaries)`."""

    def __init__(
        self,
        model: ByteLatentTransformer,
        patching: str = "space",
        entropy_model=None,
        theta=0.0,
        max_patch: int | None = 16,
    ):
        assert patching in ("space", "entropy") and (patching == "space" or entropy_model is not None)
        self.m, self.patching, self.ent, self.theta, self.max_patch = model, patching, entropy_model, theta, max_patch
        self.t, self.hist, self.n_patches, self.patch = 0, [], 0, []
        self.enc, self.dec, self.lat = ([[] for _ in s] for s in (model.encoder, model.decoder, model.latent))
        self.ent_cache: list = []
        self.z_new = self.z_read = model.bos.view(1, 1, -1)
        self.next_starts = True
        self.calls = {"encoder": 0, "decoder": 0, "entropy": 0, "latent": 0}

    def _embed(self) -> torch.Tensor:
        e, xs = self.m.embed, self.hist
        out = e.byte.weight[xs[-1]]
        for n, table in zip(e.ngrams, e.tables, strict=False):
            h = 0
            for j in range(n):  # same rolling hash as HashNgramEmbedding, zero-padded on the left
                i = len(xs) - 1 - j
                h = h * 257 + (xs[i] if i >= 0 else 0) + 1
            out = out + table.weight[h % e.buckets]
        return out.view(1, 1, -1)

    def _next_starts(self, byte: int) -> bool:
        if self.max_patch and len(self.patch) >= self.max_patch:
            return True
        if self.patching == "entropy":
            logp = F.log_softmax(self._ent_logits, -1)
            return bool(-(logp.exp() * logp).sum() > self.theta)
        return byte in _SPACELIKE and (len(self.hist) < 2 or self.hist[-2] not in _SPACELIKE)

    @torch.no_grad()
    def feed(self, byte: int) -> torch.Tensor:
        m, t = self.m, self.t
        if self.next_starts:  # this byte opens a patch: its bytes read the newest closed patch
            self.z_read = self.z_new
        self.hist.append(byte)
        h = self._embed() + m.pos.weight[t]
        for blk, c in zip(m.encoder, self.enc, strict=True):
            h = block_step(blk, h, c)
        self.patch.append(h)
        self.calls["encoder"] += 1
        if self.patching == "entropy":
            self._ent_logits = lm_step(self.ent, torch.tensor([byte]), t, self.ent_cache)[0]
            self.calls["entropy"] += 1
        self.next_starts = self._next_starts(byte)
        if self.next_starts:  # the patch is complete: one pooling + latent step, read by the next patch's bytes
            hp = torch.cat(self.patch, 1)
            z = m.pool(hp, torch.zeros(1, hp.shape[1], dtype=torch.long), 1)
            z = z + m.patch_pos.weight[min(self.n_patches, m.cfg.max_patches - 1)]
            for blk, c in zip(m.latent, self.lat, strict=True):
                z = block_step(blk, z, c)
            self.z_new, self.patch, self.n_patches = z, [], self.n_patches + 1
            self.calls["latent"] += 1
        y = h + m.from_latent(self.z_read)
        for blk, c in zip(m.decoder, self.dec, strict=True):
            y = block_step(blk, y, c)
        self.calls["decoder"] += 1
        self.t += 1
        return m.head(m.norm(y))[0, 0]


class HNetStream:
    """Batch-1 streaming H-Net: Mamba2 encoder / decoder steps per byte (fixed-size state), the router decides from
    the encoder state, and the main network takes one KV-cached step per chunk. `feed(byte)` -> logits (256,)."""

    def __init__(self, model: HNet):
        self.m, self.t, self.n = model, 0, 0
        self.enc, self.dec, self.main = ([[] for _ in s] for s in (model.encoder, model.decoder, model.main))
        self.h_prev = None
        self.zbar = torch.zeros(1, 1, model.cfg.d_outer)
        self.calls = {"encoder": 0, "decoder": 0, "main": 0}

    @torch.no_grad()
    def feed(self, byte: int) -> torch.Tensor:
        m = self.m
        h = m.embed(torch.tensor([[byte]]))
        for blk, s in zip(m.encoder, self.enc, strict=True):
            h = mamba_block_step(blk, h, s)
        if self.h_prev is None:
            p = torch.ones(())
        else:
            cos = F.cosine_similarity(m.chunker.q(h), m.chunker.k(self.h_prev), dim=-1)[0, 0]
            p = ((1 - cos) / 2).clamp(0, 1)
        self.h_prev = h
        if p >= 0.5:  # new chunk: one main-network step, then EMA smoothing with p
            z = m.up_in(h) + m.chunk_pos.weight[self.n]
            for blk, c in zip(m.main, self.main, strict=True):
                z = block_step(blk, z, c)
            self.zbar = p * m.down_out(z) + (1 - p) * self.zbar
            self.n += 1
            self.calls["main"] += 1
        y = self.zbar + m.residual(h)  # straight-through confidence is exactly 1 in the forward pass
        for blk, s in zip(m.decoder, self.dec, strict=True):
            y = mamba_block_step(blk, y, s)
        self.calls["encoder"] += 1
        self.calls["decoder"] += 1
        self.t += 1
        return m.head(m.norm(y))[0, 0]


class LMStream:
    """Batch-1 streaming byte transformer (TinyLM, softmax blocks), KV-cached."""

    def __init__(self, model: nn.Module):
        self.m, self.t, self.cache = model, 0, []
        self.calls = {"layers": 0}

    @torch.no_grad()
    def feed(self, byte: int) -> torch.Tensor:
        out = lm_step(self.m, torch.tensor([byte]), self.t, self.cache)[0]
        self.t += 1
        self.calls["layers"] += 1
        return out


@torch.no_grad()
def greedy_generate(stream, prompt: list[int], n_new: int) -> tuple[list[int], float]:
    """Feed the prompt, then greedily generate n_new bytes. Returns (new bytes, decode seconds)."""
    import time

    for b in prompt[:-1]:
        stream.feed(b)
    logits, out = stream.feed(prompt[-1]), []
    t0 = time.perf_counter()
    for _ in range(n_new):
        b = int(logits.argmax())
        out.append(b)
        logits = stream.feed(b)
    return out, time.perf_counter() - t0
