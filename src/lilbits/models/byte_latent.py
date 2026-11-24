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
    if max_patch:
        run = torch.zeros(entropy.shape[0], dtype=torch.long, device=entropy.device)
        for t in range(entropy.shape[1]):
            run = torch.where(b[:, t], torch.zeros_like(run), run + 1)
            force = run >= max_patch
            b[:, t] |= force
            run[force] = 0
    return b


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
        B, _T, _ = h.shape
        pooled = h.new_full((B, n_patches, h.shape[-1]), float("-inf"))
        pooled = pooled.scatter_reduce(1, pid[..., None].expand_as(h), h, reduce="amax", include_self=True)
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        q = self.q(pooled)
        k, v = self.kv(h).chunk(2, -1)
        H, D = self.heads, q.shape[-1] // self.heads
        sh = lambda t: t.view(B, -1, H, D).transpose(1, 2)
        mask = pid[:, None, :] == torch.arange(n_patches, device=h.device)[None, :, None]  # (B, P, T)
        y = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask[:, None])
        y = torch.nan_to_num(y)  # empty (padding) patches
        return q + self.o(y.transpose(1, 2).reshape(B, n_patches, -1))


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
        """x: (B, T) bytes, boundaries: (B, T) patch starts. Returns next-byte logits (B, T, 256)."""
        B, T = x.shape
        pid = patch_ids(boundaries).clamp(max=self.cfg.max_patches - 1)
        P = int(pid.max()) + 1
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for blk in self.encoder:
            h = blk(h)
        z = self.pool(h, pid, P) + self.patch_pos(torch.arange(P, device=x.device))
        for blk in self.latent:  # causal over patches
            z = blk(z)
        # byte in patch j reads latent output j-1 (all of whose bytes precede it); patch 0 reads BOS
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
