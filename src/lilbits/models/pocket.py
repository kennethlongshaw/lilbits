"""Pocket LM: a byte-level model for one CPU core and a small RAM budget, where every trick is a switch.

    bytes in (no tokenizer; the only vocab table is 256 byte embeddings)
      -> BYTE ENCODER: small FP GDN block(s) over bytes, streaming state
      -> ROUTER (switch `router`): which bytes open a patch. The body runs once per patch, not once per byte.
           "pair"  p_t = sigmoid(v[x_{t-1}] + u[x_t]), initialized to space patching (a patch opens after a space
                   or punctuation byte), trainable through the H-Net ratio loss; patches are capped at `max_patch`
                   bytes. A function of raw bytes and a counter only, so chunked and byte-by-byte runs make the same
                   decisions bit for bit (what parallel verify needs).
           "hnet"  H-Net cosine routing over encoder states. More expressive, but decisions near p = 0.5 can flip
                   between chunked and streaming numerics, which breaks exact verify. Kept for comparison.
           "none"  every byte is a patch (the body runs per byte, as example 09 does per token).
      -> BODY: the looped index LM (models/looped_index.py) over patch vectors: LittleBit GDN prelude, looped core
         (FP, or shared sub-1-bit signs + per-loop scales after `convert_pocket_core`; base scales INT8 with
         scale_bits=8), sparse token index over patches, coda (LittleBit 1 bpw by default: an FP16 coda block
         was two thirds of the deployed bytes); decode may halt the loop early.
      -> UPSAMPLE: each byte reads its patch's body output (H-Net confidence smoothing).
      -> BYTE DECODER: small FP GDN block(s) -> next-byte logits (tied with the byte embedding).
      -> MTP (switch `n_mtp`): byte draft heads; drafts are checked by one parallel pass with fixed loop count,
         and a rejection rolls every state back (encoder, decoder, router, body cache and index).

Causality: a patch's body input is the encoder state at its first byte, and every byte of the patch reads that
patch's output, so byte t only sees bytes <= t, and streaming decode computes what the training forward computes.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.measure import size_report
from lilbits.models.byte_latent import DynamicChunker
from lilbits.models.decision import StatefulBlock
from lilbits.models.looped_index import Cache, LoopedIndexConfig, LoopedIndexLM, MTPHead, convert_core_to_littlebit
from lilbits.models.stateful_lm import StatefulLM

N_BYTES = 256
SEPARATORS = b" \t\n\r"  # a patch opens at the first byte after whitespace (BLT-style space patching)


@dataclass
class PocketConfig:
    d_byte: int = 64
    d_body: int = 256
    heads: int = 2
    n_enc: int = 1
    n_dec: int = 1
    router: str = "pair"  # pair | hnet | none
    max_patch: int = 8  # pair router: longest patch in bytes
    target_ratio: float = 4.0  # bytes per patch the ratio loss aims for
    ratio_weight: float = 0.03
    n_loops: int = 3
    block: int = 16  # patches per index block
    top_k: int = 8
    prelude_linear: str = "littlebit"
    prelude_bpw: float = 1.0
    core_bpw: float = 0.55
    coda_linear: str = "littlebit"  # an FP coda block would be ~2/3 of the deployed bytes
    coda_bpw: float = 1.0
    scale_bits: int = 8
    n_mtp: int = 2


class PairRouter(nn.Module):
    """p_t = sigmoid(v[x_{t-1}] + u[x_t]); starts as space patching. Exact under any chunking."""

    def __init__(self, init_logit: float = 4.0):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(N_BYTES))
        v = torch.full((N_BYTES,), -init_logit)
        v[list(SEPARATORS)] = init_logit
        self.v = nn.Parameter(v)

    def forward(self, x: torch.Tensor, prev_byte: torch.Tensor | None) -> torch.Tensor:
        prev = torch.cat([prev_byte if prev_byte is not None else torch.full_like(x[:, :1], -1), x[:, :-1]], 1)
        logit = self.u[x] + torch.where(prev >= 0, self.v[prev.clamp_min(0)], torch.full_like(self.u[x], 30.0))
        return torch.sigmoid(logit)


@dataclass
class PocketCache:
    enc: list
    dec: list
    body: Cache
    prev_h: torch.Tensor | None = None  # last encoder output (hnet router)
    prev_byte: torch.Tensor | None = None  # last byte (pair router)
    run: torch.Tensor | None = None  # bytes since the last boundary (pair router cap)
    zbar: torch.Tensor | None = None  # smoothed body output of the open patch
    pos: int = 0
    patches: int = 0

    def snapshot(self) -> tuple:
        return (
            list(self.enc),
            list(self.dec),
            self.body.snapshot(),
            self.prev_h,
            self.prev_byte,
            self.run,
            self.zbar,
            self.pos,
            self.patches,
        )

    def restore(self, snap: tuple) -> None:
        enc, dec, body, self.prev_h, self.prev_byte, self.run, self.zbar, self.pos, self.patches = snap
        self.enc, self.dec = list(enc), list(dec)
        self.body.restore(body)

    def state_bytes(self) -> dict:
        def b(xs):
            out = 0
            for s in xs:
                for t in s if isinstance(s, tuple) else (s,):
                    if t is not None:
                        out += t.numel() * t.element_size()
            return out

        return {"byte_gdn_bytes": b(self.enc) + b(self.dec), **self.body.state_bytes(), **self.body.index.nbytes()}


class PocketLM(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()
        if cfg.router not in ("pair", "hnet", "none"):
            raise ValueError(f"unknown router {cfg.router!r}")
        self.cfg = cfg
        d, D = cfg.d_byte, cfg.d_body

        def lin(i, o):
            return nn.Linear(i, o, bias=False)

        self.embed = nn.Embedding(N_BYTES, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.encoder = nn.ModuleList(StatefulBlock(d, cfg.heads, lin) for _ in range(cfg.n_enc))
        self.pair = PairRouter() if cfg.router == "pair" else None
        self.chunker = DynamicChunker(d) if cfg.router == "hnet" else None
        self.up = nn.Linear(d, D, bias=False)
        self.body = LoopedIndexLM(
            LoopedIndexConfig(
                vocab_size=1,  # inputs are patch vectors; the body's embedding and head are unused
                d_model=D,
                n_heads=cfg.heads,
                n_loops=cfg.n_loops,
                block=cfg.block,
                top_k=cfg.top_k,
                prelude_linear=cfg.prelude_linear,
                prelude_bpw=cfg.prelude_bpw,
                core_bpw=cfg.core_bpw,
                coda_linear=cfg.coda_linear,
                coda_bpw=cfg.coda_bpw,
                scale_bits=cfg.scale_bits,
                n_mtp=0,
            )
        )
        self.down = nn.Linear(D, d, bias=False)
        self.residual = nn.Linear(d, d, bias=False)
        self.decoder = nn.ModuleList(StatefulBlock(d, cfg.heads, lin) for _ in range(cfg.n_dec))
        self.norm = nn.RMSNorm(d)
        self.mtp = nn.ModuleList(MTPHead(d) for _ in range(cfg.n_mtp))
        self.timing: dict | None = None  # set to {} to accumulate seconds per part (byte layers / router / body)

    @contextmanager
    def _timed(self, part: str):
        if self.timing is None:
            yield
            return
        t = time.perf_counter()
        yield
        self.timing[part] = self.timing.get(part, 0.0) + time.perf_counter() - t

    def new_cache(self, batch: int) -> PocketCache:
        return PocketCache([None] * len(self.encoder), [None] * len(self.decoder), self.body.new_cache(batch))

    def logits(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.embed.weight.t()

    def _route(self, x: torch.Tensor, h: torch.Tensor, cache: PocketCache) -> tuple[torch.Tensor, torch.Tensor]:
        """Boundary probs p and decisions b, (B, L) each."""
        cfg = self.cfg
        if cfg.router == "none":
            return torch.ones_like(h[..., 0]), torch.ones_like(x, dtype=torch.bool)
        if cfg.router == "hnet":
            assert self.chunker is not None
            ch = self.chunker
            if cache.prev_h is None:
                p, b = ch.route(h)
            else:
                cos = F.cosine_similarity(ch.q(h), ch.k(torch.cat([cache.prev_h, h[:, :-1]], 1)), dim=-1)
                p = ((1 - cos) / 2).clamp(0, 1)
                b = p >= 0.5
            cache.prev_h = h[:, -1:]
            return p, b
        assert self.pair is not None
        p = self.pair(x, cache.prev_byte)
        want = p >= 0.5
        run = cache.run if cache.run is not None else torch.zeros_like(x[:, 0])
        bs = []
        for t in range(x.shape[1]):  # cap patch length: a counter, so the decision stays exact
            bt = want[:, t] | (run >= cfg.max_patch) | (cache.pos + t == 0)
            run = torch.where(bt, torch.ones_like(run), run + 1)
            bs.append(bt)
        cache.run, cache.prev_byte = run, x[:, -1:]
        return p, torch.stack(bs, 1)

    def forward(self, x: torch.Tensor, cache: PocketCache, halt_eps: float | None = None, stats=None):
        """Run bytes x (B, L) from the cache's position on. Returns (hidden (B, L, d_byte), aux dict).
        Batch > 1 is for training from a fresh cache (patch counts differ per row, padded at the end)."""
        stats = {} if stats is None else stats
        B, L = x.shape
        with self._timed("byte_layers"):
            h = self.embed(x)
            enc = []
            for blk, s in zip(self.encoder, cache.enc, strict=True):
                h, s = blk(h, s)
                enc.append(s)
            cache.enc = enc
        with self._timed("router"):
            p, b = self._route(x, h, cache)
        cnt = b.sum(1)
        n = int(cnt.max())
        cid = b.long().cumsum(1)  # 0: the byte belongs to the patch opened before this call
        with self._timed("body"):
            if n > 0:
                assert B == 1 or cache.patches == 0, "streaming several rows needs equal patch counts"
                slot = torch.where(b, cid - 1, torch.full_like(cid, n))
                chunks = h.new_zeros(B, n + 1, h.shape[-1]).scatter(1, slot[..., None].expand_as(h), h)[:, :n]
                z = self.down(self.body.process(self.up(chunks), cache.body, halt_eps, stats))
                pc = p.new_zeros(B, n + 1).scatter(1, slot, p)[:, :n]
            else:
                z, pc = h.new_zeros(B, 0, h.shape[-1]), p.new_zeros(B, 0)
        with self._timed("byte_layers"):
            prev = cache.zbar if cache.zbar is not None else h.new_zeros(B, h.shape[-1])
            smoothed = [prev]
            for c in range(n):  # H-Net smoothing: z_bar_c = P_c z_c + (1 - P_c) z_bar_{c-1}
                prev = pc[:, c, None] * z[:, c] + (1 - pc[:, c, None]) * prev
                smoothed.append(prev)
            zs = torch.stack(smoothed, 1)
            if n > 0:
                cache.zbar = zs.gather(1, cnt.view(B, 1, 1).expand(B, 1, zs.shape[-1]))[:, 0]
            zb = zs.gather(1, cid[..., None].expand(B, L, zs.shape[-1]))
            conf = torch.where(b, p, 1 - p)
            conf = conf + (1 - conf).detach()  # straight-through: forward 1, gradient of the confidence
            y = zb * conf[..., None] + self.residual(h)
            dec = []
            for blk, s in zip(self.decoder, cache.dec, strict=True):
                y, s = blk(y, s)
                dec.append(s)
            cache.dec = dec
            y = self.norm(y)
        cache.pos += L
        cache.patches += n
        return y, {"p": p, "b": b, "n": n}

    # ---- training -------------------------------------------------------------------------------------

    def training_loss(self, x: torch.Tensor, mtp: bool = True, mask: torch.Tensor | None = None):
        """Next-byte loss + ratio loss (learned routers) + MTP losses. x: (B, T) bytes. mask (B, T - 1): weight
        per target byte (e.g. answer bytes only for instruct tuning); MTP losses are then skipped."""
        y, aux = self(x, self.new_cache(x.shape[0]))
        ce = F.cross_entropy(self.logits(y[:, :-1]).flatten(0, 1), x[:, 1:].flatten(), reduction="none")
        if mask is None:
            loss = ce.mean()
        else:
            m = mask.flatten().to(ce)
            loss = (ce * m).sum() / m.sum().clamp_min(1)
            mtp = False
        info = {"lm": loss.item(), "bytes_per_patch": aux["b"].numel() / max(1, int(aux["b"].sum()))}
        if self.cfg.router != "none":
            r = DynamicChunker.ratio_loss(aux["p"], aux["b"], self.cfg.target_ratio)
            loss = loss + self.cfg.ratio_weight * r
            info["ratio"] = r.item()
        if mtp:
            hj = y
            for j, head in enumerate(self.mtp, start=1):
                hj = head(hj[:, :-1], self.embed(x[:, j:]))
                lj = F.cross_entropy(self.logits(hj[:, :-1]).flatten(0, 1), x[:, j + 1 :].flatten())
                info[f"mtp{j}"] = lj.item()
                loss = loss + lj / len(self.mtp)
        return loss, info

    # ---- decoding -------------------------------------------------------------------------------------

    def _drafts(self, y_last, t1, k):
        drafts, hj, tok = [], y_last, t1
        for head in list(self.mtp)[:k]:
            hj = head(hj, self.embed(tok))
            tok = self.logits(hj).argmax(-1)
            drafts.append(tok)
        return drafts

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new: int, halt_eps: float | None = None):
        """Greedy byte decode (batch 1), one byte per step; the body runs only when a byte opens a patch."""
        cache, loop_stats = self.new_cache(prompt.shape[0]), {}
        y, _ = self(prompt, cache)
        start = cache.patches
        out = []
        for _ in range(n_new):
            tok = self.logits(y[:, -1:]).argmax(-1)
            out.append(tok)
            if len(out) == n_new:
                break
            y, _ = self(tok, cache, halt_eps, loop_stats)
        loops = loop_stats.get("loops") or [self.cfg.n_loops]
        passes = cache.patches - start
        stats = {
            "body_passes": passes,
            "bytes_per_body_pass": (n_new - 1) / max(1, passes),
            "mean_loops": sum(loops) / len(loops),
            "cache": cache.state_bytes(),
        }
        return torch.cat([prompt, *out], 1), stats

    @torch.no_grad()
    def generate_speculative(self, prompt: torch.Tensor, n_new: int, k: int | None = None, adaptive: bool = True):
        """MTP byte drafts + one parallel verify pass with rollback (batch 1). Equals generate() with fixed N when
        routing is exact (pair / none)."""
        assert prompt.shape[0] == 1
        k_max = len(self.mtp)
        k = k_max if k is None else min(k, k_max)
        cache = self.new_cache(1)
        y, _ = self(prompt, cache)
        t1 = self.logits(y[:, -1:]).argmax(-1)
        drafts = self._drafts(y[:, -1:], t1, k)
        out: list[torch.Tensor] = []
        passes, proposed, accepted, rate = 0, 0, 0, 0.5
        pos_prop, pos_acc = [0] * k_max, [0] * k_max  # per draft position: proposed / accepted
        while sum(o.shape[1] for o in out) < n_new:
            seq = torch.cat([t1, *drafts], 1)
            snap = cache.snapshot()
            yv, _ = self(seq, cache)
            passes += 1
            preds = self.logits(yv).argmax(-1)
            a = 0
            while a < len(drafts) and drafts[a].item() == preds[0, a].item():
                a += 1
            proposed, accepted = proposed + len(drafts), accepted + a
            for j in range(len(drafts)):
                pos_prop[j] += 1
                pos_acc[j] += int(j < a)
            if a < len(drafts):
                cache.restore(snap)
                yv, _ = self(seq[:, : a + 1], cache)
                passes += 1
            out.append(seq[:, : a + 1])
            t1 = self.logits(yv[:, -1:]).argmax(-1)
            if adaptive and drafts:
                rate = 0.7 * rate + 0.3 * a / len(drafts)
                k = max(1, k - 1) if rate < 0.3 else min(k_max, k + 1) if rate > 0.7 else k
            drafts = self._drafts(yv[:, -1:], t1, k)
        gen = torch.cat(out, 1)[:, :n_new]
        stats = {
            "passes": passes,
            "bytes_per_pass": n_new / passes,
            "acceptance": accepted / max(1, proposed),
            "acceptance_by_draft": [a / p if p else None for a, p in zip(pos_acc, pos_prop, strict=True)],
            "cache": cache.state_bytes(),
        }
        return torch.cat([prompt, gen], 1), stats


def fp16_baseline(d: int) -> StatefulLM:
    """The comparison model: a 2-layer byte-level FP GDN LM (head dim 64), deployed in FP16."""
    return StatefulLM(N_BYTES, d, d // 64, 2, partial(nn.Linear, bias=False), mixer="gdn")


def matched_fp16_width(target_bytes: float) -> int:
    """Width (multiple of 64) of `fp16_baseline` whose deployed size is closest to target_bytes."""
    best, best_gap = 64, math.inf
    for d in range(64, 8193, 64):
        with torch.device("meta"):
            gap = abs(size_report(fp16_baseline(d))["deployed_total_bytes"] - target_bytes)
        if gap < best_gap:
            best, best_gap = d, gap
    return best


def convert_pocket_core(model: PocketLM, ranks: dict[str, int] | None = None) -> None:
    """Stage 2: the body's FP looped core -> sub-1-bit LittleBit (Dual-SVID from the FP weights), per-loop scales
    kept, base scales at cfg.scale_bits."""
    convert_core_to_littlebit(model.body, ranks=ranks)


__all__ = [
    "PairRouter",
    "PocketCache",
    "PocketConfig",
    "PocketLM",
    "convert_pocket_core",
    "fp16_baseline",
    "matched_fp16_width",
]
