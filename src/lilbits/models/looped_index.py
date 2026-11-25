"""Looped index LM: a small, lean decoder that reasons by looping, remembers through fixed-size state plus a sparse
token index, and decodes several tokens per pass with draft heads and parallel verify.

    tokens in (prompt, or k draft tokens to check)
      -> embed (tied with the LM head)
      -> PRELUDE: unlooped GDN blocks (LittleBit linears). Each token is written to the INDEX here.
      -> INDEX: 4-bit K/V + sign(K) signature per token; shallow and deep summaries per block of 64 tokens.
      -> LOOPED CORE x N: one shared block, per-loop FP scales over shared signs, a GDN state per loop,
         loop memory over this token's earlier passes, and hierarchical retrieval from the index.
         Every block of 64 tokens, the mean final-loop state goes back to the index as the deep summary.
      -> CODA: higher-precision (FP) GDN block, LM head, and sequential MTP heads that draft k more tokens.
      -> VERIFY (next pass): the drafts run through the full stack in parallel, which is also their write;
         the longest correct prefix is kept, a rejection rolls the index and every state back, and the
         acceptance rate sets k.

N rule: prefill and verify run a fixed N. Plain token-by-token decode may halt early when the core state stops
changing; the GDN states of the skipped loops are still advanced with the halted state (a write-only pass: the GDN
projections and state update only), so the next token sees consistent per-loop memories.

Memory tiers:
    within a token   loop memory        up to N vectors per position, discarded after the token
    across tokens    GDN states         fixed size, one per prelude / core loop / coda layer
    across blocks    deep summaries     one per 64 tokens (plus a shallow one), cached
    raw recall       K/V + signatures   about (Dk + Dv) / 2 + 4 + Dk / 8 bytes per token, cached

Chunks are processed in segments that never cross a block boundary, so a chunk gives the same result as feeding
its tokens one at a time. That is what makes parallel verify exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.bpw_alloc import allocate_ranks, input_energy
from lilbits.layers.ffn import SwiGLU
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.layers.loop_scaled import convert_loop_linears, loop_fp_weights, loop_linear_factory, set_loop
from lilbits.mixers.loop_memory import LoopMemory
from lilbits.mixers.recurrent import GatedDeltaNet
from lilbits.mixers.sparse_index import IndexedRetrieval, TokenIndex
from lilbits.models.decision import StatefulBlock


@dataclass
class LoopedIndexConfig:
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 2
    n_prelude: int = 1
    n_coda: int = 1
    n_loops: int = 3
    block: int = 64  # tokens per index block (one shallow + one deep summary each)
    top_blocks: int = 2
    top_k: int = 16
    flat: float = 0.95  # normalized entropy of block scores above which retrieval falls back to dense
    prelude_linear: str = "littlebit"
    prelude_bpw: float = 1.0
    scale_bits: int = 16  # LittleBit h, g, l scales of the prelude and (once converted) the core: 16 or 8 (INT8)
    core_linear: str = "fp"  # stage 1 trains the core in FP; convert_core_to_littlebit() for stage 2
    core_bpw: float = 0.55
    coda_linear: str = "fp"  # "littlebit" puts the coda on LittleBit at coda_bpw (the FP coda can dominate size)
    coda_bpw: float = 1.0
    n_mtp: int = 2  # draft heads: up to n_mtp drafts per pass
    loop_memory: bool = True


def _linear(kind: str, bpw: float, scale_bits: int = 16):
    if kind == "fp":
        return partial(nn.Linear, bias=False)
    if kind == "littlebit":
        return partial(LittleBitLinear, bpw=bpw, residual=True, ste="clipped", scale_bits=scale_bits)
    raise ValueError(f"unknown linear {kind!r}")


class CoreBlock(nn.Module):
    """The shared looped block. `set_loop(core, i)` selects loop i's FP scales."""

    def __init__(self, cfg: LoopedIndexConfig):
        super().__init__()
        d, lin = cfg.d_model, loop_linear_factory(_linear(cfg.core_linear, cfg.core_bpw, cfg.scale_bits), cfg.n_loops)
        self.adapter = lin(2 * d, d)
        self.n1, self.n2, self.n3, self.n4 = (nn.RMSNorm(d) for _ in range(4))
        self.gdn = GatedDeltaNet(d, cfg.n_heads, lin, conv_size=1)  # per-head gates stay FP
        self.memory = LoopMemory(d) if cfg.loop_memory else None
        self.retrieval = IndexedRetrieval(d, cfg.n_heads, d // cfg.n_heads, lin, cfg.top_blocks, cfg.top_k, cfg.flat)
        self.mlp = SwiGLU(d, int(d * 8 / 3) // 8 * 8, lin)

    def forward(self, s, e, state, memory, index, pos0, use_index=True, distill=False):
        s = self.adapter(torch.cat([s, e], -1))
        y, state = self.gdn(self.n1(s), initial_state=state, return_state=True)
        s = s + y
        if self.memory is not None:
            s = s + self.memory(self.n2(s), memory)
        aux: dict = {}
        if use_index:
            r, aux = self.retrieval(self.n3(s), index, pos0, distill)
            s = s + r
        return s + self.mlp(self.n4(s)), state, aux

    def write_only(self, s, e, state):
        """Advance this loop's GDN state with s without running the rest of the block (skipped loops)."""
        s = self.adapter(torch.cat([s, e], -1))
        return self.gdn(self.n1(s), initial_state=state, return_state=True)[1]


class MTPHead(nn.Module):
    """Sequential multi-token prediction head (DeepSeek-V3 style, position-wise): h_j = f(h_{j-1}, emb(x_{t+j}))."""

    def __init__(self, d: int):
        super().__init__()
        self.nh, self.ne = nn.RMSNorm(d), nn.RMSNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.mlp = SwiGLU(d, int(d * 8 / 3) // 8 * 8, partial(nn.Linear, bias=False))
        self.norm = nn.RMSNorm(d)

    def forward(self, h: torch.Tensor, tok_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat([self.nh(h), self.ne(tok_emb)], -1))
        return self.norm(x + self.mlp(x))


@dataclass
class Cache:
    prelude: list
    core: list
    coda: list
    index: TokenIndex
    pos: int = 0
    e_buf: list = field(default_factory=list)  # prelude outputs of the current, incomplete block
    s_buf: list = field(default_factory=list)  # final-loop states of the current block

    def snapshot(self) -> tuple:
        # states are replaced, never modified in place, so shallow copies are enough
        return (
            list(self.prelude),
            list(self.core),
            list(self.coda),
            self.index.snapshot(),
            self.pos,
            list(self.e_buf),
            list(self.s_buf),
        )

    def restore(self, snap: tuple) -> None:
        prelude, core, coda, isnap, self.pos, e_buf, s_buf = snap
        self.prelude, self.core, self.coda = list(prelude), list(core), list(coda)
        self.e_buf, self.s_buf = list(e_buf), list(s_buf)
        self.index.restore(isnap)

    def state_bytes(self) -> dict:
        def b(xs):
            return sum(t.numel() * t.element_size() for t in xs if t is not None)

        return {"prelude_gdn_bytes": b(self.prelude), "core_gdn_bytes": b(self.core), "coda_gdn_bytes": b(self.coda)}


class LoopedIndexLM(nn.Module):
    def __init__(self, cfg: LoopedIndexConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        pre = _linear(cfg.prelude_linear, cfg.prelude_bpw, cfg.scale_bits)
        self.prelude = nn.ModuleList(StatefulBlock(d, cfg.n_heads, pre) for _ in range(cfg.n_prelude))
        self.core = CoreBlock(cfg)
        self.coda = nn.ModuleList(
            StatefulBlock(d, cfg.n_heads, _linear(cfg.coda_linear, cfg.coda_bpw, cfg.scale_bits))
            for _ in range(cfg.n_coda)
        )
        self.norm = nn.RMSNorm(d)
        self.mtp = nn.ModuleList(MTPHead(d) for _ in range(cfg.n_mtp))
        self.use_index = True  # stage switches
        self.distill = False

    # ---- running ---------------------------------------------------------------------------------------

    def new_cache(self, batch: int) -> Cache:
        cfg = self.cfg
        packed = not torch.is_grad_enabled()
        return Cache(
            [None] * cfg.n_prelude, [None] * cfg.n_loops, [None] * cfg.n_coda, TokenIndex(cfg.block, packed=packed)
        )

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.embed.weight.t()

    def _segment(self, tokens, cache: Cache, halt_eps: float | None, stats: dict) -> torch.Tensor:
        cfg = self.cfg
        x = tokens if tokens.is_floating_point() else self.embed(tokens)  # (B, L, d) inputs: used as embeddings
        for i, blk in enumerate(self.prelude):
            x, cache.prelude[i] = blk(x, cache.prelude[i])
        e = x
        self.core.retrieval.write(e, cache.index)
        s, memory = torch.zeros_like(e), []
        loops = cfg.n_loops
        for i in range(cfg.n_loops):
            set_loop(self.core, i)
            prev = s
            s, cache.core[i], aux = self.core(
                s, e, cache.core[i], memory, cache.index, cache.pos, self.use_index, self.distill
            )
            memory.append(s)
            for k, v in aux.items():
                stats.setdefault(k, []).append(v)
            if halt_eps is not None and i > 0 and i < cfg.n_loops - 1:
                change = ((s - prev).norm(dim=-1) / s.norm(dim=-1).clamp_min(1e-6)).max().item()
                if change < halt_eps:
                    loops = i + 1
                    for j in range(i + 1, cfg.n_loops):  # copy the halted state forward into skipped loops
                        set_loop(self.core, j)
                        cache.core[j] = self.core.write_only(s, e, cache.core[j])
                    break
        stats.setdefault("loops", []).append(loops)
        h = s
        for i, blk in enumerate(self.coda):
            h, cache.coda[i] = blk(h, cache.coda[i])
        cache.pos += tokens.shape[1]
        cache.e_buf.append(e)
        cache.s_buf.append(s)
        if cache.pos % cfg.block == 0:  # block complete: write its summaries
            self.core.retrieval.write_summaries(
                torch.cat(cache.e_buf, 1).mean(1), torch.cat(cache.s_buf, 1).mean(1), cache.index
            )
            cache.e_buf, cache.s_buf = [], []
        return self.norm(h)

    def process(self, tokens: torch.Tensor, cache: Cache, halt_eps: float | None = None, stats=None) -> torch.Tensor:
        """Run tokens (B, L) from the cache's position on; returns normed hidden states (B, L, d).
        A float tensor (B, L, d) is taken as input embeddings instead (e.g. byte-patch vectors)."""
        stats = {} if stats is None else stats
        outs, t = [], 0
        while t < tokens.shape[1]:
            n = min(tokens.shape[1] - t, self.cfg.block - cache.pos % self.cfg.block)
            outs.append(self._segment(tokens[:, t : t + n], cache, halt_eps, stats))
            t += n
        return torch.cat(outs, 1)

    # ---- training -------------------------------------------------------------------------------------

    def training_loss(self, tokens: torch.Tensor, mtp: bool = False, distill_weight: float = 1.0):
        """Next-token loss (+ MTP losses, + index distillation when self.distill). tokens: (B, T)."""
        stats: dict = {}
        h = self.process(tokens, self.new_cache(tokens.shape[0]), stats=stats)
        loss = F.cross_entropy(self.logits(h[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten())
        info = {"lm": loss.item()}
        if mtp:
            hj = h
            for j, head in enumerate(self.mtp, start=1):
                hj = head(hj[:, :-1], self.embed(tokens[:, j:]))  # position t sees x_{t+j}, predicts x_{t+j+1}
                lj = F.cross_entropy(self.logits(hj[:, :-1]).flatten(0, 1), tokens[:, j + 1 :].flatten())
                info[f"mtp{j}"] = lj.item()
                loss = loss + lj / len(self.mtp)
        if self.distill and stats.get("distill"):
            dl = torch.stack(stats["distill"]).mean()
            info["index_distill"] = dl.item()
            loss = loss + distill_weight * dl
        for k in ("fallback", "remote_frac"):
            if stats.get(k):
                info[k] = sum(stats[k]) / len(stats[k])
        return loss, info

    # ---- decoding -------------------------------------------------------------------------------------

    def _drafts(self, h_last: torch.Tensor, t1: torch.Tensor, k: int) -> list[torch.Tensor]:
        drafts, hj, tok = [], h_last, t1
        for head in list(self.mtp)[:k]:
            hj = head(hj, self.embed(tok))
            tok = self.logits(hj).argmax(-1)
            drafts.append(tok)
        return drafts

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new: int, halt_eps: float | None = None):
        """Plain greedy decode, one token per pass. With halt_eps, loops may stop early (state copied forward)."""
        cache, stats = self.new_cache(prompt.shape[0]), {}
        h = self.process(prompt, cache)
        out = []
        loop_stats: dict = {}
        for _ in range(n_new):
            tok = self.logits(h[:, -1:]).argmax(-1)
            out.append(tok)
            if len(out) == n_new:
                break
            h = self.process(tok, cache, halt_eps, loop_stats)
        loops = loop_stats.get("loops") or [self.cfg.n_loops]
        stats["mean_loops"] = sum(loops) / len(loops)
        stats["passes"] = n_new
        stats["cache"] = {**cache.state_bytes(), **cache.index.nbytes()}
        return torch.cat([prompt, *out], 1), stats

    @torch.no_grad()
    def generate_speculative(self, prompt: torch.Tensor, n_new: int, k: int | None = None, adaptive: bool = True):
        """Greedy decode with MTP drafts + parallel verify (batch 1). Output equals generate() with fixed N."""
        assert prompt.shape[0] == 1, "speculative decode is batch 1"
        k_max = len(self.mtp)
        k = k_max if k is None else min(k, k_max)
        cache = self.new_cache(1)
        h = self.process(prompt, cache)
        t1 = self.logits(h[:, -1:]).argmax(-1)
        drafts = self._drafts(h[:, -1:], t1, k)
        out: list[torch.Tensor] = []
        passes, proposed, accepted, rate, ks = 0, 0, 0, 0.5, []
        while sum(o.shape[1] for o in out) < n_new:
            seq = torch.cat([t1, *drafts], 1)  # t1 is the model's own prediction; drafts are guesses
            snap = cache.snapshot()
            hv = self.process(seq, cache)  # verify = write, fixed N
            passes += 1
            preds = self.logits(hv).argmax(-1)  # preds[:, i]: the model's next token after seq[:, i]
            a = 0
            while a < len(drafts) and drafts[a].item() == preds[0, a].item():
                a += 1
            proposed, accepted = proposed + len(drafts), accepted + a
            if a < len(drafts):  # reject: roll back the index and all states, re-write the accepted prefix
                cache.restore(snap)
                hv = self.process(seq[:, : a + 1], cache)
                passes += 1
            out.append(seq[:, : a + 1])
            t1 = self.logits(hv[:, -1:]).argmax(-1)
            if adaptive and drafts:
                rate = 0.7 * rate + 0.3 * a / len(drafts)
                k = max(1, k - 1) if rate < 0.3 else min(k_max, k + 1) if rate > 0.7 else k
            ks.append(k)
            drafts = self._drafts(hv[:, -1:], t1, k)
        gen = torch.cat(out, 1)[:, :n_new]
        stats = {
            "passes": passes,
            "tokens_per_pass": n_new / passes,
            "acceptance": accepted / max(1, proposed),
            "k_trace": ks,
            "cache": {**cache.state_bytes(), **cache.index.nbytes()},
        }
        return torch.cat([prompt, gen], 1), stats


@torch.no_grad()
def convert_core_to_littlebit(
    model: LoopedIndexLM, bpw: float | None = None, ranks: dict[str, int] | None = None
) -> None:
    """Stage 2: the core's shared FP linears -> LittleBit (Dual-SVID from the FP weights), keeping per-loop scales.
    `ranks` (core module name -> rank, see `allocate_core_ranks`) replaces the uniform bpw per layer."""
    bpw = model.cfg.core_bpw if bpw is None else bpw
    make = partial(LittleBitLinear, bpw=bpw, residual=True, ste="clipped", scale_bits=model.cfg.scale_bits)
    convert_loop_linears(model.core, make, ranks=ranks)
    model.cfg.core_linear = "littlebit"


@torch.no_grad()
def allocate_core_ranks(model: LoopedIndexLM, calib: torch.Tensor, bpw: float | None = None) -> tuple[dict, dict]:
    """Uneven bpw for the core: calibrate input energy of each FP core linear on `calib` tokens (B, T), then
    allocate LittleBit ranks at the same total bits as a uniform `bpw` (see `layers/bpw_alloc.py`)."""
    bpw = model.cfg.core_bpw if bpw is None else bpw
    lins = loop_fp_weights(model.core)
    was = model.training
    model.eval()
    with input_energy(lins) as act_sq:
        model.process(calib, model.new_cache(calib.shape[0]))
    model.train(was)
    return allocate_ranks({n: m.weight for n, m in lins.items()}, bpw, act_sq=act_sq)


def memory_tiers(model: LoopedIndexLM, cache_stats: dict, tokens: int) -> dict:
    """Bytes per memory tier for a cache holding `tokens` tokens."""
    d = model.cfg.d_model
    return {
        "within_token_loop_memory_bytes_per_position": model.cfg.n_loops * d * 4,
        "gdn_state_bytes": cache_stats["prelude_gdn_bytes"]
        + cache_stats["core_gdn_bytes"]
        + cache_stats["coda_gdn_bytes"],
        "summary_bytes": cache_stats["summary_bytes"],
        "kv_signature_bytes": cache_stats["kv_bytes"] + cache_stats["signature_bytes"],
        "kv_signature_bytes_per_token": (cache_stats["kv_bytes"] + cache_stats["signature_bytes"]) / max(1, tokens),
    }


__all__ = [
    "Cache",
    "CoreBlock",
    "LoopedIndexConfig",
    "LoopedIndexLM",
    "MTPHead",
    "allocate_core_ranks",
    "convert_core_to_littlebit",
    "memory_tiers",
]
