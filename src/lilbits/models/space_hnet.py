"""SpaceHNet: a two-level byte LM for CPUs. H-Net's encoder / main / decoder hierarchy, SpaceByte's patch rule,
Gated DeltaNet everywhere (fixed-size state, no KV cache).

    bytes -> embed (d_outer, tied head) -> ENCODER: n_enc GDN blocks, every byte
          -> at each patch boundary: up(norm(h)) -> MAIN: n_main GDN blocks (d_main), one step per patch
          -> every byte: y = down(norm(z of the byte's patch)) + residual(h)
          -> DECODER: n_dec GDN blocks, every byte -> RMSNorm -> next-byte logits

Patch boundaries come from a fixed rule on the bytes (no router, no ratio loss, no patcher model), so decode knows
whether a byte opens a patch as soon as it has the byte:

    align="end"    the main step runs on the first space-like byte after a word: the step has seen the whole word,
                   and its output is used from that byte on, so the next word's first byte is predicted from it
    align="start"  SpaceByte / example 07: a patch starts right after that byte (on the next word's first byte)

Space-like = not an ASCII letter / digit and not a UTF-8 continuation byte (SpaceByte). Long runs are cut every
max_patch bytes. With ~5-6 bytes per patch on English the main network's weights are read once per word, while the
narrow encoder and decoder run per byte.

`SpaceHNetStream` decodes batch 1, one byte per call, carrying every GDN state; it matches the parallel forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.models.byte_latent import cap_patches, spacelike
from lilbits.models.decision import StatefulBlock


@dataclass
class SpaceHNetConfig:
    d_outer: int = 128
    d_main: int = 256
    heads_outer: int = 4
    heads_main: int = 4
    n_enc: int = 2
    n_main: int = 6
    n_dec: int = 2
    align: str = "end"  # end | start
    max_patch: int = 16


def boundaries(x: torch.Tensor, align: str = "end", max_patch: int | None = 16) -> torch.Tensor:
    """(B, T) bytes -> (B, T) bool: True where the main network takes a step. Position 0 always does.
    Depends only on x_{<=t}, so a streaming decoder can decide it on the byte it was just given."""
    sp = spacelike(x)
    b = torch.zeros_like(sp)
    if align == "end":  # first space-like byte after a non-space-like one
        b[:, 1:] = sp[:, 1:] & ~sp[:, :-1]
    elif align == "start":  # the byte after that one
        b[:, 1:] = sp[:, :-1]
        b[:, 2:] &= ~sp[:, :-2]
    else:
        raise ValueError(f"align must be end or start, not {align!r}")
    b[:, 0] = True
    return cap_patches(b, max_patch)


def _gdn_stack(n: int, d: int, heads: int) -> nn.ModuleList:
    return nn.ModuleList(StatefulBlock(d, heads, lambda i, o: nn.Linear(i, o, bias=False), "gdn") for _ in range(n))


def _run(blocks: nn.ModuleList, x: torch.Tensor, states: list | None):
    states = states or [None] * len(blocks)
    new = []
    for blk, s in zip(blocks, states, strict=True):
        x, s = blk(x, s)
        new.append(s)
    return x, new


class SpaceHNet(nn.Module):
    def __init__(self, cfg: SpaceHNetConfig):
        super().__init__()
        self.cfg = cfg
        do, dm = cfg.d_outer, cfg.d_main
        self.embed = nn.Embedding(256, do)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.encoder = _gdn_stack(cfg.n_enc, do, cfg.heads_outer)
        self.n_up = nn.RMSNorm(do)
        self.up = nn.Linear(do, dm, bias=False)
        self.main = _gdn_stack(cfg.n_main, dm, cfg.heads_main)
        self.n_down = nn.RMSNorm(dm)
        self.down = nn.Linear(dm, do, bias=False)
        self.residual = nn.Linear(do, do, bias=False)
        self.decoder = _gdn_stack(cfg.n_dec, do, cfg.heads_outer)
        self.norm = nn.RMSNorm(do)

    def boundaries(self, x: torch.Tensor) -> torch.Tensor:
        return boundaries(x, self.cfg.align, self.cfg.max_patch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """(B, T) bytes -> next-byte logits (B, T, 256), stats."""
        B, T = x.shape
        h, _ = _run(self.encoder, self.embed(x), None)
        b = self.boundaries(x)
        pid = b.long().cumsum(1) - 1  # patch index of every byte
        n = int(pid.max()) + 1
        # gather each row's boundary positions into (B, n, d_outer); rows with fewer patches are zero-padded at the
        # end (causal main network: padding after a row's last patch never reaches its real patches)
        slot = torch.where(b, pid, torch.full_like(pid, n))
        hb = h.new_zeros(B, n + 1, h.shape[-1]).scatter(1, slot[..., None].expand_as(h), h)[:, :n]
        z, _ = _run(self.main, self.up(self.n_up(hb)), None)
        zd = self.down(self.n_down(z))
        y = zd.gather(1, pid[..., None].expand(B, T, zd.shape[-1])) + self.residual(h)
        y, _ = _run(self.decoder, y, None)
        logits = self.norm(y) @ self.embed.weight.t()
        return logits, {"bytes_per_patch": b.numel() / b.sum().item()}

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """x: (B, T+1) bytes -> mean next-byte CE (nats)."""
        logits, st = self(x[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()), st

    def macs_per_byte(self, bytes_per_patch: float) -> float:
        """Linear-layer multiply-adds per generated byte (embedding lookup excluded, tied head included)."""

        def lin(mods) -> int:
            return sum(m.in_features * m.out_features for m in mods.modules() if isinstance(m, nn.Linear))

        outer = lin(self.encoder) + lin(self.decoder) + lin(self.residual) + 256 * self.cfg.d_outer
        patch = lin(self.main) + lin(self.up) + lin(self.down)
        return outer + patch / bytes_per_patch


def plain_macs_per_byte(model: nn.Module) -> int:
    """LeanLM (no draft heads): every linear once per byte, plus the tied head."""
    lin = sum(m.in_features * m.out_features for m in model.blocks.modules() if isinstance(m, nn.Linear))
    return lin + 256 * model.embed.embedding_dim


class SpaceHNetStream:
    """Batch-1 streaming decode: `feed(byte)` -> next-byte logits (256,). The main network steps only on patch
    boundaries, decided from the bytes seen so far (the same rule as training)."""

    def __init__(self, model: SpaceHNet):
        self.m = model
        self.enc = self.main = self.dec = None
        self.zd: torch.Tensor | None = None
        self.prev_sp: bool | None = None  # was the previous byte space-like
        self.prev2_sp: bool | None = None
        self.run = 0  # bytes since the last boundary
        self.main_steps = self.bytes = 0

    def _is_boundary(self, byte: int) -> bool:
        sp = bool(spacelike(torch.tensor([byte])).item())
        if self.prev_sp is None:
            b = True
        elif self.m.cfg.align == "end":
            b = sp and not self.prev_sp
        else:
            b = bool(self.prev_sp) and not bool(self.prev2_sp)
        if not b and self.m.cfg.max_patch and self.run + 1 >= self.m.cfg.max_patch:
            b = True  # cap_patches: forced start every max_patch bytes after a natural one
        self.prev2_sp, self.prev_sp = self.prev_sp, sp
        return b

    @torch.no_grad()
    def feed(self, byte: int) -> torch.Tensor:
        m = self.m
        x = torch.tensor([[byte]])
        h, self.enc = _run(m.encoder, m.embed(x), self.enc)
        if self._is_boundary(byte):
            self.run = 0
            z, self.main = _run(m.main, m.up(m.n_up(h)), self.main)
            self.zd = m.down(m.n_down(z))
            self.main_steps += 1
        else:
            self.run += 1
        assert self.zd is not None
        y, self.dec = _run(m.decoder, self.zd + m.residual(h), self.dec)
        self.bytes += 1
        return (m.norm(y) @ m.embed.weight.t())[0, 0]
