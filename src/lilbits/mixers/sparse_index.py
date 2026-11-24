"""Sparse token index with hierarchical retrieval: block summaries -> sign signatures -> exact top-k attention.

Every token is written to the index once, at insert time, from the prelude output (so its entries do not depend on
how many loops the core runs later):

    per token:   K, V for one retrieval layer, 4-bit (absmax per token, FP16 scale), one KV head (GQA)
                 a sign(K) signature: 1 bit per key channel (Dk / 8 bytes)
    per block:   two summaries every `block` tokens: shallow (mean prelude output of the block) and
                 deep (mean final-loop core state of the block, written back by the model when the block ends)

A query at position p (any loop of the core) attends to:

    local:   all tokens of its own block up to p (causal)
    remote:  1. score every earlier complete block by q . (k_shallow + k_deep)        (L1: summaries)
             2. in the top `top_blocks` blocks, rank tokens by sign agreement
                popcount(~(sign(q) XOR signature))                                   (L2: signatures)
             3. exact softmax attention over the `top_k` best tokens                  (dequantized K/V)
             If the block scores are flat (normalized entropy > `flat`), the query falls back to dense attention
             over every earlier token.

`distill_loss` trains the block scorer to match the dense attention mass per block (KL), so the index learns which
blocks dense attention would have used. Segments passed to `forward` must not cross a block boundary (the model
splits its chunks so that they never do).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

_POPCOUNT = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.uint8)
_SHIFTS = torch.arange(8, dtype=torch.uint8)


def pack_bits(x: torch.Tensor) -> torch.Tensor:
    """(..., D) real -> (..., ceil(D/8)) uint8 of sign bits (x >= 0 -> 1)."""
    bits = (x >= 0).to(torch.uint8)
    bits = F.pad(bits, (0, (-x.shape[-1]) % 8))
    return (bits.unflatten(-1, (-1, 8)) << _SHIFTS.to(x.device)).sum(-1, dtype=torch.uint8)


def sign_agreement(q_sig: torch.Tensor, k_sig: torch.Tensor) -> torch.Tensor:
    """(B, L, nb) and (B, P, nb) uint8 -> (B, L, P) number of equal sign bits (XOR + popcount)."""
    x = q_sig.unsqueeze(2) ^ k_sig.unsqueeze(1)  # B, L, P, nb
    ham = _POPCOUNT.to(x.device)[x.long()].sum(-1, dtype=torch.int32)
    return 8 * q_sig.shape[-1] - ham


def quant4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax int4 in [-7, 7]: (..., D) -> packed uint8 (..., D/2), FP16 scale (...,)."""
    s = x.abs().amax(-1).clamp(min=1e-6) / 7
    q = (x / s.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8) + 8  # 1..15
    q = q.to(torch.uint8)
    return q[..., 0::2] | (q[..., 1::2] << 4), s.half()


def dequant4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lo, hi = (packed & 15).float() - 8, (packed >> 4).float() - 8
    return torch.stack([lo, hi], -1).flatten(-2) * scale.float().unsqueeze(-1)


def fake_quant4(x: torch.Tensor) -> torch.Tensor:
    """Training-time int4 (absmax per row) with a straight-through gradient."""
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-6).detach() / 7
    return x + ((x / s).round().clamp(-7, 7) * s - x).detach()


@dataclass
class TokenIndex:
    """The cached index for a batch. Training keeps float (fake-quant) K/V in the autograd graph; eval keeps
    packed int4 codes, which is what the byte counts report."""

    block: int
    K: list = field(default_factory=list)  # per segment: (B, L, Dk) float  |  (codes, scale) when packed
    V: list = field(default_factory=list)
    sig: list = field(default_factory=list)  # (B, L, Dk/8) uint8
    shallow: list = field(default_factory=list)  # per complete block: (B, Dk) summary keys
    deep: list = field(default_factory=list)
    length: int = 0
    packed: bool = False

    def append(self, K: torch.Tensor, V: torch.Tensor, sig: torch.Tensor) -> None:
        if self.packed:
            self.K.append(quant4(K))
            self.V.append(quant4(V))
        else:
            self.K.append(K)
            self.V.append(V)
        self.sig.append(sig)
        self.length += K.shape[1]

    def _cat(self, xs: list) -> torch.Tensor:
        if self.packed:
            return torch.cat([dequant4(c, s) for c, s in xs], 1)
        return torch.cat(xs, 1)

    def keys(self) -> torch.Tensor:
        return self._cat(self.K)

    def values(self) -> torch.Tensor:
        return self._cat(self.V)

    def signatures(self) -> torch.Tensor:
        return torch.cat(self.sig, 1)

    def snapshot(self) -> tuple:
        return (len(self.K), len(self.shallow), len(self.deep), self.length)

    def restore(self, snap: tuple) -> None:
        """Roll back to a snapshot (drops entries written since, e.g. rejected draft tokens)."""
        nseg, ns, nd, length = snap
        del self.K[nseg:], self.V[nseg:], self.sig[nseg:], self.shallow[ns:], self.deep[nd:]
        self.length = length

    def nbytes(self) -> dict:
        """Bytes held per tier. Packed: the int4 codes + FP16 scales stored. Training (float K/V): the size
        the same entries take once packed."""
        kv = 0
        for x in self.K + self.V:
            if self.packed:
                codes, scale = x
                kv += codes.numel() + 2 * scale.numel()
            else:
                kv += x.numel() // 2 + 2 * x.shape[0] * x.shape[1]
        sig = sum(t.numel() for t in self.sig)
        summ = sum(t.numel() * 2 for t in self.shallow + self.deep)  # FP16
        return {"kv_bytes": kv, "signature_bytes": sig, "summary_bytes": summ}


class IndexedRetrieval(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        linear=None,
        top_blocks: int = 2,
        top_k: int = 16,
        flat: float = 0.95,
    ):
        super().__init__()
        linear = linear or (lambda i, o: nn.Linear(i, o, bias=False))
        self.H, self.D = n_heads, head_dim
        self.top_blocks, self.top_k, self.flat = top_blocks, top_k, flat
        self.q = linear(d_model, n_heads * head_dim)
        self.o = linear(n_heads * head_dim, d_model)
        # written once per token / block: small, kept FP
        self.k = nn.Linear(d_model, head_dim, bias=False)
        self.v = nn.Linear(d_model, head_dim, bias=False)
        self.k_shallow = nn.Linear(d_model, head_dim, bias=False)
        self.k_deep = nn.Linear(d_model, head_dim, bias=False)
        self.k_norm = nn.RMSNorm(head_dim)

    def write(self, e: torch.Tensor, index: TokenIndex) -> None:
        """Insert tokens (B, L, d prelude output)."""
        K = self.k_norm(self.k(e))
        V = self.v(e)
        index.append(fake_quant4(K), fake_quant4(V), pack_bits(K))

    def write_summaries(self, shallow: torch.Tensor | None, deep: torch.Tensor | None, index: TokenIndex) -> None:
        """Per complete block: shallow = mean prelude output, deep = mean final-loop state, (B, d) each."""
        if shallow is not None:
            index.shallow.append(self.k_norm(self.k_shallow(shallow)))
        if deep is not None:
            index.deep.append(self.k_norm(self.k_deep(deep.detach())))

    def forward(self, x: torch.Tensor, index: TokenIndex, pos0: int, distill: bool = False):
        """x: (B, L, d) queries at positions pos0 .. pos0+L-1 (one block). Returns (out, aux dict)."""
        B, L, _ = x.shape
        H, D = self.H, self.D
        cur = pos0 // index.block
        assert (pos0 + L - 1) // index.block == cur, "segment crosses a block boundary"
        start = cur * index.block  # first token of the query's own block
        P = pos0 + L  # tokens visible to the last query
        q = self.q(x).view(B, L, H, D)
        K, V = index.keys()[:, :P], index.values()[:, :P]
        pos = torch.arange(P, device=x.device)
        qpos = pos0 + torch.arange(L, device=x.device)
        mask = ((pos[None, :] >= start) & (pos[None, :] <= qpos[:, None])).expand(B, L, P).clone()  # local
        aux: dict = {"fallback": 0.0, "remote_frac": 0.0}
        nb = min(len(index.shallow), len(index.deep), cur)
        if nb > 0:
            prev = start  # tokens in complete earlier blocks
            qm = q.mean(2)  # B, L, D
            summ = torch.stack(index.shallow[:nb], 1) + torch.stack(index.deep[:nb], 1)  # B, nb, D
            scores = torch.einsum("bld,bnd->bln", qm.detach(), summ) / math.sqrt(D)
            if distill:
                with torch.no_grad():
                    att = torch.einsum("blhd,bpd->blhp", q, K[:, :prev]).div(math.sqrt(D)).softmax(-1).mean(2)
                    mass = att.view(B, L, nb, index.block).sum(-1)
                aux["distill"] = F.kl_div(scores.log_softmax(-1), mass, reduction="batchmean") / L
            with torch.no_grad():
                p = scores.softmax(-1)
                ent = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(nb) if nb > 1 else torch.zeros(B, L)
                dense = ent > self.flat  # B, L: flat scores -> dense fallback
                tb = min(self.top_blocks, nb)
                chosen = torch.zeros(B, L, nb, dtype=torch.bool, device=x.device)
                chosen.scatter_(2, scores.topk(tb, -1).indices, True)
                in_chosen = chosen.repeat_interleave(index.block, -1)  # B, L, prev
                agree = sign_agreement(pack_bits(qm), index.signatures()[:, :prev]).float()
                agree = agree.masked_fill(~in_chosen, -1.0)
                k = min(self.top_k, tb * index.block)
                remote = torch.zeros(B, L, prev, dtype=torch.bool, device=x.device)
                remote.scatter_(2, agree.topk(k, -1).indices, True)
                remote &= in_chosen
                remote |= dense.unsqueeze(-1)
                mask[:, :, :prev] = remote
                aux["fallback"] = dense.float().mean().item()
                aux["remote_frac"] = remote.float().sum(-1).mean().item() / prev
        att = torch.einsum("blhd,bpd->bhlp", q, K) / math.sqrt(D)
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf")).softmax(-1)
        out = torch.einsum("bhlp,bpd->blhd", att, V).reshape(B, L, H * D)
        return self.o(out), aux
