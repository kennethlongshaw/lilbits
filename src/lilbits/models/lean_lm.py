"""LeanLM: the measured wins in one byte-level model.

    bytes -> embed (tied, vocab 256) -> N x StatefulBlock (GDN, conv_size=1: the recurrent state is all the memory)
          -> RMSNorm -> next-byte logits
                     -> k draft heads (MTP: byte t+2 .. t+k+1 from the same hidden state)

- GDN mixer: fixed-size state instead of a growing K/V cache; chunked delta rule for training and prefill.
- LittleBit block linears (`convert_linears`): train FP first, then SVD(+ITQ)-convert the block projections and keep
  training, the recipe that held up in the compare runs. Inference packs them to 1-bit signs on the LUT kernels.
- Self-speculative decoding (`speculative_generate`): the main head and the draft heads propose k+1 bytes, one call
  verifies them all from a state snapshot, and rejecting drafts replays only the accepted prefix from the snapshot.
  The output is the model's own greedy output, so drafts only change the speed, never the text.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.littlebit import LittleBitLinear
from lilbits.models.decision import StatefulBlock, clone_states


class LeanLM(nn.Module):
    def __init__(
        self, d: int, n_heads: int, n_layers: int, linear, n_draft: int = 3, draft_rank: int = 32, vocab: int = 256
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(StatefulBlock(d, n_heads, linear, "gdn") for _ in range(n_layers))
        self.norm = nn.RMSNorm(d)
        # draft head j: h + B_j A_j h (rank draft_rank, B zero init: every draft starts as the next-byte prediction).
        # Low rank keeps the heads a few percent of the deployed bytes instead of n_draft x d^2 FP16 weights.
        r = min(draft_rank, d)
        self.drafts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)) for _ in range(n_draft)
        )
        for w in self.drafts:
            up = cast("nn.Sequential", w)[1]
            nn.init.zeros_(cast("nn.Linear", up).weight)

    def hidden(self, tokens: torch.Tensor, states: list | None = None, all_states: bool = False):
        """-> (normed hidden, per-layer states). all_states: each layer's state after every token, (B, T, ...)."""
        states = states or [None] * len(self.blocks)
        x = self.embed(tokens)
        new = []
        kw = {"all_states": True} if all_states else {}
        for blk, s in zip(self.blocks, states, strict=True):
            x, s = blk(x, s, **kw)
            new.append(s)
        return self.norm(x), new

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.embed.weight.t()

    def draft_logits(self, h: torch.Tensor) -> list[torch.Tensor]:
        return [self.head(h + w(h)) for w in self.drafts]

    def forward(self, tokens: torch.Tensor, states: list | None = None) -> tuple[torch.Tensor, list]:
        h, states = self.hidden(tokens, states)
        return self.head(h), states

    def loss(self, x: torch.Tensor, mtp_weight: float = 0.3) -> tuple[torch.Tensor, dict]:
        """x: (B, T+1) bytes. Next-byte CE plus mtp_weight x mean CE of the draft heads (head j: byte t+2+j)."""
        h, _ = self.hidden(x[:, :-1])
        ce = F.cross_entropy(self.head(h).flatten(0, 1), x[:, 1:].flatten())
        info: dict = {"ce": ce.item()}
        if not self.drafts or mtp_weight == 0:
            return ce, info
        T = h.shape[1]
        mtp = []
        for j, lg in enumerate(self.draft_logits(h), start=2):
            if T - j + 1 <= 0:
                break
            mtp.append(F.cross_entropy(lg[:, : T - j + 1].flatten(0, 1), x[:, j:].flatten()))
        m = torch.stack(mtp).mean()
        info["mtp_ce"] = m.item()
        return ce + mtp_weight * m, info

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new: int) -> torch.Tensor:
        """Greedy, batch 1: one prefill call, then one single-byte call per byte."""
        logits, states = self(prompt)
        out = []
        for _ in range(n_new):
            tok = logits[:, -1:].argmax(-1)
            out.append(tok)
            logits, states = self(tok, states)
        return torch.cat(out, 1)

    @torch.no_grad()
    def speculative_generate(self, prompt: torch.Tensor, n_new: int, replay: bool = False) -> tuple[torch.Tensor, dict]:
        """Greedy output via self-drafting, batch 1. Each round: the main head's next byte (always right) plus k draft
        bytes -> one verify call from the current state -> keep the longest prefix the model agrees with. The verify
        call keeps the GDN state after every candidate byte, so rolling back to the accepted prefix is an index,
        not a replay (replay=True: rebuild it by replaying the accepted bytes from a snapshot, as in example 03)."""
        h, states = self.hidden(prompt)
        h_last = h[:, -1:]
        out: list[torch.Tensor] = []
        rounds = accepted_drafts = proposed_drafts = calls = 0
        while sum(t.shape[1] for t in out) < n_new:
            cand = torch.cat([self.head(h_last).argmax(-1), *(lg.argmax(-1) for lg in self.draft_logits(h_last))], 1)
            snap = clone_states(states) if replay else states
            vh, vstates = self.hidden(cand, states, all_states=not replay)
            calls += 1
            pred = self.head(vh).argmax(-1)  # pred[:, i] = model's byte after cand[:, : i + 1]
            ok = (pred[:, :-1] == cand[:, 1:]).long().cumprod(1)
            n_acc = 1 + int(ok.sum())
            rounds += 1
            proposed_drafts += cand.shape[1] - 1
            accepted_drafts += n_acc - 1
            if not replay:
                states, h_last = [s[:, n_acc - 1] for s in vstates], vh[:, n_acc - 1 : n_acc]
            elif n_acc == cand.shape[1]:
                states, h_last = vstates, vh[:, -1:]
            else:
                rh, states = self.hidden(cand[:, :n_acc], snap)
                calls += 1
                h_last = rh[:, -1:]
            out.append(cand[:, :n_acc])
        toks = torch.cat(out, 1)[:, :n_new]
        return toks, {
            "rounds": rounds,
            "model_calls": calls,
            "bytes_per_call": n_new / max(1, calls),
            "draft_acceptance": accepted_drafts / max(1, proposed_drafts),
        }


@torch.no_grad()
def convert_linears(model: nn.Module, bpw: float, residual: bool = True, itq_iters: int = 50, min_out: int = 32) -> int:
    """In place: every nn.Linear inside model.blocks with out_features >= min_out (so not the per-head GDN gates)
    -> LittleBitLinear SVD(+ITQ)-initialized from its weight. Returns how many layers were converted."""
    n = 0
    for parent in cast("nn.Module", model.blocks).modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child.out_features >= min_out:
                lb = LittleBitLinear(
                    child.in_features, child.out_features, bpw=bpw, residual=residual, ste="clipped"
                ).to(child.weight.device)
                lb.init_from_weight(child.weight, itq_iters)
                setattr(parent, name, lb)
                n += 1
    return n
