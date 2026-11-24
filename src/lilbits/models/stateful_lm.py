"""A small LM built only from recurrent mixers, with explicit state: prefill once, then decode token by token.

Every layer is a `StatefulBlock` (any mixer from `mixers/recurrent.py`, conv_size=1, so the mixer's recurrent
state is the layer's whole memory). `forward(tokens, states)` returns (logits, states), which is all a real decode
loop needs: no KV cache, no recomputing the prefix, and a state of fixed size whatever the context length.
"""

from __future__ import annotations

import torch
from torch import nn

from lilbits.models.decision import StatefulBlock, clone_states


def state_bytes(states: list) -> int:
    total = 0
    for s in states:
        for t in s if isinstance(s, tuple) else (s,):
            if t is not None:
                total += t.numel() * t.element_size()
    return total


class StatefulLM(nn.Module):
    def __init__(self, vocab: int, d: int, n_heads: int, n_layers: int, linear, mixer: str = "gdn"):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(StatefulBlock(d, n_heads, linear, mixer) for _ in range(n_layers))
        self.norm = nn.RMSNorm(d)
        self.mixer = mixer

    def forward(self, tokens: torch.Tensor, states: list | None = None):
        states = states or [None] * len(self.blocks)
        x = self.embed(tokens)
        new = []
        for blk, s in zip(self.blocks, states, strict=True):
            x, s = blk(x, s)
            new.append(s)
        return self.norm(x) @ self.embed.weight.t(), new

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new: int, states: list | None = None):
        """Greedy: one prefill call, then one single-token call per new token (state carried, nothing recomputed)."""
        logits, states = self(prompt, clone_states(states) if states else None)
        out = []
        for _ in range(n_new):
            tok = logits[:, -1:].argmax(-1)
            out.append(tok)
            logits, states = self(tok, states)
        return torch.cat([prompt, *out], 1), states
