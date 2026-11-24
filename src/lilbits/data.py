"""Training data: a synthetic Markov source with a known loss floor, or a text file."""

from __future__ import annotations

import numpy as np
import torch


class MarkovSource:
    """Random sparse order-`order` Markov chain over `vocab` tokens.

    Each context has `branching` allowed next tokens with Dirichlet weights,
    so the optimal achievable cross-entropy (the entropy rate) is known and
    well below log(vocab). A model must use attention to look `order` tokens back.
    """

    def __init__(self, vocab: int = 64, order: int = 2, branching: int = 4, alpha: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab, self.order = vocab, order
        n_ctx = vocab**order
        probs = torch.zeros(n_ctx, vocab)
        nxt = torch.from_numpy(rng.random((n_ctx, vocab)).argsort(axis=1)[:, :branching])
        w = torch.from_numpy(rng.dirichlet(np.full(branching, alpha), size=n_ctx)).float()
        probs.scatter_(1, nxt, w)
        self.probs = probs
        self._g = torch.Generator().manual_seed(seed + 1)

    def _ctx_index(self, seq: torch.Tensor, t: int) -> torch.Tensor:
        idx = torch.zeros(seq.shape[0], dtype=torch.long)
        for j in range(self.order):
            idx = idx * self.vocab + seq[:, t - self.order + j]
        return idx

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        seq = torch.empty(batch, seq_len + 1, dtype=torch.long)
        seq[:, : self.order] = torch.randint(0, self.vocab, (batch, self.order), generator=self._g)
        for t in range(self.order, seq_len + 1):
            p = self.probs[self._ctx_index(seq, t)]
            seq[:, t] = torch.multinomial(p, 1, generator=self._g).squeeze(1)
        return seq

    def entropy_floor(self, seq: torch.Tensor) -> float:
        """Mean true conditional entropy (nats) over predictable positions of `seq`."""
        hs = []
        for t in range(self.order, seq.shape[1]):
            p = self.probs[self._ctx_index(seq, t)]
            hs.append(-(p * p.clamp_min(1e-12).log()).sum(1))
        return torch.stack(hs).mean().item()


class TextSource:
    """Character-level LM data from a text file (e.g. tinyshakespeare)."""

    def __init__(self, path: str, seed: int = 0):
        text = open(path, encoding="utf-8").read()
        chars = sorted(set(text))
        self.vocab = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self._g = torch.Generator().manual_seed(seed)

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        ix = torch.randint(0, len(self.data) - seq_len - 1, (batch,), generator=self._g)
        return torch.stack([self.data[i : i + seq_len + 1] for i in ix])


class ByteSource:
    """Raw UTF-8 bytes (vocab 256, no tokenizer) from files or a bytes object, for byte-level models."""

    vocab = 256

    def __init__(self, data: bytes | str | list[str], seed: int = 0):
        if isinstance(data, (str, list)):
            paths = [data] if isinstance(data, str) else data
            data = b"".join(open(p, "rb").read() for p in paths)
        self.data = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
        self._g = torch.Generator().manual_seed(seed)

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        ix = torch.randint(0, len(self.data) - seq_len - 1, (batch,), generator=self._g)
        return torch.stack([self.data[i : i + seq_len + 1] for i in ix])


class PairedViewSource:
    """Two views of the same content, for JEPA-style training: (Text, Code) pairs.

    Text is a random string over `n_symbols` ids [0, n). Code is the same content written another way: the
    reversed string through a fixed random substitution into ids [n, 2n). Id 2n is free for a [PRED] token,
    so the vocabulary is 2n + 1. Code is a deterministic function of Text, so the ideal Text -> Code loss is 0.
    """

    def __init__(self, n_symbols: int = 8, length: int = 8, seed: int = 0):
        self.n, self.length = n_symbols, length
        self.vocab, self.pred_token = 2 * n_symbols + 1, 2 * n_symbols
        self._g = torch.Generator().manual_seed(seed)
        self.sub = torch.randperm(n_symbols, generator=self._g) + n_symbols

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = torch.randint(0, self.n, (batch, self.length), generator=self._g)
        return text, self.sub[text.flip(1)]


class RecallSource:
    """Long-context recall and multi-hop sequences, to measure what loss alone does not.

    Token ids: symbols [0, n_symbols), filler [n_symbols, n_symbols + n_filler), then QUERY_1HOP, QUERY_2HOP.
    A sequence of `length` tokens is filler with `n_pairs` (key, value) pairs of symbols written at random
    positions, ending with a question and its answer as the last token:

        recall    ... k v ... [Q1] k  v       value written next to k, possibly thousands of tokens back
        multi-hop ... a b ... b c ... [Q2] a  c   follow a -> b -> c across two facts

    Keys within a sequence are distinct, so the answer is unique. `sample` returns (tokens, kind) where kind is
    0 for recall and 1 for multi-hop; the answer is tokens[:, -1], predicted at position -2.
    """

    def __init__(self, n_symbols: int = 24, n_filler: int = 6, n_pairs: int = 4, length: int = 256, seed: int = 0):
        assert 2 * n_pairs + 2 <= n_symbols
        self.n_symbols, self.n_filler, self.n_pairs, self.length = n_symbols, n_filler, n_pairs, length
        self.q1, self.q2 = n_symbols + n_filler, n_symbols + n_filler + 1
        self.vocab = n_symbols + n_filler + 2
        self._g = torch.Generator().manual_seed(seed)

    def _one(self, two_hop: bool) -> torch.Tensor:
        g, L = self._g, self.length
        seq = torch.randint(self.n_symbols, self.n_symbols + self.n_filler, (L,), generator=g)
        syms = torch.randperm(self.n_symbols, generator=g)
        if two_hop:  # chain a -> b -> c, plus distractor pairs
            a, b, c = syms[:3].tolist()
            pairs = [(a, b), (b, c)] + [
                (syms[3 + 2 * i].item(), syms[4 + 2 * i].item()) for i in range(self.n_pairs - 2)
            ]
            question, answer = (self.q2, a), c
        else:
            pairs = [(syms[2 * i].item(), syms[2 * i + 1].item()) for i in range(self.n_pairs)]
            question, answer = (self.q1, pairs[0][0]), pairs[0][1]
        # non-overlapping slots in [0, L - 3), in random order
        slots = torch.randperm((L - 3) // 2, generator=g)[: len(pairs)].sort().values * 2
        order = torch.randperm(len(pairs), generator=g)
        if two_hop:  # keep a -> b before b -> c so the chain can be followed left to right
            order = torch.cat([torch.tensor([0, 1]), order[order > 1]])
        for slot, j in zip(slots.tolist(), order.tolist(), strict=True):
            seq[slot], seq[slot + 1] = pairs[j]
        seq[L - 3], seq[L - 2], seq[L - 1] = question[0], question[1], answer
        return seq

    def sample(self, batch: int, two_hop_frac: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        kind = (torch.rand(batch, generator=self._g) < two_hop_frac).long()
        return torch.stack([self._one(bool(k)) for k in kind.tolist()]), kind
