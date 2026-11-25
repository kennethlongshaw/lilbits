"""Toy Shakespeare data: the tiny-shakespeare corpus as bytes, its speeches, and exactly scorable QA pairs.

    speeches()                       the corpus parsed into (speaker, lines) turns, in order
    ShakespeareBytes(split)          random byte windows from the train (first 90% of speeches) or test text
    make_qa(split, n, seed)          question / answer pairs read off the text (see data/shakespeare/README.md)
    load_qa(split)                   the committed pairs (data/shakespeare/qa_{split}.jsonl)
    qa_tensors(pairs, seq)           "Q: ...\\nA: ...\\n" byte rows + a loss mask over the answer bytes (SFT)
    exact_match(answer_fn, pairs)    fraction of pairs whose generated answer equals the reference exactly

The data lives in the repo's data/shakespeare/ (set LILBITS_DATA to point elsewhere).
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

TEST_FRAC = 0.1
KINDS = ("speaker", "next_line", "next_speaker", "line_count")


def data_dir() -> Path:
    env = os.environ.get("LILBITS_DATA")
    if env:
        return Path(env) / "shakespeare"
    return Path(__file__).resolve().parents[2] / "data" / "shakespeare"


@cache
def corpus() -> str:
    return (data_dir() / "tiny_shakespeare.txt").read_text()


@dataclass(frozen=True)
class Speech:
    speaker: str
    lines: tuple[str, ...]


@cache
def speeches() -> tuple[Speech, ...]:
    out = []
    for block in corpus().strip().split("\n\n"):
        head, *lines = block.split("\n")
        if head.endswith(":") and lines:
            out.append(Speech(head[:-1], tuple(lines)))
    return tuple(out)


def split_speeches(split: str) -> tuple[Speech, ...]:
    sp = speeches()
    cut = int(len(sp) * (1 - TEST_FRAC))
    if split == "train":
        return sp[:cut]
    if split == "test":
        return sp[cut:]
    raise ValueError(f"split must be train or test, not {split!r}")


def text_of(split: str) -> str:
    return "\n\n".join(f"{s.speaker}:\n" + "\n".join(s.lines) for s in split_speeches(split)) + "\n"


class ShakespeareBytes:
    """Random byte windows (vocab 256) from one split's text."""

    vocab = 256

    def __init__(self, split: str = "train", seed: int = 0):
        self.data = torch.frombuffer(bytearray(text_of(split).encode()), dtype=torch.uint8).long()
        self._g = torch.Generator().manual_seed(seed)

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        ix = torch.randint(0, len(self.data) - seq_len - 1, (batch,), generator=self._g)
        return torch.stack([self.data[i : i + seq_len + 1] for i in ix])


def _usable(line: str) -> bool:
    return 20 <= len(line) <= 70


def make_qa(split: str, n: int, seed: int = 0) -> list[dict]:
    """n pairs, kinds in rotation. Lines quoted in questions occur exactly once in the whole corpus, so each
    question has a single answer."""
    counts = Counter(line for s in speeches() for line in s.lines)
    sp = split_speeches(split)
    rng = random.Random(seed)
    out: list[dict] = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        kind = KINDS[len(out) % len(KINDS)]
        i = rng.randrange(len(sp) - 1)
        s, nxt = sp[i], sp[i + 1]

        def unique(line: str) -> bool:
            return _usable(line) and counts[line] == 1

        if kind == "speaker":
            line = rng.choice(s.lines)
            if unique(line):
                out.append({"kind": kind, "q": f'Who says "{line}"?', "a": s.speaker})
        elif kind == "next_line" and len(s.lines) >= 2:
            j = rng.randrange(len(s.lines) - 1)
            if unique(s.lines[j]):
                out.append({"kind": kind, "q": f'What line comes after "{s.lines[j]}"?', "a": s.lines[j + 1]})
        elif kind == "next_speaker":
            if unique(s.lines[-1]) and nxt.speaker != s.speaker:
                q = f'Who speaks after {s.speaker} says "{s.lines[-1]}"?'
                out.append({"kind": kind, "q": q, "a": nxt.speaker})
        elif kind == "line_count" and unique(s.lines[0]):
            q = f'How many lines are in {s.speaker}\'s speech that begins "{s.lines[0]}"?'
            out.append({"kind": kind, "q": q, "a": str(len(s.lines))})
    return out


def load_qa(split: str) -> list[dict]:
    with open(data_dir() / f"qa_{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def prompt_bytes(q: str) -> bytes:
    return f"Q: {q}\nA: ".encode()


def qa_tensors(pairs: list[dict], seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of "Q: q\\nA: a\\n" bytes, right-padded with newlines to seq + 1, and a mask (B, seq) that is 1 where
    the target byte (x[:, 1:]) belongs to the answer (including its closing newline). Long questions are cut
    from the left so the answer always fits."""
    rows, masks = [], []
    for p in pairs:
        pr, ans = prompt_bytes(p["q"]), (p["a"] + "\n").encode()
        pr = pr[-(seq + 1 - len(ans)) :]
        b = pr + ans
        m = [0] * (len(pr) - 1) + [1] * len(ans)
        pad = seq + 1 - len(b)
        rows.append(list(b) + [10] * pad)
        masks.append(m + [0] * pad)
    return torch.tensor(rows), torch.tensor(masks, dtype=torch.float)


def exact_match(answer_fn: Callable[[str], str], pairs: list[dict]) -> dict:
    """answer_fn(question) -> answer string. Returns overall and per-kind exact-match accuracy."""
    hits: dict[str, list[int]] = {}
    for p in pairs:
        hits.setdefault(p["kind"], []).append(int(answer_fn(p["q"]).strip() == p["a"]))
    allh = [h for v in hits.values() for h in v]
    return {"exact_match": sum(allh) / max(1, len(allh)), **{k: sum(v) / len(v) for k, v in hits.items()}}


def write_qa(n_train: int = 4000, n_test: int = 400) -> None:
    for split, n in (("train", n_train), ("test", n_test)):
        with open(data_dir() / f"qa_{split}.jsonl", "w") as f:
            for p in make_qa(split, n, seed=0 if split == "train" else 1):
                f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    write_qa()
    print({s: len(load_qa(s)) for s in ("train", "test")})
