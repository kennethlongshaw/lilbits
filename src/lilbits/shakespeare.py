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


def segment_tensors(rows: list[list[tuple[bytes, bool]]], seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row is a list of (bytes, trained) segments. Returns bytes (B, seq + 1), right-padded with NUL, and a
    mask (B, seq) over targets x[:, 1:] that is 1 on bytes of trained segments. Rows longer than seq + 1 are cut
    from the left (the end, where the trained part is, is kept)."""
    xs, ms = [], []
    for segs in rows:
        b = b"".join(t for t, _ in segs)
        flag = [f for t, f in segs for _ in t]
        b, flag = b[-(seq + 1) :], flag[-(seq + 1) :]
        pad = seq + 1 - len(b)
        xs.append(list(b) + [0] * pad)
        ms.append([int(f) for f in flag[1:]] + [0] * pad)
    return torch.tensor(xs), torch.tensor(ms, dtype=torch.float)


def qa_tensors(pairs: list[dict], seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of "Q: q\nA: a\n" bytes, right-padded with NUL bytes to seq + 1, and a mask (B, seq) that is 1 where
    the target byte (x[:, 1:]) belongs to the answer (including its closing newline). Long questions are cut
    from the left so the answer always fits."""
    return segment_tensors([[(prompt_bytes(p["q"]), False), ((p["a"] + "\n").encode(), True)] for p in pairs], seq)


# ---- toy tasks (each exactly scored) ---------------------------------------------------------------------


@cache
def characters(split: str = "train", min_speeches: int = 20) -> tuple[str, ...]:
    c = Counter(s.speaker for s in split_speeches(split))
    return tuple(sorted(k for k, v in c.items() if v >= min_speeches and k.isalpha() and k.isupper()))


def needle_example(length: int, rng: random.Random, split: str = "test") -> tuple[bytes, str]:
    """Needle in a haystack: `length` bytes of Shakespeare with "The password is NAME." hidden at a random point,
    then the question. Returns (prompt bytes ending in "A: ", answer)."""
    text = text_of(split).encode()
    name = rng.choice(characters())
    needle = f" The password is {name}. ".encode()
    q = b"\nQ: What is the password?\nA: "
    hay = length - len(needle) - len(q)
    start = rng.randrange(len(text) - hay)
    window = text[start : start + hay]
    at = rng.randrange(hay)
    return window[:at] + needle + window[at:] + q, name


def copy_sort_example(rng: random.Random, k: int, task: str) -> tuple[bytes, str]:
    """copy: repeat a list of k character names; sort: the same list in alphabetical order."""
    names = rng.sample(characters(), k)
    target = names if task == "copy" else sorted(names)
    return f"{task.upper()}: {', '.join(names)}\n=> ".encode(), ", ".join(target)


def _excerpt(rng: random.Random, n_speeches: int, split: str, max_bytes: int) -> list[Speech]:
    """n consecutive speeches with distinct consecutive speakers, short enough to fit max_bytes."""
    sp = split_speeches(split)
    while True:
        i = rng.randrange(len(sp) - n_speeches)
        ex = [Speech(s.speaker, s.lines[:2]) for s in sp[i : i + n_speeches]]
        txt = "".join(f"{s.speaker}: {' '.join(s.lines)}\n" for s in ex)
        speakers = [s.speaker for s in ex]
        if len(txt) <= max_bytes and len(set(speakers)) == len(speakers):
            return ex


def _render(ex: list[Speech]) -> str:
    return "".join(f"{s.speaker}: {' '.join(s.lines)}\n" for s in ex)


def count_example(rng: random.Random, split: str = "train", max_bytes: int = 120) -> tuple[bytes, str]:
    """COUNT: how many times a word occurs in a short excerpt (answer: digits). Reads only the prompt."""
    ex = _excerpt(rng, 2, split, max_bytes)
    words = [w.strip(".,;:!?'\"").lower() for w in _render(ex).split()]
    words = [w for w in words if w.isalpha()]
    w = rng.choice(words)
    return f"{_render(ex)}COUNT {w} =>".encode() + b" ", str(words.count(w))


def who_said_example(rng: random.Random, split: str = "train", max_bytes: int = 160) -> tuple[bytes, str]:
    """Which speaker in the excerpt says the quoted words (a fragment of one of their lines)."""
    ex = _excerpt(rng, 3, split, max_bytes)
    s = rng.choice(ex)
    words = " ".join(s.lines).split()
    j = rng.randrange(max(1, len(words) - 2))
    frag = " ".join(words[j : j + 3])
    return f'{_render(ex)}WHO SAID "{frag}" =>'.encode() + b" ", s.speaker


def who_next_example(rng: random.Random, split: str = "train", max_bytes: int = 160) -> tuple[bytes, str]:
    """Two hops: find who says the fragment, then the speaker right after them in the excerpt."""
    ex = _excerpt(rng, 3, split, max_bytes)
    i = rng.randrange(len(ex) - 1)
    words = " ".join(ex[i].lines).split()
    j = rng.randrange(max(1, len(words) - 2))
    frag = " ".join(words[j : j + 3])
    return f'{_render(ex)}WHO ANSWERS "{frag}" =>'.encode() + b" ", ex[i + 1].speaker


TOOLS = {"search": "speaker of the speech containing the text", "next": "line after the given line"}


def run_tool(call: str) -> str:
    """Execute 'search("text")' or 'next("text")' over the whole corpus; unknown calls -> 'ERROR'."""
    import re

    m = re.fullmatch(r'(search|next)\("(.*)"\)', call.strip())
    if not m:
        return "ERROR"
    name, arg = m.groups()
    for s in speeches():
        for i, line in enumerate(s.lines):
            if arg and arg in line:
                if name == "search":
                    return s.speaker
                return s.lines[i + 1] if i + 1 < len(s.lines) else "NONE"
    return "NONE"


def tool_example(pair: dict) -> list[tuple[bytes, bool]] | None:
    """A QA pair as a tool-use trajectory: the model writes the call and the answer, the environment the result."""
    import re

    quoted = re.search(r'"(.*)"', pair["q"])
    if pair["kind"] not in ("speaker", "next_line") or not quoted:
        return None
    call = f'{"search" if pair["kind"] == "speaker" else "next"}("{quoted.group(1)}")'
    return [
        (prompt_bytes(pair["q"]).removesuffix(b"A: ") + b"CALL: ", False),
        (f"{call}\n".encode(), True),
        (f"RESULT: {run_tool(call)}\n".encode(), False),
        (b"A: ", False),
        (f"{pair['a']}\n".encode(), True),
    ]


def preference_pairs(split: str, n: int, seed: int = 0) -> list[dict]:
    """DPO pairs from QA: chosen = the right answer, rejected = the answer to another question of the same kind."""
    qa = load_qa(split)
    rng = random.Random(seed)
    by_kind: dict[str, list[dict]] = {}
    for p in qa:
        by_kind.setdefault(p["kind"], []).append(p)
    out = []
    for p in qa[:n]:
        other = rng.choice(by_kind[p["kind"]])
        while other["a"] == p["a"]:
            other = rng.choice(by_kind[p["kind"]])
        out.append({**p, "rejected": other["a"]})
    return out


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
