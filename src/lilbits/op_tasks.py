"""Algorithmic tasks over Shakespeare character names, each built to need a known chain of query operators.

Every task gives a prompt, a reference answer, an exact score, and the operators a solution should use (for
`OperatorLoop` traces). Several are ones softmax attention handles badly on its own: exact counting (attention
averages, so it reports a proportion, not a number), argmax over values (needs a max, then a pointer back), sorting
(one pointer per output, conditioned on what was already emitted) and shortest paths (repeated pointer hops plus a
min over path lengths).

`make(task, rng, level=0)` is the curriculum start: one-letter names and fewer items, same formats and op chains.

    task      prompt (bytes)                                         answer           expected ops
    lookup    "AGE OF JULIET: ROMEO 17, JULIET 13, ... =>"             13               point / attend (control)
    count     "COUNT ROMEO: ROMEO, JULIET, ROMEO, ... =>"              2                filter -> count
    argmax    "OLDEST: ROMEO 17, JULIET 13, NURSE 44 =>"               NURSE            max -> point
    sort      "SORT: TYBALT, JULIET, ROMEO =>"                         JULIET, ROMEO,.. point (per output)
    path      "EDGES: ROMEO-JULIET, JULIET-NURSE, ...; PATH ROMEO NURSE =>"  ROMEO JULIET NURSE  point chained, min

`path` is scored by checking the answer, not by string match: any valid path of minimal length scores 1, a valid
longer path 0.5 (for RL reward shaping), anything else 0.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from lilbits.shakespeare import characters


@dataclass
class TaskExample:
    prompt: bytes
    answer: str
    meta: Any = None


LEVEL = {"name_len": 8}  # set by make(level=...): level 0 uses one-letter names


def _names(rng: random.Random, k: int) -> list[str]:
    if LEVEL["name_len"] == 1:
        return rng.sample([chr(c) for c in range(65, 91)], k)
    pool = [n for n in characters() if len(n) <= 8]
    return rng.sample(pool, k)


def lookup(rng: random.Random, k: int = 4) -> TaskExample:
    names = _names(rng, k)
    ages = [rng.randrange(10, 90) for _ in names]
    i = rng.randrange(k)
    body = ", ".join(f"{n} {a}" for n, a in zip(names, ages, strict=True))
    return TaskExample(f"AGE OF {names[i]}: {body} =>".encode() + b" ", str(ages[i]))


def count(rng: random.Random, k: int = 7) -> TaskExample:
    names = _names(rng, 3)
    seq = [rng.choice(names) for _ in range(k)]
    w = rng.choice(names)
    return TaskExample(f"COUNT {w}: {', '.join(seq)} =>".encode() + b" ", str(seq.count(w)))


def argmax(rng: random.Random, k: int = 4) -> TaskExample:
    names = _names(rng, k)
    ages = rng.sample(range(10, 90), k)
    body = ", ".join(f"{n} {a}" for n, a in zip(names, ages, strict=True))
    return TaskExample(f"OLDEST: {body} =>".encode() + b" ", names[ages.index(max(ages))])


def sort(rng: random.Random, k: int = 4) -> TaskExample:
    names = _names(rng, k)
    return TaskExample(f"SORT: {', '.join(names)} =>".encode() + b" ", ", ".join(sorted(names)))


def _bfs(edges: list[tuple[str, str]], a: str, b: str) -> int | None:
    adj: dict[str, set[str]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    dist, q = {a: 0}, deque([a])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist.get(b)


def _shortest(edges, a, b) -> list[str]:
    adj: dict[str, list[str]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    prev, q = {a: a}, deque([a])
    while q:
        u = q.popleft()
        for v in sorted(adj.get(u, ())):
            if v not in prev:
                prev[v] = u
                q.append(v)
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def path(rng: random.Random, n_nodes: int = 5, n_edges: int = 6) -> TaskExample:
    """A small undirected graph over character names and a query whose shortest path has 2+ hops."""
    while True:
        names = _names(rng, n_nodes)
        edges: set[tuple[str, str]] = set()
        while len(edges) < n_edges:
            u, v = rng.sample(names, 2)
            edges.add((min(u, v), max(u, v)))
        e = sorted(edges, key=lambda _: rng.random())
        a, b = rng.sample(names, 2)
        d = _bfs(e, a, b)
        if d is not None and d >= 2:
            body = ", ".join(f"{u}-{v}" for u, v in e)
            ref = " ".join(_shortest(e, a, b))
            return TaskExample(f"EDGES: {body}; PATH {a} {b} =>".encode() + b" ", ref, (e, a, b, d))


def score_path(answer: str, ex: TaskExample) -> float:
    edges, a, b, d = ex.meta
    nodes = answer.split()
    es = {frozenset(x) for x in edges}
    if len(nodes) < 2 or nodes[0] != a or nodes[-1] != b:
        return 0.0
    if not all(frozenset((u, v)) in es for u, v in pairwise(nodes)):
        return 0.0
    return 1.0 if len(nodes) - 1 == d else 0.5


TASKS = {
    "lookup": (lookup, ({"point", "attend"},), "control: attention alone suffices"),
    "count": (count, ({"filter"}, {"count"}), "exact counting"),
    "argmax": (argmax, ({"max"}, {"point", "attend"}), "max over values, then a pointer back"),
    "sort": (sort, ({"point", "attend"},), "one pointer per output byte"),
    "path": (path, ({"point", "attend"}, {"point", "attend"}), "chained pointer hops"),
}


# level 0: a curriculum start, one-letter names and fewer items (same prompt formats and op chains)
EASY = {
    "lookup": {"k": 3},
    "count": {"k": 4},
    "argmax": {"k": 3},
    "sort": {"k": 3},
    "path": {"n_nodes": 4, "n_edges": 4},
}


def make(task: str, rng: random.Random, level: int = 1) -> TaskExample:
    """One problem. level 1: the full task; level 0: the easy curriculum start (EASY)."""
    if level >= 1:
        return TASKS[task][0](rng)
    LEVEL["name_len"] = 1
    try:
        return TASKS[task][0](rng, **EASY[task])
    finally:
        LEVEL["name_len"] = 8


# ---- dense pretraining streams ---------------------------------------------------------------------------------
# Task problems train 2-3 answer bytes per sequence. These streams make most bytes predictable by one of the ops
# (recall: pointer to a key, then read its value; copy: pointer stepping along a list; count: filter + count), and
# mark exactly those bytes as trained, so a sequence carries dozens of targets.


def _key(rng: random.Random) -> str:
    return rng.choice([chr(c) for c in range(65, 91)]) if rng.random() < 0.5 else _names(rng, 1)[0]


def recall_episode(rng: random.Random, k: int = 4, n_q: int = 6) -> list[tuple[bytes, bool]]:
    keys: list[str] = []
    while len(keys) < k:
        key = _key(rng)
        if key not in keys:
            keys.append(key)
    vals = {key: f"{rng.randrange(10, 100)}" for key in keys}
    segs: list[tuple[bytes, bool]] = [(("KV: " + ", ".join(f"{a}={vals[a]}" for a in keys) + " |").encode(), False)]
    for _ in range(n_q):
        a = rng.choice(keys)
        segs += [(f" {a}=".encode(), False), (vals[a].encode(), True)]
    return [*segs, (b"\n", False)]


def copy_episode(rng: random.Random, k: int = 5) -> list[tuple[bytes, bool]]:
    words = [_key(rng) for _ in range(k)]
    body = " ".join(words)
    return [(f"COPY: {body} |".encode(), False), (f" {body}".encode(), True), (b"\n", False)]


def count_episode(rng: random.Random, k: int = 6) -> list[tuple[bytes, bool]]:
    pool = [_key(rng) for _ in range(3)]
    seq = [rng.choice(pool) for _ in range(k)]
    segs: list[tuple[bytes, bool]] = [(f"ITEMS: {', '.join(seq)} |".encode(), False)]
    for a in dict.fromkeys(pool):
        segs += [(f" #{a}=".encode(), False), (str(seq.count(a)).encode(), True)]
    return [*segs, (b"\n", False)]


EPISODES = {"recall": recall_episode, "copy": copy_episode, "count": count_episode}


def dense_stream(rng: random.Random, n_bytes: int) -> list[tuple[bytes, bool]]:
    """Whole episodes of recall / copy / count, as (bytes, trained) segments, at most n_bytes in total (no episode
    is cut, so every trained byte has its context)."""
    segs: list[tuple[bytes, bool]] = []
    total, misses = 0, 0
    while misses < 8:
        ep = EPISODES[rng.choice(list(EPISODES))](rng)
        n = sum(len(b) for b, _ in ep)
        if total + n > n_bytes:
            misses += 1
            continue
        segs += ep
        total += n
    return segs


def score(task: str, answer: str, ex: TaskExample) -> float:
    if task == "path":
        return score_path(answer, ex)
    return float(answer.strip() == ex.answer)


def chain_match(task: str, trace: list[str]) -> float:
    """1 if the trace contains the task's expected op groups in order (as a subsequence), else 0."""
    groups, i = TASKS[task][1], 0
    for op in trace:
        if i < len(groups) and op in groups[i]:
            i += 1
    return float(i == len(groups))


__all__ = ["EPISODES", "TASKS", "TaskExample", "chain_match", "dense_stream", "make", "score", "score_path"]
