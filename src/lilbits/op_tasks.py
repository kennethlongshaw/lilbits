"""Algorithmic tasks over Shakespeare character names, each built to need a known chain of query operators.

Every task gives a prompt, a reference answer, an exact score, and the operators a solution should use (for
`OperatorLoop` traces). Several are ones softmax attention handles badly on its own: exact counting (attention
averages, so it reports a proportion, not a number), argmax over values (needs a max, then a pointer back), sorting
(one pointer per output, conditioned on what was already emitted) and shortest paths (repeated pointer hops plus a
min over path lengths).

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


def _names(rng: random.Random, k: int) -> list[str]:
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


def make(task: str, rng: random.Random) -> TaskExample:
    return TASKS[task][0](rng)


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


__all__ = ["TASKS", "TaskExample", "chain_match", "make", "score", "score_path"]
