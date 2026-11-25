"""Looped model that composes typed query operators over the context and a register workspace, per loop.

Each loop is one routed instruction (mixers/operators.py): the router picks an op (filter, point, topk, sum, mean,
max, count, attend, mix, noop), which SET / VEC registers it reads, and where it writes; a gate folds the VEC
result into the hidden state. Several loops chain into small programs, e.g. point -> mean (attention in two steps),
filter -> count, filter -> topk -> mean. Training routes softly (Gumbel-softmax, temperature annealed from 1 to
--tau-end) with a load-balance loss; inference routes hard (one op per loop per position) and records the trace.

    models   ops      OpLoopLM with every operator
             attend   the same loop with attention as its only op (the transformer-style baseline)
    tasks    sort       "SORT: N1, N2, ... =>" -> the names in alphabetical order (pointer-style)
             count      a Shakespeare excerpt + "COUNT word =>" -> occurrences (filter + count)
             who_said   an excerpt + a quoted fragment -> its speaker (filter / point + gather)
             who_next   an excerpt + a fragment -> the speaker after its speaker (two hops)
    test     exact match per loop count (1 .. --loops) for both models, op usage per loop, one sample trace
    infer    answer latency and the trace for one held-out prompt

Smoke-trained: exact match near 0 and early op usage only show the mechanics.

    uv run python examples/13_operator_loop.py --task count
"""

from __future__ import annotations

import random
from collections import Counter

import torch

from lilbits.measure import Meter
from lilbits.models.op_loop import OpLoopLM
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import copy_sort_example, count_example, segment_tensors, who_next_example, who_said_example

TASKS = {
    "sort": lambda rng, split: copy_sort_example(rng, 4, "sort"),
    "count": count_example,
    "who_said": who_said_example,
    "who_next": who_next_example,
}


def add_args(ap):
    ap.add_argument("--task", default="count", choices=list(TASKS))
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4, help="training steps (default: smoke run)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=200)
    ap.add_argument("--tau-end", type=float, default=0.3)
    ap.add_argument("--eval", type=int, default=6)


def build(args, meta):
    torch.manual_seed(args.seed)
    return {
        "ops": OpLoopLM(args.d, args.loops),
        "attend": OpLoopLM(args.d, args.loops, ops=("attend",)),
    }


def rows(args, seed: int, n: int, split: str):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p, a = TASKS[args.task](rng, split)
        out.append([(p, False), ((a + "\n").encode(), True)])
    return out


def train_fn(models, args, fab):
    out = {}
    g = torch.Generator().manual_seed(args.seed)
    for name, model in models.items():

        def batch(i, model=model):
            model.tau = 1.0 + (args.tau_end - 1.0) * i / max(1, args.steps - 1)  # anneal the router temperature
            return segment_tensors(rows(args, i, args.batch, "train"), args.seq)

        with Meter() as m:
            info = fit(
                fab, model, batch, lambda mod, b, model=model: model.loss(b[0], b[1], generator=g), args.steps, lr=2e-3
            )
        out[name] = {**info, **m.as_dict()}
    return out, {}


def test_fn(models, args):
    held = rows(args, 99_999, args.eval, "test")
    out: dict = {}
    for name, model in models.items():
        per_loops, usage = {}, [Counter() for _ in range(args.loops)]
        for n in range(1, args.loops + 1):
            hits = 0
            for r in held:
                ans, trace = model.complete(r[0][0], n_loops=n)
                hits += ans == r[1][0].decode().strip()
                if n == args.loops:
                    for i, op in enumerate(trace):
                        usage[i][op] += 1
            per_loops[f"{n} loops"] = hits / len(held)
        out[name] = {
            "exact_match_by_loops": per_loops,
            "op_usage_by_loop": {f"loop {i}": dict(c) for i, c in enumerate(usage)},
        }
    _, trace = models["ops"].complete(held[0][0][0])
    out["sample_trace"] = " -> ".join(trace)
    out["note"] = "smoke-trained: scores and op usage show the mechanics only"
    return out


def infer_fn(models, args):
    p = rows(args, 123, 1, "test")[0][0][0]
    out = {}
    for name, model in models.items():
        with Meter() as m:
            ans, trace = model.complete(p, 16)
        out[name] = {"ms": m.wall_s * 1e3, "answer": ans, "trace": " -> ".join(trace)}
    return out


EXAMPLE = Example(
    "13_operator_loop", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("eval",)
)

if __name__ == "__main__":
    run(EXAMPLE)
