"""Looped model that composes typed query operators over the context and a register workspace, per loop.

Each loop is one routed instruction (mixers/operators.py): the router picks an op (filter, point, topk, sum, mean,
max, count, attend, mix, noop), which SET / VEC registers it reads, and where it writes; a gate folds the VEC
result into the hidden state. Loops chain into small programs. Training routes softly (Gumbel-softmax, temperature
annealed from 1 to --tau-end) with a load-balance loss; inference routes hard and records the op trace.

    models   ops      OpLoopLM with every operator
             attend   the same loop with attention as its only op (the transformer-style baseline)
    tasks    (op_tasks.py, Shakespeare character names; each built to need a known op chain)
             lookup   "AGE OF X: A 17, X 13, ... =>" -> 13                 point / attend (control)
             count    "COUNT X: X, Y, X, ... =>" -> 2                      filter -> count
             argmax   "OLDEST: A 17, B 13, ... =>" -> the oldest name       max -> point
             sort     "SORT: ... =>" -> names in order                     point per output
             path     "EDGES: A-B, ...; PATH A C =>" -> "A B C"            pointer hops; SFT on BFS paths, then
                      GRPO (--grpo-steps) with reward 1 for a shortest valid path, 0.5 for a longer valid one
    test     score per loop count (1 .. --loops) for both models; for the ops model, how often the trace at the
             first answer byte contains the task's expected op chain, and op usage per loop
    infer    answer latency and the trace for one held-out prompt

    uv run python examples/13_operator_loop.py --task count
    uv run python examples/13_operator_loop.py --task path --grpo-steps 4
"""

from __future__ import annotations

import random
from collections import Counter

import torch

from lilbits.measure import Meter
from lilbits.models.op_loop import OpLoopLM
from lilbits.op_tasks import TASKS, chain_match, make, score
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import segment_tensors
from lilbits.training.preference import grpo_loss, seq_logprob


def add_args(ap):
    ap.add_argument("--task", default="count", choices=list(TASKS))
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4, help="SFT steps (default: smoke run)")
    ap.add_argument("--grpo-steps", type=int, default=0, help="path: GRPO steps after SFT")
    ap.add_argument("--group", type=int, default=4, help="GRPO samples per prompt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=200)
    ap.add_argument("--tau-end", type=float, default=0.3)
    ap.add_argument("--eval", type=int, default=6)
    ap.add_argument("--epochs", type=float, default=0.0, help="> 0: SFT steps = epochs x 4000 examples / batch")
    ap.add_argument("--schedule", default="constant", choices=["constant", "cosine"])


def build(args, meta):
    torch.manual_seed(args.seed)
    return {
        "ops": OpLoopLM(args.d, args.loops),
        "attend": OpLoopLM(args.d, args.loops, ops=("attend",)),
    }


def examples(args, seed: int, n: int):
    rng = random.Random(seed)
    return [make(args.task, rng) for _ in range(n)]


def rows(exs):
    return [[(e.prompt, False), ((e.answer + "\n").encode(), True)] for e in exs]


def train_fn(models, args, fab):
    out: dict = {}
    g = torch.Generator().manual_seed(args.seed)
    if args.epochs > 0:
        args.steps = max(1, round(args.epochs * 4000 / args.batch))
    for name, model in models.items():

        def batch(i, model=model):
            model.tau = 1.0 + (args.tau_end - 1.0) * i / max(1, args.steps - 1)  # anneal the router temperature
            return segment_tensors(rows(examples(args, i, args.batch)), args.seq)

        with Meter() as m:
            info = fit(
                fab,
                model,
                batch,
                lambda mod, b, model=model: model.loss(b[0], b[1], generator=g),
                args.steps,
                lr=2e-3,
                schedule=args.schedule,
            )
        out[f"{name}: SFT"] = {**info, **m.as_dict()}
        if args.grpo_steps:
            ref = OpLoopLM(args.d, args.loops, ops=model.ops)
            ref.load_state_dict(model.state_dict())
            ref.eval().requires_grad_(False)
            ref.tau, ref.hard = args.tau_end, True

            def grpo_step(mod, exs, model=model, ref=ref):
                rws, rewards, lengths = [], [], []
                for e in exs:
                    for comp in model.sample(e.prompt, args.group, generator=g):
                        rewards.append(score(args.task, comp.decode(errors="replace"), e))
                        lengths.append(len(comp) + 1)
                        rws.append([(e.prompt, False), (comp + b"\n", True)])
                x, mk = segment_tensors(rws, args.seq)
                with torch.no_grad():
                    rl = seq_logprob(lambda t: ref(t)[0], x, mk)
                lp = seq_logprob(lambda t: model(t)[0], x, mk)
                return grpo_loss(lp, torch.tensor(rewards), args.group, torch.tensor(lengths), rl, kl=0.05)

            with Meter() as m:
                info = fit(
                    fab, model, lambda i: examples(args, 50_000 + i, args.batch), grpo_step, args.grpo_steps, lr=5e-4
                )
            out[f"{name}: GRPO"] = {**info, **m.as_dict()}
    return out, {}


def test_fn(models, args):
    held = examples(args, 99_999, args.eval)
    out: dict = {"task": args.task, "why": TASKS[args.task][2]}
    for name, model in models.items():
        by_loops, usage, chains = {}, [Counter() for _ in range(args.loops)], []
        for n in range(1, args.loops + 1):
            total = 0.0
            for e in held:
                ans, trace = model.complete(e.prompt, n_loops=n)
                total += score(args.task, ans, e)
                if n == args.loops:
                    chains.append(chain_match(args.task, trace))
                    for i, op in enumerate(trace):
                        usage[i][op] += 1
            by_loops[f"{n} loops"] = total / len(held)
        out[name] = {"score_by_loops": by_loops}
        if len(model.ops) > 1:
            out[name]["expected_chain_rate"] = sum(chains) / len(chains)
            out[name]["op_usage_by_loop"] = {f"loop {i}": dict(c.most_common(3)) for i, c in enumerate(usage)}
    _, trace = models["ops"].complete(held[0].prompt)
    out["sample_trace"] = " -> ".join(trace)
    return out


def infer_fn(models, args):
    e = examples(args, 123, 1)[0]
    out = {}
    for name, model in models.items():
        with Meter() as m:
            ans, trace = model.complete(e.prompt, 24)
        out[name] = {"ms": m.wall_s * 1e3, "answer": ans, "reference": e.answer, "trace": " -> ".join(trace)}
    return out


EXAMPLE = Example(
    "13_operator_loop", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("eval",)
)

if __name__ == "__main__":
    run(EXAMPLE)
