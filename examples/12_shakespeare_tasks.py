"""Toy Shakespeare tasks, each with an exact score: recall, copy / sort, tool calling, DPO and GRPO.

One byte-level model per run (`--model gdn`: a 2-layer GDN StatefulLM; `--model looped`: example 09's looped index
LM on bytes), one task per run (`--task`):

    needle   a line "The password is NAME." hidden in --context bytes of held-out Shakespeare, then
             "Q: What is the password?"; exact match of NAME (long-context recall: GDN state / sparse index)
    copy     "COPY: N1, N2, ..." -> the same list;  sort: -> the list in alphabetical order (character names)
    tools    QA questions answered through a tool: the model writes CALL: search("line") or next("line"), the
             environment runs it over the corpus and appends RESULT: ..., the model writes A: ...; scored on
             valid calls and exact answers
    dpo      SFT on QA, then DPO against a frozen copy on (question, right answer, wrong answer) pairs; scored on
             held-out preference accuracy and exact match
    grpo     SFT on QA, then GRPO: sample a group of answers per question, reward 1 for an exact answer, push up the
             better-than-group-average samples; scored on greedy exact match and mean sampled reward

Smoke runs prove each pipeline end to end (scores near 0 are expected at this training length).

    uv run python examples/12_shakespeare_tasks.py --task needle
    uv run python examples/12_shakespeare_tasks.py --task grpo --steps 4
"""

from __future__ import annotations

import copy
import random
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.measure import Meter
from lilbits.models.looped_index import LoopedIndexConfig, LoopedIndexLM
from lilbits.models.stateful_lm import StatefulLM
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import (
    copy_sort_example,
    exact_match,
    load_qa,
    needle_example,
    preference_pairs,
    prompt_bytes,
    qa_tensors,
    run_tool,
    segment_tensors,
    tool_example,
)
from lilbits.training.preference import dpo_loss, grpo_loss, sample, seq_logprob

TASKS = ("needle", "copy", "sort", "tools", "dpo", "grpo")


class ByteLM(nn.Module):
    """Uniform interface over the two backbones: logits(x), step(tokens, state), greedy answer."""

    def __init__(self, kind: str, d: int):
        super().__init__()
        self.kind = kind
        if kind == "gdn":
            self.net = StatefulLM(256, d, max(2, d // 64), 2, partial(nn.Linear, bias=False), mixer="gdn")
        else:
            self.net = LoopedIndexLM(LoopedIndexConfig(vocab_size=256, d_model=d, n_heads=max(2, d // 64), n_mtp=0))

    def step(self, tokens: torch.Tensor, state):
        net = self.net
        if isinstance(net, StatefulLM):
            return net(tokens, state)
        cache = state if state is not None else net.new_cache(tokens.shape[0])
        return net.logits(net.process(tokens, cache)), cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step(x, None)[0]

    @torch.no_grad()
    def complete(self, prompt: bytes, max_bytes: int = 48) -> str:
        out = sample(self.step, torch.tensor(list(prompt))[None], max_bytes, temperature=0.0)
        return bytes(out[0].tolist()).split(b"\n")[0].decode(errors="replace")


def add_args(ap):
    ap.add_argument("--task", default="needle", choices=TASKS)
    ap.add_argument("--model", default="gdn", choices=["gdn", "looped"])
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4, help="task training steps (default: smoke run)")
    ap.add_argument("--sft-steps", type=int, default=3, help="dpo / grpo: SFT steps on QA first")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=160)
    ap.add_argument("--context", type=int, default=512, help="needle: haystack bytes")
    ap.add_argument("--k", type=int, default=4, help="copy / sort: list length")
    ap.add_argument("--group", type=int, default=4, help="grpo: samples per question")
    ap.add_argument("--beta", type=float, default=0.1, help="dpo: beta")
    ap.add_argument("--eval", type=int, default=8, help="held-out examples scored")


def build(args, meta):
    torch.manual_seed(args.seed)
    return {"model": ByteLM(args.model, args.d)}


def masked_ce(model: nn.Module, batch) -> tuple[torch.Tensor, dict]:
    x, m = batch
    ce = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten(), reduction="none")
    w = m.flatten()
    return (ce * w).sum() / w.sum().clamp_min(1), {}


def task_rows(args, split_seed: int, n: int) -> list[list[tuple[bytes, bool]]]:
    rng = random.Random(split_seed)
    if args.task == "needle":
        out = []
        for _ in range(n):
            p, a = needle_example(args.context, rng, split="train" if split_seed < 1000 else "test")
            out.append([(p, False), ((a + "\n").encode(), True)])
        return out
    if args.task in ("copy", "sort"):
        return [
            [(p, False), ((a + "\n").encode(), True)]
            for p, a in (copy_sort_example(rng, args.k, args.task) for _ in range(n))
        ]
    if args.task == "tools":
        qa = load_qa("train")
        rows = [t for t in (tool_example(p) for p in qa) if t is not None]
        return [rows[rng.randrange(len(rows))] for _ in range(n)]
    raise ValueError(args.task)


def seq_len(args) -> int:
    return args.context if args.task == "needle" else args.seq


def train_fn(models, args, fab):
    model = models["model"]
    out: dict = {}
    if args.task in ("needle", "copy", "sort", "tools"):
        with Meter() as m:
            info = fit(
                fab,
                model,
                lambda i: segment_tensors(task_rows(args, i, args.batch), seq_len(args)),
                masked_ce,
                args.steps,
                lr=2e-3,
            )
        out["sft"] = {**info, **m.as_dict()}
        return out, {}
    qa = load_qa("train")

    def qa_batch(i):
        return qa_tensors([qa[(i * args.batch + j) % len(qa)] for j in range(args.batch)], args.seq)

    with Meter() as m:
        info = fit(fab, model, qa_batch, masked_ce, args.sft_steps, lr=2e-3)
    out["sft"] = {**info, **m.as_dict()}
    ref = copy.deepcopy(model).eval().requires_grad_(False)
    if args.task == "dpo":
        prefs = preference_pairs("train", 2000, seed=args.seed)

        def dpo_batch(i):
            ps = [prefs[(i * args.batch + j) % len(prefs)] for j in range(args.batch)]
            chosen = qa_tensors(ps, args.seq)
            rejected = qa_tensors([{**p, "a": p["rejected"]} for p in ps], args.seq)
            return chosen, rejected

        def dpo_step(mod, b):
            (xc, mc), (xr, mr) = b
            with torch.no_grad():
                rc, rr = seq_logprob(ref, xc, mc), seq_logprob(ref, xr, mr)
            return dpo_loss(seq_logprob(mod, xc, mc), seq_logprob(mod, xr, mr), rc, rr, args.beta)

        with Meter() as m:
            info = fit(fab, model, dpo_batch, dpo_step, args.steps, lr=1e-3)
        out["dpo"] = {**info, **m.as_dict()}
        return out, {}
    # GRPO: short-answer questions (speaker / next speaker / line count)
    short = [p for p in qa if p["kind"] != "next_line"]
    g = torch.Generator().manual_seed(args.seed)

    def grpo_batch(i):
        return [short[(i * args.batch + j) % len(short)] for j in range(args.batch)]

    def grpo_step(mod, ps):
        rows, rewards, lengths = [], [], []
        for p in ps:
            pr = torch.tensor(list(prompt_bytes(p["q"])))[None].expand(args.group, -1)
            done = sample(model.step, pr, 24, temperature=1.0, generator=g)
            for s in done:
                txt = bytes(s.tolist()).split(b"\n")[0]
                rewards.append(float(txt.decode(errors="replace") == p["a"]))
                comp = txt + b"\n"
                lengths.append(len(comp))
                rows.append([(prompt_bytes(p["q"]), False), (comp, True)])
        x, mk = segment_tensors(rows, args.seq)
        with torch.no_grad():
            ref_lp = seq_logprob(ref, x, mk)
        return grpo_loss(
            seq_logprob(mod, x, mk), torch.tensor(rewards), args.group, torch.tensor(lengths), ref_lp, kl=0.05
        )

    with Meter() as m:
        info = fit(fab, model, grpo_batch, grpo_step, args.steps, lr=1e-3)
    out["grpo"] = {**info, **m.as_dict()}
    return out, {}


def test_fn(models, args):
    model = models["model"]
    if args.task in ("needle", "copy", "sort"):
        rows = task_rows(args, 10_000, args.eval)
        hits = [model.complete(r[0][0]) == r[1][0].decode().strip() for r in rows]
        return {"task": args.task, "exact_match": sum(hits) / len(hits)}
    qa = load_qa("test")[: 4 * args.eval]
    if args.task == "tools":
        pairs = [p for p in qa if tool_example(p) is not None][: args.eval]
        valid, hits = 0, 0
        for p in pairs:
            head = prompt_bytes(p["q"]).removesuffix(b"A: ") + b"CALL: "
            call = model.complete(head)
            result = run_tool(call)
            valid += result != "ERROR"
            ans = model.complete(head + f"{call}\nRESULT: {result}\nA: ".encode())
            hits += ans == p["a"]
        return {"task": "tools", "valid_call_rate": valid / len(pairs), "exact_match": hits / len(pairs)}
    pairs = qa[: args.eval]
    out: dict = {"task": args.task, **exact_match(lambda q: model.complete(prompt_bytes(q)), pairs)}
    if args.task == "dpo":
        ps = preference_pairs("test", args.eval, seed=1)
        xc, mc = qa_tensors(ps, args.seq)
        xr, mr = qa_tensors([{**p, "a": p["rejected"]} for p in ps], args.seq)
        out["preference_acc"] = (seq_logprob(model, xc, mc) > seq_logprob(model, xr, mr)).float().mean().item()
    return out


def infer_fn(models, args):
    model = models["model"]
    p = prompt_bytes(load_qa("test")[0]["q"])
    with Meter() as m:
        ans = model.complete(p, 32)
    return {"answer_sample": ans, "ms": m.wall_s * 1e3}


EXAMPLE = Example(
    "12_shakespeare_tasks", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("eval",)
)

if __name__ == "__main__":
    run(EXAMPLE)
