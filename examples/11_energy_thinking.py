"""Energy-based next-byte prediction that "thinks" by minimizing an energy, next to a plain LM of the same backbone.

After Energy-Based Transformers (Gladstone et al. 2025, https://arxiv.org/abs/2507.02092): the model scores
(context, candidate) pairs with a scalar energy and predicts by gradient descent on the candidate. More descent
steps at inference = more thinking; several candidates + keep the lowest energy = self-verification. Training puts
cross-entropy on the candidate after a random number of steps with a random step size, backpropagating through
the steps. Lean version (models/energy.py): a recurrent GDN backbone encodes the context once; only a small energy
head runs per thinking step.

    models   ebt  EnergyLM (GDN backbone + energy head)
             lm   StatefulLM with the same backbone and a tied linear head (one forward = one prediction)
    data     tiny-shakespeare bytes, then Shakespeare QA pairs with the loss on answer bytes (data/shakespeare)
    train    both models, same steps (smoke run)
    test     bits per byte on held-out speeches: lm, and ebt at 1 / 2 / 4 / 8 thinking steps and with 4-way
             self-verification; exact match on held-out QA pairs
    infer    decode speed: lm, and ebt at 1 / 4 / 8 steps (cost of thinking per byte)

Smoke-trained, the quality numbers only show the mechanics (energy falls as it thinks, the step-count curve is
measured), not whether thinking helps; that needs a real training run.

    uv run python examples/11_energy_thinking.py
"""

from __future__ import annotations

import math
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.measure import Meter
from lilbits.models.energy import EnergyConfig, EnergyLM
from lilbits.models.stateful_lm import StatefulLM
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import ShakespeareBytes, exact_match, load_qa, prompt_bytes, qa_tensors

THINK_STEPS = (1, 2, 4, 8)


def add_args(ap):
    ap.add_argument("--steps", type=int, default=6, help="training steps on bytes (default: smoke run)")
    ap.add_argument("--qa-steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--think", type=int, default=4, help="max thinking steps at training")
    ap.add_argument("--qa-eval", type=int, default=8)
    ap.add_argument("--gen", type=int, default=32)


def build(args, meta):
    torch.manual_seed(args.seed)
    cfg = EnergyConfig(d=args.d, heads=max(2, args.d // 64), n_layers=args.layers, steps=args.think)
    ebt = EnergyLM(cfg)
    lm = StatefulLM(256, args.d, cfg.heads, args.layers, partial(nn.Linear, bias=False), mixer="gdn")
    return {"ebt": ebt, "lm": lm}


def lm_loss(m: nn.Module, x, mask=None):
    logits, _ = m(x[:, :-1])
    ce = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="none")
    if mask is None:
        return ce.mean(), {}
    w = mask.flatten()
    return (ce * w).sum() / w.sum().clamp_min(1), {}


def train_fn(models, args, fab):
    src, qa = ShakespeareBytes("train", seed=args.seed), load_qa("train")
    g = torch.Generator().manual_seed(args.seed)

    def qa_batch(i):
        return qa_tensors([qa[(i * args.batch + j) % len(qa)] for j in range(args.batch)], args.seq)

    ebt = models["ebt"]
    out = {}
    for name, model, loss, qloss in (
        (
            "ebt",
            ebt,
            lambda m, x: ebt.training_loss(x, generator=g),
            lambda m, b: ebt.training_loss(b[0], b[1], generator=g),
        ),
        ("lm", models["lm"], lm_loss, lambda m, b: lm_loss(m, b[0], b[1])),
    ):
        with Meter() as m:
            info = fit(fab, model, lambda _: src.sample(args.batch, args.seq), loss, args.steps, lr=2e-3)
        out[f"{name}: bytes"] = {**info, **m.as_dict(), "step_ms": m.wall_s / max(1, args.steps) * 1e3}
        with Meter() as m:
            info = fit(fab, model, qa_batch, qloss, args.qa_steps, lr=1e-3)
        out[f"{name}: QA instruct"] = {**info, **m.as_dict()}
    return out, {}


def bpb(logits, x) -> float:
    return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()).item() / math.log(2)


def answerer(generate, max_bytes: int = 32):
    def answer(q: str) -> str:
        p = torch.tensor(list(prompt_bytes(q)))[None]
        gen = bytes(generate(p, max_bytes)[0, p.shape[1] :].tolist())
        return gen.split(b"\n")[0].decode(errors="replace")

    return answer


def test_fn(models, args):
    ebt, lm = models["ebt"], models["lm"]
    x = ShakespeareBytes("test", seed=123).sample(4, args.seq)
    h, _ = ebt.context(x[:, :-1])
    out: dict = {"lm_bits_per_byte": bpb(lm(x[:, :-1])[0], x)}
    curve = {}
    for s in THINK_STEPS:
        y = ebt.predict(h, s)
        _, es = ebt.think(h, s, generator=torch.Generator().manual_seed(0))
        curve[f"{s} steps"] = {"bits_per_byte": bpb(y, x), "energy_drop": (es[0] - es[-1]).mean().item()}
    curve["4 steps, 4-way verify"] = {"bits_per_byte": bpb(ebt.predict(h, 4, samples=4), x)}
    out["ebt_thinking_curve"] = curve
    pairs = load_qa("test")[: args.qa_eval]
    out["qa_exact_match"] = {
        "lm": exact_match(answerer(lambda p, n: lm.generate(p, n)[0]), pairs)["exact_match"],
        "ebt (4 steps)": exact_match(answerer(lambda p, n: ebt.generate(p, n, steps=4)), pairs)["exact_match"],
    }
    out["note"] = "smoke-trained: quality numbers show the mechanics only"
    return out


def infer_fn(models, args):
    ebt, lm = models["ebt"], models["lm"]
    prompt = ShakespeareBytes("test", seed=7).sample(1, 64)[:, :64]
    rows = {}

    def timed(name, fn):
        fn()
        t, c = time.perf_counter(), time.process_time()
        fn()
        w = time.perf_counter() - t
        rows[name] = {"bytes_per_s": args.gen / w, "cpu_ms": (time.process_time() - c) * 1e3}

    timed("lm", lambda: lm.generate(prompt, args.gen))
    for s in (1, 4, 8):
        timed(f"ebt, {s} thinking steps", lambda s=s: ebt.generate(prompt, args.gen, steps=s))
    head = sum(p.numel() for p in ebt.energy.parameters())
    return {"decode": rows, "energy_head_params": head, "backbone_params": sum(p.numel() for p in lm.parameters())}


EXAMPLE = Example(
    "11_energy_thinking", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("gen", "qa_eval")
)

if __name__ == "__main__":
    run(EXAMPLE)
