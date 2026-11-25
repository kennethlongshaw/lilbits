"""Pocket LM: a byte-level model for one CPU core, every trick a switch, next to an FP16 model of the same size.

The model (models/pocket.py): bytes -> small byte encoder -> router (which bytes open a patch) -> example 09's
looped index body, run once per patch -> byte decoder -> next byte, with MTP byte drafts and parallel verify.
Each piece can be turned off, so a benchmark can say what each one costs or saves:

    --router pair|hnet|none   exact byte-pair router (space-patching init) | H-Net cosine router | no patching
    --core fp|littlebit       stage 2 converts the looped core to sub-1-bit LittleBit (Dual-SVID), INT8 base scales
    --scale-bits 8|16         LittleBit h, g, l storage
    --no-index                body without the sparse patch index (GDN state + loop memory only)
    --mtp N                   byte draft heads (0: plain decode only)
    --kernel native|lut       infer: LittleBit prelude and looped core on the LUT kernels

The comparison model, `fp16_same_size`, is a byte-level GDN LM in FP16 whose deployed bytes match the pocket model's
(width searched at build time), trained on the same bytes for the same number of steps.

Data: the tiny-shakespeare bytes (train split: the first 90% of speeches), then an instruct stage on Shakespeare
QA pairs with the loss on answer bytes only (data/shakespeare). Test: bits per byte on held-out speeches and exact
match on held-out QA pairs, for both models.

Training is a smoke run (a few steps) that proves the code works: bits per byte here say nothing about which model
is better. What the run does measure: deployed bytes, checkpoint size, RAM, train step cost, decode speed and its
split between the byte layers and the body, and whether speculative decode reproduces greedy decode.

    uv run python examples/10_pocket_lm.py
    uv run python examples/10_pocket_lm.py --router none --core fp      # ablation: no patching, FP core
"""

from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.inference.lut import pack_lut
from lilbits.measure import Meter, size_report
from lilbits.models.pocket import PocketConfig, PocketLM, convert_pocket_core, fp16_baseline, matched_fp16_width
from lilbits.models.stateful_lm import state_bytes
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import ShakespeareBytes, exact_match, load_qa, prompt_bytes, qa_tensors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_args(ap):
    ap.add_argument("--steps", type=int, default=4, help="training steps per stage (default: smoke run)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256, help="bytes per training sequence")
    ap.add_argument("--d-byte", type=int, default=64)
    ap.add_argument("--d-body", type=int, default=512)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--router", default="pair", choices=["pair", "hnet", "none"])
    ap.add_argument("--core", default="littlebit", choices=["fp", "littlebit"])
    ap.add_argument("--core-bpw", type=float, default=0.55)
    ap.add_argument("--scale-bits", type=int, default=8, choices=[8, 16])
    ap.add_argument("--coda", default="littlebit", choices=["fp", "littlebit"], help="coda block linears")
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--mtp", type=int, default=2)
    ap.add_argument("--qa-steps", type=int, default=4, help="instruct stage: steps on Shakespeare QA pairs")
    ap.add_argument("--qa-eval", type=int, default=16, help="test: held-out QA pairs scored by exact match")
    ap.add_argument("--eval-seqs", type=int, default=8, help="test: held-out byte sequences for bits per byte")
    ap.add_argument(
        "--epochs",
        type=float,
        default=0.0,
        help="> 0: set --steps / --qa-steps to cover this many passes over the train bytes and QA pairs",
    )
    ap.add_argument("--schedule", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--prompt", type=int, default=128, help="infer: prompt bytes")
    ap.add_argument("--gen", type=int, default=64, help="infer: bytes to decode")
    ap.add_argument("--kernel", default="native", choices=["native", "lut"])


def pocket_cfg(args) -> PocketConfig:
    return PocketConfig(
        d_byte=args.d_byte,
        d_body=args.d_body,
        heads=max(2, args.d_body // 128),
        router=args.router,
        n_loops=args.loops,
        core_bpw=args.core_bpw,
        scale_bits=args.scale_bits,
        coda_linear=args.coda,
        n_mtp=args.mtp,
    )


def deployed(m: nn.Module) -> float:
    return size_report(m)["deployed_total_bytes"]


def build(args, meta):
    """Training builds the FP-core model (stage 1); a checkpoint's meta says whether the core was converted."""
    torch.manual_seed(args.seed)
    pocket = PocketLM(pocket_cfg(args))
    pocket.body.use_index = not args.no_index
    if meta.get("core_converted"):
        convert_pocket_core(pocket)
    d = meta.get("baseline_d")
    if d is None:  # match the size the pocket model is deployed at (after stage 2)
        final = copy.deepcopy(pocket)
        if args.core == "littlebit":
            convert_pocket_core(final)
        d = matched_fp16_width(deployed(final))
    return {"pocket": pocket, "fp16_same_size": fp16_baseline(d)}


def baseline_loss(m: nn.Module, x, mask=None):
    logits, _ = m(x[:, :-1])
    ce = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="none")
    if mask is None:
        return ce.mean(), {}
    w = mask.flatten()
    return (ce * w).sum() / w.sum().clamp_min(1), {}


def train_fn(models, args, fab):
    src = ShakespeareBytes("train", seed=args.seed)
    qa = load_qa("train")
    pocket = models["pocket"]
    out: dict = {}
    n_stages = 2 if args.core == "littlebit" else 1
    if args.epochs > 0:  # an "epoch" = as many sampled bytes as the train split holds (windows are random)
        args.steps = max(1, round(args.epochs * len(src.data) / (args.batch * args.seq) / n_stages))
        args.qa_steps = max(1, round(args.epochs * len(qa) / args.batch))
    out["schedule"] = {"steps_per_stage": args.steps, "qa_steps": args.qa_steps, "lr": args.schedule}
    stages = [("1 FP core", False)] + ([("2 LittleBit core", True)] if args.core == "littlebit" else [])
    for name, convert in stages:
        if convert:
            convert_pocket_core(pocket)
        with Meter() as m:
            info = fit(
                fab,
                pocket,
                lambda _: src.sample(args.batch, args.seq),
                lambda mod, x: pocket.training_loss(x),
                args.steps,
                lr=2e-3,
                schedule=args.schedule,
            )
        out[f"pocket: {name}"] = {**info, **m.as_dict(), "step_ms": m.wall_s / max(1, args.steps) * 1e3}

    def qa_batch(i):
        return qa_tensors([qa[(i * args.batch + j) % len(qa)] for j in range(args.batch)], args.seq)

    with Meter() as m:  # instruct stage: loss on answer bytes only
        info = fit(
            fab,
            pocket,
            qa_batch,
            lambda mod, b: pocket.training_loss(b[0], mask=b[1]),
            args.qa_steps,
            lr=1e-3,
            schedule=args.schedule,
        )
    out["pocket: 3 QA instruct"] = {**info, **m.as_dict()}
    steps = args.steps * len(stages)
    base = models["fp16_same_size"]
    with Meter() as m:
        info = fit(
            fab, base, lambda _: src.sample(args.batch, args.seq), baseline_loss, steps, lr=2e-3, schedule=args.schedule
        )
    out["fp16_same_size"] = {**info, **m.as_dict(), "step_ms": m.wall_s / max(1, steps) * 1e3}
    with Meter() as m:
        info = fit(
            fab,
            base,
            qa_batch,
            lambda mod, b: baseline_loss(mod, b[0], b[1]),
            args.qa_steps,
            lr=1e-3,
            schedule=args.schedule,
        )
    out["fp16_same_size: QA instruct"] = {**info, **m.as_dict()}
    return out, {
        "core_converted": args.core == "littlebit",
        "baseline_d": models["fp16_same_size"].embed.weight.shape[1],
    }


def held_out(args) -> torch.Tensor:
    return ShakespeareBytes("test", seed=123).sample(args.eval_seqs, args.seq)


def answerer(generate, max_bytes: int = 48):
    """Greedy answer to a QA prompt: bytes up to the first newline."""

    def answer(q: str) -> str:
        p = torch.tensor(list(prompt_bytes(q)))[None]
        out = generate(p, max_bytes)
        gen = bytes(out[0, p.shape[1] :].tolist())
        return gen.split(b"\n")[0].decode(errors="replace")

    return answer


def test_fn(models, args):
    x = held_out(args)
    pairs = load_qa("test")[: args.qa_eval]
    y, aux = models["pocket"](x, models["pocket"].new_cache(x.shape[0]))
    pl = F.cross_entropy(models["pocket"].logits(y[:, :-1]).flatten(0, 1), x[:, 1:].flatten()).item()
    bl, _ = baseline_loss(models["fp16_same_size"], x)
    return {
        "pocket": {
            "bits_per_byte": pl / math.log(2),
            "bytes_per_patch": aux["b"].numel() / max(1, int(aux["b"].sum())),
        },
        "fp16_same_size": {"bits_per_byte": bl.item() / math.log(2), "bytes_per_patch": 1.0},
        "qa_exact_match": {
            "pocket": exact_match(answerer(lambda p, n: models["pocket"].generate(p, n)[0]), pairs),
            "fp16_same_size": exact_match(answerer(lambda p, n: models["fp16_same_size"].generate(p, n)[0]), pairs),
        },
        "note": "smoke-trained: bits per byte and exact match do not rank the models",
    }


def infer_fn(models, args):
    pocket, base = models["pocket"], models["fp16_same_size"]
    prompt = held_out(args)[:1, : args.prompt]
    rows: dict = {}

    def timed(name, fn, n_bytes):
        fn()  # warm-up
        with Meter() as m:
            res = fn()
        rows[name] = {
            "bytes_per_s": n_bytes / m.wall_s,
            "wall_ms": m.wall_s * 1e3,
            "cpu_ms": m.cpu_s * 1e3,
            "peak_rss_growth_mib": m.peak_extra_mib,
        }
        return res

    native = None
    if args.kernel == "lut":
        native, _ = pocket.generate(prompt, args.gen)
        pack_lut(pocket.body)
    greedy, st = timed("pocket greedy", lambda: pocket.generate(prompt, args.gen), args.gen)
    rows["pocket greedy"].update(body_passes=st["body_passes"], bytes_per_body_pass=st["bytes_per_body_pass"])
    pocket.timing = {}
    pocket.generate(prompt, args.gen)
    split = {k: v * 1e3 for k, v in pocket.timing.items()}
    pocket.timing = None
    halted, st_h = timed("pocket greedy, halting eps 0.1", lambda: pocket.generate(prompt, args.gen, 0.1), args.gen)
    rows["pocket greedy, halting eps 0.1"].update(
        mean_loops=st_h["mean_loops"], same_as_greedy=torch.equal(halted, greedy)
    )
    if args.mtp:
        spec, st_s = timed("pocket MTP + verify", lambda: pocket.generate_speculative(prompt, args.gen), args.gen)
        rows["pocket MTP + verify"].update(
            acceptance=st_s["acceptance"],
            bytes_per_pass=st_s["bytes_per_pass"],
            same_as_greedy=torch.equal(spec, greedy),
        )
    # decode cost on real text: feed held-out bytes one at a time (patch rate of real text, not of what a
    # smoke-trained model happens to generate)
    real = held_out(args)[1:2, : args.gen]

    def stream_pocket():
        c = pocket.new_cache(1)
        pocket(prompt, c)
        for t in range(real.shape[1]):
            pocket(real[:, t : t + 1], c)
        return c

    def stream_base():
        _, s = base(prompt)
        for t in range(real.shape[1]):
            _, s = base(real[:, t : t + 1], s)
        return s

    c = timed("pocket, streaming real text", stream_pocket, real.shape[1])
    rows["pocket, streaming real text"]["bytes_per_body_pass"] = real.shape[1] / max(1, c.patches - 1)
    timed("fp16_same_size, streaming real text", stream_base, real.shape[1])
    (_, states) = timed("fp16_same_size greedy", lambda: base.generate(prompt, args.gen), args.gen)
    return {
        "kernel": args.kernel,
        **({"lut_same_as_native": torch.equal(native, greedy)} if native is not None else {}),
        "decode": rows,
        "pocket_decode_split_ms": split,
        "state_bytes": {"pocket": sum(st["cache"].values()), "fp16_same_size": state_bytes(states)},
        "deployed_bytes": {"pocket": deployed(pocket), "fp16_same_size": deployed(base)},
        "baseline_width": base.embed.weight.shape[1],
    }


EXAMPLE = Example(
    "10_pocket_lm",
    __doc__ or "",
    add_args,
    build,
    train_fn,
    test_fn,
    infer_fn,
    runtime_args=("prompt", "gen", "kernel"),
)

if __name__ == "__main__":
    run(EXAMPLE)
