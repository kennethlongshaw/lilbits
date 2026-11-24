"""Looped index LM: the full lean-decoder design, trained with its staged curriculum (smoke run).

    tokens -> embed (tied) -> PRELUDE (LittleBit GDN, writes the INDEX) -> LOOPED CORE x N (shared LittleBit
    signs, per-loop FP scales, GDN state per loop, loop memory, retrieval: block summaries -> sign signatures ->
    exact top-k, dense fallback) -> CODA (FP GDN, LM head, MTP draft heads) -> VERIFY next pass

See src/lilbits/models/looped_index.py for the design. This script trains it in the planned stages, next to a
dense FP reference (same architecture, FP everywhere, dense retrieval), on long-context recall and two-hop
questions:

    stage 1  looped GDN core, FP, fixed N, loop memory
    stage 2  + LittleBit core (Dual-SVID from the stage-1 weights), per-loop scales
    stage 3  + sparse index and deep summaries; the block scorer is trained to match dense attention
    stage 4  + MTP draft heads
    stage 5  + halting (decode only: no training, calibrates the early-exit threshold)

Phases:
    train   stages 1-4 for the lean model and the dense FP reference, on Lightning Fabric, cost per stage
    test    held-out loss, recall and two-hop accuracy, both models
    infer   stage 5 and decode: greedy (fixed N, and with halting thresholds), MTP drafts + parallel verify
            (checked token for token against greedy), draft acceptance, adaptive k, bytes per memory tier

Default is a smoke run (a few steps per stage) that proves the code paths work. Recall and two-hop accuracy at
this size and step count are at chance; they are there so the same script can measure a real run.

    uv run python examples/09_looped_index_lm.py
    uv run python examples/09_looped_index_lm.py --steps 300 --length 512   # an actual (still small) run
"""

from __future__ import annotations

import copy
import time

import torch
import torch.nn.functional as F

from lilbits.data import RecallSource
from lilbits.measure import Meter
from lilbits.models.looped_index import LoopedIndexConfig, LoopedIndexLM, convert_core_to_littlebit, memory_tiers
from lilbits.runner import Example, fit, run

STAGES = [
    ("1 looped GDN core, FP, fixed N, loop memory", dict(index=False, mtp=False, convert=False)),
    ("2 + LittleBit core, per-loop scales", dict(index=False, mtp=False, convert=True)),
    ("3 + sparse index, deep summaries, index distillation", dict(index=True, mtp=False, convert=False)),
    ("4 + MTP draft heads", dict(index=True, mtp=True, convert=False)),
]


@torch.no_grad()
def evaluate(model: LoopedIndexLM, x: torch.Tensor, kind: torch.Tensor) -> dict:
    model.eval()
    h = model.process(x, model.new_cache(x.shape[0]))
    logits = model.logits(h)
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item()
    hit = logits[:, -2].argmax(-1).eq(x[:, -1]).float()
    model.train()
    return {
        "loss": loss,
        "recall_acc": hit[kind == 0].mean().item() if (kind == 0).any() else float("nan"),
        "two_hop_acc": hit[kind == 1].mean().item() if (kind == 1).any() else float("nan"),
    }


def add_args(ap):
    ap.add_argument("--steps", type=int, default=6, help="training steps per stage (default: smoke run)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--length", type=int, default=128, help="sequence length (blocks of 64 tokens)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--gen", type=int, default=24, help="infer: tokens to decode")


def build(args, meta):
    src = RecallSource(length=args.length, seed=0)
    cfg = LoopedIndexConfig(vocab_size=src.vocab, d_model=args.d, n_loops=args.loops)
    ref_cfg = copy.deepcopy(cfg)
    lean = LoopedIndexLM(cfg)
    if meta.get("core_converted"):
        convert_core_to_littlebit(lean)
    ref_cfg.prelude_linear, ref_cfg.flat = "fp", -1.0  # FP everywhere; flat < 0: retrieval is always dense
    return {"lean": lean, "fp_dense_ref": LoopedIndexLM(ref_cfg)}


def train_fn(models, args, fab):
    out = {}
    for name, st in STAGES:
        if st["convert"]:
            convert_core_to_littlebit(models["lean"])
        for tag, m in models.items():
            m.use_index = m.distill = st["index"]
            src = RecallSource(length=args.length, seed=args.seed)
            with Meter() as meter:
                info = fit(
                    fab,
                    m,
                    lambda _, src=src: src.sample(args.batch)[0],
                    lambda model, x, mtp=st["mtp"]: model.training_loss(x, mtp=mtp),  # ty: ignore[not-callable]
                    args.steps,
                    lr=2e-3,
                )
            out[f"{name} [{tag}]"] = {**info, **meter.as_dict()}
    return {"stages": out}, {"core_converted": True}


def held_out(args):
    return RecallSource(length=args.length, seed=123).sample(32)


def test_fn(models, args):
    x, kind = held_out(args)
    out = {}
    for tag, m in models.items():
        m.use_index, m.distill = True, False
        out[tag] = evaluate(m, x, kind)
        m.eval()
    return out


def infer_fn(models, args):
    model = models["lean"]
    model.use_index, model.distill = True, False
    prompt = held_out(args)[0][:1, : args.length - 1]
    rows = {}

    def timed(fn):
        t, c = time.perf_counter(), time.process_time()
        res = fn()
        return res, (time.perf_counter() - t) * 1e3, (time.process_time() - c) * 1e3

    (greedy, st0), w, c = timed(lambda: model.generate(prompt, args.gen))
    rows["greedy, fixed N"] = {"wall_ms": w, "cpu_ms": c, "passes": st0["passes"], "mean_loops": st0["mean_loops"]}
    for eps in (0.05, 0.2):  # stage 5: halting thresholds, decode only
        (out, st), w, c = timed(lambda eps=eps: model.generate(prompt, args.gen, halt_eps=eps))
        rows[f"greedy, halting eps {eps}"] = {
            "wall_ms": w,
            "cpu_ms": c,
            "passes": st["passes"],
            "mean_loops": st["mean_loops"],
            "same_as_greedy": torch.equal(out, greedy),
        }
    (spec, st2), w, c = timed(lambda: model.generate_speculative(prompt, args.gen))
    rows["MTP drafts + parallel verify"] = {
        "wall_ms": w,
        "cpu_ms": c,
        "passes": st2["passes"],
        "tokens_per_pass": st2["tokens_per_pass"],
        "acceptance": st2["acceptance"],
        "k_trace": st2["k_trace"],
        "same_as_greedy": torch.equal(spec, greedy),
    }
    distinct = len(set(greedy[0, prompt.shape[1] :].tolist()))
    return {
        "decode": rows,
        "greedy_distinct_tokens": distinct,  # a smoke-trained model repeats filler: drafts are trivially accepted
        "memory_tiers": memory_tiers(model, st2["cache"], prompt.shape[1] + args.gen),
    }


EXAMPLE = Example(
    "09_looped_index_lm", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("gen",)
)

if __name__ == "__main__":
    run(EXAMPLE)
