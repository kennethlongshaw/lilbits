"""Sample-efficient in-context adaptation via sleep: a transformer that refines its KV cache between windows.

After Lee et al., "Sample-Efficient In-Context Adaptation via Sleep" (2026, arXiv:2611.32802,
https://www.alphaxiv.org/abs/2611.sample-efficient-in-context-adaptation-sleep). Every W tokens the model "sleeps"
on the window it just read: N - 1 extra passes of the whole stack (same weights), each recomputing the window's
keys and values and merging them into the cache with a learned gate. Later tokens read the refined cache, so the
model should need fewer in-context tokens to adapt. Sleep needs no new weights beyond two small gates per layer,
keeps the KV cache the same size, and runs once per window instead of once per token (models/sleep_lm.py).

    models   N=1   the plain transformer (no sleep): the baseline
             N=2   one sleep loop per window       (paper: 2.10x fewer context tokens to the same loss)
             N=4   three sleep loops per window    (paper: 2.39x)
             same width / depth / seed; `--linear littlebit` makes the block linears sub-1-bit-capable
    data     tiny-shakespeare bytes, or `--data corpus` (lilbits.corpus: ~10 MB of Shakespeare, Gutenberg books
             and webtext, the last 5% of every document held out), interleaved (`--fewshot-frac`, default half) with
             few-shot map sequences ("cKaFcK...": a fresh random key -> value map per sequence, learnable only
             from earlier examples in the same context)
    train    each model, same steps and batches (smoke run by default)
    test     held-out bits per byte on a fixed last window given 0, W, 2W, ... tokens of earlier context
             (the paper's Figure 2), and the context efficiency of N > 1: N=1's full prefix / the prefix N needs to
             reach the same loss (the paper's 2.10x / 2.39x);
             few-shot map accuracy against the number of earlier examples of the key (0 shots = chance)
    infer    decode speed and per-sleep cost, KV-cache bytes (unchanged by N), sleep-gate parameter overhead

Smoke-trained, the curves only show the mechanics; the comparison needs `--steps` in the hundreds or more:

    uv run python examples/16_sleep_lm.py
    uv run python examples/16_sleep_lm.py --steps 1500 --threads 4
    uv run python examples/16_sleep_lm.py --data corpus --d 96 --batch 16 --bytes-per-param 20 --fewshot-frac 0.25
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from lilbits.corpus import CorpusBytes
from lilbits.data import FewShotMapSource
from lilbits.measure import Meter
from lilbits.models.sleep_lm import SleepConfig, SleepLM
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import ShakespeareBytes

MAX_SHOTS = 6


def text_source(args, split: str, seed: int):
    return (CorpusBytes if args.data == "corpus" else ShakespeareBytes)(split, seed=seed)


def add_args(ap):
    ap.add_argument(
        "--data",
        default="shakespeare",
        choices=["shakespeare", "corpus"],
        help="text: tiny-shakespeare (1.1 MB) or the ~10 MB corpus (lilbits.corpus: Shakespeare, Gutenberg, webtext)",
    )
    ap.add_argument("--steps", type=int, default=4, help="training steps per model (default: smoke run)")
    ap.add_argument(
        "--bytes-per-param", type=float, default=0, help="if > 0: train this many bytes per parameter (sets --steps)"
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--window", type=int, default=32, help="W: tokens per sleep window")
    ap.add_argument("--loops", default="1,2,4", help="N values to train side by side (1 = no sleep)")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--linear", default="fp", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--bpw", type=float, default=1.0)
    ap.add_argument("--sleep-causal", action="store_true", help="causal mask inside the window during sleep")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--keys", type=int, default=16, help="keys per few-shot map")
    ap.add_argument("--fewshot-frac", type=float, default=0.5, help="fraction of training steps on few-shot maps")
    ap.add_argument("--test-batches", type=int, default=8)
    ap.add_argument("--gen", type=int, default=128)


def loops_of(args) -> list[int]:
    return [int(n) for n in str(args.loops).split(",")]


def build(args, meta):
    out = {}
    for n in loops_of(args):
        torch.manual_seed(args.seed)
        cfg = SleepConfig(
            d_model=args.d,
            n_heads=args.heads,
            n_kv_heads=args.kv_heads,
            n_layers=args.layers,
            window=args.window,
            loops=n,
            sleep_causal=args.sleep_causal,
            linear=args.linear,
            bpw=args.bpw,
        )
        out[f"N={n}"] = SleepLM(cfg)
    return out


def train_steps(args, n_params: int) -> int:
    """--steps, or with --bytes-per-param R the steps that show the model R training bytes per parameter (a
    Chinchilla-like data / weights ratio: ~20 tokens per parameter at scale)."""
    if args.bytes_per_param > 0:
        return math.ceil(args.bytes_per_param * n_params / (args.batch * args.seq))
    return args.steps


def train_fn(models, args, fab):
    first = next(iter(models.values()))
    n_params = sum(p.numel() for p in first.parameters()) - first.sleep_params()  # without the sleep gates
    steps = train_steps(args, n_params)
    out: dict = {
        "data_ratio": {
            "params_without_gates": n_params,
            "train_bytes": steps * args.batch * args.seq,
            "bytes_per_param": steps * args.batch * args.seq / n_params,
        }
    }
    for name, model in models.items():
        text, fewshot = text_source(args, "train", args.seed), FewShotMapSource(args.keys, seed=args.seed)

        def batch(step, text=text, fewshot=fewshot):
            f = args.fewshot_frac  # deterministic interleave: a fraction f of the steps are few-shot map batches
            if int((step + 1) * f) > int(step * f):
                return fewshot.sample(args.batch, args.seq)[0]
            return text.sample(args.batch, args.seq)

        with Meter() as m:
            info = fit(fab, model, batch, lambda mod, x: mod.training_loss(x), steps, lr=args.lr, schedule="cosine")
        out[name] = {**info, **m.as_dict(), "step_ms": m.wall_s / max(1, steps) * 1e3}
    return out, {}


def prefix_curve(model, args) -> tuple[list[float], float]:
    """Held-out bits per byte on a fixed target (the last window of each sequence) given 0, W, 2W, ... tokens of
    preceding context, as in the paper's Figure 2 (same sequences for every model). Also the plain bits per byte
    over the full sequences."""
    src = text_source(args, "test", 123)
    S, W = args.seq, args.window
    prefixes = list(range(0, S - W + 1, W))
    tot, full = torch.zeros(len(prefixes)), 0.0
    for _ in range(args.test_batches):
        x = src.sample(args.batch, S)
        full += F.cross_entropy(model(x[:, :-1]).transpose(1, 2), x[:, 1:]).item()
        for j, p in enumerate(prefixes):
            logits = model(x[:, S - W - p : S])[:, -W:]
            tot[j] += F.cross_entropy(logits.transpose(1, 2), x[:, S - W + 1 :]).item()
    return (tot / args.test_batches / math.log(2)).tolist(), full / args.test_batches / math.log(2)


def prefix_to_reach(curve: list[float], target: float, window: int) -> float:
    """Context tokens (linear interpolation between measured prefixes) at which a prefix curve first reaches target."""
    for i, v in enumerate(curve):
        if v <= target:
            if i == 0:
                return 0.0
            a = curve[i - 1]
            return window * (i - 1) + (a - target) / max(a - v, 1e-9) * window
    return math.inf


def fewshot_accuracy(model, args) -> dict:
    src = FewShotMapSource(args.keys, seed=321)
    hits, counts = torch.zeros(MAX_SHOTS + 1), torch.zeros(MAX_SHOTS + 1)
    for _ in range(args.test_batches):
        x, shots = src.sample(args.batch, args.seq)
        pred = model(x[:, :-1]).argmax(-1)
        ok, s = pred == x[:, 1:], shots[:, 1:]
        for k in range(MAX_SHOTS + 1):
            sel = (s == k) if k < MAX_SHOTS else (s >= k)
            hits[k] += ok[sel].sum()
            counts[k] += sel.sum()
    acc = (hits / counts.clamp_min(1)).tolist()
    return {(f"{k} shots" if k < MAX_SHOTS else f"{k}+ shots"): a for k, a in enumerate(acc)}


def test_fn(models, args):
    res = {name: prefix_curve(m, args) for name, m in models.items()}
    curves = {name: c for name, (c, _) in res.items()}
    out: dict = {"bits_per_byte": {name: b for name, (_, b) in res.items()}}
    out["last_window_bpb_by_prefix"] = {"prefix_tokens": list(range(0, args.seq - args.window + 1, args.window))}
    out["last_window_bpb_by_prefix"].update(curves)
    if "N=1" in curves:
        target, full = curves["N=1"][-1], args.seq - args.window
        eff = {}
        for name, c in curves.items():
            if name != "N=1":
                t = prefix_to_reach(c, target, args.window)
                eff[name] = full / t if 0 < t < math.inf else (None if t == math.inf else f"> {full / args.window:g}")
        out["context_efficiency_vs_N=1"] = eff
        out["note_efficiency"] = (
            f"prefix N=1 needs for its full-context loss ({target:.3f} bpb, {full} tokens) / prefix N needs for the "
            "same loss; > 1 = sleep adapts faster, None = never reached"
        )
    out["fewshot_map_accuracy"] = {name: fewshot_accuracy(m, args) for name, m in models.items()}
    out["note"] = "chance at 0 shots = 1/26; smoke-trained runs show the mechanics only"
    return out


def infer_fn(models, args):
    prompt = text_source(args, "test", 7).sample(1, args.window * 2)[:, : args.window * 2]
    rows = {}
    for name, m in models.items():
        m.generate(prompt, 8)  # warm-up
        t, c = time.perf_counter(), time.process_time()
        _, st = m.generate(prompt, args.gen)
        w = time.perf_counter() - t
        # cost of one sleep alone: time a window's sleep on a fresh state filled with one window
        s = m.init_state(1)
        m.step(prompt[:, : args.window - 1], s)
        m._wake(prompt[:, args.window - 1 : args.window], s)
        t0 = time.perf_counter()
        if m.cfg.loops > 1:
            m._sleep(s)
        sleep_ms = (time.perf_counter() - t0) * 1e3
        base = sum(p.numel() for p in m.parameters()) - m.sleep_params()
        rows[name] = {
            "bytes_per_s": args.gen / w,
            "cpu_ms": (time.process_time() - c) * 1e3,
            "sleeps": st.sleeps,
            "ms_per_sleep": sleep_ms,
            "kv_cache_bytes": st.kv_bytes(),
            "kv_bytes_per_token": st.kv_bytes() / st.pos,
            "sleep_gate_params": m.sleep_params(),
            "param_overhead_pct": 100 * m.sleep_params() / base,
        }
    return {"decode": rows, "decoded_tokens": args.gen, "prompt_tokens": prompt.shape[1]}


EXAMPLE = Example(
    "16_sleep_lm", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("gen", "test_batches")
)

if __name__ == "__main__":
    run(EXAMPLE)
