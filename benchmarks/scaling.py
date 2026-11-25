"""Scaling harness: the one place to answer "does this trick scale?".

Every model family is swept over a fixed grid of deployed sizes (MiB). At each size the family's width is solved
so its deployed bytes hit the target, for each setting of its other knobs (depth, loop count, bpw for the bit
models), one knob varied at a time around the family's defaults. Every point runs in its own single-threaded
process and records the full set:

    size      params, deployed bytes (FP16 for FP parts, packed bits for LittleBit), on-disk checkpoint bytes
    train     a short run on Shakespeare bytes (--steps, cosine): median step ms, CPU util, peak RSS growth,
              then bits per byte on held-out text (the quality signal; smoke-scale by default, so only the
              ordering and the slope mean anything, not the absolute value)
    infer     deployable form (LUT-packed LittleBit where the family has it): RSS growth to build it, prefill
              bytes / s on 512 bytes, decode ms / byte streaming 128 bytes after a 128-byte prefill (median of 3)

Then per (family, knobs) curve it fits power laws y = a * MiB^b by least squares in log-log space (bpb, decode
ms / byte, train step ms, prefill), and finds the deployed size where each curve crosses the FP16 baseline curve
(fp16, depth 2), i.e. the size above or below which the trick wins. Output: results/scaling/<out>.{md,json,png}.

Families (add one by writing a `Family` adapter; nothing else changes):
    fp16      2-layer FP GDN byte LM (the baseline every crossover is measured against); knob: depth
    pocket    examples/10 PocketLM (byte patches -> looped LittleBit core with index); knobs: loops, core bpw
    looped    LoopedIndexLM on bytes directly (no patching); knobs: loops, core bpw
    oploop    examples/13 OpLoopLM (routed operators over context + registers, FP); knob: loops

    uv run python benchmarks/scaling.py                                  # smoke: 3 sizes, every family
    uv run python benchmarks/scaling.py --families fp16,pocket --sizes 1,2,4,8 --steps 200 --seeds 3
    uv run python benchmarks/scaling.py --match flops --sizes 5,10,20     # equal compute per byte
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
MiB = 2**20
MATCH = {
    "bytes": ("deployed MiB", "deployed_bytes", MiB),
    "params": ("M params", "params", 1e6),
    "flops": ("MFLOP / byte", "mflops_per_byte", 1.0),  # forward compute per byte of text (equal-compute view)
}
CACHE_KEY = {"bytes": "bytes", "params": "params", "flops": "flops"}
AXIS = "bytes"  # set by --match: what the size grid and every fit / crossover is measured in


def xval(r: dict) -> float:
    _, key, unit = MATCH[AXIS]
    return r[key] / unit


def xunit() -> str:
    return MATCH[AXIS][0]


# ---- families -----------------------------------------------------------------------------------------------


@dataclass
class Family:
    """How to build, train, size and run one model family. `build(width, **knobs)` returns a trainable model;
    `stage2(model)` (optional) converts it mid-run (e.g. FP core -> LittleBit); `deployable(model)` returns the
    inference form; `stream(model)` returns (prefill(x) -> state, step(state, x_t) -> state)."""

    name: str
    build: Any
    loss: Any
    stream: Any
    defaults: dict
    axes: dict  # knob -> values to try (one knob at a time, others at defaults)
    eval_ce: Any = None  # model, x -> next-byte CE only (nats); defaults to loss (when loss has no aux terms)
    stage2: Any = None
    deployable: Any = None
    width_step: int = 64
    max_width: int = 4096
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _heads(width: int) -> int:
    """Largest power of two <= width / 128 (at least 2): divides every width and the pocket's 64-wide byte layers."""
    return max(2, 1 << max(0, (width // 128).bit_length() - 1))


def _fp16_build(width, depth=2):
    from lilbits.models.stateful_lm import StatefulLM

    heads = max(h for h in range(1, max(1, width // 64) + 1) if width % h == 0)
    return StatefulLM(256, width, heads, depth, partial(nn.Linear, bias=False), mixer="gdn")


def _fp16_loss(m, x):
    return F.cross_entropy(m(x[:, :-1])[0].flatten(0, 1), x[:, 1:].flatten())


def _fp16_stream(m):
    def prefill(x):
        return m(x)[1]

    def step(s, xt):
        return m(xt, s)[1]

    return prefill, step


def _pocket_build(width, loops=3, bpw=0.55):
    from lilbits.models.pocket import PocketConfig, PocketLM

    return PocketLM(PocketConfig(d_body=width, heads=_heads(width), n_loops=loops, core_bpw=bpw))


def _pocket_ce(m, x):
    y, _ = m(x, m.new_cache(x.shape[0]))
    return F.cross_entropy(m.logits(y[:, :-1]).flatten(0, 1), x[:, 1:].flatten())


def _pocket_stage2(m):
    from lilbits.models.pocket import convert_pocket_core

    convert_pocket_core(m)


def _lut(m, sub=None):
    from lilbits.inference.lut import pack_lut

    pack_lut(sub(m) if sub else m)
    return m.eval()


def _pocket_stream(m):
    def prefill(x):
        c = m.new_cache(x.shape[0])
        m(x, c)
        return c

    def step(c, xt):
        m(xt, c)
        return c

    return prefill, step


def _looped_build(width, loops=3, bpw=0.55):
    from lilbits.models.looped_index import LoopedIndexConfig, LoopedIndexLM

    cfg = LoopedIndexConfig(
        vocab_size=256, d_model=width, n_heads=_heads(width), n_loops=loops, core_bpw=bpw,
        coda_linear="littlebit", scale_bits=8, block=16, top_k=8, n_mtp=0,
    )  # fmt: skip
    return LoopedIndexLM(cfg)


def _looped_stage2(m):
    from lilbits.models.looped_index import convert_core_to_littlebit

    convert_core_to_littlebit(m)


def _looped_stream(m):
    def prefill(x):
        c = m.new_cache(x.shape[0])
        m.logits(m.process(x, c)[:, -1:])
        return c

    def step(c, xt):
        m.logits(m.process(xt, c))
        return c

    return prefill, step


def _oploop_build(width, loops=4):
    from lilbits.models.op_loop import OpLoopLM

    return OpLoopLM(width, loops, heads=max(2, width // 64))


def _oploop_loss(m, x):
    return m.loss(x, torch.ones(x.shape[0], x.shape[1] - 1))[0]


def _oploop_stream(m):
    m.hard = True

    def prefill(x):
        m(x)
        return x

    def step(x, xt):  # no recurrent state: every byte re-reads the whole context (quadratic decode)
        x = torch.cat([x, xt], 1)
        m(x)
        return x

    return prefill, step


FAMILIES: dict[str, Family] = {
    "fp16": Family("fp16", _fp16_build, _fp16_loss, _fp16_stream, {"depth": 2}, {"depth": [2, 4]}, width_step=32),
    "pocket": Family(
        "pocket",
        _pocket_build,
        lambda m, x: m.training_loss(x)[0],
        _pocket_stream,
        {"loops": 3, "bpw": 0.55},
        {"loops": [1, 3], "bpw": [0.55, 1.0]},
        eval_ce=_pocket_ce,
        stage2=_pocket_stage2,
        deployable=partial(_lut, sub=lambda m: m.body),
    ),
    "looped": Family(
        "looped",
        _looped_build,
        lambda m, x: m.training_loss(x)[0],
        _looped_stream,
        {"loops": 3, "bpw": 0.55},
        {"loops": [1, 3], "bpw": [0.55, 1.0]},
        eval_ce=lambda m, x: torch.tensor(m.training_loss(x)[1]["lm"]),
        stage2=_looped_stage2,
        deployable=_lut,
    ),
    "oploop": Family(
        "oploop", _oploop_build, _oploop_loss, _oploop_stream, {"loops": 4}, {"loops": [2, 4]}, max_width=1024
    ),
}


def settings(fam: Family) -> list[dict]:
    """Defaults, then each axis value that differs from the default (one knob at a time)."""
    out = [dict(fam.defaults)]
    for k, vals in fam.axes.items():
        out += [{**fam.defaults, k: v} for v in vals if v != fam.defaults[k]]
    return out


def label(fam: str, knobs: dict) -> str:
    return fam + " " + " ".join(f"{k}={v}" for k, v in knobs.items())


# ---- sizing -------------------------------------------------------------------------------------------------


def forward_mflops_per_byte(fam: Family, m: nn.Module) -> float:
    """Forward FLOPs per byte on real text (256 held-out bytes), counted by torch's FLOP counter. Data-dependent
    paths (byte patching) run at the rate real text gives."""
    from torch.utils.flop_counter import FlopCounterMode

    from lilbits.shakespeare import ShakespeareBytes

    x = ShakespeareBytes("test", seed=7).sample(1, 256)
    was = m.training
    m.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        (fam.eval_ce or fam.loss)(m, x)
    m.train(was)
    return fc.get_total_flops() / x.shape[1] / 1e6


def model_size(fam: Family, width: int, knobs: dict) -> tuple[int, int, float]:
    """(deployed bytes, params, forward MFLOP per byte) of the deployable model at this width."""
    from lilbits.measure import size_report

    torch.manual_seed(0)
    m = fam.build(width, **knobs)
    if fam.stage2:
        fam.stage2(m)
    return (
        size_report(m)["deployed_total_bytes"],
        sum(p.numel() for p in m.parameters()),
        forward_mflops_per_byte(fam, m),
    )


def solve_width(fam: Family, knobs: dict, target_mib: float, cache: dict) -> int | None:
    """Width (multiple of width_step) whose size (deployed MiB or M params, per --match) is closest to the
    target. Both are quadratic in width at fixed knobs, so three small real builds fit them exactly enough; None
    if out of range."""
    key = label(fam.name, knobs)
    if "flops" not in cache.get(key, {}):
        ws = [2 * fam.width_step, 4 * fam.width_step, 6 * fam.width_step]
        sizes = [model_size(fam, w, knobs) for w in ws]
        cache[key] = {"w": ws, **{k: [z[i] for z in sizes] for i, k in enumerate(("bytes", "params", "flops"))}}
    ws, ys = cache[key]["w"], cache[key][CACHE_KEY[AXIS]]
    target = target_mib * MATCH[AXIS][2]

    # quadratic through 3 points (Lagrange)
    def f(w):
        return sum(y * math.prod((w - wj) / (wi - wj) for wj in ws if wj != wi) for wi, y in zip(ws, ys, strict=True))

    cands = range(fam.width_step, fam.max_width + 1, fam.width_step)
    best = min(cands, key=lambda w: abs(math.log(max(f(w), 1)) - math.log(target)))
    if abs(math.log(max(f(best), 1) / target)) > math.log(1.35):
        return None  # cannot get within 35% of the target (too small for the family's fixed parts, or too big)
    return best


# ---- one point (own process) --------------------------------------------------------------------------------


def _median(fn, reps=3):
    fn()
    walls, cpus = [], []
    for _ in range(reps):
        w, c = time.perf_counter(), time.process_time()
        fn()
        walls.append(time.perf_counter() - w)
        cpus.append(time.process_time() - c)
    return statistics.median(walls), statistics.median(cpus)


def worker(spec: dict) -> dict:
    from lilbits.measure import Meter, size_report, trim
    from lilbits.shakespeare import ShakespeareBytes

    torch.set_num_threads(1)
    if spec.get("cpu") is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {spec["cpu"]})  # pin to one core: timings don't hop between busy and idle cores
    fam, knobs, width = FAMILIES[spec["family"]], spec["knobs"], spec["width"]
    steps, batch, seq, seed = spec["steps"], spec["batch"], spec["seq"], spec.get("seed", 0)
    torch.manual_seed(seed)
    model = fam.build(width, **knobs)
    train = ShakespeareBytes("train", seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=0.0)
    warm = max(1, steps // 20)

    def lr_at(i):
        if i < warm:
            return (i + 1) / warm
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm)))

    step_ms, losses = [], []
    model.train()
    with Meter() as mt:
        for i in range(steps):
            if fam.stage2 and i == steps // 2:  # two-stage families: FP first half, converted second half
                fam.stage2(model)
                opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=0.0)
            for g in opt.param_groups:
                g["lr"] = spec["lr"] * lr_at(i)
            x = train.sample(batch, seq)
            t0 = time.perf_counter()
            loss = fam.loss(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            losses.append(loss.item())
    if fam.stage2 and steps < 2:
        fam.stage2(model)
    out: dict = {
        "params": sum(p.numel() for p in model.parameters()),
        "deployed_bytes": size_report(model)["deployed_total_bytes"],
        "train_step_ms": statistics.median(step_ms[1:] or step_ms),
        "train_cpu_util": mt.cpu_util,
        "train_peak_rss_mib": mt.peak_extra_mib,
        "train_wall_s": mt.wall_s,
        "train_last_loss": statistics.mean(losses[-5:]),  # includes a family's aux terms (pocket: MTP, ratio)
    }
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    out["disk_bytes"] = buf.tell()
    out["mflops_per_byte"] = forward_mflops_per_byte(fam, model)
    model.eval()
    test = ShakespeareBytes("test", seed=1).sample(8, 256)
    with torch.no_grad():
        out["eval_loss"] = (fam.eval_ce or fam.loss)(model, test).item()
    out["bpb"] = out["eval_loss"] / math.log(2)
    span = max(1, len(losses) // 10)
    out["loss_curve"] = [statistics.mean(losses[i : i + span]) for i in range(0, len(losses), span)]
    trim()
    rss0 = Meter()
    with rss0:
        m = fam.deployable(model) if fam.deployable else model
        gc.collect()
    out["infer_build_rss_mib"] = rss0.peak_extra_mib
    data = ShakespeareBytes("test", seed=2).sample(1, 640)
    prompt, stream, long = data[:, :128], data[:, 128:256], data[:, :512]
    prefill, step = fam.stream(m)

    def run_stream():
        s = prefill(prompt)
        for t in range(stream.shape[1]):
            s = step(s, stream[:, t : t + 1])

    with torch.no_grad():
        with Meter() as mi:
            dec_w, dec_c = _median(run_stream)
            pre_w, pre_c = _median(lambda: prefill(prompt))
            long_w, _ = _median(lambda: prefill(long))
    out["decode_ms_per_byte"] = max(0.0, dec_w - pre_w) / stream.shape[1] * 1e3
    out["decode_cpu_ms_per_byte"] = max(0.0, dec_c - pre_c) / stream.shape[1] * 1e3
    out["prefill_bytes_per_s"] = long.shape[1] / long_w
    out["infer_peak_rss_mib"] = mi.peak_extra_mib
    return out


QUALITY = ("bpb", "eval_loss", "train_last_loss")


def merge_seeds(runs: list[dict]) -> dict:
    """One row from per-seed runs: quality metrics averaged (plus *_std), everything else the median."""
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return runs[0]
    out: dict = {"seeds": len(ok)}
    for k, v in ok[0].items():
        vals = [r[k] for r in ok]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            out[k] = v
        elif k in QUALITY:
            out[k] = statistics.mean(vals)
            out[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        else:
            out[k] = statistics.median(vals)
    return out


def load_1m() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return 0.0


def wait_for_quiet(max_load: float, timeout_s: float = 600) -> float:
    """Wait until other work on the machine is below max_load (1-min load average, minus this sweep's own process)
    or timeout_s passes; returns the load seen at start of the run."""
    t0 = time.time()
    while load_1m() - 1 > max_load and time.time() - t0 < timeout_s:
        time.sleep(15)
    return load_1m()


def _run(spec: dict, env: dict) -> dict:
    p = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(spec)], capture_output=True, text=True, env=env, cwd=ROOT
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        return {"error": p.stderr[-1500:]}
    return json.loads(line[0][7:])


# ---- fits, crossovers, report -------------------------------------------------------------------------------

METRICS = {  # key -> (label, lower is better)
    "bpb": ("bits / byte (short run)", True),
    "decode_ms_per_byte": ("decode ms / byte", True),
    "train_step_ms": ("train step ms", True),
    "prefill_bytes_per_s": ("prefill bytes / s", False),
    "infer_peak_rss_mib": ("inference peak RSS growth MiB", True),
}


def power_fit(xs: list[float], ys: list[float]) -> dict | None:
    """y = a * x^b by least squares on logs. Returns a, b, r2 (None with < 2 points or non-positive values)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    mx, my = statistics.mean(p[0] for p in pts), statistics.mean(p[1] for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = math.exp(my - b * mx)
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (math.log(a) + b * p[0])) ** 2 for p in pts)
    return {"a": a, "b": b, "r2": 1 - ss_res / ss_tot if ss_tot else 1.0}


def crossover(f: dict, base: dict, better_low: bool, xs: list[float], base_xs: list[float]) -> dict:
    """Compare curve f with the baseline over the sizes both were measured at. Returns the ratio f / baseline
    (baseline read off its fit) at each of f's sizes, a verdict, and the crossover size if the fits meet inside
    the measured range (extended 2x each way). Fits only extrapolate badly, so outside that range there is no
    crossover, just better / worse everywhere measured."""
    ratios = [f["a"] * x ** f["b"] / (base["a"] * x ** base["b"]) for x in xs]
    good = [(r < 1) == better_low for r in ratios]
    lo, hi = max(min(xs), min(base_xs)) / 2, min(max(xs), max(base_xs)) * 2
    out: dict = {"ratio_min": min(ratios), "ratio_max": max(ratios), "mib": None}
    if abs(f["b"] - base["b"]) > 1e-9:
        x = math.exp(math.log(base["a"] / f["a"]) / (f["b"] - base["b"]))
        if lo <= x <= hi:
            above = (f["b"] < base["b"]) == better_low  # f's exponent lower -> f below base for x > crossover
            out |= {"mib": x, "verdict": "better above" if above else "better below"}
            return out
    out["verdict"] = "better everywhere measured" if all(good) else "worse everywhere measured"
    if any(good) and not all(good):
        out["verdict"] = "mixed"
    return out


def analyze(rows: list[dict]) -> dict:
    curves: dict[str, list[dict]] = {}
    for r in rows:
        if "error" not in r:
            curves.setdefault(r["curve"], []).append(r)
    fits: dict = {}
    sizes = {n: [xval(p) for p in pts] for n, pts in curves.items()}
    for name, pts in curves.items():
        fits[name] = {k: power_fit(sizes[name], [p[k] for p in pts]) for k in METRICS}
        fits[name]["points"] = len(pts)
    base = next((n for n in fits if n.startswith("fp16 depth=2")), None)
    cross: dict = {}
    if base:
        for name, f in fits.items():
            if name == base:
                continue
            cross[name] = {}
            for k, (_, low) in METRICS.items():
                if f[k] and fits[base][k]:
                    cross[name][k] = crossover(f[k], fits[base][k], low, sizes[name], sizes[base])
    return {"baseline": base, "fits": fits, "crossovers": cross}


def report(rows: list[dict], an: dict, args) -> str:
    lines = [
        "# Scaling sweep",
        "",
        f"Matched on {xunit()}: {args.sizes}. Seeds per point: {args.seeds}; workers pinned to core {args.cpu}. "
        f"Short runs: {args.steps} steps x {args.batch} x {args.seq} bytes of "
        "Shakespeare, cosine; single-threaded CPU, one process per point. Smoke-scale: read slopes and orderings, "
        "not absolute bits per byte.",
        "",
        "## Points",
        "",
        "| curve | target | width | params | deployed MiB | disk MiB | train loss | eval loss | bpb | train step ms | "
        "train wall s | train CPU util | "
        "train peak RSS MiB | decode ms/B | decode CPU ms/B | prefill B/s | infer peak RSS MiB |",
        "|---" * 17 + "|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['curve']} | {r['target_mib']} | {r.get('width')} | error: see json |" + " |" * 14)
            continue
        lines.append(
            f"| {r['curve']} | {r['target_mib']} | {r['width']} | {r['params'] / 1e6:.2f}M | "
            f"{r['deployed_bytes'] / MiB:.2f} | {r['disk_bytes'] / MiB:.2f} | {r['train_last_loss']:.3f} | "
            f"{r['eval_loss']:.3f} | {r['bpb']:.3f}{f' ± {r['bpb_std']:.3f}' if r.get('seeds', 1) > 1 else ''} | {r['train_step_ms']:.0f} | {r['train_wall_s']:.1f} | "
            f"{r['train_cpu_util']:.2f} | {r['train_peak_rss_mib']:.0f} | "
            f"{r['decode_ms_per_byte']:.2f} | {r['decode_cpu_ms_per_byte']:.2f} | {r['prefill_bytes_per_s']:.0f} | "
            f"{r['infer_peak_rss_mib']:.0f} |"
        )
    noisy = [f"{r['curve']} @ {r['target_mib']}" for r in rows if r.get("load_1m", 0) - 1 > args.max_load]
    if noisy:
        lines += ["", f"Timings taken on a busy machine (load above {args.max_load} + 1): " + ", ".join(noisy)]
    if args.skipped:
        lines += ["", "Out of range (no width within 35% of the target): " + ", ".join(args.skipped)]
    lines += ["", f"## Fits (y = a * size^b, size in {xunit()})", ""]
    lines += [
        "| curve | points | " + " | ".join(m[0] for m in METRICS.values()) + " |",
        "|---" * (len(METRICS) + 2) + "|",
    ]
    for name, f in an["fits"].items():
        cells = [f"b={v['b']:+.2f} (r2 {v['r2']:.2f})" if v else "-" for v in (f[k] for k in METRICS)]
        lines.append(f"| {name} | {f['points']} | " + " | ".join(cells) + " |")
    lines += ["", f"## Crossovers against `{an['baseline']}`", ""]
    lines.append(
        "Each cell: the curve / baseline ratio over the curve's measured sizes (baseline read off its fit), then the "
        "crossover size if the two fits meet within the measured range (2x margin), else whether it is better or "
        "worse at every measured size. For prefill, higher is better; for the rest, lower. Fits from 2 points "
        "always have r2 = 1; trust slopes from 3+ points and r2 near 1."
    )
    lines += ["", "| curve | " + " | ".join(m[0] for m in METRICS.values()) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for name, c in an["crossovers"].items():
        cells = []
        for k in METRICS:
            v = c.get(k)
            if not v:
                cells.append("-")
            else:
                r = f"x{v['ratio_min']:.2f}" + (
                    f"-{v['ratio_max']:.2f}" if v["ratio_max"] / v["ratio_min"] > 1.01 else ""
                )
                where = f"crosses at {v['mib']:.2g} {xunit()}, {v['verdict']}" if v["mib"] else v["verdict"]
                cells.append(f"{r}; {where}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(rows: list[dict], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    keys = list(METRICS)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6))
    curves: dict[str, list[dict]] = {}
    for r in rows:
        if "error" not in r:
            curves.setdefault(r["curve"], []).append(r)
    for ax, k in zip(axes, keys, strict=True):
        for name, pts in curves.items():
            pts = sorted(pts, key=xval)
            style = "-" if name.endswith(("depth=2", "loops=3 bpw=0.55", "loops=4")) else "--"
            ax.plot([xval(p) for p in pts], [p[k] for p in pts], style, marker="o", label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xunit())
        ax.set_title(METRICS[k][0], fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--sizes", default="0.5,1,2", help="size grid: deployed MiB (--match bytes) or M params")
    ap.add_argument("--match", default="bytes", choices=list(MATCH), help="match points on deployed bytes or params")
    ap.add_argument("--steps", type=int, default=20, help="short-run training steps per point")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=1, help="runs per point: losses averaged (with std), timings median")
    ap.add_argument("--cpu", type=int, default=(os.cpu_count() or 1) - 1, help="core every worker is pinned to")
    ap.add_argument(
        "--max-load",
        type=float,
        default=1.0,
        help="before each run, wait (up to 10 min) until the 1-min load average minus 1 (this sweep) is below it; "
        "the load is recorded per point and the report flags noisy ones",
    )
    ap.add_argument("--defaults-only", action="store_true", help="skip the one-knob-at-a-time variations")
    ap.add_argument("--out", default="results/scaling/scaling")
    ap.add_argument("--report-only", action="store_true", help="re-fit and re-render <out>.json without running")
    args = ap.parse_args(argv)
    global AXIS
    AXIS = args.match
    if args.worker:
        print("RESULT " + json.dumps(worker(json.loads(args.worker))))
        return
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env["OMP_NUM_THREADS"] = "1"
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out.parent / "size_fits.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    sizes = [float(s) for s in args.sizes.split(",")]
    torch.set_num_threads(1)
    rows, args.skipped = [], []
    if args.report_only:
        saved = json.loads(out.with_suffix(".json").read_text())
        rows, args.skipped = saved["rows"], saved.get("skipped", [])
    for fname in [] if args.report_only else args.families.split(","):
        fam = FAMILIES[fname]
        for knobs in [fam.defaults] if args.defaults_only else settings(fam):
            curve = label(fname, knobs)
            for target in sizes:
                width = solve_width(fam, knobs, target, cache)
                cache_path.write_text(json.dumps(cache, indent=1))
                if width is None:
                    args.skipped.append(f"{curve} @ {target} {xunit()}")
                    continue
                spec = {"family": fname, "knobs": knobs, "width": width, "steps": args.steps}
                spec |= {"batch": args.batch, "seq": args.seq, "lr": args.lr, "cpu": args.cpu}
                t0 = time.time()
                runs = []
                for seed in range(args.seeds):
                    load = wait_for_quiet(args.max_load)
                    runs.append({**_run({**spec, "seed": seed}, env), "load_1m": load})
                r = {"curve": curve, "target_mib": target, "width": width, **merge_seeds(runs)}
                rows.append(r)
                msg = str(r.get("error", "")).splitlines()[-1:]
                print(
                    f"{curve:28s} {target:5g} {xunit()}  d={width:<5d} "
                    + (f"bpb {r['bpb']:.3f}  decode {r['decode_ms_per_byte']:.2f} ms/B" if not msg else str(msg))
                    + f"  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    an = analyze(rows)
    args.sizes = sizes
    md = report(rows, an, args)
    if plot(rows, out.with_suffix(".png")):
        md += f"\n![scaling]({out.with_suffix('.png').name})\n"
    out.with_suffix(".md").write_text(md)
    out.with_suffix(".json").write_text(json.dumps({"rows": rows, "skipped": args.skipped, **an}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
