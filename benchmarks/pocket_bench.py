"""Pocket LM vs an FP16 byte LM of the same deployed size: cost per byte across widths, and per-piece ablations.

No training: every number here is a cost (time, memory, bytes), which does not depend on the weights. Each row runs
in its own process so RSS is clean. Decode cost is measured by streaming real text (this repo's files) one byte
at a time after a prefill, so patching runs at the rate real text gives, not at whatever an untrained model emits.

    inference RSS   process RSS growth for building the deployable form (LUT-packed LittleBit layers for the
                    pocket model; the FP16 baseline's weights are held as FP32 on CPU, as it runs here)
    decode          per-byte wall / CPU time streaming 256 bytes after a 256-byte prefill (median of 3)
    prefill         bytes per second for one 1024-byte chunk
    train step      forward + backward of 4 x 256 bytes, and peak RSS growth during it

    uv run python benchmarks/pocket_bench.py
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

ROWS = [  # name, pocket kwargs / None for the baseline matched to the pocket kwargs given
    ("pocket d512 (all on, LUT)", dict(d_body=512)),
    ("pocket d1024 (all on, LUT)", dict(d_body=1024)),
    ("pocket d2048 (all on, LUT)", dict(d_body=2048)),
    ("  ablation d1024: native kernel", dict(d_body=1024, kernel="native")),
    ("  ablation d1024: no patching (router none)", dict(d_body=1024, router="none")),
    ("  ablation d1024: no index", dict(d_body=1024, index=False)),
    ("  ablation d1024: FP core (no LittleBit core)", dict(d_body=1024, core="fp")),
    ("  ablation d1024: 1 loop", dict(d_body=1024, n_loops=1)),
    ("  ablation d1024: FP coda", dict(d_body=1024, coda_linear="fp")),
]


def corpus() -> torch.Tensor:
    paths = sorted(
        glob.glob(str(ROOT / "docs" / "*.md")) + glob.glob(str(ROOT / "src" / "**" / "*.py"), recursive=True)
    )
    data = b"".join(open(p, "rb").read() for p in paths)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def rss_mib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def _median_ms(fn, reps: int = 3) -> tuple[float, float]:
    """Median wall and CPU seconds of fn() over reps runs (after one warm-up)."""
    fn()
    walls, cpus = [], []
    for _ in range(reps):
        w, c = time.perf_counter(), time.process_time()
        fn()
        walls.append(time.perf_counter() - w)
        cpus.append(time.process_time() - c)
    return statistics.median(walls), statistics.median(cpus)


def worker(spec: dict) -> dict:
    from lilbits.inference.lut import pack_lut
    from lilbits.measure import Meter, size_report, trim
    from lilbits.models.pocket import PocketConfig, PocketLM, convert_pocket_core, fp16_baseline, matched_fp16_width

    torch.set_num_threads(1)
    torch.manual_seed(0)
    data = corpus()
    prompt, stream, long = data[None, 5000:5256], data[None, 5256:5512], data[None, 20000:21024]
    x = data[: 4 * 257].view(4, 257)
    kw = dict(spec["pocket"])
    kernel, core, index = kw.pop("kernel", "lut"), kw.pop("core", "littlebit"), kw.pop("index", True)
    cfg = PocketConfig(heads=max(2, kw["d_body"] // 128), **kw)
    out: dict = {}

    def make_pocket() -> PocketLM:
        m = PocketLM(cfg)
        m.body.use_index = index
        if core == "littlebit":
            convert_pocket_core(m)
        return m

    gc.collect()
    trim()
    base_rss = rss_mib()
    if spec["baseline"]:
        d = spec["width"]
        base = fp16_baseline(d).eval()
        out["baseline_width"] = d
        out["deployed_kib"] = size_report(base)["deployed_total_bytes"] / 1024
        out["params"] = sum(p.numel() for p in base.parameters())
        gc.collect()
        trim()
        out["infer_rss_mib"] = rss_mib() - base_rss

        def stream_base():
            _, s = base(prompt)
            for t in range(stream.shape[1]):
                _, s = base(stream[:, t : t + 1], s)

        with torch.no_grad():
            dec_w, dec_c = _median_ms(stream_base)
            pre_w, pre_c = _median_ms(lambda: base(prompt))
            long_w, _ = _median_ms(lambda: base(long))
        del base

        def train_loss() -> torch.Tensor:
            return F.cross_entropy(tm(x[:, :-1])[0].flatten(0, 1), x[:, 1:].flatten())

        tm: torch.nn.Module = fp16_baseline(d)
    else:
        pocket = make_pocket().eval()
        out["deployed_kib"] = size_report(pocket)["deployed_total_bytes"] / 1024
        out["params"] = sum(p.numel() for p in pocket.parameters())
        out["matched_fp16_width"] = matched_fp16_width(out["deployed_kib"] * 1024)
        if kernel == "lut" and core == "littlebit":
            pack_lut(pocket.body)
        gc.collect()
        trim()
        out["infer_rss_mib"] = rss_mib() - base_rss
        info: dict = {}

        def stream_pocket():
            c = pocket.new_cache(1)
            pocket(prompt, c)
            p0 = c.patches
            for t in range(stream.shape[1]):
                pocket(stream[:, t : t + 1], c)
            info["bytes_per_body_pass"] = stream.shape[1] / max(1, c.patches - p0)
            info["state_kib"] = sum(c.state_bytes().values()) / 1024

        with torch.no_grad():
            dec_w, dec_c = _median_ms(stream_pocket)
            pre_w, pre_c = _median_ms(lambda: pocket(prompt, pocket.new_cache(1)))
            long_w, _ = _median_ms(lambda: pocket(long, pocket.new_cache(1)))
        out.update(info)
        del pocket
        tp = make_pocket()

        def train_loss() -> torch.Tensor:
            return tp.training_loss(x)[0]

        tm = tp
    # decode = streaming time minus the prompt's prefill
    out["decode_ms_per_byte"] = (dec_w - pre_w) / stream.shape[1] * 1e3
    out["decode_cpu_ms_per_byte"] = (dec_c - pre_c) / stream.shape[1] * 1e3
    out["prefill_bytes_per_s"] = long.shape[1] / long_w
    gc.collect()
    trim()
    tm.train()
    train_loss().backward()
    tm.zero_grad(set_to_none=True)
    with Meter() as mt:
        train_loss().backward()
    out["train_step_ms"] = mt.wall_s * 1e3
    out["train_peak_rss_growth_mib"] = mt.peak_extra_mib
    return out


def _run(spec: dict, env: dict) -> dict:
    p = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(spec)], capture_output=True, text=True, env=env, cwd=ROOT
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        print(p.stderr[-3000:])
        raise SystemExit(f"{spec} failed")
    return json.loads(line[0][7:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--only", help="rows whose name contains this")
    ap.add_argument("--out", default="results/pocket_bench")
    args = ap.parse_args(argv)
    if args.worker:
        print("RESULT " + json.dumps(worker(json.loads(args.worker))))
        return
    rows: list[dict] = []
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for name, kw in ROWS:
        if args.only and args.only not in name:
            continue
        r = {"row": name, **_run({"pocket": kw, "baseline": False}, env)}
        rows.append(r)
        print(r, flush=True)
        if not name.startswith("  "):  # ablations compare against their full row's baseline
            b = {
                "row": f"FP16 same size as {name}",
                **_run({"pocket": kw, "baseline": True, "width": r["matched_fp16_width"]}, env),
            }
            rows.append(b)
            print(b, flush=True)
    md = [
        "| model | params | deployed | inference RSS | decode ms/byte (CPU) | bytes/body pass | prefill B/s "
        "| train step | train peak RSS | state |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [
            r["row"],
            f"{r['params'] / 1e6:.1f} M",
            f"{r['deployed_kib'] / 1024:.1f} MiB",
            f"{r['infer_rss_mib']:.0f} MiB",
            f"{r['decode_ms_per_byte']:.2f} ({r['decode_cpu_ms_per_byte']:.2f})",
            f"{r.get('bytes_per_body_pass', 1):.2f}",
            f"{r['prefill_bytes_per_s']:.0f}",
            f"{r['train_step_ms']:.0f} ms",
            f"{r['train_peak_rss_growth_mib']:.0f} MiB",
            f"{r['state_kib']:.0f} KiB" if "state_kib" in r else "",
        ]
        md.append("| " + " | ".join(cells) + " |")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
