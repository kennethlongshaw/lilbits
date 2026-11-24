"""Inference memory + speed: FP baseline vs LittleBit (training-format with cache) vs bit-packed export.

Each case runs in its own process. Linux only (uses /proc to reset/read peak RSS).

    uv run lb-mem-bench
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import io
import json
import subprocess
import sys
import time

import torch
from torch import nn

from .littlebit import LittleBitLinear
from .model import Config, TinyLM
from .packed import model_bytes, pack_model


def _trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _status(key: str) -> float:
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1]) / 1024  # MiB
    return 0.0


def _reset_peak():
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def _time(fn, reps):
    with torch.no_grad():
        for _ in range(3):
            fn()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
    return (time.perf_counter() - t) / reps * 1e3


def build(case: str):
    kind, _, arg = case.partition(":")
    if kind.startswith("model"):
        variant = kind.split("-")[1]
        lin = "fp" if variant == "fp" else "littlebit"
        m = TinyLM(Config(vocab_size=32, seq_len=64, d_model=128, n_layers=2, linear=lin, bpw=float(arg or 1.41)))
        inputs = {"batch 32x64": torch.randint(0, 32, (32, 64)), "1 token": torch.randint(0, 32, (1, 1))}
    else:
        d_in, d_out = (int(v) for v in kind.split("-")[1].split("x"))
        variant = kind.split("-")[2]
        m = (nn.Linear(d_in, d_out, bias=False) if variant == "fp"
             else LittleBitLinear(d_in, d_out, bpw=float(arg), residual=True))
        inputs = {"1 token": torch.randn(1, d_in), "2048 tokens": torch.randn(2048, d_in)}
    m.eval()
    if variant == "packed":
        pack_model(m) if not isinstance(m, LittleBitLinear) else None
        if isinstance(m, LittleBitLinear):
            from .packed import PackedLittleBitLinear
            m = PackedLittleBitLinear(m)
    elif variant == "cached":
        with torch.no_grad():
            m(next(iter(inputs.values())))  # build the FP32 inference cache
    return m, inputs


def cache_bytes(m) -> int:
    return sum(sum(t.numel() * t.element_size() for t in mm._infer_cache[1:])
               for mm in m.modules() if isinstance(mm, LittleBitLinear) and mm._infer_cache is not None)


def run_case(case: str, reps: int) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    m, inputs = build(case)
    if case.split(":")[0].endswith("cached"):
        with torch.no_grad():
            m(next(iter(inputs.values())))
    # weights in RAM are counted exactly from tensor sizes (RSS can't isolate them reliably)
    weights = model_bytes(m) + cache_bytes(m)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    out = {"case": case, "weights_mib": weights / 2**20, "disk_mib": buf.tell() / 2**20}
    for name, x in inputs.items():
        _trim()
        _reset_peak()
        base = _status("VmRSS")
        ms = _time(lambda: m(x), reps)
        out[f"{name} ms"] = ms
        out[f"{name} peak_extra_mib"] = _status("VmHWM") - base
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--cases", default="model-fp,model-cached:1.41,model-packed:1.41,"
                    "layer-4096x4096-fp,layer-4096x4096-cached:1.0,layer-4096x4096-packed:1.0,"
                    "layer-4096x4096-packed:0.55,layer-4096x4096-packed:0.1")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.one:
        print(json.dumps(run_case(args.one, args.reps)))
        return
    rows = []
    for c in args.cases.split(","):
        o = subprocess.run([sys.executable, "-m", "littlebit_scratch.mem_bench", "--one", c, "--reps", str(args.reps)],
                           stdout=subprocess.PIPE, text=True, check=True).stdout
        rows.append(json.loads(o.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]), flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
