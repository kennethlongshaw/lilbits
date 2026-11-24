"""Inference-speed benchmark: nn.Linear vs LittleBitLinear (cached factors), single layers and TinyLM.

    uv run lb-infer-bench
    uv run lb-infer-bench --device cuda --dims 4096x4096,4096x11008
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from .littlebit import LittleBitLinear
from .model import Config, TinyLM


def _time(fn, device: str, reps: int) -> float:
    with torch.no_grad():
        for _ in range(3):
            fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def layer_rows(dims, bpws, tokens_list, device, reps):
    rows = []
    for d_in, d_out in dims:
        fp = nn.Linear(d_in, d_out, bias=False).to(device).eval()
        lbs = {b: LittleBitLinear(d_in, d_out, bpw=b, residual=True).to(device).eval() for b in bpws}
        for tokens in tokens_list:
            x = torch.randn(tokens, d_in, device=device)
            t_fp = _time(lambda: fp(x), device, reps)
            for b, lb in lbs.items():
                t_lb = _time(lambda: lb(x), device, reps)
                kind = lb._infer_cache[0]
                rows.append(dict(shape=f"{d_in}x{d_out}", tokens=tokens, bpw=lb.bpw, rank=lb.rank, kind=kind,
                                 fp_ms=t_fp, lb_ms=t_lb, speedup=t_fp / t_lb,
                                 fp_macs=d_in * d_out, lb_macs=(2 * lb.rank * (d_in + d_out) if kind == "lowrank"
                                                                else d_in * d_out)))
    return rows


def model_rows(bpws, device, reps):
    rows = []
    shapes = [("prefill 32x64", 32, 64), ("decode-like 1x1", 1, 1)]
    models = {"FP": TinyLM(Config(vocab_size=32, seq_len=64, linear="fp"))}
    for b in bpws:
        models[f"LittleBit {b} bpw"] = TinyLM(Config(vocab_size=32, seq_len=64, linear="littlebit", bpw=b))
    for name, m in models.items():
        m.to(device)
        # train-mode forward under no_grad = the old path that re-binarizes every call
        modes = [("", False)] if name == "FP" else [(" (old: rebuild factors each call)", True), ("", False)]
        for suffix, training in modes:
            m.train(training)
            for sname, B, T in shapes:
                x = torch.randint(0, 32, (B, T), device=device)
                rows.append(dict(model=name + suffix, input=sname, ms=_time(lambda: m(x), device, reps)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="1024x1024,4096x4096,4096x11008")
    ap.add_argument("--bpws", default="1.0,0.55,0.1")
    ap.add_argument("--tokens", default="1,2048")
    ap.add_argument("--model-bpws", default="1.41,1.0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    print(f"## TinyLM forward (d_model 128, 2 blocks), {args.device}, {args.threads} threads\n")
    print("| model | input | ms |\n|---|---|---|")
    for r in model_rows([float(b) for b in args.model_bpws.split(",")], args.device, args.reps):
        print(f"| {r['model']} | {r['input']} | {r['ms']:.2f} |")

    dims = [tuple(int(v) for v in d.split("x")) for d in args.dims.split(",")]
    print(f"\n## Single layer, {args.device}, {args.threads} threads\n")
    print("| in x out | tokens | LittleBit bpw (rank/path) | path | MACs vs dense | nn.Linear ms | LittleBit ms | speedup |")
    print("|---|---|---|---|---|---|---|---|")
    for r in layer_rows(dims, [float(b) for b in args.bpws.split(",")],
                        [int(t) for t in args.tokens.split(",")], args.device, args.reps):
        print(f"| {r['shape']} | {r['tokens']} | {r['bpw']:.2f} ({r['rank']}) | {r['kind']} | "
              f"{r['lb_macs'] / r['fp_macs']:.2f}x | {r['fp_ms']:.3f} | {r['lb_ms']:.3f} | {r['speedup']:.2f}x |")


if __name__ == "__main__":
    main()
