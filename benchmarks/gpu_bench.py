"""Packed LittleBit inference on GPU: FP16 nn.Linear vs packed (torch backend) vs packed (Helion kernels).

    uv sync --extra gpu
    uv run python benchmarks/gpu_bench.py                                    # first run autotunes (~a few min per shape)
    LB_HELION_AUTOTUNE=1 uv run python benchmarks/gpu_bench.py               # Helion's full autotuner instead of the candidate list

CPU smoke test (tiny shapes, correctness only):
    HELION_INTERPRET=1 uv run python benchmarks/gpu_bench.py --device cpu --dims 64x96 --tokens 1,5 --bpws 2.0
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from lilbits.layers.littlebit import LittleBitLinear
from lilbits.inference.packed import PackedLittleBitLinear


def _time(fn, device, reps):
    with torch.no_grad():
        for _ in range(3):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="4096x4096,4096x11008,11008x4096")
    ap.add_argument("--bpws", default="1.0,0.55,0.1")
    ap.add_argument("--tokens", default="1,16,2048")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev, dt = args.device, getattr(torch, args.dtype)
    if dev == "cpu" and dt == torch.float16:
        dt = torch.float32
    torch.manual_seed(0)
    print(f"device {dev} ({torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'}), dtype {dt}\n")
    print("| in x out | tokens | bpw (rank/path) | dense weights | packed weights | dense ms | packed torch ms | "
          "packed helion ms | helion vs dense | max abs err helion vs torch |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in args.dims.split(","):
        d_in, d_out = (int(v) for v in d.split("x"))
        dense = nn.Linear(d_in, d_out, bias=False, device=dev, dtype=dt).eval()
        dense_mib = d_in * d_out * dense.weight.element_size() / 2**20
        for b in (float(v) for v in args.bpws.split(",")):
            lb = LittleBitLinear(d_in, d_out, bpw=b, residual=True)
            pk = PackedLittleBitLinear(lb).to(dev)
            del lb
            pk_mib = pk.storage_bytes() / 2**20
            for t in (int(v) for v in args.tokens.split(",")):
                x = torch.randn(t, d_in, device=dev, dtype=dt)
                t_dense = _time(lambda: dense(x), dev, args.reps)
                pk.backend = "torch"
                t_torch = _time(lambda: pk(x), dev, args.reps)
                with torch.no_grad():
                    ref = pk(x).float()
                pk.backend = "helion"
                with torch.no_grad():
                    out = pk(x).float()  # first call compiles / autotunes
                t_hel = _time(lambda: pk(x), dev, args.reps)
                err = (out - ref).abs().max().item()
                print(f"| {d} | {t} | {pk.storage_bytes() * 8 / (d_in * d_out):.2f} ({pk.rank}) | {dense_mib:.1f} MiB | {pk_mib:.2f} MiB | "
                      f"{t_dense:.3f} | {t_torch:.3f} | {t_hel:.3f} | {t_dense / t_hel:.2f}x | {err:.2e} |", flush=True)


if __name__ == "__main__":
    main()
