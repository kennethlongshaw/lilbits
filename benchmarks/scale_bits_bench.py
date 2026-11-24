"""FP16 vs INT8 LittleBit scales at the same bits per weight: rank bought, stored size, approximation error.

INT8 scales (one FP16 step per vector) halve the cost of h, g, l, so at a fixed bpw budget the freed bits go to
rank. This fits each layer to a fixed target matrix with LittleBit's Dual-SVID init (no training), so it measures
how much closer the cheaper scales get at equal storage. The target has a decaying spectrum (s_i ~ i^-0.5) plus
noise, like trained weights more than a pure Gaussian.

    uv run python benchmarks/scale_bits_bench.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lilbits.inference.packed import PackedLittleBitLinear
from lilbits.layers.littlebit import LittleBitLinear


def target(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    s = torch.arange(1, n + 1).float() ** -0.5
    W = (U * s) @ V.t()
    return W / W.std() * n**-0.5 + 0.1 * torch.randn(n, n, generator=g) * n**-0.5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", default="128,512,2048")
    ap.add_argument("--bpws", default="1.0,0.55,0.3")
    ap.add_argument("--out", default="results/scale_bits_bench")
    args = ap.parse_args(argv)
    rows = []
    for n in (int(v) for v in args.widths.split(",")):
        W = target(n)
        x = torch.randn(256, n)
        ref = x @ W.t()
        for bpw in (float(b) for b in args.bpws.split(",")):
            for bits in (16, 8):
                t = time.perf_counter()
                lb = LittleBitLinear(n, n, bpw=bpw, residual=True, scale_bits=bits)
                lb.init_from_weight(W, itq_iters=20)
                lb.eval()
                with torch.no_grad():
                    rel_w = ((lb.weight() - W).norm() / W.norm()).item()
                    packed = PackedLittleBitLinear(lb)
                    rel_y = ((packed(x) - ref).norm() / ref.norm()).item()
                rows.append(
                    {
                        "width": n,
                        "bpw_budget": bpw,
                        "scale_bits": bits,
                        "rank": lb.rank,
                        "bpw": lb.bpw,
                        "packed_bytes": packed.storage_bytes(),
                        "rel_weight_err": rel_w,
                        "rel_output_err": rel_y,
                        "fit_s": time.perf_counter() - t,
                    }
                )
                print(rows[-1], flush=True)
    md = [
        "| width | bpw budget | scales | rank | actual bpw | packed bytes | weight rel. error | output rel. error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {r['width']} | {r['bpw_budget']} | {'INT8' if r['scale_bits'] == 8 else 'FP16'} | {r['rank']} "
        f"| {r['bpw']:.3f} | {r['packed_bytes']:,} | {r['rel_weight_err']:.3f} | {r['rel_output_err']:.3f} |"
        for r in rows
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
