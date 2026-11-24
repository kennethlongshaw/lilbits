"""Gated DeltaNet recurrence: naive per-token loop vs chunkwise-parallel (WY) form, forward + backward.

Reports wall ms, CPU ms and peak RSS growth per call for each sequence length, and the max output difference.

    uv run python benchmarks/gdn_bench.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from lilbits.measure import Meter
from lilbits.mixers.recurrent import chunk_gated_delta_rule, gated_delta_rule


def inputs(B, T, H, D):
    torch.manual_seed(0)
    q = (torch.randn(B, T, H, D) * D**-0.5).requires_grad_()
    k = F.normalize(torch.randn(B, T, H, D), dim=-1).requires_grad_()
    v = torch.randn(B, T, H, D).requires_grad_()
    return q, k, v, -torch.rand(B, T, H) * 0.3, torch.rand(B, T, H)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", default="64,256,1024,4096")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/gdn_bench")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    rows = []
    for T in (int(t) for t in args.lengths.split(",")):
        x = inputs(args.batch, T, args.heads, args.dim)
        ref = None
        for name, fn in (("naive loop", gated_delta_rule), ("chunked (C=64)", chunk_gated_delta_rule)):
            reps = 3 if T <= 1024 else 1
            fn(*x)[0].sum().backward()
            with Meter() as m:
                for _ in range(reps):
                    o, _ = fn(*x)
                    o.sum().backward()
            ref = o.detach() if ref is None else ref
            rows.append(
                {
                    "T": T,
                    "rule": name,
                    "fwd_bwd_ms": m.wall_s / reps * 1e3,
                    "cpu_ms": m.cpu_s / reps * 1e3,
                    "peak_rss_growth_mib": m.peak_extra_mib,
                    "max_diff_vs_naive": (o.detach() - ref).abs().max().item(),
                }
            )
            print(rows[-1], flush=True)
    md = ["| T | rule | fwd+bwd ms | CPU ms | peak RSS growth | max diff vs naive |", "|---|---|---|---|---|---|"]
    md += [
        f"| {r['T']} | {r['rule']} | {r['fwd_bwd_ms']:.1f} | {r['cpu_ms']:.1f} | {r['peak_rss_growth_mib']:.1f} MiB "
        f"| {r['max_diff_vs_naive']:.1e} |"
        for r in rows
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
