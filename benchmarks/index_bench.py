"""Indexed retrieval decode step: masked dense attention vs sparse gather, over growing context.

One query token (decode) against a packed index (int4 K/V, 1-bit signatures, block summaries). The masked path
dequantizes and scores every stored token; the gather path dequantizes and attends only to the selected top-k
tokens plus the local block. Reports wall ms, CPU ms and peak RSS growth per step, and the output difference.

    uv run python benchmarks/index_bench.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lilbits.measure import Meter
from lilbits.mixers.sparse_index import IndexedRetrieval, TokenIndex


def build(ctx, d, heads, dim, block, top_blocks, top_k):
    torch.manual_seed(0)
    r = IndexedRetrieval(d, heads, dim, top_blocks=top_blocks, top_k=top_k, flat=1.0).eval()
    idx = TokenIndex(block=block, packed=True)
    for s in range(0, ctx, block):
        e = torch.randn(1, min(block, ctx - s), d)
        r.write(e, idx)
        if e.shape[1] == block:
            r.write_summaries(e.mean(1), e.mean(1), idx)
    return r, idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contexts", default="1024,4096,16384,65536")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--top-blocks", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/index_bench")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    rows = []
    with torch.no_grad():
        for ctx in (int(c) for c in args.contexts.split(",")):
            r, idx = build(ctx, args.d_model, args.heads, args.dim, args.block, args.top_blocks, args.top_k)
            x = torch.randn(1, 1, args.d_model)
            pos0 = idx.length - 1
            ref = None
            for name, gather in (("masked dense", False), ("sparse gather", True)):
                r(x, idx, pos0, gather=gather)  # warm-up (also builds the concatenated storage)
                with Meter() as m:
                    for _ in range(args.reps):
                        out, _ = r(x, idx, pos0, gather=gather)
                ref = out if ref is None else ref
                rows.append(
                    {
                        "context": ctx,
                        "path": name,
                        "step_ms": m.wall_s / args.reps * 1e3,
                        "cpu_ms": m.cpu_s / args.reps * 1e3,
                        "peak_rss_growth_mib": m.peak_extra_mib,
                        "index_kib": sum(idx.nbytes().values()) / 1024,
                        "max_diff_vs_masked": (out - ref).abs().max().item(),
                    }
                )
                print(rows[-1], flush=True)
    md = [
        "| context | path | step ms | CPU ms | peak RSS growth | index size | max diff vs masked |",
        "|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {r['context']} | {r['path']} | {r['step_ms']:.2f} | {r['cpu_ms']:.2f} | "
        f"{r['peak_rss_growth_mib']:.1f} MiB | {r['index_kib']:.0f} KiB | {r['max_diff_vs_masked']:.1e} |"
        for r in rows
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
