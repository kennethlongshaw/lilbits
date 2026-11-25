"""Uneven vs uniform LittleBit bpw at the same total bits: approximation error, stored size, decode cost.

Part 1, synthetic layers: eight weights of different shapes and spectra (s_i ~ i^-a, a from 0.2 to 1.5) with
log-normal input energies, the spread a trained model's layers show. Part 2, the looped index LM's core: the FP
model (random init, no training) is converted once with uniform ranks and once with allocated ranks, calibrated
on RecallSource tokens, and compared to the FP model's hidden states on held-out tokens, plus prefill + decode
time and deployed bytes. No training anywhere: this measures the conversion, not a trained result.

    uv run python benchmarks/bpw_alloc_bench.py
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from lilbits.data import RecallSource
from lilbits.layers.bpw_alloc import allocate_ranks
from lilbits.layers.loop_scaled import LoopScaledLinear
from lilbits.measure import Meter
from lilbits.models.looped_index import (
    LoopedIndexConfig,
    LoopedIndexLM,
    allocate_core_ranks,
    convert_core_to_littlebit,
)


def spectral(out_f, in_f, a, g):
    n = min(out_f, in_f)
    U, _ = torch.linalg.qr(torch.randn(out_f, n, generator=g))
    V, _ = torch.linalg.qr(torch.randn(in_f, n, generator=g))
    W = (U * torch.arange(1, n + 1).float() ** -a) @ V.t()
    return W / W.std() * in_f**-0.5


def synthetic(bpws):
    g = torch.Generator().manual_seed(0)
    shapes = [(256, 256), (256, 1024), (1024, 256), (512, 512), (256, 256), (768, 256), (256, 768), (512, 512)]
    alphas = torch.linspace(0.2, 1.5, len(shapes)).tolist()
    W = {}
    for i, ((o, n), a) in enumerate(zip(shapes, alphas, strict=True)):
        W[f"L{i} {o}x{n} a={a:.1f}"] = spectral(o, n, a, g)
    act = {k: torch.exp(torch.randn(w.shape[1], generator=g) * 0.7 + torch.randn(1, generator=g)) for k, w in W.items()}
    rows = []
    for bpw in bpws:
        t = time.perf_counter()
        _, rep = allocate_ranks(W, bpw, act_sq=act)
        rows.append(
            {
                "bpw_budget": bpw,
                "uniform_bpw": rep["uniform_bpw"],
                "allocated_bpw": rep["allocated_bpw"],
                "uniform_error": rep["uniform_error"],
                "allocated_error": rep["allocated_error"],
                "error_ratio": rep["allocated_error"] / rep["uniform_error"],
                "alloc_s": time.perf_counter() - t,
                "ranks": {k: (v["uniform_rank"], v["rank"]) for k, v in rep["layers"].items()},
            }
        )
        print({k: v for k, v in rows[-1].items() if k != "ranks"}, flush=True)
    return rows


@torch.no_grad()
def hidden(model, x):
    model.use_index, model.distill = True, False
    return model.process(x, model.new_cache(x.shape[0]))


def looped(d, bpws, length):
    src = RecallSource(length=length, seed=0)
    torch.manual_seed(0)
    fp = LoopedIndexLM(LoopedIndexConfig(vocab_size=src.vocab, d_model=d)).eval()
    calib = RecallSource(length=length, seed=1).sample(8)[0]
    held = RecallSource(length=length, seed=2).sample(8)[0]
    ref = hidden(fp, held)
    rows = []
    for bpw in bpws:
        t = time.perf_counter()
        ranks, rep = allocate_core_ranks(fp, calib, bpw)
        alloc_s = time.perf_counter() - t
        for name, rk in (("uniform", None), ("allocated", ranks)):
            m = copy.deepcopy(fp)
            convert_core_to_littlebit(m, bpw, ranks=rk)
            m.eval()
            h = hidden(m, held)
            prompt = held[:1, : length - 1]
            m.generate(prompt, 2)
            with Meter() as meter:
                m.generate(prompt, 16)
            core_bits = sum(mod.storage_bits() for mod in m.core.modules() if isinstance(mod, LoopScaledLinear))
            rows.append(
                {
                    "d_model": d,
                    "bpw_budget": bpw,
                    "alloc": name,
                    "hidden_rel_err": ((h - ref).norm() / ref.norm()).item(),
                    "core_kib": core_bits / 8 / 1024,
                    "decode16_ms": meter.wall_s * 1e3,
                    "decode16_cpu_ms": meter.cpu_s * 1e3,
                    "peak_rss_growth_mib": meter.peak_extra_mib,
                    "alloc_s": alloc_s if rk is not None else 0.0,
                    "calib_error_ratio": rep["allocated_error"] / rep["uniform_error"],
                }
            )
            print(rows[-1], flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bpws", default="1.0,0.55,0.3")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/bpw_alloc_bench")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    bpws = [float(b) for b in args.bpws.split(",")]
    syn, core = synthetic(bpws), looped(args.d, bpws, args.length)
    md = [
        "Synthetic layers (output-weighted error, same total bits):",
        "",
        "| bpw budget | uniform bpw | allocated bpw | uniform error | allocated error | ratio | alloc time |",
        "|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {r['bpw_budget']} | {r['uniform_bpw']:.3f} | {r['allocated_bpw']:.3f} | {r['uniform_error']:.3g} "
        f"| {r['allocated_error']:.3g} | {r['error_ratio']:.2f} | {r['alloc_s']:.1f} s |"
        for r in syn
    ]
    md += ["", "Ranks (uniform -> allocated):", ""]
    md += [f"- bpw {r['bpw_budget']}: " + ", ".join(f"{k} {u}->{a}" for k, (u, a) in r["ranks"].items()) for r in syn]
    md += [
        "",
        f"Looped index LM core, d={args.d}, untrained FP -> LittleBit (hidden-state error vs FP on held-out tokens):",
        "",
        "| bpw budget | ranks | hidden rel. error | core size | decode 16 tok | CPU | peak RSS growth | alloc time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {r['bpw_budget']} | {r['alloc']} | {r['hidden_rel_err']:.3f} | {r['core_kib']:.0f} KiB "
        f"| {r['decode16_ms']:.0f} ms | {r['decode16_cpu_ms']:.0f} ms | {r['peak_rss_growth_mib']:.1f} MiB "
        f"| {r['alloc_s']:.1f} s |"
        for r in core
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"synthetic": syn, "looped_core": core}, indent=1))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
