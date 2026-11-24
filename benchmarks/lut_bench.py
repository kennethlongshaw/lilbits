"""Decode-speed benchmark for the lookup-table (T-MAC style) kernels vs dense and unpack-then-matmul layers.

For each weight shape and token count, every layer variant is timed on the same input. Reported per row:
wall ms per call (median of repeats), CPU ms per call, peak RSS growth during the calls, weight RAM, on-disk
state_dict size, and max error relative to the variant's own dense reference.

Variants:
  fp32 dense / bf16 dense          nn.Linear
  LittleBit {bpw} unpack           PackedLittleBitLinear (1-bit signs, unpacked every call)
  LittleBit {bpw} LUT              LUTLittleBitLinear (same 1-bit signs, LUT kernel)
  BitNet unpack                    PackedBitLinear (2-bit codes, unpacked every call)
  BitNet LUT                       LUTBitLinear (two sign planes, 2 bits/weight, LUT kernel)

Shapes default to BitNet b1.58 2B4T's projections (d 2560, FFN 6912) and a 4096 square.
The LUT layers are forced onto the kernel at every token count here (normally > 8 tokens falls back to dense).

    uv run python benchmarks/lut_bench.py                 # 1 and 4 threads
    uv run python benchmarks/lut_bench.py --threads 1 --tokens 1
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from lilbits.inference.lut import LUTBitLinear, LUTLittleBitLinear, kernel_info
from lilbits.inference.packed import PackedLittleBitLinear
from lilbits.layers.bitnet import BitLinear, PackedBitLinear
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.measure import Meter, state_dict_bytes, tensor_bytes

FORCE = 10**9  # max_lut_tokens: always use the kernel


def time_call(fn, reps: int) -> tuple[float, float]:
    """(median wall ms, mean CPU ms) per call."""
    for _ in range(3):
        fn()
    walls = []
    c0 = time.process_time()
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        walls.append(time.perf_counter() - t)
    return statistics.median(walls) * 1e3, (time.process_time() - c0) / reps * 1e3


def variants(d_in: int, d_out: int, bpws: list[float]) -> dict[str, tuple[nn.Module, torch.Tensor]]:
    """name -> (layer, dense reference weight in fp32)."""
    torch.manual_seed(0)
    out: dict[str, tuple[nn.Module, torch.Tensor]] = {}
    fp = nn.Linear(d_in, d_out, bias=False).eval()
    out["fp32 dense"] = (fp, fp.weight.detach())
    out["bf16 dense"] = (copy.deepcopy(fp).to(torch.bfloat16), fp.weight.detach())
    for b in bpws:
        lb = LittleBitLinear(d_in, d_out, bpw=b, residual=True).eval()
        with torch.no_grad():
            W = lb.weight().detach()
        out[f"LittleBit {b} bpw unpack"] = (PackedLittleBitLinear(lb), W)
        out[f"LittleBit {b} bpw LUT"] = (LUTLittleBitLinear(lb, FORCE), W)
        del lb
    bl = BitLinear(d_in, d_out).eval()
    out["BitNet b1.58 unpack"] = (PackedBitLinear(bl), torch.empty(0))
    out["BitNet b1.58 LUT"] = (LUTBitLinear(bl, FORCE), torch.empty(0))
    out["BitNet b1.58 dense ternary (cached fp32)"] = (bl, torch.empty(0))
    return out


def run(shapes, bpws, tokens_list, threads_list, reps) -> list[dict]:
    rows = []
    for d_in, d_out in shapes:
        vs = variants(d_in, d_out, bpws)
        bitnet_ref = vs["BitNet b1.58 dense ternary (cached fp32)"][0]
        for threads in threads_list:
            torch.set_num_threads(threads)
            for tokens in tokens_list:
                x = torch.randn(tokens, d_in)
                with torch.no_grad():
                    bitnet_y = bitnet_ref(x)
                    for name, (layer, W) in vs.items():
                        xin = x.to(torch.bfloat16) if "bf16" in name else x
                        y = layer(xin).float()
                        ref = bitnet_y if name.startswith("BitNet") else x @ W.t()
                        err = ((y - ref).abs().max() / ref.abs().max().clamp(min=1e-9)).item()
                        with Meter() as m:
                            wall, cpu = time_call(lambda layer=layer, xin=xin: layer(xin), reps)
                        weights = tensor_bytes([*layer.parameters(), *layer.buffers()])
                        if name.startswith("BitNet b1.58 dense"):
                            weights += tensor_bytes([layer._infer_w])
                        rows.append(
                            dict(
                                shape=f"{d_in}x{d_out}",
                                threads=threads,
                                tokens=tokens,
                                variant=name,
                                wall_ms=wall,
                                cpu_ms=cpu,
                                peak_rss_growth_mib=m.peak_extra_mib,
                                weight_ram_bytes=weights,
                                disk_bytes=state_dict_bytes(layer),
                                rel_err=err,
                            )
                        )
                        print(
                            f"{d_in}x{d_out} t={threads} n={tokens:<3} {name:<44} {wall:8.3f} ms  cpu {cpu:8.3f} ms"
                            f"  ram {weights / 2**20:7.2f} MiB  err {err:.1e}",
                            flush=True,
                        )
    return rows


# BitNet b1.58 2B4T block projections (d 2560, GQA k/v 640, FFN 6912): (d_in, d_out) per linear
BLOCK_2B4T = [(2560, 2560), (2560, 640), (2560, 640), (2560, 2560), (2560, 6912), (2560, 6912), (6912, 2560)]


def make_layer(kind: str, d_in: int, d_out: int) -> nn.Module:
    if kind == "fp32 dense":
        return nn.Linear(d_in, d_out, bias=False).eval()
    if kind == "bf16 dense":
        return nn.Linear(d_in, d_out, bias=False).to(torch.bfloat16).eval()
    if kind.startswith("LittleBit"):
        bpw = float(kind.split()[1])
        lb = LittleBitLinear(d_in, d_out, bpw=bpw, residual=True).eval()
        return LUTLittleBitLinear(lb) if kind.endswith("LUT") else PackedLittleBitLinear(lb)
    bl = BitLinear(d_in, d_out).eval()
    if kind.endswith("LUT"):
        return LUTBitLinear(bl)
    if kind.endswith("unpack"):
        return PackedBitLinear(bl)
    with torch.no_grad():
        bl(torch.zeros(1, d_in))  # cache the dense ternary weight
    bl.weight.requires_grad_(False)
    return bl


def block_decode(n_blocks: int, threads_list, reps: int, kinds: list[str]) -> list[dict]:
    """One-token decode through the linears of n_blocks 2B4T-shaped blocks (weights larger than cache for fp32)."""
    rows = []
    for kind in kinds:
        torch.manual_seed(0)
        layers = [make_layer(kind, i, o) for _ in range(n_blocks) for i, o in BLOCK_2B4T]
        dt = torch.bfloat16 if "bf16" in kind else torch.float32
        xs = {d: torch.randn(1, d, dtype=dt) for d in (2560, 6912)}

        dims = [i for _ in range(n_blocks) for i, _o in BLOCK_2B4T]

        def step(layers=layers, xs=xs, dims=dims):
            for m, d in zip(layers, dims, strict=True):
                m(xs[d])

        weights = tensor_bytes([t for m in layers for t in (*m.parameters(), *m.buffers())])
        if kind.endswith("cached fp32)"):
            weights = tensor_bytes([m._infer_w for m in layers]) + tensor_bytes([m.weight for m in layers])
        for threads in threads_list:
            torch.set_num_threads(threads)
            with torch.no_grad(), Meter() as m:
                wall, cpu = time_call(step, reps)
            per_layer = wall / n_blocks
            rows.append(
                dict(
                    variant=kind,
                    threads=threads,
                    blocks=n_blocks,
                    wall_ms=wall,
                    cpu_ms=cpu,
                    peak_rss_growth_mib=m.peak_extra_mib,
                    weight_ram_bytes=weights,
                    est_30_layer_ms_per_token=per_layer * 30,
                )
            )
            print(f"block decode t={threads} {kind:<44} {wall:8.2f} ms / token for {n_blocks} blocks", flush=True)
        del layers
    return rows


def block_table(rows: list[dict]) -> str:
    out = [
        f"\n**One-token decode through the linears of {rows[0]['blocks']} BitNet-2B4T-shaped blocks** "
        "(q, k, v, o, gate, up, down; fp32 weights exceed the cache)\n",
        "| variant | threads | wall ms / token | CPU ms | 30-layer linears, est. ms / token (tok/s) "
        "| weight RAM | peak RSS growth |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        est = r["est_30_layer_ms_per_token"]
        out.append(
            f"| {r['variant']} | {r['threads']} | {r['wall_ms']:.2f} | {r['cpu_ms']:.2f} | {est:.1f} ({1e3 / est:.0f}) "
            f"| {r['weight_ram_bytes'] / 2**20:,.1f} MiB | {r['peak_rss_growth_mib']:.1f} MiB |"
        )
    return "\n".join(out)


def table(rows: list[dict]) -> str:
    out = []
    keys = sorted({(r["shape"], r["threads"], r["tokens"]) for r in rows}, key=lambda k: (k[0], k[1], k[2]))
    for shape, threads, tokens in keys:
        sub = [r for r in rows if (r["shape"], r["threads"], r["tokens"]) == (shape, threads, tokens)]
        base = next(r for r in sub if r["variant"] == "fp32 dense")["wall_ms"]
        out += [
            f"\n**{shape}, {tokens} token{'s' if tokens > 1 else ''}, {threads} thread{'s' if threads > 1 else ''}**\n",
            "| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sub:
            out.append(
                f"| {r['variant']} | {r['wall_ms']:.3f} | {base / r['wall_ms']:.2f}× | {r['cpu_ms']:.3f} "
                f"| {r['peak_rss_growth_mib']:.1f} MiB | {r['weight_ram_bytes'] / 2**20:.2f} MiB "
                f"| {r['disk_bytes'] / 2**20:.2f} MiB | {r['rel_err']:.1e} |"
            )
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="2560x2560,2560x6912,6912x2560,4096x4096")
    ap.add_argument("--bpws", default="1.0,0.55")
    ap.add_argument("--tokens", default="1,4,16")
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=4, help="blocks for the stacked decode test (0 = skip)")
    ap.add_argument("--out", default="results/lut/lut_bench")
    args = ap.parse_args(argv)
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    info = kernel_info()
    print(f"LUT kernel: {info}")
    rows = run(
        shapes,
        [float(b) for b in args.bpws.split(",")],
        [int(t) for t in args.tokens.split(",")],
        [int(t) for t in args.threads.split(",")],
        args.reps,
    )
    threads = [int(t) for t in args.threads.split(",")]
    kinds = [
        "fp32 dense",
        "bf16 dense",
        "LittleBit 1.0 bpw unpack",
        "LittleBit 1.0 bpw LUT",
        "LittleBit 0.55 bpw LUT",
        "BitNet b1.58 unpack",
        "BitNet b1.58 LUT",
        "BitNet b1.58 dense ternary (cached fp32)",
    ]
    blocks = block_decode(args.blocks, threads, max(3, args.reps // 4), kinds) if args.blocks else []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(
            {"kernel": info, "torch": torch.__version__, "cpu_threads": threads, "rows": rows, "block_decode": blocks},
            indent=1,
        )
    )
    md = f"LUT kernel SIMD width: {info['simd']} bits; torch {torch.__version__}\n"
    md += (block_table(blocks) + "\n\n**Single layers**\n" if blocks else "") + table(rows)
    out.with_suffix(".md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
