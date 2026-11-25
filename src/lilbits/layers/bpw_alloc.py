"""Uneven bits-per-weight: spend a model-wide LittleBit bit budget where it lowers error most.

A uniform bpw gives every layer the same rank budget, but layers differ: some weights have a fast-decaying
spectrum (a low rank already fits them), some see large activations (their errors cost more at the output),
and a looped core's shared weights are used N times per token. Here every candidate layer is fitted with
LittleBit's Dual-SVID init at a grid of ranks, its error is measured as the output error the layer would cause
on calibration activations,

    err(r) = || (W - W_r) diag(sqrt(E[x^2])) ||_F^2     (plain Frobenius without calibration data)

and a greedy multiple-choice knapsack (lower convex hull, best error drop per extra bit first) picks one rank
per layer so the total storage stays within `bpw * total weights` (the uniform allocation's bits).

The per-layer error is a proxy: it lowers the summed layer error (10-15% on layers with mixed spectra, see
`benchmarks/bpw_alloc_bench.py`) but layer errors compound through a deep or looped stack, and on an
untrained looped core the end-to-end hidden-state error was worse than uniform. Validate on the trained model
before keeping an allocation.

    ranks, report = allocate_ranks(weights, bpw=0.55, act_sq=act_sq)
    convert_loop_linears(core, partial(LittleBitLinear, residual=True), ranks=ranks)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager

import torch
from torch import nn

from lilbits.layers.littlebit import FP_BITS, LittleBitLinear, bits_per_weight, rank_for_bpw


def layer_bits(in_features: int, out_features: int, rank: int, residual: bool = True, scale_bits: int = FP_BITS):
    paths = 2 if residual else 1
    return bits_per_weight(in_features, out_features, rank, paths, scale_bits) * in_features * out_features


def rank_grid(in_features: int, out_features: int, max_bpw: float, n: int = 12, residual: bool = True, **kw):
    """Up to n ranks from 1 to the rank at max_bpw, spaced geometrically (error curves are steep at low rank)."""
    top = rank_for_bpw(in_features, out_features, max_bpw, 2 if residual else 1, kw.get("scale_bits", FP_BITS))
    rs = torch.logspace(0, torch.log10(torch.tensor(float(top))).item(), n).round().int().tolist()
    return sorted(set(max(1, min(r, top)) for r in rs))


@torch.no_grad()
def error_curve(
    W: torch.Tensor,
    ranks: Iterable[int],
    act_sq: torch.Tensor | None = None,
    residual: bool = True,
    scale_bits: int = FP_BITS,
    itq_iters: int = 0,
) -> list[tuple[int, float, float]]:
    """(rank, bits, error) for a LittleBit fit of W (out x in) at each rank."""
    out_f, in_f = W.shape
    col = act_sq.clamp_min(0).sqrt().to(W) if act_sq is not None else torch.ones(in_f, dtype=W.dtype)
    rows = []
    for r in ranks:
        lb = LittleBitLinear(in_f, out_f, rank=r, residual=residual, scale_bits=scale_bits)
        lb.init_from_weight(W, itq_iters)
        err = ((lb.weight() - W.float()) * col).pow(2).sum().item()
        rows.append((r, layer_bits(in_f, out_f, r, residual, scale_bits), err))
    return rows


def _hull(curve: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """Lower convex hull of (bits, error), keeping only points that lower the error."""
    pts = sorted(curve, key=lambda t: t[1])
    hull: list[tuple[int, float, float]] = []
    for p in pts:
        if hull and p[2] >= hull[-1][2]:
            continue
        while len(hull) >= 2:
            (_, b1, e1), (_, b2, e2) = hull[-2], hull[-1]
            # drop hull[-1] if it lies above the segment hull[-2] -> p
            if (e2 - e1) * (p[1] - b1) >= (p[2] - e1) * (b2 - b1):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def greedy_allocate(curves: dict[str, list[tuple[int, float, float]]], budget_bits: float) -> dict[str, int]:
    """Pick one (rank) per layer minimizing total error with total bits <= budget (greedy on convex hulls)."""
    hulls = {k: _hull(c) for k, c in curves.items()}
    pick = dict.fromkeys(hulls, 0)
    used = sum(h[0][1] for h in hulls.values())
    if used > budget_bits:
        raise ValueError(f"budget {budget_bits:.0f} bits is below the rank-1 minimum {used:.0f}")
    while True:
        best, best_gain = None, 0.0
        for k, h in hulls.items():
            i = pick[k]
            if i + 1 < len(h):
                db, de = h[i + 1][1] - h[i][1], h[i][2] - h[i + 1][2]
                if used + db <= budget_bits and de / db > best_gain:
                    best, best_gain = k, de / db
        if best is None:
            return {k: hulls[k][i][0] for k, i in pick.items()}
        i = pick[best]
        used += hulls[best][i + 1][1] - hulls[best][i][1]
        pick[best] = i + 1


def allocate_ranks(
    weights: dict[str, torch.Tensor],
    bpw: float,
    act_sq: dict[str, torch.Tensor] | None = None,
    importance: dict[str, float] | None = None,
    residual: bool = True,
    scale_bits: int = FP_BITS,
    max_bpw_factor: float = 3.0,
    grid: int = 12,
    itq_iters: int = 0,
    min_rank_frac: float = 0.25,
) -> tuple[dict[str, int], dict]:
    """Ranks per layer for an average `bpw` over all given weights. `importance` multiplies a layer's error
    (e.g. how many times it runs per token). No layer drops below `min_rank_frac` of its uniform rank: the
    per-layer error is a proxy and starving a layer to rank 1 can cost more end to end than it saves.
    Returns (ranks, report) where the report compares against the uniform allocation at the same budget."""
    act_sq, importance = act_sq or {}, importance or {}
    total_w = sum(W.numel() for W in weights.values())
    uniform = {
        k: rank_for_bpw(W.shape[1], W.shape[0], bpw, 2 if residual else 1, scale_bits) for k, W in weights.items()
    }
    budget = sum(layer_bits(W.shape[1], W.shape[0], uniform[k], residual, scale_bits) for k, W in weights.items())
    curves = {}
    for k, W in weights.items():
        rs = set(rank_grid(W.shape[1], W.shape[0], bpw * max_bpw_factor, grid, residual, scale_bits=scale_bits))
        rs.add(uniform[k])
        floor = max(1, round(uniform[k] * min_rank_frac))
        rs = {max(r, floor) for r in rs}
        w = importance.get(k, 1.0)
        curve = error_curve(W, sorted(rs), act_sq.get(k), residual, scale_bits, itq_iters)
        curves[k] = [(r, b, e * w) for r, b, e in curve]
    ranks = greedy_allocate(curves, budget)

    def total(rk):
        return sum(next(e for r, _, e in curves[k] if r == rk[k]) for k in curves)

    def bits(rk):
        return sum(layer_bits(W.shape[1], W.shape[0], rk[k], residual, scale_bits) for k, W in weights.items())

    report = {
        "uniform_bpw": bits(uniform) / total_w,
        "allocated_bpw": bits(ranks) / total_w,
        "uniform_error": total(uniform),
        "allocated_error": total(ranks),
        "layers": {
            k: {
                "uniform_rank": uniform[k],
                "rank": ranks[k],
                "bpw": layer_bits(W.shape[1], W.shape[0], ranks[k], residual, scale_bits) / W.numel(),
            }
            for k, W in weights.items()
        },
    }
    return ranks, report


@contextmanager
def input_energy(modules: Mapping[str, nn.Module]):
    """Accumulate E[x^2] per input feature for each module while the block runs (calibration forward passes).
    Yields a dict that is filled in place: name -> (in_features,) tensor."""
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    out: dict[str, torch.Tensor] = {}

    def hook(name: str) -> Callable:
        def f(_m, args, _out):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            sums[name] = sums.get(name, 0) + x.pow(2).sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]

        return f

    handles = [m.register_forward_hook(hook(n)) for n, m in modules.items()]
    try:
        yield out
    finally:
        for h in handles:
            h.remove()
        out.update({n: sums[n] / counts[n] for n in sums})


__all__ = ["allocate_ranks", "error_curve", "greedy_allocate", "input_energy", "layer_bits", "rank_grid"]
