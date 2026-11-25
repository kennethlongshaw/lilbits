"""Tools to see what a model is doing, not just its end metrics.

    profile_parts(fn, parts)              wall time, calls, FLOPs and resident bytes per named part of a model while
                                          fn() runs (e.g. byte encoder / router / body prelude / core / coda / decoder)
    fewer_loops(looped, n)                context manager: run a looped index LM with only its first n loops
    loop_convergence(looped, run)         per-loop relative change of the core state (does it settle, or keep moving?)
    littlebit_health(model)               per LittleBit layer: rank, bpw, latent mass near the sign boundary, scales
    sign_flips(state_a, state_b)          fraction of LittleBit signs that differ between two state dicts
    patch_report(pocket, x)               byte-patch boundaries: rate, length histogram, a rendered sample
    histogram(values)                     compact {bucket: count} for lists of ints
    router_entropy(probs)                 normalized entropy of router probabilities (1 = uniform, 0 = one choice)

All are read-only: they use forward hooks or temporary settings and restore the model afterwards.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from lilbits.layers.littlebit import LittleBitLinear


def _resident_bytes(m: nn.Module) -> int:
    """Parameters + buffers + tensors cached in module __dict__ (inference caches), counted once."""
    seen, total = set(), 0

    def add(t):
        nonlocal total
        if isinstance(t, torch.Tensor) and t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()

    for mod in m.modules():
        for t in [*mod.parameters(recurse=False), *mod.buffers(recurse=False)]:
            add(t)
        for v in mod.__dict__.values():
            if isinstance(v, torch.Tensor):
                add(v)
            elif isinstance(v, (list, tuple, dict)):
                for x in v.values() if isinstance(v, dict) else v:
                    for y in x if isinstance(x, (list, tuple)) else (x,):
                        add(y)
    return total


def profile_parts(fn: Callable[[], object], root: nn.Module, parts: dict[str, str]) -> dict:
    """Run fn() with hooks on named parts of `root` (label -> submodule path, e.g. {"core": "body.core"}; parts
    must not nest). Per part: wall ms (inclusive), share of total, calls, GFLOPs, resident MiB (parameters,
    buffers and inference caches). `_total` adds the share of wall time no part accounts for (Python glue)."""
    mods = {label: root.get_submodule(path) for label, path in parts.items()}
    fn()  # warm-up: first calls build inference caches
    t_in: dict[str, list[float]] = {}
    wall = dict.fromkeys(parts, 0.0)
    calls = dict.fromkeys(parts, 0)
    handles = []
    for label, top in mods.items():
        # containers (ModuleList) are never called: hook their children
        hooked = list(top.children()) if isinstance(top, (nn.ModuleList, nn.ModuleDict)) else [top]
        for m in hooked:

            def pre(_m, _a, label=label):
                t_in.setdefault(label, []).append(time.perf_counter())

            def post(_m, _a, _o, label=label):
                wall[label] += time.perf_counter() - t_in[label].pop()
                calls[label] += 1

            handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    try:
        t0 = time.perf_counter()
        fn()
        total = time.perf_counter() - t0
    finally:
        for h in handles:
            h.remove()
    with FlopCounterMode(display=False) as fc:
        fn()
    flops = fc.get_flop_counts()
    prefix = type(root).__name__
    out: dict = {}
    for label, path in parts.items():
        keys = [f"{prefix}.{path}"]
        if isinstance(mods[label], (nn.ModuleList, nn.ModuleDict)):
            keys = [f"{prefix}.{path}.{n}" for n, _ in mods[label].named_children()]
        f = Counter()
        for k in keys:
            f.update(flops.get(k, {}))
        out[label] = {
            "wall_ms": wall[label] * 1e3,
            "share": wall[label] / total if total else 0.0,
            "calls": calls[label],
            "gflops": sum(f.values()) / 1e9,
            "resident_mib": _resident_bytes(mods[label]) / 2**20,
        }
    out["_total"] = {
        "wall_ms": total * 1e3,
        "unattributed_share": 1 - sum(v["share"] for v in out.values()),
        "gflops": sum(flops.get("Global", {}).values()) / 1e9,
        "resident_mib": _resident_bytes(root) / 2**20,
    }
    return out


@contextmanager
def fewer_loops(looped, n: int):
    """Temporarily run a LoopedIndexLM (or anything with cfg.n_loops) with its first n loops."""
    old = looped.cfg.n_loops
    looped.cfg.n_loops = min(n, old)
    try:
        yield
    finally:
        looped.cfg.n_loops = old


def loop_convergence(looped, run: Callable[[], object]) -> list[float]:
    """Mean relative change ||s_i - s_{i-1}|| / ||s_i|| of the core state per loop (loop 0 vs zeros = 1)."""
    states: list[torch.Tensor] = []
    h = looped.core.register_forward_hook(lambda _m, _a, out: states.append(out[0].detach()))
    try:
        run()
    finally:
        h.remove()
    n = looped.cfg.n_loops
    per: list[list[float]] = [[] for _ in range(n)]
    for j in range(0, len(states) - len(states) % n, n):
        prev = torch.zeros_like(states[j])
        for i in range(n):
            s = states[j + i]
            per[i].append(((s - prev).norm(dim=-1) / s.norm(dim=-1).clamp_min(1e-6)).mean().item())
            prev = s
    return [sum(v) / len(v) if v else float("nan") for v in per]


def littlebit_health(model: nn.Module, margin: float = 0.05) -> list[dict]:
    """Per LittleBit layer: rank, bpw, fraction of latent sign entries within `margin` x std of 0 (they flip
    easily: a high fraction means the signs are still unsettled), and mean |h|, |g|, |l| scales."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, LittleBitLinear):
            lat = torch.cat([torch.cat([p.U.flatten(), p.V.flatten()]) for p in m.paths]).detach()
            near = (lat.abs() < margin * lat.std()).float().mean().item()
            scales = {k: torch.cat([getattr(p, k).flatten() for p in m.paths]).abs().mean().item() for k in "hgl"}
            rows.append(
                {
                    "layer": name,
                    "shape": f"{m.out_features}x{m.in_features}",
                    "rank": m.rank,
                    "bpw": round(m.bpw, 3),
                    "near_zero_latent": near,
                    **{f"mean_|{k}|": v for k, v in scales.items()},
                }
            )
    return rows


def sign_flips(state_a: dict, state_b: dict) -> dict[str, float]:
    """Per LittleBit latent (keys ending in .U / .V): fraction of signs that differ between two state dicts."""
    out = {}
    for k, a in state_a.items():
        if (k.endswith(".U") or k.endswith(".V")) and k in state_b and state_b[k].shape == a.shape:
            out[k] = (torch.sign(a) != torch.sign(state_b[k])).float().mean().item()
    return out


@torch.no_grad()
def patch_report(pocket, x: torch.Tensor, sample_bytes: int = 160) -> dict:
    """Boundary decisions of a PocketLM on bytes x (B, T)."""
    _, aux = pocket(x, pocket.new_cache(x.shape[0]))
    b = aux["b"]
    lengths: list[int] = []
    for row in b:
        starts = [*row.nonzero().flatten().tolist(), row.numel()]
        lengths += [e - s for s, e in itertools.pairwise(starts)]
    text = bytes(x[0, :sample_bytes].tolist())
    marks = b[0, :sample_bytes].tolist()
    rendered = "".join(
        ("|" if m else "") + chr(c) if 32 <= c < 127 else ("|" if m else "") + "~"
        for c, m in zip(text, marks, strict=True)
    )
    return {
        "bytes_per_patch": b.numel() / max(1, int(b.sum())),
        "patch_length_hist": histogram(lengths),
        "sample": rendered,
    }


def histogram(values: list[int], top: int = 12) -> dict:
    return dict(sorted(Counter(values).most_common(top)))


def router_entropy(probs: torch.Tensor) -> float:
    p = probs.flatten(0, -2).clamp_min(1e-12)
    return (-(p * p.log()).sum(-1) / math.log(p.shape[-1])).mean().item()


__all__ = [
    "fewer_loops",
    "histogram",
    "littlebit_health",
    "loop_convergence",
    "patch_report",
    "profile_parts",
    "router_entropy",
    "sign_flips",
]
