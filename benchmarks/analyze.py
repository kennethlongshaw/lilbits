"""Look inside trained checkpoints of example 10 (pocket LM) and example 13 (operator loop).

    uv run python benchmarks/analyze.py pocket --ckpt results/ckpt/10_pocket_lm.pt
    uv run python benchmarks/analyze.py oploop --ckpt results/ckpt/13_operator_loop.pt

pocket: deployed bytes by part; time / FLOPs / resident memory by part for prefill and for streaming decode
(native and LUT); bits per byte when exiting after 1..N loops; per-loop state change; LittleBit layer health;
patch boundaries; which index blocks retrieval reaches; MTP acceptance per draft position.
oploop: router entropy per loop, op usage per loop on right vs wrong answers, expected-chain rate.
Writes results/analysis/<name>.{md,json}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning.fabric import Fabric

from lilbits.analysis import (
    fewer_loops,
    histogram,
    littlebit_health,
    loop_convergence,
    patch_report,
    profile_parts,
    router_entropy,
)
from lilbits.measure import size_report
from lilbits.runner import load

ROOT = Path(__file__).resolve().parent.parent


def example(name: str):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.EXAMPLE


def load_models(ex_name: str, ckpt: str, threads: int):
    ex = example(ex_name)
    args = argparse.Namespace(
        mode="test", ckpt=ckpt, threads=threads, accelerator="cpu", devices=1, precision="32-true"
    )
    fab = Fabric(accelerator="cpu", devices=1)
    models, args = load(fab, Path(ckpt), ex, args)
    return models, args


def bpb(model, x) -> float:
    y, _ = model(x, model.new_cache(x.shape[0]))
    return F.cross_entropy(model.logits(y[:, :-1]).flatten(0, 1), x[:, 1:].flatten()).item() / math.log(2)


@torch.no_grad()
def analyze_pocket(ckpt: str, threads: int) -> dict:
    from lilbits.inference.lut import pack_lut
    from lilbits.shakespeare import ShakespeareBytes

    models, _ = load_models("10_pocket_lm", ckpt, threads)
    pocket, base = models["pocket"], models["fp16_same_size"]
    x = ShakespeareBytes("test", seed=123).sample(16, 256)
    out: dict = {}
    parts = {
        "byte encoder": "encoder",
        "router": "pair",
        "up": "up",
        "body prelude": "body.prelude",
        "body core (looped)": "body.core",
        "body coda": "body.coda",
        "down": "down",
        "byte decoder": "decoder",
        "MTP heads": "mtp",
    }
    parts = {k: v for k, v in parts.items() if pocket.get_submodule(v) is not None}
    out["deployed_kib_by_part"] = {
        k: size_report(pocket.get_submodule(v))["deployed_total_bytes"] / 1024 for k, v in parts.items()
    }
    out["deployed_kib_total"] = size_report(pocket)["deployed_total_bytes"] / 1024
    # quality vs loops (exit depth) and convergence
    out["bits_per_byte_by_loops"] = {}
    for n in range(1, pocket.body.cfg.n_loops + 1):
        with fewer_loops(pocket.body, n):
            out["bits_per_byte_by_loops"][f"{n} loops"] = bpb(pocket, x)
    out["fp16_same_size_bits_per_byte"] = F.cross_entropy(
        base(x[:, :-1])[0].flatten(0, 1), x[:, 1:].flatten()
    ).item() / math.log(2)
    out["core_state_change_by_loop"] = loop_convergence(pocket.body, lambda: pocket(x[:4], pocket.new_cache(4)))
    # retrieval and patches
    stats: dict = {}
    c = pocket.new_cache(1)
    pocket(x[:1], c, stats=stats)
    for row in x[1:4]:
        c = pocket.new_cache(1)
        pocket(row[None], c, stats=stats)
    backs = [b for lst in stats.get("block_back", []) for b in lst]
    out["retrieval_blocks_back_hist"] = histogram(backs)
    out["retrieval_dense_fallback_rate"] = sum(stats.get("fallback", [0])) / max(1, len(stats.get("fallback", [0])))
    out["patches"] = patch_report(pocket, x[:4])
    # LittleBit health, summarized per part
    health = littlebit_health(pocket)
    by_part: dict[str, list[float]] = {}
    for r in health:
        by_part.setdefault(".".join(r["layer"].split(".")[:2]), []).append(r["near_zero_latent"])
    out["littlebit_near_zero_latent_by_part"] = {k: sum(v) / len(v) for k, v in by_part.items()}
    out["littlebit_layers"] = len(health)
    # MTP acceptance by draft position
    accs: list[list[float]] = []
    for row in x[4:8]:
        _, st = pocket.generate_speculative(row[None, :128], 48)
        accs.append(st["acceptance_by_draft"])
    out["mtp_acceptance_by_draft"] = [
        sum(a[j] for a in accs if a[j] is not None) / max(1, sum(a[j] is not None for a in accs))
        for j in range(len(accs[0]))
    ]
    # where time and memory go
    prompt, stream = x[8:9, :256], x[9:10, :64]

    def prefill():
        pocket(prompt, pocket.new_cache(1))

    def decode():
        cc = pocket.new_cache(1)
        pocket(prompt[:, :64], cc)
        for t in range(stream.shape[1]):
            pocket(stream[:, t : t + 1], cc)

    out["profile_native"] = {
        "prefill 256 B": profile_parts(prefill, pocket, parts),
        "prefill 64 B + decode 64 B": profile_parts(decode, pocket, parts),
    }
    pack_lut(pocket.body)
    out["profile_lut"] = {
        "prefill 256 B": profile_parts(prefill, pocket, parts),
        "prefill 64 B + decode 64 B": profile_parts(decode, pocket, parts),
    }
    return out


@torch.no_grad()
def analyze_oploop(ckpt: str, threads: int) -> dict:
    from lilbits.op_tasks import chain_match, make, score

    models, args = load_models("13_operator_loop", ckpt, threads)
    rng = random.Random(4242)
    exs = [make(args.task, rng) for _ in range(40)]
    out: dict = {"task": args.task}
    for name, m in models.items():
        right, wrong = [Counter() for _ in range(m.n_loops)], [Counter() for _ in range(m.n_loops)]
        hits, chains, ent = 0.0, 0.0, [0.0] * m.n_loops
        for e in exs:
            ans, trace = m.complete(e.prompt)
            s = score(args.task, ans, e)
            hits += s
            chains += chain_match(args.task, trace)
            for i, op in enumerate(trace):
                (right if s == 1.0 else wrong)[i][op] += 1
            saved = (m.hard, m.tau)
            m.hard, m.tau = False, 1.0  # soft router probabilities for entropy
            x = torch.tensor(list(e.prompt))[None]
            _, probs = m(x, generator=torch.Generator().manual_seed(0))
            m.hard, m.tau = saved
            for i, p in enumerate(probs):
                ent[i] += router_entropy(p[:, -1:])
        out[name] = {
            "score": hits / len(exs),
            "expected_chain_rate": chains / len(exs),
            "router_entropy_by_loop": [v / len(exs) for v in ent],
            "ops_when_right": [dict(c.most_common(3)) for c in right],
            "ops_when_wrong": [dict(c.most_common(3)) for c in wrong],
        }
    return out


def to_md(title: str, d: dict) -> str:
    lines = [f"# {title}", ""]

    def walk(k, v, depth):
        pad = "  " * depth
        if isinstance(v, dict) and not all(isinstance(x, (int, float, str)) for x in v.values()):
            lines.append(f"{pad}- **{k}**")
            for kk, vv in v.items():
                walk(kk, vv, depth + 1)
        else:
            if isinstance(v, float):
                v = f"{v:.4g}"
            elif isinstance(v, dict):
                v = ", ".join(f"{a}: {b:.4g}" if isinstance(b, float) else f"{a}: {b}" for a, b in v.items())
            elif isinstance(v, list):
                v = ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in v)
            lines.append(f"{pad}- {k}: {v}")

    for k, v in d.items():
        walk(k, v, 0)
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["pocket", "oploop"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    res = analyze_pocket(args.ckpt, args.threads) if args.what == "pocket" else analyze_oploop(args.ckpt, args.threads)
    name = args.out or f"{args.what}_{Path(args.ckpt).stem}"
    out = ROOT / "results" / "analysis" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    out.with_suffix(".md").write_text(to_md(name, res))
    print(to_md(name, res))


if __name__ == "__main__":
    main()
