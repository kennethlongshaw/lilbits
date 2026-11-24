"""Loss vs stored size: LittleBit at several bpw against a BitNet b1.58 width sweep and an FP16 width sweep.

LittleBit keeps d_model 128 and lowers bits per weight. BitNet (1.58 bpw, fixed) and FP16 can only
shrink by getting narrower, so they sweep d_model so their stored sizes bracket LittleBit's.
Every run uses the same data stream and eval batches as lb-tune (MarkovSource seeded by the run seed).

    uv sync --extra tune --extra plot
    uv run python benchmarks/frontier.py --stage grid      # small lr / optimizer grid for BitNet (seed 0)
    uv run python benchmarks/frontier.py --stage main      # all points on 3 seeds (resumes from the JSONL)
    uv run python benchmarks/frontier.py --stage plot      # PNG + markdown table
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
from concurrent.futures import ProcessPoolExecutor

import torch

from lilbits.measure import size_report
from lilbits.data import MarkovSource
from lilbits.models.tinylm import Config
from lilbits.training.train import TrainConfig, train

# Optuna winners from results/optuna (tuned at d_model 128, LittleBit at 1.41 bpw)
LB_TUNED = dict(lr=0.017513239557445673, weight_decay=0.01, warmup=100, optim="muon_ds",
                muon_lr_mult=0.4280616809539174, latent_lr_mult=0.018332324963029632,
                scale_lr_mult=0.5002803233519172, fp_warmup=600)
FP_TUNED = dict(lr=0.0037861878865426053, weight_decay=0.0, warmup=150, optim="muon",
                muon_lr_mult=1.8571470500322553, muon_ns_steps=10)

BITNET_GRID = [
    dict(optim="adamw", lr=3e-3, warmup=100),
    dict(optim="adamw", lr=6e-3, warmup=100),
    dict(optim="adamw", lr=1.2e-2, warmup=100),
    dict(optim="adamw", lr=2.4e-2, warmup=100),
    dict(optim="muon_ds", lr=4e-3, warmup=100),
    dict(optim="muon_ds", lr=1e-2, warmup=100),
    dict(optim="muon_ds", lr=2e-2, warmup=100),
    dict(optim="muon", lr=FP_TUNED["lr"], warmup=150, muon_lr_mult=FP_TUNED["muon_lr_mult"], muon_ns_steps=10),
]
GRID_D = 96


def main_specs(bitnet_tc: dict) -> list[dict]:
    lb = lambda bpw, res: dict(family="LittleBit" + ("" if res else " (1 path)"), linear="littlebit", d_model=128,
                               bpw=bpw, residual=res, itq_iters=50, init="small", tc=LB_TUNED)
    specs = [lb(1.41, True), lb(0.8, True), lb(0.55, True), lb(0.8, False), lb(0.55, False), lb(0.3, False)]
    specs += [dict(family="BitNet b1.58", linear="bitnet", d_model=d, tc=bitnet_tc) for d in (64, 80, 96, 112, 128)]
    specs += [dict(family="FP16", linear="fp", d_model=d, tc=FP_TUNED) for d in (24, 32, 40, 48)]
    return specs


def key(spec: dict, seed: int) -> str:
    return json.dumps({**{k: v for k, v in spec.items() if k != "family"}, "seed": seed}, sort_keys=True)


def run_one(job) -> dict:
    spec, seed, steps = job
    torch.set_num_threads(1)
    src = MarkovSource(vocab=32, seed=seed)
    eval_batches = [src.sample(32, 64) for _ in range(4)]
    model_kw = {k: spec[k] for k in ("linear", "d_model", "bpw", "residual", "itq_iters", "init") if k in spec}
    cfg = Config(**model_kw)
    tc = TrainConfig(steps=steps, log_every=100, seed=seed, verbose=False, accelerator="cpu", **spec["tc"])
    res = train(cfg, tc, src, eval_batches)
    sizes = size_report(res["model"])
    return {"key": key(spec, seed), "family": spec["family"], "spec": spec, "seed": seed,
            "final_eval": res["final_eval"], **sizes}


def run_jobs(jobs, out: str, workers: int):
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["key"] for line in open(out)}
    todo = [j for j in jobs if key(j[0], j[1]) not in done]
    print(f"{len(jobs) - len(todo)} done, {len(todo)} to run")
    # spawn, not fork: forked workers intermittently die with "oneMKL FATAL: cannot load libtorch_cpu" on SVD
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(run_one, todo):
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{r['family']:20s} d={r['spec']['d_model']:3d} bpw={r['block_bpw']:.3f} "
                  f"{r['deployed_total_bytes'] / 1024:6.1f} KiB  seed {r['seed']}  eval {r['final_eval']:.4f}", flush=True)


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def best_grid(path: str) -> dict:
    rows = load(path)
    if not rows:
        raise SystemExit(f"no grid results in {path}; run --stage grid first")
    return min(rows, key=lambda r: r["final_eval"])["spec"]["tc"]


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(json.dumps({k: v for k, v in r["spec"].items() if k != "tc"}, sort_keys=True), []).append(r)
    agg = []
    for rs in groups.values():
        losses = [r["final_eval"] for r in rs]
        agg.append({"family": rs[0]["family"], "d_model": rs[0]["spec"]["d_model"], "block_bpw": rs[0]["block_bpw"],
                    "total_kib": rs[0]["deployed_total_bytes"] / 1024, "block_kib": rs[0]["deployed_block_bytes"] / 1024,
                    "params": rs[0]["trainable_params"], "mean": statistics.mean(losses),
                    "std": statistics.stdev(losses) if len(losses) > 1 else 0.0, "seeds": losses})
    return sorted(agg, key=lambda a: (a["family"], a["total_kib"]))


def plot(agg: list[dict], png: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"LittleBit": ("#2a6fdb", "o"), "LittleBit (1 path)": ("#7aa7f0", "s"),
             "BitNet b1.58": ("#d9622b", "^"), "FP16": ("#6b6b6b", "D")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, xkey, xlabel in ((axes[0], "total_kib", "stored size, whole model (KiB)"),
                             (axes[1], "block_kib", "stored size, block linears only (KiB)")):
        for fam, (color, marker) in style.items():
            pts = sorted((a for a in agg if a["family"] == fam), key=lambda a: a[xkey])
            if not pts:
                continue
            ax.errorbar([a[xkey] for a in pts], [a["mean"] for a in pts], yerr=[a["std"] for a in pts],
                        color=color, marker=marker, ms=6, lw=1.8, capsize=3, label=fam)
            for a in pts:
                tag = f"{a['block_bpw']:.2f}" if fam.startswith("LittleBit") else f"d{a['d_model']}"
                ax.annotate(tag, (a[xkey], a["mean"]), textcoords="offset points", xytext=(5, 5),
                            fontsize=7.5, color=color)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("eval loss (nats), mean ± sd of 3 seeds")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Loss vs stored size (LittleBit labels: bits per weight; others: d_model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=150)


def table(agg: list[dict]) -> str:
    lines = ["| model | d_model | block bpw | block linears | whole model | trainable params | eval loss (mean ± sd) |",
             "|---|---|---|---|---|---|---|"]
    for a in sorted(agg, key=lambda a: a["total_kib"]):
        lines.append(f"| {a['family']} | {a['d_model']} | {a['block_bpw']:.2f} | {a['block_kib']:.1f} KiB | "
                     f"{a['total_kib']:.1f} KiB | {a['params']:,} | {a['mean']:.3f} ± {a['std']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["grid", "main", "plot"], required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dir", default="results/frontier")
    args = ap.parse_args(argv)
    os.makedirs(args.dir, exist_ok=True)
    grid_path, main_path = os.path.join(args.dir, "bitnet_grid.jsonl"), os.path.join(args.dir, "runs.jsonl")

    if args.stage == "grid":
        specs = [dict(family="BitNet b1.58", linear="bitnet", d_model=GRID_D, tc=tc) for tc in BITNET_GRID]
        run_jobs([(s, 0, args.steps) for s in specs], grid_path, args.workers)
        for r in sorted(load(grid_path), key=lambda r: r["final_eval"]):
            print(f"  {r['final_eval']:.4f}  {r['spec']['tc']}")
    elif args.stage == "main":
        specs = main_specs(best_grid(grid_path))
        run_jobs([(s, seed, args.steps) for s in specs for seed in range(args.seeds)], main_path, args.workers)
    else:
        agg = aggregate(load(main_path))
        plot(agg, os.path.join(args.dir, "loss_vs_size.png"))
        md = table(agg)
        with open(os.path.join(args.dir, "table.md"), "w") as f:
            f.write(md + "\n")
        print(md)


if __name__ == "__main__":
    main()
