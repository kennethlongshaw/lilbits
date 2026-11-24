"""Train FP baseline and LittleBit models on the identical data stream and compare
size, loss, tokens seen, wall time and memory.

Each config runs in its own subprocess so peak-RSS numbers don't mix.

    uv run python benchmarks/bench.py                       # defaults: d_model 128, 2 layers, 1,500 steps
    uv run python benchmarks/bench.py --device cuda --d-model 1024 --n-layers 6 --steps 5000
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from typing import Any

import torch

from lilbits.data import MarkovSource
from lilbits.measure import size_report
from lilbits.models.tinylm import Config
from lilbits.training.train import TrainConfig, train

CONFIGS: dict[str, dict[str, Any]] = {
    "fp": dict(linear="fp", optim="adamw", fp_warmup=0, init="small", itq_iters=0, label="FP16 baseline"),
    "lb_scratch": dict(
        linear="littlebit",
        optim="adamw",
        fp_warmup=0,
        init="svd",
        itq_iters=50,
        label="LittleBit ~{bpw} bpw, SVD+ITQ init (pure from scratch)",
    ),
    "lb_warm": dict(
        linear="littlebit",
        optim="adamw",
        fp_warmup=300,
        init="small",
        itq_iters=50,
        label="LittleBit ~{bpw} bpw, 300-step FP warm start + SVD+ITQ",
    ),
}


def _rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 if sys.platform != "darwin" else usage / 2**20


def _bytes(params) -> int:
    return sum(p.numel() * p.element_size() for p in params)


OPTIMS: dict[str, dict[str, Any]] = {
    "adamw": dict(optim="adamw", ns=5, label="AdamW"),
    "muon": dict(optim="muon", ns=5, label="torch Muon 5 NS"),
    "muon10": dict(optim="muon", ns=10, label="torch Muon 10 NS"),
    "muon_ds": dict(optim="muon_ds", ns=10, label="DeepSeek Muon"),
}


def run_one(spec: str, args) -> dict:
    torch.set_num_threads(args.threads)
    key, _, okey = spec.partition(":")
    c = dict(CONFIGS[key])
    o = OPTIMS[okey or "adamw"]
    source = MarkovSource(vocab=args.vocab, seed=args.seed)  # same seed -> identical stream for every config
    eval_batches = [source.sample(args.batch, args.seq_len) for _ in range(4)]
    floor = sum(source.entropy_floor(b) for b in eval_batches) / len(eval_batches)
    rss0 = _rss_mib()
    cfg = Config(
        vocab_size=args.vocab,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        linear=c["linear"],
        bpw=args.bpw,
        init=c["init"],
        itq_iters=c["itq_iters"],
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tc = TrainConfig(
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        log_every=args.steps,
        weight_decay=args.weight_decay,
        latent_lr_mult=0.1,
        seed=args.seed,
        verbose=False,
        fp_warmup=c["fp_warmup"],
        optim=o["optim"],
        muon_ns_steps=o["ns"],
        accelerator="cuda" if args.device.startswith("cuda") else "cpu",
    )
    res = train(cfg, tc, source, eval_batches)
    wall = time.perf_counter() - t0
    train_peak = torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else _rss_mib() - rss0
    model = res["model"].eval()
    sizes = size_report(model)
    # AdamW keeps exp_avg + exp_avg_sq (2x params); Muon keeps one momentum buffer for its 2D block matrices
    from lilbits.training.train import param_groups

    groups = param_groups(model, 1.0, 1.0, 0.0)
    muon_bytes = sum(_bytes(g["params"]) for g in groups if g["name"] in ("hidden", "latent"))
    opt_state_bytes = (
        2 * sizes["train_param_bytes"]
        if o["optim"] == "adamw"
        else muon_bytes + 2 * (sizes["train_param_bytes"] - muon_bytes)
    )
    xb = eval_batches[0][:, :-1].to(args.device)
    with torch.no_grad():
        for _ in range(3):
            model(xb)
        t = time.perf_counter()
        for _ in range(20):
            model(xb)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t) / 20 * 1e3
    return {
        "key": key,
        "optim": okey or "adamw",
        "label": c["label"].format(bpw=f"{args.bpw:g}") + ("" if not okey else f" [{o['label']}]"),
        "final_eval": res["final_eval"],
        "entropy_floor": floor,
        "steps": args.steps,
        "tokens": args.steps * args.batch * args.seq_len,
        "fp_warmup_steps": c["fp_warmup"],
        "wall_s": wall,
        "ms_per_step": wall / args.steps * 1e3,
        "train_peak_mib": train_peak,
        "train_state_bytes": 2 * sizes["train_param_bytes"] + opt_state_bytes,
        "infer_ms_per_batch": infer_ms,
        **sizes,
    }


def table(rows: list[dict]) -> str:
    kib = lambda b: f"{b / 1024:,.1f} KiB"
    mib = lambda b: f"{b / 2**20:,.2f} MiB"
    fp = next((r for r in rows if r["key"] == "fp"), rows[0])
    [("", lambda r: r["label"])]
    lines = []
    metrics = [
        ("Final eval loss (floor {:.3f})".format(rows[0]["entropy_floor"]), lambda r: f"{r['final_eval']:.3f}"),
        (
            "Training steps / tokens seen",
            lambda r: (
                f"{r['steps']:,} / {r['tokens'] / 1e6:.2f}M"
                + (f" ({r['fp_warmup_steps']} FP)" if r["fp_warmup_steps"] else "")
            ),
        ),
        ("Wall time (ms/step)", lambda r: f"{r['wall_s']:.0f} s ({r['ms_per_step']:.0f})"),
        ("Trainable params (FP32 while training)", lambda r: f"{r['trainable_params']:,}"),
        ("Train state: weights+grads+optimizer", lambda r: mib(r["train_state_bytes"])),
        ("Train peak memory (RSS delta / CUDA peak)", lambda r: f"{r['train_peak_mib']:.0f} MiB"),
        ("Deployed block linears", lambda r: f"{kib(r['deployed_block_bytes'])} ({r['block_bpw']:.2f} bpw)"),
        ("Deployed whole model", lambda r: kib(r["deployed_total_bytes"])),
        (
            "Compression vs FP16 (blocks / whole)",
            lambda r: (
                f"{fp['deployed_block_bytes'] / r['deployed_block_bytes']:.1f}x"
                f" / {fp['deployed_total_bytes'] / r['deployed_total_bytes']:.1f}x"
            ),
        ),
        ("Inference ms per batch (unpacked)", lambda r: f"{r['infer_ms_per_batch']:.1f}"),
    ]
    lines.append("| metric | " + " | ".join(r["label"] for r in rows) + " |")
    lines.append("|---" * (len(rows) + 1) + "|")
    for name, f in metrics:
        lines.append(f"| {name} | " + " | ".join(f(r) for r in rows) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--configs", default=",".join(CONFIGS), help="comma list of config[:optim], optim in " + ",".join(OPTIMS)
    )
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--bpw", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--parallel", action="store_true", help="run configs concurrently (CPU only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write JSON results here")
    args = ap.parse_args(argv)

    if args.one:
        print(json.dumps(run_one(args.one, args)))
        return
    passthrough = [a for a in (argv if argv is not None else sys.argv[1:])]
    keys = args.configs.split(",")
    cmds = [[sys.executable, __file__, *passthrough, "--one", k] for k in keys]
    if args.parallel:
        procs = [subprocess.Popen(c, stdout=subprocess.PIPE, text=True) for c in cmds]
        outs = [p.communicate()[0] for p in procs]
    else:
        outs = [subprocess.run(c, stdout=subprocess.PIPE, text=True, check=True).stdout for c in cmds]
    rows = [json.loads(o.strip().splitlines()[-1]) for o in outs]
    print(table(rows))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
