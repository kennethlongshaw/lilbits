"""Flagship edge LM: the current best small byte LM for CPUs, end to end. Train from scratch, export W8A8, run on
the Rust engine with no Python (`lilbits` CLI, one .lbm file). A living example: `BEST` below is whatever has the
lowest CPU and memory per byte at equal held-out bits per byte so far; results/flagship/README.md has the table.

Candidates:
    plain   models/lean_lm.py without draft heads: d x layers Gated DeltaNet, 4 heads, fixed-size state
    hnet    models/space_hnet.py: narrow GDN encoder + decoder per byte, a wider GDN main network once per word
            (SpaceByte boundaries, no router)

Data: lilbits.corpus, ~10 MB of English (tiny-shakespeare + NLTK gutenberg + webtext, last 5% of each doc held out).

    uv run python examples/flagship_edge_lm.py train runs/flagship/best.pt            # BEST config
    uv run python examples/flagship_edge_lm.py train runs/flagship/h.pt --arch hnet   # a candidate
    uv run python examples/flagship_edge_lm.py export runs/flagship/best.pt           # .w8a8.lbm + engine bpb
    uv run python examples/flagship_edge_lm.py bench runs/flagship/best.w8a8.lbm      # CLI: ms/B, CPU, RSS
    uv run python examples/flagship_edge_lm.py table runs/flagship/*.pt               # markdown table
    rust/lilbits_fused/target/release/lilbits runs/flagship/best.w8a8.lbm --prompt "Once upon a time"

Training resumes from the checkpoint if it exists (batches depend only on seed and step).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lilbits.corpus import CorpusBytes
from lilbits.measure import Meter
from lilbits.models.lean_lm import LeanLM
from lilbits.models.space_hnet import SpaceHNet, SpaceHNetConfig, plain_macs_per_byte

EVAL_BATCHES, EVAL_BATCH, EVAL_SEQ = 64, 8, 256
ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "rust" / "lilbits_fused" / "target" / "release" / "lilbits"

# The current best configuration. This example is a living one: when an experiment beats it on CPU / memory per byte
# at equal held-out bpb, update BEST and the table in results/flagship/README.md.
BEST = {
    "arch": "hnet", "d_outer": 64, "d_main": 96, "heads": 4, "n_enc": 1, "n_main": 4, "n_dec": 3, "align": "start",
    "max_patch": 16, "bytes_per_param": 20, "batch": 16, "seq": 256, "lr": 3e-3, "wd": 0.01, "fmt": "w8a8",
}  # fmt: skip


def build(cfg: dict) -> nn.Module:
    if cfg["arch"] == "plain":
        return LeanLM(cfg["d"], cfg["heads"], cfg["layers"], lambda i, o: nn.Linear(i, o, bias=False), n_draft=0)
    keys = SpaceHNetConfig.__dataclass_fields__
    return SpaceHNet(SpaceHNetConfig(**{k: v for k, v in cfg.items() if k in keys}))


def loss_fn(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if isinstance(model, LeanLM):
        return model.loss(x, 0)
    return model.loss(x)


def held_out() -> list[torch.Tensor]:
    src = CorpusBytes("test", seed=999)
    return [src.sample(EVAL_BATCH, EVAL_SEQ) for _ in range(EVAL_BATCHES)]


@torch.no_grad()
def bpb(model: nn.Module, batches) -> tuple[float, float]:
    """(bits per byte, bytes per patch) over the batches."""
    model.eval()
    tot, bpp = 0.0, []
    for x in batches:
        loss, st = loss_fn(model, x)
        tot += loss.item()
        bpp.append(st.get("bytes_per_patch", 1.0))
    return tot / len(batches) / math.log(2), sum(bpp) / len(bpp)


def macs(model: nn.Module, bytes_per_patch: float) -> float:
    return model.macs_per_byte(bytes_per_patch) if isinstance(model, SpaceHNet) else plain_macs_per_byte(model)


def params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def lr_at(step: int, steps: int, lr: float, warm: float = 0.05) -> float:
    n_warm = max(1, int(warm * steps))
    if step < n_warm:
        return lr * (step + 1) / n_warm
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - n_warm) / max(1, steps - n_warm))))


def cmd_train(a) -> None:
    path = Path(a.ckpt)
    torch.set_num_threads(a.threads)
    if path.exists():
        ck = torch.load(path, weights_only=False)
        cfg = ck["cfg"]
        print(f"resuming {path} at step {ck['step']}/{cfg['steps']}")
    else:
        cfg = {k: getattr(a, k) for k in ("arch", "d", "heads", "layers", "steps", "batch", "seq", "lr", "seed")}
        cfg.update(
            d_outer=a.d_outer, d_main=a.d_main, heads_outer=a.heads, heads_main=a.heads, n_enc=a.n_enc,
            n_main=a.n_main, n_dec=a.n_dec, align=a.align, max_patch=a.max_patch, wd=a.wd,
        )  # fmt: skip
        ck = None
    torch.manual_seed(cfg["seed"])
    model = build(cfg)
    if ck is None and a.bytes_per_param > 0:  # steps from a fixed data : weights ratio (bytes counted as tokens)
        cfg["bytes_per_param"] = a.bytes_per_param
        cfg["steps"] = math.ceil(a.bytes_per_param * params(model) / (cfg["batch"] * cfg["seq"]))
    n_bytes = cfg["steps"] * cfg["batch"] * cfg["seq"]
    cfg["train_bytes"], cfg["epochs"] = n_bytes, n_bytes / len(CorpusBytes("train"))
    if cfg["epochs"] > a.max_epochs:
        raise SystemExit(f"{cfg['epochs']:.1f} epochs > --max-epochs {a.max_epochs}: use a smaller model or ratio")
    print(f"{params(model):,} params, {cfg['steps']} steps = {n_bytes / 1e6:.1f} MB = {cfg['epochs']:.2f} epochs, "
          f"{n_bytes / params(model):.1f} bytes / param", flush=True)  # fmt: skip
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg.get("wd", 0.01))
    step, log = 0, []
    if ck is not None:
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        step, log = ck["step"], ck["log"]
    src = CorpusBytes("train", seed=cfg["seed"])
    ev = held_out()[:8]

    def save(final: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        state = {"cfg": cfg, "state": model.state_dict(), "step": step, "log": log}
        if not final:
            state["opt"] = opt.state_dict()
        torch.save(state, tmp)
        tmp.replace(path)

    model.train()
    t0, losses = time.perf_counter(), []
    with Meter() as mt:
        while step < cfg["steps"]:
            src._g.manual_seed(cfg["seed"] * 1_000_003 + step)  # batch depends only on (seed, step): resumable
            x = src.sample(cfg["batch"], cfg["seq"])
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg["steps"], cfg["lr"])
            loss, _ = loss_fn(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % a.log_every == 0 or step == cfg["steps"]:
                e, bpp = bpb(model, ev)
                model.train()
                dt = time.perf_counter() - t0
                row = {"step": step, "train_bpb": sum(losses) / len(losses) / math.log(2), "eval8_bpb": e,
                       "bytes_per_patch": bpp, "s_per_step": dt / len(losses)}  # fmt: skip
                log.append(row)
                print(json.dumps(row), flush=True)
                t0, losses = time.perf_counter(), []
                save()
    cfg["train_meter"] = mt.as_dict()
    save(final=True)
    cmd_eval(argparse.Namespace(ckpt=a.ckpt, threads=a.threads))


def load(path: str) -> tuple[nn.Module, dict]:
    ck = torch.load(path, weights_only=False)
    m = build(ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck


def cmd_eval(a) -> None:
    torch.set_num_threads(a.threads)
    m, ck = load(a.ckpt)
    t = time.perf_counter()
    b, bpp = bpb(m, held_out())
    res = {
        "ckpt": a.ckpt,
        "bpb64": b,
        "bytes_per_patch": bpp,
        "params": params(m),
        "macs_per_byte": macs(m, bpp),
        "steps": ck["step"],
        "eval_s": time.perf_counter() - t,
    }
    out = Path(a.ckpt).with_suffix(".eval.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def describe(cfg: dict) -> str:
    if cfg["arch"] == "plain":
        return f"plain GDN d{cfg['d']} x{cfg['layers']}, {cfg['heads']} heads"
    return (
        f"space H-Net ({cfg['align']}) enc {cfg['n_enc']}x{cfg['d_outer']}, main {cfg['n_main']}x{cfg['d_main']}, "
        f"dec {cfg['n_dec']}x{cfg['d_outer']}"
    )


def runtime(arrays: dict, cfg: dict):
    if cfg["arch"] == "plain":
        from lilbits.inference.fused import FusedLM

        return FusedLM(arrays)
    from lilbits.inference.fused_hnet import FusedHNet

    return FusedHNet(arrays)


def engine_bpb(lm, batches) -> float:
    """Held-out bpb through the Rust engine (each row from a fresh state), same rows as `bpb`."""
    nll, n = 0.0, 0
    for x in batches:
        for row in x.numpy():
            lm.reset()
            lg = lm.logits_all(row[:-1]).astype(np.float64)
            lg -= lg.max(1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
            nll -= lp[np.arange(len(row) - 1), row[1:]].sum()
            n += len(row) - 1
    return nll / n / math.log(2)


def cmd_export(a) -> None:
    """checkpoint -> <ckpt>.<fmt>.npz (Python runtime) and <ckpt>.<fmt>.lbm (CLI bundle); bpb through the engine."""
    from lilbits.inference import bundle, fused, fused_hnet

    m, ck = load(a.ckpt)
    arrays = fused.export(m, a.fmt) if ck["cfg"]["arch"] == "plain" else fused_hnet.export(m, a.fmt)
    base = Path(a.ckpt).with_suffix(f".{a.fmt}")
    np.savez(base.with_suffix(base.suffix + ".npz"), **arrays)
    size = bundle.write(base.with_suffix(base.suffix + ".lbm"), arrays)
    res = {"ckpt": a.ckpt, "fmt": a.fmt, "bundle_bytes": size, "engine_bpb64": engine_bpb(runtime(arrays, ck["cfg"]),
           held_out())}  # fmt: skip
    Path(f"{base}.export.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def ensure_cli() -> Path:
    if not CLI.exists():
        subprocess.run(["cargo", "build", "--release", "--manifest-path", str(CLI.parents[2] / "Cargo.toml")],
                       check=True)  # fmt: skip
    return CLI


def cmd_bench(a) -> None:
    """Fresh CLI processes per bundle (median of --procs), one thread: decode / prefill speed, CPU, RSS."""
    cli = ensure_cli()
    for path in a.bundles:
        runs = []
        for _ in range(a.procs):
            out = subprocess.run([str(cli), path, "--bench", "--gen", str(a.gen), "--rounds", str(a.rounds)],
                                 capture_output=True, text=True, check=True).stdout  # fmt: skip
            runs.append(json.loads(out))
        med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        Path(path).with_suffix(".bench.json").write_text(json.dumps(med, indent=1))
        print(path, json.dumps(med))


def cmd_table(a) -> None:
    print(
        "| model | params | train bytes / param | linear MACs / byte | bytes / patch | held-out bpb (torch) | "
        "W8A8 bpb (engine) | bundle | decode ms / B | decode CPU ms / B | prefill B/s | RSS MiB | train s / step |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in a.ckpts:
        ev = Path(p).with_suffix(".eval.json")
        if not ev.exists():
            continue
        r = json.loads(ev.read_text())
        ck = torch.load(p, weights_only=False)
        sps = sum(x["s_per_step"] for x in ck["log"]) / max(1, len(ck["log"]))
        base = Path(p).with_suffix(".w8a8")
        ex, bn = Path(f"{base}.export.json"), base.with_suffix(".w8a8.bench.json")
        e = json.loads(ex.read_text()) if ex.exists() else {}
        b = json.loads(bn.read_text()) if bn.exists() else {}
        f = lambda d, k, spec: format(d[k], spec) if k in d else ""
        mb = f"{e['bundle_bytes'] / 1e6:.2f} MB" if e else ""
        print(
            f"| {describe(ck['cfg'])} | {r['params'] / 1e6:.2f}M | "
            f"{ck['cfg']['steps'] * ck['cfg']['batch'] * ck['cfg']['seq'] / r['params']:.1f} | "
            f"{r['macs_per_byte'] / 1e6:.2f}M | "
            f"{r['bytes_per_patch']:.2f} | {r['bpb64']:.4f} | {f(e, 'engine_bpb64', '.4f')} | {mb} | "
            f"{f(b, 'decode_ms_per_byte', '.3f')} | {f(b, 'decode_cpu_ms_per_byte', '.3f')} | "
            f"{f(b, 'prefill_bytes_per_s', ',.0f')} | {f(b, 'rss_mib', '.1f')} | {sps:.2f} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("ckpt")
    t.add_argument("--arch", choices=["hnet", "plain"], default="hnet")
    t.add_argument("--d", type=int, default=200, help="plain: width")
    t.add_argument("--layers", type=int, default=8, help="plain: GDN layers")
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--d-outer", type=int, default=128)
    t.add_argument("--d-main", type=int, default=256)
    t.add_argument("--n-enc", type=int, default=2)
    t.add_argument("--n-main", type=int, default=6)
    t.add_argument("--n-dec", type=int, default=2)
    t.add_argument("--align", choices=["end", "start"], default="end")
    t.add_argument("--max-patch", type=int, default=16)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--seq", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--log-every", type=int, default=250)
    t.add_argument("--bytes-per-param", type=float, default=20.0,
                   help="training bytes per parameter (sets --steps); 0: use --steps")  # fmt: skip
    t.add_argument("--max-epochs", type=float, default=4.0, help="refuse runs that repeat the corpus more often")
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--threads", type=int, default=1)
    t.set_defaults(**{k.replace("-", "_"): v for k, v in BEST.items() if k != "fmt"})
    x = sub.add_parser("export")
    x.add_argument("ckpt")
    x.add_argument("--fmt", default=BEST["fmt"], help="f32 | f16 | i8 | i4 | w8a8")
    bn = sub.add_parser("bench")
    bn.add_argument("bundles", nargs="+", help=".lbm files")
    bn.add_argument("--procs", type=int, default=3)
    bn.add_argument("--gen", type=int, default=2048, help="decode bytes (long enough for 10 ms CPU-clock ticks)")
    bn.add_argument("--rounds", type=int, default=5)
    tb = sub.add_parser("table")
    tb.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "export": cmd_export, "bench": cmd_bench, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
