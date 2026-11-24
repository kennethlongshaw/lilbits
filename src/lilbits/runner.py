"""One train / test / infer harness for every example, on Lightning Fabric.

Each example declares four functions and hands them to `run`:

    build(args, meta)        -> {name: nn.Module}   fresh models in their final (loadable) structure
    train(models, args, fab) -> (metrics, meta)     may replace entries in `models`; meta says how to rebuild them
    test(models, args)       -> metrics             held-out quality
    infer(models, args)      -> metrics             deployment: packing, decode / latency, cache sizes

and gets the same command line everywhere:

    --mode all     train, save the checkpoint, test, infer (default)
    --mode train   train and save the checkpoint
    --mode test    load the checkpoint, test
    --mode infer   load the checkpoint, infer
    --ckpt PATH    checkpoint path (default results/ckpt/<example>.pt)
    --accelerator / --devices / --precision   passed to Lightning Fabric

Training goes through Lightning Fabric: `fab` is a launched `Fabric`, examples call `fab.setup(model, opt)` and
`fab.backward(loss)` (or `lilbits.training.train.train`, which uses Fabric too), and checkpoints are written and
read with `fab.save` / `fab.load`. Test and infer run on the same Fabric device.

Every phase is wrapped in a `Meter` (wall, CPU, peak RSS growth), and the harness adds sizes for each deployed model
(params, weight RAM, deployed bytes) plus the checkpoint file size. The last line is `METRICS {json}` with sections
"train", "test", "infer", "size"; `benchmarks/examples_bench.py` reads it. A checkpoint stores the arguments that
shaped the models, so test and infer runs rebuild the same models; only the arguments listed in `runtime_args`
(and threads) can change after training.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.fabric import Fabric
from torch import nn

from lilbits.measure import Meter, size_report, tensor_bytes

Models = dict[str, nn.Module]


@dataclass
class Example:
    name: str
    description: str
    add_args: Callable[[argparse.ArgumentParser], None]
    build: Callable[[argparse.Namespace, dict], Models]
    train: Callable[[Models, argparse.Namespace, Fabric], tuple[dict, dict]]
    test: Callable[[Models, argparse.Namespace], dict]
    infer: Callable[[Models, argparse.Namespace], dict]
    deploy: Callable[[Models], Models] | None = None  # which modules are deployed (default: all)
    runtime_args: tuple[str, ...] = ()  # arguments that may differ between training and test / infer runs


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


def _print_section(title: str, metrics: dict) -> None:
    print(f"\n== {title} ==")
    for k, v in metrics.items():
        if k != "table":
            print(f"  {k:32s} {_fmt(v)}")


def size_metrics(models: Models, ckpt: Path | None) -> dict:
    out: dict = {}
    for name, m in models.items():
        rep = size_report(m)
        out[name] = {
            "params": rep["trainable_params"],
            "weight_ram_bytes": tensor_bytes([*m.parameters(), *m.buffers()]),
            "deployed_bytes": rep["deployed_total_bytes"],
        }
    if ckpt is not None and ckpt.exists():
        out["checkpoint_file_bytes"] = os.path.getsize(ckpt)
    return out


def save(fab: Fabric, path: Path, ex: Example, args: argparse.Namespace, meta: dict, models: Models) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in vars(args).items() if k not in ("mode", "ckpt")}
    state: dict[str, Any] = {"example": ex.name, "args": keep, "meta": meta}
    state.update({f"model/{k}": m for k, m in models.items()})  # Fabric stores each module's state_dict
    fab.save(path, state)


def load(fab: Fabric, path: Path, ex: Example, args: argparse.Namespace) -> tuple[Models, argparse.Namespace]:
    ck = fab.load(path)
    assert ck["example"] == ex.name, f"{path} is a checkpoint of {ck['example']!r}, not {ex.name!r}"
    runtime = {"mode", "ckpt", "threads", "accelerator", "devices", "precision", *ex.runtime_args}
    merged = {**ck["args"], **{k: v for k, v in vars(args).items() if k in runtime}}
    args = argparse.Namespace(**merged)
    with fab.init_module():
        models = ex.build(args, ck["meta"])
    for k, m in models.items():
        m.load_state_dict(ck[f"model/{k}"])
        m.eval()
    return models, args


def run(ex: Example, argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=ex.description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="all", choices=["all", "train", "test", "infer"])
    ap.add_argument("--ckpt", default=None, help=f"checkpoint path (default results/ckpt/{ex.name}.pt)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accelerator", default="cpu", help="Fabric accelerator: cpu | cuda | mps | auto")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="32-true", help="Fabric precision, e.g. 32-true | bf16-mixed")
    ex.add_args(ap)
    args = ap.parse_args(argv)
    ckpt = Path(args.ckpt or f"results/ckpt/{ex.name}.pt")
    torch.set_num_threads(args.threads)
    fab = Fabric(accelerator=args.accelerator, devices=args.devices, precision=args.precision)
    fab.launch()
    fab.seed_everything(args.seed, verbose=False)
    out: dict = {"example": ex.name, "mode": args.mode}

    if args.mode in ("all", "train"):
        with fab.init_module():
            models = ex.build(args, {})
        for m in models.values():
            m.train()
        with Meter() as m:
            metrics, meta = ex.train(models, args, fab)
        out["train"] = {**metrics, **m.as_dict()}
        save(fab, ckpt, ex, args, meta, models)
        _print_section("train", out["train"])
        for mod in models.values():
            mod.eval()
    else:
        models, args = load(fab, ckpt, ex, args)

    for phase, fn in (("test", ex.test), ("infer", ex.infer)):
        if args.mode in ("all", phase):
            fab.seed_everything(args.seed + 1, verbose=False)
            with torch.no_grad(), Meter() as m:
                metrics = fn(models, args)
            out[phase] = {**metrics, **m.as_dict()}
            _print_section(phase, out[phase])
    deployed = ex.deploy(models) if ex.deploy else models
    out["size"] = size_metrics(deployed, ckpt)
    _print_section("size", out["size"])
    print("METRICS " + json.dumps(out, default=float))
    return out


def fit(
    fab: Fabric,
    model: nn.Module,
    batch_fn: Callable[[int], Any],
    loss_fn: Callable[[nn.Module, Any], tuple[torch.Tensor, dict]],
    steps: int,
    lr: float = 3e-3,
    optimizer: torch.optim.Optimizer | None = None,
    clip: float | None = 1.0,
    after_step: Callable[[], None] | None = None,
) -> dict:
    """Standard Fabric training loop: setup, forward, fab.backward, clip, step. Returns the last step's info."""
    opt = optimizer or torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.95))
    wrapped, opt = fab.setup(model, opt)
    info: dict = {}
    loss = torch.zeros(())
    for step in range(steps):
        loss, info = loss_fn(wrapped, fab.to_device(batch_fn(step)))
        opt.zero_grad(set_to_none=True)
        fab.backward(loss)
        if clip:
            fab.clip_gradients(wrapped, opt, max_norm=clip)
        opt.step()
        if after_step is not None:
            after_step()
    return {"steps": steps, "last_loss": loss.item(), **info}
