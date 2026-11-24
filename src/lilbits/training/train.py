"""Train TinyLM from scratch with full-precision or LittleBit linears, on Lightning Fabric.

Fabric handles device placement, precision and (multi-)device setup; the CLI is generated from the
`Config` (model) and `TrainConfig` dataclasses by jsonargparse. Examples:

    uv run lb-train --linear both --train.steps 1500
    uv run lb-train --linear littlebit --model.bpw 0.55 --model.d_model 512 --train.precision bf16-mixed
    uv run lb-train --linear littlebit --train.fp_warmup 300 --model.itq_iters 50 --train.optim muon_ds
    uv run lb-train --text input.txt --train.accelerator cuda --train.devices 1
    uv run lb-train --help
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F
from lightning.fabric import Fabric

from lilbits.data import MarkovSource, TextSource
from lilbits.layers.littlebit import LittleBitLinear, zero_margin
from lilbits.models.tinylm import Config, TinyLM, compression_report, convert_fp_to_littlebit
from lilbits.training.optim import DeepSeekMuon


@dataclass
class TrainConfig:
    steps: int = 1500
    batch: int = 32
    lr: float = 3e-3
    warmup: int = 100
    log_every: int = 100
    scale_lr_mult: float = 1.0
    latent_lr_mult: float = 0.1  # LittleBit U/V; full lr makes signs churn when training from scratch
    weight_decay: float = 0.0
    optim: str = "adamw"  # adamw | muon (torch.optim.Muon) | muon_ds (DeepSeek-V4 recipe)
    muon_lr_mult: float = 1.0
    muon_embed: bool = False  # also put embeddings / LM head on Muon
    muon_ns_steps: int = 5
    fp_warmup: int = 0  # train FP for N steps, then SVD(+ITQ)-convert to LittleBit (counts toward steps)
    seed: int = 0
    accelerator: str = "auto"  # Fabric: auto | cpu | cuda | mps
    devices: int = 1
    precision: str = "32-true"  # Fabric: 32-true | bf16-mixed | 16-mixed ...
    verbose: bool = True


def param_groups(model: torch.nn.Module, lr: float, scale_lr_mult: float, weight_decay: float,
                 latent_lr_mult: float = 1.0):
    """Labeled groups. Latent factors / FP weights get weight decay; LittleBit scales and norms don't.

    "hidden" = 2D weights inside the blocks (FP linears), "latent" = LittleBit U/V (also 2D),
    "embed" = embeddings and LM head (2D but conventionally kept off Muon).
    """
    latent, scales, other = [], [], []
    for mod in model.modules():
        if isinstance(mod, LittleBitLinear):
            for p in mod.paths:
                latent += [p.U, p.V]
                scales += [p.g, p.h, p.l]
            if mod.bias is not None:
                other.append(mod.bias)
    seen = {id(p) for p in latent + scales + other}
    hidden, embed, no_decay = [], [], []
    for name, p in model.named_parameters():
        if id(p) not in seen:
            # Only 2D matrices go to "hidden"/"embed" (Muon-eligible); conv kernels (3D) and vectors don't.
            (no_decay if p.dim() != 2 else hidden if name.startswith("blocks.") else embed).append(p)
    groups = [
        {"name": "hidden", "params": hidden, "lr": lr, "weight_decay": weight_decay},
        {"name": "embed", "params": embed, "lr": lr, "weight_decay": weight_decay},
        {"name": "latent", "params": latent, "lr": lr * latent_lr_mult, "weight_decay": weight_decay},
        {"name": "scales", "params": scales, "lr": lr * scale_lr_mult, "weight_decay": 0.0},
        {"name": "other", "params": no_decay + other, "lr": lr, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


def make_optimizers(model, tc: TrainConfig) -> list[torch.optim.Optimizer]:
    """AdamW for everything, or Muon for the 2D block matrices (FP weights and LittleBit latents;
    also embeddings/head if muon_embed) and AdamW for the rest. Muon lrs are RMS-matched to AdamW."""
    groups = param_groups(model, tc.lr, tc.scale_lr_mult, tc.weight_decay, tc.latent_lr_mult)
    if tc.optim == "adamw":
        return [torch.optim.AdamW(groups, betas=(0.9, 0.95))]
    muon_names = {"hidden", "latent"} | ({"embed"} if tc.muon_embed else set())
    muon_groups = [dict(g, lr=g["lr"] * tc.muon_lr_mult) for g in groups if g["name"] in muon_names]
    adam_groups = [g for g in groups if g["name"] not in muon_names]
    if tc.optim == "muon":
        muon = torch.optim.Muon(muon_groups, lr=tc.lr, weight_decay=tc.weight_decay,
                                adjust_lr_fn="match_rms_adamw", ns_steps=tc.muon_ns_steps)
    elif tc.optim == "muon_ds":
        muon = DeepSeekMuon(muon_groups, lr=tc.lr, weight_decay=tc.weight_decay)
    else:
        raise ValueError(f"unknown optim {tc.optim!r}")
    return [muon] + ([torch.optim.AdamW(adam_groups, betas=(0.9, 0.95))] if adam_groups else [])


def lm_loss(model, batch: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())


@torch.no_grad()
def evaluate(model, batches) -> float:
    model.eval()
    loss = sum(lm_loss(model, b).item() for b in batches) / len(batches)
    model.train()
    return loss


def latent_stats(model, prev: dict) -> tuple[float, float, dict]:
    """(fraction of latent signs flipped since `prev`, fraction of latents near zero, current signs)."""
    latents = {n: p.detach() for n, p in model.named_parameters() if n.endswith((".U", ".V"))}
    cur = {n: p >= 0 for n, p in latents.items()}
    common = [n for n in cur if n in prev]
    flips = sum((cur[n] != prev[n]).sum().item() for n in common)
    total = sum(cur[n].numel() for n in common)
    near_zero = sum(zero_margin(p) for p in latents.values()) / max(1, len(latents))
    return flips / max(1, total), near_zero, cur


def _fit(fabric: Fabric, model, label: str, tc: TrainConfig, source, seq_len: int, steps: int, offset: int,
         eval_batches, history: list, t0: float, on_eval=None):
    opts = make_optimizers(model, tc)
    model, *opts = fabric.setup(model, *opts)

    def lr_at(step: int) -> float:  # one warmup + cosine schedule over the whole run (incl. FP warm start)
        if step < tc.warmup:
            return (step + 1) / tc.warmup
        return 0.1 + 0.45 * (1 + math.cos(math.pi * (step - tc.warmup) / max(1, tc.steps - tc.warmup)))

    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda i: lr_at(offset + i)) for o in opts]
    _, _, signs = latent_stats(model, {})
    for i in range(steps):
        step = offset + i
        loss = lm_loss(model, fabric.to_device(source.sample(tc.batch, seq_len)))
        fabric.backward(loss)
        fabric.clip_gradients(model, opts[0], max_norm=1.0)
        for o, s in zip(opts, scheds):
            o.step()
            o.zero_grad(set_to_none=True)
            s.step()
        if step % tc.log_every == 0 or step == tc.steps - 1 or i == steps - 1:
            ev = evaluate(model, eval_batches)
            flip, nz, signs = latent_stats(model, signs)
            history.append({"step": step, "phase": label, "train": loss.item(), "eval": ev, "flip": flip,
                            "near_zero": nz})
            fabric.log_dict({"train_loss": loss.item(), "eval_loss": ev, "sign_flips": flip}, step=step)
            if on_eval is not None:
                on_eval(step, ev)  # e.g. Optuna pruning; may raise
            if tc.verbose:
                fabric.print(f"[{label:9s}] step {step:5d}  train {loss.item():.4f}  eval {ev:.4f}  "
                             f"sign-flips {flip:.4%}  near-zero {nz:.2%}  {time.time() - t0:.0f}s")
    return model.module


def train(cfg: Config, tc: TrainConfig, source, eval_batches=None, fabric: Fabric | None = None,
          on_eval=None) -> dict:
    """Train from scratch. With tc.fp_warmup > 0 (LittleBit only), the first fp_warmup steps train a
    full-precision copy, which is then SVD(+ITQ)-converted, as the paper does with a pretrained model."""
    fabric = fabric or Fabric(accelerator=tc.accelerator, devices=tc.devices, precision=tc.precision)
    fabric.seed_everything(tc.seed, verbose=False)
    eval_batches = eval_batches or [source.sample(tc.batch, cfg.seq_len) for _ in range(4)]
    eval_batches = [fabric.to_device(b) for b in eval_batches]
    history: list = []
    run = dict(tc=tc, source=source, seq_len=cfg.seq_len, eval_batches=eval_batches, history=history, t0=time.time(),
               on_eval=on_eval)
    offset = tc.fp_warmup if (tc.fp_warmup > 0 and cfg.linear == "littlebit") else 0
    if offset:
        with fabric.init_module():
            fp_model = TinyLM(replace(cfg, linear="fp"))
        fp_model = _fit(fabric, fp_model, "fp-warm", steps=offset, offset=0, **run)
        model = convert_fp_to_littlebit(fp_model, cfg)
        ev, (_, nz, _) = evaluate(model, eval_batches), latent_stats(model, {})
        history.append({"step": offset, "phase": "converted", "eval": ev, "near_zero": nz})
        if tc.verbose:
            fabric.print(f"[converted] eval right after SVD{'+ITQ' if cfg.itq_iters else ''} conversion: "
                         f"{ev:.4f}  near-zero {nz:.2%}")
    else:
        with fabric.init_module():
            model = TinyLM(cfg)
    model = _fit(fabric, model, cfg.linear, steps=tc.steps - offset, offset=offset, **run)
    return {"model": model, "history": history, "final_eval": history[-1]["eval"],
            "compression": compression_report(model)}


def run(linear: str = "both", model: Config = Config(), train: TrainConfig = TrainConfig(),
        text: str | None = None, out: str | None = None) -> dict:
    """Train an FP baseline and/or a LittleBit model.

    Args:
        linear: fp | littlebit | bitnet (BitNet b1.58 baseline) | both (fp, then littlebit)
        model: model / LittleBit layer config (vocab_size is taken from the data)
        train: optimization and Fabric settings
        text: char-level text file instead of the synthetic Markov source
        out: write a JSON summary (configs, final losses, histories) here
    """
    tc, fit = train, globals()["train"]
    source = TextSource(text, tc.seed) if text else MarkovSource(vocab=model.vocab_size, seed=tc.seed)
    model = replace(model, vocab_size=source.vocab)
    eval_batches = [source.sample(tc.batch, model.seq_len) for _ in range(4)]
    floor = None if text else sum(source.entropy_floor(b) for b in eval_batches) / len(eval_batches)
    if floor is not None:
        print(f"Markov source: uniform loss {math.log(source.vocab):.4f}, entropy floor ~{floor:.4f} nats")
    summary = {"model": asdict(model), "train": asdict(tc), "entropy_floor": floor, "runs": {}}
    for mode in (["fp", "littlebit"] if linear == "both" else [linear]):
        cfg = replace(model, linear=mode)
        res = fit(cfg, tc, source, eval_batches)
        comp = res["compression"]
        print(f"[{mode}] final eval {res['final_eval']:.4f} | block linears: {comp['block_linear_weights']:,} "
              f"weights at {comp['bpw']:.3f} bpw ({comp['bits'] / 8 / 1024:.1f} KiB)")
        summary["runs"][mode] = {"config": asdict(cfg), "final_eval": res["final_eval"], "compression": comp,
                                 "history": res["history"]}
    if out:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    from jsonargparse import CLI

    CLI(run, args=argv, as_positional=False)


if __name__ == "__main__":
    main()
