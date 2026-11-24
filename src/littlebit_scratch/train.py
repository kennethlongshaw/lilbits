"""Train TinyLM from scratch with full-precision or LittleBit linears.

Examples:
    uv run lb-train --linear both --steps 1500
    uv run lb-train --linear littlebit --bpw 0.55 --device cuda --d-model 512 --n-layers 4
    uv run lb-train --text input.txt --linear both --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace

import torch
import torch.nn.functional as F

from .data import MarkovSource, TextSource
from .littlebit import LittleBitLinear, zero_margin
from .optim import DeepSeekMuon
from .model import Config, TinyLM, compression_report, convert_fp_to_littlebit


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
        if id(p) in seen:
            continue
        if p.dim() < 2:
            no_decay.append(p)
        elif name.startswith("blocks."):
            hidden.append(p)
        else:
            embed.append(p)
    groups = [
        {"name": "hidden", "params": hidden, "lr": lr, "weight_decay": weight_decay},
        {"name": "embed", "params": embed, "lr": lr, "weight_decay": weight_decay},
        {"name": "latent", "params": latent, "lr": lr * latent_lr_mult, "weight_decay": weight_decay},
        {"name": "scales", "params": scales, "lr": lr * scale_lr_mult, "weight_decay": 0.0},
        {"name": "other", "params": no_decay + other, "lr": lr, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


def make_optimizers(model, lr: float, scale_lr_mult: float, weight_decay: float, latent_lr_mult: float,
                    optim: str = "adamw", muon_lr_mult: float = 1.0, muon_embed: bool = False,
                    muon_ns_steps: int = 5):
    """AdamW for everything, or torch.optim.Muon for the 2D block matrices (FP weights and
    LittleBit latents; also embeddings/head if muon_embed) and AdamW for the rest.

    Muon uses adjust_lr_fn="match_rms_adamw", so its lr is on the same scale as AdamW's.
    """
    groups = param_groups(model, lr, scale_lr_mult, weight_decay, latent_lr_mult)
    if optim == "adamw":
        return [torch.optim.AdamW(groups, betas=(0.9, 0.95))]
    if optim not in ("muon", "muon_ds"):
        raise ValueError(f"unknown optim {optim!r}")
    muon_names = {"hidden", "latent"} | ({"embed"} if muon_embed else set())
    muon_groups = [dict(g, lr=g["lr"] * muon_lr_mult) for g in groups if g["name"] in muon_names]
    adam_groups = [g for g in groups if g["name"] not in muon_names]
    if optim == "muon":
        opts = [torch.optim.Muon(muon_groups, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw",
                                 ns_steps=muon_ns_steps)]
    else:
        opts = [DeepSeekMuon(muon_groups, lr=lr, weight_decay=weight_decay)]
    if adam_groups:
        opts.append(torch.optim.AdamW(adam_groups, betas=(0.9, 0.95)))
    return opts


@torch.no_grad()
def evaluate(model, batches) -> float:
    model.eval()
    losses = [F.cross_entropy(model(b[:, :-1]).flatten(0, 1), b[:, 1:].flatten()).item() for b in batches]
    model.train()
    return sum(losses) / len(losses)


def sign_flip_rate(model, prev: dict) -> tuple[float, dict]:
    flips = total = 0
    cur = {}
    for name, p in model.named_parameters():
        if name.endswith(".U") or name.endswith(".V"):
            s = p.detach() >= 0
            cur[name] = s
            if name in prev:
                flips += (s != prev[name]).sum().item()
                total += s.numel()
    return (flips / total if total else 0.0), cur


def latent_near_zero(model) -> float:
    """Mean fraction of latent entries within 10% of their rms from zero."""
    vals = [zero_margin(p.detach()) for n, p in model.named_parameters() if n.endswith((".U", ".V"))]
    return sum(vals) / len(vals) if vals else 0.0


def _fit(model, label: str, source, steps: int, total_steps: int, step_offset: int, batch: int, seq_len: int,
         lr: float, device: str, log_every: int, scale_lr_mult: float, latent_lr_mult: float,
         weight_decay: float, warmup: int, eval_batches, verbose: bool, history: list, t0: float,
         optim: str = "adamw", muon_lr_mult: float = 1.0, muon_embed: bool = False, muon_ns_steps: int = 5):
    opts = make_optimizers(model, lr, scale_lr_mult, weight_decay, latent_lr_mult, optim, muon_lr_mult, muon_embed,
                           muon_ns_steps)
    all_groups = [g for o in opts for g in o.param_groups]
    base_lrs = [g["lr"] for g in all_groups]

    def lr_at(step: int) -> float:  # one schedule over the whole run, including any FP warm start
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    prev_signs: dict = {}
    _, prev_signs = sign_flip_rate(model, prev_signs)
    for i in range(steps):
        step = step_offset + i
        for g, base in zip(all_groups, base_lrs):
            g["lr"] = base * lr_at(step)
        xb = source.sample(batch, seq_len).to(device)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), xb[:, 1:].flatten())
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        if step % log_every == 0 or step == total_steps - 1 or i == steps - 1:
            ev = evaluate(model, eval_batches)
            flip, prev_signs = sign_flip_rate(model, prev_signs)
            nz = latent_near_zero(model)
            history.append({"step": step, "phase": label, "train": loss.item(), "eval": ev, "flip": flip,
                            "near_zero": nz})
            if verbose:
                print(f"[{label:9s}] step {step:5d}  train {loss.item():.4f}  eval {ev:.4f}  "
                      f"sign-flips {flip:.4%}  near-zero {nz:.2%}  {time.time() - t0:.0f}s", flush=True)


def train(cfg: Config, source, steps: int, batch: int, lr: float, device: str, log_every: int = 100,
          scale_lr_mult: float = 1.0, latent_lr_mult: float = 1.0, weight_decay: float = 0.0, warmup: int = 100,
          eval_batches=None, seed: int = 0, verbose: bool = True, fp_warmup: int = 0,
          optim: str = "adamw", muon_lr_mult: float = 1.0, muon_embed: bool = False,
          muon_ns_steps: int = 5) -> dict:
    """Train from scratch. With fp_warmup > 0 (LittleBit only), the first fp_warmup steps train
    a full-precision copy, which is then SVD(+ITQ)-converted, as the paper does with a pretrained model."""
    torch.manual_seed(seed)
    eval_batches = [b.to(device) for b in (eval_batches or [source.sample(batch, cfg.seq_len) for _ in range(4)])]
    history: list = []
    t0 = time.time()
    common = dict(source=source, total_steps=steps, batch=batch, seq_len=cfg.seq_len, lr=lr, device=device,
                  log_every=log_every, scale_lr_mult=scale_lr_mult, latent_lr_mult=latent_lr_mult,
                  weight_decay=weight_decay, warmup=warmup, eval_batches=eval_batches, verbose=verbose,
                  history=history, t0=t0, optim=optim, muon_lr_mult=muon_lr_mult, muon_embed=muon_embed,
                  muon_ns_steps=muon_ns_steps)
    offset = 0
    if fp_warmup > 0 and cfg.linear == "littlebit":
        fp_model = TinyLM(replace(cfg, linear="fp")).to(device)
        _fit(fp_model, "fp-warm", steps=fp_warmup, step_offset=0, **common)
        model = convert_fp_to_littlebit(fp_model, cfg)
        ev = evaluate(model, eval_batches)
        history.append({"step": fp_warmup, "phase": "converted", "eval": ev, "near_zero": latent_near_zero(model)})
        if verbose:
            print(f"[converted] eval right after SVD{'+ITQ' if cfg.itq_iters else ''} conversion: {ev:.4f}  "
                  f"near-zero {latent_near_zero(model):.2%}", flush=True)
        offset = fp_warmup
    else:
        model = TinyLM(cfg).to(device)
    _fit(model, cfg.linear, steps=steps - offset, step_offset=offset, **common)
    return {"model": model, "history": history, "final_eval": history[-1]["eval"],
            "compression": compression_report(model)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--linear", choices=["fp", "littlebit", "both"], default="both")
    ap.add_argument("--bpw", type=float, default=1.0, help="target bits/weight for LittleBit linears")
    ap.add_argument("--no-residual", action="store_true", help="single path instead of residual factorization")
    ap.add_argument("--ste", choices=["smooth", "clipped"], default="clipped")
    ap.add_argument("--smooth-k", type=float, default=100.0)
    ap.add_argument("--init", choices=["small", "bimodal", "svd"], default="small",
                    help="LittleBit latent init (svd = Dual-SVID of a random dense init)")
    ap.add_argument("--itq", type=int, default=0, metavar="ITERS",
                    help="Joint-ITQ rotation iterations before binarizing (LittleBit-2); used by svd init / FP warm start")
    ap.add_argument("--fp-warmup", type=int, default=0, metavar="STEPS",
                    help="train full precision for STEPS, then SVD-convert to LittleBit (counts toward --steps)")
    ap.add_argument("--optim", choices=["adamw", "muon", "muon_ds"], default="adamw",
                    help="muon = torch.optim.Muon, muon_ds = DeepSeek-V4 recipe (10-step hybrid Newton-Schulz, "
                         "update RMS 0.18); both on 2D block matrices (incl. LittleBit latents), AdamW for the rest")
    ap.add_argument("--muon-ns-steps", type=int, default=5, help="Newton-Schulz steps for --optim muon")
    ap.add_argument("--qk-norm", action="store_true", help="RMSNorm on per-head queries and keys")
    ap.add_argument("--muon-lr-mult", type=float, default=1.0, help="extra lr multiplier for Muon groups")
    ap.add_argument("--muon-embed", action="store_true", help="also put embeddings/LM head on Muon")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--scale-lr-mult", type=float, default=1.0)
    ap.add_argument("--latent-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--text", help="char-level text file instead of the synthetic Markov source")
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write a JSON summary here")
    args = ap.parse_args(argv)

    if args.text:
        source, floor = TextSource(args.text, args.seed), None
        vocab = source.vocab
    else:
        source = MarkovSource(vocab=args.vocab, seed=args.seed)
        vocab = args.vocab
    eval_batches = [source.sample(args.batch, args.seq_len) for _ in range(4)]
    if not args.text:
        floor = sum(source.entropy_floor(b) for b in eval_batches) / len(eval_batches)
        print(f"Markov source: uniform loss {math.log(vocab):.4f}, entropy floor ~{floor:.4f} nats")

    modes = ["fp", "littlebit"] if args.linear == "both" else [args.linear]
    summary = {"args": vars(args), "entropy_floor": floor, "runs": {}}
    for mode in modes:
        cfg = Config(vocab_size=vocab, seq_len=args.seq_len, d_model=args.d_model, n_heads=args.n_heads,
                     n_layers=args.n_layers, linear=mode, bpw=args.bpw, residual=not args.no_residual,
                     ste=args.ste, smooth_k=args.smooth_k, init=args.init, itq_iters=args.itq, qk_norm=args.qk_norm)
        res = train(cfg, source, args.steps, args.batch, args.lr, args.device,
                    scale_lr_mult=args.scale_lr_mult, latent_lr_mult=args.latent_lr_mult, weight_decay=args.weight_decay,
                    eval_batches=eval_batches, seed=args.seed, fp_warmup=args.fp_warmup,
                    optim=args.optim, muon_lr_mult=args.muon_lr_mult, muon_embed=args.muon_embed,
                    muon_ns_steps=args.muon_ns_steps)
        comp = res["compression"]
        print(f"[{mode}] final eval {res['final_eval']:.4f} | block linears: {comp['block_linear_weights']:,} "
              f"weights at {comp['bpw']:.3f} bpw ({comp['bits'] / 8 / 1024:.1f} KiB)")
        summary["runs"][mode] = {"config": asdict(cfg), "final_eval": res["final_eval"],
                                 "compression": comp, "history": res["history"]}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
