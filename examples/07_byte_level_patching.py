"""Tokenizer-free byte-level LMs with dynamic patching: BLT (entropy patches) and H-Net (learned chunks).

Instead of a tokenizer with a ~200k-entry vocabulary (an embedding + LM head of 2 x 200k x d parameters),
these models read raw bytes (vocab 256) and group them into variable-length patches, so the big inner network
runs on a sequence several times shorter than the byte stream.

    BLT   (Pagnoni et al. 2024, https://huggingface.co/papers/2412.09871): a small byte LM's next-byte entropy
          decides patch starts; local encoder -> latent transformer over patches -> local decoder.
    H-Net (Hwang, Wang & Gu 2025): a router learns chunk boundaries end to end (cosine-similarity boundaries,
          ratio loss toward a target bytes-per-chunk); Mamba2 encoder/decoder around a transformer over chunks.
    Baseline: a plain byte-level transformer of similar width (every byte goes through the whole network).

The corpus is this repo's own Markdown/Python files (no download). Default is a short smoke run that proves the
code works; raise --steps to train.

    train   the byte baseline, BLT's entropy model then BLT, and H-Net, each on Lightning Fabric (per-model cost)
    test    bits per byte, bytes per patch (how much shorter the inner sequence is), inner sequence length
    infer   forward latency for 16 x seq bytes per model, including BLT's entropy patching pass

    uv run python examples/07_byte_level_patching.py
    uv run python examples/07_byte_level_patching.py --latent-linear littlebit   # sub-1-bit inner network
"""

from __future__ import annotations

import glob
import math
import os
import time

import torch
import torch.nn.functional as F

from lilbits.data import ByteSource
from lilbits.measure import Meter
from lilbits.models.byte_latent import (
    BLTConfig,
    ByteLatentTransformer,
    HNet,
    HNetConfig,
    byte_entropy,
    entropy_boundaries,
)
from lilbits.models.tinylm import Config, TinyLM
from lilbits.runner import Example, fit, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bits_per_byte(logits, x):
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item() / math.log(2)


def add_args(ap):
    ap.add_argument("--steps", type=int, default=30, help="training steps per model (default: smoke run)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--target-bytes-per-patch", type=float, default=4.0)
    ap.add_argument("--latent-linear", default="fp", choices=["fp", "littlebit", "bitnet"])


def files() -> list[str]:
    return sorted(
        glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True)
        + glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True)
    )


def build(args, meta):
    T = args.seq
    blt = ByteLatentTransformer(
        BLTConfig(
            seq_len=T,
            d_local=64,
            d_latent=128,
            heads=4,
            n_enc=1,
            n_latent=2,
            n_dec=1,
            max_patches=T,
            linear=args.latent_linear,
        )
    )
    blt.theta = meta.get("theta", 0.0)  # entropy threshold, fixed after the entropy model is trained
    hnet = HNet(
        HNetConfig(
            seq_len=T,
            d_outer=64,
            d_main=128,
            heads=4,
            n_main=2,
            target_ratio=args.target_bytes_per_patch,
            linear=args.latent_linear,
        )
    )
    return {
        "baseline": TinyLM(Config(vocab_size=256, seq_len=T, d_model=128, n_heads=4, n_layers=3)),
        "entropy": TinyLM(Config(vocab_size=256, seq_len=T, d_model=48, n_heads=2, n_layers=1)),
        "blt": blt,
        "hnet": hnet,
    }


def lm_loss(model, x):
    return F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten()), {}


def patches(models, x):
    return entropy_boundaries(byte_entropy(models["entropy"], x), theta_global=models["blt"].theta, max_patch=16)


def train_fn(models, args, fab):
    src = ByteSource(files())
    T, B = args.seq, args.batch
    batch = lambda _: src.sample(B, T)[:, :T]
    out = {}
    for name in ("baseline", "entropy"):
        with Meter() as m:
            fit(fab, models[name], batch, lm_loss, args.steps, lr=2e-3)
        out[name] = m.as_dict()
    with torch.no_grad():
        h = byte_entropy(models["entropy"].eval(), src.sample(64, T)[:, :T])
    theta = torch.quantile(h.flatten(), 1 - 1 / args.target_bytes_per_patch).item()  # ~1 boundary per N bytes
    models["blt"].theta = theta

    def blt_loss(model, x):
        with torch.no_grad():
            bnd = patches(models, x[:, :-1])
        return F.cross_entropy(model(x[:, :-1], bnd).flatten(0, 1), x[:, 1:].flatten()), {}

    def hnet_loss(model, x):
        logits, ratio, _ = model(x[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()) + models["hnet"].cfg.ratio_weight * ratio, {}

    for name, fn in (("blt", blt_loss), ("hnet", hnet_loss)):
        with Meter() as m:
            fit(fab, models[name], batch, fn, args.steps, lr=2e-3)
        out[name] = m.as_dict()
    return {"steps_per_model": args.steps, "per_model": out}, {"theta": theta}


def test_batch(args) -> torch.Tensor:
    return ByteSource(files(), seed=1).sample(16, args.seq)[:, : args.seq]


def test_fn(models, args):
    x, T = test_batch(args), args.seq
    out = {}
    out["byte transformer"] = {
        "bits_per_byte": bits_per_byte(F.pad(models["baseline"](x[:, : T - 1]), (0, 0, 0, 1)), x),
        "bytes_per_patch": 1.0,
        "inner_len": T,
    }
    bnd = patches(models, x)
    out["BLT"] = {
        "bits_per_byte": bits_per_byte(models["blt"](x, bnd), x),
        "bytes_per_patch": bnd.numel() / bnd.sum().item(),
        "inner_len": bnd.sum(1).float().mean().item(),
    }
    logits, _, st = models["hnet"](x)
    out["H-Net"] = {
        "bits_per_byte": bits_per_byte(logits, x),
        "bytes_per_patch": st["bytes_per_chunk"],
        "inner_len": st["chunks"],
    }
    return out


def infer_fn(models, args):
    x, T = test_batch(args), args.seq

    def ms(f):
        f()
        t = time.perf_counter()
        for _ in range(3):
            f()
        return (time.perf_counter() - t) / 3 * 1e3

    return {
        "ms_per_16xseq": {
            "byte transformer": ms(lambda: models["baseline"](x[:, : T - 1])),
            "BLT (incl. entropy patching)": ms(lambda: models["blt"](x, patches(models, x))),
            "H-Net": ms(lambda: models["hnet"](x)),
        },
        "embedding_params_bytes_vocab": 256 * 128 * 2,
        "embedding_params_200k_vocab_at_d128": 2 * 200_000 * 128,
    }


EXAMPLE = Example("07_byte_level_patching", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn)

if __name__ == "__main__":
    run(EXAMPLE)
