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
code works; raise --steps to train. Reports bits per byte, bytes per patch (the compression of the inner
sequence), and for each model train/infer wall, CPU, peak RSS, parameters, checkpoint and stored size.

    uv run python examples/07_byte_level_patching.py
    uv run python examples/07_byte_level_patching.py --latent-linear littlebit   # sub-1-bit inner network
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from lilbits.data import ByteSource
from lilbits.measure import Meter, size_report, state_dict_bytes
from lilbits.models.byte_latent import (
    BLTConfig,
    ByteLatentTransformer,
    HNet,
    HNetConfig,
    byte_entropy,
    entropy_boundaries,
)
from lilbits.models.tinylm import Config, TinyLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bits_per_byte(logits, x):
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).item() / math.log(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30, help="training steps per model (default: smoke run)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--target-bytes-per-patch", type=float, default=4.0)
    ap.add_argument("--latent-linear", default="fp", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    files = sorted(
        glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True)
        + glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True)
    )
    src = ByteSource(files)
    test = ByteSource(files, seed=1).sample(16, args.seq)
    print(f"corpus: {len(src.data):,} bytes from {len(files)} repo files; seq {args.seq} bytes\n")
    T, B = args.seq, args.batch

    def fit(name, model, loss_fn, extra_params=()):
        opt = torch.optim.AdamW(list(model.parameters()) + list(extra_params), lr=2e-3, betas=(0.9, 0.95))
        with Meter() as m:
            for _ in range(args.steps):
                loss = loss_fn(src.sample(B, T)[:, :T])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        return m

    rows: dict[str, dict[str, Any]] = {}

    # 1) byte-level baseline: every byte through every layer
    base = TinyLM(Config(vocab_size=256, seq_len=T, d_model=128, n_heads=4, n_layers=3))
    lm = lambda model: lambda x: F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
    tm = fit("baseline", base, lm(base))
    base.eval()
    with torch.no_grad(), Meter() as im:
        bpb = bits_per_byte(F.pad(base(test[:, : T - 1]), (0, 0, 0, 1)), test[:, :T])
    rows["byte transformer (no patching)"] = dict(model=base, bpb=bpb, bpp=1.0, inner_len=T, train=tm, infer=im)

    # 2) BLT: stage 1 a tiny byte LM for entropies, stage 2 the patched model
    ent_model = TinyLM(Config(vocab_size=256, seq_len=T, d_model=48, n_heads=2, n_layers=1))
    tm_ent = fit("entropy", ent_model, lm(ent_model))
    ent_model.eval()
    with torch.no_grad():
        h = byte_entropy(ent_model, src.sample(64, T)[:, :T])
    theta = torch.quantile(h.flatten(), 1 - 1 / args.target_bytes_per_patch).item()  # ~1 boundary per N bytes

    def patches(x):
        return entropy_boundaries(byte_entropy(ent_model, x), theta_global=theta, max_patch=16)

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
    tm_blt = fit(
        "blt", blt, lambda x: F.cross_entropy(blt(x[:, :-1], patches(x[:, :-1])).flatten(0, 1), x[:, 1:].flatten())
    )
    tm_blt.wall_s += tm_ent.wall_s  # count the entropy model's training too
    tm_blt.cpu_s += tm_ent.cpu_s
    blt.eval()
    with torch.no_grad(), Meter() as im:
        bnd = patches(test[:, :T])
        bpb = bits_per_byte(blt(test[:, :T], bnd), test[:, :T])
    rows["BLT (entropy patches)"] = dict(
        model=blt,
        extra=ent_model,
        bpb=bpb,
        bpp=bnd.numel() / bnd.sum().item(),
        inner_len=int(bnd.sum(1).float().mean()),
        train=tm_blt,
        infer=im,
    )

    # 3) H-Net: chunk boundaries learned end to end
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

    def hnet_loss(x):
        logits, ratio, _ = hnet(x[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()) + hnet.cfg.ratio_weight * ratio

    tm_h = fit("hnet", hnet, hnet_loss)
    hnet.eval()
    with torch.no_grad(), Meter() as im:
        logits, _, st = hnet(test[:, :T])
        bpb = bits_per_byte(logits, test[:, :T])
    rows["H-Net (learned chunks)"] = dict(
        model=hnet, bpb=bpb, bpp=st["bytes_per_chunk"], inner_len=st["chunks"], train=tm_h, infer=im
    )

    print(
        f"| model | bits/byte | bytes per patch | inner sequence (of {T}) | params | train wall / CPU | "
        f"train peak RSS | infer wall (16×{T}) | checkpoint | stored |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    metrics = {}
    for name, r in rows.items():
        mods = [r["model"]] + ([r["extra"]] if "extra" in r else [])
        params = sum(p.numel() for mod in mods for p in mod.parameters())
        ckpt = sum(state_dict_bytes(mod) for mod in mods)
        stored = sum(size_report(mod)["deployed_total_bytes"] for mod in mods)
        print(
            f"| {name} | {r['bpb']:.2f} | {r['bpp']:.1f} | {r['inner_len']} | {params:,} | "
            f"{r['train'].wall_s:.1f} s / {r['train'].cpu_s:.1f} s | {r['train'].peak_extra_mib:.0f} MiB | "
            f"{r['infer'].wall_s * 1e3:.0f} ms | {ckpt / 1024:,.0f} KiB | {stored / 1024:,.0f} KiB |"
        )
        metrics[name] = {
            "bits_per_byte": r["bpb"],
            "bytes_per_patch": r["bpp"],
            "params": params,
            **r["train"].as_dict("train_"),
            **r["infer"].as_dict("infer_"),
            "checkpoint_bytes": ckpt,
            "stored_bytes": stored,
        }
    d = 128
    print(
        f"\nfor scale: a 200k-token vocabulary needs {2 * 200_000 * d:,} embedding + head params at d={d}; "
        f"bytes need {2 * 256 * d:,}."
    )
    print("METRICS " + json.dumps({"train_steps": args.steps, "models": metrics}))


if __name__ == "__main__":
    main()
