"""Tokenizer-free byte-level LMs with dynamic patching: BLT (entropy or space patches) and H-Net (learned chunks).

Instead of a tokenizer with a ~200k-entry vocabulary (an embedding + LM head of 2 x 200k x d parameters),
these models read raw bytes (vocab 256) and group them into variable-length patches, so the big inner network
runs on a sequence several times shorter than the byte stream, once per patch at decode.

    BLT   (Pagnoni et al. 2024, arXiv:2412.09871): local encoder -> latent transformer over patches -> local
          decoder, hash n-gram byte embeddings. Two patchers:
            entropy  a small byte LM's next-byte entropy decides patch starts (the published BLT recipe). At decode
                     that byte LM is one more sequential network call per byte.
            space    SpaceByte (Slagle 2024, arXiv:2404.14408): a patch starts after a run of space-like bytes
                     (not a letter / digit / UTF-8 continuation), capped at --max-patch bytes. No patcher model.
    H-Net (Hwang, Wang & Gu 2025, arXiv:2507.07955): a router learns chunk boundaries end to end; Mamba2
          encoder/decoder around a transformer over chunks.
    Baseline: a plain byte-level transformer of similar width (every byte goes through the whole network).

The corpus is this repo's own Markdown/Python files (no download): first 90% train, last 10% test. Default is a
short smoke run; raise --steps to train.

    train   byte baseline, BLT's entropy model, BLT with entropy patches, BLT with space patches, H-Net
    test    held-out bits per byte, bytes per patch, inner sequence length
    infer   batch-1 greedy decode with every network cached (models/byte_latent.py: BLTStream, HNetStream,
            LMStream; each matches its parallel forward): bytes/s, linear MACs per byte, inner steps per byte;
            and parallel prefill latency

At 800 steps (results/byte_patching.md), space patches beat entropy patches on held-out bits/byte (3.416 vs 3.428)
with 6.6 instead of 3.0 bytes per patch, so half the latent steps, no entropy model, and 1.6-2.4x the decode speed.

    uv run python examples/07_byte_level_patching.py
    uv run python examples/07_byte_level_patching.py --steps 800
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
    BLTStream,
    ByteLatentTransformer,
    HNet,
    HNetConfig,
    HNetStream,
    LMStream,
    byte_entropy,
    entropy_boundaries,
    greedy_generate,
    space_boundaries,
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
    ap.add_argument("--max-patch", type=int, default=16, help="cap on bytes per patch (both BLT patchers)")
    ap.add_argument("--decode-bytes", type=int, default=192, help="greedy bytes generated per prompt in infer")


def files() -> list[str]:
    return sorted(
        glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True)
        + glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True)
    )


def corpus(split: str) -> ByteSource:
    """Train on the first 90% of the corpus bytes, test on the last 10% (held out)."""
    data = b"".join(open(p, "rb").read() for p in files())
    cut = int(len(data) * 0.9)
    return ByteSource(data[:cut], seed=0) if split == "train" else ByteSource(data[cut:], seed=1)


def build(args, meta):
    T = args.seq
    blt = lambda: ByteLatentTransformer(
        BLTConfig(
            seq_len=T,
            d_local=64,
            d_latent=128,
            heads=4,
            n_enc=1,
            n_latent=2,
            n_dec=1,
            max_patches=T + 1,
            linear=args.latent_linear,
        )
    )
    blt_entropy, blt_space = blt(), blt()
    blt_entropy.theta = meta.get("theta", 0.0)  # entropy threshold, fixed after the entropy model is trained
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
        "blt_entropy": blt_entropy,
        "blt_space": blt_space,
        "hnet": hnet,
    }


def lm_loss(model, x):
    return F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten()), {}


def patches(models, x, kind, max_patch):
    """Patch starts for bytes x (B, T): column t says whether byte t opens a patch. Both patchers decide column t
    from bytes < t, which is what lets decode close a patch (and run the latent) before sampling the next byte."""
    if kind == "space":
        return space_boundaries(x, max_patch)
    h = byte_entropy(models["entropy"], x)
    return entropy_boundaries(h, theta_global=models["blt_entropy"].theta, max_patch=max_patch)


def train_fn(models, args, fab):
    src = corpus("train")
    T, B = args.seq, args.batch
    batch = lambda _: src.sample(B, T)[:, :T]
    out = {}
    for name in ("baseline", "entropy"):
        with Meter() as m:
            fit(fab, models[name], batch, lm_loss, args.steps, lr=2e-3)
        out[name] = m.as_dict()
    with torch.no_grad():
        h = byte_entropy(models["entropy"].eval(), src.sample(64, T)[:, :T])
    # ~1 boundary per N bytes; set before the space patcher's rate is known, so it is matched to N, not to spaces
    theta = torch.quantile(h.flatten(), 1 - 1 / args.target_bytes_per_patch).item()
    models["blt_entropy"].theta = theta

    def blt_loss(kind):
        def f(model, x):
            with torch.no_grad():
                bnd = patches(models, x, kind, args.max_patch)
            return F.cross_entropy(model(x[:, :-1], bnd).flatten(0, 1), x[:, 1:].flatten()), {}

        return f

    def hnet_loss(model, x):
        logits, ratio, _ = model(x[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()) + models["hnet"].cfg.ratio_weight * ratio, {}

    for name, fn in (("blt_entropy", blt_loss("entropy")), ("blt_space", blt_loss("space")), ("hnet", hnet_loss)):
        with Meter() as m:
            fit(fab, models[name], batch, fn, args.steps, lr=2e-3)
        out[name] = m.as_dict()
    return {"steps_per_model": args.steps, "per_model": out}, {"theta": theta}


def test_batch(args) -> torch.Tensor:
    return corpus("test").sample(32, args.seq)  # (B, seq + 1): the last byte is the target of the last position


def test_fn(models, args):
    x, T = test_batch(args), args.seq
    xi, y = x[:, :T], x[:, 1:]
    bpb = lambda logits: F.cross_entropy(logits.flatten(0, 1), y.flatten()).item() / math.log(2)
    out = {"byte transformer": {"bits_per_byte": bpb(models["baseline"](xi)), "bytes_per_patch": 1.0, "inner_len": T}}
    for name, kind in (("BLT entropy patches", "entropy"), ("BLT space patches", "space")):
        bnd = patches(models, x, kind, args.max_patch)
        b = bnd[:, :T]
        out[name] = {
            "bits_per_byte": bpb(models["blt_" + kind](xi, bnd)),
            "bytes_per_patch": b.numel() / b.sum().item(),
            "inner_len": b.sum(1).float().mean().item(),
        }
    logits, _, st = models["hnet"](xi)
    out["H-Net"] = {"bits_per_byte": bpb(logits), "bytes_per_patch": st["bytes_per_chunk"], "inner_len": st["chunks"]}
    return out


def _macs(*mods) -> int:
    """Multiply-accumulates per call of these modules at one position: every 2-D weight except embeddings."""
    seen, n = set(), 0
    for mod in mods:
        for sub in mod.modules():
            if isinstance(sub, torch.nn.Embedding):
                seen.update(id(p) for p in sub.parameters())
        for p in mod.parameters():
            if p.dim() == 2 and id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
    return n


def decode_bench(models, args) -> dict:
    """Batch-1 greedy decode with every network KV-cached / state-carried (see *Stream in models/byte_latent.py).
    Prompt 64 held-out bytes, then --decode-bytes new bytes, 4 prompts. Reports bytes/s, sequential network calls
    per generated byte, and linear-layer multiply-accumulates per byte (attention scores not counted)."""
    prompts = corpus("test").sample(4, 64)[:, :64].tolist()
    n_new = min(args.decode_bytes, args.seq - 64)
    blt_e, blt_s, hn, base, ent = (models[k] for k in ("blt_entropy", "blt_space", "hnet", "baseline", "entropy"))
    makers = {
        "byte transformer": lambda: LMStream(base),
        "BLT entropy patches": lambda: BLTStream(blt_e, "entropy", ent, blt_e.theta, args.max_patch),
        "BLT space patches": lambda: BLTStream(blt_s, "space", max_patch=args.max_patch),
        "H-Net": lambda: HNetStream(hn),
    }

    def macs(name, calls, n):  # calls are over prompt + generated bytes
        if name == "byte transformer":
            return _macs(base)
        if name.startswith("BLT"):
            m = blt_e if "entropy" in name else blt_s
            per_byte = _macs(m.encoder, m.decoder, m.from_latent, m.head)
            per_patch = _macs(m.pool, m.latent)
            return per_byte + per_patch * calls["latent"] / n + (_macs(ent) if "entropy" in name else 0)
        per_byte = _macs(hn.encoder, hn.decoder, hn.residual, hn.head, hn.chunker)
        return per_byte + _macs(hn.up_in, hn.main, hn.down_out) * calls["main"] / n

    out = {}
    for name, make in makers.items():
        for mod in models.values():
            mod.eval()
        secs, total, calls = 0.0, 0, {}
        for p in prompts:
            st = make()
            _, s = greedy_generate(st, p, n_new)
            secs, total = secs + s, total + st.t
            calls = {k: calls.get(k, 0) + v for k, v in st.calls.items()}
        n_gen = n_new * len(prompts)
        out[name] = {
            "decode_bytes_per_s": n_gen / secs,
            "ms_per_byte": secs / n_gen * 1e3,
            "linear_macs_per_byte": macs(name, calls, total),
            "inner_steps_per_byte": next(calls[k] for k in ("latent", "main", "layers") if k in calls) / total,
        }
    return out


def infer_fn(models, args):
    x, T = test_batch(args)[:16], args.seq

    def ms(f):
        f()
        t = time.perf_counter()
        for _ in range(3):
            f()
        return (time.perf_counter() - t) / 3 * 1e3

    with torch.no_grad():
        prefill = {
            "byte transformer": ms(lambda: models["baseline"](x[:, :T])),
            "BLT entropy patches (incl. patcher)": ms(
                lambda: models["blt_entropy"](x[:, :T], patches(models, x, "entropy", args.max_patch))
            ),
            "BLT space patches (incl. patcher)": ms(
                lambda: models["blt_space"](x[:, :T], patches(models, x, "space", args.max_patch))
            ),
            "H-Net": ms(lambda: models["hnet"](x[:, :T])),
        }
    return {
        "decode_batch1_greedy": decode_bench(models, args),
        "prefill_ms_per_16xseq": prefill,
        "embedding_params_bytes_vocab": 256 * 128 * 2,
        "embedding_params_200k_vocab_at_d128": 2 * 200_000 * 128,
    }


EXAMPLE = Example(
    "07_byte_level_patching",
    __doc__ or "",
    add_args,
    build,
    train_fn,
    test_fn,
    infer_fn,
    runtime_args=("decode_bytes",),
)

if __name__ == "__main__":
    run(EXAMPLE)
