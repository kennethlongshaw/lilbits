"""End-to-end inference: train LittleBit -> pack to 1-bit storage -> run on the torch, LUT and Helion backends.

    train   Lightning Fabric loop (lilbits.training.train) with a short FP warm start, SVD+ITQ conversion
    test    held-out loss of the trained (unpacked) model against the Markov entropy floor
    infer   pack the checkpoint; for each backend (torch unpack, LUT kernels, Helion): held-out loss, max logit
            difference vs the unpacked model, whether greedy generation is identical, weight RAM, wall time

Helion runs compiled on CUDA, or in its eager reference mode on CPU with HELION_INTERPRET=1 (slow; correctness
only). Without helion installed that backend is skipped.

    HELION_INTERPRET=1 uv run --extra gpu python examples/05_pack_and_helion.py
    uv run python examples/05_pack_and_helion.py --backends torch,lut
"""

from __future__ import annotations

import copy
import importlib.util
import io
import os
import time

import torch
import torch.nn.functional as F

from lilbits.data import MarkovSource
from lilbits.inference.lut import pack_lut
from lilbits.inference.packed import model_bytes, pack_model, set_backend
from lilbits.models.tinylm import Config, TinyLM
from lilbits.runner import Example, run
from lilbits.training.train import TrainConfig, train


def add_args(ap):
    ap.add_argument("--steps", type=int, default=60, help="training steps (default: smoke run)")
    ap.add_argument("--fp-warmup", type=int, default=20)
    ap.add_argument("--bpw", type=float, default=1.41)
    ap.add_argument("--gen", type=int, default=24, help="infer: greedy tokens to generate")
    ap.add_argument("--backends", default="torch,lut,helion")


def cfg(args) -> Config:
    return Config(
        vocab_size=32, seq_len=64, d_model=128, n_heads=4, n_layers=2, linear="littlebit", bpw=args.bpw, itq_iters=50
    )


def source(stream: int | None = None) -> MarkovSource:
    src = MarkovSource(vocab=32, seed=0)
    if stream is not None:
        src._g.manual_seed(stream)
    return src


def eval_loss(model, batches) -> float:
    return sum(F.cross_entropy(model(b[:, :-1]).flatten(0, 1), b[:, 1:].flatten()).item() for b in batches) / len(
        batches
    )


def generate(model, prompt: torch.Tensor, n: int, seq_len: int) -> torch.Tensor:
    out = prompt.clone()
    for _ in range(n):
        out = torch.cat([out, model(out[:, -seq_len:])[:, -1].argmax(-1, keepdim=True)], 1)
    return out


def build(args, meta):
    return {"lm": TinyLM(cfg(args))}


def train_fn(models, args, fab):
    tc = TrainConfig(steps=args.steps, log_every=args.steps, fp_warmup=args.fp_warmup, verbose=False, seed=args.seed)
    src = source()
    res = train(cfg(args), tc, src, [src.sample(32, 64) for _ in range(2)], fabric=fab)
    models["lm"] = res["model"]
    return {"steps": args.steps, "final_train_eval_loss": res["final_eval"]}, {}


def test_fn(models, args):
    src = source(999)
    held = [src.sample(16, 64) for _ in range(2)]
    floor = sum(src.entropy_floor(b) for b in held) / len(held)
    return {"held_out_loss": eval_loss(models["lm"], held), "entropy_floor": floor}


def infer_fn(models, args):
    held = [source(999).sample(16, 64) for _ in range(2)]
    ref = models["lm"].eval()
    prompt = held[0][:2, :8]
    ref_logits = ref(held[0][:, :-1])
    ref_gen = generate(ref, prompt, args.gen, 64)
    out: dict = {"unpacked_weight_ram_bytes": model_bytes(ref)}
    packed = pack_model(copy.deepcopy(ref))
    buf = io.BytesIO()
    torch.save(packed.state_dict(), buf)
    out["packed_checkpoint_bytes"] = buf.tell()
    for backend in args.backends.split(","):
        if backend == "helion" and importlib.util.find_spec("helion") is None:
            out["helion"] = "skipped (helion not installed; uv sync --extra gpu)"
            continue
        if backend == "helion" and args.accelerator == "cpu" and os.environ.get("HELION_INTERPRET") != "1":
            out["helion"] = "skipped (CPU needs HELION_INTERPRET=1)"
            continue
        m = pack_lut(copy.deepcopy(packed)) if backend == "lut" else set_backend(copy.deepcopy(packed), backend)
        t = time.perf_counter()
        loss = eval_loss(m, held)
        diff = (m(held[0][:, :-1]) - ref_logits).abs().max().item()
        same = torch.equal(generate(m, prompt, args.gen, 64), ref_gen)
        out[backend] = {
            "held_out_loss": loss,
            "max_logit_diff": diff,
            "same_greedy": same,
            "weight_ram_bytes": model_bytes(m),
            "wall_s": time.perf_counter() - t,
        }
    models["lm"] = packed  # deployed form, for the size report
    return out


EXAMPLE = Example(
    "05_pack_and_helion", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("gen", "backends")
)

if __name__ == "__main__":
    run(EXAMPLE)
