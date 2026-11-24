"""Train a tiny LM whose block linears are LittleBit (or BitNet, or FP), then pack it for inference.

train   Lightning Fabric loop (lilbits.training.train): LittleBit from a short FP warm start that is
        SVD+ITQ-converted; BitNet and FP train directly
test    held-out loss on the Markov source (entropy floor ~0.89)
infer   pack to 1-bit signs (LittleBit) / 2-bit codes (BitNet), optionally on the LUT kernels (--kernel lut),
        and time a 32x63 batch and a single-token call

uv run python examples/01_train_and_pack.py --linear littlebit --bpw 0.8
uv run python examples/01_train_and_pack.py --linear bitnet --mode train
uv run python examples/01_train_and_pack.py --linear bitnet --mode infer --kernel lut
"""

from __future__ import annotations

import time

import torch

from lilbits.data import MarkovSource
from lilbits.inference.lut import pack_lut
from lilbits.inference.packed import pack_model
from lilbits.layers.bitnet import pack_bitnet
from lilbits.models.tinylm import Config, TinyLM
from lilbits.runner import Example, run
from lilbits.training.train import TrainConfig, evaluate, train


def add_args(ap):
    ap.add_argument("--linear", default="littlebit", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--bpw", type=float, default=1.41)
    ap.add_argument("--steps", type=int, default=60, help="training steps (default: smoke run)")
    ap.add_argument("--kernel", default="unpack", choices=["unpack", "lut"], help="infer: packed kernel")


def cfg(args) -> Config:
    return Config(linear=args.linear, bpw=args.bpw, itq_iters=50)


def held_out() -> list[torch.Tensor]:
    src = MarkovSource(vocab=32, seed=0)
    src._g.manual_seed(999)  # same chain, separate sample stream
    return [src.sample(32, 64) for _ in range(4)]


def build(args, meta):
    return {"lm": TinyLM(cfg(args))}


def train_fn(models, args, fab):
    tc = TrainConfig(
        steps=args.steps,
        optim="muon_ds",
        lr=1e-2,
        verbose=False,
        fp_warmup=args.steps // 3 if args.linear == "littlebit" else 0,
        latent_lr_mult=0.02,
        seed=args.seed,
        log_every=max(1, args.steps),
    )
    src = MarkovSource(vocab=32, seed=0)
    res = train(cfg(args), tc, src, [src.sample(32, 64) for _ in range(2)], fabric=fab)
    models["lm"] = res["model"]
    return {"steps": args.steps, "final_train_eval_loss": res["final_eval"]}, {}


def test_fn(models, args):
    return {"held_out_loss": evaluate(models["lm"], held_out()), "entropy_floor_approx": 0.89}


def _ms(f, reps: int) -> float:
    f()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps * 1e3


def infer_fn(models, args):
    lm = models["lm"]
    if args.linear == "littlebit":
        pack_model(lm)
    elif args.linear == "bitnet":
        pack_bitnet(lm)
    if args.kernel == "lut" and args.linear != "fp":
        pack_lut(lm)
    ev = held_out()
    x, one = ev[0][:, :-1], ev[0][:1, :1]
    return {
        "kernel": args.kernel if args.linear != "fp" else "dense",
        "packed_held_out_loss": evaluate(lm, ev),
        "ms_per_32x63_batch": _ms(lambda: lm(x), 10),
        "ms_per_single_token_call": _ms(lambda: lm(one), 50),
    }


EXAMPLE = Example(
    "01_train_and_pack", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("kernel",)
)

if __name__ == "__main__":
    run(EXAMPLE)
