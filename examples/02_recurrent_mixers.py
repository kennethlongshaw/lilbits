"""The recurrent token mixers side by side, as small stateful LMs: train, test, then decode with carried state.

Each mixer keeps a fixed-size recurrent state instead of a softmax K/V cache that grows with length, so decoding is
one single-token call per new token with nothing recomputed.

    train   one StatefulLM per mixer (GDN, Mamba2, GLA, mLSTM, minGRU) on the Markov source, Lightning Fabric
    test    held-out loss per mixer
    infer   prefill a prompt, then decode token by token: ms per token, state bytes vs an attention K/V cache;
            --kernel lut runs LittleBit / BitNet projections on the lookup-table kernels

    uv run python examples/02_recurrent_mixers.py
    uv run python examples/02_recurrent_mixers.py --linear littlebit --mode infer --gen 64
"""

from __future__ import annotations

import time

import torch.nn.functional as F

from lilbits.data import MarkovSource
from lilbits.inference.lut import pack_lut
from lilbits.mixers.recurrent import MIXERS
from lilbits.models.stateful_lm import StatefulLM, state_bytes
from lilbits.models.tinylm import Config, make_linear_factory
from lilbits.runner import Example, fit, run


def add_args(ap):
    ap.add_argument("--linear", default="fp", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--bpw", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=15, help="training steps per mixer (default: smoke run)")
    ap.add_argument("--mixers", default=",".join(MIXERS))
    ap.add_argument("--gen", type=int, default=32, help="infer: tokens to decode")
    ap.add_argument("--kernel", default="native", choices=["native", "lut"], help="infer: LUT kernels for LB/BitNet")


def build(args, meta):
    lin = make_linear_factory(Config(linear=args.linear, bpw=args.bpw))
    return {m: StatefulLM(32, args.d, 4, args.layers, lin, m) for m in args.mixers.split(",")}


def lm_loss(model, x):
    logits, _ = model(x[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())
    return loss, {}


def held_out():
    src = MarkovSource(vocab=32, seed=0)
    src._g.manual_seed(999)
    return src.sample(32, 64)


def train_fn(models, args, fab):
    src = MarkovSource(vocab=32, seed=0)
    out = {}
    for name, m in models.items():
        out[name] = fit(fab, m, lambda _: src.sample(16, 64), lm_loss, args.steps)["last_loss"]
    return {"last_loss": out}, {}


def test_fn(models, args):
    x = held_out()
    return {"held_out_loss": {k: lm_loss(m, x)[0].item() for k, m in models.items()}}


def infer_fn(models, args):
    prompt = held_out()[:1, :32]
    ms, sizes = {}, {}
    for name, m in models.items():
        if args.kernel == "lut":
            pack_lut(m)
        m.generate(prompt, 4)  # warm up
        t = time.perf_counter()
        _, states = m.generate(prompt, args.gen)
        ms[name] = (time.perf_counter() - t) / args.gen * 1e3
        sizes[name] = state_bytes(states)
    T = prompt.shape[1] + args.gen
    kv = 2 * args.d * 4 * args.layers * T  # FP32 K and V per layer per token, softmax attention
    return {
        "ms_per_decoded_token_incl_prefill": ms,
        "state_bytes": sizes,
        "attention_kv_bytes_same_length": kv,
        "context_tokens": T,
        "kernel": args.kernel,
    }


EXAMPLE = Example(
    "02_recurrent_mixers", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("gen", "kernel")
)

if __name__ == "__main__":
    run(EXAMPLE)
