"""LLM-JEPA with an EMA target encoder: add an embedding-prediction loss to next-token training.

Follows Huang, LeCun & Balestriero (2025), "LLM-JEPA". Each example is a pair of views of the same content
(Text, Code). Next to the usual Text -> Code next-token loss, the model predicts the embedding of Code from Text
through k appended [PRED] tokens, and a cosine loss pulls the prediction onto the target's embedding of Code.
The target encoder is an exponential moving average of the model (stop-gradient), or the model itself
(--target online, as in the paper). One packed forward with a block-causal mask computes both terms.

Phases (default: a short smoke run on a synthetic paired-view source, to prove the code works):
    train   four variants on Lightning Fabric: LM loss only, LLM-JEPA (online target), LLM-JEPA (EMA target),
            LLM-JEPA (EMA) with loss dropout 0.5; per-variant cost and the training-only EMA copy's RAM
    test    Text -> Code loss and token accuracy, Text/Code embedding retrieval top-1
    infer   the deployed LM alone (no EMA copy, no [PRED] tokens): latency for 64 Text+Code pairs

At inference the model is an ordinary LM: the EMA target and the [PRED] tokens are training-only, so deployed
size and decode cost equal the LM-only baseline. The cost is at training time (an extra target forward over Code,
k extra positions, and the EMA copy in memory), which loss dropout cuts.

    uv run python examples/08_llm_jepa.py                    # FP, smoke run
    uv run python examples/08_llm_jepa.py --linear littlebit  # same with sub-1-bit-capable linears
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from lilbits.data import PairedViewSource
from lilbits.measure import Meter, tensor_bytes
from lilbits.models.jepa import LLMJEPA, JEPAConfig, retrieval_accuracy
from lilbits.models.tinylm import Config, TinyLM
from lilbits.runner import Example, fit, run

VARIANTS: dict[str, dict[str, Any]] = {
    "LM loss only": dict(lam=0.0, target="online"),
    "LLM-JEPA, online target": dict(lam=1.0, target="online"),
    "LLM-JEPA, EMA target": dict(lam=1.0, target="ema"),
    "LLM-JEPA, EMA + loss dropout 0.5": dict(lam=1.0, target="ema", loss_dropout=0.5),
}


@torch.no_grad()
def evaluate(m: LLMJEPA, text: torch.Tensor, code: torch.Tensor) -> dict:
    Lt = text.shape[1]
    logits = m.lm(torch.cat([text, code], 1))[:, Lt - 1 : -1]
    return {
        "code_loss": F.cross_entropy(logits.flatten(0, 1), code.flatten()).item(),
        "code_token_acc": logits.argmax(-1).eq(code).float().mean().item(),
        "retrieval_top1": retrieval_accuracy(m, text, code),
    }


def add_args(ap):
    ap.add_argument("--steps", type=int, default=30, help="training steps per variant (default: smoke run)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k-pred", type=int, default=2, help="number of [PRED] tokens")
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--linear", default="fp", choices=["fp", "littlebit", "bitnet"])


def source(seed: int) -> PairedViewSource:
    return PairedViewSource(n_symbols=8, length=8, seed=seed)


def build(args, meta):
    src = source(0)
    out = {}
    for name, kw in VARIANTS.items():
        lm = TinyLM(
            Config(
                vocab_size=src.vocab,
                seq_len=2 * src.length + args.k_pred,
                d_model=args.d,
                n_heads=4,
                n_layers=2,
                linear=args.linear,
            )
        )
        out[name] = LLMJEPA(
            lm, JEPAConfig(pred_token=src.pred_token, k_pred=args.k_pred, ema_decay=args.ema_decay, **kw)
        )
    return out


def train_fn(models, args, fab):
    out = {}
    for name, m in models.items():
        src = source(0)
        g = torch.Generator().manual_seed(args.seed)
        dropped = []

        def loss_fn(model, batch, g=g, dropped=dropped):
            loss, info = model.loss(*batch, generator=g)
            dropped.append(info["jepa_dropped"])
            return loss, {k: v for k, v in info.items() if k != "jepa_dropped"}

        with Meter() as meter:
            info = fit(
                fab,
                m,
                lambda _, src=src: src.sample(32),
                loss_fn,
                args.steps,
                lr=2e-3,
                clip=None,
                after_step=m.step_target,
            )
        ema = tensor_bytes(m.target.state_dict().values()) if m.target is not None else 0
        out[name] = {
            **info,
            "jepa_steps_dropped": sum(dropped) if m.cfg.lam > 0 else 0,
            "train_only_ema_bytes": ema,
            **meter.as_dict(),
        }
    return {"variants": out}, {}


def eval_pairs():
    src = source(0)
    src._g.manual_seed(12345)
    return src.sample(64)


def test_fn(models, args):
    text, code = eval_pairs()
    return {name: evaluate(m, text, code) for name, m in models.items()}


def infer_fn(models, args):
    text, code = eval_pairs()
    seq = torch.cat([text, code], 1)
    out = {}
    for name, m in models.items():
        m.lm(seq)
        t = time.perf_counter()
        for _ in range(5):
            m.lm(seq)
        out[name] = {"ms_per_64_pairs": (time.perf_counter() - t) / 5 * 1e3}
    return out


EXAMPLE = Example(
    "08_llm_jepa",
    __doc__ or "",
    add_args,
    build,
    train_fn,
    test_fn,
    infer_fn,
    deploy=lambda models: {k: m.lm for k, m in models.items() if isinstance(m, LLMJEPA)},
)

if __name__ == "__main__":
    run(EXAMPLE)
