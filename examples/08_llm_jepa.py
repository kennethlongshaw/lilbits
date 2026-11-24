"""LLM-JEPA with an EMA target encoder: add an embedding-prediction loss to next-token training.

Follows Huang, LeCun & Balestriero (2025), "LLM-JEPA". Each example is a pair of views of the same content
(Text, Code). Next to the usual Text -> Code next-token loss, the model predicts the embedding of Code from Text
through k appended [PRED] tokens, and a cosine loss pulls the prediction onto the target's embedding of Code.
The target encoder is an exponential moving average of the model (stop-gradient), or the model itself
(--target online, as in the paper). One packed forward with a block-causal mask computes both terms.

What this script shows (default: a short smoke run on a synthetic paired-view source, to prove the code works):
  1. variants trained side by side: LM loss only, LLM-JEPA (online target), LLM-JEPA (EMA target),
     and LLM-JEPA (EMA) with loss dropout 0.5
  2. per variant: Text -> Code loss and token accuracy, Text/Code embedding retrieval top-1, train and infer
     wall time, CPU time, peak RSS, training-time RAM (the EMA copy), checkpoint and deployed size
  3. METRICS line for benchmarks/examples_bench.py

At inference the model is an ordinary LM: the EMA target and the [PRED] tokens are training-only, so deployed
size and decode cost equal the LM-only baseline. The cost is at training time (an extra target forward over Code,
k extra positions, and the EMA copy in memory), which loss dropout cuts.

    uv run python examples/08_llm_jepa.py                    # FP, smoke run
    uv run python examples/08_llm_jepa.py --linear littlebit  # same with sub-1-bit-capable linears
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F

from lilbits.data import PairedViewSource
from lilbits.measure import Meter, size_report, state_dict_bytes, tensor_bytes
from lilbits.models.jepa import LLMJEPA, JEPAConfig, retrieval_accuracy
from lilbits.models.tinylm import Config, TinyLM

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30, help="training steps (default: smoke run)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k-pred", type=int, default=2, help="number of [PRED] tokens")
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--linear", default="fp", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    src = PairedViewSource(n_symbols=8, length=8, seed=0)
    eval_src = PairedViewSource(n_symbols=8, length=8, seed=0)
    eval_src._g.manual_seed(12345)
    eval_text, eval_code = eval_src.sample(64)
    seq_len = 2 * src.length + args.k_pred

    print(
        f"paired views: Text = {src.length} symbols, Code = reversed + substituted; vocab {src.vocab} "
        f"(incl. [PRED]); k = {args.k_pred}; {args.steps} steps; linear = {args.linear}\n"
    )
    print(
        "| variant | code loss | code token acc | retrieval top-1 | train wall / CPU | train peak RSS | "
        "train-only RAM (EMA copy) | infer wall (64 pairs) | checkpoint | deployed |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    metrics: dict[str, dict] = {}
    for name, kw in VARIANTS.items():
        torch.manual_seed(0)
        src._g.manual_seed(0)
        lm = TinyLM(
            Config(vocab_size=src.vocab, seq_len=seq_len, d_model=args.d, n_heads=4, n_layers=2, linear=args.linear)
        )
        m = LLMJEPA(lm, JEPAConfig(pred_token=src.pred_token, k_pred=args.k_pred, ema_decay=args.ema_decay, **kw))
        opt = torch.optim.AdamW(m.lm.parameters(), lr=2e-3, betas=(0.9, 0.95))
        g = torch.Generator().manual_seed(0)
        dropped = 0
        m.train()
        with Meter() as tm:
            for _ in range(args.steps):
                loss, info = m.loss(*src.sample(32), generator=g)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                m.step_target()
                dropped += info["jepa_dropped"] and kw["lam"] > 0
        m.eval()
        with Meter() as im:
            ev = evaluate(m, eval_text, eval_code)
        ema_bytes = tensor_bytes(m.target.state_dict().values()) if m.target is not None else 0
        ckpt = state_dict_bytes(m.lm)
        deployed = size_report(m.lm)["deployed_total_bytes"]
        print(
            f"| {name} | {ev['code_loss']:.3f} | {ev['code_token_acc']:.2f} | {ev['retrieval_top1']:.2f} | "
            f"{tm.wall_s:.1f} s / {tm.cpu_s:.1f} s | {tm.peak_extra_mib:.0f} MiB | {ema_bytes / 1024:,.0f} KiB | "
            f"{im.wall_s * 1e3:.0f} ms | {ckpt / 1024:,.0f} KiB | {deployed / 1024:,.0f} KiB |"
        )
        metrics[name] = {
            **ev,
            **tm.as_dict("train_"),
            **im.as_dict("infer_"),
            "jepa_steps_dropped": int(dropped),
            "train_only_ema_bytes": ema_bytes,
            "params": sum(p.numel() for p in m.lm.parameters()),
            "checkpoint_bytes": ckpt,
            "stored_bytes": deployed,
        }
    print(
        "\nsmoke-run numbers only prove the code paths; the paper's gains need real paired data and full training."
        "\nthe EMA target and [PRED] tokens are training-only: deployed size and decode cost equal the LM baseline."
    )
    print("METRICS " + json.dumps({"train_steps": args.steps, "variants": metrics}))


if __name__ == "__main__":
    main()
