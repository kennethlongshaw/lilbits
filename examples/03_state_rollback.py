"""Speculative-decoding state handling on a trained stateful LM: snapshot, verify k drafts in one call, roll back.

Every recurrent mixer maps (state, tokens) -> (outputs, state), so verifying k drafts is one call from a snapshot,
and rejecting drafts j.. means replaying only the accepted prefix from the snapshot. The rolled-back state must
equal the state of a model that never saw the rejected drafts.

    train   a small GDN StatefulLM on the Markov source, Lightning Fabric
    test    held-out loss
    infer   prefill -> snapshot -> verify k drafts (some wrong) in one call -> accept the longest correct prefix ->
            roll back; checks rolled-back state == clean state and times rollback vs re-prefilling from scratch

    uv run python examples/03_state_rollback.py
    uv run python examples/03_state_rollback.py --mode infer --k 8
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from lilbits.data import MarkovSource
from lilbits.models.decision import clone_states
from lilbits.models.stateful_lm import StatefulLM, state_bytes
from lilbits.models.tinylm import Config, make_linear_factory
from lilbits.runner import Example, fit, run


def add_args(ap):
    ap.add_argument("--linear", default="fp", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--mixer", default="gdn")
    ap.add_argument("--steps", type=int, default=20, help="training steps (default: smoke run)")
    ap.add_argument("--k", type=int, default=5, help="infer: draft tokens per verify")
    ap.add_argument("--prompt", type=int, default=256, help="infer: prompt length")


def build(args, meta):
    lin = make_linear_factory(Config(linear=args.linear, bpw=1.0))
    return {"lm": StatefulLM(32, 64, 4, 2, lin, args.mixer)}


def lm_loss(model, x):
    logits, _ = model(x[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()), {}


def source(stream: int) -> MarkovSource:
    src = MarkovSource(vocab=32, seed=0)
    src._g.manual_seed(stream)
    return src


def train_fn(models, args, fab):
    src = source(1)
    return fit(fab, models["lm"], lambda _: src.sample(16, 64), lm_loss, args.steps), {}


def test_fn(models, args):
    return {"held_out_loss": lm_loss(models["lm"], source(999).sample(32, 64))[0].item()}


def max_diff(a: list, b: list) -> float:
    return max((x - y).abs().max().item() for x, y in zip(a, b, strict=True))


def infer_fn(models, args):
    lm = models["lm"]
    prompt = source(7).sample(1, args.prompt)
    logits, snap = lm(prompt)  # prefill, then snapshot
    # drafts: the model's own greedy continuation with the token at position k//2 corrupted
    toks, st, lg = [], clone_states(snap), logits
    for _ in range(args.k):
        toks.append(lg[:, -1:].argmax(-1))
        lg, st = lm(toks[-1], st)
    drafts = torch.cat(toks, 1)
    j = args.k // 2
    drafts[0, j] = (drafts[0, j] + 1) % 32
    t = time.perf_counter()
    vlogits, _ = lm(drafts, clone_states(snap))  # verify all k in one call
    t_verify = time.perf_counter() - t
    preds = torch.cat([logits[:, -1:], vlogits[:, :-1]], 1).argmax(-1)
    accepted = int((preds == drafts).long().cumprod(1).sum())
    t = time.perf_counter()
    # roll back = replay the accepted prefix from the snapshot (nothing accepted: the snapshot itself)
    rolled = lm(drafts[:, :accepted], clone_states(snap))[1] if accepted else clone_states(snap)
    t_rollback = time.perf_counter() - t
    t = time.perf_counter()
    _, clean = lm(torch.cat([prompt, drafts[:, :accepted]], 1))  # reference: never saw the rejected drafts
    t_reprefill = time.perf_counter() - t
    _, dirty = lm(drafts, clone_states(snap))
    return {
        "drafts": args.k,
        "accepted": accepted,
        "rolled_back_state_max_diff_vs_clean": max_diff(rolled, clean),
        "state_with_rejected_drafts_differs": max_diff(dirty, clean) > 1e-4,
        "verify_ms": t_verify * 1e3,
        "rollback_replay_ms": t_rollback * 1e3,
        "reprefill_from_scratch_ms": t_reprefill * 1e3,
        "state_bytes": state_bytes(snap),
    }


EXAMPLE = Example(
    "03_state_rollback", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("k", "prompt")
)

if __name__ == "__main__":
    run(EXAMPLE)
