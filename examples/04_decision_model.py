"""A tiny Jev-style decision model: it never generates text, it only answers typed questions about a state.

Jev (TypeSafe AI's "System One" model; open JEV-9B from AutoTrust) takes a `state` plus questions of three
kinds and returns probabilities, not text:
    Choice  pick one of K coded options      -> softmax over the option codes
    Score   place the state on a 0..9 scale  -> ordinal distribution and E[x]
    Noul    yes / no                         -> P(yes)

This example builds that shape from the repo's primitives, on a synthetic key/value task:

    state tokens ──► prelude (GDN, LittleBit) ──► looped core ×N (shared GDN block, LittleBit) ──► coda (FP)
          │ prefilled ONCE; every layer's recurrent state is kept (fixed size, no K/V cache)
          ▼
    question forks: each question is a short suffix that starts from a copy of those states, so all
    questions about one state run as one batch and never see each other. Questions write nothing back.
          ▼
    hidden state at [ANS] ──► tied embedding rows of the allowed answer codes ──► per-type temperature ──► p

Option names are codes ([A]..[H]) followed by their definition, and codes are reassigned at random every
time, so the model has to read definitions instead of memorising letters. The eval reports accuracy,
calibration (ECE, Brier), how often a choice flips when only the codes / order are shuffled, and ms per question.

Not included yet (see docs/primitives.md): sparse state index, loop memory, per-loop scales, MoE, halting.

This is a code demo: the default is a 30-step smoke run that only shows the pieces fit together, so
the accuracies it prints are near chance. Pass --steps to actually train.

    uv run python examples/04_decision_model.py                    # LittleBit prelude + core, smoke run
    uv run python examples/04_decision_model.py --linear bitnet --loops 3
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch import nn

from littlebit_scratch.decision import StatefulBlock, TypedReadout, brier, clone_states, crps, ece, fork_states, ordinal_stats
from littlebit_scratch.measure import Meter, size_report, state_dict_bytes
from littlebit_scratch.model import Config, make_linear_factory
from littlebit_scratch.train import TrainConfig, make_optimizers

# ---- vocabulary -------------------------------------------------------------------------------
N_KEYS = N_VALS = 16
KEY, VAL = 0, 16                 # KEY+i, VAL+i
CODE, N_CODES = 32, 8            # option codes [A]..[H]
DIGIT = 40                       # 0..9 for Score
YES, NO = 50, 51
STATE, Q, CHOICE, SCORE, NOUL, ANS = 52, 53, 54, 55, 56, 57
VOCAB = 58
K_OPTIONS = 4
TYPES = ("choice", "score", "noul")


def make_batch(B: int, g: torch.Generator) -> dict:
    """B states (all 16 keys in random order with random values) and one question of each type per state."""
    vals = torch.randint(0, N_VALS, (B, N_KEYS), generator=g)
    order = torch.argsort(torch.rand(B, N_KEYS, generator=g), dim=1)
    state = torch.full((B, 1 + 2 * N_KEYS), STATE)
    state[:, 1::2] = KEY + order
    state[:, 2::2] = VAL + vals.gather(1, order)
    rows = torch.arange(B)

    # Choice: "value of key x?"  options = K coded values, one correct
    kx = torch.randint(0, N_KEYS, (B,), generator=g)
    true_v = vals[rows, kx]
    cand = torch.stack([torch.randperm(N_VALS - 1, generator=g)[: K_OPTIONS - 1] for _ in range(B)])
    cand = cand + (cand >= true_v[:, None]).long()                       # distractors != true value
    opt_vals = torch.cat([true_v[:, None], cand], 1)
    perm = torch.argsort(torch.rand(B, K_OPTIONS, generator=g), 1)
    opt_vals = opt_vals.gather(1, perm)
    codes = torch.stack([torch.randperm(N_CODES, generator=g)[:K_OPTIONS] for _ in range(B)])
    choice = torch.cat([torch.full((B, 1), Q), torch.full((B, 1), CHOICE), (KEY + kx)[:, None],
                        torch.stack([CODE + codes, VAL + opt_vals], 2).flatten(1), torch.full((B, 1), ANS)], 1)
    choice_allowed = CODE + codes                                         # readout rows for this question
    choice_target = (opt_vals == true_v[:, None]).float().argmax(1)

    # Score: "how many keys have value v?" (0..9)
    v = torch.where(torch.rand(B, generator=g) < 0.7, vals[rows, torch.randint(0, N_KEYS, (B,), generator=g)],
                    torch.randint(0, N_VALS, (B,), generator=g))
    score = torch.stack([torch.full((B,), Q), torch.full((B,), SCORE), VAL + v, torch.full((B,), ANS)], 1)
    score_target = (vals == v[:, None]).sum(1).clamp(max=9)

    # Noul: "does key x have value v?"
    yes = torch.rand(B, generator=g) < 0.5
    other = (true_v + torch.randint(1, N_VALS, (B,), generator=g)) % N_VALS
    nv = torch.where(yes, true_v, other)
    noul = torch.stack([torch.full((B,), Q), torch.full((B,), NOUL), KEY + kx, VAL + nv, torch.full((B,), ANS)], 1)

    return {"state": state,
            "choice": (choice, choice_allowed, choice_target, opt_vals),
            "score": (score, (DIGIT + torch.arange(10)).expand(B, 10), score_target),
            "noul": (noul, torch.tensor([NO, YES]).expand(B, 2), yes.long())}


# ---- model ------------------------------------------------------------------------------------
class DecisionModel(nn.Module):
    def __init__(self, d: int = 96, heads: int = 4, loops: int = 2, linear: str = "littlebit", bpw: float = 1.0,
                 mixer: str = "gdn"):
        super().__init__()
        lb = make_linear_factory(Config(linear=linear, bpw=bpw, init="svd", itq_iters=20))
        fp = make_linear_factory(Config(linear="fp"))
        self.embed = nn.Embedding(VOCAB, d)            # tied with the answer-code readout rows
        nn.init.normal_(self.embed.weight, std=0.02)
        self.prelude = StatefulBlock(d, heads, lb, mixer)
        self.core = StatefulBlock(d, heads, lb, mixer)  # one set of weights, applied `loops` times
        self.coda = StatefulBlock(d, heads, fp, mixer)  # higher precision at the end
        self.norm = nn.RMSNorm(d)
        self.loops = loops
        self.readout = TypedReadout(self.embed, TYPES)  # tied rows + per-type temperature

    def forward(self, tokens, states=None):
        """Returns hidden states and the per-layer recurrent states [prelude, loop_1..loop_N, coda]."""
        states = states or [None] * (self.loops + 2)
        x = self.embed(tokens)
        x, s0 = self.prelude(x, states[0])
        new = [s0]
        for i in range(self.loops):                    # each loop pass keeps its own GDN state
            x, si = self.core(x, states[1 + i])
            new.append(si)
        x, sc = self.coda(x, states[-1])
        return self.norm(x), new + [sc]

    def prefill(self, state_tokens):
        return self(state_tokens)[1]

    def answer_logp(self, states, suffix, allowed, kind: str):
        """Fork from `states` (one per row), run the question suffix, read out at its last ([ANS]) token."""
        h, _ = self(suffix, clone_states(states))
        return self.readout(h[:, -1], allowed, kind)


def loss_fn(model, batch) -> tuple[torch.Tensor, dict]:
    states = model.prefill(batch["state"])             # state written once, shared by all questions
    total, parts = 0.0, {}
    for kind in TYPES:
        suffix, allowed, target = batch[kind][:3]
        logp = model.answer_logp(states, suffix, allowed, kind)
        nll = F.nll_loss(logp, target)
        if kind == "score":                            # proper scoring for ordinal: add CRPS
            nll = nll + crps(logp.exp(), target).mean()
        parts[kind] = nll.item()
        total = total + nll
    return total, parts


# ---- eval -------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, batch) -> dict:
    model.eval()
    out = {}
    states = model.prefill(batch["state"])
    for kind in TYPES:
        suffix, allowed, target = batch[kind][:3]
        p = model.answer_logp(states, suffix, allowed, kind).exp()
        conf, pred = p.max(-1)
        correct = pred == target
        out[kind] = {"acc": correct.float().mean().item(), "ece": ece(conf, correct), "brier": brier(p, target).item()}
        if kind == "score":
            out[kind]["mae_E[x]"] = (ordinal_stats(p) - target).abs().mean().item()
        if kind == "choice":                           # code-swap / order invariance
            opt_vals = batch["choice"][3]
            chosen = opt_vals.gather(1, pred[:, None]).squeeze(1)
            B = suffix.shape[0]
            perm = torch.argsort(torch.rand(B, K_OPTIONS), 1)
            codes = torch.stack([torch.randperm(N_CODES)[:K_OPTIONS] for _ in range(B)])
            v2 = opt_vals.gather(1, perm)
            s2 = suffix.clone()
            s2[:, 3:3 + 2 * K_OPTIONS] = torch.stack([CODE + codes, VAL + v2], 2).flatten(1)
            p2 = model.answer_logp(states, s2, CODE + codes, kind).exp()
            chosen2 = v2.gather(1, p2.argmax(-1)[:, None]).squeeze(1)
            out[kind]["flip_rate"] = (chosen != chosen2).float().mean().item()
    model.train()
    return out


@torch.no_grad()
def timing(model, batch, n_q: int = 64) -> dict:
    """ms to prefill one state, then ms per question when n_q questions about it run as one batch."""
    model.eval()
    st = batch["state"][:1]
    t = time.perf_counter()
    for _ in range(5):
        states = model.prefill(st)
    prefill_ms = (time.perf_counter() - t) / 5 * 1e3
    suffix, allowed = batch["choice"][0][:n_q], batch["choice"][1][:n_q]
    forks = fork_states(states, n_q)
    t = time.perf_counter()
    for _ in range(5):
        model.answer_logp(forks, suffix, allowed, "choice")
    per_q = (time.perf_counter() - t) / 5 * 1e3 / n_q
    model.train()
    return {"prefill_ms_per_state": prefill_ms, "ms_per_question_batched": per_q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", default="littlebit", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--bpw", type=float, default=1.0)
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mixer", default="gdn", help="gdn | mamba2 | gla | mlstm | mingru")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    g = torch.Generator().manual_seed(0)
    model = DecisionModel(args.d, 4, args.loops, args.linear, args.bpw, args.mixer)
    tc = TrainConfig(lr=3e-3, latent_lr_mult=0.1)
    opts = make_optimizers(model, tc)
    sched = [torch.optim.lr_scheduler.OneCycleLR(o, max_lr=[pg["lr"] for pg in o.param_groups],
                                                 total_steps=args.steps, pct_start=0.1) for o in opts]
    test = make_batch(256, torch.Generator().manual_seed(123))
    t0 = time.time()
    train_meter = Meter().__enter__()
    for step in range(args.steps):
        loss, parts = loss_fn(model, make_batch(args.batch, g))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o, s in zip(opts, sched):
            o.step()
            o.zero_grad(set_to_none=True)
            s.step()
        if step % max(1, args.steps // 4) == 0 or step == args.steps - 1:
            ev = evaluate(model, test)
            print(f"step {step:5d}  loss {loss.item():.3f}  " + "  ".join(
                f"{k} acc {v['acc']:.2f}" for k, v in ev.items()) + f"  {time.time() - t0:.0f}s", flush=True)
    train_meter.__exit__()
    ev = evaluate(model, test)
    print(f"\n{args.linear} prelude+core ({args.bpw:g} bpw if littlebit), {args.loops} loops, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    for kind, m in ev.items():
        print(f"  {kind:6s} " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))
    print("  temps  " + "  ".join(f"{k} {t:.2f}" for k, t in zip(TYPES, model.readout.log_temp.exp().tolist())))
    with Meter() as infer_meter:
        t = timing(model, test)
    print("  " + "  ".join(f"{k} {v:.2f}" for k, v in t.items()))
    sizes = size_report(model)
    metrics = {"train_steps": args.steps, **train_meter.as_dict("train_"), **infer_meter.as_dict("infer_"), **t,
               "params": sizes["trainable_params"], "stored_bytes": sizes["deployed_total_bytes"],
               "ram_weights_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
               "checkpoint_bytes": state_dict_bytes(model), "choice_acc": ev["choice"]["acc"]}
    print("METRICS " + json.dumps(metrics))


if __name__ == "__main__":
    main()
