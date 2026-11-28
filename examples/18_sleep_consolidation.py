"""Continual learning with a sleep phase: new facts go to fast memory while awake, then into new low-rank experts.

After Behrouz, Hashemi, Javanmard and Mirrokni, "Language Models Need Sleep: Learning to Self-Modify and
Consolidate Memories" (Google Research, 2026, arXiv:2606.03979; blog: "The sleep framework"). A pretrained byte LM
lives through a series of sessions. Each session brings new facts it has never seen ("Tavrin Olsk's number is
4821."). While awake it writes them into fast memory (a low-rank adapter per MLP, trained on the session's text).
Then it sleeps (models/sleep_consolidation.py): it unlocks a new low-rank expert per block, distills the
fast-memory model into it on its own dreams (GKD, on- and off-policy, plus a learning-to-imitate edit-distance
reward), dreams again with random old experts forced on and keeps the dreams with the largest gradients for SFT,
and finally resets the fast memory. The base weights never change after pretraining.

    models   fast only   wake, then the fast memory is reset with nothing kept (in-context-learning analogue)
             sft         continual fine-tuning: one slow adapter trained on every session's text in turn
             sleep       wake + sleep (knowledge seeding + dreaming)
             sleep, no dreaming   ablation
             all start from the same pretrained base; `--linear littlebit` makes its block linears sub-1-bit-capable
    data     pretraining: the ~10 MB corpus (lilbits.corpus; `--data shakespeare` for 1.1 MB) mixed with fact
             sentences about random names, so the format is known but no fact is; sessions: `--sessions` x
             `--facts` new facts each
    train    pretrain, then every method lives through the same sessions; retention after every session
    test     exact-match recall of every session's numbers with no context and fast memory off (what was kept),
             in-session recall with fast memory on (what was learned), held-out bits per byte (base drift)
    infer    decode speed with every unlocked expert on vs the base, slow-memory bytes per session

Smoke-trained by default (the mechanics only); a real run:

    uv run python examples/18_sleep_consolidation.py --d 64 --bytes-per-param 20 --wake-steps 300 \\
        --consolidate-steps 200 --dream-steps 60 --expert-rank 16

At that size (results/sleep/README.md) the fast memory learns every fact and continual SFT forgets all but the
last session, but sleep does not yet keep the exact numbers: consolidation stalls at a KL of about 0.13 nats.
"""

from __future__ import annotations

import copy
import math
import random
import time

import torch

from lilbits.corpus import CorpusBytes
from lilbits.measure import Meter
from lilbits.models.sleep_consolidation import ConsolidatingLM, ConsolidationConfig, SleepSettings, sleep, wake
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import ShakespeareBytes

METHODS = ("fast only", "sft", "sleep", "sleep, no dreaming")
SYL = ("ka", "lo", "mi", "ra", "ten", "vor", "sil", "an", "dru", "el", "fen", "gor", "is", "jun", "mar", "ost")


def add_args(ap):
    ap.add_argument("--data", default="corpus", choices=["corpus", "shakespeare"])
    ap.add_argument("--steps", type=int, default=4, help="pretraining steps (default: smoke run)")
    ap.add_argument(
        "--bytes-per-param", type=float, default=0, help="if > 0: pretrain on this many bytes per base parameter"
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--linear", default="fp", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--bpw", type=float, default=1.0)
    ap.add_argument("--fast-rank", type=int, default=16)
    ap.add_argument("--expert-rank", type=int, default=8)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--facts", type=int, default=6, help="new facts per session")
    ap.add_argument("--wake-steps", type=int, default=3)
    ap.add_argument("--wake-lr", type=float, default=1e-2, help="fast memory (and the SFT baseline) learning rate")
    ap.add_argument("--consolidate-steps", type=int, default=2)
    ap.add_argument("--dream-steps", type=int, default=2)
    ap.add_argument("--n-dreams", type=int, default=32)
    ap.add_argument("--anchor", type=float, default=1.0, help="anchor KL on general text during seeding (0 = off)")
    ap.add_argument("--sleep-lr", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=0.5, help="on-policy fraction of distillation batches")
    ap.add_argument("--alpha", type=float, default=0.8, help="divergence weight vs the LTI reward")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--test-batches", type=int, default=4)
    ap.add_argument("--gen", type=int, default=64)


def build(args, meta):
    torch.manual_seed(args.seed)
    cfg = ConsolidationConfig(
        seq_len=args.seq,
        d_model=args.d,
        n_heads=args.heads,
        n_layers=args.layers,
        fast_rank=args.fast_rank,
        expert_rank=args.expert_rank,
        max_experts=max(1, args.sessions),
        linear=args.linear,
        bpw=args.bpw,
    )
    base = ConsolidatingLM(cfg)
    return {"base": base, **{m: copy.deepcopy(base) for m in METHODS}}


# ---- facts
def name(rng: random.Random) -> str:
    w = lambda: "".join(rng.choice(SYL) for _ in range(rng.randint(2, 3))).capitalize()
    return f"{w()} {w()}"


def session_facts(seed: int, session: int, n: int) -> list[tuple[str, str]]:
    rng = random.Random(10_000 * (seed + 1) + session)
    return [(name(rng), f"{rng.randint(0, 9999):04d}") for _ in range(n)]


def sentence(fact: tuple[str, str]) -> bytes:
    return f"{fact[0]}'s number is {fact[1]}.\n".encode()


def fact_stream(facts: list[tuple[str, str]], batch: int, seq: int, rng: random.Random) -> torch.Tensor:
    """(batch, seq + 1) bytes of the given facts' sentences in random order, back to back."""
    rows = []
    for _ in range(batch):
        buf = b"\n" * rng.randint(1, 24)  # a random run of newlines first, like the left-padded recall prompts
        while len(buf) < seq + 1:
            buf += sentence(rng.choice(facts))
        rows.append(list(buf[: seq + 1]))
    return torch.tensor(rows)


def left_pad(rows: list[bytes], width: int = 0) -> torch.Tensor:
    n = max(width, *(len(r) for r in rows))
    return torch.tensor([list(b"\n" * (n - len(r)) + r) for r in rows])


def prompts_from(facts: list[tuple[str, str]], batch: int, rng: random.Random) -> torch.Tensor:
    """Prefixes of the session's sentences cut at random points: the context C that dreams are sampled from."""
    rows = []
    for _ in range(batch):
        s = sentence(rng.choice(facts))
        rows.append(b"\n" + s[: rng.randint(2, len(s) - 6)])
    return left_pad(rows, width=max(len(sentence(f)) for f in facts))  # one width per session


@torch.no_grad()
def recall(model: ConsolidatingLM, facts: list[tuple[str, str]], fast: bool = False) -> float:
    """Exact match of the 4-digit number after "<name>'s number is ", greedy, no context."""
    model.fast_on, model.extra, model.forced = fast, [], None
    p = left_pad([f"\n{n}'s number is ".encode() for n, _ in facts])
    out = model.sample(p, 4, temperature=0)[:, -4:]
    model.fast_on = False
    return sum(bytes(o.tolist()).decode(errors="replace") == v for o, (_, v) in zip(out, facts, strict=True)) / len(
        facts
    )


def text_source(args, split: str, seed: int):
    return (CorpusBytes if args.data == "corpus" else ShakespeareBytes)(split, seed=seed)


def train_fn(models, args, fab):
    out: dict = {}
    base = models["base"]
    text, rng = text_source(args, "train", args.seed), random.Random(args.seed)

    def pre_batch(step):  # half corpus text, half sentences about random (never-again-seen) names
        if step % 2 == 0:
            return text.sample(args.batch, args.seq)
        return fact_stream([(name(rng), f"{rng.randint(0, 9999):04d}") for _ in range(8)], args.batch, args.seq, rng)

    base.fast_on = False
    for p in base.memory_params():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in base.base_params())
    steps = args.steps
    if args.bytes_per_param > 0:  # Chinchilla-like data / weights ratio (~20 tokens per parameter at scale)
        steps = math.ceil(args.bytes_per_param * n_params / (args.batch * args.seq))
    out["data_ratio"] = {
        "base_params": n_params,
        "train_bytes": steps * args.batch * args.seq,
        "bytes_per_param": steps * args.batch * args.seq / n_params,
    }
    with Meter() as m:
        info = fit(fab, base, pre_batch, lambda mod, x: (mod.lm_loss(x), {}), steps, lr=args.lr, schedule="cosine")
    for p in base.memory_params():
        p.requires_grad_(True)
    out["pretrain"] = {**info, **m.as_dict()}
    for meth in METHODS:
        models[meth].load_state_dict(base.state_dict())

    s = SleepSettings(
        consolidate_steps=args.consolidate_steps,
        dream_steps=args.dream_steps,
        n_dreams=args.n_dreams,
        top_k=args.n_dreams // 3,
        n_random=max(1, args.n_dreams // 12),
        lam=args.lam,
        alpha=args.alpha,
        anchor=args.anchor,
        lr=args.sleep_lr,
    )
    facts = [session_facts(args.seed, k, args.facts) for k in range(args.sessions)]
    for meth in METHODS:
        model, g = models[meth], torch.Generator().manual_seed(args.seed)
        for p in model.base_params():
            p.requires_grad_(False)
        retention, rows = [], {}
        if meth == "sft":
            model.n_active.fill_(1)  # one slow adapter, trained continually and never reset
        with Meter() as mt:
            for k in range(args.sessions):
                r = random.Random(1000 * k + 7)
                batch_fn = lambda i, k=k, r=r: (fact_stream(facts[k], args.batch, args.seq, r), None)
                row: dict = {}
                if meth == "sft":
                    model.fast_on, model.extra = False, []
                    opt = torch.optim.Adam(model.expert_params(0), lr=args.wake_lr)
                    for i in range(args.wake_steps):
                        loss = model.lm_loss(batch_fn(i)[0])
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    row["session_recall"] = recall(model, facts[k])
                else:
                    t0 = time.perf_counter()
                    row["wake_loss"] = wake(model, batch_fn, args.wake_steps, args.wake_lr)
                    row["wake_s"] = time.perf_counter() - t0
                    row["session_recall"] = recall(model, facts[k], fast=True)
                    if meth == "fast only":
                        model.reset_fast()
                    else:
                        t0 = time.perf_counter()
                        pf = lambda i, k=k, r=r: prompts_from(facts[k], s.batch, r)
                        anchor = lambda i: text.sample(s.batch, args.seq // 2)[:, :-1]
                        row.update(sleep(model, pf, s, dreaming=meth == "sleep", generator=g, anchor_fn=anchor))
                        row["sleep_s"] = time.perf_counter() - t0
                retention.append([recall(model, facts[j]) for j in range(k + 1)])
                rows[f"session {k}"] = row
        out[meth] = {"sessions": rows, "retention_after_each_session": retention, **mt.as_dict()}
    return out, {}


def bpb(model, args) -> float:
    src = text_source(args, "test", 123)
    tot = 0.0
    for _ in range(args.test_batches):
        tot += model.lm_loss(src.sample(args.batch, args.seq)).item()
    return tot / args.test_batches / math.log(2)


def test_fn(models, args):
    facts = [session_facts(args.seed, k, args.facts) for k in range(args.sessions)]
    out: dict = {"recall_no_context": {}, "held_out_bits_per_byte": {}}
    for name_, m in models.items():
        m.fast_on, m.extra, m.forced = False, [], None
        per = [recall(m, f) for f in facts]
        out["recall_no_context"][name_] = {"mean": sum(per) / len(per), "per_session": per}
        out["held_out_bits_per_byte"][name_] = bpb(m, args)
    out["chance"] = "1e-4 per fact (4 random digits)"
    out["note"] = "smoke-trained runs show the mechanics only"
    return out


def infer_fn(models, args):
    prompt = text_source(args, "test", 7).sample(1, 32)[:, :32]
    rows = {}
    for name_ in ("base", "sleep"):
        m = models[name_]
        m.sample(prompt, 4, temperature=0)
        t = time.perf_counter()
        m.sample(prompt, args.gen, temperature=0)
        rows[name_] = {
            "bytes_per_s": args.gen / (time.perf_counter() - t),
            "experts_on": int(m.n_active),
            **m.stored_memory_bytes(),
        }
    per_expert = sum(p.numel() for p in models["base"].expert_params(0)) * 2
    base_bytes = sum(p.numel() for p in models["base"].base_params()) * 2
    return {
        "decode": rows,
        "slow_bytes_per_session": per_expert,
        "slow_bytes_per_session_pct_of_base": 100 * per_expert / base_bytes,
    }


EXAMPLE = Example(
    "18_sleep_consolidation",
    __doc__ or "",
    add_args,
    build,
    train_fn,
    test_fn,
    infer_fn,
    runtime_args=("gen", "test_batches"),
)

if __name__ == "__main__":
    run(EXAMPLE)
