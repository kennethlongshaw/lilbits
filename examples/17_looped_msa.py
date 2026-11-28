"""Looped depth over MiniMax Sparse Attention (models/looped_msa.py) vs plain GDN, on the ~10 MB byte corpus.

msa_l3     2 GDN prelude blocks, then ONE shared MSA+SwiGLU core looped 3x over a K/V cache written once per byte.
           MSA: per-group index query, shared index key, max-pooled block scores, top-k blocks (own block forced),
           KL-trained indexer with stop-gradient; dense attention + KL for the first --dense-frac, then sparse
msa_l1     the same model with 1 loop (what the loop adds)
dense_l3   msa_l3 with dense attention over the cache throughout (what sparsity costs)
gdn_p      plain GDN, layers chosen so params match msa_l3
gdn_c      plain GDN with 2 + loops layers: the same number of block applications per byte as msa_l3

train   CorpusBytes (lilbits.corpus) train split, Lightning Fabric (runner.fit), cosine schedule
test    held-out bits per byte on the test split; msa_l3 also at 1..5 test-time loops and with sparsity off
infer   per model, batch 1 after a --prompt byte prefill: decode ms/byte, CPU ms/byte, attended keys per loop,
        memory that grows with context (K/V + index keys) vs GDN's fixed state, peak RSS

uv run python examples/17_looped_msa.py                                   # smoke run
uv run python examples/17_looped_msa.py --d 128 --bytes-per-param 20 --models msa_l3,gdn_p
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from lilbits.corpus import CorpusBytes
from lilbits.measure import Meter
from lilbits.models.looped_msa import LoopedMSALM
from lilbits.models.stateful_lm import StatefulLM, state_bytes
from lilbits.runner import Example, fit, run

ALL = ("msa_l3", "msa_l1", "dense_l3", "gdn_p", "gdn_c")


def add_args(ap):
    ap.add_argument("--models", default=",".join(ALL), help=f"comma list from {ALL}")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--gdn-heads", type=int, default=4)
    ap.add_argument("--prelude", type=int, default=2, help="GDN blocks before the looped core")
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--groups", type=int, default=2, help="GQA K/V groups (one index query head each)")
    ap.add_argument("--d-idx", type=int, default=32, help="index head width")
    ap.add_argument("--block", type=int, default=16, help="MSA key block, bytes")
    ap.add_argument("--topk", type=int, default=4, help="MSA blocks per query and group (own block included)")
    ap.add_argument("--dense-frac", type=float, default=0.2, help="share of steps with dense attention + KL")
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument(
        "--bytes-per-param",
        type=float,
        default=0,
        help="> 0: overrides --steps so every model sees this many training bytes per msa_l3 parameter "
        "(Chinchilla-style: 20). All models get the same bytes, so they compare at equal data.",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--prompt", type=int, default=512, help="infer: prompt bytes")
    ap.add_argument("--gen", type=int, default=128, help="infer: bytes to decode")


def _msa(args, loops: int) -> LoopedMSALM:
    return LoopedMSALM(
        args.d, args.gdn_heads, args.prelude, loops, args.attn_heads, args.groups, args.d_idx, args.block, args.topk
    )


def _gdn(args, layers: int) -> StatefulLM:
    return StatefulLM(256, args.d, args.gdn_heads, layers, lambda i, o: nn.Linear(i, o, bias=False))


def _params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def param_matched_layers(args) -> int:
    with torch.device("meta"):
        target = _params(_msa(args, args.loops))
        return min(range(1, 12), key=lambda n: abs(_params(_gdn(args, n)) - target))


def build(args, meta):
    out: dict[str, nn.Module] = {}
    for name in args.models.split(","):
        if name == "msa_l3":
            out[name] = _msa(args, args.loops)
        elif name == "msa_l1":
            out[name] = _msa(args, 1)
        elif name == "dense_l3":
            m = _msa(args, args.loops)
            m.sparse = False
            out[name] = m
        elif name == "gdn_p":
            out[name] = _gdn(args, meta.get("gdn_p_layers") or param_matched_layers(args))
        elif name == "gdn_c":
            out[name] = _gdn(args, args.prelude + args.loops)
        else:
            raise ValueError(name)
    return out


def gdn_loss(m: StatefulLM, x: torch.Tensor):
    logits, _ = m(x[:, :-1])
    ce = nn.functional.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())
    return ce, {"ce": ce.item()}


def train_fn(models, args, fab):
    out: dict = {}
    if args.bytes_per_param > 0:
        with torch.device("meta"):
            ref = _params(_msa(args, args.loops))
        args.steps = math.ceil(args.bytes_per_param * ref / (args.batch * args.seq))
    train_bytes = args.steps * args.batch * args.seq
    for name, m in models.items():
        src = CorpusBytes("train", seed=args.seed)

        def batch(_, src=src):
            return src.sample(args.batch, args.seq)

        with Meter() as mt:
            if isinstance(m, LoopedMSALM):
                sparse_final = m.sparse
                kw = args.kl_weight if sparse_final else 0.0
                n_dense = round(args.dense_frac * args.steps) if sparse_final else args.steps
                phases = [(n_dense, False), (args.steps - n_dense, True)]
                info: dict = {}
                # one optimizer and one cosine schedule across both phases: fit's step counter restarts per call,
                # so the schedule is run by hand here
                opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95))
                wrapped, opt = fab.setup(m, opt)
                n_warm, step, losses = max(1, int(0.05 * args.steps)), 0, []
                for n, sparse in phases:
                    m.sparse = sparse
                    for _ in range(n):
                        f = (
                            (step + 1) / n_warm
                            if step < n_warm
                            else 0.1 + 0.45 * (1 + math.cos(math.pi * (step - n_warm) / max(1, args.steps - n_warm)))
                        )
                        for g in opt.param_groups:
                            g["lr"] = args.lr * f
                        loss, info = m.loss(fab.to_device(batch(step)), kw)
                        opt.zero_grad(set_to_none=True)
                        fab.backward(loss)
                        fab.clip_gradients(wrapped, opt, max_norm=1.0)
                        opt.step()
                        losses.append(info["ce"])
                        step += 1
                m.sparse = sparse_final
                span = max(1, len(losses) // 10)
                info = {
                    "steps": args.steps,
                    "dense_steps": n_dense,
                    **info,
                    "ce_curve": [sum(losses[i * span : (i + 1) * span]) / span for i in range(10)]
                    if len(losses) >= 20
                    else losses,
                }
            else:
                info = fit(fab, m, batch, lambda mod, x: gdn_loss(mod, x), args.steps, lr=args.lr, schedule="cosine")
        out[name] = {
            **info,
            **mt.as_dict(),
            "params": _params(m),
            "train_bytes": train_bytes,
            "bytes_per_param": train_bytes / _params(m),
            "epochs": train_bytes / len(src),
        }
    gp = models.get("gdn_p")
    return out, {"gdn_p_layers": len(gp.blocks) if gp is not None else None}


def held_out(args):
    src = CorpusBytes("test", seed=999)
    return [src.sample(args.batch, args.seq) for _ in range(args.eval_batches)]


@torch.no_grad()
def bpb(m: nn.Module, batches, loops: int | None = None) -> float:
    tot = 0.0
    for x in batches:
        if isinstance(m, LoopedMSALM):
            logits, _, _ = m(x[:, :-1], loops=loops)
        else:
            logits, _ = m(x[:, :-1])
        tot += nn.functional.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()).item()
    return tot / len(batches) / math.log(2)


def test_fn(models, args):
    ev = held_out(args)
    out: dict = {}
    for name, m in models.items():
        m.eval()
        res: dict = {"bpb": bpb(m, ev), "params": _params(m)}
        if isinstance(m, LoopedMSALM) and m.sparse:
            res["bpb_by_test_loops"] = {n: bpb(m, ev, n) for n in range(1, 6)}
            m.sparse = False
            res["bpb_sparsity_off"] = bpb(m, ev)
            m.sparse = True
            _, _, att = m(ev[0][:, :-1])
            res["mean_attended_keys_per_query"] = att.item()
        out[name] = res
    return out


@torch.no_grad()
def _gdn_decode(m: StatefulLM, prompt: torch.Tensor, n: int):
    logits, states = m(prompt)
    tok = logits[:, -1:].argmax(-1)
    t, c = time.perf_counter(), time.process_time()
    for _ in range(n):
        logits, states = m(tok, states)
        tok = logits[:, -1:].argmax(-1)
    return (time.perf_counter() - t) / n * 1e3, (time.process_time() - c) / n * 1e3, states


@torch.no_grad()
def _msa_decode(m: LoopedMSALM, prompt: torch.Tensor, n: int):
    """Decode-only time: generate(n+1) minus a prefill-only generate(1), so the prompt prefill is not counted."""
    m.generate(prompt, 1)
    t, c = time.perf_counter(), time.process_time()
    m.generate(prompt, 1)
    pt, pc = time.perf_counter() - t, time.process_time() - c
    t, c = time.perf_counter(), time.process_time()
    _, info = m.generate(prompt, n + 1)
    wt, wc = time.perf_counter() - t, time.process_time() - c
    return (wt - pt) / n * 1e3, (wc - pc) / n * 1e3, info


def infer_fn(models, args):
    prompt = CorpusBytes("test", seed=7).sample(1, args.prompt)[:, : args.prompt]
    T = args.prompt + args.gen
    out: dict = {"threads": torch.get_num_threads(), "context_bytes": T}
    for name, m in models.items():
        m.eval()
        with Meter() as mt:
            res: dict = {}
            if isinstance(m, LoopedMSALM):
                m.generate(prompt[:, :32], 4)  # warm up
                res["decode_ms_per_byte"], res["decode_cpu_ms_per_byte"], info = _msa_decode(m, prompt, args.gen)
                res.update(info)
                _, states = m.prelude(prompt)
                res["fixed_state_bytes"] = state_bytes(states)
                res["growing_cache_bytes_fp16"] = m.cache_bytes(T)
                gen, _ = m.generate(prompt, 60)
            else:
                m.generate(prompt[:, :32], 4)
                res["decode_ms_per_byte"], res["decode_cpu_ms_per_byte"], states = _gdn_decode(m, prompt, args.gen)
                res["fixed_state_bytes"] = state_bytes(states)
                res["growing_cache_bytes_fp16"] = 0
                gen = m.generate(prompt, 60)[0][:, args.prompt :]
            res["sample"] = bytes(gen[0].tolist()).decode(errors="replace")
        out[name] = {**res, **mt.as_dict()}
    return out


EXAMPLE = Example(
    "17_looped_msa", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("gen", "prompt")
)

if __name__ == "__main__":
    run(EXAMPLE)
