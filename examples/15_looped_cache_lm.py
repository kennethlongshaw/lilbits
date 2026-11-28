"""Looped depth that keeps its weights in cache: does a weight-shared GDN block beat a plain GDN stack on CPU decode?

Decode is weight-bandwidth bound (results/fused): every byte streams every weight once, and a plain d200 x 8 W8A8
stack (5 MB) comes from L3 at ~24 GB/s. Earlier looped models lost on decode because each loop re-streamed weights
the same way. A looped block's *unique* weights are small, so this example tests whether they stay in L2 (2 MB per
core here, ~50-60 GB/s for the matvec kernel) across the loops of a byte:

    loop:D,K,R    LeanLM GDN blocks, K unique blocks applied R times per byte (K*R block applications, each with its
                  own GDN state, as example 09's per-loop state); the loops of one byte run back to back, so loop 2..R
                  reread the block that loop 1 just pulled into cache
    rloop:D,K,R   the same, trained with a random loop count in 1..R per step, so test-time depth can vary (test sweeps
                  --test-loops; infer benchmarks each)
    plain:D,L     the plain LeanLM GDN stack (L blocks), the baseline at matched quality

train   Shakespeare bytes, the results/fused recipe (AdamW 3e-3 cosine, --steps x --batch x --seq), every model on the
        same batches
test    held-out bits per byte: bpb (example 14's 8-batch set) and bpb64 (64 batches, tighter)
infer   export each model to the fused Rust engine (W8A8) and run it in fresh processes (numpy + ctypes, no torch):
          hot   looped layers alias the same weight arrays, so the engine reads one copy R times
          cold  the same model with every loop's weights copied to their own buffers: identical math and output,
                but every block application streams its own weights, the way a plain stack does. hot vs cold is
                the cache effect alone
        per (model, mode, threads): decode ms/byte and CPU ms/byte (--gen greedy bytes after a 512-byte prompt),
        prefill bytes/s, unique weight bytes, weight bytes streamed per byte, recurrent state, process RSS.
        Medians of --repeats fresh processes, run round-robin so VM drift hits every row alike.
        The lockstep threads (threads > 1) give each thread the same weight rows in every block application, so
        with aliasing a thread's slice of the shared block stays in its own core's L2.

uv run python examples/15_looped_cache_lm.py                                    # smoke run
uv run python examples/15_looped_cache_lm.py --models plain:200,8 loop:200,2,4 --steps 1000 --batch 8 --seq 128
uv run python examples/15_looped_cache_lm.py --mode infer --threads-list 1,4 --repeats 5
uv run python examples/15_looped_cache_lm.py _merge all.pt a.pt b.pt     # models trained in parallel processes
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "_bench":
    # bench worker: numpy + ctypes only, so process RSS has no torch in it. One BLAS thread: numpy's idle OpenBLAS
    # workers otherwise spin during decode and add ~0.14 ms/B of CPU that is not the engine's
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import numpy as np

    from lilbits.inference.fused import FusedLM

    def _rss_mib() -> float:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20

    def _bench(path: str, mode: str, threads: int, data: str, gen: int) -> None:
        z = dict(np.load(path))  # one read: entries below alias these arrays
        K, R = (int(x) for x in z.pop("loop"))
        layer_keys = [k for k in z if k.startswith("l") and k.split(".")[0][1:].isdigit()]
        for l in range(K, K * R):
            for k in layer_keys:
                if k.startswith(f"l{l % K}."):
                    v = z[k]
                    z[f"l{l}." + k.split(".", 1)[1]] = v if mode == "hot" else v.copy()
        m = FusedLM(z, block=64, threads=threads)
        uniq = {id(v): v.nbytes for k, v in m.arr.items() if k != "meta" and not k.endswith(".rpad")}
        ptrs = {m.arr[f"l{l}.in.w"].ctypes.data for l in range(K * R)}
        dz = np.load(data)
        prompt = dz["prompt"]
        m.prefill(prompt[:16])
        m.generate(4)  # warm up (pool start, first touch)
        m.reset()
        w, c = time.perf_counter(), time.process_time()
        m.prefill(prompt)
        pw, pc = time.perf_counter() - w, time.process_time() - c
        w, c = time.perf_counter(), time.process_time()
        out = m.generate(gen + 1)  # the first byte is the prefill's argmax
        dw, dc = time.perf_counter() - w, time.process_time() - c
        res = {
            "decode_ms_per_B": dw / gen * 1e3,
            "decode_cpu_ms_per_B": dc / gen * 1e3,
            "prefill_B_per_s": len(prompt) / pw,
            "prefill_cpu_ms_per_B": pc / len(prompt) * 1e3,
            "unique_weight_bytes": sum(uniq.values()),
            "streamed_weight_bytes_per_B": m.weight_bytes(),
            "distinct_layer_buffers": len(ptrs),
            "state_bytes": m.S.nbytes,
            "process_rss_mib": _rss_mib(),
            "tokens": out,
        }
        print("JSON " + json.dumps(res))

    _bench(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
    sys.exit(0)

import numpy as np
import torch

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "_merge":
    # _merge OUT IN...: one checkpoint with every model of the inputs (groups trained in parallel processes), so a
    # single infer run benchmarks them all round-robin. Inputs must share every argument but --models.
    cks = [torch.load(p, weights_only=False) for p in sys.argv[3:]]
    out = cks[0]
    for ck in cks[1:]:
        assert {k: v for k, v in ck["args"].items() if k != "models"} == {
            k: v for k, v in out["args"].items() if k != "models"
        }, "checkpoints differ in more than --models"
        out["args"]["models"] = out["args"]["models"] + ck["args"]["models"]
        out.update({k: v for k, v in ck.items() if k.startswith("model/")})
    torch.save(out, sys.argv[2])
    sys.exit(0)
from torch import nn

from lilbits.inference.fused import export
from lilbits.models.lean_lm import LeanLM
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import ShakespeareBytes


def source(args, split: str, seed: int):
    corpus = getattr(args, "corpus", "shakespeare")
    if corpus != "shakespeare":
        from lilbits.corpus import CorpusBytes

        return CorpusBytes(split, seed=seed, source=corpus)
    return ShakespeareBytes(split, seed=seed)


def add_args(ap):
    ap.add_argument(
        "--models",
        nargs="+",
        default=["plain:64,4", "loop:64,1,4"],
        help="plain:D,L (L blocks) or loop:D,K,R (K unique blocks x R loops); width D",
    )
    ap.add_argument("--heads", type=int, default=4, help="GDN heads (4 measured best in results/fused)")
    ap.add_argument("--steps", type=int, default=30, help="training steps per model (default: smoke run)")
    ap.add_argument(
        "--bytes-per-param",
        type=float,
        default=0.0,
        help="> 0: each model trains on this many bytes per unique parameter (overrides --steps; 20 = Chinchilla's "
        "tokens per parameter with bytes as tokens)",
    )
    ap.add_argument(
        "--corpus",
        default="shakespeare",
        choices=["shakespeare", "mixed", "fineweb_edu", "wiki40b"],
        help="shakespeare: the 1 MB Shakespeare split; mixed: lilbits.corpus (~10 MB: Shakespeare, Gutenberg, "
        "webtext); fineweb_edu / wiki40b: lilbits.corpus's ~100 MB FineWeb-Edu / Wiki40B splits",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-batches", type=int, default=64, help="bpb64 batches (the first 8 are bpb's)")
    ap.add_argument("--test-loops", default="1,2,4,8", help="looped models: loop counts for the test-time depth sweep")
    ap.add_argument("--halt", type=int, default=0, help="> 0: rloop models, halting curves on this many test batches")
    ap.add_argument("--halt-compare", type=int, default=0, help="> 0: rloop models, compute-vs-bpb of exit rules")
    ap.add_argument("--min-loops", type=int, default=1, help="--halt-compare: loops before any exit")
    ap.add_argument("--threads-list", default="1,4", help="infer: engine thread counts")
    ap.add_argument("--repeats", type=int, default=3, help="infer: fresh processes per row (median)")
    ap.add_argument("--gen", type=int, default=1024, help="infer: greedy bytes decoded after the 512-byte prompt")
    ap.add_argument("--fmt", default="w8a8", help="infer: engine weight format")
    ap.add_argument("--out", default="", help="infer: write the rows as JSON here")


def parse(spec: str) -> tuple[int, int, int]:
    """-> (width, unique blocks, loops)."""
    kind, nums = spec.split(":")
    v = [int(x) for x in nums.split(",")]
    return (v[0], v[1], 1) if kind == "plain" else (v[0], v[1], v[2])


def set_loops(m: LeanLM, k: int, r: int) -> None:
    """Point the model at its K unique blocks applied r times (test-time depth: any r, same weights)."""
    uniq = list(m.blocks)[:k]
    m.blocks = nn.ModuleList([uniq[i % k] for i in range(k * r)])


def looped_lm(d: int, k: int, r: int, heads: int) -> LeanLM:
    m = LeanLM(d, heads, k, lambda i, o: nn.Linear(i, o, bias=False), n_draft=0)
    if r > 1:  # the same K modules R times: shared weights, one GDN state per application
        m.blocks = nn.ModuleList([m.blocks[i % k] for i in range(k * r)])
    return m


def build(args, meta):
    """Each model's init is seeded from (seed, its spec), so it doesn't depend on which other models are listed:
    groups trained in parallel processes and merged give the same models as one process training them all."""
    out = {}
    for s in args.models:
        torch.manual_seed(args.seed * 1_000_003 + zlib.crc32(s.encode()))
        out[s] = looped_lm(*parse(s), args.heads)
    return out


def train_fn(models, args, fab):
    out = {}
    corpus = len(source(args, "train", 0).data)
    for name, m in models.items():
        src = source(args, "train", args.seed)  # every model sees the same batches
        n_par = sum(p.numel() for p in m.parameters())  # unique: shared blocks count once
        steps = args.steps
        if args.bytes_per_param:
            steps = math.ceil(args.bytes_per_param * n_par / (args.batch * args.seq))
        _, k, r = parse(name)
        g = torch.Generator().manual_seed(args.seed + 17)

        rand = name.startswith("rloop:")

        def loss(mod, x, m=m, k=k, r=r, g=g, rand=rand):
            if rand:  # random depth per step, so any loop count works at test time (Huginn-style)
                set_loops(m, k, int(torch.randint(1, r + 1, (1,), generator=g)))
            return mod.loss(x, 0.0)

        t = time.perf_counter()
        info = fit(
            fab,
            m,
            lambda _, src=src: src.sample(args.batch, args.seq),
            loss,
            steps,
            lr=args.lr,
            schedule="cosine",
            optimizer=torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), foreach=False),
        )
        set_loops(m, k, r)
        seen = steps * args.batch * args.seq
        out[name] = {
            "ce": info["ce"],
            "steps": steps,
            "train_bytes": seen,
            "bytes_per_param": seen / n_par,
            "epochs": seen / corpus,
            "train_s": time.perf_counter() - t,
            "loss_curve": info.get("loss_curve"),
        }
    return out, {}


def held_out(args):
    src = source(args, "test", 999)  # shakespeare: example 14's held-out sampler
    return [src.sample(8, 128) for _ in range(max(8, args.eval_batches))]


def test_fn(models, args):
    ev = held_out(args)
    out = {}
    for name, m in models.items():
        ce = [m.loss(x, 0)[0].item() for x in ev]
        out[name] = {"bpb": sum(ce[:8]) / 8 / math.log(2), "bpb64": sum(ce) / len(ce) / math.log(2)}
        out[name]["params"] = sum(p.numel() for p in m.parameters())
        _, k, r = parse(name)
        if name.startswith("loop") or name.startswith("rloop"):
            depth = {}
            for rr in (int(x) for x in args.test_loops.split(",")):  # test-time depth sweep
                set_loops(m, k, rr)
                c = [m.loss(x, 0)[0].item() for x in ev]
                depth[rr] = sum(c) / len(c) / math.log(2)
            set_loops(m, k, r)
            out[name]["bpb64_by_loops"] = depth
        if args.halt and name.startswith("rloop:"):
            hb = ev[: args.halt]
            fixed = {}
            for rr in range(1, r + 1):  # fixed depth on the same bytes (loop rr's states don't depend on deeper loops)
                set_loops(m, k, rr)
                fixed[rr] = sum(m.loss(x, 0)[0].item() for x in hb) / len(hb) / math.log(2)
            set_loops(m, k, r)
            out[name]["halt_fixed_depth_bpb"] = fixed
            for wt in (False, True):
                out[name][f"halt_{'write' if wt else 'freeze'}"] = halting_curves(m, k, r, hb, wt)
        if args.halt_compare and name.startswith("rloop:"):
            hb = ev[: args.halt_compare]
            blocks = list(m.blocks)[:k]
            full = sum(block_macs(b, b.mixer.n_heads)[0] for b in blocks)
            d = m.embed.embedding_dim
            fixed = []
            for rr in range(1, r + 1):  # fixed depth: no signal, no write-through
                set_loops(m, k, rr)
                bpb = sum(m.loss(x, 0)[0].item() for x in hb) / len(hb) / math.log(2)
                fixed.append(
                    {
                        "signal": "fixed",
                        "threshold": rr,
                        "bpb": bpb,
                        "mean_loops": rr,
                        "macs_per_byte": rr * full + 256 * d + d,
                    }
                )
            set_loops(m, k, r)
            src = source(args, "train", 4242)
            probe = fit_probe(m, k, r, [src.sample(8, 128) for _ in range(32)])
            out[name]["halt_compare"] = fixed + halting_compare(m, k, r, hb, probe, min_loops=args.min_loops)
    return out


HALT = {  # exit signal -> thresholds swept; a byte exits after loop r when the signal passes the threshold
    "kl": (1e-3, 3e-3, 1e-2, 3e-2, 0.1),  # KL(p_r || p_{r-1}) < t (Huginn)
    "dh": (0.02, 0.05, 0.1, 0.2, 0.3),  # |h_r - h_{r-1}| / |h_r| < t (relative state change, example 09)
    "accel": (0.01, 0.03, 0.1, 0.2),  # |d_r - d_{r-1}| / |h_r| < t, d_r = h_r - h_{r-1} (two-scale "acceleration")
    "conf": (0.5, 0.7, 0.8, 0.9, 0.95),  # max_b p_r(b) > t (confidence)
}


def _where_state(keep: torch.Tensor, new, old):
    def pick(n, o):
        return torch.where(keep.view(-1, *([1] * (n.dim() - 1))), n, o)

    return tuple(pick(n, o) for n, o in zip(new, old, strict=True)) if isinstance(new, tuple) else pick(new, old)


@torch.no_grad()
def halting_curves(m: LeanLM, k: int, r: int, batches, write_through: bool) -> list[dict]:
    """Byte-by-byte decode of held-out text with a per-byte exit after any loop (of K blocks), up to r loops.
    Frozen loops (write_through=False) leave their GDN states untouched for that byte ("cache holes"); with
    write_through the remaining loops still advance their states from the exited hidden state (outputs discarded)."""
    blocks = list(m.blocks)[:k]

    def run(sig: str | None, thr: float) -> dict:
        nll = loops = 0.0
        n = 0
        for x in batches:
            B, T = x.shape[0], x.shape[1] - 1
            states = [None] * (k * r)
            for t in range(T):
                h = m.embed(x[:, t : t + 1])
                active = torch.ones(B, dtype=torch.bool)
                out_logp = torch.zeros(B, 256)
                prev_h = prev_d = prev_lp = None
                for rr in range(r):
                    for j, blk in enumerate(blocks):
                        i = rr * k + j
                        h2, s2 = blk(h, states[i])
                        keep_old = states[i] is not None and not write_through
                        states[i] = _where_state(active, s2, states[i]) if keep_old else s2
                        h = torch.where(active.view(-1, 1, 1), h2, h)
                    loops += active.sum().item()
                    lp = torch.log_softmax(m.head(m.norm(h))[:, 0], -1)
                    out_logp = torch.where(active[:, None], lp, out_logp)
                    if sig is not None and rr < r - 1 and t > 0:
                        hh = h[:, 0]
                        if sig == "kl" and prev_lp is not None:
                            stop = (lp.exp() * (lp - prev_lp)).sum(-1) < thr
                        elif sig == "dh" and prev_h is not None:
                            stop = (hh - prev_h).norm(dim=-1) / hh.norm(dim=-1) < thr
                        elif sig == "accel" and prev_d is not None:
                            stop = ((hh - prev_h) - prev_d).norm(dim=-1) / hh.norm(dim=-1) < thr
                        elif sig == "conf":
                            stop = lp.exp().max(-1).values > thr
                        else:
                            stop = torch.zeros(B, dtype=torch.bool)
                        if prev_h is not None:
                            prev_d = hh - prev_h
                        prev_h, prev_lp = hh, lp
                        active = active & ~stop
                    else:
                        prev_d = None if prev_h is None else h[:, 0] - prev_h
                        prev_h, prev_lp = h[:, 0], lp
                nll -= out_logp.gather(1, x[:, t + 1 : t + 2]).sum().item()
                n += B
        return {"signal": sig or "full", "threshold": thr, "bpb": nll / n / math.log(2), "mean_loops": loops / n}

    rows = [run(None, 0.0)]
    for sig, ts in HALT.items():
        rows += [run(sig, t) for t in ts]
    return rows


COMPARE = {  # finer threshold grids for the efficiency comparison (--halt-compare)
    "kl": (3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 6e-2, 0.1, 0.2),
    "accel": (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4),
    "probe": (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2),  # predicted remaining gain (nats) < t
}


def block_macs(blk, h: int) -> tuple[int, int, int]:
    """-> (MACs of one block application, its write-only k/v/a/b projection MACs, its write-only state update MACs)."""
    mx, d = blk.mixer, blk.n1.normalized_shape[0]
    lin = sum(mod.in_features * mod.out_features for mod in blk.modules() if isinstance(mod, nn.Linear))
    dh = d // h
    rec = 4 * h * dh * dh  # S^T k, S^T q, rank-1 update (two passes)
    write = sum(getattr(mx, n).in_features * getattr(mx, n).out_features for n in ("k", "v", "a", "b"))
    return lin + rec, write, 2 * h * dh * dh


def _features(hh, prev_h, prev_d, rr: int, r: int):
    """Exit-probe features after loop rr: normed hidden, |dh|/|h|, |accel|/|h|, loop one-hot, bias."""
    nrm = hh.norm(dim=-1, keepdim=True)
    d = (hh - prev_h) if prev_h is not None else torch.zeros_like(hh)
    a = (d - prev_d) if prev_d is not None else torch.zeros_like(hh)
    oh = torch.zeros(hh.shape[0], r)
    oh[:, rr] = 1
    return torch.cat([hh / nrm, d.norm(dim=-1, keepdim=True) / nrm, a.norm(dim=-1, keepdim=True) / nrm, oh], -1)


@torch.no_grad()
def fit_probe(m: LeanLM, k: int, r: int, batches, ridge: float = 1.0) -> torch.Tensor:
    """Ouro-stage-2-style exit probe, fit post hoc on full-depth trajectories of training text: ridge regression of the
    remaining gain CE_r - min_{r' > r} CE_r' (nats, >= 0) on O(d) features; exit when the prediction is < t."""
    blocks = list(m.blocks)[:k]
    X, Y = [], []
    for x in batches:
        h = m.embed(x[:, :-1])
        ces, feats, prev_h, prev_d = [], [], None, None
        for rr in range(r):
            for blk in blocks:
                h, _ = blk(h, None)
            ce = -torch.log_softmax(m.head(m.norm(h)), -1).gather(-1, x[:, 1:, None])[..., 0]
            hh = h.reshape(-1, h.shape[-1])
            feats.append(_features(hh, prev_h, prev_d, rr, r))
            prev_d = None if prev_h is None else hh - prev_h
            prev_h = hh
            ces.append(ce.reshape(-1))
        ce = torch.stack(ces)  # (r, N)
        for rr in range(r - 1):
            X.append(feats[rr])
            Y.append((ce[rr] - ce[rr + 1 :].min(0).values).clamp(min=0))
    X, Y = torch.cat(X).double(), torch.cat(Y).double()
    w = torch.linalg.solve(X.T @ X + ridge * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y)
    return w.float()


@torch.no_grad()
def halting_compare(m: LeanLM, k: int, r: int, batches, probe: torch.Tensor, min_loops: int = 1) -> list[dict]:
    """Byte-by-byte decode with write-through exits (skipped loops still advance their GDN states from the exited
    hidden state). Each row: bpb and MACs per byte including the exit signal's own cost and the write-through cost.
    KL needs the head at every loop; accel and the probe are O(d)."""
    blocks = list(m.blocks)[:k]
    d = m.embed.embedding_dim
    full, wproj, wstate = (sum(x) for x in zip(*(block_macs(b, b.mixer.n_heads) for b in blocks), strict=True))
    head = 256 * d

    def run(sig: str | None, thr: float) -> dict:
        nll = macs = loops = 0.0
        n = 0
        for x in batches:
            B, T = x.shape[0], x.shape[1] - 1
            states = [None] * (k * r)
            for t in range(T):
                h = m.embed(x[:, t : t + 1])
                active = torch.ones(B, dtype=torch.bool)
                out_lp = torch.zeros(B, 256)
                prev_h = prev_d = prev_lp = None
                for rr in range(r):
                    was = active.clone()
                    for j, blk in enumerate(blocks):
                        h2, states[rr * k + j] = blk(h, states[rr * k + j])  # write-through for exited bytes
                        h = torch.where(active.view(-1, 1, 1), h2, h)
                    na = was.sum().item()
                    loops += na
                    macs += na * full + (B - na) * wstate  # skipped loops: state updates only
                    hh = h[:, 0]
                    need_head = sig == "kl" or rr == r - 1
                    lp = torch.log_softmax(m.head(m.norm(h))[:, 0], -1)
                    out_lp = torch.where(was[:, None], lp, out_lp)
                    if sig is not None and rr < r - 1 and t > 0 and rr + 1 >= min_loops:
                        if sig == "kl":
                            stop = (lp.exp() * (lp - prev_lp)).sum(-1) < thr if prev_lp is not None else ~was
                        elif sig == "accel":
                            stop = (
                                ((hh - prev_h) - prev_d).norm(dim=-1) / hh.norm(dim=-1) < thr
                                if prev_d is not None
                                else ~was
                            )
                        else:
                            stop = _features(hh, prev_h, prev_d, rr, r) @ probe < thr
                        macs += na * (head if need_head and rr < r - 1 else 4 * d + (d if sig == "probe" else 0))
                        newly = was & stop
                        macs += newly.sum().item() * wproj  # write-only k/v/a/b of the exited state, once
                        if sig == "kl":
                            macs -= newly.sum().item() * head  # its readout is the head already computed
                        active = active & ~stop
                    prev_d = None if prev_h is None else hh - prev_h
                    prev_h, prev_lp = hh, lp
                macs += B * head  # the readout of the exited (or last) loop
                nll -= out_lp.gather(1, x[:, t + 1 : t + 2]).sum().item()
                n += B
        return {
            "signal": sig or f"fixed {r}",
            "threshold": thr,
            "bpb": nll / n / math.log(2),
            "mean_loops": loops / n,
            "macs_per_byte": macs / n + d,  # + embedding lookup
        }

    rows = []
    for sig, ts in COMPARE.items():
        rows += [run(sig, t) for t in ts]
    return rows


def save_unique(m: LeanLM, k: int, r: int, fmt: str, path: Path) -> None:
    """Exported arrays with only the K unique blocks (l0..l{K-1}) plus loop=(K, R); the worker re-expands them."""
    arr = export(m, fmt)
    keep = {key: v for key, v in arr.items() if not (key[0] == "l" and int(key.split(".")[0][1:]) >= k)}
    keep["meta"] = keep["meta"].copy()
    keep["meta"][6] = k * r  # layer applications
    np.savez(path, **keep, loop=np.array([k, r], np.int64))


def infer_fn(models, args):
    tmp = Path(tempfile.mkdtemp(prefix="lilbits15_"))
    data = tmp / "data.npz"
    prompt = source(args, "test", 7).sample(1, 512)[0, :512]
    np.savez(data, prompt=prompt.numpy().astype(np.int32))
    jobs = []
    for name, m in models.items():
        _, k, r = parse(name)
        variants = [(name, r)]
        if name.startswith("rloop:"):  # random-depth models: decode cost at every test-time loop count
            variants = [(f"{name}@{rr}", rr) for rr in (int(x) for x in args.test_loops.split(","))]
        for label, rr in variants:
            p = tmp / f"{label.replace(':', '_').replace(',', '_').replace('@', '_at')}.npz"
            save_unique(m, k, rr, args.fmt, p)
            for th in (int(x) for x in args.threads_list.split(",")):
                for mode in ("hot", "cold") if rr > 1 and not name.startswith("rloop:") else ("hot",):
                    jobs.append((label, mode, th, p))
    runs: dict = {j[:3]: [] for j in jobs}
    for _ in range(args.repeats):  # round-robin: drift on this shared VM lands on every row alike
        for name, mode, th, p in jobs:
            cmd = [sys.executable, __file__, "_bench", str(p), mode, str(th), str(data), str(args.gen)]
            so = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            runs[(name, mode, th)].append(json.loads(so.split("JSON ", 1)[1]))
    rows = []
    for (name, mode, th), rs in runs.items():
        row = {k: statistics.median(x[k] for x in rs) for k, v in rs[0].items() if isinstance(v, (int, float))}
        row["decode_ms_per_B_best"] = min(x["decode_ms_per_B"] for x in rs)
        row.update(model=name, mode=mode, threads=th, tokens=rs[0]["tokens"])
        rows.append(row)
    for row in rows:  # hot and cold run the same math: their greedy text must match
        twin = next((x for x in rows if x["model"] == row["model"] and x["threads"] == row["threads"]), row)
        row["same_text_as_hot"] = row["tokens"] == twin["tokens"]
    for row in rows:
        row["sample"] = bytes(row.pop("tokens")[:48]).decode(errors="replace")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    cols = (
        "model | weights | threads | decode ms/B (best) | CPU ms/B | prefill B/s | unique MB | streamed MB/B | RSS MiB"
    )
    print(f"\n| {cols} |")
    print("|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        print(
            f"| {x['model']} | {x['mode']} | {x['threads']}"
            f" | {x['decode_ms_per_B']:.3f} ({x['decode_ms_per_B_best']:.3f})"
            f" | {x['decode_cpu_ms_per_B']:.3f} | {x['prefill_B_per_s']:,.0f} | {x['unique_weight_bytes'] / 1e6:.2f}"
            f" | {x['streamed_weight_bytes_per_B'] / 1e6:.2f} | {x['process_rss_mib']:.0f} |"
        )
    return {f"{x['model']} {x['mode']} t{x['threads']}": round(x["decode_ms_per_B"], 4) for x in rows}


EXAMPLE = Example(
    "15_looped_cache_lm",
    __doc__ or "",
    add_args,
    build,
    train_fn,
    test_fn,
    infer_fn,
    runtime_args=("halt", "halt_compare", "min_loops", "test_loops", "threads_list", "repeats", "gen", "fmt", "out"),
)

if __name__ == "__main__":
    run(EXAMPLE)
