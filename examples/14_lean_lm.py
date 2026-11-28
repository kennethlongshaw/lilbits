"""The clear wins in one byte-level LM (models/lean_lm.py), next to FP16 models of the same size and the same width.

lean          GDN blocks (fixed-size state, chunked training kernel) whose block linears become LittleBit
              (--bpw, SVD+ITQ conversion after --fp-frac of training in FP), MTP draft heads, packed to 1-bit
              signs on the LUT kernels for inference, self-speculative decoding with state rollback
fp16_size     the same GDN stack in FP, width solved so its deployed FP16 bytes match lean's
fp16_width    the same GDN stack in FP at lean's width (the quality ceiling lean gives up for its size)

train   Shakespeare bytes, Lightning Fabric (runner.fit), cosine schedule; lean: FP phase -> convert -> LittleBit
test    held-out bits per byte (test split)
infer   per model: prefill bytes/s, greedy decode ms/byte, recurrent state bytes vs an attention K/V cache at the
        same length, peak RSS; lean also: LUT-packed bpb, speculative decode ms/byte, draft acceptance, bytes per
        model call, and whether the speculative output equals greedy byte for byte; rollback by keeping the verify
            call's per-byte GDN states (default) vs replaying the accepted prefix from a snapshot (example 03)

uv run python examples/14_lean_lm.py                       # smoke run
uv run python examples/14_lean_lm.py --mode infer --gen 256
"""

from __future__ import annotations

import gc
import math
import time

import torch
from torch import nn

from lilbits.inference.lut import kernel_info, pack_lut
from lilbits.inference.packed import pack_model
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.measure import Meter, size_report
from lilbits.models.lean_lm import LeanLM, convert_linears
from lilbits.models.stateful_lm import state_bytes
from lilbits.runner import Example, fit, run
from lilbits.shakespeare import ShakespeareBytes

FP = nn.Linear


def add_args(ap):
    ap.add_argument("--d", type=int, default=256, help="lean width")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=0, help="> 0: heads = width // head-dim (overrides --heads)")
    ap.add_argument("--baselines", default="size,width", help="FP16 baselines to train: size, width (comma list)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--bpw", type=float, default=0.55, help="LittleBit bits per weight of the block linears")
    ap.add_argument("--drafts", type=int, default=3, help="MTP draft heads (speculative bytes per round)")
    ap.add_argument("--draft-rank", type=int, default=32, help="rank of each low-rank draft head")
    ap.add_argument("--mtp-weight", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=30, help="training steps per model (default: smoke run)")
    ap.add_argument("--grad-ckpt", action="store_true", help="lean: recompute each block in backward (big widths)")
    ap.add_argument("--itq", type=int, default=50, help="Joint-ITQ iterations in the LittleBit conversion")
    ap.add_argument("--fp-frac", type=float, default=0.3, help="lean: share of steps in FP before conversion")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--prompt", type=int, default=512, help="infer: prompt bytes")
    ap.add_argument("--gen", type=int, default=128, help="infer: bytes to decode")


def _heads(args, d: int) -> int:
    if not getattr(args, "head_dim", 0):
        return args.heads
    h = max(1, d // args.head_dim)
    while d % h:
        h -= 1
    return h


def _fp(d: int, args, drafts: int = 0) -> LeanLM:
    return LeanLM(
        d, _heads(args, d), args.layers, lambda i, o: FP(i, o, bias=False), n_draft=drafts, draft_rank=args.draft_rank
    )


def _shell(m: LeanLM, bpw: float) -> LeanLM:
    """Swap the block linears for untrained LittleBit layers of the same shapes (no SVD): for loading a converted
    checkpoint and for sizing on the meta device."""
    for parent in m.blocks.modules():
        for name, c in list(parent.named_children()):
            if isinstance(c, nn.Linear) and c.out_features >= 32:
                setattr(
                    parent, name, LittleBitLinear(c.in_features, c.out_features, bpw=bpw, residual=True, ste="clipped")
                )
    return m


def deployed(m: nn.Module) -> float:
    return size_report(m)["deployed_total_bytes"]


def matched_width(args) -> int:
    """FP width (multiple of 8) whose deployed FP16 bytes are closest to lean's, sized on the meta device."""
    with torch.device("meta"):
        target = deployed(_shell(_fp(args.d, args, args.drafts), args.bpw))

        def size(w: int) -> float:
            return deployed(_fp(w, args))

        lo, hi = 1, args.d // 8  # deployed bytes grow with width: binary search on width / 8
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if size(8 * mid) < target else (lo, mid)
        return min((8 * max(1, lo - 1), 8 * lo), key=lambda w: abs(size(w) - target))


def build(args, meta):
    if meta.get("converted"):
        lean = _shell(_fp(args.d, args, args.drafts), args.bpw)
    else:
        lean = _fp(args.d, args, args.drafts)
    models = {"lean": lean}
    base = getattr(args, "baselines", "size,width").split(",")
    if "size" in base:
        models["fp16_size"] = _fp(meta.get("fp_d") or matched_width(args), args)
    if "width" in base:
        models["fp16_width"] = _fp(args.d, args)
    return models


def bpb(model: LeanLM, batches) -> float:
    with torch.no_grad():
        return sum(model.loss(x, 0)[0].item() for x in batches) / len(batches) / math.log(2)


def held_out(args):
    src = ShakespeareBytes("test", seed=999)
    return [src.sample(args.batch, args.seq) for _ in range(args.eval_batches)]


def adamw(m: nn.Module, args) -> torch.optim.Optimizer:
    """AdamW without the multi-tensor path: foreach builds whole-model temporaries, which at ~50 MB deployed
    (~360M LittleBit latents) pushed peak RAM past 14 GB."""
    return torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), foreach=False)


def train_fn(models, args, fab):
    out: dict = {}
    for name, m in models.items():
        src = ShakespeareBytes("train", seed=args.seed)
        w = args.mtp_weight if name == "lean" else 0.0

        def loss(mod, x, w=w):
            return mod.loss(x, w)

        def batch(_, src=src):
            return src.sample(args.batch, args.seq)

        with Meter() as mt:
            if name == "lean":
                n_fp = round(args.fp_frac * args.steps)
                fp_info = (
                    fit(fab, m, batch, loss, n_fp, lr=args.lr, schedule="cosine", optimizer=adamw(m, args))
                    if n_fp
                    else {}
                )
                t = time.perf_counter()
                n_conv = convert_linears(m, args.bpw, itq_iters=args.itq)
                conv_s = time.perf_counter() - t
                gc.collect()  # the dense FP weights the conversion replaced
                m.grad_ckpt = args.grad_ckpt
                info = fit(
                    fab, m, batch, loss, args.steps - n_fp, lr=args.lr, schedule="cosine", optimizer=adamw(m, args)
                )
                info = {"fp_phase": fp_info, "converted_layers": n_conv, "convert_s": conv_s, **info}
            else:
                info = fit(fab, m, batch, loss, args.steps, lr=args.lr, schedule="cosine", optimizer=adamw(m, args))
        out[name] = {**info, **mt.as_dict(), "width": m.embed.embedding_dim}
    fp = models.get("fp16_size")
    return out, {"converted": True, "fp_d": fp.embed.embedding_dim if fp is not None else None}


def test_fn(models, args):
    ev = held_out(args)
    return {name: {"bpb": bpb(m, ev)} for name, m in models.items()}


def _decode_ms(fn, n: int) -> tuple[float, torch.Tensor]:
    t = time.perf_counter()
    toks = fn()
    return (time.perf_counter() - t) / n * 1e3, toks


@torch.no_grad()
def _decode_only_ms(m: LeanLM, prompt: torch.Tensor, greedy: torch.Tensor) -> float:
    """ms per byte of single-byte decode steps from a prefilled state, replaying the greedy bytes."""
    _, states = m(prompt)
    toks = greedy[:, :-1]  # the bytes generate() fed back after the prefill
    t = time.perf_counter()
    for i in range(toks.shape[1]):
        _, states = m(toks[:, i : i + 1], states)
    return (time.perf_counter() - t) / toks.shape[1] * 1e3


def infer_fn(models, args):
    prompt = ShakespeareBytes("test", seed=7).sample(1, args.prompt)[:, : args.prompt]
    out: dict = {"lut_kernel": kernel_info(), "threads": torch.get_num_threads()}
    ev = held_out(args)[:2]
    for name, m in models.items():
        with Meter() as mt:
            res: dict = {}
            if name == "lean":
                pack_lut(pack_model(m))
                res["packed_lut_bpb"] = bpb(m, ev)
            m.eval()
            m.generate(prompt[:, :16], 4)  # warm up
            t = time.perf_counter()
            _, states = m(prompt)
            res["prefill_bytes_per_s"] = args.prompt / (time.perf_counter() - t)
            greedy_total_ms, greedy = _decode_ms(lambda m=m: m.generate(prompt, args.gen), args.gen)
            # generate() includes the prompt prefill; report decode alone (per-byte steps after prefill)
            res["greedy_ms_per_byte"] = _decode_only_ms(m, prompt, greedy)
            res["greedy_incl_prefill_ms_per_byte"] = greedy_total_ms
            if name == "lean":
                t = time.perf_counter()
                spec, stats = m.speculative_generate(prompt, args.gen)
                res["speculative_ms_per_byte"] = (time.perf_counter() - t) / args.gen * 1e3
                res["speculative_speedup"] = res["greedy_ms_per_byte"] / res["speculative_ms_per_byte"]
                res["speculative_equals_greedy"] = bool(torch.equal(spec, greedy))
                res.update(stats)
                t = time.perf_counter()
                spec_r, stats_r = m.speculative_generate(prompt, args.gen, replay=True)
                res["replay_rollback"] = {
                    "ms_per_byte": (time.perf_counter() - t) / args.gen * 1e3,
                    "equals_greedy": bool(torch.equal(spec_r, greedy)),
                    **stats_r,
                }
            T, d = args.prompt + args.gen, m.embed.embedding_dim
            res["state_bytes"] = state_bytes(states)
            res["attention_kv_bytes_same_length_fp16"] = 2 * 2 * d * args.layers * T
            res["sample"] = bytes(greedy[0, :60].tolist()).decode(errors="replace")
        out[name] = {**res, **mt.as_dict()}
    return out


EXAMPLE = Example(
    "14_lean_lm", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn, runtime_args=("gen", "prompt")
)

if __name__ == "__main__":
    run(EXAMPLE)
