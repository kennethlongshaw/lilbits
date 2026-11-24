"""End-to-end inference demo: train LittleBit -> pack to 1-bit storage -> run with torch and Helion backends.

Compares the trained (unpacked) model, the packed model on the PyTorch backend, and the packed model
on the Helion kernels: eval loss on held-out data, logits agreement, and greedy generation.

    HELION_INTERPRET=1 uv run --extra gpu python examples/05_pack_and_helion.py  # CPU: Helion eager reference mode
    uv run --extra gpu python examples/05_pack_and_helion.py --device cuda  # GPU: compiled Helion kernels
"""

from __future__ import annotations

import argparse
import io
import json

import torch
import torch.nn.functional as F

from lilbits.data import MarkovSource
from lilbits.inference.packed import model_bytes, pack_model, set_backend
from lilbits.measure import Meter, size_report
from lilbits.models.tinylm import Config
from lilbits.training.train import TrainConfig, train


@torch.no_grad()
def eval_loss(model, batches) -> float:
    return sum(F.cross_entropy(model(b[:, :-1]).flatten(0, 1), b[:, 1:].flatten()).item() for b in batches) / len(
        batches
    )


@torch.no_grad()
def generate(model, prompt: torch.Tensor, n: int, seq_len: int) -> torch.Tensor:
    out = prompt.clone()
    for _ in range(n):
        logits = model(out[:, -seq_len:])
        out = torch.cat([out, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=60)  # smoke run; raise to actually train
    ap.add_argument("--fp-warmup", type=int, default=20)
    ap.add_argument("--bpw", type=float, default=1.41)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    dev = args.device

    src = MarkovSource(vocab=32, seed=0)
    cfg = Config(
        vocab_size=32, seq_len=64, d_model=128, n_heads=4, n_layers=2, linear="littlebit", bpw=args.bpw, itq_iters=50
    )
    print(
        f"1) training LittleBit ({args.bpw} bpw) for {args.steps} steps ({args.fp_warmup} FP warm start)...", flush=True
    )
    with Meter() as train_meter:
        res = train(
            cfg,
            TrainConfig(
                steps=args.steps,
                log_every=args.steps,
                fp_warmup=args.fp_warmup,
                verbose=False,
                accelerator="cuda" if dev == "cuda" else "cpu",
            ),
            src,
        )
    model = res["model"].eval()
    stored = size_report(model)["deployed_total_bytes"]
    print(f"   done in {train_meter.wall_s:.0f}s, eval loss {res['final_eval']:.4f}")

    held_out = [src.sample(16, 64).to(dev) for _ in range(2)]
    floor = sum(src.entropy_floor(b.cpu()) for b in held_out) / 2
    prompt = held_out[0][:2, :8]

    ref_loss = eval_loss(model, held_out)
    ref_logits = model(held_out[0][:, :-1]).detach()
    ref_gen = generate(model, prompt, args.gen, cfg.seq_len)
    ref_bytes = model_bytes(model)

    print("2) packing: 1-bit signs (uint8), FP16 scales, FP32 latents dropped")
    pack_model(model)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    rows = [("trained, unpacked (FP32 latents)", ref_loss, 0.0, True, ref_bytes, None)]
    gens = {"unpacked": ref_gen}
    meters = {}
    for backend in ("torch", "helion"):
        set_backend(model, backend)
        with Meter() as m:
            loss = eval_loss(model, held_out)
            logits = model(held_out[0][:, :-1])
            gen = generate(model, prompt, args.gen, cfg.seq_len)
        meters[backend] = m
        gens[backend] = gen
        rows.append(
            (
                f"packed, {backend} backend",
                loss,
                (logits - ref_logits).abs().max().item(),
                torch.equal(gen, ref_gen),
                model_bytes(model),
                m.wall_s,
            )
        )

    print("\n3) inference on held-out sequences\n")
    print(
        f"| model | held-out loss (floor {floor:.3f}) | max logit diff vs unpacked "
        "| greedy generation identical | weights in RAM | time |"
    )
    print("|---|---|---|---|---|---|")
    for name, loss, diff, same, nbytes, secs in rows:
        print(
            f"| {name} | {loss:.4f} | {diff:.2e} | {'yes' if same else 'NO'} | {nbytes / 1024:,.0f} KiB | "
            f"{'' if secs is None else f'{secs:.1f}s'} |"
        )
    print(f"\npacked checkpoint on disk: {buf.tell() / 1024:,.0f} KiB")
    print(
        "METRICS "
        + json.dumps(
            {
                "train_steps": args.steps,
                **train_meter.as_dict("train_"),
                **meters["torch"].as_dict("infer_torch_"),
                **meters["helion"].as_dict("infer_helion_"),
                "ram_weights_bytes": model_bytes(model),
                "checkpoint_bytes": buf.tell(),
                "stored_bytes": stored,
            }
        )
    )
    print("\ngreedy continuation of the first prompt (prompt | generated):")
    p = prompt.shape[1]
    for k, g in gens.items():
        g = g.cpu()
        ok = sum(src.probs[src._ctx_index(g, t), g[:, t]].gt(0).sum().item() for t in range(p, g.shape[1]))
        total = g.shape[0] * (g.shape[1] - p)
        print(f"  {k:9s} {g[0, :p].tolist()} | {g[0, p:].tolist()}   (legal under the true chain: {ok}/{total})")


if __name__ == "__main__":
    main()
