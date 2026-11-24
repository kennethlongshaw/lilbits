"""Train a tiny LM whose block linears are LittleBit (or BitNet), then pack it for inference.

    uv run python examples/01_train_and_pack.py --linear littlebit --bpw 0.8
    uv run python examples/01_train_and_pack.py --linear bitnet
"""

import argparse
import json

import torch

from littlebit_scratch.bitnet import pack_bitnet
from littlebit_scratch.measure import Meter, size_report, state_dict_bytes
from littlebit_scratch.data import MarkovSource
from littlebit_scratch.model import Config
from littlebit_scratch.packed import model_bytes, pack_model
from littlebit_scratch.train import TrainConfig, evaluate, train

ap = argparse.ArgumentParser()
ap.add_argument("--linear", default="littlebit", choices=["fp", "littlebit", "bitnet"])
ap.add_argument("--bpw", type=float, default=1.41)
ap.add_argument("--steps", type=int, default=60)  # smoke run; raise to actually train
args = ap.parse_args()

src = MarkovSource(vocab=32, seed=0)
evals = [src.sample(32, 64) for _ in range(4)]
cfg = Config(linear=args.linear, bpw=args.bpw, itq_iters=50)
# LittleBit trains best from a short FP warm start that is SVD+ITQ-converted; BitNet trains directly.
tc = TrainConfig(steps=args.steps, optim="muon_ds", lr=1e-2, verbose=False, accelerator="cpu",
                 fp_warmup=args.steps // 3 if args.linear == "littlebit" else 0, latent_lr_mult=0.02)
with Meter() as train_meter:
    model = train(cfg, tc, src, evals)["model"].eval()
sizes = size_report(model)
print(f"{args.linear}: eval loss {evaluate(model, evals):.3f} (floor ~0.89), block bpw {sizes['block_bpw']:.2f}, "
      f"stored {sizes['deployed_total_bytes'] / 1024:.1f} KiB")

before = model_bytes(model)
packed = pack_model(model) if args.linear == "littlebit" else pack_bitnet(model) if args.linear == "bitnet" else model
x = evals[0][:, :-1]
with torch.no_grad(), Meter() as infer_meter:
    for _ in range(10):
        packed(x)
with torch.no_grad():
    print(f"packed: {before / 1024:.0f} KiB -> {model_bytes(packed) / 1024:.0f} KiB in RAM, "
          f"eval loss {evaluate(packed, evals):.3f}, {infer_meter.wall_s / 10 * 1e3:.1f} ms per 32x63 batch")
print("METRICS " + json.dumps({"train_steps": args.steps, **train_meter.as_dict("train_"),
                               "infer_ms_per_batch": infer_meter.wall_s / 10 * 1e3,
                               "infer_cpu_ms_per_batch": infer_meter.cpu_s / 10 * 1e3,
                               "infer_peak_extra_mib": infer_meter.peak_extra_mib,
                               "ram_weights_bytes": model_bytes(packed), "checkpoint_bytes": state_dict_bytes(packed),
                               "stored_bytes": sizes["deployed_total_bytes"]}))
