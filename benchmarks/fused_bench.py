"""Fused C runtime (lilbits.inference.fused) vs the PyTorch path on Example 14's FP baselines. Results: results/fused/.

  uv run python benchmarks/fused_bench.py train runs/d200h4.pt --variant 200,8,4   # d,layers,heads; omit for the
                                                                                    # exact fp16_size baseline run
  uv run python benchmarks/fused_bench.py export runs/d200h4.pt w8a8 runs/d200h4_w8a8.npz
  uv run python benchmarks/fused_bench.py suite "torch 4t|torch|runs/d200h4.pt|4" "fused|fused|runs/d200h4_w8a8.npz|1"

Each suite entry runs 3x in fresh processes (median reported): bpb on the 8-batch held-out set Example 14 uses and on
64 batches (64 KB), prefill of a 512-byte prompt, 256 greedy decode bytes after it, CPU time per byte, RSS. The
fused runs import numpy + ctypes only, so their process RSS has no torch in it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

DATA = Path(os.environ.get("LILBITS_CACHE", Path.home() / ".cache" / "lilbits")) / "fused_bench_data.npz"
EX14 = Path(__file__).resolve().parents[1] / "examples" / "14_lean_lm.py"


def rss_mib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


class Peak:
    def __enter__(self):
        self.base = self.peak = rss_mib()
        self._stop = False

        def loop():
            while not self._stop:
                self.peak = max(self.peak, rss_mib())
                time.sleep(0.0005)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join()
        self.growth = max(self.peak, rss_mib()) - self.base


def timed(fn):
    w, c = time.perf_counter(), time.process_time()
    r = fn()
    return r, time.perf_counter() - w, time.process_time() - c


def _ex14():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ex14", EX14)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    return ex


def data():
    """Held-out batches (64 x 8 x 129 bytes; the first 8 are Example 14's eval set) and the infer prompt."""
    import numpy as np

    if not DATA.exists():
        import torch

        from lilbits.shakespeare import ShakespeareBytes

        ex = _ex14()
        ap = argparse.ArgumentParser()
        ex.add_args(ap)
        ev = torch.stack(ex.held_out(ap.parse_args(["--batch", "8", "--seq", "128", "--eval-batches", "64"])))
        prompt = ShakespeareBytes("test", seed=7).sample(1, 512)[0, :512]
        DATA.parent.mkdir(parents=True, exist_ok=True)
        np.savez(DATA, eval=ev.numpy().astype(np.int32), prompt=prompt.numpy().astype(np.int32))
    z = np.load(DATA)
    return z["eval"], z["prompt"]


def load_lm(path):
    import torch
    from torch import nn

    from lilbits.models.lean_lm import LeanLM

    ck = torch.load(path, weights_only=False)
    c = ck["cfg"]
    m = LeanLM(c["d"], c["heads"], c["layers"], lambda i, o: nn.Linear(i, o, bias=False), n_draft=0)
    m.load_state_dict(ck["state"])
    return m.eval()


def cmd_train(a):
    """Example 14's FP baseline recipe: 1000 steps x 8 x 128 B, AdamW 3e-3 cosine. No --variant: the exact fp16_size
    model and RNG stream of `14_lean_lm.py --d 1024 --layers 8 --head-dim 128 --steps 1000 --batch 8 --seq 128`."""
    import torch
    from lightning.fabric import Fabric
    from torch import nn

    from lilbits.models.lean_lm import LeanLM
    from lilbits.runner import fit
    from lilbits.shakespeare import ShakespeareBytes

    ex = _ex14()
    ap = argparse.ArgumentParser()
    ex.add_args(ap)
    args = ap.parse_args(
        [
            "--d",
            "1024",
            "--layers",
            "8",
            "--head-dim",
            "128",
            "--min-heads",
            "1",
            "--steps",
            "1000",
            "--batch",
            "8",
            "--seq",
            "128",
            "--fp-frac",
            "0",
            "--baselines",
            "size",
        ]
    )
    torch.set_num_threads(a.threads)
    fab = Fabric(accelerator="cpu", devices=1, precision="32-true")
    fab.launch()
    fab.seed_everything(a.seed, verbose=False)
    if a.variant:
        d, L, H = map(int, a.variant.split(","))
        m = LeanLM(d, H, L, lambda i, o: nn.Linear(i, o, bias=False), n_draft=0)
    else:
        with fab.init_module():
            m = ex.build(args, {})["fp16_size"]  # lean is built first: same RNG stream as the example
    m.train()
    src = ShakespeareBytes("train", seed=a.seed)
    t = time.perf_counter()
    info = fit(
        fab,
        m,
        lambda _: src.sample(args.batch, args.seq),
        lambda mod, x: mod.loss(x, 0.0),
        args.steps,
        lr=args.lr,
        schedule="cosine",
        optimizer=ex.adamw(m, args),
    )
    m.eval()
    cfg = {"d": m.embed.embedding_dim, "layers": len(m.blocks), "heads": m.blocks[0].mixer.n_heads}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "state": m.state_dict()}, a.out)
    print(
        json.dumps({**cfg, "bpb": ex.bpb(m, ex.held_out(args)), "ce": info["ce"], "train_s": time.perf_counter() - t})
    )


def cmd_export(a):
    import numpy as np

    from lilbits.inference.fused import export

    np.savez(a.out, **export(load_lm(a.ckpt), a.fmt))


def cmd_bench_torch(a):
    import torch

    torch.set_num_threads(a.threads)
    ev, prompt = data()
    ev, prompt = torch.from_numpy(ev).long(), torch.from_numpy(prompt).long()[None]
    with Peak() as pl:
        m = load_lm(a.path)
    res: dict = {"load_growth_mib": pl.growth, "weight_bytes": sum(p.numel() * 4 for p in m.parameters())}
    with torch.no_grad():
        ce = [m.loss(x, 0)[0].item() for x in ev]
        res["bpb8"], res["bpb64"] = sum(ce[:8]) / 8 / math.log(2), sum(ce) / len(ce) / math.log(2)
        m.generate(prompt[:, :16], 4)
        with Peak() as pi:
            (lg, st), w, cpu = timed(lambda: m(prompt))
            res["prefill_B_per_s"], res["prefill_cpu_ms_per_B"] = 512 / w, cpu / 512 * 1e3
            state = {"tok": lg[:, -1:].argmax(-1), "st": st, "out": []}

            def dec():
                for _ in range(256):
                    state["out"].append(state["tok"].item())
                    lg2, state["st"] = m(state["tok"], state["st"])
                    state["tok"] = lg2[:, -1:].argmax(-1)

            _, w, cpu = timed(dec)
    res.update(
        decode_ms_per_B=w / 256 * 1e3,
        decode_cpu_ms_per_B=cpu / 256 * 1e3,
        infer_growth_mib=pi.growth,
        process_rss_mib=rss_mib(),
        tokens=state["out"],
    )
    print("JSON", json.dumps(res))


def cmd_bench_fused(a):
    import numpy as np

    from lilbits.inference.fused import FusedLM

    ev, prompt = data()
    with Peak() as pl:
        m = FusedLM(dict(np.load(a.path)), block=64, threads=a.threads)
    res: dict = {"load_growth_mib": pl.growth, "weight_bytes": m.weight_bytes(), "state_bytes": m.S.nbytes}

    def ce_row(x):
        m.reset()
        lg = m.logits_all(x[:-1]).astype(np.float64)
        mx = lg.max(1)
        return (np.log(np.exp(lg - mx[:, None]).sum(1)) + mx - lg[np.arange(len(x) - 1), x[1:]]).mean()

    ce = [np.mean([ce_row(r) for r in b]) for b in ev]
    res["bpb8"], res["bpb64"] = float(np.mean(ce[:8]) / math.log(2)), float(np.mean(ce) / math.log(2))
    m.reset()
    m.prefill(prompt[:16])
    m.generate(4)
    with Peak() as pi:
        m.reset()
        _, w, cpu = timed(lambda: m.prefill(prompt))
        res["prefill_B_per_s"], res["prefill_cpu_ms_per_B"] = 512 / w, cpu / 512 * 1e3
        out, w, cpu = timed(lambda: m.generate(256))
    res.update(
        decode_ms_per_B=w / 256 * 1e3,
        decode_cpu_ms_per_B=cpu / 256 * 1e3,
        infer_growth_mib=pi.growth,
        process_rss_mib=rss_mib(),
        tokens=out,
    )
    print("JSON", json.dumps(res))


def cmd_suite(a):
    rows = []
    for spec in a.entries:
        label, kind, path, th = spec.split("|")
        runs = []
        for _ in range(a.repeats):
            out = subprocess.run(
                [sys.executable, __file__, f"bench-{kind}", path, th], capture_output=True, text=True, check=True
            ).stdout
            runs.append(json.loads(out.split("JSON ", 1)[1]))
        med = {k: statistics.median(r[k] for r in runs) for k, v in runs[0].items() if isinstance(v, (int, float))}
        med["label"] = label
        med["sample"] = bytes(runs[0]["tokens"][:60]).decode(errors="replace")
        rows.append(med)
        print(json.dumps(med), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("out")
    p.add_argument("--variant", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("export")
    p.add_argument("ckpt")
    p.add_argument("fmt", choices=["f32", "f16", "i8", "i4", "w8a8"])
    p.add_argument("out")
    for kind in ("torch", "fused"):
        p = sub.add_parser(f"bench-{kind}")
        p.add_argument("path")
        p.add_argument("threads", type=int)
    p = sub.add_parser("suite")
    p.add_argument("entries", nargs="+", help="label|torch or fused|path|threads")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="")
    a = ap.parse_args()
    cmds = {"train": cmd_train, "export": cmd_export, "bench-torch": cmd_bench_torch, "bench-fused": cmd_bench_fused}
    cmds.get(a.cmd, cmd_suite)(a)


if __name__ == "__main__":
    main()
