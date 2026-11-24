"""Full comparison of LittleBit, BitNet b1.58 and FP16 at matched stored size: training time, peak memory,
training state, learning curve, inference speed / memory in fast (cached) and packed form, stored size.

Each model trains (seed 0, tuned settings from python benchmarks/frontier.py) in its own process, single-threaded, so wall
times and peak RSS are comparable. Eval loss is also reported as the 3-seed mean from results/frontier.

    uv run python benchmarks/compare.py                      # writes results/compare/compare.{json,md}
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import io
import json
import os
import statistics
import subprocess
import sys
import time

import torch

from lilbits.measure import size_report
from lilbits.layers.bitnet import BitLinear, pack_bitnet
from lilbits.data import MarkovSource
from frontier import best_grid, key, load
from frontier import main_specs as frontier_specs
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.models.tinylm import Config
from lilbits.inference.packed import model_bytes, pack_model
from lilbits.training.train import TrainConfig, param_groups, train

TARGET = 1.10  # "steps to reach eval loss <= TARGET" learning-curve stat


def specs(frontier_dir: str) -> dict[str, dict]:
    all_specs = frontier_specs(best_grid(os.path.join(frontier_dir, "bitnet_grid.jsonl")))
    pick = lambda fam, d, bpw=None: next(s for s in all_specs if s["family"] == fam and s["d_model"] == d
                                         and (bpw is None or s.get("bpw") == bpw))
    return {
        "FP16 d128 (full size)": dict(pick("FP16", 48), d_model=128),
        "LittleBit 1.40 bpw": pick("LittleBit", 128, 1.41),
        "BitNet d128": pick("BitNet b1.58", 128),
        "BitNet d112": pick("BitNet b1.58", 112),
        "FP16 d40": pick("FP16", 40),
        "LittleBit 0.79 bpw": pick("LittleBit", 128, 0.8),
        "BitNet d96": pick("BitNet b1.58", 96),
        "LittleBit 0.55 bpw (1 path)": pick("LittleBit (1 path)", 128, 0.55),
        "BitNet d80": pick("BitNet b1.58", 80),
        "FP16 d32": pick("FP16", 32),
    }


def _trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _status(k: str) -> float:
    for line in open("/proc/self/status"):
        if line.startswith(k):
            return int(line.split()[1]) / 1024
    return 0.0


class PeakRSS:
    """Samples RSS from a background thread; .extra_mib = peak RSS during the block minus RSS at entry.

    (Resetting the kernel's peak via /proc/self/clear_refs made long torch runs here die with
    "oneMKL FATAL: cannot load libtorch_cpu.so", so the peak is sampled instead.)
    """

    def __init__(self, interval: float = 0.0005):
        self.interval = interval

    def __enter__(self):
        import threading

        self.base = self.peak = _status("VmRSS")
        self._stop = threading.Event()

        def loop():
            page = os.sysconf("SC_PAGE_SIZE") / 2**20
            with open("/proc/self/statm") as f:
                while not self._stop.is_set():
                    f.seek(0)
                    self.peak = max(self.peak, int(f.read().split()[1]) * page)
                    time.sleep(self.interval)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _status("VmRSS"))
        self.extra_mib = self.peak - self.base


def _cache_bytes(model) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, LittleBitLinear) and m._infer_cache is not None:
            total += sum(t.numel() * t.element_size() for t in m._infer_cache[1:])
        if isinstance(m, BitLinear) and m._infer_w is not None:
            total += m._infer_w.numel() * m._infer_w.element_size()
    return total


def _infer(model, inputs: dict, reps: int) -> dict:
    out = {}
    with torch.no_grad():
        model(next(iter(inputs.values())))  # build caches
        out["weights_in_ram_kib"] = (model_bytes(model) + _cache_bytes(model)) / 1024
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        out["file_kib"] = buf.tell() / 1024
        for name, x in inputs.items():
            _trim()
            with PeakRSS() as peak:
                for _ in range(3):
                    model(x)
            t, c = time.perf_counter(), time.process_time()
            for _ in range(reps):
                model(x)
            ms = (time.perf_counter() - t) / reps * 1e3
            out[f"{name}_cpu_ms"] = (time.process_time() - c) / reps * 1e3
            out[f"{name}_ms"] = ms
            out[f"{name}_tok_per_s"] = x.numel() / ms * 1e3
            out[f"{name}_peak_extra_mib"] = peak.extra_mib
    return out


def run_one(spec: dict, steps: int, reps: int) -> dict:
    torch.set_num_threads(1)
    src = MarkovSource(vocab=32, seed=0)
    eval_batches = [src.sample(32, 64) for _ in range(4)]
    model_kw = {k: spec[k] for k in ("linear", "d_model", "bpw", "residual", "itq_iters", "init") if k in spec}
    cfg = Config(**model_kw)
    tc = TrainConfig(steps=steps, log_every=50, seed=0, verbose=False, accelerator="cpu", **spec["tc"])
    _trim()
    with PeakRSS(interval=0.002) as peak:
        t0, c0 = time.perf_counter(), time.process_time()
        res = train(cfg, tc, src, eval_batches)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    train_peak = peak.extra_mib
    model = res["model"].eval()
    sizes = size_report(model)
    hist = [h for h in res["history"] if "train" in h]
    hit = next((h for h in hist if h["eval"] <= TARGET), None)
    groups = param_groups(model, 1.0, 1.0, 0.0)
    pbytes = sizes["train_param_bytes"]
    muon_bytes = sum(p.numel() * p.element_size() for g in groups if g["name"] in ("hidden", "latent")
                     for p in g["params"])
    opt_bytes = 2 * pbytes if tc.optim == "adamw" else muon_bytes + 2 * (pbytes - muon_bytes)
    out = {
        "spec": spec, "final_eval_seed0": res["final_eval"], "final_train_seed0": hist[-1]["train"],
        "wall_s": wall, "cpu_s": cpu, "ms_per_step": wall / steps * 1e3, "fp_warmup_steps": tc.fp_warmup if cfg.linear == "littlebit" else 0,
        "tokens": steps * tc.batch * cfg.seq_len, "train_peak_extra_mib": train_peak,
        "train_state_kib": (2 * pbytes + opt_bytes) / 1024, "optim": tc.optim,
        "steps_to_target": hit["step"] if hit else None,
        "secs_to_target": hit["step"] / steps * wall if hit else None, **sizes,
    }
    inputs = {"batch_32x64": eval_batches[0][:, :64], "seq_1x64": eval_batches[0][:1, :64],
              "1_token": eval_batches[0][:1, :1]}
    out["infer_fast"] = _infer(model, inputs, reps)
    if cfg.linear == "littlebit":
        out["infer_packed"] = _infer(pack_model(model), inputs, reps)
    elif cfg.linear == "bitnet":
        out["infer_packed"] = _infer(pack_bitnet(model), inputs, reps)
    return out


def seed_stats(frontier_dir: str, spec: dict) -> tuple[float, float] | None:
    rows = [r for r in load(os.path.join(frontier_dir, "runs.jsonl"))
            if key(r["spec"], r["seed"]).replace(f'"seed": {r["seed"]}', "") == key(spec, 0).replace('"seed": 0', "")]
    vals = [r["final_eval"] for r in rows]
    if not vals and spec["linear"] == "fp" and spec["d_model"] == 128:  # tuned FP d128: Optuna's 3-seed confirm
        conf = json.load(open(os.path.join(os.path.dirname(frontier_dir), "optuna", "fp_bpw1.41_steps1500_confirm.json")))
        vals = next(r["seeds"] for r in conf.values() if r["params"] == {k: v for k, v in spec["tc"].items()})
    return (statistics.mean(vals), statistics.stdev(vals)) if len(vals) > 1 else None


def table(rows: list[dict]) -> str:
    names = [r["name"] for r in rows]
    fp = rows[0]
    kib = lambda v: f"{v:,.1f} KiB"
    pk = lambda r, k: r.get("infer_packed", {}).get(k)
    fmt = lambda v, f: "n/a" if v is None else f(v)
    metrics = [
        ("**Quality**", None),
        ("Eval loss, mean ± sd of 3 seeds", lambda r: f"{r['seeds'][0]:.3f} ± {r['seeds'][1]:.3f}" if r["seeds"] else "n/a"),
        ("Eval loss, seed 0 (this run)", lambda r: f"{r['final_eval_seed0']:.3f}"),
        ("Final train loss, seed 0", lambda r: f"{r['final_train_seed0']:.3f}"),
        ("**Size**", None),
        ("d_model / block bpw", lambda r: f"{r['spec']['d_model']} / {r['block_bpw']:.2f}"),
        ("Trainable params", lambda r: f"{r['trainable_params']:,}"),
        ("Stored block linears", lambda r: kib(r["deployed_block_bytes"] / 1024)),
        ("Stored whole model", lambda r: kib(r["deployed_total_bytes"] / 1024)),
        ("vs FP16 d128 (whole model)", lambda r: f"{fp['deployed_total_bytes'] / r['deployed_total_bytes']:.1f}x smaller"),
        ("**Training** (1,500 steps, 3.07M tokens, 1 CPU thread)", None),
        ("Optimizer", lambda r: r["optim"] + (f", {r['fp_warmup_steps']} FP warm-up steps" if r["fp_warmup_steps"] else "")),
        ("Wall time / CPU time", lambda r: f"{r['wall_s']:.0f} s / " + (f"{r['cpu_s']:.0f} s" if "cpu_s" in r else "≈ wall (1 thread)")),
        ("ms per step", lambda r: f"{r['ms_per_step']:.0f}"),
        (f"Steps / seconds to eval loss ≤ {TARGET}", lambda r: "never" if r["steps_to_target"] is None
         else f"{r['steps_to_target']} / {r['secs_to_target']:.0f} s"),
        ("Peak RSS growth while training (sampled)", lambda r: f"{r['train_peak_extra_mib']:.0f} MiB"),
        ("Weights + grads + optimizer state", lambda r: kib(r["train_state_kib"])),
        ("**Inference, fast path** (FP32 weights / cached factors)", None),
        ("Weights in RAM", lambda r: kib(r["infer_fast"]["weights_in_ram_kib"])),
        ("state_dict file (training format)", lambda r: kib(r["infer_fast"]["file_kib"])),
        ("Batch 32×64: ms (tokens/s)", lambda r: f"{r['infer_fast']['batch_32x64_ms']:.1f} ({r['infer_fast']['batch_32x64_tok_per_s']:,.0f})"),
        ("1×64 sequence: ms", lambda r: f"{r['infer_fast']['seq_1x64_ms']:.2f}"),
        ("1 token: ms", lambda r: f"{r['infer_fast']['1_token_ms']:.2f}"),
        ("Peak RSS growth, batch 32×64", lambda r: f"{r['infer_fast']['batch_32x64_peak_extra_mib']:.1f} MiB"),
        ("**Inference, packed** (LittleBit: 1-bit signs; BitNet: 2-bit codes)", None),
        ("Weights in RAM", lambda r: fmt(pk(r, "weights_in_ram_kib"), kib)),
        ("state_dict file", lambda r: fmt(pk(r, "file_kib"), kib)),
        ("Batch 32×64: ms", lambda r: fmt(pk(r, "batch_32x64_ms"), lambda v: f"{v:.1f}")),
        ("1×64 sequence: ms", lambda r: fmt(pk(r, "seq_1x64_ms"), lambda v: f"{v:.2f}")),
        ("1 token: ms", lambda r: fmt(pk(r, "1_token_ms"), lambda v: f"{v:.2f}")),
        ("Peak RSS growth, batch 32×64", lambda r: fmt(pk(r, "batch_32x64_peak_extra_mib"), lambda v: f"{v:.1f} MiB")),
    ]
    lines = ["| | " + " | ".join(names) + " |", "|---" * (len(rows) + 1) + "|"]
    for label, f in metrics:
        lines.append(f"| {label} | " + " | ".join("" if f is None else f(r) for r in rows) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--frontier-dir", default="results/frontier")
    ap.add_argument("--dir", default="results/compare")
    ap.add_argument("--only", help="comma list of names to (re)run; others are loaded from compare.json")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--save", action="store_true", help="with --one: merge the result into compare.json")
    args = ap.parse_args(argv)
    sp = specs(args.frontier_dir)
    os.makedirs(args.dir, exist_ok=True)
    path = os.path.join(args.dir, "compare.json")
    done = {r["name"]: r for r in json.load(open(path))} if os.path.exists(path) else {}
    if args.one:
        r = run_one(sp[args.one], args.steps, args.reps)
        print(json.dumps(r))
        if args.save:
            done[args.one] = {"name": args.one, **r}
            json.dump(list(done.values()), open(path, "w"), indent=2)
        return
    todo = [] if args.table_only else (args.only.split(",") if args.only else [n for n in sp if n not in done])
    for name in todo:
        for attempt in range(8):  # torch subprocesses here intermittently die with "oneMKL FATAL" at load
            p = subprocess.run([sys.executable, __file__, "--one", name, "--steps",
                                str(args.steps), "--reps", str(args.reps), "--frontier-dir", args.frontier_dir],
                               stdout=subprocess.PIPE, text=True)
            if p.returncode == 0:
                break
            time.sleep(10 * (attempt + 1))
        else:
            print(f"{name}: failed after retries", flush=True)
            continue
        r = json.loads(p.stdout.strip().splitlines()[-1])
        done[name] = {"name": name, **r}
        json.dump(list(done.values()), open(path, "w"), indent=2)
        print(f"{name}: eval {r['final_eval_seed0']:.4f}  {r['wall_s']:.0f}s", flush=True)
    rows = [dict(done[n], seeds=seed_stats(args.frontier_dir, sp[n])) for n in sp if n in done]
    md = table(rows)
    open(os.path.join(args.dir, "compare.md"), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
