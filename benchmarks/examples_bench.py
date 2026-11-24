"""Run every example as its own process and report the same dimensions for each: wall time, CPU time
(user + sys, all threads), CPU utilisation, peak RSS, plus whatever the example measures itself
(train / inference wall + CPU + peak RSS growth, RAM weights, checkpoint and stored size) via its
`METRICS {json}` line.

Examples default to short smoke runs, so this checks that the code paths work and gives cost
numbers for them; it is not a quality comparison.

    uv run python benchmarks/examples_bench.py                    # writes results/examples_bench.{json,md}
    uv run python benchmarks/examples_bench.py --only 04
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("01 train+pack, LittleBit", ["examples/01_train_and_pack.py", "--linear", "littlebit"]),
    ("01 train+pack, BitNet", ["examples/01_train_and_pack.py", "--linear", "bitnet"]),
    ("01 train+pack, FP", ["examples/01_train_and_pack.py", "--linear", "fp"]),
    ("02 recurrent mixers", ["examples/02_recurrent_mixers.py"]),
    ("03 state rollback", ["examples/03_state_rollback.py"]),
    ("04 decision, LittleBit GDN", ["examples/04_decision_model.py", "--linear", "littlebit"]),
    ("04 decision, BitNet GDN", ["examples/04_decision_model.py", "--linear", "bitnet"]),
    ("04 decision, FP GDN", ["examples/04_decision_model.py", "--linear", "fp"]),
    ("04 decision, LittleBit Mamba2", ["examples/04_decision_model.py", "--mixer", "mamba2"]),
    ("05 pack + Helion (interpret)", ["examples/05_pack_and_helion.py"]),
]


def run(cmd: list[str], env: dict) -> dict:
    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, *cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    out = p.stdout.read()
    _, status, ru = os.wait4(p.pid, 0)  # per-child rusage: CPU times and peak RSS of exactly this process
    wall = time.perf_counter() - t0
    cpu = ru.ru_utime + ru.ru_stime
    metrics = {}
    for line in out.splitlines():
        if line.startswith("METRICS "):
            metrics = json.loads(line[len("METRICS "):])
    return {"ok": os.waitstatus_to_exitcode(status) == 0, "wall_s": wall, "cpu_s": cpu, "cpu_util": cpu / wall,
            "peak_rss_mib": ru.ru_maxrss / 1024, "metrics": metrics, "tail": out.strip().splitlines()[-3:]}


def table(rows: list[dict]) -> str:
    f = lambda v, fmt: "" if v is None else fmt.format(v)
    kib = lambda v: f(v / 1024 if v is not None else None, "{:,.0f} KiB")
    lines = ["| example | ok | process wall | process CPU (util) | process peak RSS | train wall / CPU | train peak RSS growth "
             "| inference (example's own measure) | RAM weights | checkpoint file | stored size |",
             "|---" * 11 + "|"]
    for r in rows:
        m = r["metrics"]
        if "ms_per_question_batched" in m:
            infer = f"{m['prefill_ms_per_state']:.1f} ms prefill / state, {m['ms_per_question_batched']:.2f} ms / question"
        elif "infer_torch_wall_s" in m:
            infer = (f"eval + generate: torch {m['infer_torch_wall_s']:.2f} s, Helion (interpret) "
                     f"{m['infer_helion_wall_s']:.2f} s wall")
        elif "infer_ms_per_batch" in m:
            infer = f"{m['infer_ms_per_batch']:.1f} ms wall, {m['infer_cpu_ms_per_batch']:.1f} ms CPU per 32×63 batch"
        else:
            infer = ""
        train = (f"{m['train_wall_s']:.1f} s / {m['train_cpu_s']:.1f} s ({m['train_steps']} steps)"
                 if "train_wall_s" in m else "")
        lines.append(
            f"| {r['name']} | {'✅' if r['ok'] else '❌'} | {r['wall_s']:.1f} s | {r['cpu_s']:.1f} s ({r['cpu_util']:.2f}) "
            f"| {r['peak_rss_mib']:.0f} MiB | {train} | {f(m.get('train_peak_extra_mib'), '{:.0f} MiB')} | {infer} "
            f"| {kib(m.get('ram_weights_bytes'))} | {kib(m.get('checkpoint_bytes'))} | {kib(m.get('stored_bytes'))} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run only examples whose name starts with this (e.g. 04)")
    ap.add_argument("--threads", type=int, default=1, help="OMP/MKL threads per example (1 = comparable CPU numbers)")
    ap.add_argument("--out", default="results/examples_bench")
    args = ap.parse_args(argv)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), HELION_INTERPRET="1")
    rows = []
    for name, cmd in RUNS:
        if args.only and not name.startswith(args.only):
            continue
        r = {"name": name, "cmd": cmd, **run(cmd, env)}
        rows.append(r)
        print(f"{name}: {'ok' if r['ok'] else 'FAILED'}  {r['wall_s']:.1f}s wall  {r['cpu_s']:.1f}s CPU  "
              f"{r['peak_rss_mib']:.0f} MiB peak", flush=True)
        if not r["ok"]:
            print("   " + "\n   ".join(r["tail"]))
    md = table(rows)
    print(md)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, args.out + ".json"), "w"), indent=2)
    open(os.path.join(ROOT, args.out + ".md"), "w").write(md + "\n")


if __name__ == "__main__":
    main()
