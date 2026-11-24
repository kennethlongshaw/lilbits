"""Run every example in three separate processes, train -> test -> infer, and report the same dimensions for each.

Every example uses the shared harness (`lilbits.runner`), so each one has `--mode train|test|infer` and a
checkpoint. Running the phases as separate processes measures each on its own: an inference process never holds
training state. Per phase this reports the process wall time, CPU time (user + sys, all threads), CPU
utilisation and peak RSS (from os.wait4 rusage), the harness's own Meter for the phase (wall, CPU, peak RSS growth),
the example's key metrics, and the deployed size and checkpoint file size.

Examples default to short smoke runs, so this checks that every code path works and gives cost numbers for
them; it is not a quality comparison.

    uv run python benchmarks/examples_bench.py                    # writes results/examples_bench.{json,md}
    uv run python benchmarks/examples_bench.py --only 04
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUT = ["--kernel", "lut"]
RUNS: list[tuple] = [
    ("01 train+pack, LittleBit", ["examples/01_train_and_pack.py", "--linear", "littlebit"]),
    ("01 train+pack, LittleBit, LUT kernel", ["examples/01_train_and_pack.py", "--linear", "littlebit"], LUT),
    ("01 train+pack, BitNet", ["examples/01_train_and_pack.py", "--linear", "bitnet"]),
    ("01 train+pack, BitNet, LUT kernel", ["examples/01_train_and_pack.py", "--linear", "bitnet"], LUT),
    ("01 train+pack, FP", ["examples/01_train_and_pack.py", "--linear", "fp"]),
    ("02 recurrent mixers (stateful LMs)", ["examples/02_recurrent_mixers.py"]),
    (
        "02 recurrent mixers, LittleBit, LUT decode",
        ["examples/02_recurrent_mixers.py", "--linear", "littlebit"],
        ["--kernel", "lut"],
    ),
    ("03 state rollback", ["examples/03_state_rollback.py"]),
    ("04 decision, LittleBit GDN", ["examples/04_decision_model.py", "--linear", "littlebit"]),
    ("04 decision, BitNet GDN", ["examples/04_decision_model.py", "--linear", "bitnet"]),
    ("04 decision, FP GDN", ["examples/04_decision_model.py", "--linear", "fp"]),
    ("04 decision, LittleBit Mamba2", ["examples/04_decision_model.py", "--mixer", "mamba2"]),
    ("05 pack + LUT + Helion (interpret)", ["examples/05_pack_and_helion.py"]),
    ("06 looped LM, FP", ["examples/06_looped_reasoning_lm.py"]),
    ("06 looped LM, LittleBit core", ["examples/06_looped_reasoning_lm.py", "--core-linear", "littlebit"]),
    ("06 looped LM, Ouro exit gate", ["examples/06_looped_reasoning_lm.py", "--exit-gate"]),
    ("07 byte-level: baseline / BLT / H-Net", ["examples/07_byte_level_patching.py"]),
    ("08 LLM-JEPA (LM / online / EMA / EMA+dropout)", ["examples/08_llm_jepa.py"]),
    ("09 looped index LM (5 stages + decode modes)", ["examples/09_looped_index_lm.py"]),
    ("09 looped index LM, LUT decode", ["examples/09_looped_index_lm.py"], ["--kernel", "lut"]),
]
METER_KEYS = ("wall_s", "cpu_s", "cpu_util", "peak_extra_mib")


def run(cmd: list[str], env: dict) -> dict:
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [sys.executable, *cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert p.stdout is not None
    out = p.stdout.read()
    _, status, ru = os.wait4(p.pid, 0)  # per-child rusage: CPU times and peak RSS of exactly this process
    wall = time.perf_counter() - t0
    metrics = {}
    for line in out.splitlines():
        if line.startswith("METRICS "):
            metrics = json.loads(line[len("METRICS ") :])
    cpu = ru.ru_utime + ru.ru_stime
    return {
        "ok": status == 0 and bool(metrics),
        "wall_s": wall,
        "cpu_s": cpu,
        "cpu_util": cpu / wall if wall else 0.0,
        "peak_rss_mib": ru.ru_maxrss / 1024,
        "metrics": metrics,
        "tail": out.strip().splitlines()[-3:],
    }


def _short(v) -> str | None:
    if isinstance(v, bool) or isinstance(v, int) or isinstance(v, str):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3g}"
    return None


def headline(d: dict, n: int = 3) -> str:
    """The first n metrics of a phase that aren't the harness's Meter fields; nested dicts show 3 entries."""
    items = []
    for k, v in d.items():
        if k in METER_KEYS:
            continue
        if isinstance(v, dict):
            inner = [f"{a}: {_short(b)}" for a, b in v.items() if _short(b) is not None][:3]
            if not inner:
                continue
            v = "{" + ", ".join(inner) + "}"
        else:
            v = _short(v)
            if v is None:
                continue
        items.append(f"{k} = {v}")
        if len(items) == n:
            break
    return "; ".join(items)


def table(rows: list[dict]) -> str:
    lines = [
        "| example | phase | ok | process wall / CPU (util) | process peak RSS | phase wall / CPU "
        "| phase peak RSS growth | key metrics | deployed | checkpoint file |",
        "|---" * 10 + "|",
    ]
    for r in rows:
        for phase in ("train", "test", "infer"):
            p = r["phases"].get(phase)
            if p is None:
                continue
            m = p["metrics"]
            ph = m.get(phase, {})
            size = m.get("size", {})
            deployed = sum(v["deployed_bytes"] for v in size.values() if isinstance(v, dict))
            ckpt = size.get("checkpoint_file_bytes")
            lines.append(
                f"| {r['name'] if phase == 'train' else ''} | {phase} | {'✅' if p['ok'] else '❌'} "
                f"| {p['wall_s']:.1f} s / {p['cpu_s']:.1f} s ({p['cpu_util']:.2f}) | {p['peak_rss_mib']:.0f} MiB "
                f"| {ph.get('wall_s', 0):.2f} s / {ph.get('cpu_s', 0):.2f} s | {ph.get('peak_extra_mib', 0):.0f} MiB "
                f"| {headline(ph)} | {deployed / 1024:,.0f} KiB | {'' if ckpt is None else f'{ckpt / 1024:,.0f} KiB'} |"
            )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run only examples whose name starts with this (e.g. 04)")
    ap.add_argument("--threads", type=int, default=1, help="OMP/MKL threads per example (1 = comparable CPU numbers)")
    ap.add_argument("--out", default="results/examples_bench")
    args = ap.parse_args(argv)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), HELION_INTERPRET="1")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, cmd, *extra) in enumerate(RUNS):
            if args.only and not name.startswith(args.only):
                continue
            ckpt = os.path.join(tmp, f"run{i}.pt")
            phases = {}
            for phase in ("train", "test", "infer"):
                flags = ["--mode", phase, "--ckpt", ckpt, "--threads", str(args.threads)]
                runtime = extra[0] if extra and phase != "train" else []
                p = phases[phase] = run([*cmd, *flags, *runtime], env)
                print(
                    f"{name} [{phase}]: {'ok' if p['ok'] else 'FAILED'}  {p['wall_s']:.1f}s wall  "
                    f"{p['cpu_s']:.1f}s CPU  {p['peak_rss_mib']:.0f} MiB peak",
                    flush=True,
                )
                if not p["ok"]:
                    print("   " + "\n   ".join(p["tail"]))
                    break
            rows.append({"name": name, "cmd": cmd, "phases": phases})
    md = table(rows)
    print(md)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, args.out + ".json"), "w"), indent=2)
    open(os.path.join(ROOT, args.out + ".md"), "w").write(md + "\n")


if __name__ == "__main__":
    main()
