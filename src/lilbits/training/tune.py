"""Optuna sweep over training / optimizer settings (lr, lr multipliers, weight decay, warmup, optimizer,
Muon options, FP warm-start length) for the LittleBit or FP model.

Workers are separate processes sharing an Optuna journal file, so CPU runs use all cores.

    uv sync --extra tune
    uv run lb-tune --linear littlebit --trials 48 --workers 4
    uv run lb-tune --linear fp --trials 24 --workers 4
    uv run lb-tune --linear littlebit --confirm 3     # re-run the top configs on 3 seeds
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from lilbits.data import MarkovSource
from lilbits.models.tinylm import Config
from lilbits.training.train import TrainConfig, train


def suggest(trial: optuna.Trial, linear: str) -> tuple[dict, dict]:
    """Returns (model overrides, TrainConfig kwargs)."""
    tc = dict(
        lr=trial.suggest_float("lr", 5e-4, 2e-2, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.03, 0.1, 0.3]),
        warmup=trial.suggest_int("warmup", 0, 400, step=50),
        optim=trial.suggest_categorical("optim", ["adamw", "muon", "muon_ds"]),
    )
    if tc["optim"] != "adamw":
        tc["muon_lr_mult"] = trial.suggest_float("muon_lr_mult", 0.3, 3.0, log=True)
    if tc["optim"] == "muon":
        tc["muon_ns_steps"] = trial.suggest_categorical("muon_ns_steps", [5, 10])
    model = {}
    if linear == "littlebit":
        tc["latent_lr_mult"] = trial.suggest_float("latent_lr_mult", 0.01, 1.0, log=True)
        tc["scale_lr_mult"] = trial.suggest_float("scale_lr_mult", 0.1, 10.0, log=True)
        tc["fp_warmup"] = trial.suggest_categorical("fp_warmup", [0, 150, 300, 600])
        model = {"itq_iters": 50, "init": "svd" if tc["fp_warmup"] == 0 else "small"}
    return model, tc


def run_trial(params: dict, args, seed: int = 0, trial: optuna.Trial | None = None) -> float:
    torch.set_num_threads(1)
    fixed = optuna.trial.FixedTrial(params) if trial is None else trial
    model_over, tc_kw = suggest(fixed, args.linear)
    src = MarkovSource(vocab=32, seed=seed)
    eval_batches = [src.sample(32, 64) for _ in range(4)]
    cfg = Config(linear=args.linear, bpw=args.bpw, **model_over)
    tc = TrainConfig(steps=args.steps, log_every=100, seed=seed, verbose=False, accelerator="cpu", **tc_kw)

    def on_eval(step: int, loss: float):
        if trial is not None:
            trial.report(loss, step)
            if step >= args.prune_after and trial.should_prune():
                raise optuna.TrialPruned()

    return train(cfg, tc, src, eval_batches, on_eval=on_eval)["final_eval"]


def worker(args):
    storage = JournalStorage(JournalFileBackend(args.journal))
    study = optuna.load_study(study_name=args.study, storage=storage,
                              sampler=optuna.samplers.TPESampler(seed=args.worker_id, multivariate=True,
                                                                 group=True, n_startup_trials=8),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=args.prune_after))
    study.optimize(lambda t: run_trial({}, args, trial=t), n_trials=args.trials_per_worker)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--linear", choices=["fp", "littlebit"], default="littlebit")
    ap.add_argument("--bpw", type=float, default=1.41)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prune-after", type=int, default=800, help="no pruning before this step")
    ap.add_argument("--journal", default="results/optuna/journal.log")
    ap.add_argument("--study", default=None)
    ap.add_argument("--confirm", type=int, default=0, help="re-run the top-3 configs (and the default, trial 0) on this many seeds")
    ap.add_argument("--worker-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--trials-per-worker", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.study = args.study or f"{args.linear}_bpw{args.bpw}_steps{args.steps}"
    os.makedirs(os.path.dirname(args.journal) or ".", exist_ok=True)

    if args.worker_id >= 0:
        return worker(args)

    storage = JournalStorage(JournalFileBackend(args.journal))
    study = optuna.create_study(study_name=args.study, storage=storage, direction="minimize", load_if_exists=True)
    if args.trials and not args.confirm:
        # seed the search with the hand-tuned default so it's always a candidate
        if not study.trials:
            default = {"lr": 3e-3, "weight_decay": 0.0, "warmup": 100, "optim": "adamw"}
            if args.linear == "littlebit":
                default |= {"latent_lr_mult": 0.1, "scale_lr_mult": 1.0, "fp_warmup": 300}
            study.enqueue_trial(default)
        per = -(-args.trials // args.workers)
        base = [a for a in (argv if argv is not None else sys.argv[1:])]
        procs = [subprocess.Popen([sys.executable, "-m", "lilbits.training.tune", *base, "--study", args.study,
                                   "--worker-id", str(w), "--trials-per-worker", str(per)])
                 for w in range(args.workers)]
        for p in procs:
            p.wait()
        study = optuna.load_study(study_name=args.study, storage=storage)

    done = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"\nstudy {args.study}: {len(done)} complete, {pruned} pruned")
    for t in done[:8]:
        print(f"  #{t.number:3d}  {t.value:.4f}  {t.params}")
    if len(done) >= 8:
        try:
            imp = optuna.importance.get_param_importances(study)
            print("  importance:", {k: round(v, 3) for k, v in imp.items()})
        except Exception as e:  # noqa: BLE001
            print("  importance unavailable:", e)

    if args.confirm:
        from concurrent.futures import ProcessPoolExecutor

        top = done[:3] + [t for t in done if t.number == 0 and t not in done[:3]]  # + hand-tuned default
        jobs = [(t.number, t.params, s) for t in top for s in range(args.confirm)]
        with ProcessPoolExecutor(args.workers) as ex:
            vals = list(ex.map(_confirm_job, [(p, args, s) for _, p, s in jobs]))
        res = {}
        for (n, p, s), v in zip(jobs, vals):
            res.setdefault(n, {"params": p, "seeds": []})["seeds"].append(v)
        print(f"\nconfirmation on seeds 0..{args.confirm - 1}:")
        for n, r in res.items():
            m = sum(r["seeds"]) / len(r["seeds"])
            print(f"  #{n:3d}  mean {m:.4f}  seeds {[round(v, 4) for v in r['seeds']]}  {r['params']}")
        with open(os.path.join(os.path.dirname(args.journal), f"{args.study}_confirm.json"), "w") as f:
            json.dump(res, f, indent=2)


def _confirm_job(a):
    params, args, seed = a
    return run_trial(params, args, seed=seed)


if __name__ == "__main__":
    main()
