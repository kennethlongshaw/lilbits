"""Looped (recurrent-depth) LM: think in latent space by iterating a shared core, then answer.

Mirrors Huginn (Geiping et al. 2025, "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth
Approach") with the Ouro / LoopLM exit gate (Zhu et al. 2025) as an option. A reasoning LLM spends test-time
compute by decoding thinking tokens; this model spends it by looping its core block on each position's hidden
state, freezing the state once the prediction stops changing, and reading the answer from it. No thought tokens
are decoded, so there is no extra decoding latency or KV cache for them.

What this script shows (default: a short smoke run on the Markov source, to prove the code paths work):
  1. training with Huginn's sampled depth + truncated backprop (or Ouro's gated objective with --exit-gate)
  2. test-time depth scaling: eval loss at 1, 2, 4, ... iterations of the same weights
  3. adaptive exit: per-position iteration counts when states are frozen at convergence
  4. cost of thinking: per-answer FLOPs and cache of r latent loops vs r decoded thinking tokens
  5. METRICS line: train/infer wall, CPU, peak RSS, RAM weights, checkpoint and stored size

    uv run python examples/06_looped_reasoning_lm.py                       # FP, smoke run
    uv run python examples/06_looped_reasoning_lm.py --core-linear littlebit   # sub-1-bit-capable looped core
    uv run python examples/06_looped_reasoning_lm.py --exit-gate           # Ouro-style learned exit
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from lilbits.data import MarkovSource
from lilbits.measure import Meter, size_report, state_dict_bytes
from lilbits.models.recurrent_depth import RecurrentDepthConfig, RecurrentDepthLM


def block_flops_per_token(d: int, T: int) -> float:
    """Forward FLOPs of one sandwich block for one token attending to T positions (2 * MACs)."""
    hidden = int(d * 8 / 3) // 8 * 8
    return 2 * (4 * d * d + 3 * d * hidden) + 2 * 2 * T * d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=40, help="training steps (default: smoke run)")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--mean-recurrence", type=int, default=6)
    ap.add_argument("--backprop-depth", type=int, default=3)
    ap.add_argument("--linear", default="fp", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--core-linear", default=None, choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--exit-gate", action="store_true", help="Ouro-style learned exit gate")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    cfg = RecurrentDepthConfig(
        vocab_size=32,
        seq_len=64,
        d_model=args.d,
        n_heads=4,
        n_prelude=1,
        n_core=2,
        n_coda=1,
        mean_recurrence=args.mean_recurrence,
        max_recurrence=4 * args.mean_recurrence,
        backprop_depth=args.backprop_depth,
        linear=args.linear,
        core_linear=args.core_linear,
        exit_gate=args.exit_gate,
    )
    model = RecurrentDepthLM(cfg)
    src = MarkovSource(vocab=32, seed=0)
    evals = [src.sample(32, 64) for _ in range(2)]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(0)

    print(
        f"1) training {args.steps} steps: depth r ~ log-normal Poisson (mean {cfg.mean_recurrence}), "
        f"backprop through last {cfg.backprop_depth}" + (", Ouro exit gate" if cfg.exit_gate else "")
    )
    with Meter() as train_meter:
        for step in range(args.steps):
            loss, info = model.training_loss(src.sample(32, 64), generator=g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % max(1, args.steps // 4) == 0 or step == args.steps - 1:
                print(f"   step {step:4d}  r={info['r']:2d}  loss {loss.item():.3f}", flush=True)
    model.eval()

    @torch.no_grad()
    def eval_loss(**kw):
        tot = 0.0
        for b in evals:
            torch.manual_seed(0)
            tot += F.cross_entropy(model(b[:, :-1], **kw).flatten(0, 1), b[:, 1:].flatten()).item()
        return tot / len(evals)

    print("\n2) test-time depth scaling (same weights, more latent iterations)")
    depth_curve = {}
    for r in (1, 2, 4, 8, 16, 32):
        if r > cfg.max_recurrence:
            break
        with Meter() as m:
            depth_curve[r] = eval_loss(num_steps=r)
        print(f"   r={r:2d}  eval loss {depth_curve[r]:.3f}  ({m.wall_s * 1e3 / len(evals):.0f} ms per 32x63 batch)")

    print(
        "\n3) adaptive exit: freeze each position once KL(p_i || p_i-1) < 5e-4"
        + (" (or gate CDF >= 0.5)" if cfg.exit_gate else "")
    )
    with torch.no_grad(), Meter() as infer_meter:
        torch.manual_seed(0)
        logp, steps = model.adaptive_forward(evals[0][:, :-1], kl_threshold=5e-4, gate_q=0.5 if cfg.exit_gate else None)
    ad_loss = F.nll_loss(logp.flatten(0, 1), evals[0][:, 1:].flatten()).item()
    print(
        f"   eval loss {ad_loss:.3f}, iterations per position: mean {steps.float().mean():.1f}, "
        f"min {steps.min().item()}, max {steps.max().item()}  ({infer_meter.wall_s * 1e3:.0f} ms)"
    )

    print("\n4) cost of thinking for one answer token (context T=64, r iterations vs r thinking tokens)")
    d, T = cfg.d_model, 64
    per_block = block_flops_per_token(d, T)
    fixed = (cfg.n_prelude + cfg.n_coda) * per_block
    core = cfg.n_core * per_block + 2 * 2 * d * d  # + adapter
    for r in (4, 16):
        latent = fixed + r * core
        # an unlooped model of the same depth would decode r thinking tokens, each a full forward over a growing cache
        unlooped_depth = cfg.n_prelude + cfg.n_core + cfg.n_coda
        tokens = sum(unlooped_depth * block_flops_per_token(d, T + j) for j in range(r + 1))
        print(
            f"   r={r:2d}: latent loop {latent / 1e6:5.1f} MFLOP, {r} core passes in 1 decode step, 0 extra cache | "
            f"{r} thinking tokens {tokens / 1e6:5.1f} MFLOP, {r + 1} sequential decode steps, "
            f"+{r} positions of KV cache ({r * unlooped_depth * 2 * d * 2 / 1024:.1f} KiB FP16)"
        )
    print(
        "   (analytic forward FLOPs. Both routes run r sequential passes, but the loop runs only the core,\n"
        "    adds no cache and emits nothing; the token route also needs a model trained to produce useful thoughts)"
    )

    sizes = size_report(model)
    print(
        "METRICS "
        + json.dumps(
            {
                "train_steps": args.steps,
                **train_meter.as_dict("train_"),
                **infer_meter.as_dict("infer_adaptive_"),
                "adaptive_mean_steps": steps.float().mean().item(),
                "depth_curve": depth_curve,
                "ram_weights_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                "checkpoint_bytes": state_dict_bytes(model),
                "stored_bytes": sizes["deployed_total_bytes"],
                "params": sizes["trainable_params"],
            }
        )
    )


if __name__ == "__main__":
    main()
