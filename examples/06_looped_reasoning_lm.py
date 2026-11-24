"""Looped (recurrent-depth) LM: think in latent space by iterating a shared core, then answer.

Mirrors Huginn (Geiping et al. 2025, "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth
Approach") with the Ouro / LoopLM exit gate (Zhu et al. 2025) as an option. A reasoning LLM spends test-time
compute by decoding thinking tokens; this model spends it by looping its core block on each position's hidden
state, freezing the state once the prediction stops changing, and reading the answer from it. No thought tokens
are decoded, so there is no extra decoding latency or KV cache for them.

Phases (default: a short smoke run on the Markov source, to prove the code paths work):
    train   Huginn's sampled depth + truncated backprop (or Ouro's gated objective with --exit-gate), on Fabric
    test    test-time depth scaling: held-out loss at 1, 2, 4, ... iterations of the same weights
    infer   adaptive exit (per-position iteration counts when states freeze at convergence), and the cost of
            thinking: per-answer FLOPs and cache of r latent loops vs r decoded thinking tokens

    uv run python examples/06_looped_reasoning_lm.py                       # FP, smoke run
    uv run python examples/06_looped_reasoning_lm.py --core-linear littlebit   # sub-1-bit-capable looped core
    uv run python examples/06_looped_reasoning_lm.py --exit-gate           # Ouro-style learned exit
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lilbits.data import MarkovSource
from lilbits.models.recurrent_depth import RecurrentDepthConfig, RecurrentDepthLM
from lilbits.runner import Example, fit, run


def block_flops_per_token(d: int, T: int) -> float:
    """Forward FLOPs of one sandwich block for one token attending to T positions (2 * MACs)."""
    hidden = int(d * 8 / 3) // 8 * 8
    return 2 * (4 * d * d + 3 * d * hidden) + 2 * 2 * T * d


def add_args(ap):
    ap.add_argument("--steps", type=int, default=40, help="training steps (default: smoke run)")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--mean-recurrence", type=int, default=6)
    ap.add_argument("--backprop-depth", type=int, default=3)
    ap.add_argument("--linear", default="fp", choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--core-linear", default=None, choices=["fp", "littlebit", "bitnet"])
    ap.add_argument("--exit-gate", action="store_true", help="Ouro-style learned exit gate")


def cfg(args) -> RecurrentDepthConfig:
    return RecurrentDepthConfig(
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


def held_out() -> list[torch.Tensor]:
    src = MarkovSource(vocab=32, seed=0)
    src._g.manual_seed(999)
    return [src.sample(32, 64) for _ in range(2)]


def build(args, meta):
    return {"lm": RecurrentDepthLM(cfg(args))}


def train_fn(models, args, fab):
    src = MarkovSource(vocab=32, seed=0)
    g = torch.Generator().manual_seed(args.seed)
    rs: list[int] = []

    def loss_fn(model, x):
        loss, info = model.training_loss(x, generator=g)
        rs.append(info["r"])
        return loss, {}

    out = fit(fab, models["lm"], lambda _: src.sample(32, 64), loss_fn, args.steps, lr=1e-3)
    return {**out, "mean_sampled_depth": sum(rs) / len(rs)}, {}


def eval_loss(model, batches, **kw) -> float:
    tot = 0.0
    for b in batches:
        torch.manual_seed(0)
        tot += F.cross_entropy(model(b[:, :-1], **kw).flatten(0, 1), b[:, 1:].flatten()).item()
    return tot / len(batches)


def test_fn(models, args):
    model, c = models["lm"], cfg(args)
    curve = {r: eval_loss(model, held_out(), num_steps=r) for r in (1, 2, 4, 8, 16, 32) if r <= c.max_recurrence}
    return {"held_out_loss_by_depth": curve}


def infer_fn(models, args):
    model, c = models["lm"], cfg(args)
    x = held_out()[0]
    torch.manual_seed(0)
    logp, steps = model.adaptive_forward(x[:, :-1], kl_threshold=5e-4, gate_q=0.5 if c.exit_gate else None)
    out: dict = {
        "adaptive_exit_held_out_loss": F.nll_loss(logp.flatten(0, 1), x[:, 1:].flatten()).item(),
        "adaptive_mean_loops": steps.float().mean().item(),
        "adaptive_min_max_loops": [steps.min().item(), steps.max().item()],
    }
    d, T = c.d_model, 64
    per_block = block_flops_per_token(d, T)
    fixed = (c.n_prelude + c.n_coda) * per_block
    core = c.n_core * per_block + 2 * 2 * d * d  # + adapter
    depth = c.n_prelude + c.n_core + c.n_coda
    for r in (4, 16):  # analytic: r latent loops in one decode step vs r decoded thinking tokens
        out[f"r{r}_latent_loop_mflop"] = (fixed + r * core) / 1e6
        out[f"r{r}_thinking_tokens_mflop"] = sum(depth * block_flops_per_token(d, T + j) for j in range(r + 1)) / 1e6
        out[f"r{r}_thinking_tokens_extra_kv_bytes_fp16"] = r * depth * 2 * d * 2
    return out


EXAMPLE = Example("06_looped_reasoning_lm", __doc__ or "", add_args, build, train_fn, test_fn, infer_fn)

if __name__ == "__main__":
    run(EXAMPLE)
