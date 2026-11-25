"""Post-training losses for byte LMs: sequence log-probs, GRPO (group-relative policy gradient) and DPO.

Models are anything that maps bytes (B, T) to logits (B, T, V) through `logits_fn`. Completions are scored only
on their own bytes (a mask), so prompts of any length can be batched.

    seq_logprob(logits_fn, x, mask)        sum of log p(x_t | x_<t) over masked target positions, per row
    sample(step_fn, prompt, n, ...)        temperature sampling of n bytes from a stateful step function
    grpo_loss(logp, rewards, group, ...)   -(A * logp) with A = (r - mean_group) / std_group, + optional KL
    dpo_loss(pc, pr, rc, rr, beta)         -log sigmoid(beta * ((pc - rc) - (pr - rr)))
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def seq_logprob(logits_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, mask: torch.Tensor):
    """x (B, T+1) bytes, mask (B, T) over targets x[:, 1:] -> (B,) summed log-probs of the masked bytes."""
    logp = logits_fn(x[:, :-1]).log_softmax(-1).gather(-1, x[:, 1:, None]).squeeze(-1)
    return (logp * mask).sum(-1)


@torch.no_grad()
def sample(
    step_fn: Callable[[torch.Tensor, object], tuple[torch.Tensor, object]],
    prompt: torch.Tensor,
    n: int,
    temperature: float = 1.0,
    stop: int | None = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """step_fn(tokens, state) -> (logits (B, L, V), state). Samples n bytes per row after prompt (B, P); rows
    that emitted `stop` keep emitting it. Returns (B, n) sampled bytes."""
    logits, state = step_fn(prompt, None)
    out, done = [], torch.zeros(prompt.shape[0], dtype=torch.bool)
    for _ in range(n):
        p = (logits[:, -1] / max(temperature, 1e-6)).softmax(-1)
        tok = torch.multinomial(p, 1, generator=generator).squeeze(-1) if temperature > 0 else p.argmax(-1)
        if stop is not None:
            tok = torch.where(done, torch.full_like(tok, stop), tok)
            done |= tok == stop
        out.append(tok)
        logits, state = step_fn(tok[:, None], state)
    return torch.stack(out, 1)


def group_advantages(rewards: torch.Tensor, group: int, eps: float = 1e-4) -> torch.Tensor:
    """rewards (B,) laid out as consecutive groups of `group` samples of the same prompt."""
    r = rewards.view(-1, group)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).flatten()


def grpo_loss(
    logp: torch.Tensor,
    rewards: torch.Tensor,
    group: int,
    lengths: torch.Tensor | None = None,
    ref_logp: torch.Tensor | None = None,
    kl: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """On-policy GRPO step (one update per batch of samples, so the PPO ratio is 1 and clipping is inactive).
    logp: (B,) summed log-probs of each sampled completion under the current policy; lengths normalize them."""
    adv = group_advantages(rewards, group)
    norm = lengths.clamp_min(1) if lengths is not None else torch.ones_like(logp)
    loss = -(adv * logp / norm).mean()
    info = {"reward": rewards.mean().item(), "adv_abs": adv.abs().mean().item()}
    if ref_logp is not None and kl:
        k = ((logp - ref_logp) / norm).mean()  # sample estimate of KL(pi || ref) per byte
        loss = loss + kl * k
        info["kl"] = k.item()
    return loss, info


def dpo_loss(
    pol_chosen: torch.Tensor,
    pol_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    margin = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, {"dpo": loss.item(), "pref_acc": (margin > 0).float().mean().item()}


__all__ = ["dpo_loss", "group_advantages", "grpo_loss", "sample", "seq_logprob"]
