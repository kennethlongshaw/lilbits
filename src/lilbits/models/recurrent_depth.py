"""Recurrent-depth ("looped") language model: think by iterating a shared block in latent space, then answer.

Follows Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"
(2025, the Huginn-3.5B model), with the exit gate of Ouro / LoopLM (Zhu et al., "Scaling Latent Reasoning
via Looped Language Models", 2025) as an option:

    tokens -> embed -> PRELUDE (unlooped) -> e
    s_0 ~ TruncNormal(0, 2/5)                               random latent state
    s_i = CORE(adapter([s_{i-1}; e]))   for i = 1..r       same weights every iteration; e re-injected each time
    logits = head(CODA(s_r))

Instead of emitting chain-of-thought tokens, the model spends compute on r latent iterations per token. Once
a position's prediction stops changing, its state is frozen and the coda reads the answer from it: the
"thinking" never becomes tokens, so it costs no decoding steps and no KV-cache growth from thought tokens.

Training (Huginn): r is sampled per batch from a log-normal Poisson with mean `mean_recurrence`, and
gradients flow only through the last `backprop_depth` iterations (truncated BPTT), so cost is bounded.
Optional Ouro-style training adds a per-step exit gate lambda_t = sigmoid(w . s_t) and minimizes the
expected loss over exit steps minus beta * entropy of the exit distribution.

Inference: `forward(idx, num_steps=..)` runs a fixed depth. `adaptive_forward(idx, ..)` freezes each position
once KL(p_i || p_{i-1}) < threshold (Huginn's zero-shot exit) or once the gate's exit CDF >= q (Ouro).

Any block linear can be FP, LittleBit or BitNet via `linear`, so the looped core can be sub-1-bit: its weights
are stored once and reused every iteration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.ffn import SwiGLU
from lilbits.models.tinylm import Config, make_linear_factory, make_mixer


@dataclass
class RecurrentDepthConfig:
    vocab_size: int = 32
    seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_prelude: int = 1
    n_core: int = 2
    n_coda: int = 1
    mean_recurrence: int = 8  # training: mean of the sampled iteration count (Huginn uses 32)
    max_recurrence: int = 32
    backprop_depth: int = 4  # truncated BPTT through the last k iterations (Huginn uses 8)
    state_std: float = math.sqrt(2 / 5)
    linear: str = "fp"  # fp | littlebit | bitnet, for prelude / core / coda linears
    core_linear: str | None = None  # override for the looped core only (e.g. "littlebit")
    bpw: float = 1.0
    mixer: str = "softmax"
    exit_gate: bool = False  # Ouro-style learned exit gate + entropy-regularized objective
    gate_beta: float = 0.05


class SandwichBlock(nn.Module):
    """Pre- and post-norm around each sublayer (Huginn's "sandwich" layout keeps deep recurrence stable)."""

    def __init__(self, d: int, heads: int, linear, mixer: str = "softmax"):
        super().__init__()
        self.n1, self.n2, self.n3, self.n4 = (nn.RMSNorm(d) for _ in range(4))
        self.attn = make_mixer(d, heads, linear, mixer)
        self.mlp = SwiGLU(d, int(d * 8 / 3) // 8 * 8, linear)

    def forward(self, x):
        x = self.n2(x + self.attn(self.n1(x)))
        return self.n4(x + self.mlp(self.n3(x)))


def sample_recurrence(mean: int, max_r: int, sigma: float = 0.5, generator=None) -> int:
    """Huginn's heavy-tailed depth sampler: tau ~ N(log(mean) - sigma^2/2, sigma), r = Poisson(e^tau) + 1."""
    tau = torch.normal(math.log(mean) - sigma**2 / 2, sigma, (1,), generator=generator)
    return int(min(max_r, torch.poisson(tau.exp(), generator=generator).item() + 1))


class RecurrentDepthLM(nn.Module):
    def __init__(self, cfg: RecurrentDepthConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        mk = lambda kind: make_linear_factory(Config(linear=kind, bpw=cfg.bpw, init="svd", itq_iters=20))
        lin, core_lin = mk(cfg.linear), mk(cfg.core_linear or cfg.linear)
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.seq_len, d)
        nn.init.normal_(self.tok.weight, std=d**-0.5)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.prelude = nn.ModuleList(SandwichBlock(d, cfg.n_heads, lin, cfg.mixer) for _ in range(cfg.n_prelude))
        self.adapter = nn.Linear(2 * d, d, bias=False)  # [s; e] -> core input (concat beats add at scale)
        self.core = nn.ModuleList(SandwichBlock(d, cfg.n_heads, core_lin, cfg.mixer) for _ in range(cfg.n_core))
        self.core_norm = nn.RMSNorm(d)
        self.coda = nn.ModuleList(SandwichBlock(d, cfg.n_heads, lin, cfg.mixer) for _ in range(cfg.n_coda))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.gate = nn.Linear(d, 1) if cfg.exit_gate else None

    # -- pieces ---------------------------------------------------------------------------------
    def embed(self, idx):
        x = self.tok(idx) * self.cfg.d_model**0.5 + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.prelude:
            x = b(x)
        return x

    def init_state(self, e, generator=None):
        s = torch.empty_like(e)
        nn.init.trunc_normal_(
            s, std=self.cfg.state_std, a=-3 * self.cfg.state_std, b=3 * self.cfg.state_std, generator=generator
        )
        return s

    def step(self, s, e):
        x = self.adapter(torch.cat([s, e], -1))
        for b in self.core:
            x = b(x)
        return self.core_norm(x)

    def readout(self, s):
        for b in self.coda:
            s = b(s)
        return self.head(self.norm(s))

    # -- fixed depth ------------------------------------------------------------------------------
    def forward(self, idx, num_steps: int | None = None, backprop_depth: int | None = None):
        """Logits after `num_steps` latent iterations (default mean_recurrence). With backprop_depth=k, only
        the last k iterations carry gradients (truncated BPTT, as in training)."""
        r = num_steps or self.cfg.mean_recurrence
        k = r if backprop_depth is None else min(backprop_depth, r)
        e = self.embed(idx)
        s = self.init_state(e)
        with torch.no_grad():
            for _ in range(r - k):
                s = self.step(s, e)
        for _ in range(k):
            s = self.step(s, e)
        return self.readout(s)

    def training_loss(self, idx, generator=None) -> tuple[torch.Tensor, dict]:
        """Next-token loss with a sampled depth (Huginn), or the Ouro expected-loss objective with an exit gate."""
        cfg = self.cfg
        r = sample_recurrence(cfg.mean_recurrence, cfg.max_recurrence, generator=generator)
        inp, tgt = idx[:, :-1], idx[:, 1:]
        if not cfg.exit_gate:
            logits = self(inp, num_steps=r, backprop_depth=cfg.backprop_depth)
            loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten())
            return loss, {"r": r}
        # Ouro: L = sum_t p(t) L_t - beta H(p), p(t) = lambda_t prod_{j<t}(1 - lambda_j), last step takes the rest
        e = self.embed(inp)
        s = self.init_state(e)
        k = min(cfg.backprop_depth, r)
        with torch.no_grad():
            for _ in range(r - k):
                s = self.step(s, e)
        losses, lams = [], []
        for _ in range(k):
            s = self.step(s, e)
            logits = self.readout(s)
            losses.append(F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), reduction="none").view(tgt.shape))
            assert self.gate is not None
            lams.append(torch.sigmoid(self.gate(s)).squeeze(-1))
        lam = torch.stack(lams, -1)  # (B, T, k)
        surv = torch.cumprod(torch.cat([torch.ones_like(lam[..., :1]), 1 - lam[..., :-1]], -1), -1)
        p = torch.cat([(lam * surv)[..., :-1], surv[..., -1:]], -1)
        Lt = torch.stack(losses, -1)
        entropy = -(p * p.clamp_min(1e-9).log()).sum(-1)
        loss = ((p * Lt).sum(-1) - cfg.gate_beta * entropy).mean()
        return loss, {"r": r, "final_step_loss": Lt[..., -1].mean().item()}

    # -- adaptive depth ---------------------------------------------------------------------------
    @torch.no_grad()
    def adaptive_forward(
        self,
        idx,
        max_steps: int | None = None,
        kl_threshold: float = 5e-4,
        gate_q: float | None = None,
        min_steps: int = 1,
    ):
        """Iterate until each position has converged, then freeze its state and read the answer from it.

        Exit rule: KL(p_i || p_{i-1}) < kl_threshold (Huginn), or, with an exit gate and gate_q set, exit CDF >= gate_q
        (Ouro). Frozen positions keep their state while unconverged ones continue; later positions still attend to
        the frozen states, as they would to a fixed KV cache. Returns (logits, steps used per position).
        """
        max_steps = max_steps or self.cfg.max_recurrence
        e = self.embed(idx)
        s = self.init_state(e)
        done = torch.zeros(idx.shape, dtype=torch.bool, device=idx.device)
        steps = torch.zeros(idx.shape, dtype=torch.long, device=idx.device)
        logp_prev, cdf = None, torch.zeros(idx.shape, device=idx.device)
        for i in range(max_steps):
            s_new = self.step(s, e)
            s = torch.where(done[..., None], s, s_new)  # frozen positions keep their state
            steps += (~done).long()
            logp = F.log_softmax(self.readout(s), -1)
            if i + 1 >= min_steps:
                if gate_q is not None and self.gate is not None:
                    lam = torch.sigmoid(self.gate(s)).squeeze(-1)
                    cdf = 1 - (1 - cdf) * (1 - lam)
                    done |= cdf >= gate_q
                elif logp_prev is not None:
                    kl = (logp.exp() * (logp - logp_prev)).sum(-1)
                    done |= kl < kl_threshold
            logp_prev = logp
            if done.all():
                break
        return logp, steps
