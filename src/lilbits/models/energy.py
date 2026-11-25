"""Energy-based next-byte prediction with "thinking" by energy minimization (after Energy-Based Transformers,
Gladstone et al. 2025, https://arxiv.org/abs/2507.02092), in a lean form.

An EBT scores a (context, candidate prediction) pair with a scalar energy and predicts by gradient descent on the
candidate: start from noise, step y <- y - alpha * dE/dy a few times, read off the final prediction. More steps =
more thinking; drawing several candidates and keeping the lowest-energy one is self-verification. Training runs
the optimization and puts cross-entropy on the final candidate, backpropagating through the steps (second order).

The lean part: the paper re-runs the whole transformer on every step. Here a recurrent backbone (GDN blocks with
explicit state) encodes the context once per position, and only a small energy head runs per thinking step, so
a step costs about one MLP over (d + d_pred), not a full forward. The candidate is logits over the 256 bytes;
the head sees it as the expected byte embedding under softmax(y), so the energy is smooth in y.

    E(h, y) = w . silu(W2 silu(W1 [norm(h), softmax(y) @ C]))      h: context state, C: candidate embeddings

Training tricks from the paper kept: random number of steps, random step size (the paper finds thinking gains
vanish without it), Langevin noise. Not kept: the replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.models.decision import StatefulBlock


@dataclass
class EnergyConfig:
    vocab: int = 256
    d: int = 256
    heads: int = 4
    n_layers: int = 2
    d_pred: int = 64  # candidate embedding width
    d_hidden: int = 512  # energy head width
    steps: int = 4  # thinking steps used at training (randomized in [min_steps, steps])
    min_steps: int = 2
    alpha: float = 10.0  # initial step size (learned, in log space)
    noise: float = 0.05  # Langevin noise std at training
    init_std: float = 1.0  # std of the initial candidate logits


class EnergyHead(nn.Module):
    def __init__(self, cfg: EnergyConfig):
        super().__init__()
        self.cand = nn.Parameter(torch.randn(cfg.vocab, cfg.d_pred) * 0.1)
        self.norm = nn.RMSNorm(cfg.d)
        self.w1 = nn.Linear(cfg.d + cfg.d_pred, cfg.d_hidden)
        self.w2 = nn.Linear(cfg.d_hidden, cfg.d_hidden)
        self.out = nn.Linear(cfg.d_hidden, 1)

    def forward(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """h (..., d), y (..., V) logits -> energy (...)."""
        e = y.softmax(-1) @ self.cand
        z = F.silu(self.w1(torch.cat([self.norm(h), e], -1)))
        return self.out(F.silu(self.w2(z))).squeeze(-1)


class EnergyLM(nn.Module):
    def __init__(self, cfg: EnergyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        nn.init.normal_(self.embed.weight, std=0.02)
        lin = partial(nn.Linear, bias=False)
        self.blocks = nn.ModuleList(StatefulBlock(cfg.d, cfg.heads, lin) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d)
        self.energy = EnergyHead(cfg)
        self.log_alpha = nn.Parameter(torch.tensor(float(cfg.alpha)).log())

    def context(self, x: torch.Tensor, states: list | None = None) -> tuple[torch.Tensor, list]:
        states = states or [None] * len(self.blocks)
        h, new = self.embed(x), []
        for blk, s in zip(self.blocks, states, strict=True):
            h, s = blk(h, s)
            new.append(s)
        return self.norm(h), new

    def think(
        self,
        h: torch.Tensor,
        steps: int,
        y0: torch.Tensor | None = None,
        step_scale: float | torch.Tensor = 1.0,
        noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Minimize E(h, y) over candidate logits y by `steps` gradient steps. Returns (y, energies per step).
        In training mode the steps stay in the graph (second-order gradients); otherwise only y is tracked."""
        train = self.training and torch.is_grad_enabled()
        shape = (*h.shape[:-1], self.cfg.vocab)
        y = y0 if y0 is not None else torch.randn(shape, generator=generator).to(h) * self.cfg.init_std
        alpha = self.log_alpha.exp() * step_scale
        energies = []
        with torch.enable_grad():
            if not train:
                h = h.detach()
            if not y.requires_grad:
                y = y.detach().requires_grad_()
            for _ in range(steps):
                E = self.energy(h, y)
                energies.append(E.detach())
                (g,) = torch.autograd.grad(E.sum(), y, create_graph=train)
                y = y - alpha * g
                if noise:
                    y = y + noise * torch.randn(y.shape, generator=generator).to(y)
                if not train:
                    y = y.detach().requires_grad_()
            energies.append(self.energy(h, y).detach())
        return (y if train else y.detach()), energies

    def training_loss(self, x: torch.Tensor, mask: torch.Tensor | None = None, generator=None):
        """CE on the candidate after a random number of thinking steps with a random step size."""
        cfg = self.cfg
        h, _ = self.context(x[:, :-1])
        steps = int(torch.randint(cfg.min_steps, cfg.steps + 1, (), generator=generator))
        scale = 0.5 + torch.rand((), generator=generator).item()
        y, energies = self.think(h, steps, step_scale=scale, noise=cfg.noise, generator=generator)
        ce = F.cross_entropy(y.flatten(0, 1), x[:, 1:].flatten(), reduction="none")
        if mask is None:
            loss = ce.mean()
        else:
            m = mask.flatten().to(ce)
            loss = (ce * m).sum() / m.sum().clamp_min(1)
        info = {
            "ce": loss.item(),
            "steps": steps,
            "energy_drop": (energies[0] - energies[-1]).mean().item(),
            "alpha": self.log_alpha.exp().item(),
        }
        return loss, info

    @torch.no_grad()
    def predict(self, h: torch.Tensor, steps: int, samples: int = 1, seed: int = 0) -> torch.Tensor:
        """Logits after thinking. samples > 1: self-verification, keep each position's lowest-energy candidate."""
        g = torch.Generator().manual_seed(seed)
        best_y, best_e = None, None
        for _ in range(samples):
            y, es = self.think(h, steps, generator=g)
            e = es[-1]
            if best_y is None or best_e is None:
                best_y, best_e = y, e
            else:
                pick = e < best_e
                best_y = torch.where(pick[..., None], y, best_y)
                best_e = torch.where(pick, e, best_e)
        assert best_y is not None
        return best_y

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n_new: int, steps: int = 4, samples: int = 1):
        h, states = self.context(prompt)
        out = []
        for i in range(n_new):
            tok = self.predict(h[:, -1:], steps, samples, seed=i).argmax(-1)
            out.append(tok)
            h, states = self.context(tok, states)
        return torch.cat([prompt, *out], 1)


__all__ = ["EnergyConfig", "EnergyHead", "EnergyLM"]
