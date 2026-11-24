"""LittleBit-style low-rank binarized linear layer.

Each weight matrix W (out x in) is represented as

    W_hat = diag(h) @ sign(U) @ diag(l) @ sign(V)^T @ diag(g)

with U (out x r) and V (in x r) latent full-precision factors that are
binarized in the forward pass, and three learned FP scale vectors:
h (row / output scale), g (column / input scale) and l (latent scale).
The dense W is never materialized. The scales are folded into the small factors
(A = g*sign(V)*l, B = h*sign(U)) and the forward is x @ A @ B^T, costing r*(in+out)
MACs per token instead of in*out.

Optionally a second "residual" path with its own factors approximates what
the first path misses (the paper's residual factorization).

The sign function is trained with a straight-through estimator. The default
is SmoothSign (forward: sign, backward: derivative of tanh(k*x)), as used
in LittleBit's QAT; a clipped identity STE is available too.

Unlike the paper, nothing here is initialized from a pretrained weight
(no SVD init, no Joint-ITQ rotation): the latents are random and the scales
are set so the layer starts with roughly unit-variance-preserving outputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

FP_BITS = 16  # bits charged for each full-precision scale entry


def _sign(x: torch.Tensor) -> torch.Tensor:
    # sign with sign(0) = +1 so every entry is a genuine +-1 bit
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


class SmoothSign(torch.autograd.Function):
    """sign(x) forward, d/dx tanh(k x) backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.k = k
        return _sign(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        t = torch.tanh(ctx.k * x)
        return grad_out * ctx.k * (1.0 - t * t), None


class ClippedSTE(torch.autograd.Function):
    """sign(x) forward, identity gradient inside |x| <= 1 backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return _sign(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad_out * (x.abs() <= 1).to(grad_out.dtype)


def binarize(x: torch.Tensor, ste: str = "smooth", k: float = 100.0) -> torch.Tensor:
    if ste == "smooth":
        return SmoothSign.apply(x, k)
    if ste == "clipped":
        return ClippedSTE.apply(x)
    raise ValueError(f"unknown ste {ste!r}")


def bits_per_weight(in_features: int, out_features: int, rank: int, paths: int = 1) -> float:
    """Storage bits / dense weight count, matching the paper's accounting.

    Per path: r*(in+out) sign bits + FP16 scales h (out), g (in), l (r).
    """
    per_path = rank * (in_features + out_features) + FP_BITS * (in_features + out_features + rank)
    return paths * per_path / (in_features * out_features)


def rank_for_bpw(in_features: int, out_features: int, bpw: float, paths: int = 1) -> int:
    """Largest rank whose bits_per_weight does not exceed `bpw` (at least 1)."""
    budget = bpw * in_features * out_features / paths - FP_BITS * (in_features + out_features)
    r = math.floor(budget / (in_features + out_features + FP_BITS))
    return max(1, min(r, min(in_features, out_features)))


def _bimodal(shape, center: float, std: float) -> torch.Tensor:
    """Values at +-center plus noise: far from the sign-flip point at zero."""
    signs = _sign(torch.randn(shape))
    return signs * (center + std * torch.randn(shape)).abs()


@torch.no_grad()
def joint_itq(U: torch.Tensor, V: torch.Tensor, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """LittleBit-2 Joint-ITQ: orthogonal R (r x r) aligning both factors with the hypercube.

    U V^T == (U R)(V R)^T for any orthogonal R, so rotating changes nothing in full
    precision but moves latent entries away from zero, where signs are unstable.
    Minimizes ||sign(Z R) - Z R|| over R with Z = [U; V] (each block normalized).
    """
    r = U.shape[1]
    Z = torch.cat([U / U.norm(), V / V.norm()]) * math.sqrt(r)
    g = torch.Generator().manual_seed(seed)
    R, _ = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=Z.dtype).to(Z.device))
    for _ in range(iters):
        B = _sign(Z @ R)
        A, _, Ct = torch.linalg.svd(Z.t() @ B)
        R = A @ Ct
    return R


def zero_margin(x: torch.Tensor, frac: float = 0.1) -> float:
    """Fraction of latent entries within `frac` * rms of zero (likely to flip)."""
    return (x.abs() < frac * x.pow(2).mean().sqrt()).float().mean().item()


def _rank1_nonneg(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-1 approximation a b^T of a nonnegative matrix, with a, b >= 0."""
    Uu, S, Vh = torch.linalg.svd(M, full_matrices=False)
    a, b = Uu[:, 0] * S[0].sqrt(), Vh[0] * S[0].sqrt()
    if a.sum() < 0:
        a, b = -a, -b
    return a.clamp_min(1e-8), b.clamp_min(1e-8)


class _Path(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, gain: float, latent_std: float,
                 init: str = "small", bimodal_center: float = 0.5):
        super().__init__()
        if init == "bimodal":
            self.U = nn.Parameter(_bimodal((out_features, rank), bimodal_center, latent_std))
            self.V = nn.Parameter(_bimodal((in_features, rank), bimodal_center, latent_std))
        else:
            self.U = nn.Parameter(torch.randn(out_features, rank) * latent_std)
            self.V = nn.Parameter(torch.randn(in_features, rank) * latent_std)
        # x*g @ sign(V): each latent unit is a sum of `in` +-x terms -> var ~ in
        # @ sign(U)^T: sum of `rank` terms -> var ~ rank. Normalize both.
        self.g = nn.Parameter(torch.ones(in_features))
        self.l = nn.Parameter(torch.full((rank,), 1.0 / math.sqrt(in_features)))
        self.h = nn.Parameter(torch.full((out_features,), gain / math.sqrt(rank)))

    @torch.no_grad()
    def init_from_weight(self, W: torch.Tensor, itq_iters: int = 0) -> None:
        """Paper-style (Dual-SVID) init: SVD, optional Joint-ITQ, rank-1 fit of magnitudes."""
        r = self.U.shape[1]
        Uu, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        Up = Uu[:, :r] * S[:r].sqrt()
        Vp = Vh[:r].t() * S[:r].sqrt()
        if itq_iters > 0:
            R = joint_itq(Up, Vp, itq_iters)
            Up, Vp = Up @ R, Vp @ R
        h, lu = _rank1_nonneg(Up.abs())
        g, lv = _rank1_nonneg(Vp.abs())
        self.h.copy_(h)
        self.g.copy_(g)
        self.l.copy_(lu * lv)
        # Keep latent magnitudes (they set flip margins) but bring them inside |x| <= 1.
        self.U.copy_(Up / Up.abs().max())
        self.V.copy_(Vp / Vp.abs().max())

    def forward(self, x: torch.Tensor, ste: str, k: float) -> torch.Tensor:
        Vb = binarize(self.V, ste, k)
        Ub = binarize(self.U, ste, k)
        z = (x * self.g) @ Vb
        z = z * self.l
        return (z @ Ub.t()) * self.h

    def weight(self) -> torch.Tensor:
        Ub, Vb = _sign(self.U), _sign(self.V)
        return self.h[:, None] * ((Ub * self.l) @ Vb.t()) * self.g[None, :]


class LittleBitLinear(nn.Module):
    """Drop-in replacement for nn.Linear using LittleBit factorization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int | None = None,
        bpw: float | None = None,
        bias: bool = False,
        residual: bool = False,
        ste: str = "smooth",
        smooth_k: float = 100.0,
        latent_std: float | None = None,
        gain: float = 1.0,
        init: str = "small",
        itq_iters: int = 0,
    ):
        super().__init__()
        paths = 2 if residual else 1
        if rank is None:
            if bpw is None:
                raise ValueError("give rank or bpw")
            rank = rank_for_bpw(in_features, out_features, bpw, paths)
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.ste, self.smooth_k = ste, smooth_k
        # Keep latents inside the STE's active region at init.
        if init not in ("small", "bimodal", "svd"):
            raise ValueError(f"unknown init {init!r}")
        if latent_std is None:
            latent_std = 1.0 / smooth_k if ste == "smooth" else 0.1
        path_gain = gain / math.sqrt(paths)
        self.paths = nn.ModuleList(
            _Path(in_features, out_features, rank, path_gain, latent_std,
                  init="bimodal" if init == "bimodal" else "small")
            for _ in range(paths)
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        if init == "svd":
            # No teacher: factorize what nn.Linear would have started from.
            W = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            self.init_from_weight(W, itq_iters, match_norm=True)

    @torch.no_grad()
    def init_from_weight(self, W: torch.Tensor, itq_iters: int = 0, match_norm: bool = False) -> None:
        """Initialize all paths from a dense weight; each residual path fits what's left."""
        target = W.float()
        for p in self.paths:
            p.init_from_weight(target, itq_iters)
            target = target - p.weight()
        if match_norm:
            # A rank-r slice of a random matrix keeps only ~r/n of its energy; restore it.
            ratio = W.norm() / self.weight().norm().clamp_min(1e-8)
            for p in self.paths:
                p.h.mul_(ratio)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale-folded factors A (in x R) and B (out x R), R = rank * paths.

        A = [diag(g_i) sign(V_i) diag(l_i)]_i and B = [diag(h_i) sign(U_i)]_i concatenated
        along the rank axis, so sum_i W_i = B A^T. The scales touch only weight-sized
        tensors, never activations, and all paths run as one pair of matmuls.
        """
        A = [p.g[:, None] * binarize(p.V, self.ste, self.smooth_k) * p.l[None, :] for p in self.paths]
        B = [p.h[:, None] * binarize(p.U, self.ste, self.smooth_k) for p in self.paths]
        if len(self.paths) == 1:
            return A[0], B[0]
        return torch.cat(A, dim=1), torch.cat(B, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A, B = self.factors()
        return F.linear(F.linear(x, A.t()), B, self.bias)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Unfused per-path forward (scales applied to activations); for tests."""
        y = self.paths[0](x, self.ste, self.smooth_k)
        for p in self.paths[1:]:
            y = y + p(x, self.ste, self.smooth_k)
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def weight(self) -> torch.Tensor:
        """Materialize the dense effective weight (for inspection/tests only)."""
        return sum(p.weight() for p in self.paths)

    @property
    def bpw(self) -> float:
        return bits_per_weight(self.in_features, self.out_features, self.rank, len(self.paths))

    def storage_bits(self) -> int:
        return round(self.bpw * self.in_features * self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"paths={len(self.paths)}, bpw={self.bpw:.3f}, ste={self.ste}"
        )
