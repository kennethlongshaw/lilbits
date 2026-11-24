"""Linear-attention token mixers with pluggable (FP or LittleBit) projections.

Only the d_model x d_model projections go through the `linear` factory, so they are the
ones LittleBit compresses. The small, numerically sensitive pieces -- per-head decay and
write-strength gates, short convolutions, A_log / dt_bias -- stay full precision unless
`lb_gates=True`. That mirrors how the attention softmax itself is never quantized.

GatedDeltaNet follows Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule"
(ICLR 2025), the mixer used in the Qwen3-Next / Qwen3.5 / Qwen3.8-Flash-Next hybrids and the
`fla` library's GatedDeltaNet layer:

    S_t = exp(g_t) * S_{t-1}                       (gated decay, g_t <= 0)
    S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T   (delta-rule write, ||k_t|| = 1)
    o_t = S_t^T q_t

Two equivalent implementations: `gated_delta_rule`, the naive per-token loop (exact, easy to read, used for
single-token decode), and `chunk_gated_delta_rule`, the chunkwise-parallel form (WY representation, as in
`fla`'s chunk kernels) that does C tokens per step with matmuls and is what `GatedDeltaNet` uses for training and
prefill. `fla.ops.gated_delta_rule.chunk_gated_delta_rule` is the fused Triton version of the same algorithm.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

LinearFactory = Callable[[int, int], nn.Module]


class ShortConv(nn.Module):
    """Causal depthwise conv + SiLU over the time axis, (B, T, C) -> (B, T, C)."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x.transpose(1, 2), (self.kernel - 1, 0)))
        return F.silu(y).transpose(1, 2)


def gated_delta_rule(q, k, v, g, beta, initial_state=None):
    """Naive recurrent gated delta rule.

    q, k: (B, T, H, Dk), unit-norm k; v: (B, T, H, Dv); g (log decay, <= 0), beta: (B, T, H).
    Returns o (B, T, H, Dv) and the final state (B, H, Dk, Dv).
    """
    B, T, H, Dk = k.shape
    S = q.new_zeros(B, H, Dk, v.shape[-1]) if initial_state is None else initial_state
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        delta = (v[:, t] - torch.einsum("bhkv,bhk->bhv", S, kt)) * beta[:, t, :, None]
        S = S + torch.einsum("bhk,bhv->bhkv", kt, delta)
        out.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(out, dim=1), S


def chunk_gated_delta_rule(q, k, v, g, beta, initial_state=None, chunk: int = 64):
    """Chunkwise-parallel gated delta rule: same inputs, outputs and final state as `gated_delta_rule`.

    Within a chunk of C tokens, the delta-rule writes form a unit lower-triangular system: with
    G_ij = exp(gcum_i - gcum_j) and M = strictly_lower((beta k) k^T * G), the chunk's effective writes are
    U = (I + M)^-1 (beta v) and W = (I + M)^-1 (beta k exp(gcum)) (the WY representation). Then
        new values      V' = U - W S
        outputs         o  = (q exp(gcum)) S + lower((q k^T) * G) V'
        next state      S  = exp(gcum_C) S + (k exp(gcum_C - gcum))^T V'
    so each chunk is a handful of (C x C) and (C x D) matmuls instead of C sequential rank-1 updates.
    """
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    C = min(chunk, T)
    pad = (-T) % C
    if pad:  # padded tokens: beta = 0, g = 0 -> no write, no decay
        q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
        g, beta = F.pad(g, (0, 0, 0, pad)), F.pad(beta, (0, 0, 0, pad))
    n = (T + pad) // C

    def chunks(t):  # (B, T, H, ...) -> (B, H, n, C, ...)
        return t.view(B, n, C, H, *t.shape[3:]).transpose(1, 3).transpose(2, 3)

    q, k, v, g, beta = map(chunks, (q, k, v, g, beta))
    gc = g.cumsum(-1)  # B, H, n, C
    G = (gc[..., :, None] - gc[..., None, :]).tril().exp().tril()  # decay from j to i, i >= j
    kb = k * beta[..., None]
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    M = ((kb @ k.transpose(-1, -2)) * G).tril(-1)
    Tinv = torch.linalg.solve_triangular(eye + M, eye.expand_as(M), upper=False)
    U = Tinv @ (v * beta[..., None])
    W = Tinv @ (kb * gc.exp()[..., None])
    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state
    causal = (q @ k.transpose(-1, -2)) * G  # already lower-triangular through G
    outs = []
    for i in range(n):
        Vn = U[:, :, i] - W[:, :, i] @ S
        outs.append((q[:, :, i] * gc[:, :, i, :, None].exp()) @ S + causal[:, :, i] @ Vn)
        last = gc[:, :, i, -1]
        S = (
            S * last.exp()[..., None, None]
            + (k[:, :, i] * (last[..., None] - gc[:, :, i]).exp()[..., None]).transpose(-1, -2) @ Vn
        )
    o = torch.stack(outs, 2).reshape(B, H, n * C, Dv).transpose(1, 2)[:, :T]
    return o, S


class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int, n_heads: int, linear: LinearFactory, conv_size: int = 4, lb_gates: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        # Large projections: compressed when linear is LittleBit.
        self.q = linear(d_model, d_model)
        self.k = linear(d_model, d_model)
        self.v = linear(d_model, d_model)
        self.gate = linear(d_model, d_model)  # output gate
        self.o = linear(d_model, d_model)
        # Per-head scalars: d_model -> n_heads, a negligible share of the weights.
        small = linear if lb_gates else partial(nn.Linear, bias=False)
        self.a = small(d_model, n_heads)  # decay
        self.b = small(d_model, n_heads)  # write strength
        self.conv_q = ShortConv(d_model, conv_size)
        self.conv_k = ShortConv(d_model, conv_size)
        self.conv_v = ShortConv(d_model, conv_size)
        # Mamba2-style decay init: A in [1, 16], dt in [1e-3, 1e-1].
        self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.o_norm = nn.RMSNorm(self.head_dim)
        self.chunk_min = 16  # sequences at least this long use the chunked rule; shorter (decode) the naive loop

    def forward(self, x: torch.Tensor, initial_state=None, return_state: bool = False):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, D)

        q = F.normalize(heads(self.conv_q(self.q(x))), dim=-1) * D**-0.5
        k = F.normalize(heads(self.conv_k(self.k(x))), dim=-1)
        v = heads(self.conv_v(self.v(x)))
        beta = torch.sigmoid(self.b(x))
        g = -self.A_log.exp() * F.softplus(self.a(x) + self.dt_bias)
        rule = chunk_gated_delta_rule if T >= self.chunk_min else gated_delta_rule
        o, S = rule(q, k, v, g, beta, initial_state)
        o = self.o_norm(o) * F.silu(heads(self.gate(x)))
        out = self.o(o.reshape(B, T, C))
        return (out, S) if return_state else out


# ---------------------------------------------------------------------------------------------
# More recurrent mixers. Each recurrence is a plain per-token loop that takes an optional initial
# state and returns the final state, so callers can carry, snapshot or roll back state (prefill,
# speculative verify). Large projections go through `linear`; per-head gates stay FP.
# ---------------------------------------------------------------------------------------------


def gated_linear_rule(q, k, v, log_decay, initial_state=None):
    """S_t = decay_t * S_{t-1} + k_t v_t^T,  o_t = S_t^T q_t.

    q, k: (B, T, H, Dk); v: (B, T, H, Dv); log_decay (<= 0): (B, T, H) for a scalar per head
    (Mamba2 / SSD, RetNet-style) or (B, T, H, Dk) for a per-key-channel decay (GLA, RWKV-6 style).
    """
    B, T, H, Dk = k.shape
    S = q.new_zeros(B, H, Dk, v.shape[-1]) if initial_state is None else initial_state
    per_channel = log_decay.dim() == 4
    out = []
    for t in range(T):
        a = log_decay[:, t].exp()
        S = S * (a[..., None] if per_channel else a[..., None, None])
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        out.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(out, dim=1), S


class Mamba2(nn.Module):
    """Mamba2 / SSD-style selective SSM (Dao & Gu, 2024), multi-head with scalar decay per head.

    h_t = exp(dt_t * A) h_{t-1} + dt_t * B_t x_t^T,   y_t = C_t^T h_t + D * x_t,   out = o(norm(y) * silu(z)).
    Written through `gated_linear_rule` with q = C, k = B, v = dt * x (the SSD / linear-attention duality).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        linear: LinearFactory,
        d_state: int = 32,
        conv_size: int = 4,
        lb_gates: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim, self.d_state = n_heads, d_model // n_heads, d_state
        self.x = linear(d_model, d_model)
        self.z = linear(d_model, d_model)  # output gate
        self.o = linear(d_model, d_model)
        small = linear if lb_gates else partial(nn.Linear, bias=False)
        self.bc = small(d_model, 2 * n_heads * d_state)  # B and C, per head
        self.dt = small(d_model, n_heads)
        self.conv = ShortConv(d_model, conv_size)
        self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.D = nn.Parameter(torch.ones(n_heads))
        self.norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, initial_state=None, return_state: bool = False):
        B, T, C = x.shape
        H, P, N = self.n_heads, self.head_dim, self.d_state
        xs = self.conv(self.x(x)).view(B, T, H, P)
        b, c = self.bc(x).view(B, T, 2, H, N).unbind(2)
        dt = F.softplus(self.dt(x) + self.dt_bias)  # (B, T, H)
        log_decay = -self.A_log.exp() * dt
        y, S = gated_linear_rule(c, b, xs * dt[..., None], log_decay, initial_state)
        y = y + self.D[:, None] * xs
        y = self.norm(y) * F.silu(self.z(x).view(B, T, H, P))
        out = self.o(y.reshape(B, T, C))
        return (out, S) if return_state else out


class GatedLinearAttention(nn.Module):
    """GLA (Yang et al., 2024) / RWKV-6-style: linear attention with a data-dependent per-channel decay.

    The decay comes from a low-rank FP projection (d -> 16 -> d), as in GLA, kept in log space.
    """

    def __init__(self, d_model: int, n_heads: int, linear: LinearFactory, gate_rank: int = 16, lb_gates: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.q, self.k, self.v = linear(d_model, d_model), linear(d_model, d_model), linear(d_model, d_model)
        self.g = linear(d_model, d_model)  # output gate
        self.o = linear(d_model, d_model)
        small = linear if lb_gates else partial(nn.Linear, bias=False)
        self.decay = nn.Sequential(small(d_model, gate_rank), nn.Linear(gate_rank, d_model, bias=True))
        self.norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, initial_state=None, return_state: bool = False):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        heads = lambda t: t.view(B, T, H, D)
        q, k, v = heads(self.q(x)) * D**-0.5, heads(self.k(x)), heads(self.v(x))
        log_decay = F.logsigmoid(heads(self.decay(x))) / 16  # GLA's temperature: decays start near 1
        o, S = gated_linear_rule(q, k, v, log_decay, initial_state)
        o = self.norm(o) * F.silu(heads(self.g(x)))
        out = self.o(o.reshape(B, T, C))
        return (out, S) if return_state else out


def mlstm_rule(q, k, v, i_pre, f_pre, initial_state=None):
    """xLSTM mLSTM cell (Beck et al., 2024), stabilized, per-token loop.

    C_t = f_t C_{t-1} + i_t k_t v_t^T,  n_t = f_t n_{t-1} + i_t k_t,  h_t = C_t^T q_t / max(|n_t . q_t|, exp(-m_t))
    with i_t = exp(i_pre), f_t = sigmoid(f_pre), and running max m_t for stability.
    State = (C, n, m).
    """
    B, T, H, Dk = k.shape
    if initial_state is None:
        Cm = q.new_zeros(B, H, Dk, v.shape[-1])
        n = q.new_zeros(B, H, Dk)
        m = q.new_zeros(B, H)
    else:
        Cm, n, m = initial_state
    log_f = F.logsigmoid(f_pre)
    out = []
    for t in range(T):
        m_new = torch.maximum(log_f[:, t] + m, i_pre[:, t])
        f = (log_f[:, t] + m - m_new).exp()
        i = (i_pre[:, t] - m_new).exp()
        Cm = f[..., None, None] * Cm + i[..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        n = f[..., None] * n + i[..., None] * k[:, t]
        num = torch.einsum("bhkv,bhk->bhv", Cm, q[:, t])
        den = torch.maximum((n * q[:, t]).sum(-1).abs(), (-m_new).exp())
        out.append(num / den[..., None])
        m = m_new
    return torch.stack(out, dim=1), (Cm, n, m)


class MLSTM(nn.Module):
    """xLSTM matrix-memory mixer: exponential input gate, sigmoid forget gate, normalizer state."""

    def __init__(self, d_model: int, n_heads: int, linear: LinearFactory, conv_size: int = 4, lb_gates: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.q, self.k, self.v = linear(d_model, d_model), linear(d_model, d_model), linear(d_model, d_model)
        self.z = linear(d_model, d_model)
        self.o = linear(d_model, d_model)
        small = linear if lb_gates else partial(nn.Linear, bias=True)
        self.i_gate, self.f_gate = small(d_model, n_heads), small(d_model, n_heads)
        if not lb_gates:
            assert isinstance(self.f_gate, nn.Linear) and isinstance(self.i_gate, nn.Linear)
            nn.init.constant_(self.f_gate.bias, 3.0)  # start remembering (f ~ 0.95), as in xLSTM
            nn.init.constant_(self.i_gate.bias, -3.0)
        self.conv = ShortConv(d_model, conv_size)
        self.norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, initial_state=None, return_state: bool = False):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        heads = lambda t: t.view(B, T, H, D)
        xc = self.conv(x)
        q, k, v = heads(self.q(xc)), heads(self.k(xc)) * D**-0.5, heads(self.v(x))
        h, S = mlstm_rule(q, k, v, self.i_gate(xc), self.f_gate(xc), initial_state)
        h = self.norm(h) * F.silu(heads(self.z(x)))
        out = self.o(h.reshape(B, T, C))
        return (out, S) if return_state else out


def mingru_rule(z, h_tilde, initial_state=None):
    """minGRU (Feng et al., 2024, "Were RNNs All We Needed?"): h_t = (1 - z_t) h_{t-1} + z_t h~_t.

    Gates depend only on the input, so this is also computable with a parallel scan; the loop is the reference.
    """
    h = torch.zeros_like(h_tilde[:, 0]) if initial_state is None else initial_state
    out = []
    for t in range(h_tilde.shape[1]):
        h = torch.lerp(h, h_tilde[:, t], z[:, t])
        out.append(h)
    return torch.stack(out, dim=1), h


class MinGRU(nn.Module):
    """Element-wise RNN mixer: vector state of size d_model (the smallest state of these mixers)."""

    def __init__(self, d_model: int, n_heads: int, linear: LinearFactory, conv_size: int = 4, lb_gates: bool = False):
        super().__init__()
        self.z = linear(d_model, d_model)
        self.h = linear(d_model, d_model)
        self.o = linear(d_model, d_model)
        self.conv = ShortConv(d_model, conv_size)

    def forward(self, x: torch.Tensor, initial_state=None, return_state: bool = False):
        xc = self.conv(x)
        h, S = mingru_rule(torch.sigmoid(self.z(xc)), self.h(xc), initial_state)
        out = self.o(h)
        return (out, S) if return_state else out


MIXERS = {"gdn": GatedDeltaNet, "mamba2": Mamba2, "gla": GatedLinearAttention, "mlstm": MLSTM, "mingru": MinGRU}
