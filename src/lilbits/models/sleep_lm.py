"""Transformer LM that "sleeps" on each finished context window to refine its KV cache.

After Lee et al., "Sample-Efficient In-Context Adaptation via Sleep" (2026, arXiv:2611.32802). The context is cut
into windows of W tokens. Tokens are read normally, one causal ("wake") pass each, and only the wake pass makes
next-token predictions. When a window fills, the model sleeps on it: it runs the whole layer stack N - 1 more
times over that window, feeding the last layer's output of one loop back in as the next loop's input, with the
same weights every loop. Each sleep loop recomputes keys and values for the window's tokens and merges them into
the window's cache entries with a learned elementwise GRU-style gate (Eqs. 4-5 of the paper):

    alpha_K = sigmoid([K_new, K] theta_K^T + b_K)
    K      <- (1 - alpha_K) * K_new + alpha_K * K          (and the same for V with its own theta_V, b_V)

Later windows then read the refined cache. Sleep attention is bidirectional inside the window (the paper's Table 4
finds the causal mask slightly worse) and sees the committed cache of earlier windows. Training backpropagates
through every loop and every window, without truncation; the loss is the wake pass's next-token loss only, so
sleep is trained only by how much it helps the predictions of later windows.

What it costs: N = 1 is the plain transformer. N > 1 adds only the two gates per layer (d_kv x 2 d_kv each, under
2% of the block weights with grouped-query attention), leaves the KV cache the same size, and spends N - 1 extra
stack passes per W tokens, off the per-token decode path. The paper's 1.2B models reach the no-sleep model's
held-out code loss with 2.10x (N = 2) and 2.39x (N = 4) fewer in-context tokens.

Simplifications here: plain RoPE with NoPE every `nope_every`-th layer (as in the paper), no Attention Residuals,
no short convolutions, Adam-family training. Block linears can be FP, LittleBit or BitNet via `linear`; the
gates stay FP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.ffn import SwiGLU
from lilbits.models.tinylm import Config, make_linear_factory


@dataclass
class SleepConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 1  # grouped-query attention: keeps the KV cache and the gates small
    n_layers: int = 4
    mlp_ratio: float = 8 / 3
    window: int = 32  # W: tokens per sleep window (the paper uses 256 at 4k context)
    loops: int = 2  # N: total passes per window; 1 = no sleep (plain transformer)
    sleep_causal: bool = False  # causal mask inside the window during sleep (paper default: bidirectional)
    nope_every: int = 4  # every k-th layer has no positional encoding (0 = RoPE everywhere)
    rope_base: float = 10000.0
    gate_bias: float = 1.0  # initial gate bias: sigmoid(1) ~ 0.73 of the old cache is kept at the start
    linear: str = "fp"  # fp | littlebit | bitnet, for the block linears
    bpw: float = 1.0


def rope(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """Rotary embedding of x (..., T, dh) at absolute positions pos (T,)."""
    half = x.shape[-1] // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.to(torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class SleepBlock(nn.Module):
    """Pre-norm GQA attention + SwiGLU block that can attend over an external cache, plus the K / V sleep gates."""

    def __init__(self, cfg: SleepConfig, use_rope: bool):
        super().__init__()
        d, H, Hk = cfg.d_model, cfg.n_heads, cfg.n_kv_heads
        assert d % H == 0 and H % Hk == 0
        self.H, self.Hk, self.dh, self.use_rope, self.base = H, Hk, d // H, use_rope, cfg.rope_base
        linear = make_linear_factory(Config(linear=cfg.linear, bpw=cfg.bpw))
        dkv = Hk * self.dh
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q, self.k, self.v, self.o = linear(d, d), linear(d, dkv), linear(d, dkv), linear(d, d)
        self.q_norm, self.k_norm = nn.RMSNorm(self.dh), nn.RMSNorm(self.dh)
        self.ffn = SwiGLU(d, int(cfg.mlp_ratio * d), linear)
        self.gate_k: nn.Linear | None = None
        self.gate_v: nn.Linear | None = None
        if cfg.loops > 1:
            self.gate_k, self.gate_v = nn.Linear(2 * dkv, dkv), nn.Linear(2 * dkv, dkv)
            for g in (self.gate_k, self.gate_v):
                nn.init.normal_(g.weight, std=0.02)
                nn.init.constant_(g.bias, cfg.gate_bias)

    def qkv(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries (B, H, T, dh) and pre-RoPE keys / values (B, Hk, T, dh): the cache stores these."""
        B, T, _ = h.shape
        x = self.n1(h)
        q = self.q_norm(self.q(x).view(B, T, self.H, self.dh).transpose(1, 2))
        k = self.k_norm(self.k(x).view(B, T, self.Hk, self.dh).transpose(1, 2))
        v = self.v(x).view(B, T, self.Hk, self.dh).transpose(1, 2)
        return q, k, v

    def attend(self, h, q, k, v, q_pos, k_pos, mask: torch.Tensor | None) -> torch.Tensor:
        """Attention of q (at q_pos) over k / v (at k_pos), then the residual MLP. mask: bool (Tq, Tk) or None."""
        if self.use_rope:
            q, k = rope(q, q_pos, self.base), rope(k, k_pos, self.base)
        rep = self.H // self.Hk
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        B, _, T, _ = q.shape
        h = h + self.o(y.transpose(1, 2).reshape(B, T, -1))
        return h + self.ffn(self.n2(h))

    @staticmethod
    def refine(gate: nn.Linear, new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        """GRU-style elementwise merge over the whole KV width: (1 - a) * new + a * old."""
        B, Hk, T, dh = new.shape
        n, o = new.transpose(1, 2).reshape(B, T, -1), old.transpose(1, 2).reshape(B, T, -1)
        a = torch.sigmoid(gate(torch.cat([n, o], -1)))
        return ((1 - a) * n + a * o).view(B, T, Hk, dh).transpose(1, 2)


@dataclass
class SleepState:
    """Per-layer KV cache: committed (slept-on) windows, plus the window being filled and its wake outputs."""

    k: list[torch.Tensor]
    v: list[torch.Tensor]
    pk: list[torch.Tensor]
    pv: list[torch.Tensor]
    ph: torch.Tensor  # (B, pending, d): last-layer wake outputs of the pending tokens (sleep loop 2's input)
    pos: int = 0
    sleeps: int = field(default=0)

    @property
    def pending(self) -> int:
        return self.ph.shape[1]

    def kv_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.k, *self.v, *self.pk, *self.pv))


class SleepLM(nn.Module):
    def __init__(self, cfg: SleepConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            SleepBlock(cfg, use_rope=not (cfg.nope_every and (i + 1) % cfg.nope_every == 0))
            for i in range(cfg.n_layers)
        )
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied embeddings, as in the paper
        nn.init.normal_(self.embed.weight, std=0.02)

    def sleep_params(self) -> int:
        return sum(
            p.numel() for b in self.blocks for g in (b.gate_k, b.gate_v) if g is not None for p in g.parameters()
        )

    def init_state(self, batch: int, device=None) -> SleepState:
        c, dh = self.cfg, self.cfg.d_model // self.cfg.n_heads
        dt = self.embed.weight.dtype
        e = lambda: torch.zeros(batch, c.n_kv_heads, 0, dh, device=device, dtype=dt)
        L = c.n_layers
        return SleepState(
            [e() for _ in range(L)],
            [e() for _ in range(L)],
            [e() for _ in range(L)],
            [e() for _ in range(L)],
            torch.zeros(batch, 0, c.d_model, device=device, dtype=dt),
        )

    def _wake(self, x: torch.Tensor, st: SleepState) -> torch.Tensor:
        """One causal pass over n new tokens (all inside the current window); appends them to the pending window."""
        n, dev = x.shape[1], x.device
        q_pos = torch.arange(st.pos, st.pos + n, device=dev)
        k_pos = torch.arange(0, st.pos + n, device=dev)
        mask = k_pos[None, :] <= q_pos[:, None]
        h = self.embed(x)
        for i, blk in enumerate(self.blocks):
            q, k, v = blk.qkv(h)
            K = torch.cat([st.k[i], st.pk[i], k], 2)
            V = torch.cat([st.v[i], st.pv[i], v], 2)
            h = blk.attend(h, q, K, V, q_pos, k_pos, mask)
            st.pk[i], st.pv[i] = torch.cat([st.pk[i], k], 2), torch.cat([st.pv[i], v], 2)
        st.ph = torch.cat([st.ph, h], 1)
        st.pos += n
        return self.head(self.norm(h))

    def _sleep(self, st: SleepState) -> None:
        """N - 1 recurrent passes of the whole stack over the pending window, gating its cache entries each loop."""
        W, dev = st.pending, st.ph.device
        q_pos = torch.arange(st.pos - W, st.pos, device=dev)
        k_pos = torch.arange(0, st.pos, device=dev)
        mask = (k_pos[None, :] <= q_pos[:, None]) if self.cfg.sleep_causal else None
        h = st.ph
        for _ in range(self.cfg.loops - 1):
            for i, blk in enumerate(self.blocks):
                assert blk.gate_k is not None and blk.gate_v is not None
                q, kn, vn = blk.qkv(h)
                K, V = torch.cat([st.k[i], kn], 2), torch.cat([st.v[i], vn], 2)
                h = blk.attend(h, q, K, V, q_pos, k_pos, mask)
                st.pk[i] = blk.refine(blk.gate_k, kn, st.pk[i])
                st.pv[i] = blk.refine(blk.gate_v, vn, st.pv[i])
        st.sleeps += 1

    def _commit(self, st: SleepState) -> None:
        for i in range(self.cfg.n_layers):
            st.k[i], st.v[i] = torch.cat([st.k[i], st.pk[i]], 2), torch.cat([st.v[i], st.pv[i]], 2)
            st.pk[i], st.pv[i] = st.pk[i][:, :, :0], st.pv[i][:, :, :0]
        st.ph = st.ph[:, :0]

    def step(self, x: torch.Tensor, st: SleepState, sleep_on_full: bool = True) -> torch.Tensor:
        """Read x (B, n) into the state, window by window; returns wake logits (B, n, vocab). Every window that
        fills is slept on (N > 1) and committed. sleep_on_full=False skips the sleep of a window that fills at
        the very end of x (training: nothing later reads it)."""
        W, out, i = self.cfg.window, [], 0
        while i < x.shape[1]:
            n = min(W - st.pending, x.shape[1] - i)
            out.append(self._wake(x[:, i : i + n], st))
            i += n
            if st.pending == W:
                if self.cfg.loops > 1 and (sleep_on_full or i < x.shape[1]):
                    self._sleep(st)
                self._commit(st)
        return torch.cat(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training / scoring forward: logits for every position of x, sleeping on every finished window but the
        last one. Gradients flow through all sleep loops and across windows."""
        return self.step(x, self.init_state(x.shape[0], x.device), sleep_on_full=False)

    def training_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits = self(x[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()), {}

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, n: int, temperature: float = 0.0) -> tuple[torch.Tensor, SleepState]:
        """Greedy (or sampled) decode: one wake pass per token, a sleep whenever a window fills."""
        st = self.init_state(prompt.shape[0], prompt.device)
        logits = self.step(prompt, st)[:, -1]
        out = [prompt]
        for _ in range(n):
            if temperature > 0:
                tok = torch.multinomial(F.softmax(logits / temperature, -1), 1)
            else:
                tok = logits.argmax(-1, keepdim=True)
            out.append(tok)
            logits = self.step(tok, st)[:, -1]
        return torch.cat(out, 1), st
