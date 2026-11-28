"""A small LM with a two-level memory and a sleep phase that consolidates fast memory into new low-rank experts.

After Behrouz, Hashemi, Javanmard and Mirrokni, "Language Models Need Sleep: Learning to Self-Modify and
Consolidate Memories" (Google Research, 2026, arXiv:2606.03979). The model alternates between two phases:

    wake    new information is written into FAST memory: a low-rank adapter on every MLP, trained by gradient steps
            on the session's text (the high-frequency level of a Continuum Memory System; it stands in for the
            paper's in-context / high-frequency memory)
    sleep   1. parameter expansion: a new low-rank expert {A, B} with its own token router is unlocked in every
               block's SLOW memory (pre-allocated and masked until then, as the paper suggests)
            2. knowledge seeding (upward distillation): teacher = the model with its fast memory, student = the
               model with the fast memory off and the new expert on; only the new expert and its router train.
               Generalized knowledge distillation on "dreams" sampled from the teacher (1 - lambda) and on the
               student's own samples (lambda), plus learning-to-imitate (LTI): the student completes a prefix of
               a teacher dream and is rewarded by Levenshtein similarity, r_abs = 1 - z / max(|d^|, |d|) if
               z / max(...) <= z0, else 0 (REINFORCE with a mean baseline)
            3. dreaming: teacher dreams sampled with a random earlier expert forced on (mixing old knowledge in),
               scored by the gradient norm of their SFT loss on the new expert; the top-k plus b random ones are
               used for SFT of the new expert
            4. synaptic pruning: the fast memory is reset; the new expert stays, frozen from then on

The base weights never change after pretraining, and each session's knowledge goes to its own expert, so later
sessions cannot overwrite earlier ones (no catastrophic forgetting by construction, at the cost of rank x
(d_in + d_out) + d_in parameters per block per session).

Left out at this scale: the semantic reward model (gamma = 0), ReSTEM's isolated-copy evaluation of dreams, multi-
level (> 2) CMS hierarchies and MoE base experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.layers.ffn import SwiGLU
from lilbits.mixers.attention import Attention
from lilbits.models.tinylm import Config, make_linear_factory


@dataclass
class ConsolidationConfig:
    vocab_size: int = 256
    seq_len: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: float = 8 / 3
    fast_rank: int = 16  # rank of the fast (wake) memory
    expert_rank: int = 8  # rank of each slow expert unlocked at sleep
    max_experts: int = 8  # pre-allocated slow experts (one per sleep)
    router_bias: float = 0.0  # initial router bias of a new expert (gate = sigmoid(bias) before training)
    linear: str = "fp"  # fp | littlebit | bitnet for the base block linears
    bpw: float = 1.0


class LowRank(nn.Module):
    """x -> gate(x) * B A x with A (r, d_in), B (d_out, r). B starts at zero, so a fresh expert changes nothing."""

    def __init__(self, d_in: int, d_out: int, rank: int, routed: bool, router_bias: float = 0.0):
        super().__init__()
        self.router_bias = router_bias
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        self.router = nn.Linear(d_in, 1) if routed else None
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        nn.init.normal_(self.A, std=self.A.shape[1] ** -0.5)
        self.B.zero_()
        if self.router is not None:
            nn.init.zeros_(self.router.weight)
            nn.init.constant_(self.router.bias, self.router_bias)

    def forward(self, x: torch.Tensor, force: bool = False) -> torch.Tensor:
        y = F.linear(F.linear(x, self.A), self.B)
        if self.router is None or force:
            return y
        return torch.sigmoid(self.router(x)) * y


class MemoryBlock(nn.Module):
    """Pre-norm attention + SwiGLU block whose MLP output gets the fast memory and the active slow experts added."""

    def __init__(self, cfg: ConsolidationConfig):
        super().__init__()
        d = cfg.d_model
        linear = make_linear_factory(Config(linear=cfg.linear, bpw=cfg.bpw))
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = Attention(d, cfg.n_heads, linear, qk_norm=True)
        self.mlp = SwiGLU(d, round(d * cfg.mlp_ratio / 8) * 8, linear)
        self.fast = LowRank(d, d, cfg.fast_rank, routed=False)
        self.experts = nn.ModuleList(
            LowRank(d, d, cfg.expert_rank, routed=True, router_bias=cfg.router_bias) for _ in range(cfg.max_experts)
        )

    def forward(self, h: torch.Tensor, fast: bool, active: list[int], forced: int | None) -> torch.Tensor:
        h = h + self.attn(self.n1(h))
        x = self.n2(h)
        y = self.mlp(x)
        if fast:
            y = y + self.fast(x)
        for e in active:
            y = y + self.experts[e](x, force=e == forced)
        return h + y


class ConsolidatingLM(nn.Module):
    def __init__(self, cfg: ConsolidationConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(MemoryBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        # memory state (saved with the checkpoint): how many slow experts are unlocked, and whether fast memory is on
        self.register_buffer("n_active", torch.zeros((), dtype=torch.long))
        self.fast_on = False
        self.forced: int | None = None  # dreaming: one expert forced fully on (gate = 1)
        self.extra: list[int] = []  # experts switched on beyond the unlocked ones (the one being trained)
        self.skip: list[int] = []  # unlocked experts switched off (the teacher does not have the new expert)

    # ---- which parameters belong to which memory level
    def base_params(self) -> list[nn.Parameter]:
        mem = {id(p) for p in self.memory_params()}
        return [p for p in self.parameters() if id(p) not in mem]

    def memory_params(self) -> list[nn.Parameter]:
        return [p for b in self.blocks for m in (b.fast, *b.experts) for p in m.parameters()]

    def fast_params(self) -> list[nn.Parameter]:
        return [p for b in self.blocks for p in b.fast.parameters()]

    def expert_params(self, e: int) -> list[nn.Parameter]:
        return [p for b in self.blocks for p in b.experts[e].parameters()]

    def active(self) -> list[int]:
        on = [e for e in range(int(self.n_active)) if e not in self.skip]
        return on + [e for e in self.extra if e not in on]

    def reset_fast(self) -> None:
        for b in self.blocks:
            b.fast.reset()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        act = self.active()
        for b in self.blocks:
            h = b(h, self.fast_on, act, self.forced)
        return self.head(self.norm(h))

    def lm_loss(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ce = F.cross_entropy(self(x[:, :-1]).transpose(1, 2), x[:, 1:], reduction="none")
        if mask is None:
            return ce.mean()
        m = mask[:, 1:].float()
        return (ce * m).sum() / m.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, prompt: torch.Tensor, n: int, temperature: float = 1.0, generator=None) -> torch.Tensor:
        """Sample n bytes after prompt (B, P); full recompute per byte (sequences here are short)."""
        x = prompt
        for _ in range(n):
            logits = self(x[:, -self.cfg.seq_len :])[:, -1]
            if temperature <= 0:
                tok = logits.argmax(-1, keepdim=True)
            else:
                tok = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=generator)
            x = torch.cat([x, tok], 1)
        return x

    def stored_memory_bytes(self, fp_bytes: int = 2) -> dict:
        """FP16 bytes of the fast memory and of the unlocked slow experts."""
        fast = sum(p.numel() for p in self.fast_params())
        slow = sum(p.numel() for e in range(int(self.n_active)) for p in self.expert_params(e))
        return {"fast_bytes": fast * fp_bytes, "slow_expert_bytes": slow * fp_bytes}


def levenshtein(a: list[int], b: list[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def imitation_reward(student: list[int], teacher: list[int], z0: float = 0.5) -> float:
    """r_abs of the paper's Eq. 4: 1 - z / max(|d^|, |d|) when that normalized distance is at most z0, else 0."""
    n = max(len(student), len(teacher), 1)
    z = levenshtein(student, teacher) / n
    return 1.0 - z if z <= z0 else 0.0


@dataclass
class SleepSettings:
    consolidate_steps: int = 60
    dream_steps: int = 30
    batch: int = 16
    gen_len: int = 24  # bytes per dream after its prompt
    lam: float = 0.5  # fraction of on-policy (student-sampled) distillation batches
    alpha: float = 0.8  # weight of the divergence vs the LTI reward
    z0: float = 0.5  # LTI similarity threshold
    n_dreams: int = 48  # dreams generated for selection
    top_k: int = 16  # kept by gradient norm
    n_random: int = 4  # plus this many random ones, for diversity
    lr: float = 1e-2
    anchor: float = 1.0  # weight of the anchor KL on general text (keeps the base's behaviour where no fact is)


def _kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Forward KL(teacher || student) per position, averaged."""
    t = F.log_softmax(teacher_logits, -1)
    return (t.exp() * (t - F.log_softmax(student_logits, -1))).sum(-1).mean()


class _Roles:
    """Switch one model between the teacher (fast memory on, new expert off) and the student (the reverse)."""

    def __init__(self, model: ConsolidatingLM, new: int):
        self.m, self.new = model, new

    def teacher(self) -> ConsolidatingLM:
        self.m.fast_on, self.m.extra, self.m.forced = True, [], None
        return self.m

    def student(self) -> ConsolidatingLM:
        self.m.fast_on, self.m.extra, self.m.forced = False, [self.new], None
        return self.m


def wake(model: ConsolidatingLM, batch_fn, steps: int, lr: float = 3e-3) -> float:
    """Write a session into fast memory: gradient steps on the fast adapters only. Returns the last loss."""
    model.fast_on, model.extra, model.forced = True, [], None
    opt = torch.optim.Adam(model.fast_params(), lr=lr)
    loss = torch.zeros(())
    for i in range(steps):
        x, mask = batch_fn(i)
        loss = model.lm_loss(x, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item()


def sleep(
    model: ConsolidatingLM, prompt_fn, s: SleepSettings, dreaming: bool = True, generator=None, anchor_fn=None
) -> dict:
    """One sleep: unlock a new expert, seed it from the fast memory (GKD + LTI), dream, then prune fast memory.

    prompt_fn(i) -> (B, P) byte prompts drawn from the session's context (the paper samples dreams from LM(. | C)).
    anchor_fn(i) -> (B, T) general text (optional): the student is also distilled there from the model as it was
    before this session (no fast memory, no new expert), so the new expert stays quiet where it has nothing to add.
    """
    new = int(model.n_active)
    assert new < model.cfg.max_experts, "all pre-allocated experts are in use"
    for b in model.blocks:
        b.experts[new].reset()
    roles, g = _Roles(model, new), generator
    opt = torch.optim.Adam(model.expert_params(new), lr=s.lr)
    info: dict = {"expert": new}
    kls, rewards = [], []
    for i in range(s.consolidate_steps):
        p = prompt_fn(i)
        on_policy = torch.rand((), generator=g).item() < s.lam
        with torch.no_grad():
            src = roles.student() if on_policy else roles.teacher()
            seq = src.sample(p, s.gen_len, generator=g)
            t_logits = roles.teacher()(seq[:, :-1])
        kl = _kl(t_logits, roles.student()(seq[:, :-1]))
        loss = s.alpha * kl
        if anchor_fn is not None and s.anchor > 0:
            a = anchor_fn(i)
            with torch.no_grad():
                model.fast_on, model.extra = False, []
                a_logits = model(a)
            loss = loss + s.anchor * _kl(a_logits, roles.student()(a))
        if s.alpha < 1:  # learning to imitate: complete a prefix of a teacher dream, reward by edit distance
            with torch.no_grad():
                dream = roles.teacher().sample(p, s.gen_len, generator=g)
                cut = p.shape[1] + int(torch.randint(1, s.gen_len // 2 + 1, (), generator=g))
                comp = roles.student().sample(dream[:, :cut], dream.shape[1] - cut, generator=g)
                r = torch.tensor(
                    [
                        imitation_reward(c[cut:].tolist(), d[cut:].tolist(), s.z0)
                        for c, d in zip(comp, dream, strict=True)
                    ]
                )
            logp = F.log_softmax(roles.student()(comp[:, :-1]), -1)
            tok_lp = logp.gather(-1, comp[:, 1:, None])[..., 0][:, cut - 1 :].sum(-1)
            loss = loss - (1 - s.alpha) * ((r - r.mean()) * tok_lp).mean()
            rewards.append(r.mean().item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        kls.append(kl.item())
    info["seed_kl_first"], info["seed_kl_last"] = kls[0], sum(kls[-5:]) / len(kls[-5:])
    if rewards:
        info["lti_reward_last"] = sum(rewards[-5:]) / len(rewards[-5:])

    if dreaming and s.dream_steps > 0:
        # dreams from the teacher with a random earlier expert forced on, when there is one
        dreams = []
        for i in range(0, s.n_dreams, s.batch):
            t = roles.teacher()
            if new > 0:
                t.forced = int(torch.randint(0, new, (), generator=g))
            with torch.no_grad():
                dreams.append(t.sample(prompt_fn(10_000 + i), s.gen_len, generator=g))
        pool = torch.cat(dreams)[: s.n_dreams]
        # gradient-based selection: norm of the SFT gradient on the new expert, per dream
        st, params = roles.student(), model.expert_params(new)
        scores = []
        for d in pool:
            grads = torch.autograd.grad(st.lm_loss(d[None]), params, allow_unused=True)
            scores.append(sum((x.norm() ** 2 for x in grads if x is not None), torch.zeros(())).sqrt().item())
        order = torch.tensor(scores).argsort(descending=True)
        keep = order[: s.top_k]
        rest = order[s.top_k :]
        if len(rest):
            keep = torch.cat([keep, rest[torch.randperm(len(rest), generator=g)[: s.n_random]]])
        chosen = pool[keep]
        for _ in range(s.dream_steps):
            ix = torch.randint(0, len(chosen), (min(s.batch, len(chosen)),), generator=g)
            loss = roles.student().lm_loss(chosen[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        info["dreams_kept"], info["dream_sft_last"] = len(chosen), loss.item()

    # the new expert joins slow memory (frozen from now on); synaptic pruning resets fast memory
    model.n_active += 1
    model.reset_fast()
    model.fast_on, model.extra, model.forced = False, [], None
    return info
