"""LLM-JEPA: a JEPA embedding loss next to the usual next-token loss, with an EMA target encoder.

Follows Huang, LeCun & Balestriero (2025), "LLM-JEPA: Large Language Models Meet Joint Embedding Predictive
Architectures". Training data comes as paired views of the same thing (Text, Code), e.g. a natural-language
request and its regex / SQL. The loss is

    L = L_LLM(Text -> Code) + lambda * d(Pred(Enc(Text)), Enc(Code))

  * Enc(x): last-layer hidden state of the last token of x.
  * Pred: the LLM itself (tied weights). k [PRED] tokens are appended after Text and Enc is read at the last one,
    so k sets how much predictor compute is spent.
  * d: cosine distance, 1 - cos.
  * Packing: one forward runs [Text | Code | PRED * k] under a block-causal mask in which the PRED rows cannot see
    Code. The Text+Code part is plain causal LM (so L_LLM is unchanged), and the PRED rows read only Text.
    PRED tokens take the position ids right after Text, so the packed predictor equals running [Text | PRED].
  * Target: the paper encodes Code with the same model. With target="ema" Code is encoded by an exponential
    moving average of the online weights with stop-gradient (BYOL / I-JEPA style), which avoids collapse
    without needing gradients through the target pass. The EMA copy is training-only: it is not deployed.
  * Loss dropout: with probability p a step skips the JEPA term and runs the plain LM pass only (no PRED tokens,
    no target pass), which cuts the extra training compute by about p.

The model is a TinyLM with mixer="softmax" (the block-causal mask needs attention). Any linear works.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.models.tinylm import TinyLM


def jepa_mask(n_text: int, n_code: int, n_pred: int, device=None) -> torch.Tensor:
    """(T, T) bool, True = may attend. Causal over [Text | Code | PRED], with PRED rows blind to Code."""
    T = n_text + n_code + n_pred
    m = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    m[n_text + n_code :, n_text : n_text + n_code] = False
    return m


def jepa_positions(n_text: int, n_code: int, n_pred: int, device=None) -> torch.Tensor:
    """Position ids for [Text | Code | PRED]: Code and PRED both continue from the end of Text."""
    a = torch.arange(n_text + max(n_code, n_pred), device=device)
    return torch.cat([a[: n_text + n_code], a[n_text : n_text + n_pred]])


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - F.cosine_similarity(a, b, dim=-1)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, decay: float) -> None:
    """target <- decay * target + (1 - decay) * online, for parameters and float buffers."""
    for t, o in zip(target.parameters(), online.parameters(), strict=True):
        t.lerp_(o, 1 - decay)
    for t, o in zip(target.buffers(), online.buffers(), strict=True):
        if t.dtype.is_floating_point:
            t.lerp_(o, 1 - decay)
        else:
            t.copy_(o)


@dataclass
class JEPAConfig:
    pred_token: int  # vocab id reserved for [PRED]
    k_pred: int = 2  # number of [PRED] tokens (predictor depth knob)
    lam: float = 1.0  # weight of the JEPA term
    target: str = "ema"  # "ema" (stop-grad EMA encoder) or "online" (paper: same model, gradients flow)
    ema_decay: float = 0.99
    loss_dropout: float = 0.0  # probability of skipping the JEPA term on a step


class LLMJEPA(nn.Module):
    def __init__(self, lm: TinyLM, cfg: JEPAConfig):
        super().__init__()
        assert lm.cfg.mixer == "softmax", "LLM-JEPA packing needs softmax attention"
        assert cfg.target in ("ema", "online")
        self.lm, self.cfg = lm, cfg
        self.target: TinyLM | None = None
        if cfg.target == "ema":
            self.target = copy.deepcopy(lm).requires_grad_(False)

    def encode_target(self, code: torch.Tensor) -> torch.Tensor:
        """Enc(Code): last-token hidden. Stop-grad EMA encoder, or the online model."""
        if self.target is None:
            return self.lm.hidden(code)[:, -1]
        with torch.no_grad():
            return self.target.hidden(code)[:, -1]

    def predict(self, text: torch.Tensor) -> torch.Tensor:
        """Pred(Enc(Text)) on its own: [Text | PRED * k], read at the last PRED."""
        pred = text.new_full((text.shape[0], self.cfg.k_pred), self.cfg.pred_token)
        return self.lm.hidden(torch.cat([text, pred], 1))[:, -1]

    def loss(self, text: torch.Tensor, code: torch.Tensor, generator: torch.Generator | None = None) -> tuple:
        """Returns (total loss, info dict). text (B, Lt), code (B, Lc) token ids."""
        B, Lt = text.shape
        Lc, k = code.shape[1], self.cfg.k_pred
        drop = self.cfg.loss_dropout > 0 and torch.rand((), generator=generator).item() < self.cfg.loss_dropout
        if drop or self.cfg.lam == 0:
            logits = self.lm(torch.cat([text, code], 1))
            h_pred = None
        else:
            pred = text.new_full((B, k), self.cfg.pred_token)
            seq = torch.cat([text, code, pred], 1)
            h = self.lm.hidden(seq, jepa_mask(Lt, Lc, k, text.device), jepa_positions(Lt, Lc, k, text.device))
            logits = self.lm.head(h[:, : Lt + Lc])
            h_pred = h[:, -1]
        # next-token loss on Code given Text: positions Lt-1 .. Lt+Lc-2 predict code[0 .. Lc-1]
        lm_logits = logits[:, Lt - 1 : Lt + Lc - 1]
        lm_loss = F.cross_entropy(lm_logits.flatten(0, 1), code.flatten())
        info = {"lm": lm_loss.item(), "jepa_dropped": h_pred is None}
        if h_pred is None:
            return lm_loss, info
        jepa = cosine_distance(h_pred, self.encode_target(code)).mean()
        info["jepa"] = jepa.item()
        return lm_loss + self.cfg.lam * jepa, info

    @torch.no_grad()
    def step_target(self) -> None:
        """Call after each optimizer step."""
        if self.target is not None:
            ema_update(self.target, self.lm, self.cfg.ema_decay)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.target is not None:
            self.target.eval()  # the target never trains (and keeps inference caches valid)
        return self


@torch.no_grad()
def retrieval_accuracy(model: LLMJEPA, text: torch.Tensor, code: torch.Tensor) -> float:
    """Top-1 of matching each Pred(Enc(Text_i)) to its own Enc(Code_i) among the batch, by cosine."""
    p = F.normalize(model.predict(text), dim=-1)
    c = F.normalize(model.encode_target(code), dim=-1)
    return (p @ c.t()).argmax(-1).eq(torch.arange(len(p), device=p.device)).float().mean().item()
