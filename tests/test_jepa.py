import math

import torch

from lilbits.data import PairedViewSource
from lilbits.models.jepa import (
    LLMJEPA,
    JEPAConfig,
    cosine_distance,
    ema_update,
    jepa_mask,
    jepa_positions,
    retrieval_accuracy,
)
from lilbits.models.tinylm import Config, TinyLM


def make(target="ema", **kw):
    src = PairedViewSource(n_symbols=6, length=5, seed=0)
    lm = TinyLM(Config(vocab_size=src.vocab, seq_len=16, d_model=32, n_heads=2, n_layers=2, **kw))
    return src, LLMJEPA(lm, JEPAConfig(pred_token=src.pred_token, k_pred=2, target=target))


def test_mask_blocks_pred_from_code():
    m = jepa_mask(3, 2, 2)
    assert m[5:, 3:5].sum() == 0  # PRED rows never see Code
    assert m[5:, :3].all()  # but see all of Text
    assert torch.equal(m[:5, :5], torch.ones(5, 5, dtype=torch.bool).tril())  # Text+Code: plain causal


def test_packed_lm_loss_matches_plain_pass():
    """Appending PRED tokens under the block-causal mask leaves the Text->Code logits unchanged."""
    torch.manual_seed(0)
    src, m = make()
    text, code = src.sample(3)
    plain = m.lm(torch.cat([text, code], 1))
    pred = text.new_full((3, 2), src.pred_token)
    seq = torch.cat([text, code, pred], 1)
    packed = m.lm.head(m.lm.hidden(seq, jepa_mask(5, 5, 2), jepa_positions(5, 5, 2)))[:, :10]
    torch.testing.assert_close(packed, plain, rtol=1e-4, atol=1e-5)


def test_packed_pred_matches_separate_predictor():
    """The PRED readout from the packed pass equals running [Text | PRED] alone."""
    torch.manual_seed(0)
    src, m = make()
    text, code = src.sample(3)
    pred = text.new_full((3, 2), src.pred_token)
    h = m.lm.hidden(torch.cat([text, code, pred], 1), jepa_mask(5, 5, 2), jepa_positions(5, 5, 2))[:, -1]
    torch.testing.assert_close(h, m.predict(text), rtol=1e-4, atol=1e-5)


def test_ema_target_gets_no_grad_and_tracks_online():
    torch.manual_seed(0)
    src, m = make("ema")
    assert m.target is not None
    loss, info = m.loss(*src.sample(4))
    loss.backward()
    assert math.isfinite(loss.item()) and "jepa" in info
    assert all(p.grad is None for p in m.target.parameters())
    assert all(p.grad is not None for p in m.lm.parameters())
    with torch.no_grad():
        for p in m.lm.parameters():
            p.add_(1.0)
    before = next(m.target.parameters()).clone()
    m.step_target()
    after, online = next(m.target.parameters()), next(m.lm.parameters())
    torch.testing.assert_close(after, 0.99 * before + 0.01 * online)


def test_ema_update_decay_one_and_zero():
    a, b = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    w = a.weight.clone()
    ema_update(a, b, 1.0)
    torch.testing.assert_close(a.weight, w)
    ema_update(a, b, 0.0)
    torch.testing.assert_close(a.weight, b.weight)


def test_online_target_and_loss_dropout():
    torch.manual_seed(0)
    src, m = make("online")
    assert m.target is None
    loss, info = m.loss(*src.sample(4))
    loss.backward()
    assert "jepa" in info
    m.cfg.loss_dropout = 1.0
    loss, info = m.loss(*src.sample(4))
    assert info["jepa_dropped"] and "jepa" not in info


def test_littlebit_and_retrieval_runs():
    torch.manual_seed(0)
    src, m = make("ema", linear="littlebit", bpw=1.0)
    loss, _ = m.loss(*src.sample(4))
    loss.backward()
    acc = retrieval_accuracy(m.eval(), *src.sample(8))
    assert 0.0 <= acc <= 1.0
    assert cosine_distance(torch.ones(2, 4), torch.ones(2, 4)).abs().max() < 1e-6


def test_paired_source_is_deterministic_view():
    src = PairedViewSource(n_symbols=4, length=3, seed=1)
    text, code = src.sample(5)
    assert code.min() >= 4 and code.max() < 8 and src.vocab == 9
    torch.testing.assert_close(code, src.sub[text.flip(1)])
