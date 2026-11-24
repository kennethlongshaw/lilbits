import math

import pytest
import torch
from torch import nn

from lilbits.data import MarkovSource
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.mixers.recurrent import GatedDeltaNet, gated_delta_rule
from lilbits.models.tinylm import Config, TinyLM, TransformerBlock, convert_fp_to_littlebit, make_linear_factory
from lilbits.training.train import TrainConfig, make_optimizers, train

TINY = dict(vocab_size=16, seq_len=16, d_model=32, n_heads=2, n_layers=1, mixer="gdn")


@pytest.mark.parametrize("linear", ["fp", "littlebit"])
def test_gdn_block_shape_and_causality(linear):
    torch.manual_seed(0)
    cfg = Config(**TINY, linear=linear)
    block = TransformerBlock(32, 2, 64, make_linear_factory(cfg), mixer="gdn")
    x = torch.randn(2, 10, 32)
    y = block(x)
    assert y.shape == x.shape
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 32)
    torch.testing.assert_close(block(x2)[:, :6], y[:, :6])


def test_delta_rule_stores_and_recalls_with_orthonormal_keys():
    """beta=1, no decay, orthonormal keys: the state is an exact key->value memory."""
    torch.manual_seed(0)
    T, D = 4, 8
    keys = torch.linalg.qr(torch.randn(D, D))[0][:T]  # T orthonormal rows
    v = torch.randn(1, T, 1, D)
    k = keys[None, :, None, :]
    zeros = torch.zeros(1, T, 1)
    _, S = gated_delta_rule(k, k, v, g=zeros, beta=torch.ones(1, T, 1))
    recalled = torch.einsum("bhkv,tk->tv", S, keys)
    torch.testing.assert_close(recalled, v[0, :, 0], atol=1e-5, rtol=1e-5)
    # Overwrite: writing key 0 again with a new value replaces the old one (the "delta" in delta rule).
    v_new = torch.randn(1, 1, 1, D)
    _, S2 = gated_delta_rule(k[:, :1], k[:, :1], v_new, g=zeros[:, :1], beta=torch.ones(1, 1, 1), initial_state=S)
    torch.testing.assert_close(torch.einsum("bhkv,k->v", S2, keys[0]), v_new[0, 0, 0], atol=1e-5, rtol=1e-5)


def test_delta_rule_state_carries_across_chunks():
    torch.manual_seed(0)
    B, T, H, D = 2, 12, 3, 4
    q, k = torch.randn(B, T, H, D), torch.nn.functional.normalize(torch.randn(B, T, H, D), dim=-1)
    v, g, beta = torch.randn(B, T, H, D), -torch.rand(B, T, H), torch.rand(B, T, H)
    full, S = gated_delta_rule(q, k, v, g, beta)
    a, Sa = gated_delta_rule(q[:, :5], k[:, :5], v[:, :5], g[:, :5], beta[:, :5])
    b, Sb = gated_delta_rule(q[:, 5:], k[:, 5:], v[:, 5:], g[:, 5:], beta[:, 5:], initial_state=Sa)
    torch.testing.assert_close(torch.cat([a, b], 1), full)
    torch.testing.assert_close(Sb, S)


@pytest.mark.parametrize("lb_gates, n_lb", [(False, 8), (True, 10)])
def test_gdn_littlebit_placement(lb_gates, n_lb):
    model = TinyLM(Config(**TINY, linear="littlebit", lb_gates=lb_gates))
    lb = [n for n, m in model.named_modules() if isinstance(m, LittleBitLinear)]
    assert len(lb) == n_lb  # q k v gate o (+ a b) in the mixer, gate up down in the MLP
    mixer = model.blocks[0].attn
    assert isinstance(mixer, GatedDeltaNet)
    assert isinstance(mixer.a, LittleBitLinear if lb_gates else nn.Linear)


def test_gdn_fp_to_littlebit_conversion_keeps_fp_parts():
    torch.manual_seed(0)
    fp = TinyLM(Config(**TINY, linear="fp"))
    lb = convert_fp_to_littlebit(fp, Config(**TINY, linear="littlebit", bpw=8.0, itq_iters=10))
    for name in ["blocks.0.attn.A_log", "blocks.0.attn.dt_bias", "blocks.0.attn.a.weight",
                 "blocks.0.attn.conv_q.conv.weight", "blocks.0.attn.o_norm.weight"]:
        torch.testing.assert_close(lb.state_dict()[name], fp.state_dict()[name])
    x = torch.randint(0, 16, (2, 16))
    assert torch.isfinite(lb(x)).all()


def test_muon_gets_only_2d_matrices_with_gdn():
    model = TinyLM(Config(**dict(TINY, d_model=64), linear="littlebit", bpw=4.0))
    muon, adam = make_optimizers(model, TrainConfig(lr=1e-3, optim="muon"))
    assert all(p.dim() == 2 for g in muon.param_groups for p in g["params"])
    conv = model.blocks[0].attn.conv_q.conv.weight
    assert any(p is conv for g in adam.param_groups for p in g["params"])


@pytest.mark.parametrize("linear", ["fp", "littlebit"])
def test_gdn_training_from_scratch_learns(linear):
    src = MarkovSource(vocab=8, order=1, branching=2, seed=0)
    cfg = Config(vocab_size=8, seq_len=16, d_model=32, n_heads=2, n_layers=1, linear=linear, mixer="gdn",
                 bpw=2.0, residual=False, ste="clipped")
    res = train(cfg, TrainConfig(steps=300, batch=32, lr=3e-3, log_every=100, verbose=False, accelerator="cpu"), src)
    hist = res["history"]
    assert hist[-1]["eval"] < hist[0]["eval"] - 0.3
    assert hist[-1]["eval"] < math.log(8) - 0.3
