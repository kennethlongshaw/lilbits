import math

import pytest
import torch

from lilbits.data import MarkovSource
from lilbits.mixers.recurrent import MIXERS, gated_linear_rule, mingru_rule, mlstm_rule
from lilbits.models.tinylm import Config, TinyLM, TransformerBlock, make_linear_factory
from lilbits.training.train import TrainConfig, make_optimizers, train

NEW = ["mamba2", "gla", "mlstm", "mingru"]


@pytest.mark.parametrize("mixer", [*NEW, "gdn"])
@pytest.mark.parametrize("linear", ["fp", "littlebit", "bitnet"])
def test_block_shape_and_causality(mixer, linear):
    torch.manual_seed(0)
    cfg = Config(vocab_size=16, seq_len=16, d_model=32, n_heads=2, n_layers=1, linear=linear, mixer=mixer)
    block = TransformerBlock(32, 2, 64, make_linear_factory(cfg), mixer=mixer)
    x = torch.randn(2, 10, 32)
    y = block(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 32)
    torch.testing.assert_close(block(x2)[:, :6], y[:, :6])


@pytest.mark.parametrize("mixer", [*NEW, "gdn"])
def test_module_returns_state_and_gradients_flow(mixer):
    torch.manual_seed(0)
    m = MIXERS[mixer](32, 2, make_linear_factory(Config(linear="fp")))
    x = torch.randn(2, 7, 32, requires_grad=True)
    y, state = m(x, return_state=True)
    assert y.shape == x.shape and state is not None
    y.square().mean().backward()
    assert all(p.grad is not None for p in m.parameters())


@pytest.mark.parametrize("per_channel", [False, True])
def test_gated_linear_rule_matches_closed_form_and_chunks(per_channel):
    torch.manual_seed(0)
    B, T, H, D = 2, 9, 2, 3
    q, k, v = torch.randn(B, T, H, D), torch.randn(B, T, H, D), torch.randn(B, T, H, D)
    ld = -torch.rand(B, T, H, D) if per_channel else -torch.rand(B, T, H)
    o, S = gated_linear_rule(q, k, v, ld)
    # closed form: o_t = sum_{s<=t} (q_t * prod_{s<r<=t} a_r) . k_s  v_s
    cum = ld.cumsum(1)
    ref = torch.zeros_like(o)
    for t in range(T):
        for s in range(t + 1):
            w = (cum[:, t] - cum[:, s]).exp()
            w = w if per_channel else w[..., None]
            ref[:, t] += ((q[:, t] * w * k[:, s]).sum(-1, keepdim=True)) * v[:, s]
    torch.testing.assert_close(o, ref, atol=1e-5, rtol=1e-4)
    a, Sa = gated_linear_rule(q[:, :4], k[:, :4], v[:, :4], ld[:, :4])
    b, Sb = gated_linear_rule(q[:, 4:], k[:, 4:], v[:, 4:], ld[:, 4:], initial_state=Sa)
    torch.testing.assert_close(torch.cat([a, b], 1), o)
    torch.testing.assert_close(Sb, S)


def test_mlstm_stabilized_matches_naive_and_chunks():
    torch.manual_seed(0)
    B, T, H, D = 2, 8, 2, 4
    q, k, v = torch.randn(B, T, H, D), torch.randn(B, T, H, D) / 2, torch.randn(B, T, H, D)
    i_pre, f_pre = torch.randn(B, T, H), torch.randn(B, T, H) + 2
    h, _st = mlstm_rule(q, k, v, i_pre, f_pre)
    C = torch.zeros(B, H, D, D)
    n = torch.zeros(B, H, D)
    for t in range(T):  # unstabilized reference (fine for these small gate values)
        f, i = torch.sigmoid(f_pre[:, t]), i_pre[:, t].exp()
        C = f[..., None, None] * C + i[..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        n = f[..., None] * n + i[..., None] * k[:, t]
        ref = torch.einsum("bhkv,bhk->bhv", C, q[:, t]) / (n * q[:, t]).sum(-1).abs().clamp_min(1.0)[..., None]
        torch.testing.assert_close(h[:, t], ref, atol=1e-4, rtol=1e-4)
    a, sa = mlstm_rule(q[:, :3], k[:, :3], v[:, :3], i_pre[:, :3], f_pre[:, :3])
    b, _sb = mlstm_rule(q[:, 3:], k[:, 3:], v[:, 3:], i_pre[:, 3:], f_pre[:, 3:], initial_state=sa)
    torch.testing.assert_close(torch.cat([a, b], 1), h)


def test_mingru_is_a_lerp_and_chunks():
    torch.manual_seed(0)
    z, ht = torch.rand(2, 6, 5), torch.randn(2, 6, 5)
    h, _S = mingru_rule(z, ht)
    ref, prev = [], torch.zeros(2, 5)
    for t in range(6):
        prev = (1 - z[:, t]) * prev + z[:, t] * ht[:, t]
        ref.append(prev)
    torch.testing.assert_close(h, torch.stack(ref, 1))
    a, Sa = mingru_rule(z[:, :2], ht[:, :2])
    b, _ = mingru_rule(z[:, 2:], ht[:, 2:], initial_state=Sa)
    torch.testing.assert_close(torch.cat([a, b], 1), h)


def test_muon_only_gets_2d_matrices():
    for mixer in NEW:
        model = TinyLM(Config(vocab_size=16, seq_len=16, d_model=64, n_heads=2, n_layers=1, mixer=mixer))
        muon, _ = make_optimizers(model, TrainConfig(optim="muon"))
        assert all(p.dim() == 2 for g in muon.param_groups for p in g["params"])


@pytest.mark.parametrize("mixer", NEW)
def test_training_learns(mixer):
    src = MarkovSource(vocab=8, order=1, branching=2, seed=0)
    cfg = Config(vocab_size=8, seq_len=16, d_model=32, n_heads=2, n_layers=1, mixer=mixer)
    res = train(cfg, TrainConfig(steps=200, batch=32, lr=3e-3, log_every=100, verbose=False, accelerator="cpu"), src)
    hist = res["history"]
    assert hist[-1]["eval"] < math.log(8) - 0.3


@pytest.mark.parametrize("T,chunk", [(5, 64), (64, 16), (77, 32)])
def test_chunked_gated_delta_rule_matches_naive(T, chunk):
    import torch.nn.functional as F

    from lilbits.mixers.recurrent import chunk_gated_delta_rule, gated_delta_rule

    torch.manual_seed(0)
    B, H, D = 2, 2, 8
    q = (torch.randn(B, T, H, D) * D**-0.5).requires_grad_()
    k = F.normalize(torch.randn(B, T, H, D), dim=-1).requires_grad_()
    v = torch.randn(B, T, H, D).requires_grad_()
    g = (-torch.rand(B, T, H) * 0.5).requires_grad_()
    beta = torch.rand(B, T, H).requires_grad_()
    S0 = torch.randn(B, H, D, D)
    o1, S1 = gated_delta_rule(q, k, v, g, beta, S0)
    grads1 = torch.autograd.grad((o1.square().sum() + S1.sum()), (q, k, v, g, beta))
    o2, S2 = chunk_gated_delta_rule(q, k, v, g, beta, S0, chunk=chunk)
    grads2 = torch.autograd.grad((o2.square().sum() + S2.sum()), (q, k, v, g, beta))
    torch.testing.assert_close(o2, o1, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(S2, S1, rtol=1e-4, atol=1e-4)
    for a, b in zip(grads1, grads2, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-3, atol=1e-3)
