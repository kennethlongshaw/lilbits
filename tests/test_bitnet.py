import math

import torch

from lilbits.layers.bitnet import BitLinear, act_quant, ternary
from lilbits.models.tinylm import Config, TinyLM, compression_report


def test_ternary_values_and_scale():
    w = torch.randn(64, 32)
    q = ternary(w)
    s = w.abs().mean()
    assert set(torch.unique(torch.round(q / s)).tolist()) <= {-1.0, 0.0, 1.0}


def test_act_quant_levels():
    x = torch.randn(4, 16)
    q = act_quant(x, 8)
    step = x.abs().amax(-1, keepdim=True) / 127
    assert torch.allclose(q / step, (q / step).round(), atol=1e-4)
    assert (q - x).abs().max() <= step.max() / 2 + 1e-6


def test_ste_gradients_reach_latent_weight():
    m = BitLinear(16, 8)
    x = torch.randn(3, 16, requires_grad=True)
    m(x).sum().backward()
    assert m.weight.grad is not None and m.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_storage_bits():
    m = BitLinear(128, 64)
    assert math.isclose(m.storage_bits(), math.log2(3) * 128 * 64 + 16)


def test_tinylm_bitnet_trains_one_step():
    torch.manual_seed(0)
    model = TinyLM(Config(linear="bitnet"))
    rep = compression_report(model)
    assert 1.58 < rep["bpw"] < 1.59
    idx = torch.randint(0, 32, (2, 16))
    loss = torch.nn.functional.cross_entropy(model(idx).flatten(0, 1), idx.flatten())
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_eval_cache_and_packed_match_training_forward():
    from lilbits.layers.bitnet import PackedBitLinear

    torch.manual_seed(0)
    m = BitLinear(37, 11)
    x = torch.randn(5, 37)
    ref = m(x).detach()
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(m(x), ref)
        p = PackedBitLinear(m)
        torch.testing.assert_close(p(x), ref, atol=2e-3, rtol=2e-3)  # FP16 scale
    assert p.codes.numel() == -(-37 * 11 // 4)
    m.train()
    assert m._infer_w is None
