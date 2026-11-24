import math
from typing import Any

import pytest
import torch

from lilbits.models.recurrent_depth import RecurrentDepthConfig, RecurrentDepthLM, sample_recurrence

TINY: dict[str, Any] = dict(
    vocab_size=16, seq_len=16, d_model=32, n_heads=2, mean_recurrence=4, max_recurrence=8, backprop_depth=2
)


@pytest.mark.parametrize("core_linear", [None, "littlebit", "bitnet"])
def test_shapes_and_causality(core_linear):
    torch.manual_seed(0)
    m = RecurrentDepthLM(RecurrentDepthConfig(**TINY, core_linear=core_linear)).eval()
    x = torch.randint(0, 16, (2, 12))
    torch.manual_seed(1)
    y = m(x, num_steps=3)
    assert y.shape == (2, 12, 16)
    x2 = x.clone()
    x2[:, 7:] = torch.randint(0, 16, (2, 5))
    torch.manual_seed(1)  # same random initial state
    torch.testing.assert_close(m(x2, num_steps=3)[:, :7], y[:, :7])


def test_truncated_backprop_still_trains_every_weight():
    torch.manual_seed(0)
    m = RecurrentDepthLM(RecurrentDepthConfig(**TINY))
    loss, info = m.training_loss(torch.randint(0, 16, (2, 12)))
    loss.backward()
    assert math.isfinite(loss.item()) and 1 <= info["r"] <= 8
    assert all(p.grad is not None for n, p in m.named_parameters())


def test_sampler_mean_and_cap():
    g = torch.Generator().manual_seed(0)
    rs = [sample_recurrence(8, 32, generator=g) for _ in range(2000)]
    assert max(rs) <= 32 and min(rs) >= 1
    assert 7 < sum(rs) / len(rs) < 10  # mean ~ mean_recurrence + 1


def test_adaptive_exit_freezes_converged_positions():
    torch.manual_seed(0)
    m = RecurrentDepthLM(RecurrentDepthConfig(**TINY)).eval()
    x = torch.randint(0, 16, (2, 10))
    logp, steps = m.adaptive_forward(x, max_steps=8, kl_threshold=1e9)  # everything converges after step 2
    assert (steps == 2).all() and torch.allclose(logp.exp().sum(-1), torch.ones(2, 10))
    _, steps = m.adaptive_forward(x, max_steps=5, kl_threshold=-1.0)  # nothing converges
    assert (steps == 5).all()


def test_ouro_exit_gate_objective_and_cdf_exit():
    torch.manual_seed(0)
    m = RecurrentDepthLM(RecurrentDepthConfig(**TINY, exit_gate=True))
    loss, _info = m.training_loss(torch.randint(0, 16, (2, 12)))
    loss.backward()
    assert m.gate is not None
    assert math.isfinite(loss.item()) and m.gate.weight.grad is not None
    _, steps = m.eval().adaptive_forward(torch.randint(0, 16, (2, 6)), max_steps=6, gate_q=0.0)
    assert (steps == 1).all()  # CDF >= 0 at the first step
