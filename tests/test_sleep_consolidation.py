import torch

from lilbits.models.sleep_consolidation import (
    ConsolidatingLM,
    ConsolidationConfig,
    SleepSettings,
    imitation_reward,
    levenshtein,
    sleep,
    wake,
)


def small():
    torch.manual_seed(0)
    return ConsolidatingLM(ConsolidationConfig(seq_len=48, d_model=32, n_heads=2, n_layers=2, max_experts=3))


def test_wake_writes_fast_memory_and_sleep_moves_it_into_a_new_expert_only():
    m = small()
    x = torch.randint(0, 256, (2, 20))
    before = m(x).detach()
    base = [p.detach().clone() for p in m.base_params()]
    for p in m.base_params():
        p.requires_grad_(False)
    wake(m, lambda i: (x, None), 3)
    assert not torch.allclose(m(x), before)  # fast memory on: the output changed
    m.fast_on = False
    assert torch.allclose(m(x), before, atol=1e-6)  # fast memory off: unchanged base
    s = SleepSettings(consolidate_steps=2, dream_steps=2, batch=2, gen_len=4, n_dreams=4, top_k=2, n_random=1)
    info = sleep(m, lambda i: x[:, :5], s, generator=torch.Generator().manual_seed(0))
    assert int(m.n_active) == 1 and info["expert"] == 0 and info["dreams_kept"] == 3
    assert all(torch.equal(a, b) for a, b in zip(base, m.base_params(), strict=True))
    assert all(b.fast.B.abs().sum() == 0 for b in m.blocks)  # synaptic pruning reset the fast memory
    assert any(b.experts[0].B.abs().sum() > 0 for b in m.blocks)
    assert all(b.experts[1].B.abs().sum() == 0 for b in m.blocks)  # still locked
    info = sleep(m, lambda i: x[:, :5], s, generator=torch.Generator().manual_seed(1))
    assert int(m.n_active) == 2 and info["expert"] == 1


def test_fresh_expert_is_a_no_op():
    m = small().eval()
    x = torch.randint(0, 256, (1, 10))
    a = m(x)
    m.extra = [2]
    assert torch.allclose(a, m(x))


def test_imitation_reward():
    assert levenshtein([1, 2, 3], [1, 3]) == 1
    assert imitation_reward([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert imitation_reward([1, 2, 3, 4], [1, 2, 3, 5]) == 0.75
    assert imitation_reward([1, 1, 1, 1], [2, 2, 2, 2]) == 0.0  # beyond z0
