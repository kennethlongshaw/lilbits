import torch

from lilbits.models.energy import EnergyConfig, EnergyLM


def small():
    torch.manual_seed(0)
    return EnergyLM(EnergyConfig(d=32, heads=2, n_layers=1, d_pred=16, d_hidden=32, steps=3))


def test_training_backprops_through_thinking_steps():
    m = small()
    x = torch.randint(0, 256, (2, 17))
    loss, info = m.training_loss(x, generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert m.log_alpha.grad is not None and m.energy.w1.weight.grad is not None
    assert any(p.grad is not None for p in m.blocks.parameters())
    assert info["steps"] >= 2


def test_thinking_lowers_energy_at_inference():
    m = small().eval()
    h, _ = m.context(torch.randint(0, 256, (1, 9)))
    _, es = m.think(h, 8, generator=torch.Generator().manual_seed(0))
    assert (es[-1] <= es[0] + 1e-6).float().mean() > 0.9


def test_self_verification_picks_lowest_energy_and_generate_is_streaming():
    m = small().eval()
    h, _ = m.context(torch.randint(0, 256, (1, 5)))
    y1 = m.predict(h, 4, samples=1)
    y4 = m.predict(h, 4, samples=4)
    e1, e4 = m.energy(h, y1), m.energy(h, y4)
    assert (e4 <= e1 + 1e-6).all()
    p = torch.randint(0, 256, (1, 6))
    out = m.generate(p, 5, steps=2)
    assert out.shape == (1, 11) and torch.equal(out[:, :6], p)
