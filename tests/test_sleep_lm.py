import torch

from lilbits.data import FewShotMapSource
from lilbits.models.sleep_lm import SleepConfig, SleepLM


def small(loops=2, **kw):
    torch.manual_seed(0)
    return SleepLM(SleepConfig(d_model=32, n_heads=4, n_kv_heads=2, n_layers=2, window=4, loops=loops, **kw))


def test_streaming_decode_matches_full_forward():
    for loops in (1, 3):
        m = small(loops).eval()
        x = torch.randint(0, 256, (2, 13))
        full = m(x)
        st = m.init_state(2)
        parts = [m.step(x[:, :5], st)] + [m.step(x[:, t : t + 1], st) for t in range(5, 13)]
        assert torch.allclose(full, torch.cat(parts, 1), atol=1e-5)
        assert st.sleeps == (3 if loops > 1 else 0) and st.pos == 13


def test_sleep_changes_later_windows_only_and_trains_gates():
    m1, m2 = small(1).eval(), small(2).eval()
    m2.load_state_dict(m1.state_dict(), strict=False)  # same weights, plus gates
    x = torch.randint(0, 256, (1, 12))
    a, b = m1(x), m2(x)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)  # first window: nothing slept on yet
    assert not torch.allclose(a[:, 4:], b[:, 4:], atol=1e-4)
    m2.train()
    loss, _ = m2.training_loss(torch.randint(0, 256, (2, 13)))
    loss.backward()
    blk = m2.blocks[0]
    assert blk.gate_k.weight.grad is not None and blk.gate_k.weight.grad.abs().sum() > 0
    assert m1.sleep_params() == 0 and 0 < m2.sleep_params()


def test_kv_cache_size_does_not_grow_with_loops_and_generate():
    x = torch.randint(0, 256, (1, 6))
    sizes = []
    for loops in (1, 4):
        out, st = small(loops).eval().generate(x, 7)
        assert out.shape == (1, 13) and torch.equal(out[:, :6], x)
        sizes.append(st.kv_bytes())
    assert sizes[0] == sizes[1]


def test_few_shot_map_source():
    toks, shots = FewShotMapSource(n_keys=4, seed=0).sample(3, 63)
    assert toks.shape == shots.shape == (3, 64)
    for b in range(3):
        m = {}
        for t in range(1, 64, 2):
            k, v = toks[b, t - 1].item(), toks[b, t].item()
            assert shots[b, t].item() == sum(1 for j in range(1, t, 2) if toks[b, j - 1] == k)
            assert m.setdefault(k, v) == v
