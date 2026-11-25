import pytest
import torch

from lilbits.inference.lut import pack_lut
from lilbits.models.pocket import PairRouter, PocketConfig, PocketLM, convert_pocket_core

TEXT = b"The pocket model reads bytes, opens a patch after spaces, and loops its core. " * 4


def small(**kw):
    torch.manual_seed(0)
    return PocketLM(PocketConfig(d_byte=32, d_body=64, block=4, top_k=4, **kw))


def text(n, off=0):
    return torch.tensor(list(TEXT[off : off + n]))[None]


@pytest.mark.parametrize("router", ["pair", "hnet", "none"])
def test_streaming_bytes_equals_one_chunk(router):
    m = small(router=router).eval()
    x = text(60)
    with torch.no_grad():
        c = m.new_cache(1)
        full, _ = m(x, c)
        c2 = m.new_cache(1)
        steps = torch.cat([m(x[:, t : t + 1], c2)[0] for t in range(60)], 1)
    assert c.patches == c2.patches > 4
    torch.testing.assert_close(full, steps, rtol=1e-3, atol=1e-3)


def test_pair_router_starts_as_space_patching_with_cap():
    m = small(max_patch=5).eval()
    x = torch.tensor(list(b"ab cdefghijkl m"))[None]
    with torch.no_grad():
        _, aux = m(x, m.new_cache(1))
    starts = aux["b"][0].nonzero().flatten().tolist()
    assert starts == [0, 3, 8, 13, 14]  # after spaces, and every 5 bytes inside a long word


def test_pair_router_is_exact_under_any_chunking():
    r = PairRouter()
    x = text(30)
    whole = r(x, None)
    parts = torch.cat([r(x[:, :7], None), r(x[:, 7:], x[:, 6:7])], 1)
    assert torch.equal(whole, parts)


def test_patches_are_causal():
    m = small().eval()
    x = text(40)
    y = x.clone()
    y[:, 25:] = (y[:, 25:] + 1) % 256
    with torch.no_grad():
        a, _ = m(x, m.new_cache(1))
        b, _ = m(y, m.new_cache(1))
    torch.testing.assert_close(a[:, :25], b[:, :25])


@pytest.mark.parametrize("router", ["pair", "none"])
def test_speculative_equals_greedy_and_rollback(router):
    m = small(router=router)
    convert_pocket_core(m)
    m.eval()
    for off in (0, 9, 31):
        p = text(12, off)
        g, _ = m.generate(p, 20)
        s, st = m.generate_speculative(p, 20)
        assert torch.equal(g, s) and st["passes"] >= 1
    with torch.no_grad():
        c = m.new_cache(1)
        m(text(12), c)
        snap = c.snapshot()
        a, _ = m(text(5, 12), c)
        c.restore(snap)
        b, _ = m(text(5, 12), c)
    torch.testing.assert_close(a, b)


def test_training_backward_all_switches():
    for router in ("pair", "hnet", "none"):
        m = small(router=router)
        loss, info = m.training_loss(torch.randint(0, 256, (2, 40)))
        loss.backward()
        assert torch.isfinite(loss) and "mtp1" in info
    assert m.mtp[0].proj.weight.grad is not None


def test_littlebit_int8_core_and_lut_decode_match_native():
    m = small()
    convert_pocket_core(m)
    m.eval()
    bits = {mod.base.scale_bits for mod in m.body.core.modules() if hasattr(mod, "n_loops") and hasattr(mod, "base")}
    assert bits == {8}
    p = text(10)
    g, _ = m.generate(p, 12)
    pack_lut(m.body)
    g2, _ = m.generate(p, 12)
    assert torch.equal(g, g2)
