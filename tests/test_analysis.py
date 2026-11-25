import torch
from torch import nn

from lilbits.analysis import (
    fewer_loops,
    histogram,
    littlebit_health,
    loop_convergence,
    patch_report,
    profile_parts,
    router_entropy,
    sign_flips,
)
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.models.pocket import PocketConfig, PocketLM


def small():
    torch.manual_seed(0)
    return PocketLM(PocketConfig(d_byte=32, d_body=64, block=4, top_k=4)).eval()


def text(n):
    return torch.tensor(list((b"to be or not to be that is the question " * 8)[:n]))[None]


def test_profile_parts_covers_named_parts():
    m, x = small(), text(64)
    with torch.no_grad():
        out = profile_parts(lambda: m(x, m.new_cache(1)), m, {"core": "body.core"})
    assert out["core"]["calls"] >= 1 and 0 < out["core"]["share"] <= 1
    assert out["core"]["resident_mib"] > 0


def test_fewer_loops_restores_and_convergence_has_one_entry_per_loop():
    m, x = small(), text(64)
    n = m.body.cfg.n_loops
    with torch.no_grad():
        conv = loop_convergence(m.body, lambda: m(x, m.new_cache(1)))
        with fewer_loops(m.body, 1):
            assert m.body.cfg.n_loops == 1
    assert m.body.cfg.n_loops == n and len(conv) == n
    assert abs(conv[0] - 1.0) < 1e-4


def test_littlebit_health_and_sign_flips():
    a = nn.Sequential(LittleBitLinear(16, 16, rank=4))
    rows = littlebit_health(a)
    assert len(rows) == 1 and 0 <= rows[0]["near_zero_latent"] <= 1
    sa = {k: v.clone() for k, v in a.state_dict().items()}
    sb = {k: -v if k.endswith((".U", ".V")) else v for k, v in sa.items()}
    flips = sign_flips(sa, sb)
    assert flips and all(v > 0.99 for v in flips.values())


def test_patch_report_and_small_helpers():
    m = small()
    rep = patch_report(m, text(80))
    assert rep["bytes_per_patch"] >= 1 and "|" in rep["sample"]
    assert histogram([1, 1, 2]) == {1: 2, 2: 1}
    assert abs(router_entropy(torch.full((3, 4), 0.25)) - 1.0) < 1e-5
    assert router_entropy(torch.tensor([[1.0, 0, 0, 0]])) < 1e-3
