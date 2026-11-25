import itertools

import torch

from lilbits.layers.bpw_alloc import _hull, allocate_ranks, greedy_allocate, input_energy, layer_bits
from lilbits.models.looped_index import LoopedIndexConfig, LoopedIndexLM, allocate_core_ranks, convert_core_to_littlebit


def test_greedy_respects_budget_and_matches_brute_force_on_convex_curves():
    curves = {
        "a": [(1, 10.0, 100.0), (2, 20.0, 40.0), (3, 30.0, 20.0), (4, 40.0, 15.0)],
        "b": [(1, 10.0, 50.0), (2, 20.0, 45.0), (3, 30.0, 41.0), (4, 40.0, 38.0)],
    }
    for budget in (20.0, 35.0, 50.0, 80.0):
        pick = greedy_allocate(curves, budget)
        bits = sum(curves[k][pick[k] - 1][1] for k in curves)
        err = sum(curves[k][pick[k] - 1][2] for k in curves)
        assert bits <= budget
        best = min(
            sum(c[2] for c in combo)
            for combo in itertools.product(*curves.values())
            if sum(c[1] for c in combo) <= budget
        )
        assert err == best


def test_hull_drops_points_that_do_not_help():
    h = _hull([(1, 1.0, 10.0), (2, 2.0, 9.9), (3, 3.0, 2.0), (4, 4.0, 3.0)])
    assert [r for r, _, _ in h] == [1, 3]


def test_allocation_keeps_total_bits_and_lowers_proxy_error():
    torch.manual_seed(0)
    flat = torch.randn(128, 128) / 128**0.5
    U, _ = torch.linalg.qr(torch.randn(128, 128))
    steep = (U * torch.arange(1, 129).float() ** -1.5) @ U.t()
    W = {"flat": flat, "steep": steep / steep.std() / 128**0.5}
    ranks, rep = allocate_ranks(W, 1.0)
    assert rep["allocated_bpw"] <= rep["uniform_bpw"] + 1e-9
    assert rep["allocated_error"] <= rep["uniform_error"]
    assert ranks["flat"] > ranks["steep"]  # the flat spectrum needs the rank
    total = sum(layer_bits(128, 128, r) for r in ranks.values())
    assert total / (2 * 128 * 128) == rep["allocated_bpw"]


def test_input_energy_hook():
    lin = torch.nn.Linear(4, 2)
    with input_energy({"l": lin}) as e:
        lin(torch.tensor([[1.0, 2.0, 0.0, -1.0], [3.0, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(e["l"], torch.tensor([5.0, 2.0, 0.0, 1.0]))
    assert not lin._forward_hooks


def test_core_conversion_with_allocated_ranks():
    torch.manual_seed(0)
    m = LoopedIndexLM(LoopedIndexConfig(vocab_size=20, d_model=128, n_heads=2, block=8, top_k=4)).eval()
    x = torch.randint(0, 20, (2, 16))
    ranks, _ = allocate_core_ranks(m, x, bpw=1.0)
    convert_core_to_littlebit(m, 1.0, ranks=ranks)
    got = {n: mod.base.rank for n, mod in m.core.named_modules() if hasattr(mod, "n_loops") and hasattr(mod, "base")}
    assert got == ranks
    with torch.no_grad():
        assert torch.isfinite(m.process(x, m.new_cache(2))).all()
