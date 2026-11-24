import math

import pytest
import torch
from torch import nn

from lilbits.data import RecallSource
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.layers.loop_scaled import LoopScaledLinear, convert_loop_linears, set_loop
from lilbits.mixers.loop_memory import LoopMemory
from lilbits.mixers.sparse_index import (
    IndexedRetrieval,
    TokenIndex,
    dequant4,
    pack_bits,
    quant4,
    sign_agreement,
)
from lilbits.models.looped_index import LoopedIndexConfig, LoopedIndexLM, convert_core_to_littlebit, memory_tiers


def test_loop_scaled_starts_as_base_and_loops_differ():
    torch.manual_seed(0)
    x = torch.randn(3, 24)
    fp = nn.Linear(24, 16)
    m = LoopScaledLinear(fp, 3)
    torch.testing.assert_close(m(x), fp(x))
    lb = LittleBitLinear(24, 16, rank=6, residual=True, ste="clipped")
    m = LoopScaledLinear(lb, 3)
    torch.testing.assert_close(m(x), lb.forward_reference(x), rtol=1e-4, atol=1e-5)
    assert m.l is not None
    with torch.no_grad():
        m.l[1].mul_(2.0)
    set_loop(m, 1)
    assert not torch.allclose(m(x), lb.forward_reference(x))
    assert m.storage_bits() < 16 * 24 * 16 + 16 * 3 * (24 + 16 + 6)


def test_convert_loop_linears_from_fp():
    torch.manual_seed(0)
    core = nn.Sequential(LoopScaledLinear(nn.Linear(32, 32, bias=False), 2))
    convert_loop_linears(core, lambda i, o: LittleBitLinear(i, o, rank=32, residual=True, ste="clipped"))
    m = core[0]
    assert isinstance(m, LoopScaledLinear) and isinstance(m.base, LittleBitLinear) and m.l is not None


def test_quant4_and_signatures():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    codes, s = quant4(x)
    assert codes.shape == (2, 5, 8) and codes.dtype == torch.uint8
    assert (dequant4(codes, s) - x).abs().max() <= x.abs().amax() / 7
    a, b = torch.randn(1, 3, 20), torch.randn(1, 4, 20)
    agree = sign_agreement(pack_bits(a), pack_bits(b))
    brute = ((a[0, :, None] >= 0) == (b[0, None] >= 0)).sum(-1) + 4  # 4 padding bits agree
    assert torch.equal(agree[0], brute.int())


def test_retrieval_with_everything_selected_is_dense_causal_attention():
    torch.manual_seed(0)
    r = IndexedRetrieval(16, 2, 8, top_blocks=10, top_k=10_000)
    idx = TokenIndex(block=4)
    e = torch.randn(1, 12, 16)
    for b in range(3):
        r.write(e[:, 4 * b : 4 * b + 4], idx)
        r.write_summaries(e[:, 4 * b : 4 * b + 4].mean(1), e[:, 4 * b : 4 * b + 4].mean(1), idx)
    x = torch.randn(1, 4, 16)
    out, _ = r(x, idx, pos0=8)
    q = r.q(x).view(1, 4, 2, 8)
    K, V = idx.keys(), idx.values()
    att = torch.einsum("blhd,bpd->bhlp", q, K) / math.sqrt(8)
    causal = torch.arange(12)[None, :] <= (8 + torch.arange(4))[:, None]
    att = att.masked_fill(~causal, float("-inf")).softmax(-1)
    ref = r.o(torch.einsum("bhlp,bpd->blhd", att, V).reshape(1, 4, 16))
    torch.testing.assert_close(out, ref)


def test_index_rollback():
    r = IndexedRetrieval(16, 2, 8)
    idx = TokenIndex(block=4)
    r.write(torch.randn(1, 3, 16), idx)
    snap = idx.snapshot()
    r.write(torch.randn(1, 2, 16), idx)
    idx.restore(snap)
    assert idx.length == 3 and idx.keys().shape[1] == 3


def test_loop_memory_starts_as_noop():
    m = LoopMemory(16)
    x = torch.randn(2, 3, 16)
    assert m(x, [torch.randn_like(x)]).abs().max() == 0


def small(**kw):
    torch.manual_seed(0)
    return LoopedIndexLM(LoopedIndexConfig(vocab_size=20, d_model=32, n_heads=2, block=8, top_k=4, **kw))


def test_chunked_equals_token_by_token():
    m = small().eval()
    x = torch.randint(0, 20, (1, 21))
    with torch.no_grad():
        full = m.process(x, m.new_cache(1))
        c = m.new_cache(1)
        steps = torch.cat([m.process(x[:, t : t + 1], c) for t in range(21)], 1)
    torch.testing.assert_close(full, steps, rtol=1e-3, atol=1e-3)  # BLAS batching order only


@pytest.mark.parametrize("core", ["fp", "littlebit"])
def test_speculative_equals_greedy(core):
    m = small()
    if core == "littlebit":
        convert_core_to_littlebit(m)
    m.eval()
    p = torch.randint(0, 20, (1, 11))
    g1, _ = m.generate(p, 14)
    g2, st = m.generate_speculative(p, 14)
    assert torch.equal(g1, g2)
    assert st["passes"] >= 1 and 0 <= st["acceptance"] <= 1


def test_rollback_restores_cache():
    m = small().eval()
    x = torch.randint(0, 20, (1, 14))
    with torch.no_grad():
        c = m.new_cache(1)
        m.process(x[:, :6], c)
        snap = c.snapshot()
        a = m.process(x[:, 6:10], c)
        c.restore(snap)
        b = m.process(x[:, 6:10], c)
    torch.testing.assert_close(a, b)


def test_halting_uses_fewer_loops_and_keeps_states():
    m = small(n_loops=4).eval()
    p = torch.randint(0, 20, (1, 9))
    _, st = m.generate(p, 6, halt_eps=10.0)  # always halts after loop 2
    assert st["mean_loops"] == 2
    _, st_full = m.generate(p, 6)
    assert st_full["mean_loops"] == 4


def test_training_all_stages_backward():
    m = small()
    x = torch.randint(0, 20, (2, 24))
    for stage in range(1, 5):
        if stage == 2:
            convert_core_to_littlebit(m)
        m.use_index, m.distill = stage >= 3, stage >= 3
        loss, info = m.training_loss(x, mtp=stage >= 4)
        loss.backward()
        assert math.isfinite(loss.item())
        assert ("index_distill" in info) == (stage >= 3) and ("mtp1" in info) == (stage >= 4)
        m.zero_grad()
    m.eval()
    _, st = m.generate(torch.randint(0, 20, (1, 17)), 4)
    tiers = memory_tiers(m, st["cache"], 20)
    assert tiers["kv_signature_bytes_per_token"] > 0 and tiers["summary_bytes"] > 0


def test_recall_source():
    s = RecallSource(n_symbols=12, n_pairs=3, length=40, seed=0)
    x, kind = s.sample(16)
    assert x.shape == (16, 40) and x.max() < s.vocab
    for row, k in zip(x.tolist(), kind.tolist(), strict=True):
        assert row[-3] == (s.q2 if k else s.q1)
        pairs = {row[i]: row[i + 1] for i in range(len(row) - 3) if row[i] < s.n_symbols}
        ans = pairs[pairs[row[-2]]] if k else pairs[row[-2]]
        assert ans == row[-1]
