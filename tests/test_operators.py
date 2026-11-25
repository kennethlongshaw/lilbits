import torch

from lilbits.mixers.operators import ALL_OPS, OperatorLoop, gumbel_probs
from lilbits.models.op_loop import OpLoopLM


def test_attend_only_loop_is_causal_attention():
    torch.manual_seed(0)
    loop = OperatorLoop(16, 2, ops=("attend",)).eval()
    h, ctx = torch.randn(1, 5, 16), torch.randn(1, 5, 16)
    regs = loop.new_registers(1, 5, ctx)
    out, _, probs = loop(h, ctx, regs, 0, hard=True)
    vin = torch.stack([h, *regs[1]], -2).mul(probs["vec_in"][..., None]).sum(-2)  # the routed query input
    q = loop.q(vin)
    att = (q @ loop.k_proj(ctx).transpose(1, 2) / 4).masked_fill(~torch.ones(5, 5, dtype=torch.bool).tril(), -1e9)
    vec = loop.o(att.softmax(-1) @ loop.v_proj(ctx))
    h2 = h + torch.sigmoid(loop.gate(h)) * loop.write(vec)
    torch.testing.assert_close(out, h2 + loop.mlp(h2))
    assert probs["op"].sum(-1).eq(1).all()


def test_ops_are_causal_and_hard_routing_is_one_hot():
    torch.manual_seed(0)
    m = OpLoopLM(32, 3)
    m.hard = True
    x = torch.randint(0, 256, (2, 12))
    y = x.clone()
    y[:, 8:] = (y[:, 8:] + 1) % 256
    a, ta = m(x)
    b, _ = m(y)
    torch.testing.assert_close(a[:, :8], b[:, :8])
    assert all(((p == 0) | (p == 1)).all() for p in ta)


def test_soft_training_backward_and_load_balance():
    torch.manual_seed(0)
    m = OpLoopLM(32, 3)
    x = torch.randint(0, 256, (2, 13))
    loss, info = m.loss(x, torch.ones(2, 12), generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert m.loop.router.weight.grad is not None and info["load_balance"] >= 1.0 - 1e-6
    uniform = torch.full((4, len(ALL_OPS)), 1 / len(ALL_OPS))
    assert abs(m.loop.load_balance(uniform).item() - 1.0) < 1e-6
    p = gumbel_probs(torch.randn(3, 4), 0.5, False)
    torch.testing.assert_close(p.sum(-1), torch.ones(3))


def test_count_op_counts_set_positions():
    loop = OperatorLoop(8, 1, ops=("count",))
    S = torch.zeros(1, 4, 4)
    S[0, 3, :3] = 1.0
    with torch.no_grad():
        loop.count_u.fill_(1.0)
    z = torch.zeros(1, 4, 8)
    _, (_, vecs), _ = loop(z, z, ([S, S], [z, z]), 0, hard=True)
    assert vecs[0][0, 3, 0].item() == 3.0 or vecs[1][0, 3, 0].item() == 3.0


def test_router_teacher_forcing_pulls_loop_toward_expected_ops():
    from lilbits.models.op_loop import OpLoopLM
    from lilbits.shakespeare import segment_tensors

    torch.manual_seed(0)
    m = OpLoopLM(32, 2)
    t = m.op_targets([({"filter"}, {"count"}), None])
    assert t.shape == (2, 2, len(m.ops)) and t[0, 0, m.ops.index("filter")] and not t[1].any()
    x, mk = segment_tensors([[(b"COUNT A: A, B =>", False), (b" 1\n", True)]] * 2, 24)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = None
    for _ in range(15):
        loss, info = m.loss(x, mk, op_targets=t, router_weight=5.0)
        first = first if first is not None else info["router_ce"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert info["router_ce"] < 0.5 * first


def test_op_loop_decode_cache_matches_full_forward():
    from lilbits.models.op_loop import OpLoopLM

    torch.manual_seed(0)
    m = OpLoopLM(32, 3).eval()
    m.hard = True
    x = torch.randint(0, 256, (2, 20))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        parts = [m(x[:, :12], cache=cache)[0]] + [m(x[:, t : t + 1], cache=cache)[0] for t in range(12, 20)]
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-4)


def test_straight_through_routing_trains_and_skips_unchosen_ops():
    from lilbits.models.op_loop import OpLoopLM

    torch.manual_seed(0)
    m = OpLoopLM(32, 2)
    m.st, m.tau = True, 0.5
    x = torch.randint(0, 256, (2, 16))
    loss, _ = m.loss(x, torch.ones(2, 15))
    loss.backward()
    assert m.loop.router.weight.grad is not None and m.loop.router.weight.grad.abs().sum() > 0
    _, trace = m(x)
    assert all(((p == 0) | (p == 1)).all() for p in trace)  # one-hot in the forward pass


def test_word_pool_means_over_words_and_marks_word_ends():
    from lilbits.mixers.operators import word_pool

    x = torch.tensor([list(b"AB, CDE X")])
    ctx = torch.arange(9, dtype=torch.float).view(1, 9, 1).expand(1, 9, 2).contiguous()
    wctx, wend = word_pool(ctx, x)
    assert wend[0].tolist() == [False, True, False, False, False, False, True, False, False]
    assert wctx[0, :, 0].tolist() == [0, 0.5, 2, 3, 4, 4.5, 5, 7, 8]


def test_word_keys_stay_causal_and_cached_decode_matches():
    from lilbits.models.op_loop import OpLoopLM

    torch.manual_seed(0)
    m = OpLoopLM(32, 3, word_keys=True).eval()
    m.hard = True
    x = torch.tensor([list(b"COUNT AB: AB, CD, AB => 2")])
    with torch.no_grad():
        full, _ = m(x)
        changed = x.clone()
        changed[0, -3:] = torch.tensor(list(b"XYZ"))
        assert torch.allclose(m(changed)[0][:, :-3], full[:, :-3], atol=1e-5)  # later bytes never leak back
        cache = m.new_cache()
        parts = [m(x[:, :10], cache=cache)[0]] + [m(x[:, t : t + 1], cache=cache)[0] for t in range(10, x.shape[1])]
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-4)
