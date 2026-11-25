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
