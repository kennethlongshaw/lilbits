import torch
from torch import nn

from lilbits.inference.lut import pack_lut
from lilbits.inference.packed import pack_model
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.models.lean_lm import LeanLM, convert_linears


def fp(i, o):
    return nn.Linear(i, o, bias=False)


def make(seed=0, d=32, drafts=3):
    torch.manual_seed(seed)
    return LeanLM(d, 4, 2, fp, n_draft=drafts, draft_rank=8).eval()


def test_loss_includes_mtp():
    m = make()
    x = torch.randint(0, 256, (2, 17))
    loss, info = m.loss(x, 0.3)
    assert torch.isfinite(loss) and "mtp_ce" in info
    assert abs(loss.item() - info["ce"] - 0.3 * info["mtp_ce"]) < 1e-4
    loss.backward()
    assert m.drafts[0][1].weight.grad is not None


def test_drafts_start_as_next_byte_head():
    m = make()
    h, _ = m.hidden(torch.randint(0, 256, (1, 5)))
    for lg in m.draft_logits(h):
        assert torch.allclose(lg, m.head(h))


def test_speculative_equals_greedy_random_model():
    m = make(1)
    p = torch.randint(0, 256, (1, 20))
    spec, stats = m.speculative_generate(p, 24)
    assert torch.equal(spec, m.generate(p, 24))
    assert stats["model_calls"] >= stats["rounds"]


def test_speculative_full_acceptance_path():
    # blocks reduced to the identity: greedy repeats a fixed byte, so every draft is right
    m = make(2)
    for blk in m.blocks:
        nn.init.zeros_(blk.mixer.o.weight)
        nn.init.zeros_(blk.mlp.down.weight)
    p = torch.randint(0, 256, (1, 10))
    spec, stats = m.speculative_generate(p, 20)
    assert torch.equal(spec, m.generate(p, 20))
    assert stats["draft_acceptance"] == 1.0 and stats["bytes_per_call"] > 3


def test_convert_and_pack_keep_greedy_path():
    m = make(3, d=64)
    n = convert_linears(m, bpw=0.55, itq_iters=5)
    lbs = [x for x in m.modules() if isinstance(x, LittleBitLinear)]
    assert n == len(lbs) == 2 * (5 + 3)  # per block: q k v gate o + SwiGLU gate up down; GDN a/b gates stay FP
    x = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        ref = m(x)[0]
        pack_lut(pack_model(m))
        out = m(x)[0]
    # packed layers store FP16 scales, so logits match to FP16 rounding, not bit for bit
    assert (ref - out).abs().max() < 0.02 * ref.abs().max()
    spec, _ = m.speculative_generate(x, 8)
    assert torch.equal(spec, m.generate(x, 8))


def test_kept_states_match_replay():
    m = make(4)
    p = torch.randint(0, 256, (1, 15))
    a, sa = m.speculative_generate(p, 30)
    b, sb = m.speculative_generate(p, 30, replay=True)
    assert torch.equal(a, b) and torch.equal(a, m.generate(p, 30))
    assert sa["model_calls"] == sa["rounds"] <= sb["model_calls"]


def test_gdn_all_states_last_equals_final():
    from lilbits.mixers.recurrent import GatedDeltaNet

    torch.manual_seed(0)
    g = GatedDeltaNet(32, 4, fp, conv_size=1)
    x = torch.randn(2, 6, 32)
    o1, s1 = g(x, return_state=True)
    o2, s2 = g(x, return_state=True, all_states=True)
    assert s2.shape[1] == 6 and torch.allclose(s2[:, -1], s1, atol=1e-5) and torch.allclose(o1, o2, atol=1e-5)


def test_grad_ckpt_same_gradients():
    m = make(5).train()
    x = torch.randint(0, 256, (2, 20))
    m.loss(x)[0].backward()
    ref = [p.grad.clone() for p in m.parameters() if p.grad is not None]
    m.zero_grad()
    m.grad_ckpt = True
    m.loss(x)[0].backward()
    got = [p.grad for p in m.parameters() if p.grad is not None]
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ref, got, strict=True))
