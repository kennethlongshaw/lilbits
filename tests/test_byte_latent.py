import pytest
import torch

from lilbits.models.byte_latent import (
    BLTConfig,
    BLTStream,
    ByteLatentTransformer,
    DynamicChunker,
    HashNgramEmbedding,
    HNet,
    HNetConfig,
    HNetStream,
    LMStream,
    PatchPooling,
    byte_entropy,
    entropy_boundaries,
    patch_ids,
    space_boundaries,
)
from lilbits.models.tinylm import Config, TinyLM

TEXT = torch.tensor([list("def foo(bar):\n    return bar + 1  # café, ok\nsome text here!".encode())])


def test_patch_ids_and_entropy_boundaries():
    b = torch.tensor([[False, False, True, False, True]])
    assert patch_ids(b).tolist() == [[0, 0, 1, 1, 2]]
    ent = torch.tensor([[5.0, 0.1, 0.2, 3.0, 0.1, 0.1, 0.1, 0.1]])
    assert entropy_boundaries(ent, theta_global=1.0)[0].tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    rel = entropy_boundaries(ent, theta_rel=0.05)[0].tolist()
    assert rel[2] and rel[3] and not rel[4]
    forced = entropy_boundaries(ent, theta_global=10.0, max_patch=3)[0].tolist()
    assert forced == [True, False, False, True, False, False, True, False]


def test_hash_embedding_depends_on_local_context_only():
    torch.manual_seed(0)
    emb = HashNgramEmbedding(16, ngrams=(3,), buckets=97)
    x = torch.randint(0, 256, (1, 10))
    x2 = x.clone()
    x2[0, 2] = (x[0, 2] + 1) % 256
    diff = (emb(x) - emb(x2)).abs().sum(-1)[0]
    assert diff[:2].sum() == 0 and diff[2:5].gt(0).all() and diff[5:].sum() == 0  # trigram window


def test_blt_is_causal_and_shapes():
    torch.manual_seed(0)
    m = ByteLatentTransformer(BLTConfig(seq_len=32, d_local=32, d_latent=48, heads=2, max_patches=32)).eval()
    x = torch.randint(0, 256, (2, 24))
    b = torch.zeros(2, 24, dtype=torch.bool)
    b[:, ::4] = True
    b[1, 5] = True
    y = m(x, b)
    assert y.shape == (2, 24, 256)
    x2 = x.clone()
    x2[:, 10:] = torch.randint(0, 256, (2, 14))
    torch.testing.assert_close(m(x2, b)[:, :10], y[:, :10])


def test_chunker_ratio_loss_minimum_and_up_down_shapes():
    p = torch.full((1, 400), 0.25)
    b = torch.zeros(1, 400, dtype=torch.bool)
    b[:, ::4] = True
    at_target = DynamicChunker.ratio_loss(p, b, 4.0)
    assert torch.isclose(at_target, torch.tensor(1.0))
    b2 = torch.ones_like(b)
    assert DynamicChunker.ratio_loss(torch.ones_like(p), b2, 4.0) > at_target
    x = torch.randn(1, 400, 8)
    chunks, cid, n = DynamicChunker.downsample(x, b)
    assert n == 100 and torch.equal(chunks[0, 1], x[0, 4])
    up = DynamicChunker.upsample(chunks, p, b, cid)
    assert up.shape == x.shape


def test_hnet_is_causal_and_trains():
    torch.manual_seed(0)
    m = HNet(HNetConfig(seq_len=32, d_outer=32, d_main=48, heads=2))
    x = torch.randint(0, 256, (2, 24))
    m.eval()
    y, _, stats = m(x)
    assert y.shape == (2, 24, 256) and stats["chunks"] >= 1
    x2 = x.clone()
    x2[:, 10:] = torch.randint(0, 256, (2, 14))
    torch.testing.assert_close(m(x2)[0][:, :10], y[:, :10])
    m.train()
    logits, ratio, _ = m(x)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()) + 0.03 * ratio
    loss.backward()
    assert m.chunker.q.weight.grad is not None and m.chunker.q.weight.grad.abs().sum() > 0


def test_space_boundaries_open_a_patch_after_a_space_run():
    x = torch.tensor([list(b"ab  cd.e\xc3\xa9f")])
    b = space_boundaries(x, max_patch=None)[0].tolist()
    #       a     b      ' '    ' '    c      d      .      e     \xc3   \xa9   f
    assert b == [True, False, False, True, False, False, False, True, False, True, False]
    capped = space_boundaries(torch.tensor([list(b"abcdefgh")]), max_patch=3)[0].tolist()
    assert capped == [True, False, False, True, False, False, True, False]


def test_segment_softmax_pooling_matches_dense_mask():
    torch.manual_seed(0)
    pool = PatchPooling(16, 32, 4)
    h = torch.randn(2, 20, 16)
    b = torch.rand(2, 20) < 0.3
    pid = patch_ids(b)
    P = int(pid.max()) + 1
    got = pool(h, pid, P)
    # reference: one query per patch attends to its own bytes through a dense (P, T) mask
    pooled = h.new_full((2, P, 16), float("-inf")).scatter_reduce(1, pid[..., None].expand_as(h), h, "amax")
    q = pool.q(torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled))
    k, v = pool.kv(h).chunk(2, -1)
    sh = lambda t: t.view(2, -1, 4, 8).transpose(1, 2)
    mask = pid[:, None, :] == torch.arange(P)[None, :, None]
    y = torch.nan_to_num(torch.nn.functional.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask[:, None]))
    torch.testing.assert_close(got, q + pool.o(y.transpose(1, 2).reshape(2, P, -1)))


@pytest.mark.parametrize("patching", ["space", "entropy"])
def test_blt_stream_matches_forward(patching):
    torch.manual_seed(0)
    m = ByteLatentTransformer(BLTConfig(seq_len=64, d_local=32, d_latent=48, heads=2, max_patches=65)).eval()
    ent = TinyLM(Config(vocab_size=256, seq_len=64, d_model=48, n_heads=2, n_layers=1)).eval()
    x1 = torch.cat([TEXT, TEXT[:, :1]], 1)  # one extra byte: its column is the stream's next-patch decision
    if patching == "space":
        bnd, st = space_boundaries(x1, 6), BLTStream(m, "space", max_patch=6)
    else:
        h = byte_entropy(ent, x1)
        theta = h.median().item() + 1e-4
        bnd, st = entropy_boundaries(h, theta_global=theta, max_patch=6), BLTStream(m, "entropy", ent, theta, 6)
    out = torch.stack([st.feed(int(b)) for b in TEXT[0]])
    torch.testing.assert_close(out, m(TEXT, bnd)[0], atol=1e-5, rtol=1e-4)
    assert st.calls["latent"] == int(bnd[0, 1:].sum())  # the latent ran once per closed patch


def test_hnet_and_lm_streams_match_forward():
    torch.manual_seed(0)
    hn = HNet(HNetConfig(seq_len=64, d_outer=32, d_main=48, heads=2)).eval()
    st = HNetStream(hn)
    out = torch.stack([st.feed(int(b)) for b in TEXT[0]])
    ref, _, stats = hn(TEXT)
    torch.testing.assert_close(out, ref[0], atol=1e-4, rtol=1e-4)
    assert st.calls["main"] == stats["chunks"]
    lm = TinyLM(Config(vocab_size=256, seq_len=64, d_model=48, n_heads=2, n_layers=2)).eval()
    st = LMStream(lm)
    torch.testing.assert_close(torch.stack([st.feed(int(b)) for b in TEXT[0]]), lm(TEXT)[0], atol=1e-5, rtol=1e-4)
