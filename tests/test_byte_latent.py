import torch

from lilbits.models.byte_latent import (
    BLTConfig,
    ByteLatentTransformer,
    DynamicChunker,
    HashNgramEmbedding,
    HNet,
    HNetConfig,
    entropy_boundaries,
    patch_ids,
)


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
