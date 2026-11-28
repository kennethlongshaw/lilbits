import torch

from lilbits.models.looped_msa import LoopedMSALM, select_blocks


def test_select_blocks_keeps_own_block_and_k_blocks():
    torch.manual_seed(0)
    T, block, k = 64, 8, 3
    causal = torch.arange(T)[None] <= torch.arange(T)[:, None]
    s = torch.randn(1, 1, T, T).masked_fill(~causal, float("-inf"))
    m = select_blocks(s, block, k, torch.arange(T))[0, 0]
    assert not (m & ~causal).any()
    for t in range(T):
        assert m[t, t]  # own block always kept
        blocks = m[t].view(-1, block).any(-1).sum().item()
        assert blocks == min(k, t // block + 1)


def test_decode_matches_parallel_forward():
    torch.manual_seed(0)
    for sparse in (True, False):
        m = LoopedMSALM(d=64, gdn_heads=2, n_prelude=1, loops=2, attn_heads=4, groups=2, d_idx=8, block=4, topk=2)
        m.sparse = sparse
        m.eval()
        prompt = torch.randint(0, 256, (1, 9))
        gen, info = m.generate(prompt, 12)
        full = torch.cat([prompt, gen], 1)
        logits, _, _ = m(full[:, :-1])
        ref = logits[:, prompt.shape[1] - 1 :].argmax(-1)
        assert torch.equal(ref, gen), sparse
        if sparse:
            assert info["attended_keys_per_loop"] <= 2 * 4


def test_loss_backward_index_grad_isolated():
    torch.manual_seed(0)
    m = LoopedMSALM(d=64, gdn_heads=2, n_prelude=1, loops=2, attn_heads=4, groups=2, d_idx=8, block=4, topk=2)
    x = torch.randint(0, 256, (2, 33))
    _, kl, _ = m(x[:, :-1], kl=True)
    kl.backward()
    # the KL reaches only the index projections
    assert m.core.q_idx.weight.grad is not None and m.core.k_idx.weight.grad is not None
    assert m.core.q.weight.grad is None and m.embed.weight.grad is None
    loss, info = m.loss(x)
    loss.backward()
    assert "index_kl" in info
