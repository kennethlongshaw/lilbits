import numpy as np
import pytest
import torch
from torch import nn

from lilbits.inference.fused import FusedLM, export
from lilbits.models.lean_lm import LeanLM


def make(d=200, heads=1, layers=3, seed=0):
    torch.manual_seed(seed)
    m = LeanLM(d, heads, layers, lambda i, o: nn.Linear(i, o, bias=False), n_draft=0).eval()
    with torch.no_grad():  # non-trivial norm gains and conv scales, so the folding is exercised
        for n, p in m.named_parameters():
            if "norm" in n or "conv" in n:
                p.mul_(1 + 0.3 * torch.randn_like(p))
    return m


@pytest.mark.parametrize(("d", "heads"), [(200, 1), (256, 4), (96, 3)])
def test_f32_matches_torch(d, heads):
    m = make(d, heads)
    x = torch.randint(0, 256, (1, 100))
    with torch.no_grad():
        ref, _ = m(x)
    lm = FusedLM(export(m, "f32"), block=16)
    out = lm.logits_all(x[0].numpy())
    assert np.abs(out - ref[0].numpy()).max() < 1e-4


def test_f32_greedy_and_state_continue():
    m = make(128, 2)
    x = torch.randint(0, 256, (1, 50))
    with torch.no_grad():
        ref = m.generate(x, 20)[0].tolist()
    lm = FusedLM(export(m, "f32"))
    lm.prefill(x[0, :30].numpy())
    lm.prefill(x[0, 30:].numpy())  # prefill in two parts: state carries over
    assert lm.generate(20) == ref


@pytest.mark.parametrize("fmt", ["f16", "i8", "i4", "w8a8"])
def test_quantized_close(fmt):
    m = make(128, 2)
    x = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        ref = torch.log_softmax(m(x)[0][0], -1).numpy()
    lm = FusedLM(export(m, fmt))
    out = torch.log_softmax(torch.from_numpy(lm.logits_all(x[0].numpy())), -1).numpy()
    # KL(ref || out) per position stays small for every format
    kl = (np.exp(ref) * (ref - out)).sum(-1).mean()
    assert kl < {"f16": 1e-5, "i8": 1e-3, "w8a8": 2e-3, "i4": 2e-2}[fmt]


@pytest.mark.parametrize("fmt", ["f32", "w8a8"])
def test_threads_match_one_thread(fmt):
    m = make(128, 4)
    arrays = export(m, fmt)
    x = torch.randint(0, 256, (100,)).numpy()
    outs = []
    for th in (1, 3, 4):
        lm = FusedLM(arrays, block=16, threads=th)
        logits = lm.logits_all(x[:60])
        lm.reset()
        lm.prefill(x[:60])
        outs.append((logits, lm.generate(30)))
    for logits, toks in outs[1:]:
        assert np.array_equal(logits, outs[0][0]) and toks == outs[0][1]
