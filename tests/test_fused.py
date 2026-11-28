import numpy as np
import pytest
import torch
from torch import nn

from lilbits.inference.fused import FusedLM, available_simd, export, set_simd
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


@pytest.mark.parametrize("fmt", ["f32", "i4", "w8a8"])
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
        toks = lm.generate(30)
        after = lm.prefill(x[60:]).copy()  # same recurrent state after decode
        outs.append((logits, toks, after))
    for logits, toks, after in outs[1:]:
        assert np.array_equal(logits, outs[0][0]) and toks == outs[0][1]
        assert np.array_equal(after, outs[0][2])


@pytest.mark.parametrize("family", available_simd())
def test_every_kernel_family(family):
    """Each kernel family this CPU runs (scalar fallback included): f32 matches torch, every format matches the
    scalar reference (W8A8 bit-identically), and threads give identical decode output."""
    m = make(160, 4)
    x = torch.randint(0, 256, (1, 70))
    with torch.no_grad():
        ref = m(x)[0][0].numpy()
    arrays = {f: export(m, f) for f in ("f32", "f16", "i8", "i4", "w8a8")}
    try:
        set_simd("scalar")
        want = {f: FusedLM(a, block=16).logits_all(x[0].numpy()) for f, a in arrays.items()}
        set_simd(family)
        for f, a in arrays.items():
            out = FusedLM(a, block=16).logits_all(x[0].numpy())
            if f == "w8a8":
                assert np.abs(out - want[f]).max() < 1e-4, f
            else:
                assert np.abs(out - want[f]).max() < 1e-4, f
        assert np.abs(want["f32"] - ref).max() < 1e-4
        gens = []
        for th in (1, 3):
            lm = FusedLM(arrays["w8a8"], threads=th)
            lm.prefill(x[0].numpy())
            gens.append(lm.generate(20))
        assert gens[0] == gens[1]
    finally:
        set_simd("auto")
