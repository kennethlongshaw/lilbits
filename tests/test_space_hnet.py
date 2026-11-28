import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from lilbits.inference import bundle, fused, fused_hnet
from lilbits.models.lean_lm import LeanLM
from lilbits.models.space_hnet import SpaceHNet, SpaceHNetConfig, SpaceHNetStream, boundaries

TEXT = b"It was the best of times, it was the worst of times; it was the age of wisdom...  \n\nHAMLET:\nTo be, or not."
BIN = Path(__file__).resolve().parents[1] / "rust" / "lilbits_fused" / "target" / "release" / "lilbits"


def make(align="end", seed=0):
    torch.manual_seed(seed)
    m = SpaceHNet(SpaceHNetConfig(d_outer=64, d_main=96, heads_outer=2, heads_main=3, n_enc=1, n_main=2, n_dec=2,
                                  align=align, max_patch=8))  # fmt: skip
    with torch.no_grad():  # non-trivial norm gains / conv scales, so every fold in the export is exercised
        for n, p in m.named_parameters():
            if "norm" in n or "conv" in n or n.startswith(("n_up", "n_down")):
                p.mul_(1 + 0.3 * torch.randn_like(p))
    return m.eval()


def toks(n=None):
    x = torch.tensor(list(TEXT))[None]
    return x if n is None else x[:, :n]


def test_boundaries_rule():
    x = torch.tensor([list(b"ab cd,  ef")])
    end = boundaries(x, "end", None)[0].tolist()
    start = boundaries(x, "start", None)[0].tolist()
    #                  a  b  _  c  d  ,  _  _  e  f
    assert end == [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    assert start == [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    capped = boundaries(torch.tensor([list(b"abcdefghij")]), "end", 4)[0].tolist()
    assert capped == [1, 0, 0, 0, 1, 0, 0, 0, 1, 0]


@pytest.mark.parametrize("align", ["end", "start"])
def test_stream_matches_parallel(align):
    m = make(align)
    x = toks()
    with torch.no_grad():
        ref, _ = m(x)
    s = SpaceHNetStream(m)
    out = torch.stack([s.feed(int(b)) for b in x[0]])
    assert (out - ref[0]).abs().max() < 1e-4
    assert s.main_steps == int(m.boundaries(x).sum())


@pytest.mark.parametrize("align", ["end", "start"])
def test_fused_f32_matches_torch(align):
    m = make(align)
    x = toks()
    with torch.no_grad():
        ref, _ = m(x)
    lm = fused_hnet.FusedHNet(fused_hnet.export(m, "f32"), block=16)
    assert np.abs(lm.logits_all(x[0].numpy()) - ref[0].numpy()).max() < 1e-4
    lm.reset()  # state carries over a split prefill
    lm.prefill(x[0, :37].numpy())
    assert np.abs(lm.logits_all(x[0, 37:].numpy()) - ref[0, 37:].numpy()).max() < 1e-4


def test_fused_greedy_matches_stream():
    m = make()
    lm = fused_hnet.FusedHNet(fused_hnet.export(m, "f32"))
    lm.prefill(toks(40)[0].numpy())
    got = lm.generate(30)
    s = SpaceHNetStream(m)
    for b in toks(40)[0]:
        lg = s.feed(int(b))
    ref = []
    for _ in range(30):
        ref.append(int(lg.argmax()))
        lg = s.feed(ref[-1])
    assert got == ref


def test_w8a8_close():
    m = make()
    x = toks()[0].numpy()
    a = fused_hnet.FusedHNet(fused_hnet.export(m, "f32")).logits_all(x)
    b = fused_hnet.FusedHNet(fused_hnet.export(m, "w8a8")).logits_all(x)
    pa, pb = (np.exp(v - v.max(1, keepdims=True)) for v in (a, b))
    pa, pb = pa / pa.sum(1, keepdims=True), pb / pb.sum(1, keepdims=True)
    assert (pa * (np.log(pa) - np.log(pb))).sum(1).mean() < 0.02  # mean KL in nats


def test_bundle_roundtrip(tmp_path):
    arr = fused_hnet.export(make(), "w8a8")
    bundle.write(tmp_path / "m.lbm", arr)
    back = bundle.read(tmp_path / "m.lbm")
    assert set(back) == set(arr)
    assert all(back[k].tobytes() == np.ascontiguousarray(arr[k]).tobytes() for k in arr)


@pytest.mark.skipif(shutil.which("cargo") is None and not BIN.exists(), reason="needs cargo or a built CLI")
@pytest.mark.parametrize("kind", ["hnet", "plain"])
def test_cli_greedy_matches_python(tmp_path, kind):
    if not BIN.exists():
        subprocess.run(["cargo", "build", "--release", "--manifest-path", str(BIN.parents[2] / "Cargo.toml")],
                       check=True, capture_output=True)  # fmt: skip
    if kind == "hnet":
        arr, cls = fused_hnet.export(make(), "w8a8"), fused_hnet.FusedHNet
    else:
        torch.manual_seed(0)
        lean = LeanLM(64, 4, 2, lambda i, o: nn.Linear(i, o, bias=False), n_draft=0).eval()
        arr, cls = fused.export(lean, "w8a8"), fused.FusedLM
    bundle.write(tmp_path / "m.lbm", arr)
    prompt = "It was a dark and stormy night; "
    lm = cls(arr)
    lm.prefill(list(prompt.encode()))
    ref = bytes(lm.generate(50))
    out = subprocess.run(
        [str(BIN), str(tmp_path / "m.lbm"), "--prompt", prompt, "-n", "50", "--temperature", "0"],
        capture_output=True,
        check=True,
    ).stdout
    assert out == prompt.encode() + ref + b"\n"
