import io

import pytest
import torch

from littlebit_scratch.littlebit import LittleBitLinear
from littlebit_scratch.model import Config, TinyLM
from littlebit_scratch.packed import PackedLittleBitLinear, model_bytes, pack_model, pack_signs, unpack_signs


@pytest.mark.parametrize("r", [1, 7, 8, 13, 64])
def test_pack_unpack_roundtrip(r):
    torch.manual_seed(0)
    x = torch.randn(10, r)
    x[0, 0] = 0.0
    p = pack_signs(x)
    assert p.dtype == torch.uint8 and p.shape == (10, (r + 7) // 8)
    torch.testing.assert_close(unpack_signs(p, r), torch.where(x >= 0, 1.0, -1.0))


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("rank,chunk", [(5, 8), (40, 16), (40, 128)])
def test_packed_layer_matches(residual, rank, chunk):
    torch.manual_seed(0)
    layer = LittleBitLinear(48, 32, rank=rank, residual=residual, bias=True)
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    x = torch.randn(6, 48)
    ref = layer.forward_reference(x).detach()
    packed = PackedLittleBitLinear(layer, chunk=chunk)
    with torch.no_grad():
        out = packed(x)
    # only difference: FP16 scales
    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)
    assert not any(p.requires_grad for p in packed.buffers())


def test_pack_model_shrinks_and_matches():
    torch.manual_seed(0)
    model = TinyLM(Config(vocab_size=32, seq_len=16, d_model=128, n_heads=4, n_layers=2, linear="littlebit"))
    x = torch.randint(0, 32, (2, 16))
    model.eval()
    with torch.no_grad():
        ref = model(x)
    before = model_bytes(model)
    pack_model(model)
    assert not any(isinstance(m, LittleBitLinear) for m in model.modules())
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    assert model_bytes(model) < before / 4
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    assert buf.tell() < before / 3


def test_empty_packed_layer_loads_state():
    torch.manual_seed(0)
    layer = LittleBitLinear(40, 24, rank=11, residual=True, bias=True)
    src = PackedLittleBitLinear(layer)
    dst = PackedLittleBitLinear.empty(40, 24, 11, 2, bias=True)
    dst.load_state_dict(src.state_dict())
    x = torch.randn(3, 40)
    with torch.no_grad():
        torch.testing.assert_close(dst(x), src(x))
