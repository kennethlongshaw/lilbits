import pytest
import torch

from lilbits.inference import lut
from lilbits.inference.packed import PackedLittleBitLinear, pack_model
from lilbits.layers.bitnet import BitLinear, PackedBitLinear
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.models.tinylm import Config, TinyLM


@pytest.mark.parametrize("shape", [(37, 22, 1), (64, 64, 3), (1000, 513, 5)])
def test_codes_roundtrip_and_matmul(shape):
    O, I, N = shape
    torch.manual_seed(0)
    S, x = torch.randn(O, I), torch.randn(N, I)
    codes = lut.pack_sign_codes(S)
    assert codes.numel() * 8 >= O * I and codes.numel() * 8 < (O + 32) * (I + 4)  # 1 bit / weight + padding
    signs = torch.where(S >= 0, 1.0, -1.0)
    assert torch.equal(lut.unpack_sign_codes(codes, O, I), signs)
    ref = x @ signs.t()
    torch.testing.assert_close(lut.lut_matmul_ref(x, codes, O), ref, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(lut.lut_matmul(x, codes, O), ref, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(lut.lut_matmul(x, codes, O, scalar=True), ref, rtol=1e-5, atol=1e-4)


def test_kernel_compiles_here():
    info = lut.kernel_info()
    assert info["compiled"], info


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("tokens", [1, 4, 20])  # 20 > max_lut_tokens: dense prefill path
def test_littlebit_lut_matches_packed(residual, tokens):
    torch.manual_seed(0)
    layer = LittleBitLinear(72, 50, rank=12, residual=residual, bias=True).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    packed = PackedLittleBitLinear(layer)
    x = torch.randn(2, tokens, 72)
    with torch.no_grad():
        ref = packed(x)
        for src in (layer, packed):
            torch.testing.assert_close(lut.LUTLittleBitLinear(src)(x), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("tokens", [1, 3, 20])
def test_bitnet_lut_matches_bitlinear(tokens):
    torch.manual_seed(0)
    layer = BitLinear(66, 40, bias=True).eval()
    x = torch.randn(tokens, 66)
    with torch.no_grad():
        ref = layer(x)
        for src in (layer, PackedBitLinear(layer)):
            m = lut.LUTBitLinear(src)
            assert m.P_codes.numel() + m.Q_codes.numel() <= 2 * (40 + 32) * (66 + 4) / 8  # 2 bits / weight
            torch.testing.assert_close(m(x), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("linear", ["littlebit", "bitnet"])
def test_pack_lut_model(linear):
    torch.manual_seed(0)
    model = TinyLM(Config(vocab_size=16, seq_len=16, d_model=32, n_heads=2, n_layers=1, linear=linear)).eval()
    x = torch.randint(0, 16, (1, 6))
    with torch.no_grad():
        ref = model(x)
        if linear == "littlebit":
            pack_model(model)
        lut.pack_lut(model)
        assert not any(isinstance(m, (LittleBitLinear, BitLinear, PackedLittleBitLinear)) for m in model.modules())
        torch.testing.assert_close(model(x), ref, rtol=1e-3, atol=1e-3)


def test_torch_fallback_without_compiler(monkeypatch):
    torch.manual_seed(0)
    lb = LittleBitLinear(40, 24, rank=8, residual=True).eval()
    bl = BitLinear(40, 24).eval()
    x = torch.randn(2, 40)
    with torch.no_grad():
        fused = lut.LUTLittleBitLinear(lb)(x), lut.LUTBitLinear(bl)(x)
        monkeypatch.setattr(lut, "_load", lambda: None)
        fallback = lut.LUTLittleBitLinear(lb)(x), lut.LUTBitLinear(bl)(x)
    for a, b in zip(fused, fallback, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
