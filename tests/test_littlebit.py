import pytest
import torch

from lilbits.layers.littlebit import (
    LittleBitLinear,
    binarize,
    bits_per_weight,
    rank_for_bpw,
)


@pytest.mark.parametrize("ste", ["smooth", "clipped"])
def test_binarize_forward_is_pm1(ste):
    x = torch.randn(50, 7)
    x[0, 0] = 0.0
    b = binarize(x, ste)
    assert set(b.unique().tolist()) <= {-1.0, 1.0}
    assert b[0, 0] == 1.0
    assert torch.equal(b, torch.where(x >= 0, 1.0, -1.0))


def test_smoothsign_gradient_is_tanh_derivative():
    k = 10.0
    x = torch.linspace(-0.5, 0.5, 21, requires_grad=True)
    binarize(x, "smooth", k).sum().backward()
    expected = k * (1 - torch.tanh(k * x.detach()) ** 2)
    torch.testing.assert_close(x.grad, expected)


def test_clipped_ste_gradient():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    binarize(x, "clipped").sum().backward()
    assert x.grad is not None
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_forward_matches_materialized_weight(residual, bias):
    torch.manual_seed(0)
    layer = LittleBitLinear(24, 40, rank=6, residual=residual, bias=bias)
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(3, 5, 24)
    expected = x @ layer.weight().t()
    if bias:
        assert layer.bias is not None
        expected = expected + layer.bias
    torch.testing.assert_close(layer(x), expected, rtol=1e-4, atol=1e-5)


def test_only_latent_signs_matter():
    torch.manual_seed(0)
    layer = LittleBitLinear(16, 16, rank=4)
    x = torch.randn(8, 16)
    y0 = layer(x)
    with torch.no_grad():
        for p in layer.paths:
            p.U.mul_(torch.rand_like(p.U) * 5 + 0.1)  # positive rescale keeps signs
            p.V.mul_(torch.rand_like(p.V) * 5 + 0.1)
    torch.testing.assert_close(layer(x), y0)


def test_effective_weight_rank_bounded():
    layer = LittleBitLinear(32, 32, rank=5)
    assert torch.linalg.matrix_rank(layer.weight()).item() <= 5
    layer2 = LittleBitLinear(32, 32, rank=5, residual=True)
    assert torch.linalg.matrix_rank(layer2.weight()).item() <= 10


def test_all_parameters_get_gradients():
    layer = LittleBitLinear(16, 12, rank=4, residual=True, bias=True)
    layer(torch.randn(4, 16)).pow(2).sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("shape", [(128, 128), (128, 344), (1024, 4096)])
@pytest.mark.parametrize("bpw", [0.1, 0.55, 1.0])
@pytest.mark.parametrize("paths", [1, 2])
def test_rank_for_bpw_respects_budget(shape, bpw, paths):
    i, o = shape
    r = rank_for_bpw(i, o, bpw, paths)
    if r > 1 and r < min(i, o):
        assert bits_per_weight(i, o, r, paths) <= bpw
        assert bits_per_weight(i, o, r + 1, paths) > bpw


def test_large_layer_hits_sub_1_bit():
    layer = LittleBitLinear(4096, 4096, bpw=0.1, residual=True)
    assert 0.09 < layer.bpw <= 0.1
    assert layer.rank > 50


def test_init_preserves_scale():
    torch.manual_seed(0)
    layer = LittleBitLinear(256, 256, rank=64)
    x = torch.randn(512, 256)
    ratio = layer(x).std() / x.std()
    assert 0.5 < ratio.item() < 2.0


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("ste", ["smooth", "clipped"])
def test_fused_forward_matches_reference_including_grads(residual, ste):
    torch.manual_seed(0)
    layer = LittleBitLinear(24, 40, rank=6, residual=residual, bias=True, ste=ste)
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    x = torch.randn(3, 5, 24)
    y_fused = layer(x)
    g_fused = torch.autograd.grad(y_fused.pow(2).sum(), list(layer.parameters()))
    y_ref = layer.forward_reference(x)
    g_ref = torch.autograd.grad(y_ref.pow(2).sum(), list(layer.parameters()))
    torch.testing.assert_close(y_fused, y_ref, rtol=1e-4, atol=1e-5)
    for a, b in zip(g_fused, g_ref):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("rank", [2, 30])  # low-rank path and dense fallback
def test_inference_cache_matches_training_forward(rank):
    torch.manual_seed(0)
    layer = LittleBitLinear(32, 48, rank=rank, residual=True, bias=True)
    x = torch.randn(4, 32)
    y_train = layer(x).detach()
    layer.eval()
    with torch.no_grad():
        y_inf = layer(x)
        assert layer._infer_cache is not None
        assert layer._infer_cache[0] == ("dense" if rank == 30 else "lowrank")
    torch.testing.assert_close(y_inf, y_train, rtol=1e-4, atol=1e-5)


def test_inference_cache_cleared_when_training_resumes():
    torch.manual_seed(0)
    layer = LittleBitLinear(16, 16, rank=4)
    x = torch.randn(2, 16)
    layer.eval()
    with torch.no_grad():
        layer(x)
    assert layer._infer_cache is not None
    layer.train()
    assert layer._infer_cache is None
    with torch.no_grad():
        layer.paths[0].h.mul_(2)
    layer.eval()
    with torch.no_grad():
        torch.testing.assert_close(layer(x), layer.forward_reference(x), rtol=1e-4, atol=1e-5)


def test_int8_scales_rank_storage_and_packing():
    from lilbits.inference.lut import LUTLittleBitLinear
    from lilbits.inference.packed import PackedLittleBitLinear
    from lilbits.layers.littlebit import rank_for_bpw

    # cheaper scales buy rank at the same budget, most at small widths
    assert rank_for_bpw(128, 128, 0.55, 2, scale_bits=8) > rank_for_bpw(128, 128, 0.55, 2)
    torch.manual_seed(0)
    l16 = LittleBitLinear(128, 128, bpw=0.55, residual=True)
    l8 = LittleBitLinear(128, 128, bpw=0.55, residual=True, scale_bits=8)
    assert l8.bpw <= 0.55 and l8.rank > l16.rank
    with torch.no_grad():
        for p in l8.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    l8.eval()
    x = torch.randn(3, 128)
    with torch.no_grad():
        ref = l8(x)
        torch.testing.assert_close(l8.forward_reference(x), ref, rtol=1e-4, atol=1e-4)
        packed = PackedLittleBitLinear(l8)
        assert packed.h_q.dtype == torch.int8 and "h" not in packed.state_dict()
        torch.testing.assert_close(packed(x), ref, rtol=1e-3, atol=1e-3)
        for src in (l8, packed):
            torch.testing.assert_close(LUTLittleBitLinear(src)(x), ref, rtol=1e-3, atol=1e-3)
        # storage: byte-padded signs + INT8 scales + one FP16 step per vector, per path
        r = l8.rank
        assert packed.storage_bytes() == 2 * (256 * ((r + 7) // 8) + (256 + r) + 3 * 2)
        # a packed INT8 checkpoint reloads into an empty layer
        empty = PackedLittleBitLinear.empty(128, 128, l8.rank, 2, scale_bits=8)
        empty.load_state_dict(packed.state_dict())
        torch.testing.assert_close(empty(x), packed(x))
