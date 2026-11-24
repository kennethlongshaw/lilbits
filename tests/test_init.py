from typing import Any

import torch

from lilbits.data import MarkovSource
from lilbits.layers.littlebit import LittleBitLinear, joint_itq, zero_margin
from lilbits.models.tinylm import Config, TinyLM, convert_fp_to_littlebit
from lilbits.training.train import TrainConfig, train


def _factors(seed=0, out=96, inp=64, r=16):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out, inp, generator=g)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return W, U[:, :r] * S[:r].sqrt(), Vh[:r].t() * S[:r].sqrt()


def test_joint_itq_is_orthogonal_and_function_preserving():
    _, Up, Vp = _factors()
    R = joint_itq(Up, Vp, iters=30)
    torch.testing.assert_close(R @ R.t(), torch.eye(R.shape[0]), atol=1e-5, rtol=0)
    torch.testing.assert_close((Up @ R) @ (Vp @ R).t(), Up @ Vp.t(), atol=1e-4, rtol=1e-4)


def test_joint_itq_moves_latents_away_from_zero():
    _, Up, Vp = _factors()
    Z = torch.cat([Up / Up.norm(), Vp / Vp.norm()])
    R = joint_itq(Up, Vp, iters=50)
    ZR = Z @ R
    assert zero_margin(ZR) < zero_margin(Z)

    # closer to the binary hypercube: sign(ZR) explains more of ZR
    def fit(X):
        Xs = X / X.norm()
        return (X.sign() * Xs).sum() / X.sign().norm()

    assert fit(ZR) > fit(Z)


def test_init_from_weight_beats_random_and_residual_helps():
    torch.manual_seed(0)
    # low-rank-ish target so a small rank can capture most of it
    W = torch.randn(64, 8) @ torch.randn(8, 48) + 0.1 * torch.randn(64, 48)
    rand = LittleBitLinear(48, 64, rank=8)
    one = LittleBitLinear(48, 64, rank=8)
    one.init_from_weight(W)
    two = LittleBitLinear(48, 64, rank=8, residual=True)
    two.init_from_weight(W)
    err = lambda m: ((m.weight() - W).norm() / W.norm()).item()
    assert err(one) < 0.9 < err(rand)
    assert err(two) < err(one)


def test_svd_init_matches_dense_init_norm():
    torch.manual_seed(0)
    layer = LittleBitLinear(128, 128, rank=16, init="svd", itq_iters=20)
    ref = torch.empty(128, 128)
    torch.nn.init.kaiming_uniform_(ref, a=5**0.5)
    assert abs(layer.weight().norm() / ref.norm() - 1) < 0.05
    assert all(p.U.abs().max() <= 1 + 1e-6 for p in layer.paths)


def test_bimodal_init_has_no_latents_near_zero():
    torch.manual_seed(0)
    layer = LittleBitLinear(64, 64, rank=16, init="bimodal")
    assert all(zero_margin(p.U) == 0 and zero_margin(p.V) == 0 for p in layer.paths)


def test_convert_fp_to_littlebit_keeps_fp_parts():
    torch.manual_seed(0)
    base: dict[str, Any] = dict(vocab_size=16, seq_len=16, d_model=64, n_heads=2, n_layers=1)
    fp = TinyLM(Config(**base, linear="fp"))
    lb = convert_fp_to_littlebit(fp, Config(**base, linear="littlebit", bpw=4.0, itq_iters=10))
    torch.testing.assert_close(lb.tok.weight, fp.tok.weight)
    torch.testing.assert_close(lb.head.weight, fp.head.weight)
    assert torch.isfinite(lb(torch.randint(0, 16, (2, 8)))).all()


def test_train_with_fp_warmup_records_phases():
    src = MarkovSource(vocab=8, order=1, branching=2, seed=0)
    cfg = Config(
        vocab_size=8,
        seq_len=16,
        d_model=32,
        n_heads=2,
        n_layers=1,
        linear="littlebit",
        bpw=2.0,
        residual=False,
        itq_iters=10,
    )
    res = train(
        cfg, TrainConfig(steps=20, batch=8, lr=3e-3, log_every=5, fp_warmup=10, verbose=False, accelerator="cpu"), src
    )
    phases = [h["phase"] for h in res["history"]]
    assert phases[0] == "fp-warm" and "converted" in phases and phases[-1] == "littlebit"
    assert res["history"][-1]["step"] == 19
