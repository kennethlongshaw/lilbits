import math

import pytest
import torch

from lilbits.data import MarkovSource
from lilbits.models.tinylm import Config, TinyLM
from lilbits.training.optim import DeepSeekMuon, hybrid_newton_schulz
from lilbits.training.train import TrainConfig, make_optimizers, train


@pytest.mark.parametrize("shape", [(64, 64), (96, 32), (32, 96)])
def test_hybrid_newton_schulz_pins_singular_values(shape):
    torch.manual_seed(0)
    G = torch.randn(*shape)
    s = torch.linalg.svdvals(hybrid_newton_schulz(G))
    assert s.min() > 0.9 and s.max() < 1.1
    # torch's default (5 fast steps, no stabilizing phase) leaves a much wider spread
    s_fast = torch.linalg.svdvals(hybrid_newton_schulz(G, fast_steps=5, stable_steps=0))
    assert (s_fast.max() - s_fast.min()) > (s.max() - s.min())


def test_deepseek_muon_update_rms():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(64, 32))
    opt = DeepSeekMuon([p], lr=1.0, momentum=0.0, nesterov=False)
    p.grad = torch.randn(64, 32)
    opt.step()
    assert abs(p.detach().pow(2).mean().sqrt().item() - 0.18) < 0.02


def test_deepseek_muon_rejects_1d():
    with pytest.raises(ValueError):
        DeepSeekMuon([torch.nn.Parameter(torch.zeros(4))])


def test_make_optimizers_muon_ds():
    model = TinyLM(
        Config(vocab_size=16, seq_len=16, d_model=64, n_heads=2, n_layers=1, linear="littlebit", bpw=4.0, qk_norm=True)
    )
    ds, adam = make_optimizers(model, TrainConfig(lr=1e-3, optim="muon_ds"))
    assert isinstance(ds, DeepSeekMuon) and isinstance(adam, torch.optim.AdamW)


def test_qk_norm_training_with_deepseek_muon_learns():
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
        qk_norm=True,
    )
    res = train(
        cfg,
        TrainConfig(
            steps=300,
            batch=32,
            lr=3e-3,
            log_every=100,
            latent_lr_mult=1.0,
            optim="muon_ds",
            verbose=False,
            accelerator="cpu",
        ),
        src,
    )
    assert res["history"][-1]["eval"] < math.log(8) - 0.3
