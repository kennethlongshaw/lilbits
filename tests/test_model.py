import math

import pytest
import torch

from lilbits.data import MarkovSource
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.models.tinylm import Config, TinyLM, TransformerBlock, compression_report, make_linear_factory
from lilbits.training.train import TrainConfig, train

TINY = dict(vocab_size=16, seq_len=16, d_model=32, n_heads=2, n_layers=1)


@pytest.mark.parametrize("linear", ["fp", "littlebit"])
def test_block_shape_and_causality(linear):
    torch.manual_seed(0)
    cfg = Config(**TINY, linear=linear)
    block = TransformerBlock(32, 2, 64, make_linear_factory(cfg))
    x = torch.randn(2, 10, 32)
    y = block(x)
    assert y.shape == x.shape
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 32)
    torch.testing.assert_close(block(x2)[:, :6], y[:, :6])


def test_littlebit_model_uses_littlebit_everywhere_in_blocks():
    model = TinyLM(Config(**TINY, linear="littlebit"))
    lb = [m for m in model.modules() if isinstance(m, LittleBitLinear)]
    assert len(lb) == 7  # q k v o gate up down
    assert isinstance(model.head, torch.nn.Linear)
    # tiny layers can't fit the FP16 scales in a 1-bit budget, so rank clamps to 1
    assert all(m.rank == 1 for m in lb)
    big = TinyLM(Config(vocab_size=16, seq_len=16, d_model=128, n_heads=2, n_layers=1, linear="littlebit"))
    assert 0.9 < compression_report(big)["bpw"] <= 1.0
    assert compression_report(TinyLM(Config(**TINY, linear="fp")))["bpw"] == 16


def test_markov_source():
    src = MarkovSource(vocab=16, order=2, branching=3, seed=0)
    s = src.sample(8, 32)
    assert s.shape == (8, 33) and s.min() >= 0 and s.max() < 16
    floor = src.entropy_floor(s)
    assert 0 < floor < math.log(3) + 1e-6
    assert torch.equal(MarkovSource(vocab=16, seed=0).probs, MarkovSource(vocab=16, seed=0).probs)


@pytest.mark.parametrize("linear", ["fp", "littlebit"])
def test_training_from_scratch_learns(linear):
    """A short from-scratch run beats the best context-free (unigram) predictor."""
    src = MarkovSource(vocab=8, order=1, branching=2, seed=0)
    cfg = Config(vocab_size=8, seq_len=16, d_model=32, n_heads=2, n_layers=1, linear=linear,
                 bpw=2.0, residual=False, ste="clipped")
    res = train(cfg, TrainConfig(steps=300, batch=32, lr=3e-3, log_every=100, latent_lr_mult=0.1, verbose=False, accelerator="cpu"), src)
    eval_b = res["history"]
    assert eval_b[-1]["eval"] < eval_b[0]["eval"] - 0.3
    assert eval_b[-1]["eval"] < math.log(8) - 0.3


def test_muon_gets_2d_block_matrices_only():
    from lilbits.training.train import make_optimizers

    model = TinyLM(Config(vocab_size=16, seq_len=16, d_model=64, n_heads=2, n_layers=1, linear="littlebit", bpw=4.0))
    muon, adam = make_optimizers(model, TrainConfig(lr=1e-3, optim="muon"))
    assert isinstance(muon, torch.optim.Muon) and isinstance(adam, torch.optim.AdamW)
    muon_params = [p for g in muon.param_groups for p in g["params"]]
    assert muon_params and all(p.dim() == 2 for p in muon_params)
    ids = {id(p) for p in muon_params}
    assert id(model.tok.weight) not in ids and id(model.head.weight) not in ids
    lb = next(m for m in model.modules() if isinstance(m, LittleBitLinear))
    assert id(lb.paths[0].U) in ids and id(lb.paths[0].h) not in ids
    n_total = sum(1 for _ in model.parameters())
    assert len(muon_params) + sum(len(g["params"]) for g in adam.param_groups) == n_total


def test_training_with_muon_learns():
    src = MarkovSource(vocab=8, order=1, branching=2, seed=0)
    cfg = Config(vocab_size=8, seq_len=16, d_model=32, n_heads=2, n_layers=1, linear="littlebit",
                 bpw=2.0, residual=False)
    res = train(cfg, TrainConfig(steps=300, batch=32, lr=3e-3, log_every=100, latent_lr_mult=1.0, optim="muon", verbose=False, accelerator="cpu"), src)
    assert res["history"][-1]["eval"] < math.log(8) - 0.3
