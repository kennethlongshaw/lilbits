"""Helion kernels: Triton codegen for every candidate config, and numerics in Helion's eager
reference mode (HELION_INTERPRET=1, runs on CPU). Real-GPU speed is measured by python benchmarks/gpu_bench.py."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytest.importorskip("helion")


def test_triton_codegen_all_configs():
    from littlebit_scratch import helion_kernels as hk

    m, k, n, r = 1, 512, 768, 100
    nb = (r + 7) // 8
    x = torch.randn(m, k, dtype=torch.float16)
    g = torch.rand(k, dtype=torch.float16)
    vb = torch.randint(0, 256, (k, nb), dtype=torch.uint8)
    l8 = torch.rand(8, nb)
    zp = torch.rand(8, m, nb)
    ub = torch.randint(0, 256, (n, nb), dtype=torch.uint8)
    h = torch.rand(n, dtype=torch.float16)
    y = torch.zeros(m, n, dtype=torch.float16)
    bd, bu = hk.littlebit_down.bind((x, g, vb, l8)), hk.littlebit_up.bind((zp, ub, h, y))
    for c in hk.CONFIGS_DOWN:
        assert "tl.dot" in bd.to_triton_code(c)
    for c in hk.CONFIGS_UP:
        assert "tl.dot" in bu.to_triton_code(c)


def test_l_planes_layout():
    from littlebit_scratch.helion_kernels import l_planes

    l = torch.arange(1, 12, dtype=torch.float32)  # r = 11 -> nb = 2
    l8 = l_planes(l, 2)
    assert l8.shape == (8, 2)
    assert l8[3, 0] == 4 and l8[2, 1] == 11 and l8[3, 1] == 0  # l8[j, b] = l[8b + j], padding 0


def test_helion_matches_torch_backend_in_ref_mode():
    code = textwrap.dedent("""
        import torch
        from littlebit_scratch.littlebit import LittleBitLinear
        from littlebit_scratch.model import Config, TinyLM
        from littlebit_scratch.packed import PackedLittleBitLinear, pack_model, set_backend
        torch.manual_seed(0)
        for i, o, r, res, m in [(48, 32, 13, True, 5), (64, 40, 16, False, 1), (40, 72, 21, True, 3)]:
            layer = LittleBitLinear(i, o, rank=r, residual=res, bias=True)
            with torch.no_grad():
                for p in layer.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            pk = PackedLittleBitLinear(layer)
            x = torch.randn(2, m, i)
            with torch.no_grad():
                a = pk(x); pk.backend = "helion"; b = pk(x)
            assert b.shape == a.shape
            torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-4)
        model = pack_model(TinyLM(Config(vocab_size=16, seq_len=8, d_model=64, n_heads=2, n_layers=1,
                                         linear="littlebit", bpw=4.0)))
        x = torch.randint(0, 16, (2, 8))
        with torch.no_grad():
            a = model(x); b = set_backend(model, "helion")(x)
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-4)
        print("OK")
    """)
    env = dict(os.environ, HELION_INTERPRET="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-3000:]
