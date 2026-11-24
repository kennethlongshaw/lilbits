import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXAMPLE


@pytest.mark.parametrize("name", ["03_state_rollback", "02_recurrent_mixers"])
def test_train_then_separate_test_and_infer(tmp_path, name):
    from lilbits.runner import run

    ex = load_example(name)
    ck = str(tmp_path / "ck.pt")
    out = run(
        ex, ["--mode", "train", "--steps", "2", "--ckpt", ck] + (["--mixers", "gdn"] if name.startswith("02") else [])
    )
    assert "train" in out and "test" not in out and Path(ck).exists()
    t = run(ex, ["--mode", "test", "--ckpt", ck])
    assert "test" in t and "train" not in t and "infer" not in t
    i = run(ex, ["--mode", "infer", "--ckpt", ck])
    assert "infer" in i and i["size"]["checkpoint_file_bytes"] > 0


def test_all_mode_matches_separate_test(tmp_path):
    from lilbits.runner import run

    ex = load_example("03_state_rollback")
    ck = str(tmp_path / "ck.pt")
    a = run(ex, ["--steps", "2", "--ckpt", ck])
    b = run(ex, ["--mode", "test", "--ckpt", ck])
    assert a["test"]["held_out_loss"] == pytest.approx(b["test"]["held_out_loss"], rel=1e-5)
