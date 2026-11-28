"""Fused CPU runtime for SpaceHNet (models/space_hnet.py) on the Rust engine (`rust/lilbits_fused/src/hnet.rs`).

`export(model, fmt)` packs the three GDN stacks the way `fused.export` packs LeanLM's blocks (norm gains, conv scales
and o_norm folded in; q/k/v/gate/a/b and SwiGLU gate/up merged; tile-major), plus the up / down / residual
projections (n_up, n_down folded into up / down) and the tied embedding / head (final norm folded into the head).
`FusedHNet(arrays)` runs it with numpy + ctypes only; the main stack steps only on patch boundaries.

    arrays = export(model, "w8a8"); np.savez("m.npz", **arrays)
    lm = FusedHNet(dict(np.load("m.npz")))
    lm.prefill(prompt_bytes); out = lm.generate(256)
"""

from __future__ import annotations

import ctypes as C

import numpy as np

from lilbits.inference.fused import FMT, _Layer, _Mat, _Model, _rows, _tiles, _up32, load_kernel

ALIGN = {"end": 0, "start": 1}


def _stack(sd: dict, prefix: str, n: int, d: int, H: int, fmt: str, out: dict, name: str) -> int:
    dp = _up32(d)
    hid = sd[f"{prefix}0.mlp.gate.weight"].shape[0]
    hp = _up32(hid)
    for l in range(n):
        g = lambda k, p=f"{prefix}{l}.": sd[p + k]
        conv = lambda k, g=g: g(f"mixer.conv_{k}.conv.weight")[:, 0, :]
        w_in = np.concatenate(
            [
                g("mixer.q.weight") * conv("q"),
                g("mixer.k.weight") * conv("k"),
                g("mixer.v.weight") * conv("v"),
                g("mixer.gate.weight"),
                g("mixer.a.weight"),
                g("mixer.b.weight"),
            ]
        )
        put = lambda k, parts, l=l: out.update({f"{name}.l{l}.{k}.{x}": v for x, v in parts.items()})
        put("in", _tiles(w_in * g("n1.weight")[None], fmt, dp))
        put("o", _tiles(g("mixer.o.weight") * np.tile(g("mixer.o_norm.weight"), H)[None], fmt, dp))
        put("up", _tiles(np.concatenate([g("mlp.gate.weight"), g("mlp.up.weight")]) * g("n2.weight")[None], fmt, dp))
        put("down", _tiles(g("mlp.down.weight"), fmt, hp))
        out[f"{name}.l{l}.negA"] = -np.exp(g("mixer.A_log")).astype(np.float32)
        out[f"{name}.l{l}.dt_bias"] = g("mixer.dt_bias").astype(np.float32)
    out[f"{name}.meta"] = np.array([d, dp, H, d // H, hid, hp, n, 256, FMT[fmt]], np.int64)
    return hid


def export(model, fmt: str = "w8a8") -> dict[str, np.ndarray]:
    if fmt not in FMT:
        raise ValueError(f"fmt must be one of {list(FMT)}")
    c = model.cfg
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    do, dm = c.d_outer, c.d_main
    po, pm = _up32(do), _up32(dm)
    out: dict[str, np.ndarray] = {}
    _stack(sd, "encoder.", c.n_enc, do, c.heads_outer, fmt, out, "enc")
    _stack(sd, "main.", c.n_main, dm, c.heads_main, fmt, out, "main")
    _stack(sd, "decoder.", c.n_dec, do, c.heads_outer, fmt, out, "dec")
    E = sd["embed.weight"]
    out.update({f"emb.{k}": v for k, v in _rows(E, fmt, po).items()})
    out.update({f"head.{k}": v for k, v in _tiles(E * sd["norm.weight"][None], fmt, po).items()})
    out.update({f"up.{k}": v for k, v in _tiles(sd["up.weight"] * sd["n_up.weight"][None], fmt, po).items()})
    out.update({f"down.{k}": v for k, v in _tiles(sd["down.weight"] * sd["n_down.weight"][None], fmt, pm).items()})
    out.update({f"res.{k}": v for k, v in _tiles(sd["residual.weight"], fmt, po).items()})
    out["hmeta"] = np.array([ALIGN[c.align], c.max_patch, FMT[fmt]], np.int64)
    return out


class _HModel(C.Structure):
    _fields_ = [
        ("enc", _Model),
        ("main", _Model),
        ("dec", _Model),
        ("up", _Mat),
        ("down", _Mat),
        ("res", _Mat),
        ("align", C.c_int),
        ("max_patch", C.c_int),
    ]


def _bind_hnet(lib: C.CDLL) -> C.CDLL:
    if getattr(lib, "_hnet_bound", False):
        return lib
    fp = np.ctypeslib.ndpointer
    m = C.POINTER(_HModel)
    lib.hnet_work_floats.argtypes = [m, C.c_int]
    lib.hnet_work_floats.restype = C.c_long
    lib.hnet_state_floats.argtypes = [m]
    lib.hnet_state_floats.restype = C.c_long
    f32, i32 = fp(np.float32), fp(np.int32)
    lib.hnet_forward.argtypes = [m, i32, C.c_int, f32, f32, i32, f32, C.c_int, f32, C.c_int]
    lib.hnet_forward.restype = C.c_long
    lib.hnet_generate.argtypes = [m, C.c_int, C.c_int, f32, f32, i32, f32, f32, i32]
    lib.hnet_generate.restype = C.c_long
    lib._hnet_bound = True
    return lib


class FusedHNet:
    """Batch-1, one-thread inference on an exported SpaceHNet. State (GDN states, current patch row, boundary
    state) lives on the object; prefill / generate continue from it."""

    def __init__(self, arrays: dict[str, np.ndarray], block: int = 64):
        self.lib = _bind_hnet(load_kernel())
        self.arr = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
        align, max_patch, fmt = (int(x) for x in self.arr["hmeta"])
        self.fmt = next(k for k, v in FMT.items() if v == fmt)

        def mat(name: str, rows: int, cols: int, cpad: int, f: int = fmt) -> _Mat:
            s = self.arr.get(name + ".s")
            rpad = int(self.arr[name + ".rpad"][0]) if name + ".rpad" in self.arr else 0
            return _Mat(
                rows, cols, cpad, f, self.arr[name + ".w"].ctypes.data, None if s is None else s.ctypes.data, rpad
            )

        self._keep = []

        def model(name: str, with_io: bool) -> _Model:
            d, dp, H, D, hid, hp, L, V, _ = (int(x) for x in self.arr[f"{name}.meta"])
            layers = (_Layer * L)()
            for l in range(L):
                p = f"{name}.l{l}"
                layers[l] = _Layer(
                    mat(f"{p}.in", 4 * d + 2 * H, d, dp),
                    mat(f"{p}.o", d, d, dp),
                    mat(f"{p}.up", 2 * hid, d, dp),
                    mat(f"{p}.down", d, hid, hp),
                    self.arr[f"{p}.negA"].ctypes.data,
                    self.arr[f"{p}.dt_bias"].ctypes.data,
                )
            self._keep.append(layers)
            emb_fmt = FMT["i8"] if self.fmt == "w8a8" else fmt
            emb = mat("emb", V, d, dp, emb_fmt) if with_io else _Mat()
            head = mat("head", V, d, dp) if with_io else _Mat()
            return _Model(d, dp, H, D, hid, hp, L, V, np.finfo(np.float32).eps, emb, head, layers)

        enc, main, dec = model("enc", True), model("main", False), model("dec", False)
        do, dm = enc.d, main.d
        self.m = _HModel(
            enc,
            main,
            dec,
            mat("up", dm, do, enc.dpad),
            mat("down", do, dm, main.dpad),
            mat("res", do, do, enc.dpad),
            align,
            max_patch,
        )
        self.block = block
        self.work = np.zeros(self.lib.hnet_work_floats(C.byref(self.m), block), np.float32)
        self.S = np.zeros(self.lib.hnet_state_floats(C.byref(self.m)), np.float32)
        self.zd = np.zeros(enc.dpad, np.float32)
        self.bst = np.array([-1, -1, 0], np.int32)
        self.logits = np.zeros(256, np.float32)
        self.main_steps = 0

    def weight_bytes(self) -> int:
        return sum(v.nbytes for k, v in self.arr.items() if not k.endswith(("meta", ".rpad")))

    def state_bytes(self) -> int:
        return self.S.nbytes + self.zd.nbytes

    def reset(self) -> None:
        self.S[:] = 0
        self.zd[:] = 0
        self.bst[:] = (-1, -1, 0)
        self.main_steps = 0

    def prefill(self, toks) -> np.ndarray:
        t = np.ascontiguousarray(toks, np.int32)
        self.main_steps += self.lib.hnet_forward(
            C.byref(self.m), t, len(t), self.S, self.zd, self.bst, self.logits, 0, self.work, self.block
        )
        return self.logits

    def logits_all(self, toks) -> np.ndarray:
        t = np.ascontiguousarray(toks, np.int32)
        out = np.zeros((len(t), 256), np.float32)
        self.main_steps += self.lib.hnet_forward(
            C.byref(self.m), t, len(t), self.S, self.zd, self.bst, out, 1, self.work, self.block
        )
        return out

    def generate(self, n: int) -> list[int]:
        """Greedy n bytes after a prefill (same convention as FusedLM.generate)."""
        first = int(self.logits.argmax())
        if n <= 1:
            return [first][:n]
        out = np.zeros(n - 1, np.int32)
        self.main_steps += self.lib.hnet_generate(
            C.byref(self.m), first, n - 1, self.S, self.zd, self.bst, self.logits, self.work, out
        )
        return [first, *out.tolist()]
