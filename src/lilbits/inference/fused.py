"""Fused CPU runtime for LeanLM with FP block linears (the Example 14 `fp16_size` / `fp16_width` models).

`export(model, fmt)` folds every RMSNorm gain, the kernel-1 "conv" scales and the GDN o_norm gain into the weights,
merges q/k/v/gate/a/b into one matrix and SwiGLU gate/up into one, pads to multiples of 32 and packs them tile-major
for the Rust engine in `rust/lilbits_fused` (SIMD through fearless_simd, runtime dispatch). A layer is then 4 matmuls
plus the gated delta rule, and decode is a native loop with no Python or op dispatch per byte (the main cost of
PyTorch's path at these sizes).

    fmt   weights                                     note
    f32   FP32                                        logits match PyTorch to ~1e-5
    f16   FP16
    i8    int8, per-row scale                         FP32 activations
    i4    int4, group 32, MSE-searched scales         +0.003 bpb on the d200 baseline
    w8a8  int8 x per-token int8 activations, VNNI     fastest; +0.000 bpb at d200, +0.001 at d512

`FusedLM(arrays)` runs it with numpy + ctypes only (torch is needed for `export`, not for inference), so a
deployment process does not pay for the torch import. Batch 1. Matmul kernels are picked at runtime: AVX-512 (+VNNI),
AVX2 + AVX-VNNI (Alder / Raptor Lake), AVX2, NEON + sdot (Apple M, Graviton 2+), NEON, else portable scalar.
W8A8 gives bit-identical results on all of them; the float formats agree to float rounding. `set_simd(name)` (or
LILBITS_SIMD=name) forces one, for tests and benchmarks. The engine is built with cargo on first use (cached by source
hash); LILBITS_FUSED_LIB=path loads a prebuilt library with the same C ABI instead.

    arrays = export(model, "w8a8"); np.savez("m.npz", **arrays)
    lm = FusedLM(dict(np.load("m.npz")))
    lm.prefill(prompt_bytes); text = lm.generate(256)
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_CRATE = Path(__file__).resolve().parents[3] / "rust" / "lilbits_fused"
FMT = {"f32": 0, "f16": 1, "i8": 2, "i4": 3, "w8a8": 4}
_lib: C.CDLL | None = None
SIMD = {"auto": 0, "scalar": 1, "sse4.2": 2, "avx2": 3, "avx-vnni": 4, "avx512": 5, "neon": 6, "neon-dotprod": 7}


# ---------------------------------------------------------------------------------------------------------------
# export (needs torch only to read the state dict)
# ---------------------------------------------------------------------------------------------------------------


def _up32(n: int) -> int:
    return (n + 31) // 32 * 32


_SHRINK = np.linspace(0.6, 1.0, 21)


def _best_scales(g: np.ndarray, grid: np.ndarray = _SHRINK) -> np.ndarray:
    """Per-group int4 scale: absmax/7 shrunk by the factor (from grid) with the lowest squared rounding error."""
    amax = np.abs(g).max(-1) / 7 + 1e-12
    best, err = amax, np.full(amax.shape, np.inf)
    for f in grid:
        s = amax * f
        e = ((np.clip(np.round(g / s[..., None]), -7, 7) * s[..., None] - g) ** 2).sum(-1)
        m = e < err
        best, err = np.where(m, s, best), np.where(m, e, err)
    return best


def _int8_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.abs(w).max(1) / 127 + 1e-12
    return np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32)


def _rows(w: np.ndarray, fmt: str, cpad: int) -> dict:
    """Row-major packing (the embedding table: lookups, no matmul)."""
    wp = np.zeros((w.shape[0], cpad), np.float32)
    wp[:, : w.shape[1]] = w
    if fmt == "f32":
        return {"w": wp}
    if fmt == "f16":
        return {"w": wp.astype(np.float16).view(np.uint16)}
    if fmt in ("i8", "w8a8"):
        q, s = _int8_rows(wp)
        return {"w": q, "s": s}
    g = wp.reshape(wp.shape[0], -1, 32)
    s = _best_scales(g)
    q = (np.clip(np.round(g / s[..., None]), -7, 7) + 8).astype(np.uint8)
    return {"w": (q[..., :16] | (q[..., 16:] << 4)).reshape(wp.shape[0], -1), "s": s.astype(np.float32)}


def _tiles(w: np.ndarray, fmt: str, cpad: int) -> dict:
    """Tile-major packing [tile of 32 rows][column][32] (w8a8: [tile][column / 4][32][4]) for the matmuls."""
    rows = w.shape[0]
    rpad = _up32(rows)
    wp = np.zeros((rpad, cpad), np.float32)
    wp[:rows, : w.shape[1]] = w
    nt = rpad // 32
    out: dict = {"rpad": np.array([rpad], np.int64)}
    if fmt == "f32":
        out["w"] = wp.reshape(nt, 32, cpad).transpose(0, 2, 1)
    elif fmt == "f16":
        out["w"] = wp.astype(np.float16).view(np.uint16).reshape(nt, 32, cpad).transpose(0, 2, 1)
    elif fmt == "i8":
        q, s = _int8_rows(wp)
        out["w"], out["s"] = q.reshape(nt, 32, cpad).transpose(0, 2, 1), s
    elif fmt == "w8a8":
        q, s = _int8_rows(wp)
        corr = (128 * q.astype(np.int32).sum(1)).astype(np.int32)  # removes the +128 activation offset
        out["w"] = q.reshape(nt, 32, cpad // 4, 4).transpose(0, 2, 1, 3)
        out["s"] = np.concatenate([s, corr.view(np.float32)])
    else:
        g = wp.reshape(rpad, cpad // 32, 32)
        s = _best_scales(g)
        q = (np.clip(np.round(g / s[..., None]), -7, 7) + 8).astype(np.uint8).reshape(nt, 32, cpad).transpose(0, 2, 1)
        out["w"] = q[..., :16] | (q[..., 16:] << 4)
        out["s"] = s.reshape(nt, 32, cpad // 32).transpose(0, 2, 1).astype(np.float32)
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


def export(model, fmt: str = "w8a8") -> dict[str, np.ndarray]:
    """LeanLM (FP linears, conv_size=1) -> flat dict of arrays for FusedLM (np.savez-able). Draft heads are dropped."""
    if fmt not in FMT:
        raise ValueError(f"fmt must be one of {list(FMT)}")
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    mix = model.blocks[0].mixer
    d, L, H = model.embed.embedding_dim, len(model.blocks), mix.n_heads
    if any(blk.mixer.conv_q.kernel != 1 for blk in model.blocks):
        raise ValueError("fused runtime supports conv_size=1 GDN blocks only")
    hid = sd["blocks.0.mlp.gate.weight"].shape[0]
    dp, hp = _up32(d), _up32(hid)
    out = {"meta": np.array([d, dp, H, d // H, hid, hp, L, 256, FMT[fmt]], np.int64)}

    def put(name: str, parts: dict):
        out.update({f"{name}.{k}": v for k, v in parts.items()})

    E = sd["embed.weight"]
    put("emb", _rows(E, fmt, dp))
    put("head", _tiles(E * sd["norm.weight"][None], fmt, dp))
    for l in range(L):
        g = lambda n, p=f"blocks.{l}.": sd[p + n]
        conv = lambda n, g=g: g(f"mixer.conv_{n}.conv.weight")[:, 0, :]
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
        put(f"l{l}.in", _tiles(w_in * g("n1.weight")[None], fmt, dp))
        put(f"l{l}.o", _tiles(g("mixer.o.weight") * np.tile(g("mixer.o_norm.weight"), H)[None], fmt, dp))
        w_up = np.concatenate([g("mlp.gate.weight"), g("mlp.up.weight")]) * g("n2.weight")[None]
        put(f"l{l}.up", _tiles(w_up, fmt, dp))
        put(f"l{l}.down", _tiles(g("mlp.down.weight"), fmt, hp))
        out[f"l{l}.negA"] = -np.exp(g("mixer.A_log")).astype(np.float32)
        out[f"l{l}.dt_bias"] = g("mixer.dt_bias").astype(np.float32)
    return out


# ---------------------------------------------------------------------------------------------------------------
# runtime (numpy + ctypes)
# ---------------------------------------------------------------------------------------------------------------


class _Mat(C.Structure):
    _fields_ = [
        ("rows", C.c_int),
        ("cols", C.c_int),
        ("cpad", C.c_int),
        ("fmt", C.c_int),
        ("w", C.c_void_p),
        ("s", C.c_void_p),
        ("rpad", C.c_int),
    ]


class _Layer(C.Structure):
    _fields_ = [("in_", _Mat), ("o", _Mat), ("up", _Mat), ("down", _Mat), ("negA", C.c_void_p), ("dt_bias", C.c_void_p)]


class _Model(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("d", "dpad", "H", "D", "hid", "hpad", "L", "vocab")] + [
        ("eps", C.c_float),
        ("emb", _Mat),
        ("head", _Mat),
        ("layers", C.POINTER(_Layer)),
    ]


def _cache_dir() -> Path:
    return Path(os.environ.get("LILBITS_CACHE", Path.home() / ".cache" / "lilbits"))


def _crate_hash() -> str:
    h = hashlib.sha256()
    for f in sorted([_CRATE / "Cargo.toml", _CRATE / "Cargo.lock", *(_CRATE / "src").glob("*.rs")]):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def load_kernel() -> C.CDLL:
    """Build (once, cached by source hash) and load the Rust engine. LILBITS_FUSED_LIB: load that library instead."""
    global _lib
    if _lib is not None:
        return _lib
    if os.environ.get("LILBITS_FUSED_LIB"):
        _lib = _bind(C.CDLL(os.environ["LILBITS_FUSED_LIB"]))
        return _lib
    so = _cache_dir() / f"lilbits_fused_{_crate_hash()}.so"
    if not so.exists():
        cargo = os.environ.get("CARGO") or shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
        if not Path(cargo).exists() and shutil.which(cargo) is None:
            raise RuntimeError("cargo not found: install Rust (https://rustup.rs) or set LILBITS_FUSED_LIB")
        target = _cache_dir() / "cargo-target"
        subprocess.run(
            [cargo, "build", "--release", "--manifest-path", str(_CRATE / "Cargo.toml"), "--target-dir", str(target)],
            check=True,
            capture_output=True,
        )
        so.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(target / "release" / "liblilbits_fused.so", so)
    _lib = _bind(C.CDLL(str(so)))
    if os.environ.get("LILBITS_SIMD"):
        set_simd(os.environ["LILBITS_SIMD"])
    return _lib


def set_simd(name: str) -> str:
    """Force the kernel family (a key of SIMD; "auto" = best for this CPU), process-wide. Returns the one in use."""
    got = load_kernel().set_simd(SIMD[name])
    if got < 0:
        raise ValueError(f"this CPU can't run the {name!r} kernels")
    return simd()


def simd() -> str:
    """The kernel family in use."""
    code = load_kernel().simd_isa()
    return next(k for k, v in SIMD.items() if v == code)


def available_simd() -> list[str]:
    """Every kernel family this CPU can run (best last)."""
    lib, cur = load_kernel(), load_kernel().simd_isa()
    names = [k for k, v in SIMD.items() if v and lib.set_simd(v) >= 0]
    lib.set_simd(cur)
    return names


def _bind(lib: C.CDLL) -> C.CDLL:
    fp = np.ctypeslib.ndpointer
    m = C.POINTER(_Model)
    lib.forward.argtypes = [
        m,
        fp(np.int32),
        C.c_int,
        fp(np.float32),
        fp(np.float32),
        C.c_int,
        fp(np.float32),
        C.c_int,
        C.c_int,
    ]
    lib.generate.argtypes = [m, C.c_int, C.c_int, fp(np.float32), fp(np.float32), fp(np.float32), fp(np.int32), C.c_int]
    lib.work_floats.argtypes = [m, C.c_int]
    lib.work_floats.restype = C.c_long
    lib.set_simd.argtypes = [C.c_int]
    lib.set_simd.restype = C.c_int
    lib.simd_isa.argtypes = []
    lib.simd_isa.restype = C.c_int
    return lib


def cpu_supported(fmt: str = "w8a8") -> bool:
    """Whether the AVX-512 path runs (AVX-512F/BW/VL, +VNNI for w8a8) per Linux /proc/cpuinfo. Other CPUs use
    their own kernels (see `available_simd`)."""
    try:
        flags = Path("/proc/cpuinfo").read_text().split()
    except OSError:
        return False
    need = {"avx512f", "avx512bw", "avx512vl", "f16c"} | ({"avx512_vnni"} if fmt == "w8a8" else set())
    return need <= set(flags)


class FusedLM:
    """Batch-1 inference on an `export`ed model. State lives in self.S; prefill/step/generate continue from it."""

    def __init__(self, arrays: dict[str, np.ndarray], block: int = 64, threads: int = 1):
        self.lib = load_kernel()
        self.arr = {k: np.ascontiguousarray(v) for k, v in arrays.items()}  # the C structs point into these
        d, dp, H, D, hid, hp, L, V, fmt = (int(x) for x in self.arr["meta"])
        self.fmt = next(k for k, v in FMT.items() if v == fmt)

        def mat(name: str, rows: int, cols: int, cpad: int, f: int = fmt) -> _Mat:
            s = self.arr.get(name + ".s")
            rpad = int(self.arr[name + ".rpad"][0]) if name + ".rpad" in self.arr else 0
            return _Mat(
                rows, cols, cpad, f, self.arr[name + ".w"].ctypes.data, None if s is None else s.ctypes.data, rpad
            )

        self._layers = (_Layer * L)()
        for l in range(L):
            self._layers[l] = _Layer(
                mat(f"l{l}.in", 4 * d + 2 * H, d, dp),
                mat(f"l{l}.o", d, d, dp),
                mat(f"l{l}.up", 2 * hid, d, dp),
                mat(f"l{l}.down", d, hid, hp),
                self.arr[f"l{l}.negA"].ctypes.data,
                self.arr[f"l{l}.dt_bias"].ctypes.data,
            )
        emb_fmt = FMT["i8"] if self.fmt == "w8a8" else fmt  # lookups read int8 rows
        self.m = _Model(
            d,
            dp,
            H,
            D,
            hid,
            hp,
            L,
            V,
            np.finfo(np.float32).eps,
            mat("emb", V, d, dp, emb_fmt),
            mat("head", V, d, dp),
            self._layers,
        )
        self.block, self.threads = block, threads
        self.work = np.zeros(self.lib.work_floats(C.byref(self.m), block), np.float32)
        self.S = np.zeros(L * H * D * D, np.float32)
        self.logits = np.zeros(V, np.float32)

    def weight_bytes(self) -> int:
        return sum(v.nbytes for k, v in self.arr.items() if k != "meta" and not k.endswith(".rpad"))

    def reset(self) -> None:
        self.S[:] = 0

    def prefill(self, toks) -> np.ndarray:
        """Run bytes from the current state; returns the last byte's next-byte logits."""
        t = np.ascontiguousarray(toks, np.int32)
        self.lib.forward(C.byref(self.m), t, len(t), self.S, self.logits, 0, self.work, self.block, self.threads)
        return self.logits

    def logits_all(self, toks) -> np.ndarray:
        """(T, 256) next-byte logits for every position, from the current state."""
        t = np.ascontiguousarray(toks, np.int32)
        out = np.zeros((len(t), self.m.vocab), np.float32)
        self.lib.forward(C.byref(self.m), t, len(t), self.S, out, 1, self.work, self.block, self.threads)
        return out

    def generate(self, n: int) -> list[int]:
        """Greedy n bytes after a prefill (same convention as LeanLM.generate: the first is the prefill's argmax)."""
        first = int(self.logits.argmax())
        if n <= 1:
            return [first][:n]
        out = np.zeros(n - 1, np.int32)
        self.lib.generate(C.byref(self.m), first, n - 1, self.S, self.logits, self.work, out, self.threads)
        return [first, *out.tolist()]
