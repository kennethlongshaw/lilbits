"""Lookup-table (T-MAC style) CPU kernels for packed LittleBit and BitNet decode.

Packed layers in `packed.py` unpack every sign before each matmul, so single-token decode pays for the full
unpack and ends up slower than a dense FP matmul. A lookup-table kernel never unpacks. It groups 4 inputs,
builds each group's 16 possible signed sums once per token, and then computes every output as one table lookup
per group (T-MAC, Wei et al. 2024; bitnet.cpp's TL kernels, Wang et al. 2025). The weights stay at 1 bit
(LittleBit signs) or 2 bits (BitNet ternary) in memory the whole time.

    y = S x,  S in {-1,+1}^(out x in):   T_g[c] = sum_j (+-x)[4g+j],   y_o = sum_g T_g[code(o, g)]

LittleBit:  y = sum_paths h * (S_U (l * (S_V^T (g * x))))  -> two sign LUT matmuls per path.
BitNet:     W in {-1,0,+1} is written as (P + Q) / 2 with P, Q in {-1,+1} (+1 = (+1,+1), -1 = (-1,-1),
            0 = (+1,-1)), so W x = (P x + Q x) / 2: two sign LUT matmuls that share the same tables.
            2 bits per weight, the same as bitnet.cpp's I2_S layout. Activations are absmax int8-quantized
            values, as in training.

The C kernel (`lut_kernel.c`) is compiled on first use with the system C compiler (-O3 -march=native -fopenmp)
and loaded with ctypes. AVX-512 and AVX2 paths use in-register table lookups (vpermps); other CPUs use a scalar
loop. Without a compiler, `lut_matmul` falls back to a pure-torch reference (correct but slow).

Use `pack_lut(model)` to swap LittleBitLinear / PackedLittleBitLinear / BitLinear / PackedBitLinear for LUT layers.
The kernel is for decode (few tokens). For more than `max_lut_tokens` rows, the layer unpacks once and runs a
dense matmul instead, which is faster for prefill.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from lilbits.inference.packed import PackedLittleBitLinear, persistent_bytes, set_scales, unpack_signs
from lilbits.layers.bitnet import BitLinear, PackedBitLinear, act_quant
from lilbits.layers.littlebit import LittleBitLinear
from lilbits.layers.loop_scaled import LoopScaledLinear

_SRC = Path(__file__).with_name("lut_kernel.c")
_lib: ctypes.CDLL | None = None
_lib_error: str | None = None


def _cache_dir() -> Path:
    return Path(os.environ.get("LILBITS_CACHE", Path.home() / ".cache" / "lilbits"))


def _load() -> ctypes.CDLL | None:
    """Compile (once, cached by source hash) and load the kernel. None if no C compiler is available."""
    global _lib, _lib_error
    if _lib is not None or _lib_error is not None:
        return _lib
    src = _SRC.read_bytes()
    flags = ["-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-std=c11"]
    key = hashlib.sha256(src + " ".join(flags).encode()).hexdigest()[:16]
    so = _cache_dir() / f"lut_kernel_{key}.so"
    try:
        if not so.exists():
            cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
            if cc is None:
                raise RuntimeError("no C compiler found (set CC)")
            so.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / so.name
                subprocess.run([cc, *flags, str(_SRC), "-o", str(out)], check=True, capture_output=True)
                shutil.move(str(out), so)
        lib = ctypes.CDLL(str(so))
    except (RuntimeError, OSError, subprocess.CalledProcessError) as e:
        _lib_error = getattr(e, "stderr", None) or str(e)
        return None
    i64, ptr = ctypes.c_int64, ctypes.c_void_p
    lib.lut_matmul.argtypes = [ptr, ptr, ptr, ptr, i64, i64, i64, i64, ctypes.c_int]
    lib.lut_matmul_scalar.argtypes = [ptr, ptr, ptr, ptr, i64, i64, i64, i64]
    lib.lut_ternary.argtypes = [ptr, ptr, ptr, ptr, ptr, i64, i64, i64, i64, ctypes.c_float, ctypes.c_int]
    lib.lut_littlebit.argtypes = [ptr] * 12 + [i64] * 9 + [ctypes.c_int]
    lib.lut_simd_width.restype = ctypes.c_int
    _lib = lib
    return lib


def kernel_info() -> dict:
    lib = _load()
    if lib is None:
        return {"compiled": False, "error": _lib_error, "simd": 0}
    return {"compiled": True, "simd": int(lib.lut_simd_width())}


def pack_sign_codes(S: torch.Tensor) -> torch.Tensor:
    """(out, in) real -> (ceil(out/32), ceil(in/4), 16) uint8 LUT codes of sign(S) (>= 0 -> +1).

    Inputs are padded to a multiple of 4 (padding signs are +1, and padded x is 0, so they add nothing) and
    outputs to a multiple of 32 (padded rows are dropped after the kernel).
    """
    O, I = S.shape
    bits = (S >= 0).to(torch.uint8)
    bits = F.pad(bits, (0, (-I) % 4, 0, (-O) % 32), value=1)
    Op, Ip = bits.shape
    nib = (bits.view(Op, Ip // 4, 4) << torch.arange(4, dtype=torch.uint8)).sum(-1, dtype=torch.uint8)  # Op, G
    nib = nib.view(Op // 32, 2, 16, Ip // 4).permute(0, 3, 2, 1)  # OB, G, 16 (k), 2 (low / high half)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()


def unpack_sign_codes(codes: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Inverse of pack_sign_codes: (out, in) of +-1 float32."""
    OB, G, _ = codes.shape
    nib = torch.stack([codes & 15, codes >> 4], -1).permute(0, 3, 2, 1).reshape(OB * 32, G)  # Op, G
    bits = (nib.unsqueeze(-1) >> torch.arange(4, dtype=torch.uint8)) & 1  # Op, G, 4
    return (bits.reshape(OB * 32, G * 4).float() * 2 - 1)[:out_features, :in_features]


def lut_matmul_ref(x: torch.Tensor, codes: torch.Tensor, out_features: int) -> torch.Tensor:
    """Pure-torch reference of the kernel: same tables and lookups, vectorized with gather."""
    N, I = x.shape
    OB, G, _ = codes.shape
    xp = F.pad(x.float(), (0, 4 * G - I)).view(N, G, 4)
    c = torch.arange(16)
    signs = ((c[:, None] >> torch.arange(4)) & 1).float() * 2 - 1  # 16, 4
    T = xp @ signs.t()  # N, G, 16
    idx = torch.stack([codes & 15, codes >> 4], -1).permute(0, 3, 2, 1).reshape(OB * 32, G).long()  # Op, G
    y = T.gather(2, idx.t().unsqueeze(0).expand(N, G, OB * 32)).sum(1)  # N, Op
    return y[:, :out_features]


def lut_matmul(
    x: torch.Tensor, codes: torch.Tensor, out_features: int, threads: int | None = None, scalar: bool = False
) -> torch.Tensor:
    """y = x @ S^T for the +-1 matrix S encoded in `codes`. x: (N, in) -> (N, out) float32."""
    lib = _load()
    if lib is None or x.device.type != "cpu":
        return lut_matmul_ref(x, codes, out_features)
    N, I = x.shape
    OB, G, _ = codes.shape
    xp = x.float().contiguous()
    if 4 * G != I:
        xp = F.pad(xp, (0, 4 * G - I))
    y = torch.empty(N, out_features, dtype=torch.float32)
    tables = torch.empty(N, G, 16, dtype=torch.float32)
    if scalar:
        lib.lut_matmul_scalar(xp.data_ptr(), codes.data_ptr(), tables.data_ptr(), y.data_ptr(), N, G, OB, out_features)
    else:
        nt = threads or torch.get_num_threads()
        lib.lut_matmul(xp.data_ptr(), codes.data_ptr(), tables.data_ptr(), y.data_ptr(), N, G, OB, out_features, nt)
    return y


def _f32(m: nn.Module, *names: str) -> list[torch.Tensor]:
    """FP32 copies of FP16 scale buffers, cached on the module (not saved; recomputed after load)."""
    cache = m.__dict__.setdefault("_f32", {})
    out = []
    for n in names:
        b = getattr(m, n)
        if n not in cache or cache[n][0] is not b:
            cache[n] = (b, None if b is None else b.float().contiguous())
        out.append(cache[n][1])
    return out


def _ptr(t: torch.Tensor | None) -> int | None:
    return None if t is None else t.data_ptr()


def _sign(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t >= 0, 1.0, -1.0)


class LUTLittleBitLinear(nn.Module):
    """Inference-only LittleBit layer on the LUT kernel. Same 1-bit storage as PackedLittleBitLinear."""

    V_codes: torch.Tensor  # P, ceil(r/32), ceil(in/4), 16: S_V^T, (r x in)
    U_codes: torch.Tensor  # P, ceil(out/32), ceil(r/4), 16: S_U, (out x r)
    h: torch.Tensor
    g: torch.Tensor
    l: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, layer: LittleBitLinear | PackedLittleBitLinear, max_lut_tokens: int = 8):
        super().__init__()
        self.in_features, self.out_features, self.rank = layer.in_features, layer.out_features, layer.rank
        self.max_lut_tokens = max_lut_tokens
        with torch.no_grad():
            if isinstance(layer, LittleBitLinear):
                paths = list(layer.paths)
                Us = [_sign(p.U) for p in paths]
                Vs = [_sign(p.V) for p in paths]
                h, g, l = (torch.stack([getattr(p, k) for p in paths]) for k in "hgl")
                bias, bits = layer.bias, layer.scale_bits
            else:
                P, r = layer.paths, layer.rank
                Us = [unpack_signs(layer.U_bits[i], r) for i in range(P)]
                Vs = [unpack_signs(layer.V_bits[i], r) for i in range(P)]
                h, g, l, bias, bits = layer.h, layer.g, layer.l, layer.bias, getattr(layer, "scale_bits", 16)
            self.register_buffer("V_codes", torch.stack([pack_sign_codes(V.t()) for V in Vs]))
            self.register_buffer("U_codes", torch.stack([pack_sign_codes(U) for U in Us]))
            set_scales(self, h.float(), g.float(), l.float(), bits)
            self.register_buffer("bias", None if bias is None else bias.detach().half().clone())

    @property
    def paths(self) -> int:
        return self.U_codes.shape[0]

    def weight(self, scales: tuple | None = None) -> torch.Tensor:
        """Dense effective weight (prefill path and tests). scales: optional (g, l, h) FP32 overrides."""
        g, l, h = scales if scales is not None else (self.g.float(), self.l.float(), self.h.float())
        W = torch.zeros(self.out_features, self.in_features)
        for i in range(self.paths):
            Su = unpack_sign_codes(self.U_codes[i], self.out_features, self.rank)
            Sv = unpack_sign_codes(self.V_codes[i], self.rank, self.in_features)
            W += h[i][:, None] * ((Su * l[i]) @ Sv) * g[i][None, :]
        return W

    def _float_signs(self):
        """Per path (S_V (in x r), S_U^T (r x out)) as +-1 floats, for prefill. Unpacked on every call and freed
        after use (one path at a time), never cached: a cache would hold rank * (in + out) FP32 values per path,
        ~32x the packed weights, for the life of the model."""
        for i in range(self.paths):
            Su = unpack_sign_codes(self.U_codes[i], self.out_features, self.rank)
            Sv = unpack_sign_codes(self.V_codes[i], self.rank, self.in_features)
            yield Sv.t().contiguous(), Su.t().contiguous()

    def forward(self, x: torch.Tensor, scales: tuple | None = None, key=None) -> torch.Tensor:
        """scales: optional (g, l, h) FP32 (P, n) overrides, e.g. per-loop effective scales. `key` is accepted
        for callers that name scale sets; prefill shares one set of unpacked signs across them."""
        lead, dt = x.shape[:-1], x.dtype
        x2 = x.reshape(-1, self.in_features).float()
        if x2.shape[0] > self.max_lut_tokens:  # prefill: low-rank matmuls over the signs, unpacked once
            g, l, h = scales if scales is not None else (self.g.float(), self.l.float(), self.h.float())
            y = torch.zeros(x2.shape[0], self.out_features)
            for i, (SvT, SuT) in enumerate(self._float_signs()):
                y += (((x2 * g[i]) @ SvT) * l[i]) @ SuT * h[i]
        elif (lib := _load()) is not None and x2.device.type == "cpu":
            g, l, h, bias = _f32(self, "g", "l", "h", "bias")
            if scales is not None:
                g, l, h = scales
            x2 = x2.contiguous()
            N, (P, OBr, Gi, _), (_, OBo, Gr, _) = x2.shape[0], self.V_codes.shape, self.U_codes.shape
            xs, ti = torch.empty(4 * Gi), torch.empty(Gi * 16)
            t, tr = torch.empty(4 * Gr), torch.empty(Gr * 16)
            y = torch.empty(N, self.out_features)
            lib.lut_littlebit(
                *(_ptr(v) for v in (x2, self.V_codes, self.U_codes, g, l, h, bias, xs, ti, t, tr, y)),
                *(N, P, self.in_features, Gi, OBr, self.rank, Gr, OBo, self.out_features),
                torch.get_num_threads(),
            )
            return y.to(dt).view(*lead, self.out_features)
        else:
            g, l, h = scales if scales is not None else (self.g.float(), self.l.float(), self.h.float())
            y = torch.zeros(x2.shape[0], self.out_features)
            for i in range(self.paths):
                t = lut_matmul(x2 * g[i], self.V_codes[i], self.rank)
                y += lut_matmul(t * l[i], self.U_codes[i], self.out_features) * h[i]
        if self.bias is not None:
            y += self.bias.float()
        return y.to(dt).view(*lead, self.out_features)

    def storage_bytes(self) -> int:
        return persistent_bytes(self)


class LUTBitLinear(nn.Module):
    """Inference-only BitNet b1.58 layer on the LUT kernel: ternary as two sign planes, 2 bits per weight."""

    P_codes: torch.Tensor
    Q_codes: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, layer: BitLinear | PackedBitLinear, max_lut_tokens: int = 8):
        super().__init__()
        self.in_features, self.out_features = layer.in_features, layer.out_features
        self.act_bits, self.norm, self.max_lut_tokens = layer.act_bits, layer.norm, max_lut_tokens
        with torch.no_grad():
            if isinstance(layer, PackedBitLinear):
                scale = layer.scale.float()
                q = (layer.weight() / scale).round()
            else:
                w = layer.weight.detach()
                scale = w.abs().mean().clamp(min=1e-5)
                q = (w / scale).round().clamp(-1, 1)
            # +1 -> (+1, +1), -1 -> (-1, -1), 0 -> (+1, -1)
            self.register_buffer("P_codes", pack_sign_codes(torch.where(q >= 0, 1.0, -1.0)))
            self.register_buffer("Q_codes", pack_sign_codes(torch.where(q > 0, 1.0, -1.0)))
            self.register_buffer("scale", scale.half())
            self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().half().clone())

    def weight(self) -> torch.Tensor:
        P = unpack_sign_codes(self.P_codes, self.out_features, self.in_features)
        Q = unpack_sign_codes(self.Q_codes, self.out_features, self.in_features)
        return (P + Q) * (0.5 * self.scale.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead, dt = x.shape[:-1], x.dtype
        x2 = x.reshape(-1, self.in_features).float()
        x2 = F.rms_norm(x2, (self.in_features,)) if self.norm else x2
        x2 = act_quant(x2, self.act_bits) if self.act_bits else x2
        if x2.shape[0] > self.max_lut_tokens:
            cache = self.__dict__.get("_dense")
            if cache is None:
                cache = self.__dict__["_dense"] = self.weight()
            y = x2 @ cache.t()
        elif (lib := _load()) is not None and x2.device.type == "cpu":
            x2 = x2.contiguous()
            OB, G, _ = self.P_codes.shape
            if 4 * G != self.in_features:
                x2 = F.pad(x2, (0, 4 * G - self.in_features))
            y = torch.empty(x2.shape[0], self.out_features)
            tables = torch.empty(x2.shape[0], G, 16)
            lib.lut_ternary(
                *(_ptr(v) for v in (x2, self.P_codes, self.Q_codes, tables, y)),
                *(x2.shape[0], G, OB, self.out_features),
                float(self.scale),
                torch.get_num_threads(),
            )
        else:
            P = lut_matmul(x2, self.P_codes, self.out_features)
            y = (P + lut_matmul(x2, self.Q_codes, self.out_features)) * (0.5 * self.scale.float())
        if self.bias is not None:
            y += self.bias.float()
        return y.to(dt).view(*lead, self.out_features)

    def storage_bytes(self) -> int:
        return persistent_bytes(self)


class LUTLoopScaledLinear(nn.Module):
    """Inference form of `LoopScaledLinear`: one LUT layer (signs stored once) plus per-loop FP16 scales.

    Over LittleBit, loop i runs the fused kernel with effective scales (g_p g_i, l_p l_i, h_p h_i), built once per
    loop and cached. Over other bases (BitNet LUT, FP), it computes base(x * g_i) * h_i."""

    g: torch.Tensor
    h: torch.Tensor
    l: torch.Tensor | None

    def __init__(self, layer: LoopScaledLinear, max_lut_tokens: int = 8):
        super().__init__()
        base = layer.base
        self.in_features, self.out_features, self.n_loops = layer.in_features, layer.out_features, layer.n_loops
        self.loop = 0
        if isinstance(base, LittleBitLinear):
            self.base: nn.Module = LUTLittleBitLinear(base, max_lut_tokens)
        elif isinstance(base, BitLinear):
            self.base = LUTBitLinear(base, max_lut_tokens)
        else:
            self.base = base
        self.register_buffer("g", layer.g.detach().half().clone())
        self.register_buffer("h", layer.h.detach().half().clone())
        self.register_buffer("l", None if layer.l is None else layer.l.detach().half().clone())

    def _scales(self, i: int) -> tuple:
        cache = self.__dict__.setdefault("_loop_scales", {})
        if i not in cache:
            b = self.base
            assert isinstance(b, LUTLittleBitLinear) and self.l is not None
            g, l, h = (getattr(b, n).float() for n in "glh")
            cache[i] = tuple(
                t.contiguous() for t in (g * self.g[i].float(), l * self.l[i].float(), h * self.h[i].float())
            )
        return cache[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i = self.loop
        if isinstance(self.base, LUTLittleBitLinear):
            y = self.base(x, self._scales(i), key=i)
            if self.base.bias is not None:  # the loop's h_i scales the bias too; the kernel added it unscaled
                y = y + (self.base.bias.float() * (self.h[i].float() - 1)).to(y.dtype)
            return y
        return self.base(x * self.g[i].to(x.dtype)) * self.h[i].to(x.dtype)

    def storage_bytes(self) -> int:
        fn = getattr(self.base, "storage_bytes", None)
        base = fn() if callable(fn) else persistent_bytes(self.base)
        return base + sum(t.numel() * t.element_size() for t in (self.g, self.h, self.l) if t is not None)


@torch.no_grad()
def pack_lut(model: nn.Module, max_lut_tokens: int = 8) -> nn.Module:
    """In place: LittleBit / BitNet linears (latent or packed, plain or loop-scaled) -> LUT layers. Returns the
    model in eval mode."""
    for name, child in list(model.named_children()):
        if isinstance(child, LoopScaledLinear):
            setattr(model, name, LUTLoopScaledLinear(child, max_lut_tokens))
        elif isinstance(child, (LittleBitLinear, PackedLittleBitLinear)):
            setattr(model, name, LUTLittleBitLinear(child, max_lut_tokens))
        elif isinstance(child, (BitLinear, PackedBitLinear)):
            setattr(model, name, LUTBitLinear(child, max_lut_tokens))
        else:
            pack_lut(child, max_lut_tokens)
    return model.eval()
