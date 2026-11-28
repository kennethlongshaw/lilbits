//! Fused CPU inference for lilbits' LeanLM (GDN byte LM with FP block linears, conv_size = 1).
//!
//! Loaded by `lilbits.inference.fused` through a C ABI (`forward`, `generate`, `work_floats`, `set_simd`; it
//! replaced the original C kernel with the same ABI and weight layouts), on arrays written by
//! `lilbits.inference.fused.export`. SIMD goes through `fearless_simd`: the forward pass is generic over `Simd` and
//! dispatched at runtime (so the elementwise code and the gated delta rule are compiled for AVX-512 / AVX2 / SSE4.2 /
//! NEON / scalar). The matmul tiles have AVX-512 (+VNNI), AVX2 (+AVX-VNNI) and NEON (+sdot) kernels and a portable
//! scalar one; `isa.rs` picks them per CPU.
//!
//! `hnet.rs` chains three GDN stacks into the space-patched H-Net (`hnet_forward`, `hnet_generate`), `bundle.rs` loads
//! a one-file `.lbm` bundle for the standalone CLI (`src/bin/lilbits.rs`), which runs either model without Python.
//!
//! Weight formats: 0 f32, 1 f16, 2 int8 (per-row scale), 3 int4 (group 32), 4 int8 x int8 activations (VNNI).
//! The embedding table is row-major (lookups); every other matrix is tile-major: [32-row tile][column][32]
//! (W8A8: [tile][column / 4][32][4]). `unsafe` turns the FFI pointers into slices, splits shared buffers across
//! threads, and calls the kernels whose CPU features `isa.rs` detected at runtime.

use core::ffi::c_void;
use fearless_simd::{Level, Simd, dispatch, f32x16, i32x16, prelude::*};

pub mod bundle;
mod hnet;
mod isa;
#[cfg(target_arch = "x86_64")]
mod kern_avx2;
#[cfg(target_arch = "aarch64")]
mod kern_neon;
mod pool;
mod tiles;

use pool::{SyncPtr, split};

// ---------------------------------------------------------------------------------------------------------------
// FFI structs (layout shared with fused.py's ctypes structures)
// ---------------------------------------------------------------------------------------------------------------

#[repr(C)]
pub struct Mat {
    rows: i32,
    cols: i32,
    cpad: i32,
    fmt: i32,
    w: *const c_void,
    s: *const f32,
    rpad: i32,
}

#[repr(C)]
pub struct Layer {
    in_: Mat,
    o: Mat,
    up: Mat,
    down: Mat,
    neg_a: *const f32,
    dt_bias: *const f32,
}

#[repr(C)]
pub struct Model {
    d: i32,
    dpad: i32,
    h: i32,
    dh: i32,
    hid: i32,
    hpad: i32,
    l: i32,
    vocab: i32,
    eps: f32,
    emb: Mat,
    head: Mat,
    layers: *const Layer,
}

// ---------------------------------------------------------------------------------------------------------------
// Safe views
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum W<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    I8(&'a [i8]),
    I4(&'a [u8]),
    W8A8(&'a [i8]),
}

#[derive(Clone, Copy)]
pub struct MatV<'a> {
    pub rows: usize,
    pub cpad: usize,
    /// rows padded to 32; 0 = row-major (embedding lookups)
    pub rpad: usize,
    pub w: W<'a>,
    pub s: &'a [f32],
}

pub struct LayerV<'a> {
    pub inp: MatV<'a>,
    pub o: MatV<'a>,
    pub up: MatV<'a>,
    pub down: MatV<'a>,
    pub neg_a: &'a [f32],
    pub dt_bias: &'a [f32],
}

pub struct ModelV<'a> {
    pub d: usize,
    pub dp: usize,
    pub h: usize,
    pub dh: usize,
    pub hid: usize,
    pub hp: usize,
    pub vocab: usize,
    pub eps: f32,
    pub emb: MatV<'a>,
    pub head: MatV<'a>,
    pub layers: Vec<LayerV<'a>>,
}

/// # Safety
/// `m` must describe arrays of the sizes `export` writes for its rows / cpad / rpad / fmt, alive for 'a.
unsafe fn view_mat<'a>(m: &Mat) -> MatV<'a> {
    let (rows, cpad, rpad) = (m.rows as usize, m.cpad as usize, m.rpad.max(0) as usize);
    let nr = if rpad > 0 { rpad } else { rows };
    let n = nr * cpad;
    let p = m.w;
    // Safety: the caller guarantees these lengths.
    let (w, ns) = unsafe {
        match m.fmt {
            0 => (W::F32(core::slice::from_raw_parts(p as *const f32, n)), 0),
            1 => (W::F16(core::slice::from_raw_parts(p as *const u16, n)), 0),
            2 => (W::I8(core::slice::from_raw_parts(p as *const i8, n)), nr),
            3 => (
                W::I4(core::slice::from_raw_parts(p as *const u8, n / 2)),
                nr * (cpad / 32),
            ),
            _ => (W::W8A8(core::slice::from_raw_parts(p as *const i8, n)), 2 * nr),
        }
    };
    let s = if ns > 0 {
        unsafe { core::slice::from_raw_parts(m.s, ns) }
    } else {
        &[][..]
    };
    MatV { rows, cpad, rpad, w, s }
}

/// # Safety
/// `m` must be a model whose pointers come from `lilbits.inference.fused` (arrays alive for 'a).
unsafe fn view_model<'a>(m: &'a Model) -> ModelV<'a> {
    let (h, l) = (m.h as usize, m.l as usize);
    let layers = unsafe { core::slice::from_raw_parts(m.layers, l) };
    let layers = layers
        .iter()
        .map(|ly| unsafe {
            LayerV {
                inp: view_mat(&ly.in_),
                o: view_mat(&ly.o),
                up: view_mat(&ly.up),
                down: view_mat(&ly.down),
                neg_a: core::slice::from_raw_parts(ly.neg_a, h),
                dt_bias: core::slice::from_raw_parts(ly.dt_bias, h),
            }
        })
        .collect();
    ModelV {
        d: m.d as usize,
        dp: m.dpad as usize,
        h,
        dh: m.dh as usize,
        hid: m.hid as usize,
        hp: m.hpad as usize,
        vocab: m.vocab as usize,
        eps: m.eps,
        emb: unsafe { view_mat(&m.emb) },
        head: unsafe { view_mat(&m.head) },
        layers,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Scalar helpers (compiled per SIMD level through the generic forward)
// ---------------------------------------------------------------------------------------------------------------

#[inline(always)]
#[cfg(test)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// exp(y) on 16 lanes (Cephes expf: range reduction by ln 2, degree-5 polynomial, 2^n through the exponent bits).
/// Portable: compiled for whichever SIMD level `dispatch!` picked. ~1-2 ulp on the clamped range.
#[inline(always)]
fn exp16<S: Simd>(y: f32x16<S>) -> f32x16<S> {
    let y = y.max(-87.3).min(88.7);
    let n = (y * core::f32::consts::LOG2_E).round_ties_even();
    let r = n.mul_add(-0.693_359_4, y);
    let r = n.mul_add(2.121_944_4e-4, r);
    let mut p = f32x16::splat(y.simd, 1.987_569_1e-4);
    for c in [1.398_199_9e-3, 8.333_452e-3, 4.166_579_6e-2, 0.166_666_65, 0.5] {
        p = p.mul_add(r, c);
    }
    let e = (p * r).mul_add(r, r) + 1.0;
    let bits = (i32x16::truncate_from(n) + 127) << 23;
    e * bits.bitcast::<f32x16<S>>()
}

/// x <- silu(x), 16 lanes at a time.
#[inline(always)]
fn silu_inplace<S: Simd>(simd: S, x: &mut [f32]) {
    let mut ch = x.chunks_exact_mut(16);
    for c in &mut ch {
        let v = f32x16::from_slice(simd, c);
        (v / (exp16(-v) + 1.0)).store_slice(c);
    }
    let r = ch.into_remainder();
    if !r.is_empty() {
        // the tail goes through the same exp16, so a value's result does not depend on where a slice starts
        let mut t = [0f32; 16];
        t[..r.len()].copy_from_slice(r);
        let v = f32x16::from_slice(simd, &t);
        (v / (exp16(-v) + 1.0)).store_slice(&mut t);
        r.copy_from_slice(&t[..r.len()]);
    }
}

/// out[i] = silu(a[i]) * b[i]
#[inline(always)]
fn silu_mul<S: Simd>(simd: S, out: &mut [f32], a: &[f32], b: &[f32]) {
    let n = out.len() / 16 * 16;
    for i in (0..n).step_by(16) {
        let (va, vb) = (
            f32x16::from_slice(simd, &a[i..i + 16]),
            f32x16::from_slice(simd, &b[i..i + 16]),
        );
        (va / (exp16(-va) + 1.0) * vb).store_slice(&mut out[i..i + 16]);
    }
    let k = out.len() - n;
    if k > 0 {
        let (mut ta, mut tb) = ([0f32; 16], [0f32; 16]);
        ta[..k].copy_from_slice(&a[n..n + k]);
        tb[..k].copy_from_slice(&b[n..n + k]);
        let (va, vb) = (f32x16::from_slice(simd, &ta), f32x16::from_slice(simd, &tb));
        (va / (exp16(-va) + 1.0) * vb).store_slice(&mut ta);
        out[n..].copy_from_slice(&ta[..k]);
    }
}

#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

#[inline(always)]
fn rms(x: &[f32], y: &mut [f32], n: usize, eps: f32) {
    let ss: f32 = x[..n].iter().map(|v| v * v).sum();
    let r = 1.0 / (ss / n as f32 + eps).sqrt();
    for (o, v) in y[..n].iter_mut().zip(&x[..n]) {
        *o = v * r;
    }
    y[n..].fill(0.0);
}

#[inline(always)]
fn rms_inplace(x: &mut [f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let r = 1.0 / (ss / n as f32 + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= r);
}

#[inline(always)]
fn l2n(x: &mut [f32], mul: f32) {
    let nr = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let r = mul / nr.max(1e-12);
    x.iter_mut().for_each(|v| *v *= r);
}

fn get_row(m: &MatV, r: usize, out: &mut [f32]) {
    let c = m.cpad;
    match m.w {
        W::F32(w) => out[..c].copy_from_slice(&w[r * c..(r + 1) * c]),
        W::F16(w) => {
            for (o, &h) in out[..c].iter_mut().zip(&w[r * c..(r + 1) * c]) {
                *o = f16_to_f32(h);
            }
        }
        W::I8(w) | W::W8A8(w) => {
            let s = m.s[r];
            for (o, &q) in out[..c].iter_mut().zip(&w[r * c..(r + 1) * c]) {
                *o = q as f32 * s;
            }
        }
        W::I4(w) => {
            let (w, s) = (&w[r * c / 2..(r + 1) * c / 2], &m.s[r * (c / 32)..(r + 1) * (c / 32)]);
            for g in 0..c / 32 {
                for j in 0..16 {
                    let b = w[g * 16 + j];
                    out[g * 32 + j] = ((b & 15) as f32 - 8.0) * s[g];
                    out[g * 32 + 16 + j] = ((b >> 4) as f32 - 8.0) * s[g];
                }
            }
        }
    }
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 1024 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | (e2 << 23) | ((m2 & 1023) << 13)
        }
    } else if e == 31 {
        (s << 31) | (255 << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------------------------------------------
// Matmul: Y[t, r] (+)= W[r] . X[t]
// ---------------------------------------------------------------------------------------------------------------

/// Per-token int8 activations (+128 -> u8), packed 4 per i32 word for vpdpbusd broadcasts; returns the scales.
/// Plain scalar reference for the SIMD versions (`quant_simd`, AVX-512 `tiles::quant_row`), used by the tests.
#[cfg(test)]
pub(crate) fn quant_act(x: &[f32], ldx: usize, t: usize, n: usize, cpad: usize, xw: &mut [i32]) -> Vec<f32> {
    let c4 = cpad / 4;
    let mut ub = vec![128u8; cpad];
    (0..t)
        .map(|ti| {
            let row = &x[ti * ldx..ti * ldx + n];
            let mut lanes = [0f32; 16];
            let mut ch = row.chunks_exact(16);
            for c in &mut ch {
                for (m, v) in lanes.iter_mut().zip(c) {
                    *m = m.max(v.abs());
                }
            }
            let amax = ch
                .remainder()
                .iter()
                .fold(lanes.iter().fold(0f32, |a, &b| a.max(b)), |a, v| a.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let r = 1.0 / s;
            for (o, &v) in ub[..n].iter_mut().zip(row) {
                *o = ((v * r).round_ties_even() as i32 + 128) as u8;
            }
            for (w, b) in xw[ti * c4..(ti + 1) * c4].iter_mut().zip(ub.chunks_exact(4)) {
                *w = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            s
        })
        .collect()
}

/// `quant_act` for one row through fearless_simd vectors (AVX2 / SSE / NEON levels); same results.
#[inline(always)]
fn quant_row_simd<S: Simd>(simd: S, row: &[f32], out: &mut [i32]) -> f32 {
    let mut m = f32x16::splat(simd, 0.0);
    let mut ch = row.chunks_exact(16);
    for c in &mut ch {
        m = m.max(f32x16::from_slice(simd, c).abs());
    }
    let mut lanes = [0f32; 16];
    m.store_slice(&mut lanes);
    let amax = ch
        .remainder()
        .iter()
        .fold(lanes.iter().fold(0f32, |a, &b| a.max(b)), |a, v| a.max(v.abs()));
    let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let r = 1.0 / s;
    let (mut tmp, mut q) = ([0f32; 16], [0i32; 16]);
    for (i, o) in out.chunks_exact_mut(4).enumerate() {
        let base = i * 16;
        let v = if base + 16 <= row.len() {
            f32x16::from_slice(simd, &row[base..base + 16])
        } else {
            tmp.fill(0.0);
            if base < row.len() {
                tmp[..row.len() - base].copy_from_slice(&row[base..]);
            }
            f32x16::from_slice(simd, &tmp)
        };
        (i32x16::truncate_from((v * r).round_ties_even()) + 128).store_slice(&mut q);
        for (j, w) in o.iter_mut().enumerate() {
            *w = i32::from_le_bytes([
                q[4 * j] as u8,
                q[4 * j + 1] as u8,
                q[4 * j + 2] as u8,
                q[4 * j + 3] as u8,
            ]);
        }
    }
    s
}

pub(crate) fn quant_simd(
    level: Level,
    x: &[f32],
    ldx: usize,
    t: usize,
    n: usize,
    cpad: usize,
    xw: &mut [i32],
) -> Vec<f32> {
    let c4 = cpad / 4;
    dispatch!(level, simd => (0..t)
        .map(|ti| quant_row_simd(simd, &x[ti * ldx..ti * ldx + n], &mut xw[ti * c4..(ti + 1) * c4]))
        .collect())
}

pub struct MatIn<'a> {
    pub x: &'a [f32],
    pub ldx: usize,
    pub t: usize,
    pub cols: usize,
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn matmul<S: Simd>(simd: S, m: &MatV, inp: MatIn, y: &mut [f32], ldy: usize, acc: bool, nth: usize) {
    let level = simd.level();
    let ntile = m.rpad / 32;
    let c4 = m.cpad / 4;
    let (mut xw, mut xs) = (Vec::new(), Vec::new());
    if let W::W8A8(_) = m.w {
        xw = vec![0i32; inp.t * c4];
        xs = tiles::quant(level, inp.x, inp.ldx, inp.t, inp.cols, m.cpad, &mut xw);
    }
    let one = |tile: usize, t0: usize, buf: &mut [f32; 32 * tiles::NTB]| {
        let nt = (inp.t - t0).min(tiles::NTB);
        let xq = if xw.is_empty() { &xw[..] } else { &xw[t0 * c4..] };
        let xsq = if xs.is_empty() { &xs[..] } else { &xs[t0..] };
        tiles::tile(level, m, tile, &inp.x[t0 * inp.ldx..], inp.ldx, xq, xsq, nt, buf);
    };
    let write = |y: &mut [f32], tile: usize, t0: usize, buf: &[f32]| {
        let r0 = tile * 32;
        let nr = (m.rows - r0).min(32);
        let nt = (inp.t - t0).min(tiles::NTB);
        for t in 0..nt {
            let dst = &mut y[(t0 + t) * ldy + r0..(t0 + t) * ldy + r0 + nr];
            let src = &buf[t * 32..t * 32 + nr];
            if acc {
                dst.iter_mut().zip(src).for_each(|(a, b)| *a += b);
            } else {
                dst.copy_from_slice(src);
            }
        }
    };
    let fast = inp.t == 1 && matches!(m.w, W::W8A8(_));
    if fast && nth <= 1 {
        let yr = &mut y[..m.rows];
        if tiles::matvec_w8a8(level, m, 0, ntile, &xw, xs[0], yr, acc) {
            return;
        }
    }
    if nth > 1 {
        // Thread tid always owns the same tiles (rows), so its share of the weights stays in its own L2.
        let (yp, ylen) = (SyncPtr(y.as_mut_ptr()), y.len());
        let job = |tid: usize| {
            let (lo, hi) = split(ntile, nth, tid);
            if fast && lo < hi {
                let (r0, r1) = (lo * 32, (hi * 32).min(m.rows));
                assert!(r1 <= ylen);
                // Safety: rows r0..r1 belong to this thread's tiles, split disjointly across threads.
                let yr = unsafe { core::slice::from_raw_parts_mut(yp.get().add(r0), r1 - r0) };
                if tiles::matvec_w8a8(level, m, lo, hi, &xw, xs[0], yr, acc) {
                    return;
                }
            }
            let mut buf = [0f32; 32 * tiles::NTB];
            for tile in lo..hi {
                for t0 in (0..inp.t).step_by(tiles::NTB) {
                    one(tile, t0, &mut buf);
                    let (r0, nt) = (tile * 32, (inp.t - t0).min(tiles::NTB));
                    let nr = (m.rows - r0).min(32);
                    for t in 0..nt {
                        let base = (t0 + t) * ldy + r0;
                        assert!(base + nr <= ylen);
                        // Safety: rows r0..r0+nr belong to this tile, and tiles are split disjointly across threads.
                        let dst = unsafe { core::slice::from_raw_parts_mut(yp.get().add(base), nr) };
                        let src = &buf[t * 32..t * 32 + nr];
                        if acc {
                            dst.iter_mut().zip(src).for_each(|(a, b)| *a += b);
                        } else {
                            dst.copy_from_slice(src);
                        }
                    }
                }
            }
        };
        pool::get(nth).run(&job);
    } else {
        let mut buf = [0f32; 32 * tiles::NTB];
        for tile in 0..ntile {
            for t0 in (0..inp.t).step_by(tiles::NTB) {
                one(tile, t0, &mut buf);
                write(y, tile, t0, &buf);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------------------------------------------

pub fn work_len(m: &ModelV, bt: usize) -> usize {
    let rin = 4 * m.d + 2 * m.h;
    bt * (m.dp + m.dp + rin + m.dp + 2 * m.hp + m.hp) + 64
}

pub struct Run<'a> {
    pub toks: &'a [i32],
    pub state: &'a mut [f32],
    pub logits: &'a mut [f32],
    pub all_logits: bool,
    pub work: &'a mut [f32],
    pub bt: usize,
    pub nth: usize,
}

/// One GDN head for one token: normalize q, k; gated delta-rule update of the head's state S (D x D, row-major);
/// write o = RMSNorm(S^T q) * silu(gate).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn head_step<S: Simd>(
    simd: S,
    qh: &mut [f32],
    kh: &mut [f32],
    vh: &[f32],
    gt: &[f32],
    (a, b): (f32, f32),
    (neg_a, dt_bias): (f32, f32),
    oh: &mut [f32],
    sh: &mut [f32],
    scratch: &mut [f32],
    qs: f32,
    eps: f32,
) {
    let dh = qh.len();
    let (u, rest) = scratch.split_at_mut(dh);
    let (wv, rest) = rest.split_at_mut(dh);
    let gbuf = &mut rest[..dh];
    l2n(qh, qs);
    l2n(kh, 1.0);
    let beta = 1.0 / (1.0 + (-b).exp());
    let decay = (neg_a * softplus(a + dt_bias)).exp();
    // u = S^T k, w = S^T q (read pass); delta = (v - decay u) beta; S <- decay S + k delta^T; o = decay w + (k.q) delta
    u.fill(0.0);
    wv.fill(0.0);
    let mut kq = 0f32;
    for (i, row) in sh.chunks_exact(dh).enumerate() {
        let (ki, qi) = (kh[i], qh[i]);
        kq += ki * qi;
        for ((uj, wj), &sj) in u.iter_mut().zip(wv.iter_mut()).zip(row) {
            *uj += sj * ki;
            *wj += sj * qi;
        }
    }
    for j in 0..dh {
        u[j] = (vh[j] - decay * u[j]) * beta;
        oh[j] = decay * wv[j] + kq * u[j];
    }
    for (i, row) in sh.chunks_exact_mut(dh).enumerate() {
        let ki = kh[i];
        for (sj, &uj) in row.iter_mut().zip(u.iter()) {
            *sj = decay * *sj + ki * uj;
        }
    }
    rms_inplace(oh, eps);
    gbuf.copy_from_slice(oh);
    silu_mul(simd, oh, gt, gbuf);
}

/// The GDN layers of `m` on bt rows of the residual stream xb (row stride dp), in place, carrying `state`
/// (L*H*D*D). `rest` is the rest of the work buffer after xb (work_len(bt_max) - bt_max * dp floats). Embedding
/// and head are the caller's: `forward_impl` wraps this for token models, `hnet` chains three stacks.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn stack_impl<S: Simd>(
    simd: S,
    m: &ModelV,
    xb: &mut [f32],
    rest: &mut [f32],
    bt_max: usize,
    bt: usize,
    state: &mut [f32],
    nth: usize,
) {
    let (d, dp, h, dh, hid, hp) = (m.d, m.dp, m.h, m.dh, m.hid, m.hp);
    let rin = 4 * d + 2 * h;
    let (nb, rest) = rest.split_at_mut(bt_max * dp);
    let (pb, rest) = rest.split_at_mut(bt_max * rin);
    let (ob, rest) = rest.split_at_mut(bt_max * dp);
    let (ub, rest) = rest.split_at_mut(bt_max * 2 * hp);
    let (mb, _) = rest.split_at_mut(bt_max * hp);
    let qs = 1.0 / (dh as f32).sqrt();
    let mut scratch = vec![0f32; 3 * dh];
    for (l, ly) in m.layers.iter().enumerate() {
        let sl = &mut state[l * h * dh * dh..(l + 1) * h * dh * dh];
        for t in 0..bt {
            rms(&xb[t * dp..(t + 1) * dp], &mut nb[t * dp..(t + 1) * dp], d, m.eps);
        }
        matmul(
            simd,
            &ly.inp,
            MatIn {
                x: nb,
                ldx: dp,
                t: bt,
                cols: d,
            },
            pb,
            rin,
            false,
            nth,
        );
        for t in 0..bt {
            silu_inplace(simd, &mut pb[t * rin..t * rin + 3 * d]);
        }
        if nth > 1 && h > 1 {
            // heads split across threads (each thread keeps its heads' state in its own cache)
            let (pp, op, sp) = (
                SyncPtr(pb.as_mut_ptr()),
                SyncPtr(ob.as_mut_ptr()),
                SyncPtr(sl.as_mut_ptr()),
            );
            let (np, no, ns) = (pb.len(), ob.len(), sl.len());
            let nth = nth.min(h);
            let job = |tid: usize| {
                if tid >= nth {
                    return; // more threads than heads
                }
                let (h0, h1) = split(h, nth, tid);
                if h0 == h1 {
                    return;
                }
                simd.vectorize(
                    #[inline(always)]
                    || {
                        let mut scratch = vec![0f32; 3 * dh];
                        for t in 0..bt {
                            for hh in h0..h1 {
                                let (pq, pk) = (t * rin + hh * dh, t * rin + d + hh * dh);
                                let (po, ps) = (t * dp + hh * dh, hh * dh * dh);
                                assert!(t * rin + rin <= np && po + dh <= no && ps + dh * dh <= ns);
                                // Safety: q, k, o and S ranges of head hh are disjoint from every other head's
                                // (heads are split disjointly across threads); v, gate, a, b are only read.
                                let (qh, kh, vh, gt, oh, sh, ab) = unsafe {
                                    let p = pp.get();
                                    (
                                        core::slice::from_raw_parts_mut(p.add(pq), dh),
                                        core::slice::from_raw_parts_mut(p.add(pk), dh),
                                        core::slice::from_raw_parts(p.add(t * rin + 2 * d + hh * dh), dh),
                                        core::slice::from_raw_parts(p.add(t * rin + 3 * d + hh * dh), dh),
                                        core::slice::from_raw_parts_mut(op.get().add(po), dh),
                                        core::slice::from_raw_parts_mut(sp.get().add(ps), dh * dh),
                                        (*p.add(t * rin + 4 * d + hh), *p.add(t * rin + 4 * d + h + hh)),
                                    )
                                };
                                let g = (ly.neg_a[hh], ly.dt_bias[hh]);
                                head_step(simd, qh, kh, vh, gt, ab, g, oh, sh, &mut scratch, qs, m.eps);
                            }
                        }
                    },
                )
            };
            pool::get(nth).run(&job);
        } else {
            for t in 0..bt {
                let p = &mut pb[t * rin..(t + 1) * rin];
                let (qkv, rest) = p.split_at_mut(3 * d);
                let (gt, ab) = rest.split_at(d);
                let (q, kv) = qkv.split_at_mut(d);
                let (k, vv) = kv.split_at_mut(d);
                let o = &mut ob[t * dp..(t + 1) * dp];
                for hh in 0..h {
                    head_step(
                        simd,
                        &mut q[hh * dh..(hh + 1) * dh],
                        &mut k[hh * dh..(hh + 1) * dh],
                        &vv[hh * dh..(hh + 1) * dh],
                        &gt[hh * dh..(hh + 1) * dh],
                        (ab[hh], ab[h + hh]),
                        (ly.neg_a[hh], ly.dt_bias[hh]),
                        &mut o[hh * dh..(hh + 1) * dh],
                        &mut sl[hh * dh * dh..(hh + 1) * dh * dh],
                        &mut scratch,
                        qs,
                        m.eps,
                    );
                }
            }
        }
        for t in 0..bt {
            ob[t * dp + d..(t + 1) * dp].fill(0.0);
        }
        matmul(
            simd,
            &ly.o,
            MatIn {
                x: ob,
                ldx: dp,
                t: bt,
                cols: d,
            },
            xb,
            dp,
            true,
            nth,
        );
        for t in 0..bt {
            rms(&xb[t * dp..(t + 1) * dp], &mut nb[t * dp..(t + 1) * dp], d, m.eps);
        }
        matmul(
            simd,
            &ly.up,
            MatIn {
                x: nb,
                ldx: dp,
                t: bt,
                cols: d,
            },
            ub,
            2 * hp,
            false,
            nth,
        );
        for t in 0..bt {
            let uu = &ub[t * 2 * hp..(t + 1) * 2 * hp];
            let mh = &mut mb[t * hp..(t + 1) * hp];
            silu_mul(simd, &mut mh[..hid], &uu[..hid], &uu[hid..2 * hid]);
            mh[hid..].fill(0.0);
        }
        matmul(
            simd,
            &ly.down,
            MatIn {
                x: mb,
                ldx: hp,
                t: bt,
                cols: hid,
            },
            xb,
            dp,
            true,
            nth,
        );
    }
}

#[inline(always)]
fn forward_impl<S: Simd>(simd: S, m: &ModelV, r: Run) {
    let (d, dp, v) = (m.d, m.dp, m.vocab);
    let bt_max = r.bt;
    let (xb, rest) = r.work.split_at_mut(bt_max * dp);
    let tt = r.toks.len();
    let mut t0 = 0;
    while t0 < tt {
        let bt = (tt - t0).min(bt_max);
        for t in 0..bt {
            get_row(&m.emb, r.toks[t0 + t] as usize, &mut xb[t * dp..(t + 1) * dp]);
        }
        stack_impl(simd, m, xb, rest, bt_max, bt, r.state, r.nth);
        let nb = &mut rest[..bt_max * dp];
        if r.all_logits {
            for t in 0..bt {
                rms(&xb[t * dp..(t + 1) * dp], &mut nb[t * dp..(t + 1) * dp], d, m.eps);
            }
            let inp = MatIn {
                x: nb,
                ldx: dp,
                t: bt,
                cols: d,
            };
            matmul(simd, &m.head, inp, &mut r.logits[t0 * v..], v, false, r.nth);
        } else if t0 + bt == tt {
            rms(&xb[(bt - 1) * dp..bt * dp], &mut nb[..dp], d, m.eps);
            matmul(
                simd,
                &m.head,
                MatIn {
                    x: nb,
                    ldx: dp,
                    t: 1,
                    cols: d,
                },
                r.logits,
                v,
                false,
                r.nth,
            );
        }
        t0 += bt;
    }
}

/// One thread's rows of a decode matvec y (+)= W x: tiles split(ntile, nth, tid) of y (ylen floats, shared).
/// x: the full (padded) input row, cols real columns; xw: scratch for the quantized activations.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn rows_mv<S: Simd>(
    simd: S,
    m: &MatV,
    x: &[f32],
    cols: usize,
    xw: &mut [i32],
    (tid, nth): (usize, usize),
    (y, ylen): (SyncPtr<f32>, usize),
    acc: bool,
) {
    let level = simd.level();
    let (lo, hi) = split(m.rpad / 32, nth, tid);
    if lo == hi {
        return;
    }
    let (r0, r1) = (lo * 32, (hi * 32).min(m.rows));
    assert!(r1 <= ylen);
    // Safety: rows r0..r1 of y belong to this thread's tiles, which are split disjointly across the threads.
    let yr = unsafe { core::slice::from_raw_parts_mut(y.get().add(r0), r1 - r0) };
    let mut xs = [1f32];
    let c4 = m.cpad / 4;
    if let W::W8A8(_) = m.w {
        xs[0] = tiles::quant(level, x, cols, 1, cols, m.cpad, &mut xw[..c4])[0];
        if tiles::matvec_w8a8(level, m, lo, hi, &xw[..c4], xs[0], yr, acc) {
            return;
        }
    }
    let mut buf = [0f32; 32 * tiles::NTB];
    for tile in lo..hi {
        tiles::tile(level, m, tile, x, cols, &xw[..c4], &xs, 1, &mut buf);
        let (a, n) = (tile * 32 - r0, (r1 - tile * 32).min(32));
        let dst = &mut yr[a..a + n];
        if acc {
            dst.iter_mut().zip(&buf[..n]).for_each(|(o, b)| *o += b);
        } else {
            dst.copy_from_slice(&buf[..n]);
        }
    }
}

/// Greedy decode with nth threads in lockstep: one pool hand-off for the whole loop. Every thread runs the forward
/// pass, redoes the cheap per-element steps (RMSNorm, activation quantization, SwiGLU, argmax) on private copies,
/// computes its own rows of every matmul and its own GDN heads into shared buffers, and meets the others at a
/// barrier after each phase (5 per layer, 1 per byte for the logits).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn generate_lockstep<S: Simd>(
    simd: S,
    m: &ModelV,
    first: i32,
    out: &mut [i32],
    state: &mut [f32],
    logits: &mut [f32],
    work: &mut [f32],
    nth: usize,
) {
    let (d, dp, h, dh, hid, hp, v) = (m.d, m.dp, m.h, m.dh, m.hid, m.hp, m.vocab);
    let rin = 4 * d + 2 * h;
    let (xb, rest) = work.split_at_mut(dp);
    let (_, rest) = rest.split_at_mut(dp);
    let (pb, rest) = rest.split_at_mut(rin);
    let (ob, rest) = rest.split_at_mut(dp);
    let ub = &mut rest[..2 * hp];
    ob.fill(0.0); // the padding columns stay zero
    let (xp, pp, op, up) = (
        SyncPtr(xb.as_mut_ptr()),
        SyncPtr(pb.as_mut_ptr()),
        SyncPtr(ob.as_mut_ptr()),
        SyncPtr(ub.as_mut_ptr()),
    );
    let (sp, lp, outp) = (
        SyncPtr(state.as_mut_ptr()),
        SyncPtr(logits.as_mut_ptr()),
        SyncPtr(out.as_mut_ptr()),
    );
    let (ns, nout) = (state.len(), out.len());
    assert!(m.layers.len() * h * dh * dh <= ns && logits.len() >= v);
    let qs = 1.0 / (dh as f32).sqrt();
    let bar = pool::Barrier::new(nth);
    // Shared views. Safety (all uses below): a phase only reads a buffer that no thread writes in that phase, or
    // writes a part of it that no other thread touches in that phase; phases are separated by bar.wait().
    let view = |p: SyncPtr<f32>, n: usize| unsafe { core::slice::from_raw_parts(p.get(), n) };
    let job = |tid: usize| {
        simd.vectorize(
            #[inline(always)]
            || {
                let mut nb = vec![0f32; dp];
                let mut mb = vec![0f32; hp];
                let mut xw = vec![0i32; dp.max(hp) / 4 + 16];
                let mut scratch = vec![0f32; 3 * dh];
                let (h0, h1) = split(h, nth, tid);
                let me = (tid, nth);
                let mut tok = first as usize;
                for i in 0..nout {
                    if tid == 0 {
                        get_row(&m.emb, tok, unsafe { core::slice::from_raw_parts_mut(xp.get(), dp) });
                    }
                    bar.wait();
                    for (l, ly) in m.layers.iter().enumerate() {
                        rms(view(xp, dp), &mut nb, d, m.eps);
                        rows_mv(simd, &ly.inp, &nb, d, &mut xw, me, (pp, rin), false);
                        bar.wait();
                        for hh in h0..h1 {
                            let (qh, kh, vh, gt, oh, sh, ab) = unsafe {
                                let p = pp.get();
                                let ps = (l * h + hh) * dh * dh;
                                (
                                    core::slice::from_raw_parts_mut(p.add(hh * dh), dh),
                                    core::slice::from_raw_parts_mut(p.add(d + hh * dh), dh),
                                    core::slice::from_raw_parts_mut(p.add(2 * d + hh * dh), dh),
                                    core::slice::from_raw_parts(p.add(3 * d + hh * dh), dh),
                                    core::slice::from_raw_parts_mut(op.get().add(hh * dh), dh),
                                    core::slice::from_raw_parts_mut(sp.get().add(ps), dh * dh),
                                    (*p.add(4 * d + hh), *p.add(4 * d + h + hh)),
                                )
                            };
                            silu_inplace(simd, qh);
                            silu_inplace(simd, kh);
                            silu_inplace(simd, vh);
                            let g = (ly.neg_a[hh], ly.dt_bias[hh]);
                            head_step(simd, qh, kh, vh, gt, ab, g, oh, sh, &mut scratch, qs, m.eps);
                        }
                        bar.wait();
                        rows_mv(simd, &ly.o, view(op, dp), d, &mut xw, me, (xp, dp), true);
                        bar.wait();
                        rms(view(xp, dp), &mut nb, d, m.eps);
                        rows_mv(simd, &ly.up, &nb, d, &mut xw, me, (up, 2 * hp), false);
                        bar.wait();
                        let u = view(up, 2 * hp);
                        silu_mul(simd, &mut mb[..hid], &u[..hid], &u[hid..2 * hid]);
                        rows_mv(simd, &ly.down, &mb, hid, &mut xw, me, (xp, dp), true);
                        bar.wait();
                    }
                    rms(view(xp, dp), &mut nb, d, m.eps);
                    rows_mv(simd, &m.head, &nb, d, &mut xw, me, (lp, v), false);
                    bar.wait();
                    let lg = view(lp, v);
                    tok = lg
                        .iter()
                        .enumerate()
                        .fold(0, |b, (j, &x)| if x > lg[b] { j } else { b });
                    if tid == 0 {
                        unsafe { *outp.get().add(i) = tok as i32 };
                    }
                }
            },
        )
    };
    pool::get(nth).run(&job);
    let _ = ns;
}

pub fn forward_safe(level: Level, m: &ModelV, r: Run) {
    dispatch!(level, simd => forward_impl(simd, m, r));
}

// ---------------------------------------------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------------------------------------------

/// Force the kernel family: 0 auto, 1 scalar, 2 SSE4.2, 3 AVX2, 4 AVX-VNNI, 5 AVX-512, 6 NEON, 7 NEON + dotprod.
/// Returns the family now in use, or -1 (nothing changed) if this CPU can't run `code`. Process-wide; for tests and
/// benchmarks.
#[unsafe(no_mangle)]
pub extern "C" fn set_simd(code: i32) -> i32 {
    isa::force_code(code)
}

/// The kernel family in use (codes as in `set_simd`).
#[unsafe(no_mangle)]
pub extern "C" fn simd_isa() -> i32 {
    isa::current() as i32
}

/// # Safety
/// `m` must come from `lilbits.inference.fused` (FusedLM builds it).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn work_floats(m: *const Model, bt: i32) -> i64 {
    let mv = unsafe { view_model(&*m) };
    work_len(&mv, bt.max(1) as usize) as i64
}

/// # Safety
/// toks[t], state[L*H*D*D], logits[(all ? t : 1) * vocab],
/// work[work_floats(bt)].
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn forward(
    m: *const Model,
    toks: *const i32,
    t: i32,
    state: *mut f32,
    logits: *mut f32,
    all_logits: i32,
    work: *mut f32,
    bt: i32,
    nth: i32,
) {
    let mv = unsafe { view_model(&*m) };
    let (t, bt) = (t.max(0) as usize, bt.max(1) as usize);
    let ns = mv.layers.len() * mv.h * mv.dh * mv.dh;
    let nl = if all_logits != 0 { t * mv.vocab } else { mv.vocab };
    let nw = work_len(&mv, bt);
    let r = unsafe {
        Run {
            toks: core::slice::from_raw_parts(toks, t),
            state: core::slice::from_raw_parts_mut(state, ns),
            logits: core::slice::from_raw_parts_mut(logits, nl),
            all_logits: all_logits != 0,
            work: core::slice::from_raw_parts_mut(work, nw),
            bt,
            nth: nth.max(1) as usize,
        }
    };
    forward_safe(isa::level(isa::current()), &mv, r);
}

/// # Safety
/// As `forward`; out[n]. nth: threads (1 = this thread only).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn generate(
    m: *const Model,
    first: i32,
    n: i32,
    state: *mut f32,
    logits: *mut f32,
    work: *mut f32,
    out: *mut i32,
    nth: i32,
) {
    let mv = unsafe { view_model(&*m) };
    let ns = mv.layers.len() * mv.h * mv.dh * mv.dh;
    let nw = work_len(&mv, 1);
    let (state, logits, work, out) = unsafe {
        (
            core::slice::from_raw_parts_mut(state, ns),
            core::slice::from_raw_parts_mut(logits, mv.vocab),
            core::slice::from_raw_parts_mut(work, nw),
            core::slice::from_raw_parts_mut(out, n.max(0) as usize),
        )
    };
    let level = isa::level(isa::current());
    if nth > 1 {
        let nth = nth as usize;
        dispatch!(level, simd => generate_lockstep(simd, &mv, first, out, state, logits, work, nth));
        return;
    }
    let mut tok = [first];
    for o in out.iter_mut() {
        let r = Run {
            toks: &tok,
            state: &mut *state,
            logits: &mut *logits,
            all_logits: false,
            work: &mut *work,
            bt: 1,
            nth: nth.max(1) as usize,
        };
        forward_safe(level, &mv, r);
        let best = logits
            .iter()
            .enumerate()
            .fold(0, |b, (j, &x)| if x > logits[b] { j } else { b });
        *o = best as i32;
        tok[0] = best as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f32 {
            (self.next() % 20001) as f32 / 10000.0 - 1.0
        }
    }

    /// Owned storage for one random matrix in format `fmt` (tile-major unless `row_major`).
    struct OwnedMat {
        w: Vec<u8>,
        s: Vec<f32>,
        m: Mat,
    }

    fn rand_mat(rng: &mut Rng, fmt: i32, rows: usize, cpad: usize, row_major: bool) -> OwnedMat {
        let rpad = if row_major { 0 } else { rows.div_ceil(32) * 32 };
        let nr = if row_major { rows } else { rpad };
        let n = nr * cpad;
        let amp = 1.0 / (cpad as f32).sqrt();
        let (w, s): (Vec<u8>, Vec<f32>) = match fmt {
            0 => (
                (0..n).flat_map(|_| (rng.f() * amp * 1.7).to_le_bytes()).collect(),
                vec![],
            ),
            1 => (
                (0..n).flat_map(|_| half(rng.f() * amp * 1.7).to_le_bytes()).collect(),
                vec![],
            ),
            2 => ((0..n).map(|_| rng.next() as u8 % 255).collect(), vec![amp * 0.014; nr]),
            3 => (
                (0..n / 2).map(|_| rng.next() as u8).collect(),
                vec![amp * 0.2; nr * (cpad / 32)],
            ),
            _ => {
                let w: Vec<u8> = (0..n).map(|_| (rng.next() % 255) as u8).collect();
                let mut s = vec![amp * 0.014; nr];
                for r in 0..nr {
                    let sum: i32 = if row_major {
                        (0..cpad).map(|i| w[r * cpad + i] as i8 as i32).sum()
                    } else {
                        let (tl, j) = (r / 32, r % 32);
                        (0..cpad)
                            .map(|i| w[tl * cpad * 32 + (i / 4) * 128 + j * 4 + i % 4] as i8 as i32)
                            .sum()
                    };
                    s.push(f32::from_bits((128 * sum) as u32));
                }
                (w, s)
            }
        };
        let mut o = OwnedMat {
            w,
            s,
            m: Mat {
                rows: rows as i32,
                cols: cpad as i32,
                cpad: cpad as i32,
                fmt,
                w: core::ptr::null(),
                s: core::ptr::null(),
                rpad: rpad as i32,
            },
        };
        o.m.w = o.w.as_ptr() as *const c_void;
        o.m.s = o.s.as_ptr();
        o
    }

    fn half(v: f32) -> u16 {
        let b = v.to_bits();
        let s = ((b >> 16) & 0x8000) as u16;
        let e = ((b >> 23) & 255) as i32 - 127 + 15;
        if e <= 0 {
            s
        } else {
            s | (((e as u32) << 10) | ((b & 0x7f_ffff) >> 13)) as u16
        }
    }

    /// Runs a small random model (every weight format) through prefill + greedy decode under every kernel family
    /// this CPU supports and 1 / 3 threads: logits agree with the scalar reference, and thread counts agree exactly.
    #[test]
    fn forward_all_families() {
        let (d, h, hid, l, v) = (64usize, 2usize, 96usize, 2usize, 256usize);
        let (dp, hp, dh) = (64usize, 96usize, d / h);
        let rin = 4 * d + 2 * h;
        for fmt in 0..5 {
            let mut rng = Rng(0xDEAD_BEEF + fmt as u64);
            let emb = rand_mat(&mut rng, fmt, v, dp, true);
            let head = rand_mat(&mut rng, fmt, v, dp, false);
            let mats: Vec<[OwnedMat; 4]> = (0..l)
                .map(|_| {
                    [
                        rand_mat(&mut rng, fmt, rin, dp, false),
                        rand_mat(&mut rng, fmt, d, dp, false),
                        rand_mat(&mut rng, fmt, 2 * hid, dp, false),
                        rand_mat(&mut rng, fmt, d, hp, false),
                    ]
                })
                .collect();
            let (neg_a, dtb): (Vec<f32>, Vec<f32>) = ((0..h).map(|_| -0.5).collect(), (0..h).map(|_| 0.1).collect());
            let layers: Vec<Layer> = mats
                .iter()
                .map(|[a, b, c, e]| Layer {
                    in_: Mat { ..a.m },
                    o: Mat { ..b.m },
                    up: Mat { ..c.m },
                    down: Mat { ..e.m },
                    neg_a: neg_a.as_ptr(),
                    dt_bias: dtb.as_ptr(),
                })
                .collect();
            let model = Model {
                d: d as i32,
                dpad: dp as i32,
                h: h as i32,
                dh: dh as i32,
                hid: hid as i32,
                hpad: hp as i32,
                l: l as i32,
                vocab: v as i32,
                eps: 1e-6,
                emb: Mat { ..emb.m },
                head: Mat { ..head.m },
                layers: layers.as_ptr(),
            };
            let toks: Vec<i32> = (0..21).map(|_| (rng.next() % 256) as i32).collect();
            let run = |nth: i32| {
                let mut state = vec![0f32; l * h * dh * dh];
                let mut all = vec![0f32; toks.len() * v];
                let mut logits = vec![0f32; v];
                let mut work = vec![0f32; unsafe { work_floats(&model, 8) } as usize];
                let mut out = vec![0i32; 12];
                unsafe {
                    let (sp, wp) = (state.as_mut_ptr(), work.as_mut_ptr());
                    forward(
                        &model,
                        toks.as_ptr(),
                        toks.len() as i32,
                        sp,
                        all.as_mut_ptr(),
                        1,
                        wp,
                        8,
                        nth,
                    );
                    state.fill(0.0);
                    forward(
                        &model,
                        toks.as_ptr(),
                        toks.len() as i32,
                        sp,
                        logits.as_mut_ptr(),
                        0,
                        wp,
                        8,
                        nth,
                    );
                    let first = (0..v).fold(0, |b, j| if logits[j] > logits[b] { j } else { b }) as i32;
                    generate(&model, first, 12, sp, logits.as_mut_ptr(), wp, out.as_mut_ptr(), nth);
                }
                (all, out, logits)
            };
            let mut reference: Option<Vec<f32>> = None;
            for isa_ in isa::ALL.into_iter().filter(|&i| isa::supported(i)) {
                assert!(isa::force(Some(isa_)));
                let (all1, out1, last1) = run(1);
                let (all3, out3, last3) = run(3);
                assert!(
                    all1 == all3 && out1 == out3 && last1 == last3,
                    "{isa_:?} fmt {fmt}: threads differ"
                );
                match &reference {
                    None => reference = Some(all1),
                    Some(r) => {
                        let worst = r.iter().zip(&all1).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                        let scale = r.iter().map(|a| a.abs()).fold(0f32, f32::max);
                        assert!(
                            worst <= 1e-3 * scale,
                            "{isa_:?} fmt {fmt}: logits off by {worst} (max {scale})"
                        );
                    }
                }
            }
            isa::force(None);
        }
    }

    #[test]
    fn exp16_matches_libm() {
        let level = isa::level(isa::current());
        let xs: Vec<f32> = (0..4096).map(|i| -80.0 + 160.0 * i as f32 / 4095.0).collect();
        let mut worst = 0f32;
        dispatch!(level, simd => {
            for c in xs.chunks_exact(16) {
                let mut out = [0f32; 16];
                exp16(f32x16::from_slice(simd, c)).store_slice(&mut out);
                for (o, &x) in out.iter().zip(c) {
                    worst = worst.max(((o - x.exp()) / x.exp()).abs());
                }
            }
        });
        assert!(worst < 1e-6, "max rel err {worst}");
    }

    #[test]
    fn silu_matches_scalar() {
        let level = isa::level(isa::current());
        let mut v: Vec<f32> = (0..103).map(|i| (i as f32 - 50.0) * 0.37).collect();
        let want: Vec<f32> = v.iter().map(|&x| silu(x)).collect();
        dispatch!(level, simd => silu_inplace(simd, &mut v));
        for (a, b) in v.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
}
