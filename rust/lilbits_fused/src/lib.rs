//! Fused CPU inference for lilbits' LeanLM (GDN byte LM with FP block linears, conv_size = 1).
//!
//! Loaded by `lilbits.inference.fused` through a C ABI (`forward`, `generate`, `work_floats`; it replaced the
//! original C kernel with the same ABI and weight layouts), on arrays written by `lilbits.inference.fused.export`. SIMD goes through `fearless_simd`: the forward pass
//! is generic over `Simd` and dispatched to the best level at runtime (so the elementwise code and the gated delta
//! rule are compiled for AVX-512 / AVX2 / NEON / scalar), and the matmul tiles use AVX-512 intrinsics inside
//! `fearless_simd::kernel!` (target features checked, loads from bounds-checked slices). Without AVX-512, a
//! portable scalar matmul runs instead, so the library works on any CPU.
//!
//! Weight formats: 0 f32, 1 f16, 2 int8 (per-row scale), 3 int4 (group 32), 4 int8 x int8 activations (VNNI).
//! The embedding table is row-major (lookups); every other matrix is tile-major: [32-row tile][column][32]
//! (W8A8: [tile][column / 4][32][4]). The only `unsafe` is turning the FFI pointers into slices.

use core::ffi::c_void;
use fearless_simd::{Level, Simd, dispatch, f32x16, i32x16, prelude::*};

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
    ch.into_remainder().iter_mut().for_each(|v| *v = silu(*v));
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
    for i in n..out.len() {
        out[i] = silu(a[i]) * b[i];
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
/// Written as lane-parallel loops (16 running maxima, a byte buffer) so they vectorize without fast-math.
#[inline(always)]
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
    if nth > 1 {
        // Thread tid always owns the same tiles (rows), so its share of the weights stays in its own L2.
        let (yp, ylen) = (SyncPtr(y.as_mut_ptr()), y.len());
        let job = |tid: usize| {
            let (lo, hi) = split(ntile, nth, tid);
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

#[inline(always)]
fn forward_impl<S: Simd>(simd: S, m: &ModelV, r: Run) {
    let (d, dp, h, dh, hid, hp, v) = (m.d, m.dp, m.h, m.dh, m.hid, m.hp, m.vocab);
    let rin = 4 * d + 2 * h;
    let bt_max = r.bt;
    let (xb, rest) = r.work.split_at_mut(bt_max * dp);
    let (nb, rest) = rest.split_at_mut(bt_max * dp);
    let (pb, rest) = rest.split_at_mut(bt_max * rin);
    let (ob, rest) = rest.split_at_mut(bt_max * dp);
    let (ub, rest) = rest.split_at_mut(bt_max * 2 * hp);
    let (mb, _) = rest.split_at_mut(bt_max * hp);
    let qs = 1.0 / (dh as f32).sqrt();
    let mut scratch = vec![0f32; 3 * dh];
    let tt = r.toks.len();
    let mut t0 = 0;
    while t0 < tt {
        let bt = (tt - t0).min(bt_max);
        for t in 0..bt {
            get_row(&m.emb, r.toks[t0 + t] as usize, &mut xb[t * dp..(t + 1) * dp]);
        }
        for (l, ly) in m.layers.iter().enumerate() {
            let sl = &mut r.state[l * h * dh * dh..(l + 1) * h * dh * dh];
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
                r.nth,
            );
            for t in 0..bt {
                silu_inplace(simd, &mut pb[t * rin..t * rin + 3 * d]);
            }
            if r.nth > 1 && h > 1 {
                // heads split across threads (each thread keeps its heads' state in its own cache)
                let (pp, op, sp) = (
                    SyncPtr(pb.as_mut_ptr()),
                    SyncPtr(ob.as_mut_ptr()),
                    SyncPtr(sl.as_mut_ptr()),
                );
                let (np, no, ns) = (pb.len(), ob.len(), sl.len());
                let nth = r.nth.min(h);
                let job = |tid: usize| {
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
                pool::get(r.nth).run(&job);
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
                r.nth,
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
                r.nth,
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
                r.nth,
            );
        }
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

pub fn forward_safe(level: Level, m: &ModelV, r: Run) {
    dispatch!(level, simd => forward_impl(simd, m, r));
}

// ---------------------------------------------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------------------------------------------

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
    forward_safe(Level::new(), &mv, r);
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
    let level = Level::new();
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

    #[test]
    fn exp16_matches_libm() {
        let level = Level::new();
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
        let level = Level::new();
        let mut v: Vec<f32> = (0..103).map(|i| (i as f32 - 50.0) * 0.37).collect();
        let want: Vec<f32> = v.iter().map(|&x| silu(x)).collect();
        dispatch!(level, simd => silu_inplace(simd, &mut v));
        for (a, b) in v.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
}
