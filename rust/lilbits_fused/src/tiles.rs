//! One 32-row tile of a tile-major matmul for up to NTB tokens: out[t * 32 + j] = W[tile * 32 + j] . x[t].
//!
//! AVX-512 kernels (`fearless_simd::kernel!`: the intrinsics are safe to call inside, loads come from slices through
//! fearless_simd's vector types) stream the tile's weights once, broadcast one activation per column and keep 32
//! outputs per token in two zmm accumulators. Anything else runs the portable scalar tile.

use crate::{MatV, W, f16_to_f32};
use fearless_simd::Level;

pub const NTB: usize = 4;
const PF_DIST: usize = 2048;

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{NTB, PF_DIST};
    use core::arch::x86_64::*;
    use fearless_simd::{Avx512, f32x16, i8x16, i8x64, prelude::*, u8x16, u16x16};

    /// Float-accumulating tile for formats with 32 values per column (f32, f16, int8 weights).
    macro_rules! float_tile {
        ($name:ident, $ty:ty, |$avx:ident, $col:ident| $load:expr) => {
            fearless_simd::kernel!(
                #[inline]
                pub fn $name($avx: Avx512, wt: &[$ty], x: &[f32], ldx: usize, nt: usize, buf: &mut [f32]) {
                    let ld = |$col: &[$ty]| -> (__m512, __m512) { $load };
                    let z = _mm512_setzero_ps();
                    let (mut a0, mut a1) = ([z; NTB], [z; NTB]);
                    if nt == 1 {
                        // decode: two independent FMA chains (even / odd columns)
                        let (mut b0, mut b1) = (z, z);
                        let mut pairs = wt.chunks_exact(64);
                        let mut i = 0;
                        for p in &mut pairs {
                            let (w0, w1) = ld(&p[..32]);
                            let (v0, v1) = ld(&p[32..]);
                            let (xa, xb) = (_mm512_set1_ps(x[i]), _mm512_set1_ps(x[i + 1]));
                            a0[0] = _mm512_fmadd_ps(w0, xa, a0[0]);
                            a1[0] = _mm512_fmadd_ps(w1, xa, a1[0]);
                            b0 = _mm512_fmadd_ps(v0, xb, b0);
                            b1 = _mm512_fmadd_ps(v1, xb, b1);
                            i += 2;
                        }
                        for col in pairs.remainder().chunks_exact(32) {
                            let (w0, w1) = ld(col);
                            let xa = _mm512_set1_ps(x[i]);
                            a0[0] = _mm512_fmadd_ps(w0, xa, a0[0]);
                            a1[0] = _mm512_fmadd_ps(w1, xa, a1[0]);
                            i += 1;
                        }
                        a0[0] = _mm512_add_ps(a0[0], b0);
                        a1[0] = _mm512_add_ps(a1[0], b1);
                    } else {
                        for (i, col) in wt.chunks_exact(32).enumerate() {
                            let (w0, w1) = ld(col);
                            for t in 0..nt {
                                let xb = _mm512_set1_ps(x[t * ldx + i]);
                                a0[t] = _mm512_fmadd_ps(w0, xb, a0[t]);
                                a1[t] = _mm512_fmadd_ps(w1, xb, a1[t]);
                            }
                        }
                    }
                    for t in 0..nt {
                        let (lo, hi): (f32x16<Avx512>, f32x16<Avx512>) = (a0[t].simd_into($avx), a1[t].simd_into($avx));
                        lo.store_slice(&mut buf[t * 32..t * 32 + 16]);
                        hi.store_slice(&mut buf[t * 32 + 16..t * 32 + 32]);
                    }
                }
            );
        };
    }

    float_tile!(tile_f32, f32, |avx, col| (
        f32x16::from_slice(avx, &col[..16]).into(),
        f32x16::from_slice(avx, &col[16..32]).into()
    ));
    float_tile!(tile_f16, u16, |avx, col| (
        _mm512_cvtph_ps(u16x16::from_slice(avx, &col[..16]).into()),
        _mm512_cvtph_ps(u16x16::from_slice(avx, &col[16..32]).into())
    ));
    float_tile!(tile_i8, i8, |avx, col| (
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(i8x16::from_slice(avx, &col[..16]).into())),
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(i8x16::from_slice(avx, &col[16..32]).into()))
    ));

    fearless_simd::kernel!(
        /// int4: 16 bytes per column (low nibble rows 0..15, high 16..31, value - 8), a scale per 32-column group.
        #[inline]
        pub fn tile_i4(avx: Avx512, wt: &[u8], st: &[f32], x: &[f32], ldx: usize, nt: usize, buf: &mut [f32]) {
            let z = _mm512_setzero_ps();
            let (mut a0, mut a1) = ([z; NTB], [z; NTB]);
            let k = _mm_set1_epi8(0x0F);
            let e = _mm512_set1_epi32(8);
            for (g, grp) in wt.chunks_exact(16 * 32).enumerate() {
                let (mut g0, mut g1) = ([z; NTB], [z; NTB]);
                for (ii, col) in grp.chunks_exact(16).enumerate() {
                    let b: __m128i = u8x16::from_slice(avx, col).into();
                    let w0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_and_si128(b, k)), e));
                    let hi = _mm_and_si128(_mm_srli_epi16::<4>(b), k);
                    let w1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), e));
                    let i = g * 32 + ii;
                    for t in 0..nt {
                        let xb = _mm512_set1_ps(x[t * ldx + i]);
                        g0[t] = _mm512_fmadd_ps(w0, xb, g0[t]);
                        g1[t] = _mm512_fmadd_ps(w1, xb, g1[t]);
                    }
                }
                let s0: __m512 = f32x16::from_slice(avx, &st[g * 32..g * 32 + 16]).into();
                let s1: __m512 = f32x16::from_slice(avx, &st[g * 32 + 16..g * 32 + 32]).into();
                for t in 0..nt {
                    a0[t] = _mm512_fmadd_ps(g0[t], s0, a0[t]);
                    a1[t] = _mm512_fmadd_ps(g1[t], s1, a1[t]);
                }
            }
            for t in 0..nt {
                let (lo, hi): (f32x16<Avx512>, f32x16<Avx512>) = (a0[t].simd_into(avx), a1[t].simd_into(avx));
                lo.store_slice(&mut buf[t * 32..t * 32 + 16]);
                hi.store_slice(&mut buf[t * 32 + 16..t * 32 + 32]);
            }
        }
    );

    fearless_simd::kernel!(
        /// W8A8 on VNNI: weights [column / 4][32 rows][4] int8, activations u8 (= int8 + 128) per token.
        /// sc: the tile's 32 row scales, corr: its 32 (128 * row sum) as i32 bit patterns.
        #[inline]
        #[allow(clippy::too_many_arguments)]
        pub fn tile_vnni(
            avx: Avx512,
            wt: &[i8],
            xw: &[i32],
            c4: usize,
            xs: &[f32],
            sc: &[f32],
            corr: &[f32],
            nt: usize,
            buf: &mut [f32],
        ) {
            let z = _mm512_setzero_si512();
            let (mut a0, mut a1) = ([z; NTB], [z; NTB]);
            if nt == 1 {
                let (mut b0, mut b1) = (z, z);
                let mut pairs = wt.chunks_exact(256);
                let mut xp = xw[..c4].chunks_exact(2);
                for (p, x2) in (&mut pairs).zip(&mut xp) {
                    let (xa, xb) = (_mm512_set1_epi32(x2[0]), _mm512_set1_epi32(x2[1]));
                    a0[0] = _mm512_dpbusd_epi32(a0[0], xa, i8x64::from_slice(avx, &p[..64]).into());
                    a1[0] = _mm512_dpbusd_epi32(a1[0], xa, i8x64::from_slice(avx, &p[64..128]).into());
                    b0 = _mm512_dpbusd_epi32(b0, xb, i8x64::from_slice(avx, &p[128..192]).into());
                    b1 = _mm512_dpbusd_epi32(b1, xb, i8x64::from_slice(avx, &p[192..]).into());
                }
                for (p, &x1) in pairs.remainder().chunks_exact(128).zip(xp.remainder()) {
                    let xa = _mm512_set1_epi32(x1);
                    a0[0] = _mm512_dpbusd_epi32(a0[0], xa, i8x64::from_slice(avx, &p[..64]).into());
                    a1[0] = _mm512_dpbusd_epi32(a1[0], xa, i8x64::from_slice(avx, &p[64..]).into());
                }
                a0[0] = _mm512_add_epi32(a0[0], b0);
                a1[0] = _mm512_add_epi32(a1[0], b1);
            } else {
                for (c, p) in wt.chunks_exact(128).enumerate() {
                    let w0: __m512i = i8x64::from_slice(avx, &p[..64]).into();
                    let w1: __m512i = i8x64::from_slice(avx, &p[64..]).into();
                    for t in 0..nt {
                        let xb = _mm512_set1_epi32(xw[t * c4 + c]);
                        a0[t] = _mm512_dpbusd_epi32(a0[t], xb, w0);
                        a1[t] = _mm512_dpbusd_epi32(a1[t], xb, w1);
                    }
                }
            }
            let s0: __m512 = f32x16::from_slice(avx, &sc[..16]).into();
            let s1: __m512 = f32x16::from_slice(avx, &sc[16..32]).into();
            let c0 = _mm512_castps_si512(f32x16::from_slice(avx, &corr[..16]).into());
            let c1 = _mm512_castps_si512(f32x16::from_slice(avx, &corr[16..32]).into());
            for t in 0..nt {
                let xsv = _mm512_set1_ps(xs[t]);
                let o0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a0[t], c0)), _mm512_mul_ps(s0, xsv));
                let o1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a1[t], c1)), _mm512_mul_ps(s1, xsv));
                let (lo, hi): (f32x16<Avx512>, f32x16<Avx512>) = (o0.simd_into(avx), o1.simd_into(avx));
                lo.store_slice(&mut buf[t * 32..t * 32 + 16]);
                hi.store_slice(&mut buf[t * 32 + 16..t * 32 + 32]);
            }
        }
    );
    fearless_simd::kernel!(
        /// Decode W8A8 matvec over whole tiles: wt holds the range's tiles back to back, y gets 32 outputs per tile
        /// (the last tile may be cut at y.len()). Four independent dpbusd chains per 16 rows, no per-tile buffer.
        #[inline]
        #[allow(clippy::too_many_arguments)]
        pub fn matvec_vnni(
            avx: Avx512,
            wt: &[i8],
            xw: &[i32],
            c4: usize,
            xs: f32,
            sc: &[f32],
            corr: &[f32],
            y: &mut [f32],
            acc: bool,
        ) {
            let xsv = _mm512_set1_ps(xs);
            let xw = &xw[..c4];
            for (ti, tw) in wt.chunks_exact(c4 * 128).enumerate() {
                let z = _mm512_setzero_si512();
                let (mut a0, mut a1, mut b0, mut b1) = (z, z, z, z);
                let (mut c0, mut c1, mut d0, mut d1) = (z, z, z, z);
                let mut q = tw.chunks_exact(512);
                let mut xq = xw.chunks_exact(4);
                for (p, x4) in (&mut q).zip(&mut xq) {
                    // software prefetch ~2 KB ahead: hardware prefetch alone leaves the L3 / DRAM stream ~5% slower
                    let pf = p.as_ptr().wrapping_add(PF_DIST);
                    for l in 0..8 {
                        _mm_prefetch::<_MM_HINT_T0>(pf.wrapping_add(64 * l));
                    }
                    let (xa, xb) = (_mm512_set1_epi32(x4[0]), _mm512_set1_epi32(x4[1]));
                    let (xc, xd) = (_mm512_set1_epi32(x4[2]), _mm512_set1_epi32(x4[3]));
                    a0 = _mm512_dpbusd_epi32(a0, xa, i8x64::from_slice(avx, &p[..64]).into());
                    a1 = _mm512_dpbusd_epi32(a1, xa, i8x64::from_slice(avx, &p[64..128]).into());
                    b0 = _mm512_dpbusd_epi32(b0, xb, i8x64::from_slice(avx, &p[128..192]).into());
                    b1 = _mm512_dpbusd_epi32(b1, xb, i8x64::from_slice(avx, &p[192..256]).into());
                    c0 = _mm512_dpbusd_epi32(c0, xc, i8x64::from_slice(avx, &p[256..320]).into());
                    c1 = _mm512_dpbusd_epi32(c1, xc, i8x64::from_slice(avx, &p[320..384]).into());
                    d0 = _mm512_dpbusd_epi32(d0, xd, i8x64::from_slice(avx, &p[384..448]).into());
                    d1 = _mm512_dpbusd_epi32(d1, xd, i8x64::from_slice(avx, &p[448..]).into());
                }
                for (p, &x1) in q.remainder().chunks_exact(128).zip(xq.remainder()) {
                    let xa = _mm512_set1_epi32(x1);
                    a0 = _mm512_dpbusd_epi32(a0, xa, i8x64::from_slice(avx, &p[..64]).into());
                    a1 = _mm512_dpbusd_epi32(a1, xa, i8x64::from_slice(avx, &p[64..]).into());
                }
                let s0 = _mm512_add_epi32(_mm512_add_epi32(a0, b0), _mm512_add_epi32(c0, d0));
                let s1 = _mm512_add_epi32(_mm512_add_epi32(a1, b1), _mm512_add_epi32(c1, d1));
                let r = ti * 32;
                let k0 = _mm512_castps_si512(f32x16::from_slice(avx, &corr[r..r + 16]).into());
                let k1 = _mm512_castps_si512(f32x16::from_slice(avx, &corr[r + 16..r + 32]).into());
                let g0 = _mm512_mul_ps(f32x16::from_slice(avx, &sc[r..r + 16]).into(), xsv);
                let g1 = _mm512_mul_ps(f32x16::from_slice(avx, &sc[r + 16..r + 32]).into(), xsv);
                let o0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(s0, k0)), g0);
                let o1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(s1, k1)), g1);
                let (lo, hi): (f32x16<Avx512>, f32x16<Avx512>) = (o0.simd_into(avx), o1.simd_into(avx));
                if r + 32 <= y.len() {
                    let dst = &mut y[r..r + 32];
                    if acc {
                        let (l, h) = dst.split_at_mut(16);
                        (f32x16::from_slice(avx, l) + lo).store_slice(l);
                        (f32x16::from_slice(avx, h) + hi).store_slice(h);
                    } else {
                        lo.store_slice(&mut dst[..16]);
                        hi.store_slice(&mut dst[16..]);
                    }
                } else {
                    let mut buf = [0f32; 32];
                    lo.store_slice(&mut buf[..16]);
                    hi.store_slice(&mut buf[16..]);
                    let n = y.len() - r;
                    let dst = &mut y[r..];
                    if acc {
                        dst.iter_mut().zip(&buf[..n]).for_each(|(a, b)| *a += b);
                    } else {
                        dst.copy_from_slice(&buf[..n]);
                    }
                }
            }
        }
    );
    fearless_simd::kernel!(
        /// One token's int8 activations for W8A8: absmax scale, round to nearest even, +128, 4 bytes per i32 word.
        /// row: n floats; out: cpad / 4 words (columns past n are 0.0, i.e. the zero point 128). Returns the scale.
        #[inline]
        pub fn quant_row(avx: Avx512, row: &[f32], out: &mut [i32]) -> f32 {
            let mut m = _mm512_setzero_ps();
            let mut ch = row.chunks_exact(16);
            for c in &mut ch {
                m = _mm512_max_ps(m, _mm512_abs_ps(f32x16::from_slice(avx, c).into()));
            }
            let amax = ch
                .remainder()
                .iter()
                .fold(_mm512_reduce_max_ps(m), |a, v| a.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let r = _mm512_set1_ps(1.0 / s);
            let off = _mm512_set1_epi32(128);
            let mut tmp = [0f32; 16];
            for (i, o) in out.chunks_exact_mut(4).enumerate() {
                // 16 columns -> 16 bytes -> 4 words
                let base = i * 16;
                let v: __m512 = if base + 16 <= row.len() {
                    f32x16::from_slice(avx, &row[base..base + 16]).into()
                } else {
                    tmp.fill(0.0);
                    if base < row.len() {
                        tmp[..row.len() - base].copy_from_slice(&row[base..]);
                    }
                    f32x16::from_slice(avx, &tmp).into()
                };
                let q = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v, r)), off);
                let b: __m128i = _mm512_cvtusepi32_epi8(q);
                let bytes: [u8; 16] = u8x16::simd_from(avx, b).into();
                for (j, w) in o.iter_mut().enumerate() {
                    *w = i32::from_le_bytes([bytes[4 * j], bytes[4 * j + 1], bytes[4 * j + 2], bytes[4 * j + 3]]);
                }
            }
            s
        }
    );
}

/// Portable tile: dequantize each weight on the fly (W8A8 uses the same quantized activations as the VNNI path).
#[allow(clippy::too_many_arguments)]
fn tile_scalar(m: &MatV, tile: usize, x: &[f32], ldx: usize, xw: &[i32], xs: &[f32], nt: usize, buf: &mut [f32]) {
    let c = m.cpad;
    buf[..nt * 32].fill(0.0);
    for j in 0..32 {
        let r = tile * 32 + j;
        for t in 0..nt {
            let mut acc = 0f32;
            match m.w {
                W::F32(w) => (0..c).for_each(|i| acc += w[(tile * c + i) * 32 + j] * x[t * ldx + i]),
                W::F16(w) => (0..c).for_each(|i| acc += f16_to_f32(w[(tile * c + i) * 32 + j]) * x[t * ldx + i]),
                W::I8(w) => {
                    (0..c).for_each(|i| acc += w[(tile * c + i) * 32 + j] as f32 * x[t * ldx + i]);
                    acc *= m.s[r];
                }
                W::I4(w) => {
                    for g in 0..c / 32 {
                        let mut ga = 0f32;
                        for ii in 0..32 {
                            let i = g * 32 + ii;
                            let b = w[(tile * c + i) * 16 + (j % 16)];
                            let q = if j < 16 { b & 15 } else { b >> 4 };
                            ga += (q as f32 - 8.0) * x[t * ldx + i];
                        }
                        acc += ga * m.s[(tile * (c / 32) + g) * 32 + j];
                    }
                }
                W::W8A8(w) => {
                    let mut ia = 0i32;
                    for i in 0..c {
                        let xu = ((xw[t * (c / 4) + i / 4] as u32 >> (8 * (i % 4))) & 255) as i32;
                        ia += w[tile * c * 32 + (i / 4) * 128 + j * 4 + i % 4] as i32 * (xu - 128);
                    }
                    acc = ia as f32 * m.s[r] * xs[t];
                }
            }
            buf[t * 32 + j] = acc;
        }
    }
}

/// x: activations from the first token of this block (row stride ldx); xw / xs: the block's quantized
/// activations (W8A8 only: 4 u8 per word, cpad / 4 words per token) and their scales.
#[allow(clippy::too_many_arguments)]
pub fn tile(
    level: Level,
    m: &MatV,
    tile: usize,
    x: &[f32],
    ldx: usize,
    xw: &[i32],
    xs: &[f32],
    nt: usize,
    buf: &mut [f32],
) {
    let c = m.cpad;
    #[cfg(target_arch = "x86_64")]
    if let Some(avx) = level.as_avx512() {
        match m.w {
            W::F32(w) => avx512::tile_f32(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf),
            W::F16(w) => avx512::tile_f16(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf),
            W::I8(w) => {
                avx512::tile_i8(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf);
                let s = &m.s[tile * 32..tile * 32 + 32];
                for t in 0..nt {
                    buf[t * 32..t * 32 + 32].iter_mut().zip(s).for_each(|(o, s)| *o *= s);
                }
            }
            W::I4(w) => {
                let st = &m.s[tile * (c / 32) * 32..(tile + 1) * (c / 32) * 32];
                avx512::tile_i4(avx, &w[tile * c * 16..(tile + 1) * c * 16], st, x, ldx, nt, buf)
            }
            W::W8A8(w) => {
                let (sc, corr) = (
                    &m.s[tile * 32..tile * 32 + 32],
                    &m.s[m.rpad + tile * 32..m.rpad + tile * 32 + 32],
                );
                avx512::tile_vnni(
                    avx,
                    &w[tile * c * 32..(tile + 1) * c * 32],
                    xw,
                    c / 4,
                    xs,
                    sc,
                    corr,
                    nt,
                    buf,
                )
            }
        }
        return;
    }
    tile_scalar(m, tile, x, ldx, xw, xs, nt, buf);
}

/// Quantize t token rows of W8A8 activations (see quant_row); portable fallback without AVX-512.
pub fn quant(level: Level, x: &[f32], ldx: usize, t: usize, n: usize, cpad: usize, xw: &mut [i32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if let Some(avx) = level.as_avx512() {
        let c4 = cpad / 4;
        return (0..t)
            .map(|ti| avx512::quant_row(avx, &x[ti * ldx..ti * ldx + n], &mut xw[ti * c4..(ti + 1) * c4]))
            .collect();
    }
    crate::quant_act(x, ldx, t, n, cpad, xw)
}

/// Decode (one token) W8A8 matvec over tiles lo..hi straight into y (= rows lo * 32 .. min(hi * 32, rows)).
/// Returns false when there is no AVX-512 path (the caller then uses the tile loop).
#[allow(clippy::too_many_arguments)]
pub fn matvec_w8a8(
    level: Level,
    m: &MatV,
    lo: usize,
    hi: usize,
    xw: &[i32],
    xs: f32,
    y: &mut [f32],
    acc: bool,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if let (Some(avx), W::W8A8(w)) = (level.as_avx512(), &m.w) {
        let c = m.cpad;
        avx512::matvec_vnni(
            avx,
            &w[lo * c * 32..hi * c * 32],
            xw,
            c / 4,
            xs,
            &m.s[lo * 32..hi * 32],
            &m.s[m.rpad + lo * 32..m.rpad + hi * 32],
            y,
            acc,
        );
        return true;
    }
    let _ = (level, m, lo, hi, xw, xs, y, acc);
    false
}
