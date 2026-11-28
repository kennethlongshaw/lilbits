//! One 32-row tile of a tile-major matmul for up to NTB tokens: out[t * 32 + j] = W[tile * 32 + j] . x[t].
//!
//! AVX-512 kernels (`fearless_simd::kernel!`: the intrinsics are safe to call inside, loads come from slices through
//! fearless_simd's vector types) stream the tile's weights once, broadcast one activation per column and keep 32
//! outputs per token in two zmm accumulators. Anything else runs the portable scalar tile.

use crate::isa::Isa;
use crate::{MatV, W, f16_to_f32};
use fearless_simd::Level;

pub const NTB: usize = 4;
#[cfg(target_arch = "x86_64")]
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
                    acc = ia as f32 * (m.s[r] * xs[t]); // same rounding as the SIMD kernels
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
    tile_isa(crate::isa::current(), level, m, tile, x, ldx, xw, xs, nt, buf)
}

/// `tile` with the kernel family given (`level` must be `isa::level(isa)`).
#[allow(clippy::too_many_arguments)]
pub fn tile_isa(
    isa: Isa,
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
    let _ = (isa, level);
    // per-format slices of this tile
    let (sc, corr) = match m.w {
        W::W8A8(_) => (
            &m.s[tile * 32..tile * 32 + 32],
            &m.s[m.rpad + tile * 32..m.rpad + tile * 32 + 32],
        ),
        W::I8(_) => (&m.s[tile * 32..tile * 32 + 32], &[][..]),
        W::I4(_) => (&m.s[tile * (c / 32) * 32..(tile + 1) * (c / 32) * 32], &[][..]),
        _ => (&[][..], &[][..]),
    };
    let scale_rows = |buf: &mut [f32]| {
        for t in 0..nt {
            buf[t * 32..t * 32 + 32].iter_mut().zip(sc).for_each(|(o, s)| *o *= s);
        }
    };
    #[cfg(target_arch = "x86_64")]
    if let (Isa::Avx512, Some(avx)) = (isa, level.as_avx512()) {
        match m.w {
            W::F32(w) => avx512::tile_f32(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf),
            W::F16(w) => avx512::tile_f16(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf),
            W::I8(w) => {
                avx512::tile_i8(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf);
                scale_rows(buf);
            }
            W::I4(w) => avx512::tile_i4(avx, &w[tile * c * 16..(tile + 1) * c * 16], sc, x, ldx, nt, buf),
            W::W8A8(w) => avx512::tile_vnni(
                avx,
                &w[tile * c * 32..(tile + 1) * c * 32],
                xw,
                c / 4,
                xs,
                sc,
                corr,
                nt,
                buf,
            ),
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if let (Isa::Avx2 | Isa::AvxVnni, Some(avx)) = (isa, level.as_avx2()) {
        use crate::kern_avx2 as k;
        match m.w {
            W::F32(w) => k::tile_f32(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf),
            W::F16(w) => k::tile_f16(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf),
            W::I8(w) => {
                k::tile_i8(avx, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf);
                scale_rows(buf);
            }
            W::I4(w) => k::tile_i4(avx, &w[tile * c * 16..(tile + 1) * c * 16], sc, x, ldx, nt, buf),
            W::W8A8(w) => {
                let wt = &w[tile * c * 32..(tile + 1) * c * 32];
                if isa == Isa::AvxVnni {
                    // Safety: Isa::AvxVnni is only selected when AVX2 and AVX-VNNI were detected (isa::supported).
                    unsafe { k::tile_vnni(wt, xw, c / 4, xs, sc, corr, nt, buf) }
                } else {
                    k::tile_w8a8(avx, wt, xw, c / 4, xs, sc, corr, nt, buf)
                }
            }
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if let (Isa::Neon | Isa::NeonDot, Some(neon)) = (isa, level.as_neon()) {
        use crate::kern_neon as k;
        match m.w {
            W::F32(w) => k::tile_f32(neon, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf),
            W::F16(w) => k::tile_f16(neon, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf),
            W::I8(w) => {
                k::tile_i8(neon, &w[tile * c * 32..(tile + 1) * c * 32], x, ldx, nt, buf);
                scale_rows(buf);
            }
            W::I4(w) => k::tile_i4(neon, &w[tile * c * 16..(tile + 1) * c * 16], sc, x, ldx, nt, buf),
            W::W8A8(w) => {
                let wt = &w[tile * c * 32..(tile + 1) * c * 32];
                if isa == Isa::NeonDot {
                    // Safety: Isa::NeonDot is only selected when dotprod was detected (isa::supported).
                    unsafe { k::tile_sdot(wt, xw, c / 4, xs, sc, nt, buf) }
                } else {
                    k::tile_w8a8(neon, wt, xw, c / 4, xs, sc, nt, buf)
                }
            }
        }
        return;
    }
    let _ = (corr, scale_rows);
    tile_scalar(m, tile, x, ldx, xw, xs, nt, buf);
}

/// Quantize t token rows of W8A8 activations (see quant_row); portable fallback without AVX-512.
pub fn quant(level: Level, x: &[f32], ldx: usize, t: usize, n: usize, cpad: usize, xw: &mut [i32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if let (Isa::Avx512, Some(avx)) = (crate::isa::current(), level.as_avx512()) {
        let c4 = cpad / 4;
        return (0..t)
            .map(|ti| avx512::quant_row(avx, &x[ti * ldx..ti * ldx + n], &mut xw[ti * c4..(ti + 1) * c4]))
            .collect();
    }
    crate::quant_simd(level, x, ldx, t, n, cpad, xw)
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
    if let (Isa::Avx512, Some(avx), W::W8A8(w)) = (crate::isa::current(), level.as_avx512(), &m.w) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::{self, ALL};

    /// xorshift
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

    /// Every kernel family this CPU runs matches the scalar reference, for each format, token count and a
    /// column count with an odd number of column pairs; W8A8 bit-exactly.
    #[test]
    fn kernels_match_scalar() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (ntile, cpad) = (3usize, 224usize);
        let rpad = ntile * 32;
        let n = rpad * cpad;
        let f32w: Vec<f32> = (0..n).map(|_| rng.f()).collect();
        let f16w: Vec<u16> = f32w.iter().map(|&v| half_bits(v)).collect();
        let i8w: Vec<i8> = (0..n).map(|_| (rng.next() % 255) as i8).collect();
        let i4w: Vec<u8> = (0..n / 2).map(|_| rng.next() as u8).collect();
        let rows_s: Vec<f32> = (0..rpad).map(|_| 0.01 + rng.f().abs() * 0.02).collect();
        let i4s: Vec<f32> = (0..rpad * cpad / 32).map(|_| 0.01 + rng.f().abs() * 0.02).collect();
        // W8A8: scales then corr = 128 * row sum (as f32 bits), layout [tile][c / 4][32][4]
        let mut w8s = rows_s.clone();
        for r in 0..rpad {
            let (tl, j) = (r / 32, r % 32);
            let sum: i32 = (0..cpad)
                .map(|i| i8w[tl * cpad * 32 + (i / 4) * 128 + j * 4 + i % 4] as i32)
                .sum();
            w8s.push(f32::from_bits((128 * sum) as u32));
        }
        let mats = [
            (
                "f32",
                MatV {
                    rows: 90,
                    cpad,
                    rpad,
                    w: W::F32(&f32w),
                    s: &[],
                },
            ),
            (
                "f16",
                MatV {
                    rows: 90,
                    cpad,
                    rpad,
                    w: W::F16(&f16w),
                    s: &[],
                },
            ),
            (
                "i8",
                MatV {
                    rows: 90,
                    cpad,
                    rpad,
                    w: W::I8(&i8w),
                    s: &rows_s,
                },
            ),
            (
                "i4",
                MatV {
                    rows: 90,
                    cpad,
                    rpad,
                    w: W::I4(&i4w),
                    s: &i4s,
                },
            ),
            (
                "w8a8",
                MatV {
                    rows: 90,
                    cpad,
                    rpad,
                    w: W::W8A8(&i8w),
                    s: &w8s,
                },
            ),
        ];
        let cols = 200;
        let mut x = vec![0f32; NTB * cpad];
        for t in 0..NTB {
            for i in 0..cols {
                x[t * cpad + i] = rng.f();
            }
        }
        let mut xw = vec![0i32; NTB * cpad / 4];
        let xs = crate::quant_act(&x, cpad, NTB, cols, cpad, &mut xw);
        let mut tested = vec![];
        for isa in ALL.into_iter().filter(|&i| isa::supported(i)) {
            let level = isa::level(isa);
            tested.push(isa);
            for (name, m) in &mats {
                for nt in 1..=NTB {
                    for tl in 0..ntile {
                        let (mut want, mut got) = ([0f32; 32 * NTB], [0f32; 32 * NTB]);
                        tile_scalar(m, tl, &x, cpad, &xw, &xs, nt, &mut want);
                        tile_isa(isa, level, m, tl, &x, cpad, &xw, &xs, nt, &mut got);
                        for k in 0..nt * 32 {
                            let (a, b) = (want[k], got[k]);
                            let ok = if *name == "w8a8" {
                                a.to_bits() == b.to_bits() // same integer sums, same final rounding
                            } else {
                                (a - b).abs() <= 1e-4 * a.abs().max(1.0)
                            };
                            assert!(ok, "{isa:?} {name} nt={nt} tile={tl} row={k}: {a} vs {b}");
                        }
                    }
                }
            }
        }
        eprintln!("kernel families tested: {tested:?}");
    }

    /// The W8A8 families agree bit for bit (the forward pass relies on this for identical output across CPUs).
    #[test]
    fn w8a8_families_bit_identical() {
        let mut rng = Rng(12345);
        let (cpad, rpad) = (544usize, 64usize);
        let w: Vec<i8> = (0..rpad * cpad).map(|_| (rng.next() % 255) as i8).collect();
        let mut s: Vec<f32> = (0..rpad).map(|_| 0.001 + rng.f().abs() * 0.01).collect();
        for r in 0..rpad {
            let (tl, j) = (r / 32, r % 32);
            let sum: i32 = (0..cpad)
                .map(|i| w[tl * cpad * 32 + (i / 4) * 128 + j * 4 + i % 4] as i32)
                .sum();
            s.push(f32::from_bits((128 * sum) as u32));
        }
        let m = MatV {
            rows: rpad,
            cpad,
            rpad,
            w: W::W8A8(&w),
            s: &s,
        };
        let x: Vec<f32> = (0..NTB * cpad)
            .map(|i| if i % cpad < 528 { rng.f() * 3.0 } else { 0.0 })
            .collect();
        let mut xw = vec![0i32; NTB * cpad / 4];
        let xs = crate::quant_act(&x, cpad, NTB, 528, cpad, &mut xw);
        let mut first: Option<(Isa, Vec<f32>)> = None;
        for isa in ALL
            .into_iter()
            .filter(|&i| isa::supported(i) && i != Isa::Scalar && i != Isa::Sse)
        {
            let mut got = vec![0f32; 2 * 32 * NTB];
            for tl in 0..2 {
                tile_isa(
                    isa,
                    isa::level(isa),
                    &m,
                    tl,
                    &x,
                    cpad,
                    &xw,
                    &xs,
                    NTB,
                    &mut got[tl * 32 * NTB..],
                );
            }
            match &first {
                None => first = Some((isa, got)),
                Some((i0, g0)) => assert!(g0 == &got, "{isa:?} differs from {i0:?}"),
            }
        }
    }

    #[test]
    fn quant_paths_agree() {
        let mut rng = Rng(7);
        let (t, n, cpad) = (5usize, 203usize, 224usize);
        let x: Vec<f32> = (0..t * cpad).map(|_| rng.f() * 4.0).collect();
        let (mut a, mut b) = (vec![0i32; t * cpad / 4], vec![0i32; t * cpad / 4]);
        let sa = crate::quant_act(&x, cpad, t, n, cpad, &mut a);
        for isa in ALL.into_iter().filter(|&i| isa::supported(i)) {
            let sb = quant(isa::level(isa), &x, cpad, t, n, cpad, &mut b);
            assert!(sa == sb && a == b, "{isa:?}");
        }
    }

    fn half_bits(v: f32) -> u16 {
        // round-to-nearest f32 -> f16 for normal values in (-2, 2)
        let b = v.to_bits();
        let s = ((b >> 16) & 0x8000) as u16;
        let e = ((b >> 23) & 255) as i32 - 127 + 15;
        if e <= 0 {
            return s;
        }
        let m = b & 0x7f_ffff;
        let h = ((e as u32) << 10) | (m >> 13);
        let h = h + (((m >> 12) & 1) & (((m & 0x1fff) != 0x1000) as u32 | (h & 1)));
        s | h as u16
    }
}
