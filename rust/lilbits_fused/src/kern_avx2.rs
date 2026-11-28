//! AVX2 + FMA tiles (x86-64-v3: Haswell and later, Zen 1+), plus the AVX-VNNI W8A8 tile (Alder Lake, Raptor Lake,
//! Zen 5 without AVX-512...). Same layouts and results as the AVX-512 tiles: 32 rows = four 8-lane registers.
//!
//! W8A8 without VNNI: vpmaddubsw needs one unsigned operand, so it takes |w| and sign(x, w) (= x * sign w) with the
//! signed activation (the stored u8 xor 0x80): pairs of |w x| <= 127 * 127 never saturate i16. The integer sums
//! equal the VNNI path's (u8 sums minus 128 * row sum) exactly, so all W8A8 paths are bit-identical.

use core::arch::x86_64::*;
use fearless_simd::{Avx2, f32x8, i8x16, i8x32, prelude::*, u8x16, u16x8};

type R = [__m256; 4];

fearless_simd::kernel!(
    #[inline]
    fn store(avx: Avx2, a: &R, out: &mut [f32]) {
        for (r, v) in a.iter().enumerate() {
            let v: f32x8<Avx2> = (*v).simd_into(avx);
            v.store_slice(&mut out[r * 8..r * 8 + 8]);
        }
    }
);

/// Float-accumulating tile for formats with 32 values per column (f32, f16, int8 weights).
macro_rules! float_tile {
    ($name:ident, $ty:ty, |$avx:ident, $col:ident| $load:expr) => {
        fearless_simd::kernel!(
            #[inline]
            pub fn $name($avx: Avx2, wt: &[$ty], x: &[f32], ldx: usize, nt: usize, buf: &mut [f32]) {
                let ld = |$col: &[$ty]| -> R { $load };
                let z = _mm256_setzero_ps();
                let mut t = 0;
                while t < nt {
                    let (mut a, mut b) = ([z; 4], [z; 4]);
                    if nt - t >= 2 {
                        // two tokens per pass over the tile (8 accumulators + 4 weights fit the 16 registers)
                        for (i, col) in wt.chunks_exact(32).enumerate() {
                            let w = ld(col);
                            let (xa, xb) = (
                                _mm256_set1_ps(x[t * ldx + i]),
                                _mm256_set1_ps(x[(t + 1) * ldx + i]),
                            );
                            for r in 0..4 {
                                a[r] = _mm256_fmadd_ps(w[r], xa, a[r]);
                                b[r] = _mm256_fmadd_ps(w[r], xb, b[r]);
                            }
                        }
                        store($avx, &a, &mut buf[t * 32..t * 32 + 32]);
                        store($avx, &b, &mut buf[(t + 1) * 32..(t + 1) * 32 + 32]);
                        t += 2;
                    } else {
                        // one token: even / odd columns in two independent chains
                        let xr = &x[t * ldx..];
                        let mut pairs = wt.chunks_exact(64);
                        let mut i = 0;
                        for p in &mut pairs {
                            let (w, v) = (ld(&p[..32]), ld(&p[32..]));
                            let (xa, xb) = (_mm256_set1_ps(xr[i]), _mm256_set1_ps(xr[i + 1]));
                            for r in 0..4 {
                                a[r] = _mm256_fmadd_ps(w[r], xa, a[r]);
                                b[r] = _mm256_fmadd_ps(v[r], xb, b[r]);
                            }
                            i += 2;
                        }
                        for col in pairs.remainder().chunks_exact(32) {
                            let w = ld(col);
                            let xa = _mm256_set1_ps(xr[i]);
                            for r in 0..4 {
                                a[r] = _mm256_fmadd_ps(w[r], xa, a[r]);
                            }
                            i += 1;
                        }
                        for r in 0..4 {
                            a[r] = _mm256_add_ps(a[r], b[r]);
                        }
                        store($avx, &a, &mut buf[t * 32..t * 32 + 32]);
                        t += 1;
                    }
                }
            }
        );
    };
}

float_tile!(tile_f32, f32, |avx, col| core::array::from_fn(|r| f32x8::from_slice(
    avx,
    &col[r * 8..r * 8 + 8]
)
.into()));
float_tile!(tile_f16, u16, |avx, col| core::array::from_fn(|r| _mm256_cvtph_ps(
    u16x8::from_slice(avx, &col[r * 8..r * 8 + 8]).into()
)));
float_tile!(tile_i8, i8, |avx, col| {
    let (lo, hi): (__m128i, __m128i) = (
        i8x16::from_slice(avx, &col[..16]).into(),
        i8x16::from_slice(avx, &col[16..32]).into(),
    );
    let f = |v: __m128i| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
    [f(lo), f(_mm_srli_si128::<8>(lo)), f(hi), f(_mm_srli_si128::<8>(hi))]
});

fearless_simd::kernel!(
    /// int4: 16 bytes per column (low nibble rows 0..15, high 16..31, value - 8), a scale per 32-column group.
    #[inline]
    pub fn tile_i4(avx: Avx2, wt: &[u8], st: &[f32], x: &[f32], ldx: usize, nt: usize, buf: &mut [f32]) {
        let z = _mm256_setzero_ps();
        let k = _mm_set1_epi8(0x0F);
        let e = _mm256_set1_epi32(8);
        let f = |v: __m128i| _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(v), e));
        for t in 0..nt {
            let mut a = [z; 4];
            for (g, grp) in wt.chunks_exact(16 * 32).enumerate() {
                let mut ga = [z; 4];
                for (ii, col) in grp.chunks_exact(16).enumerate() {
                    let b: __m128i = u8x16::from_slice(avx, col).into();
                    let (lo, hi) = (_mm_and_si128(b, k), _mm_and_si128(_mm_srli_epi16::<4>(b), k));
                    let w = [f(lo), f(_mm_srli_si128::<8>(lo)), f(hi), f(_mm_srli_si128::<8>(hi))];
                    let xb = _mm256_set1_ps(x[t * ldx + g * 32 + ii]);
                    for r in 0..4 {
                        ga[r] = _mm256_fmadd_ps(w[r], xb, ga[r]);
                    }
                }
                for r in 0..4 {
                    let s: __m256 = f32x8::from_slice(avx, &st[g * 32 + r * 8..g * 32 + r * 8 + 8]).into();
                    a[r] = _mm256_fmadd_ps(ga[r], s, a[r]);
                }
            }
            store(avx, &a, &mut buf[t * 32..t * 32 + 32]);
        }
    }
);

/// W8A8 tile body shared by the AVX2 and AVX-VNNI kernels. `$xf`: the broadcast word as the dot product wants it;
/// `$dot(acc, x, w)`: acc += 4-byte dot products per i32 lane; `$corr`: subtract 128 * row sum (u8 activations).
macro_rules! w8a8_body {
    ($avx:ident, $wt:ident, $xw:ident, $c4:ident, $xs:ident, $sc:ident, $corr:ident, $nt:ident, $buf:ident,
     $xf:expr, $dot:expr, $use_corr:expr) => {{
        let z = _mm256_setzero_si256();
        let xf = $xf;
        let dot = $dot;
        for t in 0..$nt {
            let xr = &$xw[t * $c4..(t + 1) * $c4];
            let (mut a, mut b) = ([z; 4], [z; 4]);
            let mut pairs = $wt.chunks_exact(256);
            let mut xp = xr.chunks_exact(2);
            for (p, x2) in (&mut pairs).zip(&mut xp) {
                let (xa, xb) = (_mm256_set1_epi32(xf(x2[0])), _mm256_set1_epi32(xf(x2[1])));
                for r in 0..4 {
                    a[r] = dot(a[r], xa, i8x32::from_slice($avx, &p[r * 32..r * 32 + 32]).into());
                    b[r] = dot(b[r], xb, i8x32::from_slice($avx, &p[128 + r * 32..160 + r * 32]).into());
                }
            }
            for (p, &x1) in pairs.remainder().chunks_exact(128).zip(xp.remainder()) {
                let xa = _mm256_set1_epi32(xf(x1));
                for r in 0..4 {
                    a[r] = dot(a[r], xa, i8x32::from_slice($avx, &p[r * 32..r * 32 + 32]).into());
                }
            }
            let xsv = _mm256_set1_ps($xs[t]);
            let mut o = [_mm256_setzero_ps(); 4];
            for r in 0..4 {
                let mut acc = _mm256_add_epi32(a[r], b[r]);
                if $use_corr {
                    let c: __m256 = f32x8::from_slice($avx, &$corr[r * 8..r * 8 + 8]).into();
                    acc = _mm256_sub_epi32(acc, _mm256_castps_si256(c));
                }
                let s: __m256 = f32x8::from_slice($avx, &$sc[r * 8..r * 8 + 8]).into();
                o[r] = _mm256_mul_ps(_mm256_cvtepi32_ps(acc), _mm256_mul_ps(s, xsv));
            }
            store($avx, &o, &mut $buf[t * 32..t * 32 + 32]);
        }
    }};
}

fearless_simd::kernel!(
    /// W8A8 without VNNI (weights [column / 4][32 rows][4] int8; xw: u8 activations, 4 per word).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn tile_w8a8(
        avx: Avx2,
        wt: &[i8],
        xw: &[i32],
        c4: usize,
        xs: &[f32],
        sc: &[f32],
        corr: &[f32],
        nt: usize,
        buf: &mut [f32],
    ) {
        let ones = _mm256_set1_epi16(1);
        w8a8_body!(
            avx,
            wt,
            xw,
            c4,
            xs,
            sc,
            corr,
            nt,
            buf,
            |w: i32| w ^ 0x8080_8080u32 as i32,
            |acc: __m256i, x: __m256i, w: __m256i| {
                let p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
                _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones))
            },
            false
        )
    }
);

/// W8A8 on AVX-VNNI (256-bit vpdpbusd, u8 activations x int8 weights).
///
/// # Safety
/// The CPU must support AVX2 (x86-64-v3) and AVX-VNNI (`is_x86_feature_detected!("avxvnni")`).
#[target_feature(enable = "avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,fxsr,lzcnt,movbe,popcnt,xsave,avxvnni")]
#[allow(clippy::too_many_arguments)]
pub fn tile_vnni(wt: &[i8], xw: &[i32], c4: usize, xs: &[f32], sc: &[f32], corr: &[f32], nt: usize, buf: &mut [f32]) {
    let avx = Avx2::assume_supported();
    w8a8_body!(
        avx,
        wt,
        xw,
        c4,
        xs,
        sc,
        corr,
        nt,
        buf,
        |w: i32| w,
        |acc: __m256i, x: __m256i, w: __m256i| _mm256_dpbusd_avx_epi32(acc, x, w),
        true
    )
}
