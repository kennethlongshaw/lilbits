//! NEON tiles for aarch64 (32 rows = eight 4-lane registers), and the W8A8 tile with sdot (ARMv8.2 dotprod).
//!
//! W8A8 uses signed activations (the stored u8 xor 0x80), so no 128 * row-sum correction: smull / sadalp (plain
//! NEON) or sdot both give the same integer sums as the x86 paths, so W8A8 results are bit-identical everywhere.
//! sdot and the f16 -> f32 widening go through `asm!`: their intrinsics are not stable Rust yet.

use core::arch::aarch64::*;
use core::arch::asm;
use fearless_simd::{Neon, f32x4, i8x16, prelude::*, u8x16, u16x8};

type R = [float32x4_t; 8];

fearless_simd::kernel!(
    #[inline]
    fn store(neon: Neon, a: &R, out: &mut [f32]) {
        for (r, v) in a.iter().enumerate() {
            let v: f32x4<Neon> = (*v).simd_into(neon);
            v.store_slice(&mut out[r * 4..r * 4 + 4]);
        }
    }
);

/// f16 x 8 -> two f32 x 4 (fcvtl / fcvtl2).
#[inline(always)]
fn h2f(v: uint16x8_t) -> (float32x4_t, float32x4_t) {
    let (lo, hi): (float32x4_t, float32x4_t);
    // Safety: fcvtl / fcvtl2 are base ARMv8 Advanced SIMD; register-only, no memory or flags.
    unsafe {
        asm!(
            "fcvtl {lo:v}.4s, {v:v}.4h",
            "fcvtl2 {hi:v}.4s, {v:v}.8h",
            v = in(vreg) v,
            lo = out(vreg) lo,
            hi = out(vreg) hi,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    (lo, hi)
}

macro_rules! float_tile {
    ($name:ident, $ty:ty, |$neon:ident, $col:ident| $load:expr) => {
        fearless_simd::kernel!(
            #[inline]
            pub fn $name($neon: Neon, wt: &[$ty], x: &[f32], ldx: usize, nt: usize, buf: &mut [f32]) {
                let ld = |$col: &[$ty]| -> R { $load };
                let z = vdupq_n_f32(0.0);
                let mut t = 0;
                while t < nt {
                    let (mut a, mut b) = ([z; 8], [z; 8]);
                    if nt - t >= 2 {
                        for (i, col) in wt.chunks_exact(32).enumerate() {
                            let w = ld(col);
                            let (xa, xb) = (x[t * ldx + i], x[(t + 1) * ldx + i]);
                            for r in 0..8 {
                                a[r] = vfmaq_n_f32(a[r], w[r], xa);
                                b[r] = vfmaq_n_f32(b[r], w[r], xb);
                            }
                        }
                        store($neon, &a, &mut buf[t * 32..t * 32 + 32]);
                        store($neon, &b, &mut buf[(t + 1) * 32..(t + 1) * 32 + 32]);
                        t += 2;
                    } else {
                        let xr = &x[t * ldx..];
                        let mut pairs = wt.chunks_exact(64);
                        let mut i = 0;
                        for p in &mut pairs {
                            let (w, v) = (ld(&p[..32]), ld(&p[32..]));
                            for r in 0..8 {
                                a[r] = vfmaq_n_f32(a[r], w[r], xr[i]);
                                b[r] = vfmaq_n_f32(b[r], v[r], xr[i + 1]);
                            }
                            i += 2;
                        }
                        for col in pairs.remainder().chunks_exact(32) {
                            let w = ld(col);
                            for r in 0..8 {
                                a[r] = vfmaq_n_f32(a[r], w[r], xr[i]);
                            }
                            i += 1;
                        }
                        for r in 0..8 {
                            a[r] = vaddq_f32(a[r], b[r]);
                        }
                        store($neon, &a, &mut buf[t * 32..t * 32 + 32]);
                        t += 1;
                    }
                }
            }
        );
    };
}

float_tile!(tile_f32, f32, |neon, col| core::array::from_fn(|r| f32x4::from_slice(
    neon,
    &col[r * 4..r * 4 + 4]
)
.into()));
float_tile!(tile_f16, u16, |neon, col| {
    let mut w = [vdupq_n_f32(0.0); 8];
    for q in 0..4 {
        let (lo, hi) = h2f(u16x8::from_slice(neon, &col[q * 8..q * 8 + 8]).into());
        (w[2 * q], w[2 * q + 1]) = (lo, hi);
    }
    w
});
float_tile!(tile_i8, i8, |neon, col| {
    let mut w = [vdupq_n_f32(0.0); 8];
    for q in 0..2 {
        let b: int8x16_t = i8x16::from_slice(neon, &col[q * 16..q * 16 + 16]).into();
        let (l, h) = (vmovl_s8(vget_low_s8(b)), vmovl_high_s8(b));
        w[4 * q] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(l)));
        w[4 * q + 1] = vcvtq_f32_s32(vmovl_high_s16(l));
        w[4 * q + 2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(h)));
        w[4 * q + 3] = vcvtq_f32_s32(vmovl_high_s16(h));
    }
    w
});

fearless_simd::kernel!(
    /// int4: 16 bytes per column (low nibble rows 0..15, high 16..31, value - 8), a scale per 32-column group.
    #[inline]
    pub fn tile_i4(neon: Neon, wt: &[u8], st: &[f32], x: &[f32], ldx: usize, nt: usize, buf: &mut [f32]) {
        let z = vdupq_n_f32(0.0);
        let k = vdupq_n_u8(0x0F);
        let e = vdupq_n_s32(8);
        let wid = |b: uint8x16_t, w: &mut [float32x4_t]| {
            let (l, h) = (vmovl_u8(vget_low_u8(b)), vmovl_high_u8(b));
            let f = |v: uint32x4_t| vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(v), e));
            w[0] = f(vmovl_u16(vget_low_u16(l)));
            w[1] = f(vmovl_high_u16(l));
            w[2] = f(vmovl_u16(vget_low_u16(h)));
            w[3] = f(vmovl_high_u16(h));
        };
        for t in 0..nt {
            let mut a = [z; 8];
            for (g, grp) in wt.chunks_exact(16 * 32).enumerate() {
                let mut ga = [z; 8];
                for (ii, col) in grp.chunks_exact(16).enumerate() {
                    let b: uint8x16_t = u8x16::from_slice(neon, col).into();
                    let mut w = [z; 8];
                    wid(vandq_u8(b, k), &mut w[..4]);
                    wid(vshrq_n_u8::<4>(b), &mut w[4..]);
                    let xv = x[t * ldx + g * 32 + ii];
                    for r in 0..8 {
                        ga[r] = vfmaq_n_f32(ga[r], w[r], xv);
                    }
                }
                for r in 0..8 {
                    let s: float32x4_t = f32x4::from_slice(neon, &st[g * 32 + r * 4..g * 32 + r * 4 + 4]).into();
                    a[r] = vfmaq_f32(a[r], ga[r], s);
                }
            }
            store(neon, &a, &mut buf[t * 32..t * 32 + 32]);
        }
    }
);

fearless_simd::kernel!(
    /// Scale the integer row sums: out = sum * (row scale * token scale), 32 rows.
    #[inline]
    fn finish(neon: Neon, acc: &[int32x4_t; 8], sc: &[f32], xs: f32, out: &mut [f32]) {
        let mut o = [vdupq_n_f32(0.0); 8];
        for r in 0..8 {
            let s: float32x4_t = f32x4::from_slice(neon, &sc[r * 4..r * 4 + 4]).into();
            o[r] = vmulq_f32(vcvtq_f32_s32(acc[r]), vmulq_n_f32(s, xs));
        }
        store(neon, &o, out);
    }
);

fearless_simd::kernel!(
    /// W8A8 without dotprod: smull (8 x i8 -> i16) and sadalp (pairwise add into i32), then a pairwise reduce.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn tile_w8a8(neon: Neon, wt: &[i8], xw: &[i32], c4: usize, xs: &[f32], sc: &[f32], nt: usize, buf: &mut [f32]) {
        let z = vdupq_n_s32(0);
        for t in 0..nt {
            // acc[2r]: rows 4r, 4r+1 (two partial sums each); acc[2r+1]: rows 4r+2, 4r+3
            let mut acc = [z; 16];
            for (p, &xv) in wt.chunks_exact(128).zip(&xw[t * c4..(t + 1) * c4]) {
                let x = vreinterpretq_s8_s32(vdupq_n_s32(xv ^ 0x8080_8080u32 as i32));
                let xl = vget_low_s8(x);
                for r in 0..8 {
                    let w: int8x16_t = i8x16::from_slice(neon, &p[r * 16..r * 16 + 16]).into();
                    acc[2 * r] = vpadalq_s16(acc[2 * r], vmull_s8(vget_low_s8(w), xl));
                    acc[2 * r + 1] = vpadalq_s16(acc[2 * r + 1], vmull_high_s8(w, x));
                }
            }
            let sums: [int32x4_t; 8] = core::array::from_fn(|r| vpaddq_s32(acc[2 * r], acc[2 * r + 1]));
            finish(neon, &sums, sc, xs[t], &mut buf[t * 32..t * 32 + 32]);
        }
    }
);

/// acc += sdot(w, x) per 4-byte lane.
#[inline(always)]
fn sdot(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    let mut a = acc;
    // Safety: only called from `tile_sdot`, which requires (and enables) the dotprod feature; registers only.
    unsafe {
        asm!(
            ".arch_extension dotprod",
            "sdot {a:v}.4s, {w:v}.16b, {x:v}.16b",
            a = inout(vreg) a,
            w = in(vreg) w,
            x = in(vreg) x,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    a
}

/// W8A8 with sdot.
///
/// # Safety
/// The CPU must support dotprod (`is_aarch64_feature_detected!("dotprod")`).
#[target_feature(enable = "neon,dotprod")]
#[allow(clippy::too_many_arguments)]
pub fn tile_sdot(wt: &[i8], xw: &[i32], c4: usize, xs: &[f32], sc: &[f32], nt: usize, buf: &mut [f32]) {
    let neon = Neon::assume_supported();
    let z = vdupq_n_s32(0);
    for t in 0..nt {
        let xr = &xw[t * c4..(t + 1) * c4];
        let (mut a, mut b) = ([z; 8], [z; 8]);
        let mut pairs = wt.chunks_exact(256);
        let mut xp = xr.chunks_exact(2);
        let ld = |p: &[i8]| -> int8x16_t { i8x16::from_slice(neon, p).into() };
        let bx = |v: i32| vreinterpretq_s8_s32(vdupq_n_s32(v ^ 0x8080_8080u32 as i32));
        for (p, x2) in (&mut pairs).zip(&mut xp) {
            let (xa, xb) = (bx(x2[0]), bx(x2[1]));
            for r in 0..8 {
                a[r] = sdot(a[r], ld(&p[r * 16..r * 16 + 16]), xa);
                b[r] = sdot(b[r], ld(&p[128 + r * 16..144 + r * 16]), xb);
            }
        }
        for (p, &x1) in pairs.remainder().chunks_exact(128).zip(xp.remainder()) {
            let xa = bx(x1);
            for r in 0..8 {
                a[r] = sdot(a[r], ld(&p[r * 16..r * 16 + 16]), xa);
            }
        }
        let sums: [int32x4_t; 8] = core::array::from_fn(|r| vaddq_s32(a[r], b[r]));
        finish(neon, &sums, sc, xs[t], &mut buf[t * 32..t * 32 + 32]);
    }
}
