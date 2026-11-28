//! Which matmul kernels run: picked once from the CPU, or forced (tests, benchmarks) through `set_simd`.
//!
//! The matmul kernel family and fearless_simd's `Level` (used by the elementwise code) are chosen together:
//!
//! | Isa | matmul kernels | elementwise Level |
//! |---|---|---|
//! | Scalar | portable scalar | Fallback (no SIMD) |
//! | Sse | portable scalar | Sse4_2 |
//! | Avx2 | AVX2 + FMA (W8A8 through vpmaddubsw) | Avx2 |
//! | AvxVnni | AVX2 + AVX-VNNI (256-bit vpdpbusd: Alder Lake / Raptor Lake, Zen 5 without AVX-512) | Avx2 |
//! | Avx512 | AVX-512 + VNNI | Avx512 |
//! | Neon | NEON (W8A8 through smull / sadalp) | Neon |
//! | NeonDot | NEON + sdot (ARMv8.2 dotprod: Apple M, Graviton 2+, recent Cortex / Snapdragon) | Neon |

use fearless_simd::Level;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, Ordering::Relaxed};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Isa {
    Scalar = 1,
    Sse = 2,
    Avx2 = 3,
    AvxVnni = 4,
    Avx512 = 5,
    Neon = 6,
    NeonDot = 7,
}

pub const ALL: [Isa; 7] = [
    Isa::Scalar,
    Isa::Sse,
    Isa::Avx2,
    Isa::AvxVnni,
    Isa::Avx512,
    Isa::Neon,
    Isa::NeonDot,
];

static FORCED: AtomicI32 = AtomicI32::new(0);

fn from_code(c: i32) -> Option<Isa> {
    ALL.into_iter().find(|i| *i as i32 == c)
}

#[cfg(target_arch = "x86_64")]
fn vnni256() -> bool {
    std::arch::is_x86_feature_detected!("avxvnni")
}

#[cfg(target_arch = "aarch64")]
fn dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

/// The best kernels this CPU supports.
pub fn detected() -> Isa {
    static D: OnceLock<Isa> = OnceLock::new();
    *D.get_or_init(|| {
        let base = Level::new();
        #[cfg(target_arch = "x86_64")]
        {
            if base.as_avx512().is_some() {
                return Isa::Avx512;
            }
            if base.as_avx2().is_some() {
                return if vnni256() { Isa::AvxVnni } else { Isa::Avx2 };
            }
            if base.as_sse4_2().is_some() {
                return Isa::Sse;
            }
        }
        #[cfg(target_arch = "aarch64")]
        if base.as_neon().is_some() {
            return if dotprod() { Isa::NeonDot } else { Isa::Neon };
        }
        let _ = base;
        Isa::Scalar
    })
}

/// Whether `isa` can run here.
pub fn supported(isa: Isa) -> bool {
    let base = Level::new();
    let _ = base;
    match isa {
        Isa::Scalar => true,
        #[cfg(target_arch = "x86_64")]
        Isa::Sse => base.as_sse4_2().is_some(),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => base.as_avx2().is_some(),
        #[cfg(target_arch = "x86_64")]
        Isa::AvxVnni => base.as_avx2().is_some() && vnni256(),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => base.as_avx512().is_some(),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => base.as_neon().is_some(),
        #[cfg(target_arch = "aarch64")]
        Isa::NeonDot => base.as_neon().is_some() && dotprod(),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// The kernels in use: forced by `force`, else the detected best.
#[inline]
pub fn current() -> Isa {
    from_code(FORCED.load(Relaxed)).unwrap_or_else(detected)
}

/// Force `isa` (None = back to auto). Returns false (and changes nothing) if this CPU can't run it.
pub fn force(isa: Option<Isa>) -> bool {
    match isa {
        Some(i) if !supported(i) => false,
        _ => {
            FORCED.store(isa.map_or(0, |i| i as i32), Relaxed);
            true
        }
    }
}

/// Force by code (see the table; 0 = auto). Returns the code now in use, or -1 if `code` can't run here.
pub fn force_code(code: i32) -> i32 {
    let isa = if code == 0 { None } else { from_code(code) };
    if (code != 0 && isa.is_none()) || !force(isa) {
        return -1;
    }
    current() as i32
}

/// The fearless_simd level for the elementwise code that goes with `isa`.
pub fn level(isa: Isa) -> Level {
    let base = Level::new();
    match isa {
        Isa::Scalar => Level::fallback(),
        #[cfg(target_arch = "x86_64")]
        Isa::Sse => base.as_sse4_2().map_or(base, Level::Sse4_2),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::AvxVnni => base.as_avx2().map_or(base, Level::Avx2),
        _ => base,
    }
}
