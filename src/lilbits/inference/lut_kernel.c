// Lookup-table (T-MAC style) matmul for +-1 weight matrices, CPU.
//
//   y[n, o] = sum_i x[n, i] * S[o, i],   S in {-1, +1}
//
// Weights are grouped 4 inputs at a time: each group's 4 signs form a 4-bit code. For each token the kernel
// first builds, per group g, a 16-entry table T_g[c] = sum_j (bit j of c ? +x : -x)[4g + j] (4 FMAs per group).
// Then every output is a sum of table lookups, one per group, instead of 4 multiply-adds.
//
// With AVX-512 a group's whole table is one register, and one vpermps does 16 lookups for 16 outputs at
// once. AVX2 uses two 8-entry permutes and a blend. Other CPUs use a scalar loop.
//
// Code layout (built by lilbits.inference.lut.pack_sign_codes): codes[ob][g][k], uint8, for output blocks of
// 32: the low nibble is the code of output 32*ob + k and the high nibble of output 32*ob + 16 + k. Storage is
// 1 bit per weight, the same as plain sign packing.

#include <stdint.h>
#include <string.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

static void build_tables(const float *x, int64_t G, float *T) {
    for (int64_t g = 0; g < G; g++) {
        const float a = x[4 * g], b = x[4 * g + 1], c = x[4 * g + 2], d = x[4 * g + 3];
        float *t = T + 16 * g;
        for (int code = 0; code < 16; code++)
            t[code] = ((code & 1) ? a : -a) + ((code & 2) ? b : -b) + ((code & 4) ? c : -c) + ((code & 8) ? d : -d);
    }
}

static void block_scalar(const float *T, const uint8_t *codes, int64_t G, float *acc /* 32 */) {
    for (int k = 0; k < 32; k++) acc[k] = 0.f;
    for (int64_t g = 0; g < G; g++) {
        const float *t = T + 16 * g;
        const uint8_t *c = codes + 16 * g;
        for (int k = 0; k < 16; k++) {
            acc[k] += t[c[k] & 15];
            acc[16 + k] += t[c[k] >> 4];
        }
    }
}

#if defined(__AVX512F__)
static void block_simd(const float *T, const uint8_t *codes, int64_t G, float *acc) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();  // second pair breaks the add dependency chain
    int64_t g = 0;
    for (; g + 1 < G; g += 2) {
        __m512 t = _mm512_loadu_ps(T + 16 * g), u = _mm512_loadu_ps(T + 16 * g + 16);
        __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(codes + 16 * g)));
        __m512i w = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(codes + 16 * g + 16)));
        a0 = _mm512_add_ps(a0, _mm512_permutexvar_ps(v, t));  // uses the low 4 bits of each index
        a1 = _mm512_add_ps(a1, _mm512_permutexvar_ps(_mm512_srli_epi32(v, 4), t));
        b0 = _mm512_add_ps(b0, _mm512_permutexvar_ps(w, u));
        b1 = _mm512_add_ps(b1, _mm512_permutexvar_ps(_mm512_srli_epi32(w, 4), u));
    }
    for (; g < G; g++) {
        __m512 t = _mm512_loadu_ps(T + 16 * g);
        __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(codes + 16 * g)));
        a0 = _mm512_add_ps(a0, _mm512_permutexvar_ps(v, t));
        a1 = _mm512_add_ps(a1, _mm512_permutexvar_ps(_mm512_srli_epi32(v, 4), t));
    }
    _mm512_storeu_ps(acc, _mm512_add_ps(a0, b0));
    _mm512_storeu_ps(acc + 16, _mm512_add_ps(a1, b1));
}
#define HAVE_SIMD 512
#elif defined(__AVX2__)
static inline __m256 lookup8x2(__m256 lo, __m256 hi, __m256i idx) {
    // idx in [0, 16): permute both halves by idx & 7, pick hi where bit 3 is set
    __m256 m = _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28));
    return _mm256_blendv_ps(_mm256_permutevar8x32_ps(lo, idx), _mm256_permutevar8x32_ps(hi, idx), m);
}
static void block_simd(const float *T, const uint8_t *codes, int64_t G, float *acc) {
    __m256 a[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    const __m256i nib = _mm256_set1_epi32(15);
    for (int64_t g = 0; g < G; g++) {
        __m256 lo = _mm256_loadu_ps(T + 16 * g), hi = _mm256_loadu_ps(T + 16 * g + 8);
        __m128i raw = _mm_loadu_si128((const __m128i *)(codes + 16 * g));
        __m256i v0 = _mm256_cvtepu8_epi32(raw), v1 = _mm256_cvtepu8_epi32(_mm_srli_si128(raw, 8));
        a[0] = _mm256_add_ps(a[0], lookup8x2(lo, hi, _mm256_and_si256(v0, nib)));
        a[1] = _mm256_add_ps(a[1], lookup8x2(lo, hi, _mm256_and_si256(v1, nib)));
        a[2] = _mm256_add_ps(a[2], lookup8x2(lo, hi, _mm256_srli_epi32(v0, 4)));
        a[3] = _mm256_add_ps(a[3], lookup8x2(lo, hi, _mm256_srli_epi32(v1, 4)));
    }
    for (int i = 0; i < 4; i++) _mm256_storeu_ps(acc + 8 * i, a[i]);
}
#define HAVE_SIMD 256
#else
#define HAVE_SIMD 0
#endif

int lut_simd_width(void) { return HAVE_SIMD; }

// x: (N, 4G) float32, codes: (OB, G, 16) uint8, tables: (N, G, 16) float32 scratch,
// y: (N, O) float32 with O <= 32 * OB. Each token's tables are built once, then output blocks are split over
// OpenMP threads.
void lut_matmul(const float *x, const uint8_t *codes, float *tables, float *y, int64_t N, int64_t G, int64_t OB,
                int64_t O, int nthreads) {
    for (int64_t n = 0; n < N; n++) build_tables(x + n * 4 * G, G, tables + n * 16 * G);
#pragma omp parallel for collapse(2) num_threads(nthreads) schedule(static)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t ob = 0; ob < OB; ob++) {
            float acc[32];
#if HAVE_SIMD
            block_simd(tables + n * 16 * G, codes + ob * G * 16, G, acc);
#else
            block_scalar(tables + n * 16 * G, codes + ob * G * 16, G, acc);
#endif
            int64_t o0 = ob * 32, m = O - o0 < 32 ? O - o0 : 32;
            memcpy(y + n * O + o0, acc, m * sizeof(float));
        }
    }
}

static inline void block(const float *T, const uint8_t *codes, int64_t G, float *acc) {
#if HAVE_SIMD
    block_simd(T, codes, G, acc);
#else
    block_scalar(T, codes, G, acc);
#endif
}

// BitNet ternary W = (P + Q) / 2 * scale: both sign planes read the same tables, built once per token.
void lut_ternary(const float *x, const uint8_t *pc, const uint8_t *qc, float *tables, float *y, int64_t N, int64_t G,
                 int64_t OB, int64_t O, float scale, int nthreads) {
    for (int64_t n = 0; n < N; n++) build_tables(x + n * 4 * G, G, tables + n * 16 * G);
    const float s = 0.5f * scale;
#pragma omp parallel for collapse(2) num_threads(nthreads) schedule(static)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t ob = 0; ob < OB; ob++) {
            float a[32], b[32];
            block(tables + n * 16 * G, pc + ob * G * 16, G, a);
            block(tables + n * 16 * G, qc + ob * G * 16, G, b);
            int64_t o0 = ob * 32, m = O - o0 < 32 ? O - o0 : 32;
            for (int64_t k = 0; k < m; k++) y[n * O + o0 + k] = (a[k] + b[k]) * s;
        }
    }
}

// Whole LittleBit layer in one call: y = sum_p h_p * (S_U,p (l_p * (S_V,p^T (g_p * x)))) (+ bias).
// vc: (P, OBr, Gi, 16) codes of S_V^T (R x I); uc: (P, OBo, Gr, 16) codes of S_U (O x R); g (P, I), l (P, R),
// h (P, O) float32. Scratch: xs (4 Gi), ti (Gi * 16), t (4 Gr, zero padded), tr (Gr * 16). One token at a time.
void lut_littlebit(const float *x, const uint8_t *vc, const uint8_t *uc, const float *g, const float *l,
                   const float *h, const float *bias, float *xs, float *ti, float *t, float *tr, float *y, int64_t N,
                   int64_t P, int64_t I, int64_t Gi, int64_t OBr, int64_t R, int64_t Gr, int64_t OBo, int64_t O,
                   int nthreads) {
    for (int64_t n = 0; n < N; n++) {
        float *yn = y + n * O;
        for (int64_t o = 0; o < O; o++) yn[o] = bias ? bias[o] : 0.f;
        for (int64_t p = 0; p < P; p++) {
            const float *gp = g + p * I, *lp = l + p * R, *hp = h + p * O;
            for (int64_t i = 0; i < 4 * Gi; i++) xs[i] = i < I ? x[n * I + i] * gp[i] : 0.f;
            build_tables(xs, Gi, ti);
            const uint8_t *vp = vc + p * OBr * Gi * 16, *up = uc + p * OBo * Gr * 16;
#pragma omp parallel for num_threads(nthreads) schedule(static)
            for (int64_t ob = 0; ob < OBr; ob++) {
                float acc[32];
                block(ti, vp + ob * Gi * 16, Gi, acc);
                int64_t r0 = ob * 32, m = R - r0 < 32 ? R - r0 : 32;
                for (int64_t k = 0; k < m; k++) t[r0 + k] = acc[k] * lp[r0 + k];
            }
            for (int64_t r = R; r < 4 * Gr; r++) t[r] = 0.f;
            build_tables(t, Gr, tr);
#pragma omp parallel for num_threads(nthreads) schedule(static)
            for (int64_t ob = 0; ob < OBo; ob++) {
                float acc[32];
                block(tr, up + ob * Gr * 16, Gr, acc);
                int64_t o0 = ob * 32, m = O - o0 < 32 ? O - o0 : 32;
                for (int64_t k = 0; k < m; k++) yn[o0 + k] += acc[k] * hp[o0 + k];
            }
        }
    }
}

// Scalar path, always compiled, for testing the SIMD path against.
void lut_matmul_scalar(const float *x, const uint8_t *codes, float *tables, float *y, int64_t N, int64_t G,
                       int64_t OB, int64_t O) {
    for (int64_t n = 0; n < N; n++) {
        build_tables(x + n * 4 * G, G, tables);
        for (int64_t ob = 0; ob < OB; ob++) {
            float acc[32];
            block_scalar(tables, codes + ob * G * 16, G, acc);
            int64_t o0 = ob * 32, m = O - o0 < 32 ? O - o0 : 32;
            memcpy(y + n * O + o0, acc, m * sizeof(float));
        }
    }
}
