// Fused CPU inference for the Example 14 GDN byte LM (LeanLM with FP block linears, conv_size=1).
// All RMSNorm gains, the kernel-1 "conv" scales and the GDN o_norm gain are folded into the weights at export,
// q/k/v/gate/a/b share one matrix and SwiGLU gate/up share one, so a layer is 4 matmuls + the recurrence.
// Weight formats: 0 f32, 1 f16, 2 int8 (per-row scale), 3 int4 (group 32, symmetric, per-group f32 scale),
// 4 int8 weights x int8 activations on VNNI (W8A8). Columns are padded to a multiple of 32; activation buffers are
// zero-padded to match. The embedding table is row-major (lookups); every other matrix is tile-major (matmuls).
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct { int rows, cols, cpad, fmt; void *w; float *s; int rpad; } Mat;  // rpad > 0: tile-major
typedef struct { Mat in, o, up, down; float *negA, *dt_bias; } Layer;
typedef struct { int d, dpad, H, D, hid, hpad, L, vocab; float eps; Mat emb, head; Layer *layers; } Model;

// Tile-major weights (all block matrices and the head): rows in tiles of 32, layout [tile][col][32 rows], so a matmul
// streams the weights once, broadcasts one activation per column and accumulates 32 outputs per FMA pair with no
// horizontal sums. f32 / f16 / int8 (per-row scale) / int4 ([tile][col][16 bytes]: low nibble rows 0..15, high
// 16..31, value - 8, scales [tile][col/32][32]).
static inline void load_w(const Mat *m, size_t tile, int i, __m512 *w0, __m512 *w1) {
    size_t c = m->cpad;
    switch (m->fmt) {
    case 0: { const float *p = (const float *)m->w + (tile * c + i) * 32; *w0 = _mm512_loadu_ps(p); *w1 = _mm512_loadu_ps(p + 16); break; }
    case 1: { const uint16_t *p = (const uint16_t *)m->w + (tile * c + i) * 32;
              *w0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p)); *w1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(p + 16))); break; }
    case 2: { __m256i v = _mm256_loadu_si256((const __m256i *)((const int8_t *)m->w + (tile * c + i) * 32));
              *w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(v)));
              *w1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(v, 1))); break; }
    default: { __m128i b = _mm_loadu_si128((const __m128i *)((const uint8_t *)m->w + (tile * c + i) * 16));
               const __m128i k = _mm_set1_epi8(0x0F); const __m512i e = _mm512_set1_epi32(8);
               *w0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_and_si128(b, k)), e));
               *w1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(b, 4), k)), e)); }
    }
}

#define NTB 4
// One tile of 32 rows x nt (<= NTB) tokens.
static void tile_kernel(const Mat *m, size_t tile, const float *X, int ldx, int nt, __m512 *o0, __m512 *o1) {
    __m512 a0[NTB], a1[NTB], g0[NTB], g1[NTB];
    for (int t = 0; t < NTB; t++) a0[t] = a1[t] = g0[t] = g1[t] = _mm512_setzero_ps();
    const int C = m->cpad;
    if (m->fmt == 3) {
        for (int i0 = 0; i0 < C; i0 += 32) {
            for (int t = 0; t < nt; t++) g0[t] = g1[t] = _mm512_setzero_ps();
            for (int i = i0; i < i0 + 32; i++) {
                __m512 w0, w1; load_w(m, tile, i, &w0, &w1);
                for (int t = 0; t < nt; t++) { __m512 x = _mm512_set1_ps(X[(size_t)t * ldx + i]);
                    g0[t] = _mm512_fmadd_ps(w0, x, g0[t]); g1[t] = _mm512_fmadd_ps(w1, x, g1[t]); }
            }
            const float *sc = m->s + (tile * (C / 32) + i0 / 32) * 32;
            __m512 s0 = _mm512_loadu_ps(sc), s1 = _mm512_loadu_ps(sc + 16);
            for (int t = 0; t < nt; t++) { a0[t] = _mm512_fmadd_ps(g0[t], s0, a0[t]); a1[t] = _mm512_fmadd_ps(g1[t], s1, a1[t]); }
        }
    } else if (nt == 1) {  // decode: two independent chains over even / odd columns
        int i = 0;
        for (; i + 1 < C; i += 2) {
            __m512 w0, w1, v0, v1; load_w(m, tile, i, &w0, &w1); load_w(m, tile, i + 1, &v0, &v1);
            __m512 x = _mm512_set1_ps(X[i]), y = _mm512_set1_ps(X[i + 1]);
            a0[0] = _mm512_fmadd_ps(w0, x, a0[0]); a1[0] = _mm512_fmadd_ps(w1, x, a1[0]);
            g0[0] = _mm512_fmadd_ps(v0, y, g0[0]); g1[0] = _mm512_fmadd_ps(v1, y, g1[0]);
        }
        for (; i < C; i++) { __m512 w0, w1; load_w(m, tile, i, &w0, &w1); __m512 x = _mm512_set1_ps(X[i]);
            a0[0] = _mm512_fmadd_ps(w0, x, a0[0]); a1[0] = _mm512_fmadd_ps(w1, x, a1[0]); }
        a0[0] = _mm512_add_ps(a0[0], g0[0]); a1[0] = _mm512_add_ps(a1[0], g1[0]);
    } else {
        for (int i = 0; i < C; i++) {
            __m512 w0, w1; load_w(m, tile, i, &w0, &w1);
            for (int t = 0; t < nt; t++) { __m512 x = _mm512_set1_ps(X[(size_t)t * ldx + i]);
                a0[t] = _mm512_fmadd_ps(w0, x, a0[t]); a1[t] = _mm512_fmadd_ps(w1, x, a1[t]); }
        }
    }
    if (m->fmt == 2) { __m512 s0 = _mm512_loadu_ps(m->s + tile * 32), s1 = _mm512_loadu_ps(m->s + tile * 32 + 16);
        for (int t = 0; t < nt; t++) { a0[t] = _mm512_mul_ps(a0[t], s0); a1[t] = _mm512_mul_ps(a1[t], s1); } }
    for (int t = 0; t < nt; t++) { o0[t] = a0[t]; o1[t] = a1[t]; }
}

// fmt 4: int8 weights x int8 activations on AVX-512 VNNI (W8A8). Weights [tile][col/4][32 rows][4] int8, per-row
// scale s[rpad], then corr[rpad] = 128 * row sum (int32, stored after the scales). Activations are quantized per
// token (absmax / 127) and offset by 128 to u8 for vpdpbusd; the offset is removed with corr.
static void tile_vnni(const Mat *m, size_t tile, const uint8_t *XU, int ldu, const float *xs, int nt, __m512 *o0, __m512 *o1) {
    __m512i a0[NTB], a1[NTB], b0 = _mm512_setzero_si512(), b1 = b0;
    for (int t = 0; t < NTB; t++) a0[t] = a1[t] = _mm512_setzero_si512();
    const int C4 = m->cpad / 4;
    const int8_t *W = (const int8_t *)m->w + tile * (size_t)C4 * 128;
    if (nt == 1) {
        int c = 0;
        for (; c + 1 < C4; c += 2) {
            __m512i x = _mm512_set1_epi32(*(const int32_t *)(XU + 4 * c)), y = _mm512_set1_epi32(*(const int32_t *)(XU + 4 * c + 4));
            a0[0] = _mm512_dpbusd_epi32(a0[0], x, _mm512_loadu_si512(W + c * 128));
            a1[0] = _mm512_dpbusd_epi32(a1[0], x, _mm512_loadu_si512(W + c * 128 + 64));
            b0 = _mm512_dpbusd_epi32(b0, y, _mm512_loadu_si512(W + c * 128 + 128));
            b1 = _mm512_dpbusd_epi32(b1, y, _mm512_loadu_si512(W + c * 128 + 192));
        }
        for (; c < C4; c++) {
            __m512i x = _mm512_set1_epi32(*(const int32_t *)(XU + 4 * c));
            a0[0] = _mm512_dpbusd_epi32(a0[0], x, _mm512_loadu_si512(W + c * 128));
            a1[0] = _mm512_dpbusd_epi32(a1[0], x, _mm512_loadu_si512(W + c * 128 + 64));
        }
        a0[0] = _mm512_add_epi32(a0[0], b0); a1[0] = _mm512_add_epi32(a1[0], b1);
    } else {
        for (int c = 0; c < C4; c++) {
            __m512i w0 = _mm512_loadu_si512(W + c * 128), w1 = _mm512_loadu_si512(W + c * 128 + 64);
            for (int t = 0; t < nt; t++) {
                __m512i x = _mm512_set1_epi32(*(const int32_t *)(XU + (size_t)t * ldu + 4 * c));
                a0[t] = _mm512_dpbusd_epi32(a0[t], x, w0); a1[t] = _mm512_dpbusd_epi32(a1[t], x, w1);
            }
        }
    }
    const float *sc = m->s + tile * 32;
    const int32_t *corr = (const int32_t *)(m->s + m->rpad) + tile * 32;
    __m512 s0 = _mm512_loadu_ps(sc), s1 = _mm512_loadu_ps(sc + 16);
    __m512i c0 = _mm512_loadu_si512(corr), c1 = _mm512_loadu_si512(corr + 16);
    for (int t = 0; t < nt; t++) {
        __m512 x = _mm512_set1_ps(xs[t]);
        o0[t] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a0[t], c0)), _mm512_mul_ps(s0, x));
        o1[t] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(a1[t], c1)), _mm512_mul_ps(s1, x));
    }
}

static void quant_act(const float *x, int n, int npad, uint8_t *u, float *scale) {
    float amax = 0.f;
    for (int i = 0; i < n; i++) amax = fmaxf(amax, fabsf(x[i]));
    float s = amax > 0.f ? amax / 127.f : 1.f, r = 1.f / s;
    for (int i = 0; i < n; i++) u[i] = (uint8_t)((int)lrintf(x[i] * r) + 128);
    for (int i = n; i < npad; i++) u[i] = 128;
    *scale = s;
}

// Y[t, r] (+)= W[r] . X[t] for t < T, r < rows.
static void matmul(const Mat *m, const float *X, int ldx, int T, float *Y, int ldy, int accumulate, int nth) {
    const int ntile = m->rpad / 32;
    uint8_t XU[m->fmt == 4 ? (size_t)T * m->cpad : 1];
    float xs[m->fmt == 4 ? T : 1];
    if (m->fmt == 4)
        for (int t = 0; t < T; t++) quant_act(X + (size_t)t * ldx, m->cols, m->cpad, XU + (size_t)t * m->cpad, xs + t);
#pragma omp parallel for num_threads(nth) if (nth > 1 && T > 1) schedule(static)
    for (int tile = 0; tile < ntile; tile++) {
        float buf[32];
        for (int t0 = 0; t0 < T; t0 += NTB) {
            int nt = T - t0 < NTB ? T - t0 : NTB;
            __m512 o0[NTB], o1[NTB];
            if (m->fmt == 4) tile_vnni(m, tile, XU + (size_t)t0 * m->cpad, m->cpad, xs + t0, nt, o0, o1);
            else tile_kernel(m, tile, X + (size_t)t0 * ldx, ldx, nt, o0, o1);
            int r0 = tile * 32, nr = m->rows - r0 < 32 ? m->rows - r0 : 32;
            for (int t = 0; t < nt; t++) {
                float *y = Y + (size_t)(t0 + t) * ldy + r0;
                _mm512_storeu_ps(buf, o0[t]); _mm512_storeu_ps(buf + 16, o1[t]);
                if (accumulate) for (int j = 0; j < nr; j++) y[j] += buf[j]; else memcpy(y, buf, nr * 4);
            }
        }
    }
}

static inline float silu(float x) { return x / (1.f + expf(-x)); }
static inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }
static inline float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

static void rms(const float *x, float *y, int n, int npad, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r;
    for (int i = n; i < npad; i++) y[i] = 0.f;
}
static void l2n(float *x, int n, float mul) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float nr = sqrtf(ss);
    float r = mul / (nr > 1e-12f ? nr : 1e-12f);
    for (int i = 0; i < n; i++) x[i] *= r;
}

static void get_row(const Mat *m, int r, float *out) {
    int c = m->cpad;
    switch (m->fmt) {
    case 0: memcpy(out, (const float *)m->w + (size_t)r * c, c * 4); break;
    case 1: for (int i = 0; i < c; i += 16) _mm512_storeu_ps(out + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const uint16_t *)m->w + (size_t)r * c + i)))); break;
    case 2: for (int i = 0; i < c; i++) out[i] = ((const int8_t *)m->w)[(size_t)r * c + i] * m->s[r]; break;
    default: {
        const uint8_t *w = (const uint8_t *)m->w + (size_t)r * (c / 2);
        const float *s = m->s + (size_t)r * (c / 32);
        for (int g = 0; g < c / 32; g++)
            for (int j = 0; j < 16; j++) {
                out[g * 32 + j] = ((w[g * 16 + j] & 15) - 8) * s[g];
                out[g * 32 + 16 + j] = ((w[g * 16 + j] >> 4) - 8) * s[g];
            }
    }
    }
}

// Work buffer floats needed for a block of BT tokens.
long work_floats(const Model *M, int BT) {
    long d = M->dpad, rin = 4L * M->d + 2L * M->H, hp = M->hpad;
    return BT * (d /*X*/ + d /*N*/ + rin /*P*/ + d /*O*/ + 2 * hp /*U*/ + hp /*Mh*/) + 64;
}

// Run T tokens from state S (L x H x D x D, updated in place). logits: (T x vocab) if all_logits else (vocab) for the
// last token. BT: tokens per block (matmuls share weight reads across a block). nth: OpenMP threads for matmuls.
void forward(const Model *M, const int *toks, int T, float *S, float *logits, int all_logits, float *work, int BT, int nth) {
    const int d = M->d, dp = M->dpad, H = M->H, D = M->D, hid = M->hid, hp = M->hpad, V = M->vocab;
    const int rin = 4 * d + 2 * H;
    float *X = work, *N = X + BT * dp, *P = N + BT * dp, *O = P + BT * rin, *U = O + BT * dp, *Mh = U + BT * 2 * hp;
    const float qs = 1.f / sqrtf((float)D);
    for (int t0 = 0; t0 < T; t0 += BT) {
        int bt = T - t0 < BT ? T - t0 : BT;
        for (int t = 0; t < bt; t++) get_row(&M->emb, toks[t0 + t], X + t * dp);
        for (int l = 0; l < M->L; l++) {
            const Layer *ly = &M->layers[l];
            float *Sl = S + (size_t)l * H * D * D;
            for (int t = 0; t < bt; t++) rms(X + t * dp, N + t * dp, d, dp, M->eps);
            matmul(&ly->in, N, dp, bt, P, rin, 0, nth);
            for (int t = 0; t < bt; t++) {
                float *p = P + t * rin, *q = p, *k = p + d, *v = p + 2 * d, *gt = p + 3 * d, *a = p + 4 * d, *b = a + H;
                float *o = O + t * dp;
                for (int i = 0; i < 3 * d; i++) p[i] = silu(p[i]);
                for (int h = 0; h < H; h++) {
                    float *qh = q + h * D, *kh = k + h * D, *vh = v + h * D, *oh = o + h * D, *Sh = Sl + (size_t)h * D * D;
                    l2n(qh, D, qs);
                    l2n(kh, D, 1.f);
                    float beta = sigm(b[h]);
                    float decay = expf(ly->negA[h] * softplus(a[h] + ly->dt_bias[h]));
                    // u = S^T k, w = S^T q (read-only pass); delta = (v - decay u) beta;
                    // S <- decay S + k delta^T; o = S^T q = decay w + (k.q) delta
                    float u[1024], w[1024];
                    for (int j = 0; j < D; j++) u[j] = w[j] = 0.f;
                    float kq = 0.f;
                    for (int i = 0; i < D; i++) {
                        const float *row = Sh + (size_t)i * D;
                        float ki = kh[i], qi = qh[i];
                        kq += ki * qi;
                        for (int j = 0; j < D; j++) { u[j] += row[j] * ki; w[j] += row[j] * qi; }
                    }
                    for (int j = 0; j < D; j++) { u[j] = (vh[j] - decay * u[j]) * beta; oh[j] = decay * w[j] + kq * u[j]; }
                    for (int i = 0; i < D; i++) {
                        float *row = Sh + (size_t)i * D, ki = kh[i];
                        for (int j = 0; j < D; j++) row[j] = decay * row[j] + ki * u[j];
                    }
                    rms(oh, oh, D, D, M->eps);  // o_norm (gain folded into W_o)
                    for (int j = 0; j < D; j++) oh[j] *= silu(gt[h * D + j]);
                }
                for (int i = d; i < dp; i++) o[i] = 0.f;
            }
            matmul(&ly->o, O, dp, bt, X, dp, 1, nth);
            for (int t = 0; t < bt; t++) rms(X + t * dp, N + t * dp, d, dp, M->eps);
            matmul(&ly->up, N, dp, bt, U, 2 * hp, 0, nth);
            for (int t = 0; t < bt; t++) {
                float *u = U + t * 2 * hp, *mh = Mh + t * hp;
                for (int i = 0; i < hid; i++) mh[i] = silu(u[i]) * u[hid + i];
                for (int i = hid; i < hp; i++) mh[i] = 0.f;
            }
            matmul(&ly->down, Mh, hp, bt, X, dp, 1, nth);
        }
        if (all_logits) {
            for (int t = 0; t < bt; t++) rms(X + t * dp, N + t * dp, d, dp, M->eps);
            matmul(&M->head, N, dp, bt, logits + (size_t)t0 * V, V, 0, nth);
        } else if (t0 + bt == T) {
            rms(X + (bt - 1) * dp, N, d, dp, M->eps);
            matmul(&M->head, N, dp, 1, logits, V, 0, nth);
        }
    }
}

// Greedy decode n bytes after the state; first = byte to feed first. Writes out[n]. Returns nothing.
void generate(const Model *M, int first, int n, float *S, float *logits, float *work, int *out) {
    int tok = first;
    for (int i = 0; i < n; i++) {
        forward(M, &tok, 1, S, logits, 0, work, 1, 1);
        int best = 0;
        for (int j = 1; j < M->vocab; j++) if (logits[j] > logits[best]) best = j;
        out[i] = tok = best;
    }
}
