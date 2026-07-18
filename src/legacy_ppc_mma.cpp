// legacy_ppc_mma.cpp
//
// POWER10/POWER11 MMA GEMM for the legacy 32-block quants:
//   Q4_1 x Q8_1     value = d*q + m          (q in 0..15)
//   Q5_0 x Q8_0     value = d*(q - 16)       (q in 0..31, qh 5th bits)
//   Q5_1 x Q8_1     value = d*q + m          (q in 0..31)
//
// All codes are unsigned -> v3 orientation (codes on the unsigned GER
// operand, untouched int8 activations on the signed operand), 16x8 tile
// on all 8 accumulators, one 32-element block per chunk.
//
// The affine/offset terms are nearly free:
//   * Q4_1/Q5_1: per-block result = dA*dB*P + m*s, where s = dB*sum(y)
//     is ALREADY STORED in block_q8_1 -- the min term is one vec_madd
//     with no sum computation anywhere.
//   * Q5_0: t = q - 16 shares the dA factor, so the correction folds
//     into the main FMA (vec_msub) with TS = 16*dB*S, S summed once per
//     (col, chunk) in the activation pack.
//
// Element layout (all three): elems j / j+16 from qs[j] low/high nibble;
// Q5_x add bit j (elems 0..15) / bit j+16 (elems 16..31) of the 32-bit
// qh, shifted up to bit 4.
//
// Conventions as the qbit kernels; k % 32 == 0.  Tests: -DLEGACY_TEST.

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32

typedef uint16_t ggml_half;

typedef struct { ggml_half d, m; uint8_t qs[QK4_1/2]; } block_q4_1;
typedef struct { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; } block_q5_0;
typedef struct { ggml_half d, m; uint8_t qh[4]; uint8_t qs[QK5_1/2]; } block_q5_1;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
typedef struct { ggml_half d, s; int8_t qs[QK8_1]; } block_q8_1;

static_assert(sizeof(block_q4_1) == 4 + 16, "bad q4_1");
static_assert(sizeof(block_q5_0) == 2 + 4 + 16, "bad q5_0");
static_assert(sizeof(block_q5_1) == 4 + 4 + 16, "bad q5_1");
static_assert(sizeof(block_q8_1) == 4 + 32, "bad q8_1");

static inline float fp16_to_fp32(ggml_half h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4);
    return f;
}

#if defined(__MMA__)

typedef vector unsigned char  vuc;
typedef vector signed char    vsc;
typedef vector unsigned int   vui;
typedef vector signed int     vsi;
typedef vector float          vfl;

#define KC_CH 64                          // 32-elem chunks per slab (2048)
#define MR    16
#define NR    8

typedef struct {
    vuc v[KC_CH][32];                     // 8 depth-steps x 4 rowgroups
    vfl dA[KC_CH][4];                     // per-block scale per rowgroup
    vfl mA[KC_CH][4];                     // per-block min (Q4_1/Q5_1; 0 for Q5_0)
} aleg_t;

typedef struct {
    vuc v[KC_CH][16];                     // 8 depth-steps x 2 colgroups
    vfl dB[KC_CH][2];                     // per-block activation scale
    vfl SB[KC_CH][2];                     // Q8_1: s = dB*sum(y); Q8_0: 16*dB*sum(y)
} bleg_t;

static inline void mma_transpose4(const vui rows[4], vuc * out, int stride) {
    vui t0 = vec_mergeh(rows[0], rows[1]);
    vui t1 = vec_mergel(rows[0], rows[1]);
    vui t2 = vec_mergeh(rows[2], rows[3]);
    vui t3 = vec_mergel(rows[2], rows[3]);
    out[0*stride] = (vuc)vec_xxpermdi(t0, t2, 0);
    out[1*stride] = (vuc)vec_xxpermdi(t0, t2, 3);
    out[2*stride] = (vuc)vec_xxpermdi(t1, t3, 0);
    out[3*stride] = (vuc)vec_xxpermdi(t1, t3, 3);
}

static inline void nibbles32(const uint8_t * qs, vuc v[2]) {
    vuc raw = vec_xl(0, (const unsigned char *)qs);
    v[0] = vec_and(raw, vec_splats((unsigned char)0xF));
    v[1] = vec_sr(raw, vec_splats((unsigned char)4));
}

// merge qh bits: elems 0..15 use bits 0..15 (bytes 0-1), elems 16..31
// use bits 16..31 (bytes 2-3); bit -> position 4.
static inline void qh_merge(const uint8_t * qh, vuc v[2]) {
    vuc raw = vec_xl(0, (const unsigned char *)qh);   // first 4 bytes used
    const vuc repL = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc repH = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc sh   = { 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7 };
    const vuc one  = vec_splats((unsigned char)1);
    const vuc four = vec_splats((unsigned char)4);
    vuc b0 = vec_sl(vec_and(vec_sr(vec_perm(raw, raw, repL), sh), one), four);
    vuc b1 = vec_sl(vec_and(vec_sr(vec_perm(raw, raw, repH), sh), one), four);
    v[0] = vec_or(v[0], b0);
    v[1] = vec_or(v[1], b1);
}

static inline int64_t rt(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl(int64_t k) { return (k/32 + KC_CH - 1) / KC_CH; }

extern "C" size_t leg_apack_size(int64_t m, int64_t k) {
    return (size_t)(rt(m) * sl(k)) * sizeof(aleg_t);
}
extern "C" size_t leg_bpack_size(int64_t n, int64_t k) {
    return (size_t)(ct(n) * sl(k)) * sizeof(bleg_t);
}

// generic weight repack over a per-block "codes + d + m" extractor
template <typename BLK, void (*CODES)(const BLK *, vuc[2]), bool HAS_M>
static void repack_generic(const BLK * A, int64_t lda, int64_t m, int64_t k, aleg_t * P) {
    const int64_t kb = k/32, ns = sl(k);
    for (int64_t it = 0; it < rt(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aleg_t * T = &P[it*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            vuc t[MR][2]; float d[MR], mm[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                const BLK * bp = &A[rr*lda + b0 + b];
                d[r]  = fp16_to_fp32(bp->d);
                mm[r] = HAS_M ? fp16_to_fp32(((const ggml_half *)bp)[1]) : 0.0f;
                CODES(bp, t[r]);
            }
            for (int g = 0; g < 4; g++) {
                T->dA[b][g] = (vfl){ d[4*g], d[4*g+1], d[4*g+2], d[4*g+3] };
                T->mA[b][g] = (vfl){ mm[4*g], mm[4*g+1], mm[4*g+2], mm[4*g+3] };
            }
            vui rows4[4];
            for (int g = 0; g < 4; g++)
                for (int h = 0; h < 2; h++) {
                    for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                    mma_transpose4(rows4, &T->v[b][16*h + g], 4);
                }
        }
    }
}

static void codes_q4_1(const block_q4_1 * bp, vuc v[2]) { nibbles32(bp->qs, v); }
static void codes_q5_0(const block_q5_0 * bp, vuc v[2]) { nibbles32(bp->qs, v); qh_merge(bp->qh, v); }
static void codes_q5_1(const block_q5_1 * bp, vuc v[2]) { nibbles32(bp->qs, v); qh_merge(bp->qh, v); }

extern "C" void leg_repack_q4_1(const block_q4_1 * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_generic<block_q4_1, codes_q4_1, true>(A, lda, m, k, (aleg_t *)p);
}
extern "C" void leg_repack_q5_0(const block_q5_0 * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_generic<block_q5_0, codes_q5_0, false>(A, lda, m, k, (aleg_t *)p);
}
extern "C" void leg_repack_q5_1(const block_q5_1 * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_generic<block_q5_1, codes_q5_1, true>(A, lda, m, k, (aleg_t *)p);
}

// activation packs: SB = dB*sum(y) computed (Q8_0, x16 for the Q5_0
// offset) or read from the precomputed s field (Q8_1).
template <typename YBLK, bool READ_S, int SFACT>
static void pack_b_generic(const YBLK * B, int64_t ldb, int64_t n, int64_t k, bleg_t * P) {
    const int64_t kb = k/32, ns = sl(k);
    for (int64_t jt = 0; jt < ct(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        bleg_t * T = &P[jt*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            const YBLK * yb[NR]; float dB[NR], SB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + b0 + b];
                dB[j] = fp16_to_fp32(yb[j]->d);
                if (READ_S) {
                    SB[j] = fp16_to_fp32(((const ggml_half *)yb[j])[1]);
                } else {
                    vsi z = vec_splats(0);
                    vsi sm = vec_sum4s((vsc)vec_xl(0,  (const unsigned char *)yb[j]->qs), z);
                    sm = vec_sum4s((vsc)vec_xl(16, (const unsigned char *)yb[j]->qs), sm);
                    SB[j] = (float)SFACT * dB[j] * (float)(sm[0]+sm[1]+sm[2]+sm[3]);
                }
            }
            T->dB[b][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            T->dB[b][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            T->SB[b][0] = (vfl){ SB[0], SB[1], SB[2], SB[3] };
            T->SB[b][1] = (vfl){ SB[4], SB[5], SB[6], SB[7] };
            vui rows4[4];
            for (int a = 0; a < 2; a++)
                for (int h = 0; h < 2; h++) {
                    for (int j = 0; j < 4; j++)
                        rows4[j] = (vui)vec_xl(16*h, (const unsigned char *)yb[4*a + j]->qs);
                    mma_transpose4(rows4, &T->v[b][8*h + a], 2);
                }
        }
    }
}

extern "C" void leg_pack_b_q8_0(const block_q8_0 * B, int64_t ldb, int64_t n, int64_t k, void * p) {
    pack_b_generic<block_q8_0, false, 16>(B, ldb, n, k, (bleg_t *)p);
}
extern "C" void leg_pack_b_q8_1(const block_q8_1 * B, int64_t ldb, int64_t n, int64_t k, void * p) {
    pack_b_generic<block_q8_1, true, 1>(B, ldb, n, k, (bleg_t *)p);
}

// MODE 0 (offset, Q5_0):  fin += dA ⊙ (dB_j*P - SB_j)          [SB = 16*dB*S]
// MODE 1 (affine, Q4_1/Q5_1): fin += dA*(dB_j*P) + mA*SB_j     [SB = dB*sum(y)]
template <int MODE>
static void kernel_leg_16x8(const aleg_t * PA, const bleg_t * PB,
                            int64_t nb, vfl fin[NR][4]) {
    for (int64_t b = 0; b < nb; b++) {
        const vuc * a = PA->v[b];
        const vuc * y = PB->v[b];
        if (b + 1 < nb) {
            __builtin_prefetch(PA->v[b + 1], 0, 3);
            __builtin_prefetch((const char *)PA->v[b + 1] + 256, 0, 3);
            __builtin_prefetch(PB->v[b + 1], 0, 3);
        }
        __vector_quad acc[2][4];
        for (int i = 0; i < 2; i++)
            for (int g = 0; g < 4; g++)
                __builtin_mma_xxsetaccz(&acc[i][g]);
        for (int x = 0; x < 8; x++) {
            const vuc y0 = y[2*x], y1 = y[2*x + 1];
            const vuc w0 = a[4*x], w1 = a[4*x+1], w2 = a[4*x+2], w3 = a[4*x+3];
            __builtin_mma_xvi8ger4pp(&acc[0][0], y0, w0);
            __builtin_mma_xvi8ger4pp(&acc[0][1], y0, w1);
            __builtin_mma_xvi8ger4pp(&acc[0][2], y0, w2);
            __builtin_mma_xvi8ger4pp(&acc[0][3], y0, w3);
            __builtin_mma_xvi8ger4pp(&acc[1][0], y1, w0);
            __builtin_mma_xvi8ger4pp(&acc[1][1], y1, w1);
            __builtin_mma_xvi8ger4pp(&acc[1][2], y1, w2);
            __builtin_mma_xvi8ger4pp(&acc[1][3], y1, w3);
        }
        vsi pr[2][4][4];
        for (int i = 0; i < 2; i++)
            for (int g = 0; g < 4; g++)
                __builtin_mma_disassemble_acc(pr[i][g], &acc[i][g]);
        for (int i = 0; i < 2; i++) {
            const vfl dB = PB->dB[b][i];
            const vfl SB = PB->SB[b][i];
            for (int g = 0; g < 4; g++) {
                const vfl dA = PA->dA[b][g];
                const vfl mA = PA->mA[b][g];
                for (int r = 0; r < 4; r++) {
                    const vfl res = vec_ctf(pr[i][g][r], 0);
                    if (MODE == 0) {
                        vfl t = vec_msub(res, vec_splat(dB, r), vec_splat(SB, r));
                        fin[4*i + r][g] = vec_madd(t, dA, fin[4*i + r][g]);
                    } else {
                        vfl vs = vec_mul(vec_splat(dB, r), dA);
                        fin[4*i + r][g] = vec_madd(res, vs, fin[4*i + r][g]);
                        fin[4*i + r][g] = vec_madd(mA, vec_splat(SB, r), fin[4*i + r][g]);
                    }
                }
            }
        }
    }
}

template <int MODE>
static void leg_gemm_core(int64_t m, int64_t n, int64_t k,
                          const aleg_t * PA, const bleg_t * PB,
                          float * C, int64_t ldc, int ith, int nth) {
    const int64_t kb = k/32, ns = sl(k), mt = rt(m), njt = ct(n);
    const int64_t tpt = (mt + nth - 1) / nth;
    const int64_t t0 = ith*tpt, t1 = (ith+1)*tpt < mt ? (ith+1)*tpt : mt;
    vfl fin[NR][4];
    for (int64_t it = t0; it < t1; it++) {
        const int64_t i = it*MR;
        const int64_t rows = (m - i) < MR ? (m - i) : MR;
        for (int64_t jt = 0; jt < njt; jt++) {
            for (int j = 0; j < NR; j++)
                for (int g = 0; g < 4; g++) fin[j][g] = vec_splats(0.0f);
            for (int64_t s = 0; s < ns; s++) {
                const int64_t b0 = s*KC_CH;
                const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
                kernel_leg_16x8<MODE>(&PA[it*ns + s], &PB[jt*ns + s], nb, fin);
            }
            const int64_t j0 = jt*NR;
            const int64_t cols = (n - j0) < NR ? (n - j0) : NR;
            for (int64_t cj = 0; cj < cols; cj++) {
                float * dst = C + i + (j0 + cj)*ldc;
                if (rows == MR) {
                    for (int g = 0; g < 4; g++) vec_xst(fin[cj][g], 16*g, dst);
                } else {
                    for (int64_t r = 0; r < rows; r++)
                        dst[r] = fin[cj][r >> 2][r & 3];
                }
            }
        }
    }
}

extern "C" void leg_gemm_offset(int64_t m, int64_t n, int64_t k,
        const void * PA, const void * PB, float * C, int64_t ldc, int ith, int nth) {
    leg_gemm_core<0>(m, n, k, (const aleg_t *)PA, (const bleg_t *)PB, C, ldc, ith, nth);
}
extern "C" void leg_gemm_affine(int64_t m, int64_t n, int64_t k,
        const void * PA, const void * PB, float * C, int64_t ldc, int ith, int nth) {
    leg_gemm_core<1>(m, n, k, (const aleg_t *)PA, (const bleg_t *)PB, C, ldc, ith, nth);
}

#endif // __MMA__

// ---------------------------------------------------------------------------
#ifdef LEGACY_TEST
#include <cstdio>
#include <cmath>

static uint32_t rng = 0x13572468;
static uint32_t xr() { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }

static ggml_half f32_to_f16_approx(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0)   return (ggml_half)sign;
    if (exp >= 31)  return (ggml_half)(sign | 0x7bff);
    return (ggml_half)(sign | (exp << 10) | mant);
}

static int q5code(const uint8_t * qs, const uint8_t * qh4, int j) {
    uint32_t qh; memcpy(&qh, qh4, 4);
    if (j < 16) return (qs[j] & 0xF) | (((qh >> j) << 4) & 0x10);
    return (qs[j-16] >> 4) | ((qh >> (j - 16 + 12)) & 0x10);
}

int main() {
    int fails = 0;
    struct { int64_t m, n, k; } cases[] = {
        { 16, 8, 128 }, { 32, 16, 512 }, { 13, 7, 384 },
        { 64, 16, 2048 }, { 40, 24, 4096 }, { 17, 9, 2304 },
        { 32, 1, 1024 }, { 9, 3, 2176 },
    };
    for (int type = 0; type < 3; type++) {          // 0=q4_1 1=q5_0 2=q5_1
        for (auto & tc : cases) {
            const int64_t m = tc.m, n = tc.n, k = tc.k;
            const int64_t lda = k/32, ldb = k/32, ldc = m;
            block_q4_1 * A1 = nullptr; block_q5_0 * A50 = nullptr; block_q5_1 * A51 = nullptr;
            if (type == 0) {
                A1 = (block_q4_1*)aligned_alloc(64, m*lda*sizeof(block_q4_1));
                for (int64_t i = 0; i < m*lda; i++) {
                    A1[i].d = f32_to_f16_approx(0.0005f + (xr()%1000)/400000.0f);
                    A1[i].m = f32_to_f16_approx(-0.01f + (xr()%1000)/50000.0f);
                    for (int b = 0; b < 16; b++) A1[i].qs[b] = (uint8_t)(xr() & 0xff);
                }
            } else if (type == 1) {
                A50 = (block_q5_0*)aligned_alloc(64, m*lda*sizeof(block_q5_0));
                for (int64_t i = 0; i < m*lda; i++) {
                    A50[i].d = f32_to_f16_approx(0.0005f + (xr()%1000)/400000.0f);
                    for (int b = 0; b < 4; b++)  A50[i].qh[b] = (uint8_t)(xr() & 0xff);
                    for (int b = 0; b < 16; b++) A50[i].qs[b] = (uint8_t)(xr() & 0xff);
                }
            } else {
                A51 = (block_q5_1*)aligned_alloc(64, m*lda*sizeof(block_q5_1));
                for (int64_t i = 0; i < m*lda; i++) {
                    A51[i].d = f32_to_f16_approx(0.0005f + (xr()%1000)/400000.0f);
                    A51[i].m = f32_to_f16_approx(-0.01f + (xr()%1000)/50000.0f);
                    for (int b = 0; b < 4; b++)  A51[i].qh[b] = (uint8_t)(xr() & 0xff);
                    for (int b = 0; b < 16; b++) A51[i].qs[b] = (uint8_t)(xr() & 0xff);
                }
            }
            const bool use_q81 = (type != 1);
            block_q8_0 * B0 = nullptr; block_q8_1 * B1 = nullptr;
            if (use_q81) {
                B1 = (block_q8_1*)aligned_alloc(64, n*ldb*sizeof(block_q8_1));
                for (int64_t i = 0; i < n*ldb; i++) {
                    float d = 0.001f + (xr()%1000)/500000.0f;
                    B1[i].d = f32_to_f16_approx(d);
                    int s = 0;
                    for (int b = 0; b < 32; b++) { B1[i].qs[b] = (int8_t)((int)(xr()%255) - 127); s += B1[i].qs[b]; }
                    B1[i].s = f32_to_f16_approx(fp16_to_fp32(B1[i].d) * (float)s);
                }
            } else {
                B0 = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
                for (int64_t i = 0; i < n*ldb; i++) {
                    B0[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/500000.0f);
                    for (int b = 0; b < 32; b++) B0[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
                }
            }
            float * C = (float*)aligned_alloc(64, m*n*sizeof(float));
            void * PA = aligned_alloc(64, leg_apack_size(m, k));
            void * PB = aligned_alloc(64, leg_bpack_size(n, k));
            if (type == 0) { leg_repack_q4_1(A1, lda, m, k, PA); leg_pack_b_q8_1(B1, ldb, n, k, PB);
                             leg_gemm_affine(m, n, k, PA, PB, C, ldc, 0, 2);
                             leg_gemm_affine(m, n, k, PA, PB, C, ldc, 1, 2); }
            if (type == 1) { leg_repack_q5_0(A50, lda, m, k, PA); leg_pack_b_q8_0(B0, ldb, n, k, PB);
                             leg_gemm_offset(m, n, k, PA, PB, C, ldc, 0, 2);
                             leg_gemm_offset(m, n, k, PA, PB, C, ldc, 1, 2); }
            if (type == 2) { leg_repack_q5_1(A51, lda, m, k, PA); leg_pack_b_q8_1(B1, ldb, n, k, PB);
                             leg_gemm_affine(m, n, k, PA, PB, C, ldc, 0, 2);
                             leg_gemm_affine(m, n, k, PA, PB, C, ldc, 1, 2); }

            double emax = 0, scale = 0;
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++) {
                    double ref = 0;
                    for (int64_t b = 0; b < k/32; b++) {
                        double bacc = 0;
                        for (int l = 0; l < 32; l++) {
                            double w;
                            if (type == 0) {
                                const block_q4_1 * bp = &A1[i*lda + b];
                                int q = l < 16 ? (bp->qs[l] & 0xF) : (bp->qs[l-16] >> 4);
                                w = fp16_to_fp32(bp->d)*q + fp16_to_fp32(bp->m);
                            } else if (type == 1) {
                                const block_q5_0 * bp = &A50[i*lda + b];
                                w = fp16_to_fp32(bp->d)*(q5code(bp->qs, bp->qh, l) - 16);
                            } else {
                                const block_q5_1 * bp = &A51[i*lda + b];
                                w = fp16_to_fp32(bp->d)*q5code(bp->qs, bp->qh, l) + fp16_to_fp32(bp->m);
                            }
                            const int8_t yv = use_q81 ? B1[j*ldb + b].qs[l] : B0[j*ldb + b].qs[l];
                            bacc += w * yv;
                        }
                        const double dy = use_q81 ? fp16_to_fp32(B1[j*ldb + b].d)
                                                  : fp16_to_fp32(B0[j*ldb + b].d);
                        // affine ref uses the STORED s for the min term, as the kernel does
                        if (type != 1) {
                            const block_q8_1 * yb = &B1[j*ldb + b];
                            double P = 0;
                            for (int l = 0; l < 32; l++) {
                                int q;
                                if (type == 0) {
                                    const block_q4_1 * bp = &A1[i*lda + b];
                                    q = l < 16 ? (bp->qs[l] & 0xF) : (bp->qs[l-16] >> 4);
                                    ref += fp16_to_fp32(bp->d)*q*dy*yb->qs[l];
                                } else {
                                    const block_q5_1 * bp = &A51[i*lda + b];
                                    q = q5code(bp->qs, bp->qh, l);
                                    ref += fp16_to_fp32(bp->d)*q*dy*yb->qs[l];
                                }
                                (void)q; (void)P;
                            }
                            const double mm = type == 0 ? fp16_to_fp32(A1[i*lda + b].m)
                                                        : fp16_to_fp32(A51[i*lda + b].m);
                            ref += mm * (double)fp16_to_fp32(yb->s);
                        } else {
                            ref += bacc * dy;
                        }
                    }
                    scale += fabs(ref);
                    double e = fabs((double)C[i + j*ldc] - ref);
                    if (e > emax) emax = e;
                }
            scale = scale/(m*n) + 1e-30; emax /= scale;
            bool ok = emax < 1e-5;
            const char * nm = type == 0 ? "q4_1" : type == 1 ? "q5_0" : "q5_1";
            printf("%s m=%3lld n=%3lld k=%5lld  err=%.3g  %s\n",
                   nm, (long long)m,(long long)n,(long long)k, emax, ok?"OK":"FAIL");
            if (!ok) fails++;
            free(A1); free(A50); free(A51); free(B0); free(B1);
            free(C); free(PA); free(PB);
        }
    }
    printf(fails ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fails ? 1 : 0;
}
#endif
