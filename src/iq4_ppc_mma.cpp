// iq4_ppc_mma.cpp
//
// POWER10/POWER11 MMA GEMM for the IQ4 codebook quants:
//   IQ4_NL x Q8_0   (32-element blocks)
//   IQ4_XS x Q8_K   (256 superblocks, per-32 6-bit signed scales)
//
// These are the widely used modern imatrix 4-bit formats. Unlike every
// other kernel in this project, the weight values are SIGNED (a 16-entry
// int8 codebook, kvalues_iq4nl, -127..+113), which flips the operand
// orientation back to the v1/v2 scheme:
//
//   * codebook lookup is ONE vec_perm per 16 codes (nibbles index the
//     table vector directly),
//   * signed weights go on the signed GER operand; activations are
//     XOR-0x80 flipped onto the unsigned operand,
//   * bias correction  sum(w*(y+128)) = sum(w*y) + 128*sum(w)  uses the
//     per-(row, chunk) codebook-value sums W computed once at repack,
//     pre-folded as C128 = 128*W*dA so the fixup is
//         fin += dB ⊙ (P*dA - C128)      (one vec_msub + one vec_madd).
//
// Because the accumulator orientation flips with the operands (rows =
// weight rows here), this file uses an 8x8 tile on 4 accumulators --
// which the K-quant register-pressure analysis (DESIGN.md) suggests may
// be the better shape anyway.  MR = NR = 8.
//
// Conventions: A row-major in blocks (lda: iq4_nl blocks of 32, or
// iq4_xs superblocks of 256), B row-major in q8_0 blocks / q8_K
// superblocks, C column-major float.  k % 32 == 0 (NL), k % 256 == 0
// (XS).  Tests: -DIQ4_TEST.

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define QK4_NL 32
#define QK8_0  32
#define QK_K   256

typedef uint16_t ggml_half;

typedef struct { ggml_half d; uint8_t qs[QK4_NL/2]; } block_iq4_nl;
#define QK_MXFP4 32
typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4;
typedef struct {
    ggml_half d;
    uint16_t  scales_h;
    uint8_t   scales_l[QK_K/64];
    uint8_t   qs[QK_K/2];
} block_iq4_xs;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
#define QK4_0 32
typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, "bad iq4_nl");
static_assert(sizeof(block_mxfp4) == 1 + QK_MXFP4/2, "bad mxfp4");
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) + 2 + QK_K/64 + QK_K/2, "bad iq4_xs");

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
// MXFP4 e2m1 values, pre-doubled to integers (the E8M0 "half" scale
// supplies the /2): {0,1,2,3,4,6,8,12} with sign.
static const int8_t kvalues_mxfp4_[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};
#include <cmath>
static inline float e8m0_half_to_fp32(uint8_t e) { return ldexpf(1.0f, (int)e - 128); }

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

// Unaligned 16-byte load via memcpy: compiles to a single lxv (which is
// alignment-agnostic on POWER) while staying well-defined C++ for any
// source alignment -- several block structs place qs at odd offsets.
static inline vuc load16u(const void * p) { vuc v; memcpy(&v, p, 16); return v; }


#define KC_ELEMS  2048                    // K slab
#define KC_CH     (KC_ELEMS / 32)         // 32-element chunks per slab
#define MR        8
#define NR        8

// packed weights: per chunk, 8 depth-steps x 2 rowgroups of signed values
typedef struct {
    vuc v[KC_CH][16];
    vfl sA  [KC_CH][2];                   // per-row scale (dA or d*(ls-32))
    vfl C128[KC_CH][2];                   // 128 * W * scale, pre-folded
} aiq4_t;

// packed activations: per chunk, 8 depth-steps x 2 colgroups, flipped
typedef struct {
    vuc v[KC_CH][16];
    vfl dB[KC_CH][2];                     // per-chunk column scales
} biq4_t;

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

// 32 signed codebook values from 16 packed bytes; elements j / j+16
// come from qs[j] low / high nibble.  One vec_perm each.
static inline void iq4_lookup32t(const uint8_t * qs, const int8_t * table, vsc v[2]) {
    vsc tbl; memcpy(&tbl, table, 16);
    vuc raw = load16u((const uint8_t *)(qs) + (0));
    vuc lo  = vec_and(raw, vec_splats((unsigned char)0xF));
    vuc hi  = vec_sr(raw, vec_splats((unsigned char)4));
    v[0] = vec_perm(tbl, tbl, lo);
    v[1] = vec_perm(tbl, tbl, hi);
}

static inline int hsum(vsi s) { return s[0] + s[1] + s[2] + s[3]; }

// pack one chunk's 8 weight rows into T at chunk ch; w[r] = row values,
// scale[r] = per-row scale for this chunk.
static void iq4_place_chunk(aiq4_t * T, int64_t ch,
                            const vsc w[MR][2], const float scale[MR]) {
    for (int g = 0; g < 2; g++) {
        float W[4];
        vui rows4[4];
        for (int h = 0; h < 2; h++) {
            for (int r = 0; r < 4; r++) rows4[r] = (vui)w[4*g + r][h];
            mma_transpose4(rows4, &T->v[ch][8*h + g], 2);
        }
        for (int r = 0; r < 4; r++) {
            vsi z = vec_splats(0);
            vsi s = vec_sum4s(w[4*g + r][0], z);
            s = vec_sum4s(w[4*g + r][1], s);
            W[r] = (float)hsum(s);
        }
        T->sA  [ch][g] = (vfl){ scale[4*g], scale[4*g+1], scale[4*g+2], scale[4*g+3] };
        // exact integer correction 128*W (|W|<=32*127 so 128*W < 2^24:
        // exactly representable; the fixup subtracts it from the exact
        // integer accumulator BEFORE scaling, so the subtraction is
        // exact and the result matches ggml's scalar rounding order)
        T->C128[ch][g] = (vfl){ 128.0f*W[0], 128.0f*W[1],
                                128.0f*W[2], 128.0f*W[3] };
    }
}

static inline int64_t rt8(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct8(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl32(int64_t k) { return (k/32 + KC_CH - 1) / KC_CH; }

extern "C" size_t iq4_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt8(m) * sl32(k)) * sizeof(aiq4_t)) + 63) & ~(size_t)63;
}
extern "C" size_t iq4_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct8(n) * sl32(k)) * sizeof(biq4_t)) + 63) & ~(size_t)63;
}

// ---- weight repack: IQ4_NL ----
extern "C" void iq4nl_repack_a(const block_iq4_nl * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    aiq4_t * P = (aiq4_t *)packed;
    const int64_t kb = k/32, ns = sl32(k);
    for (int64_t it = 0; it < rt8(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aiq4_t * T = &P[it*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            vsc w[MR][2]; float sc[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                const block_iq4_nl * bp = &A[rr*lda + b0 + b];
                sc[r] = fp16_to_fp32(bp->d);
                iq4_lookup32t(bp->qs, kvalues_iq4nl, w[r]);
            }
            iq4_place_chunk(T, b, w, sc);
        }
    }
}

// ---- weight repack: Q8_0 (signed int8 weights, identity decode) ----
extern "C" void q8_0_repack_a(const block_q8_0 * A, int64_t lda,
                              int64_t m, int64_t k, void * packed) {
    aiq4_t * P = (aiq4_t *)packed;
    const int64_t kb = k/32, ns = sl32(k);
    for (int64_t it = 0; it < rt8(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aiq4_t * T = &P[it*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            vsc w[MR][2]; float sc[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                const block_q8_0 * bp = &A[rr*lda + b0 + b];
                sc[r] = fp16_to_fp32(bp->d);
                w[r][0] = (vsc)load16u(bp->qs);
                w[r][1] = (vsc)load16u(bp->qs + 16);
            }
            iq4_place_chunk(T, b, w, sc);
        }
    }
}

// ---- weight repack: MXFP4 (same shape as IQ4_NL, different table+scale) ----
extern "C" void mxfp4_repack_a(const block_mxfp4 * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    aiq4_t * P = (aiq4_t *)packed;
    const int64_t kb = k/32, ns = sl32(k);
    for (int64_t it = 0; it < rt8(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aiq4_t * T = &P[it*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            vsc w[MR][2]; float sc[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                const block_mxfp4 * bp = &A[rr*lda + b0 + b];
                sc[r] = e8m0_half_to_fp32(bp->e);
                iq4_lookup32t(bp->qs, kvalues_mxfp4_, w[r]);
            }
            iq4_place_chunk(T, b, w, sc);
        }
    }
}

// ---- weight repack: IQ4_XS ----
extern "C" void iq4xs_repack_a(const block_iq4_xs * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    aiq4_t * P = (aiq4_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl32(k);
    for (int64_t it = 0; it < rt8(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aiq4_t * T = &P[it*ns + s];
        const int64_t sb0 = (s*KC_CH)/8;
        const int64_t nsl = (nsb - sb0) < KC_CH/8 ? (nsb - sb0) : KC_CH/8;
        for (int64_t sb = 0; sb < nsl; sb++) {
            const block_iq4_xs * bp[MR]; float d[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                bp[r] = &A[rr*lda + sb0 + sb];
                d[r]  = fp16_to_fp32(bp[r]->d);
            }
            for (int ib = 0; ib < 8; ib++) {           // 32-groups
                vsc w[MR][2]; float sc[MR];
                for (int r = 0; r < MR; r++) {
                    const int ls = ((bp[r]->scales_l[ib/2] >> 4*(ib%2)) & 0xF)
                                 | (((bp[r]->scales_h >> 2*ib) & 3) << 4);
                    sc[r] = d[r] * (float)(ls - 32);
                    iq4_lookup32t(bp[r]->qs + 16*ib, kvalues_iq4nl, w[r]);
                }
                iq4_place_chunk(T, 8*sb + ib, w, sc);
            }
        }
    }
}

// ---- activation packs ----
extern "C" void iq4_pack_b_q8_0(const block_q8_0 * B, int64_t ldb,
                                int64_t n, int64_t k, void * packed) {
    biq4_t * P = (biq4_t *)packed;
    const int64_t kb = k/32, ns = sl32(k);
    const vuc flip = vec_splats((unsigned char)0x80);
    for (int64_t jt = 0; jt < ct8(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        biq4_t * T = &P[jt*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            const block_q8_0 * yb[NR]; float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + b0 + b];
                dB[j] = fp16_to_fp32(yb[j]->d);
            }
            T->dB[b][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            T->dB[b][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            vui rows4[4];
            for (int a = 0; a < 2; a++)
                for (int h = 0; h < 2; h++) {
                    for (int j = 0; j < 4; j++)
                        rows4[j] = (vui)vec_xor(
                            load16u((const uint8_t *)(yb[4*a + j]->qs) + (16*h)), flip);
                    mma_transpose4(rows4, &T->v[b][8*h + a], 2);
                }
        }
    }
}

extern "C" void iq4_pack_b_q8_K(const block_q8_K * B, int64_t ldb,
                                int64_t n, int64_t k, void * packed) {
    biq4_t * P = (biq4_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl32(k);
    const vuc flip = vec_splats((unsigned char)0x80);
    for (int64_t jt = 0; jt < ct8(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        biq4_t * T = &P[jt*ns + s];
        const int64_t sb0 = (s*KC_CH)/8;
        const int64_t nsl = (nsb - sb0) < KC_CH/8 ? (nsb - sb0) : KC_CH/8;
        for (int64_t sb = 0; sb < nsl; sb++) {
            const block_q8_K * yb[NR]; float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + sb0 + sb];
                dB[j] = yb[j]->d;
            }
            for (int ib = 0; ib < 8; ib++) {
                const int64_t ch = 8*sb + ib;
                T->dB[ch][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
                T->dB[ch][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
                vui rows4[4];
                for (int a = 0; a < 2; a++)
                    for (int h = 0; h < 2; h++) {
                        for (int j = 0; j < 4; j++)
                            rows4[j] = (vui)vec_xor(
                                load16u((const uint8_t *)(yb[4*a + j]->qs) + (32*ib + 16*h)), flip);
                        mma_transpose4(rows4, &T->v[ch][8*h + a], 2);
                    }
            }
        }
    }
}

// ---- 8x8 microkernel on 4 accumulators (weights on the signed operand;
//      acc rows = weight rows) ----
static void kernel_iq4_8x8(const aiq4_t * PA, const biq4_t * PB,
                           int64_t nch, vfl fin[MR][2]) {
    for (int64_t ch = 0; ch < nch; ch++) {
        const vuc * a = PA->v[ch];
        const vuc * y = PB->v[ch];
        if (ch + 1 < nch) {
            // each chunk's packed panel spans two 128B lines; touch both
            __builtin_prefetch(PA->v[ch + 1], 0, 3);
            __builtin_prefetch((const char *)PA->v[ch + 1] + 128, 0, 3);
            __builtin_prefetch(PB->v[ch + 1], 0, 3);
            __builtin_prefetch((const char *)PB->v[ch + 1] + 128, 0, 3);
        }
        __vector_quad acc[2][2];
        for (int g = 0; g < 2; g++)
            for (int cgi = 0; cgi < 2; cgi++)
                __builtin_mma_xxsetaccz(&acc[g][cgi]);
        for (int x = 0; x < 8; x++) {
            const vuc w0 = a[2*x], w1 = a[2*x + 1];
            const vuc y0 = y[2*x], y1 = y[2*x + 1];
            __builtin_mma_xvi8ger4pp(&acc[0][0], w0, y0);
            __builtin_mma_xvi8ger4pp(&acc[0][1], w0, y1);
            __builtin_mma_xvi8ger4pp(&acc[1][0], w1, y0);
            __builtin_mma_xvi8ger4pp(&acc[1][1], w1, y1);
        }
        for (int g = 0; g < 2; g++) {
            const vfl sA   = PA->sA  [ch][g];
            const vfl C128 = PA->C128[ch][g];
            for (int cgi = 0; cgi < 2; cgi++) {
                vsi rowsP[4];
                __builtin_mma_disassemble_acc(rowsP, &acc[g][cgi]);
                const vfl dB = PB->dB[ch][cgi];
                // fin += dB ⊙ (P*sA_r - C128_r)  per weight row r
                for (int r = 0; r < 4; r++) {
                    // exact int subtract, then one (dA*dB) product, then
                    // fma -- the same rounding sequence as ggml's scalar
                    // vec_dot, restoring temp-0 bit identity (field FAIL,
                    // Q8_0 27B, 2026-07-21)
                    vfl t = vec_sub(vec_ctf(rowsP[r],0), vec_splats(C128[r]));
                    vfl sc = vec_mul(vec_splats(sA[r]), dB);
                    fin[4*g + r][cgi] = vec_madd(t, sc, fin[4*g + r][cgi]);
                }
            }
        }
    }
}

static void iq4_gemm_core(int64_t m, int64_t n, int64_t k,
                          const aiq4_t * PA, const biq4_t * PB,
                          float * C, int64_t ldc, int ith, int nth) {
    const int64_t kb = k/32, ns = sl32(k), mt = rt8(m), njt = ct8(n);
    const int64_t tpt = (mt + nth - 1) / nth;
    const int64_t t0 = ith*tpt, t1 = (ith+1)*tpt < mt ? (ith+1)*tpt : mt;
    vfl fin[MR][2];
    for (int64_t it = t0; it < t1; it++) {
        const int64_t i = it*MR;
        const int64_t rows = (m - i) < MR ? (m - i) : MR;
        for (int64_t jt = 0; jt < njt; jt++) {
            for (int r = 0; r < MR; r++) fin[r][0] = fin[r][1] = vec_splats(0.0f);
            for (int64_t s = 0; s < ns; s++) {
                const int64_t b0 = s*KC_CH;
                const int64_t nch = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
                kernel_iq4_8x8(&PA[it*ns + s], &PB[jt*ns + s], nch, fin);
            }
            const int64_t j0 = jt*NR;
            const int64_t cols = (n - j0) < NR ? (n - j0) : NR;
            for (int64_t r = 0; r < rows; r++)
                for (int64_t cj = 0; cj < cols; cj++)
                    C[(i + r) + (j0 + cj)*ldc] = fin[r][cj >> 2][cj & 3];
        }
    }
}

extern "C" void iq4_gemm_packed(int64_t m, int64_t n, int64_t k,
                                const void * packedA, const void * packedB,
                                float * C, int64_t ldc, int ith, int nth) {
    iq4_gemm_core(m, n, k, (const aiq4_t *)packedA, (const biq4_t *)packedB,
                  C, ldc, ith, nth);
}

#endif // __MMA__

// ---------------------------------------------------------------------------
#ifdef IQ4_TEST
#include <cstdio>
#include <cmath>

static uint32_t rng = 0x77777771;
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

static double dref_nl(int64_t k, const block_iq4_nl * x, const block_q8_0 * y) {
    double acc = 0;
    for (int64_t b = 0; b < k/32; b++) {
        const double d = fp16_to_fp32(x[b].d);
        const double dy = fp16_to_fp32(y[b].d);
        long s = 0;
        for (int j = 0; j < 16; j++) {
            s += (long)kvalues_iq4nl[x[b].qs[j] & 0xF] * y[b].qs[j];
            s += (long)kvalues_iq4nl[x[b].qs[j] >>  4] * y[b].qs[j + 16];
        }
        acc += d * dy * (double)s;
    }
    return acc;
}

static double dref_xs(int64_t k, const block_iq4_xs * x, const block_q8_K * y) {
    double acc = 0;
    for (int64_t sb = 0; sb < k/QK_K; sb++) {
        const double d = fp16_to_fp32(x[sb].d);
        double bacc = 0;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((x[sb].scales_l[ib/2] >> 4*(ib%2)) & 0xF)
                         | (((x[sb].scales_h >> 2*ib) & 3) << 4);
            const double dl = d * (ls - 32);
            long s = 0;
            for (int j = 0; j < 16; j++) {
                s += (long)kvalues_iq4nl[x[sb].qs[16*ib + j] & 0xF] * y[sb].qs[32*ib + j];
                s += (long)kvalues_iq4nl[x[sb].qs[16*ib + j] >>  4] * y[sb].qs[32*ib + j + 16];
            }
            bacc += dl * (double)s;
        }
        acc += (double)y[sb].d * bacc;
    }
    return acc;
}

int main() {
    int fails = 0;
    struct { int64_t m, n, k; } cases[] = {
        { 8, 8, 256 }, { 16, 16, 512 }, { 13, 7, 768 },
        { 64, 16, 2048 }, { 40, 24, 4096 }, { 17, 9, 2304 },
        { 32, 1, 2048 }, { 9, 3, 4352 },
    };
    // ---- IQ4_NL ----
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        const int64_t lda = k/32, ldb = k/32, ldc = m;
        block_iq4_nl * A = (block_iq4_nl*)aligned_alloc(64, m*lda*sizeof(block_iq4_nl));
        block_q8_0 * B = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
        float * C = (float*)aligned_alloc(64, m*n*sizeof(float));
        for (int64_t i = 0; i < m*lda; i++) {
            A[i].d = f32_to_f16_approx(0.0005f + (xr()%1000)/400000.0f);
            for (int b = 0; b < 16; b++) A[i].qs[b] = (uint8_t)(xr() & 0xff);
        }
        for (int64_t i = 0; i < n*ldb; i++) {
            B[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/500000.0f);
            for (int b = 0; b < 32; b++) B[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
        }
        void * PA = aligned_alloc(64, iq4_apack_size(m, k));
        void * PB = aligned_alloc(64, iq4_bpack_size(n, k));
        iq4nl_repack_a(A, lda, m, k, PA);
        iq4_pack_b_q8_0(B, ldb, n, k, PB);
        iq4_gemm_packed(m, n, k, PA, PB, C, ldc, 0, 2);
        iq4_gemm_packed(m, n, k, PA, PB, C, ldc, 1, 2);
        double emax = 0, scale = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double ref = dref_nl(k, A + i*lda, B + j*ldb);
                scale += fabs(ref);
                double e = fabs((double)C[i + j*ldc] - ref);
                if (e > emax) emax = e;
            }
        scale = scale/(m*n) + 1e-30; emax /= scale;
        bool ok = emax < 1e-5;
        printf("iq4_nl m=%3lld n=%3lld k=%5lld  err=%.3g  %s\n",
               (long long)m,(long long)n,(long long)k, emax, ok?"OK":"FAIL");
        if (!ok) fails++;
        free(A); free(B); free(C); free(PA); free(PB);
    }
    // ---- Q8_0 x Q8_0 ----
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        const int64_t lda = k/32, ldb = k/32, ldc = m;
        block_q8_0 * A = (block_q8_0*)aligned_alloc(64, ((m*lda*sizeof(block_q8_0))+63)&~63ul);
        block_q8_0 * B = (block_q8_0*)aligned_alloc(64, ((n*ldb*sizeof(block_q8_0))+63)&~63ul);
        float * C = (float*)aligned_alloc(64, ((m*n*sizeof(float))+63)&~63ul);
        for (int64_t i = 0; i < m*lda; i++) {
            A[i].d = f32_to_f16_approx(0.0005f + (xr()%1000)/400000.0f);
            for (int b = 0; b < 32; b++) A[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
        }
        for (int64_t i = 0; i < n*ldb; i++) {
            B[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/500000.0f);
            for (int b = 0; b < 32; b++) B[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
        }
        void * PA = aligned_alloc(64, iq4_apack_size(m, k));
        void * PB = aligned_alloc(64, iq4_bpack_size(n, k));
        q8_0_repack_a(A, lda, m, k, PA);
        iq4_pack_b_q8_0(B, ldb, n, k, PB);
        iq4_gemm_packed(m, n, k, PA, PB, C, ldc, 0, 2);
        iq4_gemm_packed(m, n, k, PA, PB, C, ldc, 1, 2);
        double emax = 0, scale = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double ref = 0;
                for (int64_t b = 0; b < k/32; b++) {
                    long s2 = 0;
                    for (int l = 0; l < 32; l++) s2 += (long)A[i*lda+b].qs[l] * B[j*ldb+b].qs[l];
                    ref += fp16_to_fp32(A[i*lda+b].d) * (double)fp16_to_fp32(B[j*ldb+b].d) * (double)s2;
                }
                scale += fabs(ref);
                double e = fabs((double)C[i + j*ldc] - ref);
                if (e > emax) emax = e;
            }
        scale = scale/(m*n) + 1e-30; emax /= scale;
        bool ok = emax < 1e-5;
        printf("q8_0   m=%3lld n=%3lld k=%5lld  err=%.3g  %s\n",
               (long long)m,(long long)n,(long long)k, emax, ok?"OK":"FAIL");
        if (!ok) fails++;
        free(A); free(B); free(C); free(PA); free(PB);
    }
    // ---- MXFP4 ----
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        const int64_t lda = k/32, ldb = k/32, ldc = m;
        block_mxfp4 * A = (block_mxfp4*)aligned_alloc(64, ((m*lda*sizeof(block_mxfp4))+63)&~63ul);
        block_q8_0 * B = (block_q8_0*)aligned_alloc(64, ((n*ldb*sizeof(block_q8_0))+63)&~63ul);
        float * C = (float*)aligned_alloc(64, ((m*n*sizeof(float))+63)&~63ul);
        for (int64_t i = 0; i < m*lda; i++) {
            A[i].e = (uint8_t)(120 + xr()%12);
            for (int b = 0; b < 16; b++) A[i].qs[b] = (uint8_t)(xr() & 0xff);
        }
        for (int64_t i = 0; i < n*ldb; i++) {
            B[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/500000.0f);
            for (int b = 0; b < 32; b++) B[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
        }
        void * PA = aligned_alloc(64, iq4_apack_size(m, k));
        void * PB = aligned_alloc(64, iq4_bpack_size(n, k));
        mxfp4_repack_a(A, lda, m, k, PA);
        iq4_pack_b_q8_0(B, ldb, n, k, PB);
        iq4_gemm_packed(m, n, k, PA, PB, C, ldc, 0, 2);
        iq4_gemm_packed(m, n, k, PA, PB, C, ldc, 1, 2);
        double emax = 0, scale = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double ref = 0;
                for (int64_t b = 0; b < k/32; b++) {
                    const double d = e8m0_half_to_fp32(A[i*lda + b].e);
                    const double dy = fp16_to_fp32(B[j*ldb + b].d);
                    long s = 0;
                    for (int jj = 0; jj < 16; jj++) {
                        s += (long)kvalues_mxfp4_[A[i*lda+b].qs[jj] & 0xF] * B[j*ldb+b].qs[jj];
                        s += (long)kvalues_mxfp4_[A[i*lda+b].qs[jj] >>  4] * B[j*ldb+b].qs[jj+16];
                    }
                    ref += d * dy * (double)s;
                }
                scale += fabs(ref);
                double e = fabs((double)C[i + j*ldc] - ref);
                if (e > emax) emax = e;
            }
        scale = scale/(m*n) + 1e-30; emax /= scale;
        bool ok = emax < 1e-5;
        printf("mxfp4  m=%3lld n=%3lld k=%5lld  err=%.3g  %s\n",
               (long long)m,(long long)n,(long long)k, emax, ok?"OK":"FAIL");
        if (!ok) fails++;
        free(A); free(B); free(C); free(PA); free(PB);
    }
    // ---- IQ4_XS ----
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        if (k % QK_K) continue;
        const int64_t lda = k/QK_K, ldb = k/QK_K, ldc = m;
        block_iq4_xs * A = (block_iq4_xs*)aligned_alloc(64, m*lda*sizeof(block_iq4_xs));
        block_q8_K * B = (block_q8_K*)aligned_alloc(64, n*ldb*sizeof(block_q8_K));
        float * C = (float*)aligned_alloc(64, m*n*sizeof(float));
        for (int64_t i = 0; i < m*lda; i++) {
            A[i].d = f32_to_f16_approx(0.0005f + (xr()%1000)/400000.0f);
            A[i].scales_h = (uint16_t)(xr() & 0xffff);
            for (int b = 0; b < QK_K/64; b++) A[i].scales_l[b] = (uint8_t)(xr() & 0xff);
            for (int b = 0; b < QK_K/2;  b++) A[i].qs[b] = (uint8_t)(xr() & 0xff);
        }
        for (int64_t i = 0; i < n*ldb; i++) {
            B[i].d = 0.001f + (xr()%1000)/500000.0f;
            for (int b = 0; b < QK_K; b++) B[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
            for (int g = 0; g < QK_K/16; g++) {
                int s = 0; for (int l = 0; l < 16; l++) s += B[i].qs[16*g + l];
                B[i].bsums[g] = (int16_t)s;
            }
        }
        void * PA = aligned_alloc(64, iq4_apack_size(m, k));
        void * PB = aligned_alloc(64, iq4_bpack_size(n, k));
        iq4xs_repack_a(A, lda, m, k, PA);
        iq4_pack_b_q8_K(B, ldb, n, k, PB);
        iq4_gemm_packed(m, n, k, PA, PB, C, ldc, 0, 2);
        iq4_gemm_packed(m, n, k, PA, PB, C, ldc, 1, 2);
        double emax = 0, scale = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double ref = dref_xs(k, A + i*lda, B + j*ldb);
                scale += fabs(ref);
                double e = fabs((double)C[i + j*ldc] - ref);
                if (e > emax) emax = e;
            }
        scale = scale/(m*n) + 1e-30; emax /= scale;
        bool ok = emax < 1e-5;
        printf("iq4_xs m=%3lld n=%3lld k=%5lld  err=%.3g  %s\n",
               (long long)m,(long long)n,(long long)k, emax, ok?"OK":"FAIL");
        if (!ok) fails++;
        free(A); free(B); free(C); free(PA); free(PB);
    }
    printf(fails ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fails ? 1 : 0;
}
#endif
