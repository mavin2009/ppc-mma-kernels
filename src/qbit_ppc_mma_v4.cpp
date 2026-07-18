// qbit_ppc_mma_v4.cpp
//
// v4: production-shaped API for the Q1_0/Q2_0 x Q8_0 POWER MMA kernels.
// Fixes the performance debt documented in DESIGN.md for v3:
//
//   1. WEIGHT REPACK IS ONE-TIME.  qbit_repack_q1/q2() converts the whole
//      weight matrix into MMA-ready packed form once (in a ggml
//      integration this runs in repack.cpp at model load); the GEMM hot
//      path no longer unpacks bits per slab.
//   2. ACTIVATION PACK IS SHARED.  qbit_pack_b() packs B once for all
//      threads (belongs next to activation quantization); the GEMM takes
//      the packed buffer instead of re-packing per thread.
//   3. GEMV PATH FOR TOKEN GENERATION.  For n <= QBIT_GEMV_NMAX the
//      kernel switches to a select-and-sum path that reads the RAW 1/2-bit
//      weights (8x/4x less memory traffic than packed int8 -- and GEMV is
//      bandwidth-bound), exploiting:
//         q1:  sum(t*y) = 2*sum_{bit=1}(y) - sum(y)
//         q2:  sum(t*y) = sum_{c0=1}(y) + 2*sum_{c1=1}(y) - sum(y)
//      with the per-chunk activation sums S computed once and shared
//      across all rows.  No multiplies in the inner loop at all.
//
// Same conventions as v3 (see that file / docs/DESIGN.md); k % 128 == 0.
//
// Build tests: -DQBIT4_TEST ; bench: -DQBIT4_BENCH.

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define QK1_0 128
#define QK2_0 128
#define QK8_0 32

typedef uint16_t ggml_half;
typedef struct { ggml_half d; uint8_t qs[QK1_0 / 8]; } block_q1_0;
typedef struct { ggml_half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;
typedef struct { ggml_half d; int8_t  qs[QK8_0];     } block_q8_0;

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


#define KC_BLKS   16
#define KC_CHUNKS (KC_BLKS * 4)
#define MR        16
#define NR        8
#define QBIT_GEMV_NMAX 2   // n <= this uses the raw-weight GEMV path

typedef struct {
    vuc v[KC_CHUNKS][32];
    vfl dA[KC_BLKS][4];
} apack_t;

typedef struct {
    vuc v[KC_CHUNKS][16];
    vfl dB[KC_CHUNKS][2];
    vfl E[KC_BLKS][2];
} bpack_t;

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

static inline void q1_codes32(vuc raw, int c, vuc v[2]) {
    const vuc rep0 = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc rep1 = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc sh   = { 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7 };
    const vuc m1   = vec_splats((unsigned char)1);
    const vuc off  = vec_splats((unsigned char)(4*c));
    v[0] = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep0, off)), sh), m1);
    v[1] = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep1, off)), sh), m1);
}

static inline void q2_codes32(vuc lo, vuc hi, int c, vuc v[2]) {
    const vuc rep0 = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
    const vuc rep1 = { 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7 };
    const vuc sh   = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    const vuc m3   = vec_splats((unsigned char)3);
    const vuc off  = vec_splats((unsigned char)(8*c));
    v[0] = vec_and(vec_sr(vec_perm(lo, hi, vec_add(rep0, off)), sh), m3);
    v[1] = vec_and(vec_sr(vec_perm(lo, hi, vec_add(rep1, off)), sh), m3);
}

// ---------------- one-time packing API ----------------

static inline int64_t n_row_tiles(int64_t m)  { return (m + MR - 1) / MR; }
static inline int64_t n_col_tiles(int64_t n)  { return (n + NR - 1) / NR; }
static inline int64_t n_slabs(int64_t k)      { return (k/128 + KC_BLKS - 1) / KC_BLKS; }

extern "C" size_t qbit_apack_size(int64_t m, int64_t k) {
    return (((size_t)(n_row_tiles(m) * n_slabs(k)) * sizeof(apack_t)) + 63) & ~(size_t)63;
}
extern "C" size_t qbit_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(n_col_tiles(n) * n_slabs(k)) * sizeof(bpack_t)) + 63) & ~(size_t)63;
}

// packed layout: tile-major, slab-minor:  P[tile * n_slabs + slab]

template <typename BLK, void (*CODES)(const BLK *, int, vuc[2])>
static void repack_rows(const BLK * A, int64_t lda, int64_t m, int64_t k, apack_t * P) {
    const int64_t kb = k / 128, ns = n_slabs(k);
    for (int64_t it = 0; it < n_row_tiles(m); it++) {
        for (int64_t s = 0; s < ns; s++) {
            apack_t * T = &P[it*ns + s];
            const int64_t blk0 = s*KC_BLKS;
            const int64_t nblk = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
            for (int64_t b = 0; b < nblk; b++) {
                const BLK * bp[MR]; float d[MR];
                for (int r = 0; r < MR; r++) {
                    int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                    bp[r] = &A[rr*lda + blk0 + b];
                    d[r]  = fp16_to_fp32(bp[r]->d);
                }
                for (int g = 0; g < 4; g++)
                    T->dA[b][g] = (vfl){ d[4*g], d[4*g+1], d[4*g+2], d[4*g+3] };
                for (int c = 0; c < 4; c++) {
                    vuc t[MR][2];
                    for (int r = 0; r < MR; r++) CODES(bp[r], c, t[r]);
                    vui rows4[4];
                    for (int g = 0; g < 4; g++)
                        for (int h = 0; h < 2; h++) {
                            for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                            mma_transpose4(rows4, &T->v[4*b + c][16*h + g], 4);
                        }
                }
            }
        }
    }
}

static void q1_blk_codes(const block_q1_0 * bp, int c, vuc v[2]) {
    q1_codes32(load16u((const uint8_t *)(bp->qs) + (0)), c, v);
}
static void q2_blk_codes(const block_q2_0 * bp, int c, vuc v[2]) {
    q2_codes32(load16u((const uint8_t *)(bp->qs) + (0)),
               load16u((const uint8_t *)(bp->qs) + (16)), c, v);
}

extern "C" void qbit_repack_q1(const block_q1_0 * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    repack_rows<block_q1_0, q1_blk_codes>(A, lda, m, k, (apack_t *)packed);
}
extern "C" void qbit_repack_q2(const block_q2_0 * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    repack_rows<block_q2_0, q2_blk_codes>(A, lda, m, k, (apack_t *)packed);
}

// Pack all of B once; call from one thread (or split by col tile).
extern "C" void qbit_pack_b(const block_q8_0 * B, int64_t ldb,
                            int64_t n, int64_t k, void * packed) {
    bpack_t * P = (bpack_t *)packed;
    const int64_t kb = k / 128, ns = n_slabs(k);
    for (int64_t jt = 0; jt < n_col_tiles(n); jt++) {
        for (int64_t s = 0; s < ns; s++) {
            bpack_t * T = &P[jt*ns + s];
            const int64_t blk0 = s*KC_BLKS;
            const int64_t nblk = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
            for (int64_t b = 0; b < nblk; b++) {
                vfl E0 = vec_splats(0.0f), E1 = vec_splats(0.0f);
                for (int c = 0; c < 4; c++) {
                    const int64_t ch = 4*b + c;
                    const block_q8_0 * yb[NR];
                    float dB[NR], S[NR];
                    for (int j = 0; j < NR; j++) {
                        int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                        yb[j] = &B[jj*ldb + 4*(blk0 + b) + c];
                        dB[j] = fp16_to_fp32(yb[j]->d);
                    }
                    vui rows4[4];
                    for (int a = 0; a < 2; a++) {
                        vuc q[4][2];
                        for (int j = 0; j < 4; j++) {
                            q[j][0] = load16u((const uint8_t *)(yb[4*a + j]->qs) + (0));
                            q[j][1] = load16u((const uint8_t *)(yb[4*a + j]->qs) + (16));
                            vsi z = vec_splats(0);
                            vsi sm = vec_sum4s((vsc)q[j][0], z);
                            sm = vec_sum4s((vsc)q[j][1], sm);
                            S[4*a + j] = (float)(sm[0] + sm[1] + sm[2] + sm[3]);
                        }
                        for (int h = 0; h < 2; h++) {
                            for (int j = 0; j < 4; j++) rows4[j] = (vui)q[j][h];
                            mma_transpose4(rows4, &T->v[ch][8*h + a], 2);
                        }
                    }
                    vfl dB0 = (vfl){ dB[0], dB[1], dB[2], dB[3] };
                    vfl dB1 = (vfl){ dB[4], dB[5], dB[6], dB[7] };
                    T->dB[ch][0] = dB0; T->dB[ch][1] = dB1;
                    E0 = vec_madd(dB0, (vfl){ S[0],S[1],S[2],S[3] }, E0);
                    E1 = vec_madd(dB1, (vfl){ S[4],S[5],S[6],S[7] }, E1);
                }
                T->E[b][0] = E0; T->E[b][1] = E1;
            }
        }
    }
}

// ---------------- GEMM from packed operands ----------------

static void kernel_16x8(const apack_t * PA, const bpack_t * PB,
                        int64_t nblk, float alpha, vfl fin[NR][4]) {
    const vfl valpha = vec_splats(alpha);
    for (int64_t b = 0; b < nblk; b++) {
        for (int c = 0; c < 4; c++) {
            const int64_t ch = 4*b + c;
            const vuc * a = PA->v[ch];
            const vuc * y = PB->v[ch];
            if (ch + 1 < 4*nblk) {
                __builtin_prefetch(PA->v[ch + 1], 0, 3);
                __builtin_prefetch((const char *)PA->v[ch + 1] + 256, 0, 3);
                __builtin_prefetch(PB->v[ch + 1], 0, 3);
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
            const vfl dBa0 = vec_mul(PB->dB[ch][0], valpha);
            const vfl dBa1 = vec_mul(PB->dB[ch][1], valpha);
            for (int i = 0; i < 2; i++) {
                const vfl dBa = i ? dBa1 : dBa0;
                for (int g = 0; g < 4; g++) {
                    vsi rowsP[4];
                    __builtin_mma_disassemble_acc(rowsP, &acc[i][g]);
                    const vfl s0 = vec_mul(vec_splat(dBa, 0), PA->dA[b][g]);
                    const vfl s1 = vec_mul(vec_splat(dBa, 1), PA->dA[b][g]);
                    const vfl s2 = vec_mul(vec_splat(dBa, 2), PA->dA[b][g]);
                    const vfl s3 = vec_mul(vec_splat(dBa, 3), PA->dA[b][g]);
                    fin[4*i + 0][g] = vec_madd(vec_ctf(rowsP[0], 0), s0, fin[4*i + 0][g]);
                    fin[4*i + 1][g] = vec_madd(vec_ctf(rowsP[1], 0), s1, fin[4*i + 1][g]);
                    fin[4*i + 2][g] = vec_madd(vec_ctf(rowsP[2], 0), s2, fin[4*i + 2][g]);
                    fin[4*i + 3][g] = vec_madd(vec_ctf(rowsP[3], 0), s3, fin[4*i + 3][g]);
                }
            }
        }
    }
    for (int64_t b = 0; b < nblk; b++)
        for (int g = 0; g < 4; g++) {
            const vfl dA = PA->dA[b][g];
            fin[0][g] = vec_nmsub(dA, vec_splat(PB->E[b][0], 0), fin[0][g]);
            fin[1][g] = vec_nmsub(dA, vec_splat(PB->E[b][0], 1), fin[1][g]);
            fin[2][g] = vec_nmsub(dA, vec_splat(PB->E[b][0], 2), fin[2][g]);
            fin[3][g] = vec_nmsub(dA, vec_splat(PB->E[b][0], 3), fin[3][g]);
            fin[4][g] = vec_nmsub(dA, vec_splat(PB->E[b][1], 0), fin[4][g]);
            fin[5][g] = vec_nmsub(dA, vec_splat(PB->E[b][1], 1), fin[5][g]);
            fin[6][g] = vec_nmsub(dA, vec_splat(PB->E[b][1], 2), fin[6][g]);
            fin[7][g] = vec_nmsub(dA, vec_splat(PB->E[b][1], 3), fin[7][g]);
        }
}

// GEMM over pre-packed operands.  packedA from qbit_repack_*,
// packedB from qbit_pack_b.  Threads split row tiles.
extern "C" void qbit_gemm_packed(int64_t m, int64_t n, int64_t k, float alpha,
                                 const void * packedA, const void * packedB,
                                 float * C, int64_t ldc, int ith, int nth) {
    const apack_t * PA = (const apack_t *)packedA;
    const bpack_t * PB = (const bpack_t *)packedB;
    const int64_t kb = k/128, ns = n_slabs(k), mt = n_row_tiles(m), njt = n_col_tiles(n);
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
                const int64_t blk0 = s*KC_BLKS;
                const int64_t nblk = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
                kernel_16x8(&PA[it*ns + s], &PB[jt*ns + s], nblk, alpha, fin);
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

// ---------------- GEMV path (n <= QBIT_GEMV_NMAX), raw weights ----------------

typedef struct { float dB; float S; } gemv_bmeta_t;   // per chunk

static void gemv_prep_b(const block_q8_0 * y, int64_t nch, gemv_bmeta_t * M) {
    for (int64_t c = 0; c < nch; c++) {
        M[c].dB = fp16_to_fp32(y[c].d);
        vsi z = vec_splats(0);
        vsi s = vec_sum4s((vsc)load16u((const uint8_t *)(y[c].qs) + (0)), z);
        s = vec_sum4s((vsc)load16u((const uint8_t *)(y[c].qs) + (16)), s);
        M[c].S = (float)(s[0] + s[1] + s[2] + s[3]);
    }
}

static inline int hsum(vsi s) { return s[0] + s[1] + s[2] + s[3]; }

static float gemv_row_q1(const block_q1_0 * a, const block_q8_0 * y,
                         const gemv_bmeta_t * M, int64_t kb) {
    const vuc rep0 = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc rep1 = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc bitsel = { 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128 };
    float sumf = 0.0f;
    for (int64_t b = 0; b < kb; b++) {
        const float dA = fp16_to_fp32(a[b].d);
        vuc raw = load16u((const uint8_t *)(a[b].qs) + (0));
        for (int c = 0; c < 4; c++) {
            const vuc off = vec_splats((unsigned char)(4*c));
            vuc e0 = vec_perm(raw, raw, vec_add(rep0, off));
            vuc e1 = vec_perm(raw, raw, vec_add(rep1, off));
            vuc m0 = (vuc)vec_cmpeq(vec_and(e0, bitsel), bitsel);
            vuc m1 = (vuc)vec_cmpeq(vec_and(e1, bitsel), bitsel);
            const int8_t * q = y[4*b + c].qs;
            vuc y0 = load16u((const uint8_t *)(q) + (0));
            vuc y1 = load16u((const uint8_t *)(q) + (16));
            vsi z = vec_splats(0);
            vsi p = vec_sum4s((vsc)vec_and(y0, m0), z);
            p = vec_sum4s((vsc)vec_and(y1, m1), p);
            const gemv_bmeta_t * mm = &M[4*b + c];
            sumf += dA * mm->dB * (2.0f*(float)hsum(p) - mm->S);
        }
    }
    return sumf;
}

static float gemv_row_q2(const block_q2_0 * a, const block_q8_0 * y,
                         const gemv_bmeta_t * M, int64_t kb) {
    const vuc rep0 = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
    const vuc rep1 = { 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7 };
    const vuc sel0 = { 1,4,16,64, 1,4,16,64, 1,4,16,64, 1,4,16,64 };
    const vuc sel1 = { 2,8,32,128, 2,8,32,128, 2,8,32,128, 2,8,32,128 };
    float sumf = 0.0f;
    for (int64_t b = 0; b < kb; b++) {
        const float dA = fp16_to_fp32(a[b].d);
        vuc lo = load16u((const uint8_t *)(a[b].qs) + (0));
        vuc hi = load16u((const uint8_t *)(a[b].qs) + (16));
        for (int c = 0; c < 4; c++) {
            const vuc off = vec_splats((unsigned char)(8*c));
            vuc e0 = vec_perm(lo, hi, vec_add(rep0, off));
            vuc e1 = vec_perm(lo, hi, vec_add(rep1, off));
            vuc ma0 = (vuc)vec_cmpeq(vec_and(e0, sel0), sel0);
            vuc mb0 = (vuc)vec_cmpeq(vec_and(e0, sel1), sel1);
            vuc ma1 = (vuc)vec_cmpeq(vec_and(e1, sel0), sel0);
            vuc mb1 = (vuc)vec_cmpeq(vec_and(e1, sel1), sel1);
            const int8_t * q = y[4*b + c].qs;
            vuc y0 = load16u((const uint8_t *)(q) + (0));
            vuc y1 = load16u((const uint8_t *)(q) + (16));
            vsi z = vec_splats(0);
            vsi p0 = vec_sum4s((vsc)vec_and(y0, ma0), z);
            p0 = vec_sum4s((vsc)vec_and(y1, ma1), p0);
            vsi p1 = vec_sum4s((vsc)vec_and(y0, mb0), z);
            p1 = vec_sum4s((vsc)vec_and(y1, mb1), p1);
            const gemv_bmeta_t * mm = &M[4*b + c];
            sumf += dA * mm->dB * ((float)hsum(p0) + 2.0f*(float)hsum(p1) - mm->S);
        }
    }
    return sumf;
}

extern "C" void qbit_gemv_q1(int64_t m, int64_t n, int64_t k,
        const block_q1_0 * A, int64_t lda, const block_q8_0 * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    const int64_t kb = k/128;
    gemv_bmeta_t * M = (gemv_bmeta_t *)malloc(sizeof(gemv_bmeta_t)*4*kb);
    const int64_t rpt = (m + nth - 1)/nth, i0 = ith*rpt, i1 = (ith+1)*rpt < m ? (ith+1)*rpt : m;
    for (int64_t j = 0; j < n; j++) {
        gemv_prep_b(B + j*ldb, 4*kb, M);
        for (int64_t i = i0; i < i1; i++)
            C[i + j*ldc] = gemv_row_q1(A + i*lda, B + j*ldb, M, kb);
    }
    free(M);
}

extern "C" void qbit_gemv_q2(int64_t m, int64_t n, int64_t k,
        const block_q2_0 * A, int64_t lda, const block_q8_0 * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    const int64_t kb = k/128;
    gemv_bmeta_t * M = (gemv_bmeta_t *)malloc(sizeof(gemv_bmeta_t)*4*kb);
    const int64_t rpt = (m + nth - 1)/nth, i0 = ith*rpt, i1 = (ith+1)*rpt < m ? (ith+1)*rpt : m;
    for (int64_t j = 0; j < n; j++) {
        gemv_prep_b(B + j*ldb, 4*kb, M);
        for (int64_t i = i0; i < i1; i++)
            C[i + j*ldc] = gemv_row_q2(A + i*lda, B + j*ldb, M, kb);
    }
    free(M);
}

#endif // __MMA__

// ---------------------------------------------------------------------------
#ifdef QBIT4_TEST
#include <cstdio>
#include <cmath>

static uint32_t rng = 0x2545f491;
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

static double dref_q1(int64_t k, const block_q1_0 * x, const block_q8_0 * y) {
    double acc = 0;
    for (int64_t bi = 0; bi < k/128; bi++) {
        double d0 = fp16_to_fp32(x[bi].d);
        for (int c = 0; c < 4; c++) {
            const block_q8_0 * yb = &y[4*bi + c];
            const uint8_t * qs = &x[bi].qs[4*c];
            long s = 0;
            for (int j = 0; j < 32; j++)
                s += (2*((qs[j/8] >> (j%8)) & 1) - 1) * yb->qs[j];
            acc += d0 * (double)fp16_to_fp32(yb->d) * (double)s;
        }
    }
    return acc;
}
static double dref_q2(int64_t k, const block_q2_0 * x, const block_q8_0 * y) {
    double acc = 0;
    for (int64_t bi = 0; bi < k/128; bi++) {
        double d0 = fp16_to_fp32(x[bi].d);
        for (int c = 0; c < 4; c++) {
            const block_q8_0 * yb = &y[4*bi + c];
            const uint8_t * qs = &x[bi].qs[8*c];
            long s = 0;
            for (int b = 0; b < 8; b++)
                for (int t = 0; t < 4; t++)
                    s += (((qs[b] >> (2*t)) & 3) - 1) * yb->qs[4*b + t];
            acc += d0 * (double)fp16_to_fp32(yb->d) * (double)s;
        }
    }
    return acc;
}

int main() {
    int fails = 0;
    struct { int64_t m, n, k; } cases[] = {
        { 16, 8, 128 }, { 32, 16, 512 }, { 13, 7, 384 },
        { 64, 16, 1152 }, { 40, 24, 4096 }, { 17, 9, 2048 },
        { 32, 1, 1024 }, { 9, 2, 2176 },   // GEMV shapes
    };
    for (int type = 0; type < 2; type++) {
        for (auto & tc : cases) {
            const int64_t m = tc.m, n = tc.n, k = tc.k;
            const int64_t lda = k/128, ldb = k/QK8_0, ldc = m;
            block_q1_0 * A1 = nullptr; block_q2_0 * A2 = nullptr;
            if (type == 0) {
                A1 = (block_q1_0*)aligned_alloc(64, m*lda*sizeof(block_q1_0));
                for (int64_t i = 0; i < m*lda; i++) {
                    A1[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
                    for (int b = 0; b < 16; b++) A1[i].qs[b] = (uint8_t)(xr() & 0xff);
                }
            } else {
                A2 = (block_q2_0*)aligned_alloc(64, m*lda*sizeof(block_q2_0));
                for (int64_t i = 0; i < m*lda; i++) {
                    A2[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
                    for (int b = 0; b < 32; b++) {
                        uint8_t byte = 0;
                        for (int s = 0; s < 4; s++) byte |= (uint8_t)(xr() % 4) << (2*s);
                        A2[i].qs[b] = byte;
                    }
                }
            }
            block_q8_0 * Bm = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
            for (int64_t i = 0; i < n*ldb; i++) {
                Bm[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
                for (int b = 0; b < QK8_0; b++) Bm[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
            }
            float * Cg = (float*)aligned_alloc(64, m*n*sizeof(float));
            float * Cv = (float*)aligned_alloc(64, m*n*sizeof(float));

            // packed GEMM path
            void * PA = aligned_alloc(64, qbit_apack_size(m, k));
            void * PB = aligned_alloc(64, qbit_bpack_size(n, k));
            if (type == 0) qbit_repack_q1(A1, lda, m, k, PA);
            else           qbit_repack_q2(A2, lda, m, k, PA);
            qbit_pack_b(Bm, ldb, n, k, PB);
            qbit_gemm_packed(m, n, k, type == 0 ? 2.0f : 1.0f, PA, PB, Cg, ldc, 0, 2);
            qbit_gemm_packed(m, n, k, type == 0 ? 2.0f : 1.0f, PA, PB, Cg, ldc, 1, 2);

            // GEMV path
            if (type == 0) { qbit_gemv_q1(m, n, k, A1, lda, Bm, ldb, Cv, ldc, 0, 2);
                             qbit_gemv_q1(m, n, k, A1, lda, Bm, ldb, Cv, ldc, 1, 2); }
            else           { qbit_gemv_q2(m, n, k, A2, lda, Bm, ldb, Cv, ldc, 0, 2);
                             qbit_gemv_q2(m, n, k, A2, lda, Bm, ldb, Cv, ldc, 1, 2); }

            double eg = 0, ev = 0, scale = 0;
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++) {
                    double ref = type == 0 ? dref_q1(k, A1 + i*lda, Bm + j*ldb)
                                           : dref_q2(k, A2 + i*lda, Bm + j*ldb);
                    scale += fabs(ref);
                    double a = fabs((double)Cg[i + j*ldc] - ref); if (a > eg) eg = a;
                    double b = fabs((double)Cv[i + j*ldc] - ref); if (b > ev) ev = b;
                }
            scale = scale/(m*n) + 1e-30;
            eg /= scale; ev /= scale;
            bool ok = eg < 5e-6 && ev < 5e-6;
            printf("%s m=%3lld n=%3lld k=%5lld  gemm=%.3g gemv=%.3g  %s\n",
                   type == 0 ? "q1" : "q2",
                   (long long)m, (long long)n, (long long)k, eg, ev, ok ? "OK":"FAIL");
            if (!ok) fails++;
            free(A1); free(A2); free(Bm); free(Cg); free(Cv); free(PA); free(PB);
        }
    }
    printf(fails ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fails ? 1 : 0;
}
#endif

#ifdef QBIT4_BENCH
#include <cstdio>
#include <ctime>
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
                      return t.tv_sec + 1e-9*t.tv_nsec; }
int main() {
    // token-generation shape: n = 1
    const int64_t m = 4096, n = 1, k = 2048, iters = 8;
    const int64_t lda = k/128, ldb = k/QK8_0, ldc = m;
    block_q1_0 * A = (block_q1_0*)aligned_alloc(64, m*lda*sizeof(block_q1_0));
    block_q8_0 * B = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
    float * C = (float*)aligned_alloc(64, m*n*sizeof(float));
    memset(A, 0x5a, m*lda*sizeof(block_q1_0));
    memset(B, 0x11, n*ldb*sizeof(block_q8_0));

    void * PA = aligned_alloc(64, qbit_apack_size(m, k));
    void * PB = aligned_alloc(64, qbit_bpack_size(n, k));
    double t0 = now();
    qbit_repack_q1(A, lda, m, k, PA);
    double t_repack = now() - t0;

    t0 = now();
    for (int64_t it = 0; it < iters; it++) {
        qbit_pack_b(B, ldb, n, k, PB);
        qbit_gemm_packed(m, n, k, 2.0f, PA, PB, C, ldc, 0, 1);
    }
    double t_gemm = (now() - t0)/iters;

    t0 = now();
    for (int64_t it = 0; it < iters; it++)
        qbit_gemv_q1(m, n, k, A, lda, B, ldb, C, ldc, 0, 1);
    double t_gemv = (now() - t0)/iters;

    printf("n=1 m=%lld k=%lld: repack(once)=%.3fs  gemm/iter=%.3fs  gemv/iter=%.3fs\n",
           (long long)m, (long long)k, t_repack, t_gemm, t_gemv);
    printf("(qemu = instruction-count proxy; GEMV is bandwidth-bound on hardware,\n"
           " where its 8x smaller weight traffic matters far more than shown here)\n");
    return 0;
}
#endif
