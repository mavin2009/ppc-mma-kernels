// q1_0_ppc_mma_v2.cpp
//
// Optimized POWER10 MMA GEMM kernel for PrismML 1-bit weights (Q1_0)
// x Q8_0 activations. Changes vs v1 (q1_0_ppc_mma.cpp):
//
//   1. Packing hoisted out of the compute loop, GEMM-style, with K
//      blocked into slabs (KC): A is bit-unpacked to int8 in MMA layout
//      once per (thread, slab); B is flipped/transposed once per slab.
//      v1 redid the A unpack for every column tile and the B transpose
//      for every row tile.
//   2. 8x8 microkernel on 4 of the 8 MMA accumulators: A and B vector
//      loads amortized over 4x the outputs.
//   3. FP16 scales converted once at pack time (dB pre-broadcast as
//      vector float per column group), not per inner iteration.
//   4. Per-row ternary chunk sums (bias correction) computed during the
//      A pack, for free.
//
// Same format and conventions as v1; k must be a multiple of 128.
// In a real ggml integration the A pack belongs in repack.cpp (done once
// at model load: Q1_0 -> "Q1_0x8" layout) and the B pack belongs next to
// the activation quantization; here both live inside the kernel with
// per-thread buffers, which duplicates the B pack across threads.
//
// Build (cross):
//   powerpc64le-linux-gnu-g++ -O3 -mcpu=power10 -DQ1MMA_TEST \
//       q1_0_ppc_mma_v2.cpp -o q1mma_v2_test
//   qemu-ppc64le -cpu power10 -L /usr/powerpc64le-linux-gnu ./q1mma_v2_test

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define QK1_0 128
#define QK8_0 32

typedef uint16_t ggml_half;

typedef struct { ggml_half d; uint8_t qs[QK1_0 / 8]; } block_q1_0;
typedef struct { ggml_half d; int8_t  qs[QK8_0];     } block_q8_0;

static_assert(sizeof(block_q1_0) == sizeof(ggml_half) + QK1_0 / 8, "bad q1_0");
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0,     "bad q8_0");

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

// K slab: 16 Q1_0 blocks = 2048 elements = 64 q8 chunks.
#define KC_BLKS   16
#define KC        (KC_BLKS * QK1_0)
#define KC_CHUNKS (KC / QK8_0)

// ---- packed A: per 8-row group, per chunk: 16 vuc (8 depth-steps x lo/hi),
//      plus per-row csum (int) and per-row per-q1-block scale (float).
typedef struct {
    vuc   v[KC_CHUNKS][16];      // MMA-layout signed weight bytes
    float corr[KC_CHUNKS][8];    // -128 * chunk sum, pre-multiplied
    float dA[KC_BLKS][8];        // per-row scale per q1 block
} apack_t;

// ---- packed B: per 8-col group, per chunk: 16 vuc + dB lo/hi vfl.
typedef struct {
    vuc v[KC_CHUNKS][16];
    vfl dB[KC_CHUNKS][2];        // scales for cols 0-3 / 4-7
} bpack_t;

// Unpack chunk c (32 elements) of a Q1_0 block from its preloaded 16 qs
// bytes.  Loading the block's qs exactly once keeps the read inside the
// 18-byte struct (a straight vec_xl at qs+4*c would read past it on the
// last chunk).
static inline void q1_unpack32_c(vuc raw, int c, vsc v[2]) {
    const vuc rep0 = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc rep1 = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc sh   = { 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7 };
    const vuc m1   = vec_splats((unsigned char)1);
    const vsc one  = vec_splats((signed char)1);
    const vuc off  = vec_splats((unsigned char)(4*c));
    vuc b0 = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep0, off)), sh), m1);
    vuc b1 = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep1, off)), sh), m1);
    v[0] = vec_sub((vsc)vec_add(b0, b0), one);
    v[1] = vec_sub((vsc)vec_add(b1, b1), one);
}

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

// Pack one 8-row slab of A (kc elements starting at block blk0).
static void pack_A8(const block_q1_0 * A, int64_t lda, int64_t rows,
                    int64_t blk0, int64_t nblk, apack_t * P) {
    for (int64_t b = 0; b < nblk; b++) {
        const block_q1_0 * blkp[8];
        vuc raw[8];
        for (int r = 0; r < 8; r++) {
            int64_t rr = r < rows ? r : rows - 1;      // clamp (dup last row)
            blkp[r] = &A[rr*lda + blk0 + b];
            P->dA[b][r] = fp16_to_fp32(blkp[r]->d);
            raw[r] = vec_xl(0, (const unsigned char *)blkp[r]->qs);
        }
        for (int c = 0; c < 4; c++) {
            const int ch = 4*b + c;
            vsc t[8][2];
            for (int r = 0; r < 8; r++) {
                q1_unpack32_c(raw[r], c, t[r]);
                // sum of 32 values in {-1,+1} = 2*popcount - 32,
                // correction = -128*sum = 4096 - 256*popcount
                uint32_t w; memcpy(&w, blkp[r]->qs + 4*c, 4);
                P->corr[ch][r] = (float)(4096 - 256*(int)__builtin_popcount(w));
            }
            vui rows4[4];
            for (int g = 0; g < 2; g++)                // row groups 0-3 / 4-7
                for (int h = 0; h < 2; h++) {          // depth halves
                    for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                    mma_transpose4(rows4, &P->v[ch][8*h + g], 2);
                }
        }
    }
}

// Pack one 8-col slab of B.
static void pack_B8(const block_q8_0 * B, int64_t ldb, int64_t cols,
                    int64_t q8blk0, int64_t nchunk, bpack_t * P) {
    const vuc flip = vec_splats((unsigned char)0x80);
    for (int64_t ch = 0; ch < nchunk; ch++) {
        const block_q8_0 * yb[8];
        float dB[8];
        for (int j = 0; j < 8; j++) {
            int64_t jj = j < cols ? j : cols - 1;
            yb[j] = &B[jj*ldb + q8blk0 + ch];
            dB[j] = fp16_to_fp32(yb[j]->d);
        }
        P->dB[ch][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
        P->dB[ch][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
        vui rows4[4];
        for (int g = 0; g < 2; g++)
            for (int h = 0; h < 2; h++) {
                for (int j = 0; j < 4; j++)
                    rows4[j] = (vui)vec_xor(
                        vec_xl(16*h, (const unsigned char *)yb[4*g + j]->qs), flip);
                mma_transpose4(rows4, &P->v[ch][8*h + g], 2);
            }
    }
}

// 8x8 microkernel over one packed slab; accumulates into fin[8][2].
static inline void kernel_8x8(const apack_t * PA, const bpack_t * PB,
                              int64_t nchunk, vfl fin[8][2]) {
    for (int64_t ch = 0; ch < nchunk; ch++) {
        const vuc * a = PA->v[ch];
        const vuc * b = PB->v[ch];
        __vector_quad acc00, acc01, acc10, acc11;
        __builtin_mma_xxsetaccz(&acc00);
        __builtin_mma_xxsetaccz(&acc01);
        __builtin_mma_xxsetaccz(&acc10);
        __builtin_mma_xxsetaccz(&acc11);
        for (int x = 0; x < 8; x++) {
            const vuc alo = a[2*x], ahi = a[2*x + 1];
            const vuc blo = b[2*x], bhi = b[2*x + 1];
            __builtin_mma_xvi8ger4pp(&acc00, alo, blo);
            __builtin_mma_xvi8ger4pp(&acc01, alo, bhi);
            __builtin_mma_xvi8ger4pp(&acc10, ahi, blo);
            __builtin_mma_xvi8ger4pp(&acc11, ahi, bhi);
        }
        const float * corr = PA->corr[ch];
        const float * dA   = PA->dA[ch >> 2];
        const vfl dBlo = PB->dB[ch][0], dBhi = PB->dB[ch][1];
        vsi c00[4], c01[4], c10[4], c11[4];
        __builtin_mma_disassemble_acc(c00, &acc00);
        __builtin_mma_disassemble_acc(c01, &acc01);
        __builtin_mma_disassemble_acc(c10, &acc10);
        __builtin_mma_disassemble_acc(c11, &acc11);
        for (int i = 0; i < 4; i++) {
            const vfl crl = vec_splats(corr[i]),     crh = vec_splats(corr[i+4]);
            const vfl sl  = vec_splats(dA[i]),       sh2 = vec_splats(dA[i+4]);
            fin[i][0]   = vec_madd(vec_add(vec_ctf(c00[i],0), crl), vec_mul(sl,  dBlo), fin[i][0]);
            fin[i][1]   = vec_madd(vec_add(vec_ctf(c01[i],0), crl), vec_mul(sl,  dBhi), fin[i][1]);
            fin[i+4][0] = vec_madd(vec_add(vec_ctf(c10[i],0), crh), vec_mul(sh2, dBlo), fin[i+4][0]);
            fin[i+4][1] = vec_madd(vec_add(vec_ctf(c11[i],0), crh), vec_mul(sh2, dBhi), fin[i+4][1]);
        }
    }
}

#endif // __MMA__

// Scalar reference dot (matches dequantize_row_q1_0 semantics).
static float q1_q8_dot_ref(int64_t k, const block_q1_0 * x, const block_q8_0 * y) {
    const int64_t nb = k / QK1_0;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const float d0 = fp16_to_fp32(x[i].d);
        float sumi = 0.0f;
        for (int c = 0; c < 4; c++) {
            const block_q8_0 * yb = &y[4*i + c];
            const uint8_t * qs = &x[i].qs[4*c];
            int s = 0;
            for (int j = 0; j < 32; j++) {
                const int bit = (qs[j / 8] >> (j % 8)) & 1;
                s += (2*bit - 1) * yb->qs[j];
            }
            sumi += fp16_to_fp32(yb->d) * s;
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

extern "C" void gemm_q1_0_q8_0_ppc_v2(int64_t m, int64_t n, int64_t k,
                                      const block_q1_0 * A, int64_t lda,
                                      const block_q8_0 * B, int64_t ldb,
                                      float * C, int64_t ldc,
                                      int ith, int nth) {
    assert(k % QK1_0 == 0);

    const int64_t mt  = (m + 7) / 8;
    const int64_t tpt = (mt + nth - 1) / nth;
    const int64_t i0  = 8 * (ith * tpt);
    const int64_t i1  = m < 8 * ((ith + 1) * tpt) ? m : 8 * ((ith + 1) * tpt);
    if (i0 >= m) return;

#if defined(__MMA__)
    const int64_t kb   = k / QK1_0;
    const int64_t njt  = (n + 7) / 8;

    apack_t * PA = (apack_t*)aligned_alloc(64, sizeof(apack_t));
    bpack_t * PB = (bpack_t*)aligned_alloc(64, njt * sizeof(bpack_t));
    vfl (*fin)[8][2] = (vfl(*)[8][2])aligned_alloc(64, njt * sizeof(vfl[8][2]));

    for (int64_t i = i0; i < i1; i += 8) {
        const int64_t rows = (i1 - i) < 8 ? (i1 - i) : 8;
        for (int64_t jt = 0; jt < njt; jt++)
            for (int r = 0; r < 8; r++)
                fin[jt][r][0] = fin[jt][r][1] = vec_splats(0.0f);

        for (int64_t blk0 = 0; blk0 < kb; blk0 += KC_BLKS) {
            const int64_t nblk   = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
            const int64_t nchunk = 4 * nblk;

            pack_A8(A + i*lda, lda, rows, blk0, nblk, PA);
            for (int64_t jt = 0; jt < njt; jt++) {
                const int64_t j    = 8*jt;
                const int64_t cols = (n - j) < 8 ? (n - j) : 8;
                // PB[jt] holds only the current slab: with a single slab it
                // can be packed once (first i tile) and reused; with multiple
                // slabs it must be repacked per (i, slab).
                if (i == i0 || kb > KC_BLKS)
                    pack_B8(B + j*ldb, ldb, cols, 4*blk0, nchunk, &PB[jt]);
                kernel_8x8(PA, &PB[jt], nchunk, fin[jt]);
            }
        }

        for (int64_t jt = 0; jt < njt; jt++) {
            const int64_t j    = 8*jt;
            const int64_t cols = (n - j) < 8 ? (n - j) : 8;
            for (int64_t r = 0; r < rows; r++)
                for (int64_t cjj = 0; cjj < cols; cjj++)
                    C[(i + r) + (j + cjj)*ldc] =
                        cjj < 4 ? fin[jt][r][0][cjj] : fin[jt][r][1][cjj - 4];
        }
    }
    free(PA); free(PB); free(fin);
#else
    for (int64_t i = i0; i < i1; i++)
        for (int64_t j = 0; j < n; j++)
            C[i + j*ldc] = q1_q8_dot_ref(k, A + i*lda, B + j*ldb);
#endif
}

// ---------------------------------------------------------------------------
#ifdef Q1MMA_TEST
#include <cstdio>
#include <cmath>
#include <ctime>

static uint32_t rng = 0x9e3779b9;
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

int main() {
    struct { int64_t m, n, k; } cases[] = {
        { 8,  8, 128 }, { 8, 8, 256 }, { 16, 16, 512 },
        { 13, 7, 384 }, { 32, 1, 1024 }, { 64, 16, 1152 },
        { 24, 24, 4096 },   // multi-slab k
        { 9, 3, 2176 },     // ragged + multi-slab
    };
    int fails = 0;
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        const int64_t lda = k / QK1_0, ldb = k / QK8_0, ldc = m;

        block_q1_0 * A = (block_q1_0*)aligned_alloc(64, m*lda*sizeof(block_q1_0));
        block_q8_0 * Bm = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
        float * C = (float*)aligned_alloc(64, m*n*sizeof(float));

        for (int64_t i = 0; i < m*lda; i++) {
            A[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
            for (int b = 0; b < QK1_0/8; b++) A[i].qs[b] = (uint8_t)(xr() & 0xff);
        }
        for (int64_t i = 0; i < n*ldb; i++) {
            Bm[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
            for (int b = 0; b < QK8_0; b++) Bm[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
        }

        gemm_q1_0_q8_0_ppc_v2(m, n, k, A, lda, Bm, ldb, C, ldc, 0, 2);
        gemm_q1_0_q8_0_ppc_v2(m, n, k, A, lda, Bm, ldb, C, ldc, 1, 2);

        double emma = 0, eref = 0, scale = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                const block_q1_0 * x = A + i*lda;
                const block_q8_0 * y = Bm + j*ldb;
                double acc = 0;
                for (int64_t bi = 0; bi < k/QK1_0; bi++) {
                    double d0 = fp16_to_fp32(x[bi].d);
                    for (int c = 0; c < 4; c++) {
                        const block_q8_0 * yb = &y[4*bi + c];
                        const uint8_t * qs = &x[bi].qs[4*c];
                        long s = 0;
                        for (int jj = 0; jj < 32; jj++)
                            s += (2*((qs[jj/8] >> (jj%8)) & 1) - 1) * yb->qs[jj];
                        acc += d0 * (double)fp16_to_fp32(yb->d) * (double)s;
                    }
                }
                scale += fabs(acc);
                double a = fabs((double)C[i + j*ldc] - acc);
                double b = fabs((double)q1_q8_dot_ref(k, x, y) - acc);
                if (a > emma) emma = a;
                if (b > eref) eref = b;
            }
        scale = scale/(m*n) + 1e-30;
        emma /= scale; eref /= scale;
        bool ok = emma < 4*eref + 1e-6;
        printf("m=%3lld n=%3lld k=%5lld  err(mma)=%.3g err(f32 ref)=%.3g  %s\n",
               (long long)m, (long long)n, (long long)k, emma, eref, ok ? "OK":"FAIL");
        if (!ok) fails++;
        free(A); free(Bm); free(C);
    }
    printf(fails ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fails ? 1 : 0;
}
#endif

#ifdef Q1MMA_BENCH
#include <cstdio>
#include <ctime>
int main() {
    const int64_t m = 512, n = 64, k = 2048, iters = 20;
    const int64_t lda = k/QK1_0, ldb = k/QK8_0, ldc = m;
    block_q1_0 * A = (block_q1_0*)aligned_alloc(64, m*lda*sizeof(block_q1_0));
    block_q8_0 * Bm = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
    float * C = (float*)aligned_alloc(64, m*n*sizeof(float));
    memset(A, 0x5a, m*lda*sizeof(block_q1_0));
    memset(Bm, 0x11, n*ldb*sizeof(block_q8_0));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int64_t it = 0; it < iters; it++)
        gemm_q1_0_q8_0_ppc_v2(m, n, k, A, lda, Bm, ldb, C, ldc, 0, 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);
    printf("v2: %.3f s for %lld iters (m=%lld n=%lld k=%lld), C[0]=%g\n",
           s, (long long)iters, (long long)m, (long long)n, (long long)k, (double)C[0]);
    return 0;
}
#endif
