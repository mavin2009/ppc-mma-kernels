// q1_0_ppc_mma.cpp
//
// POWER10 MMA GEMM kernel for PrismML 1-bit weights (GGUF type Q1_0)
// against Q8_0-quantized activations. Companion to q2_0_ppc_mma.cpp;
// same structure, only the unpack differs.
//
// Format (matches ggml-common.h, PrismML fork / upstream llama.cpp):
//   Q1_0: 128 weights per block, one FP16 scale.
//         1 bit per weight, LSB-first: value = 2*bit - 1  ->  {-1, +1}
//   Q8_0: 32 int8 values per block, one FP16 scale.
//   One Q1_0 block spans exactly 4 Q8_0 blocks along K.
//
// xvi8ger4pp(acc, va, vb): va SIGNED int8 (4 rows x 4 depth bytes,
// row-major), vb UNSIGNED (verified empirically under qemu -cpu power10).
// Weights t in {-1,+1} go on the signed operand; activations are XORed
// with 0x80 onto the unsigned operand; bias corrected with
// -128 * (per-row chunk sum of t), then scaled by d_A*d_B per q8 block.
//
// Layout conventions follow llamafile sgemm (see q2_0_ppc_mma.cpp).
// k must be a multiple of 128 (QK1_0).
//
// Build (cross):
//   powerpc64le-linux-gnu-g++ -O3 -mcpu=power10 -DQ1MMA_TEST \
//       q1_0_ppc_mma.cpp -o q1mma_test
//   qemu-ppc64le -cpu power10 -L /usr/powerpc64le-linux-gnu ./q1mma_test

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstring>

#define QK1_0 128
#define QK8_0 32

typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[QK1_0 / 8];  // 1 bit per element, LSB-first
} block_q1_0;

typedef struct {
    ggml_half d;              // scale
    int8_t    qs[QK8_0];
} block_q8_0;

static_assert(sizeof(block_q1_0) == sizeof(ggml_half) + QK1_0 / 8, "bad q1_0");
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0,     "bad q8_0");

static inline float fp16_to_fp32(ggml_half h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

#if defined(__MMA__)

typedef vector unsigned char  vuc;
typedef vector signed char    vsc;
typedef vector unsigned int   vui;
typedef vector signed int     vsi;
typedef vector float          vfl;

// Unpack chunk c (32 elements) of a Q1_0 block from its preloaded 16 qs
// bytes: v[0] holds elements 0..15, v[1] 16..31.  The block's qs is
// loaded once (a vec_xl at qs+4*c would read past the 18-byte struct on
// the last chunk).
static inline void q1_unpack32_c(vuc raw, int c, vsc v[2]) {
    const vuc rep0 = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc rep1 = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc sh   = { 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7 };
    const vuc m1   = vec_splats((unsigned char)1);
    const vsc one  = vec_splats((signed char)1);
    const vuc off  = vec_splats((unsigned char)(4*c));
    vuc b0 = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep0, off)), sh), m1);
    vuc b1 = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep1, off)), sh), m1);
    v[0] = vec_sub((vsc)vec_add(b0, b0), one);   // 2*bit - 1
    v[1] = vec_sub((vsc)vec_add(b1, b1), one);
}

static inline int q1_chunk_sum(const vsc v[2]) {
    vsi z = vec_splats(0);
    vsi s = vec_sum4s(v[0], z);
    s = vec_sum4s(v[1], s);
    return s[0] + s[1] + s[2] + s[3];
}

static inline void mma_transpose4(const vui rows[4], vuc out[4]) {
    vui t0 = vec_mergeh(rows[0], rows[1]);
    vui t1 = vec_mergel(rows[0], rows[1]);
    vui t2 = vec_mergeh(rows[2], rows[3]);
    vui t3 = vec_mergel(rows[2], rows[3]);
    out[0] = (vuc)vec_xxpermdi(t0, t2, 0);
    out[1] = (vuc)vec_xxpermdi(t0, t2, 3);
    out[2] = (vuc)vec_xxpermdi(t1, t3, 0);
    out[3] = (vuc)vec_xxpermdi(t1, t3, 3);
}

static void q1mma_tile4x4(int64_t k,
                          const block_q1_0 * a0, const block_q1_0 * a1,
                          const block_q1_0 * a2, const block_q1_0 * a3,
                          const block_q8_0 * b0, const block_q8_0 * b1,
                          const block_q8_0 * b2, const block_q8_0 * b3,
                          float * C, int64_t ldc, int64_t ii, int64_t jj) {
    const block_q1_0 * A[4] = { a0, a1, a2, a3 };
    const block_q8_0 * B[4] = { b0, b1, b2, b3 };
    const int64_t nblk = k / QK1_0;
    const vuc flip = vec_splats((unsigned char)0x80);

    vfl fin[4] = { vec_splats(0.0f), vec_splats(0.0f),
                   vec_splats(0.0f), vec_splats(0.0f) };

    for (int64_t blk = 0; blk < nblk; blk++) {
        float dA[4];
        for (int i = 0; i < 4; i++) dA[i] = fp16_to_fp32(A[i][blk].d);

        vuc rawA[4];
        for (int i = 0; i < 4; i++)
            rawA[i] = vec_xl(0, (const unsigned char *)A[i][blk].qs);

        for (int c = 0; c < 4; c++) {          // 4 q8 sub-chunks per q1 block
            vsc  ta[4][2];
            int  csum[4];
            for (int i = 0; i < 4; i++) {
                q1_unpack32_c(rawA[i], c, ta[i]);
                csum[i] = q1_chunk_sum(ta[i]);
            }
            vuc vecA[8];
            {
                vui rows[4];
                for (int i = 0; i < 4; i++) rows[i] = (vui)ta[i][0];
                mma_transpose4(rows, vecA);
                for (int i = 0; i < 4; i++) rows[i] = (vui)ta[i][1];
                mma_transpose4(rows, vecA + 4);
            }

            const block_q8_0 * yb[4];
            vuc vecB[8];
            {
                vui rows[4];
                vuc lo[4], hi[4];
                for (int j = 0; j < 4; j++) {
                    yb[j] = &B[j][4*blk + c];
                    lo[j] = vec_xor(vec_xl(0,  (const unsigned char *)yb[j]->qs), flip);
                    hi[j] = vec_xor(vec_xl(16, (const unsigned char *)yb[j]->qs), flip);
                }
                for (int j = 0; j < 4; j++) rows[j] = (vui)lo[j];
                mma_transpose4(rows, vecB);
                for (int j = 0; j < 4; j++) rows[j] = (vui)hi[j];
                mma_transpose4(rows, vecB + 4);
            }

            __vector_quad acc;
            __builtin_mma_xxsetaccz(&acc);
            for (int x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(&acc, vecA[x], vecB[x]);
            }

            vfl dB = { fp16_to_fp32(yb[0]->d), fp16_to_fp32(yb[1]->d),
                       fp16_to_fp32(yb[2]->d), fp16_to_fp32(yb[3]->d) };
            vsi rawC[4];
            __builtin_mma_disassemble_acc(rawC, &acc);
            for (int i = 0; i < 4; i++) {
                vfl corr = vec_splats((float)(-128 * csum[i]));
                vfl res  = vec_add(vec_ctf(rawC[i], 0), corr);
                vfl vs   = vec_mul(vec_splats(dA[i]), dB);
                fin[i]   = vec_madd(res, vs, fin[i]);
            }
        }
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            C[(ii + i) + (jj + j)*ldc] = fin[i][j];
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

extern "C" void gemm_q1_0_q8_0_ppc(int64_t m, int64_t n, int64_t k,
                                   const block_q1_0 * A, int64_t lda,
                                   const block_q8_0 * B, int64_t ldb,
                                   float * C, int64_t ldc,
                                   int ith, int nth) {
    assert(k % QK1_0 == 0);

    const int64_t mt   = (m + 3) / 4;
    const int64_t tpt  = (mt + nth - 1) / nth;
    const int64_t i0   = 4 * (ith * tpt);
    const int64_t i1   = m < 4 * ((ith + 1) * tpt) ? m : 4 * ((ith + 1) * tpt);
    if (i0 >= m) return;

    int64_t i = i0;
#if defined(__MMA__)
    for (; i + 4 <= i1; i += 4) {
        int64_t j = 0;
        for (; j + 4 <= n; j += 4) {
            q1mma_tile4x4(k,
                A + (i+0)*lda, A + (i+1)*lda, A + (i+2)*lda, A + (i+3)*lda,
                B + (j+0)*ldb, B + (j+1)*ldb, B + (j+2)*ldb, B + (j+3)*ldb,
                C, ldc, i, j);
        }
        for (; j < n; j++)
            for (int r = 0; r < 4; r++)
                C[(i+r) + j*ldc] = q1_q8_dot_ref(k, A + (i+r)*lda, B + j*ldb);
    }
#endif
    for (; i < i1; i++)
        for (int64_t j = 0; j < n; j++)
            C[i + j*ldc] = q1_q8_dot_ref(k, A + i*lda, B + j*ldb);
}

// ---------------------------------------------------------------------------
#ifdef Q1MMA_TEST
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
        { 4,  4, 128 }, { 8, 8, 256 }, { 16, 12, 512 },
        { 13, 7, 384 },
        { 32, 1, 1024 },
        { 64, 16, 1152 },
    };
    int fails = 0;
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        const int64_t lda = k / QK1_0, ldb = k / QK8_0, ldc = m;

        block_q1_0 * A = (block_q1_0*)aligned_alloc(64, m*lda*sizeof(block_q1_0));
        block_q8_0 * Bm = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
        float * C    = (float*)aligned_alloc(64, m*n*sizeof(float));
        float * Cref = (float*)aligned_alloc(64, m*n*sizeof(float));

        for (int64_t i = 0; i < m*lda; i++) {
            A[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
            for (int b = 0; b < QK1_0/8; b++) A[i].qs[b] = (uint8_t)(xr() & 0xff);
        }
        for (int64_t i = 0; i < n*ldb; i++) {
            Bm[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
            for (int b = 0; b < QK8_0; b++) Bm[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
        }

        gemm_q1_0_q8_0_ppc(m, n, k, A, lda, Bm, ldb, C, ldc, 0, 2);
        gemm_q1_0_q8_0_ppc(m, n, k, A, lda, Bm, ldb, C, ldc, 1, 2);

        double * Cd = (double*)malloc(m*n*sizeof(double));
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                Cref[i + j*ldc] = q1_q8_dot_ref(k, A + i*lda, Bm + j*ldb);
                const block_q1_0 * x = A + i*lda;
                const block_q8_0 * y = Bm + j*ldb;
                double acc = 0;
                for (int64_t bi = 0; bi < k/QK1_0; bi++) {
                    double d0 = fp16_to_fp32(x[bi].d);
                    for (int c = 0; c < 4; c++) {
                        const block_q8_0 * yb = &y[4*bi + c];
                        const uint8_t * qs = &x[bi].qs[4*c];
                        long s = 0;
                        for (int jj = 0; jj < 32; jj++) {
                            const int bit = (qs[jj / 8] >> (jj % 8)) & 1;
                            s += (2*bit - 1) * yb->qs[jj];
                        }
                        acc += d0 * (double)fp16_to_fp32(yb->d) * (double)s;
                    }
                }
                Cd[i + j*ldc] = acc;
            }

        double scale = 0;
        for (int64_t x = 0; x < m*n; x++) scale += fabs(Cd[x]);
        scale = scale / (m*n) + 1e-30;

        double emma = 0, eref = 0;
        for (int64_t x = 0; x < m*n; x++) {
            double a = fabs((double)C[x]    - Cd[x]);
            double b = fabs((double)Cref[x] - Cd[x]);
            if (a > emma) emma = a;
            if (b > eref) eref = b;
        }
        emma /= scale; eref /= scale;
        bool ok = emma < 4*eref + 1e-6;
        printf("m=%3lld n=%3lld k=%5lld  err(mma)=%.3g err(f32 ref)=%.3g  %s\n",
               (long long)m, (long long)n, (long long)k, emma, eref,
               ok ? "OK" : "FAIL");
        if (!ok) fails++;
        free(A); free(Bm); free(C); free(Cref); free(Cd);
    }
    printf(fails ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fails ? 1 : 0;
}
#endif
