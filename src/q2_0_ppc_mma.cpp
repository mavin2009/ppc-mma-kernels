// q2_0_ppc_mma.cpp
//
// POWER10 MMA GEMM kernel for PrismML ternary weights (GGUF type Q2_0)
// against Q8_0-quantized activations, in the style of llama.cpp's
// tinyBLAS_Q0_PPC (ggml/src/ggml-cpu/llamafile/sgemm.cpp).
//
// Format (matches ggml-common.h in the PrismML fork, `prism` branch):
//   Q2_0: 128 weights per block, one FP16 scale.
//         2-bit codes packed LSB-first, value = (code - 1)  ->  {-1, 0, +1, +2}
//   Q8_0: 32 int8 values per block, one FP16 scale.
//   One Q2_0 block spans exactly 4 Q8_0 blocks along K.
//
// Core idea
// ---------
// xvi8ger4pp(acc, va, vb) computes a 4x4 rank-4 int32 GER update where
// va is treated as SIGNED int8 (4 rows x 4 depth bytes, row-major) and
// vb as UNSIGNED int8 (4 cols x 4 depth bytes).  (Verified empirically
// under qemu -cpu power10.)
//
// We therefore:
//   * unpack ternary codes to signed int8 t = code-1 and place them on
//     the signed (A) operand,
//   * XOR the int8 activations with 0x80 (y + 128, now unsigned) on the
//     B operand,
//   * accumulate 32 depth elements (= one Q8_0 block) with 8 GER updates,
//   * correct the bias:  sum(t*(y+128)) = sum(t*y) + 128*sum(t),
//     so subtract 128 * (per-row ternary chunk sum) -- the same
//     "comparray" trick tinyBLAS_Q0_PPC uses, but with the roles of the
//     correction reversed (weight sums instead of activation sums, and
//     the ternary sums are nearly free to compute),
//   * scale by d_A * d_B per (row, q8-block) and accumulate in float.
//
// Layout conventions follow llamafile sgemm:
//   A: m x k ternary weights, row-major in blocks, lda = row stride in
//      Q2_0 blocks.
//   B: n x k activations, row-major in blocks, ldb = row stride in Q8_0
//      blocks.  (Each "column" of the GEMM is a row of B.)
//   C: column-major float, C[i + j*ldc], ldc >= m.
//   k must be a multiple of 128 (QK2_0).
//
// Build (cross):
//   powerpc64le-linux-gnu-g++ -O3 -mcpu=power10 -DQ2MMA_TEST \
//       q2_0_ppc_mma.cpp -o q2mma_test
//   qemu-ppc64le -cpu power10 -L /usr/powerpc64le-linux-gnu ./q2mma_test

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstring>

#define QK2_0 128
#define QK8_0 32

typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[QK2_0 / 4];  // 2 bits per element, LSB-first
} block_q2_0;

typedef struct {
    ggml_half d;              // scale
    int8_t    qs[QK8_0];
} block_q8_0;

static_assert(sizeof(block_q2_0) == sizeof(ggml_half) + QK2_0 / 4, "bad q2_0");
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0,     "bad q8_0");

static inline float fp16_to_fp32(ggml_half h) {
    // scalar IEEE binary16 -> binary32 (no __fp16 dependency)
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {                       // subnormal
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

// Unpack chunk c (32 elements) of a Q2_0 block from its preloaded 32 qs
// bytes (lo = qs[0..15], hi = qs[16..31]).  Loading the block's qs exactly
// twice keeps reads inside the 34-byte struct (a vec_xl at qs+8*c would
// read past it on the last chunks).  vec_perm indices span both sources.
static inline void q2_unpack32_c(vuc lo, vuc hi, int c, vsc v[2]) {
    const vuc rep0 = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
    const vuc rep1 = { 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7 };
    const vuc sh   = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    const vuc m3   = vec_splats((unsigned char)3);
    const vsc one  = vec_splats((signed char)1);
    const vuc off  = vec_splats((unsigned char)(8*c));
    vuc e0 = vec_perm(lo, hi, vec_add(rep0, off));   // each byte repeated 4x
    vuc e1 = vec_perm(lo, hi, vec_add(rep1, off));
    v[0] = vec_sub((vsc)vec_and(vec_sr(e0, sh), m3), one);
    v[1] = vec_sub((vsc)vec_and(vec_sr(e1, sh), m3), one);
}

// Horizontal sum of 32 signed int8 values held in v[0], v[1].
static inline int q2_chunk_sum(const vsc v[2]) {
    vsi z = vec_splats(0);
    vsi s = vec_sum4s(v[0], z);
    s = vec_sum4s(v[1], s);
    return s[0] + s[1] + s[2] + s[3];
}

// 4x4 word transpose: rows[r] holds 4 consecutive depth words of GEMM
// row/col r; out[x] gets word x of every r, i.e. the byte layout
// xvi8ger4 wants (4 lanes x 4 depth bytes).
static inline void mma_transpose4(const vui rows[4], vuc out[4]) {
    vui t0 = vec_mergeh(rows[0], rows[1]);   // r0w0 r1w0 r0w1 r1w1
    vui t1 = vec_mergel(rows[0], rows[1]);   // r0w2 r1w2 r0w3 r1w3
    vui t2 = vec_mergeh(rows[2], rows[3]);
    vui t3 = vec_mergel(rows[2], rows[3]);
    out[0] = (vuc)vec_xxpermdi(t0, t2, 0);   // r0w0 r1w0 r2w0 r3w0
    out[1] = (vuc)vec_xxpermdi(t0, t2, 3);
    out[2] = (vuc)vec_xxpermdi(t1, t3, 0);
    out[3] = (vuc)vec_xxpermdi(t1, t3, 3);
}

// Process one 4-row x 4-col x k GEMM tile.
static void q2mma_tile4x4(int64_t k,
                          const block_q2_0 * a0, const block_q2_0 * a1,
                          const block_q2_0 * a2, const block_q2_0 * a3,
                          const block_q8_0 * b0, const block_q8_0 * b1,
                          const block_q8_0 * b2, const block_q8_0 * b3,
                          float * C, int64_t ldc, int64_t ii, int64_t jj) {
    const block_q2_0 * A[4] = { a0, a1, a2, a3 };
    const block_q8_0 * B[4] = { b0, b1, b2, b3 };
    const int64_t nblk = k / QK2_0;
    const vuc flip = vec_splats((unsigned char)0x80);

    vfl fin[4] = { vec_splats(0.0f), vec_splats(0.0f),
                   vec_splats(0.0f), vec_splats(0.0f) };

    for (int64_t blk = 0; blk < nblk; blk++) {
        float dA[4];
        for (int i = 0; i < 4; i++) dA[i] = fp16_to_fp32(A[i][blk].d);

        vuc rawLo[4], rawHi[4];
        for (int i = 0; i < 4; i++) {
            rawLo[i] = vec_xl(0,  (const unsigned char *)A[i][blk].qs);
            rawHi[i] = vec_xl(16, (const unsigned char *)A[i][blk].qs);
        }

        for (int c = 0; c < 4; c++) {          // 4 q8 sub-chunks per q2 block
            // ---- pack A: unpack ternary, transpose to MMA layout ----
            vsc  ta[4][2];
            int  csum[4];
            for (int i = 0; i < 4; i++) {
                q2_unpack32_c(rawLo[i], rawHi[i], c, ta[i]);
                csum[i] = q2_chunk_sum(ta[i]);
            }
            vuc vecA[8];
            {
                vui rows[4];
                for (int i = 0; i < 4; i++) rows[i] = (vui)ta[i][0];
                mma_transpose4(rows, vecA);
                for (int i = 0; i < 4; i++) rows[i] = (vui)ta[i][1];
                mma_transpose4(rows, vecA + 4);
            }

            // ---- pack B: flip to unsigned, transpose ----
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

            // ---- 32-deep integer GER ----
            __vector_quad acc;
            __builtin_mma_xxsetaccz(&acc);
            for (int x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(&acc, vecA[x], vecB[x]);
            }

            // ---- correct bias, scale, accumulate ----
            vfl dB = { fp16_to_fp32(yb[0]->d), fp16_to_fp32(yb[1]->d),
                       fp16_to_fp32(yb[2]->d), fp16_to_fp32(yb[3]->d) };
            vsi rawC[4];
            __builtin_mma_disassemble_acc(rawC, &acc);
            for (int i = 0; i < 4; i++) {
                vfl corr = vec_splats((float)(-128 * csum[i]));
                vfl res  = vec_add(vec_ctf(rawC[i], 0), corr);   // int dot t*y
                vfl vs   = vec_mul(vec_splats(dA[i]), dB);       // dA_i * dB_j
                fin[i]   = vec_madd(res, vs, fin[i]);
            }
        }
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            C[(ii + i) + (jj + j)*ldc] = fin[i][j];
}

#endif // __MMA__

// Scalar reference dot (mirrors ggml_vec_dot_q2_0_q8_0_generic).
static float q2_q8_dot_ref(int64_t k, const block_q2_0 * x, const block_q8_0 * y) {
    const int64_t nb = k / QK2_0;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const float d0 = fp16_to_fp32(x[i].d);
        float sumi = 0.0f;
        for (int c = 0; c < 4; c++) {
            const block_q8_0 * yb = &y[4*i + c];
            const uint8_t * qs = &x[i].qs[8*c];
            int s = 0;
            for (int b = 0; b < 8; b++) {
                s += (((qs[b] >> 0) & 3) - 1) * yb->qs[4*b + 0];
                s += (((qs[b] >> 2) & 3) - 1) * yb->qs[4*b + 1];
                s += (((qs[b] >> 4) & 3) - 1) * yb->qs[4*b + 2];
                s += (((qs[b] >> 6) & 3) - 1) * yb->qs[4*b + 3];
            }
            sumi += fp16_to_fp32(yb->d) * s;
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

// Public entry point.  Partitions rows of A across threads; 4x4 MMA tiles
// with scalar cleanup on the m/n edges.
extern "C" void gemm_q2_0_q8_0_ppc(int64_t m, int64_t n, int64_t k,
                                   const block_q2_0 * A, int64_t lda,
                                   const block_q8_0 * B, int64_t ldb,
                                   float * C, int64_t ldc,
                                   int ith, int nth) {
    assert(k % QK2_0 == 0);
    const int64_t kb2 = k / QK2_0;   (void)kb2;

    // static row partition per thread, in tiles of 4 where possible
    const int64_t mt   = (m + 3) / 4;                 // number of row tiles
    const int64_t tpt  = (mt + nth - 1) / nth;        // tiles per thread
    const int64_t i0   = 4 * (ith * tpt);
    const int64_t i1   = m < 4 * ((ith + 1) * tpt) ? m : 4 * ((ith + 1) * tpt);
    if (i0 >= m) return;

    int64_t i = i0;
#if defined(__MMA__)
    for (; i + 4 <= i1; i += 4) {
        int64_t j = 0;
        for (; j + 4 <= n; j += 4) {
            q2mma_tile4x4(k,
                A + (i+0)*lda, A + (i+1)*lda, A + (i+2)*lda, A + (i+3)*lda,
                B + (j+0)*ldb, B + (j+1)*ldb, B + (j+2)*ldb, B + (j+3)*ldb,
                C, ldc, i, j);
        }
        for (; j < n; j++)                             // n edge
            for (int r = 0; r < 4; r++)
                C[(i+r) + j*ldc] = q2_q8_dot_ref(k, A + (i+r)*lda, B + j*ldb);
    }
#endif
    for (; i < i1; i++)                                // m edge / no-MMA path
        for (int64_t j = 0; j < n; j++)
            C[i + j*ldc] = q2_q8_dot_ref(k, A + i*lda, B + j*ldb);
}

// ---------------------------------------------------------------------------
#ifdef Q2MMA_TEST
#include <cstdio>
#include <cstdlib>
#include <cmath>

static uint32_t rng = 0x12345678;
static uint32_t xr() { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }

static ggml_half f32_to_f16_approx(float f) {
    // good enough for test scales in [~0.001, ~0.1]
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
        { 13, 7, 384 },                  // ragged edges
        { 32, 1, 1024 },                 // token generation shape
        { 64, 16, 1152 },                // prompt processing shape
    };
    int fails = 0;
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        const int64_t lda = k / QK2_0, ldb = k / QK8_0, ldc = m;

        block_q2_0 * A = (block_q2_0*)aligned_alloc(64, m*lda*sizeof(block_q2_0));
        block_q8_0 * Bm = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
        float * C    = (float*)aligned_alloc(64, m*n*sizeof(float));
        float * Cref = (float*)aligned_alloc(64, m*n*sizeof(float));

        for (int64_t i = 0; i < m*lda; i++) {
            A[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
            for (int b = 0; b < QK2_0/4; b++) {
                uint8_t byte = 0;
                for (int s = 0; s < 4; s++) byte |= (uint8_t)(xr() % 3) << (2*s); // codes 0..2 (set %4 to also exercise the +2 code)
                A[i].qs[b] = byte;
            }
        }
        for (int64_t i = 0; i < n*ldb; i++) {
            Bm[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
            for (int b = 0; b < QK8_0; b++) Bm[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
        }

        // simulate 2 threads
        gemm_q2_0_q8_0_ppc(m, n, k, A, lda, Bm, ldb, C, ldc, 0, 2);
        gemm_q2_0_q8_0_ppc(m, n, k, A, lda, Bm, ldb, C, ldc, 1, 2);

        // float reference (ggml ordering) and exact double reference
        double * Cd = (double*)malloc(m*n*sizeof(double));
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                Cref[i + j*ldc] = q2_q8_dot_ref(k, A + i*lda, Bm + j*ldb);
                // double ref
                const block_q2_0 * x = A + i*lda;
                const block_q8_0 * y = Bm + j*ldb;
                double acc = 0;
                for (int64_t bi = 0; bi < k/QK2_0; bi++) {
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
                Cd[i + j*ldc] = acc;
            }

        // scale for relative error: typical magnitude of the outputs
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
        bool ok = emma < 4*eref + 1e-6;  // MMA no worse than float ref rounding
        printf("m=%3lld n=%3lld k=%5lld  err(mma)=%.3g err(f32 ref)=%.3g  %s\n",
               (long long)m, (long long)n, (long long)k, emma, eref,
               ok ? "OK" : "FAIL");
        if (!ok) fails++;
        free(Cd);
        free(A); free(Bm); free(C); free(Cref);
    }
    printf(fails ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fails ? 1 : 0;
}
#endif
