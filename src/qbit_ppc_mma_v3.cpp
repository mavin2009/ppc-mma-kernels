// qbit_ppc_mma_v3.cpp
//
// POWER10/POWER11 MMA GEMM kernels, v3, for PrismML low-bit weights:
//   Q1_0 (1-bit, {-1,+1})   x Q8_0 activations
//   Q2_0 (2-bit, {-1,0,1,2}) x Q8_0 activations
// Both formats: 128 weights/block, FP16 scale; Q8_0: 32 int8 + FP16 scale.
//
// ---------------------------------------------------------------------
// The algebra (what changed vs v2)
// ---------------------------------------------------------------------
// xvi8ger4pp multiplies SIGNED bytes (first operand) by UNSIGNED bytes
// (second operand).  v1/v2 put signed weights on the signed side and
// XOR-flipped the activations, paying a per-(row,chunk) bias correction
// -128*sum(t).  v3 instead uses the codes' natural unsignedness:
//
//     weight value t = alpha*code - 1,   alpha = 2 (Q1_0), 1 (Q2_0)
//     => sum(t*y) = alpha*sum(code*y) - sum(y)
//
// with raw codes on the unsigned operand and untouched int8 activations
// on the signed operand.  Consequences:
//   * no XOR flip in the activation pack,
//   * no per-row chunk sums / popcounts in the weight pack,
//   * the -sum(y) term is SEPARABLE in the scales:
//         corr(i,j) = sum_blocks dA(i,blk) * E(j,blk),
//         E(j,blk)  = sum_{chunks c in blk} dB(j,c) * sum(y_c)
//     i.e. a rank-1-per-block outer product applied ONCE per tile at
//     slab end (a few hundred FMAs per 128 outputs over 2048 depth)
//     instead of a correction add on every chunk of the inner loop.
//     E lives in the activation pack, whose cost is amortized over all
//     weight rows -- the favorable side, since m >> n in LLM layers.
//
// ---------------------------------------------------------------------
// The microarchitecture (why the kernel is shaped like this)
// ---------------------------------------------------------------------
// Power10/Power11 cores have one MMA engine per SMT4 core sustaining one
// 512-bit accumulator GER per cycle; xvi8ger4pp is 64 int8 MACs, so the
// GER stream is the throughput floor and everything else must hide in
// the four 128-bit VSU slices running alongside.  GER results carry
// multi-cycle latency into the SAME accumulator, so consecutive updates
// of one accumulator serialize: saturating the engine needs many
// independent accumulator chains.  v3 therefore runs a 16x8 tile on ALL
// 8 accumulators (8 independent chains, 8 GERs per depth step).  ISA
// 3.1 accumulators ACC0-7 alias VSRs 0-31, leaving VSRs 32-63 for the
// 6 operand vectors per step plus loop state -- exactly why a 16x8 tile
// fits and a larger one would spill.  The swapped operand orientation
// also makes accumulator rows correspond to activation columns, so the
// finished tile stores to column-major C with plain unaligned vector
// stores instead of lane extraction.  Power11 keeps the same MMA ISA
// and per-core engine (more cores, higher clocks), so the same kernel
// binary is the right one there; build with -mcpu=power10.
//
// Packing (hoisted, K-blocked into 2048-element slabs as in v2): the
// per-slab A pack is 32 KiB -- sized against the 32 KiB L1D so the
// compute loop streams packed operands from L1; software prefetch
// (__builtin_prefetch / dcbt) pulls the next chunk's packed lines.
// In a real ggml integration the A pack belongs in repack.cpp at model
// load and the B pack next to activation quantization (shared across
// threads; here it is per-thread).
//
// Conventions as v1/v2: A m x k row-major in blocks, B n x k row-major
// in q8 blocks, C column-major float (C[i + j*ldc]); k % 128 == 0.
//
// Build (cross):
//   powerpc64le-linux-gnu-g++ -O3 -mcpu=power10 -DQBIT_TEST \
//       qbit_ppc_mma_v3.cpp -o qbit_v3_test
//   qemu-ppc64le -cpu power10 -L /usr/powerpc64le-linux-gnu ./qbit_v3_test

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

static_assert(sizeof(block_q1_0) == 18, "bad q1_0");
static_assert(sizeof(block_q2_0) == 34, "bad q2_0");
static_assert(sizeof(block_q8_0) == 34, "bad q8_0");

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

#define KC_BLKS   16                       // 2048-element K slab
#define KC_CHUNKS (KC_BLKS * 4)
#define MR        16                       // weight rows per tile
#define NR        8                        // activation cols per tile

// Packed weights: unsigned codes in MMA layout.
// v[ch][4*x + b]: depth step x (0..7), row group b (0..3).
typedef struct {
    vuc v[KC_CHUNKS][32];
    vfl dA[KC_BLKS][4];                    // per-block scales, 4 rows per vfl
} apack_t;

// Packed activations: signed int8, untouched.
// v[ch][2*x + a]: depth step x, col group a (0..1).
typedef struct {
    vuc v[KC_CHUNKS][16];
    vfl dB[KC_CHUNKS][2];                  // per-chunk scales, cols 0-3 / 4-7
    vfl E[KC_BLKS][2];                     // sum_c dB(j,c)*sum(y_c) per block
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

// ---- weight unpack: raw UNSIGNED codes (no sign mapping needed) ----

// Q1_0: chunk c from the block's 16 qs bytes -> bits 0/1 as bytes.
static inline void q1_codes32(vuc raw, int c, vuc v[2]) {
    const vuc rep0 = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc rep1 = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc sh   = { 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7 };
    const vuc m1   = vec_splats((unsigned char)1);
    const vuc off  = vec_splats((unsigned char)(4*c));
    v[0] = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep0, off)), sh), m1);
    v[1] = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep1, off)), sh), m1);
}

// Q2_0: chunk c from the block's 32 qs bytes (lo/hi) -> codes 0..3.
static inline void q2_codes32(vuc lo, vuc hi, int c, vuc v[2]) {
    const vuc rep0 = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
    const vuc rep1 = { 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7 };
    const vuc sh   = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    const vuc m3   = vec_splats((unsigned char)3);
    const vuc off  = vec_splats((unsigned char)(8*c));
    v[0] = vec_and(vec_sr(vec_perm(lo, hi, vec_add(rep0, off)), sh), m3);
    v[1] = vec_and(vec_sr(vec_perm(lo, hi, vec_add(rep1, off)), sh), m3);
}

// ---- packers ----

static void pack_A16_q1(const block_q1_0 * A, int64_t lda, int64_t rows,
                        int64_t blk0, int64_t nblk, apack_t * P) {
    for (int64_t b = 0; b < nblk; b++) {
        vuc raw[MR]; float d[MR];
        for (int r = 0; r < MR; r++) {
            int64_t rr = r < rows ? r : rows - 1;
            const block_q1_0 * bp = &A[rr*lda + blk0 + b];
            d[r]   = fp16_to_fp32(bp->d);
            raw[r] = vec_xl(0, (const unsigned char *)bp->qs);
        }
        for (int g = 0; g < 4; g++)
            P->dA[b][g] = (vfl){ d[4*g], d[4*g+1], d[4*g+2], d[4*g+3] };
        for (int c = 0; c < 4; c++) {
            const int ch = 4*b + c;
            vuc t[MR][2];
            for (int r = 0; r < MR; r++) q1_codes32(raw[r], c, t[r]);
            vui rows4[4];
            for (int g = 0; g < 4; g++)
                for (int h = 0; h < 2; h++) {
                    for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                    mma_transpose4(rows4, &P->v[ch][16*h + g], 4);
                }
        }
    }
}

static void pack_A16_q2(const block_q2_0 * A, int64_t lda, int64_t rows,
                        int64_t blk0, int64_t nblk, apack_t * P) {
    for (int64_t b = 0; b < nblk; b++) {
        vuc lo[MR], hi[MR]; float d[MR];
        for (int r = 0; r < MR; r++) {
            int64_t rr = r < rows ? r : rows - 1;
            const block_q2_0 * bp = &A[rr*lda + blk0 + b];
            d[r]  = fp16_to_fp32(bp->d);
            lo[r] = vec_xl(0,  (const unsigned char *)bp->qs);
            hi[r] = vec_xl(16, (const unsigned char *)bp->qs);
        }
        for (int g = 0; g < 4; g++)
            P->dA[b][g] = (vfl){ d[4*g], d[4*g+1], d[4*g+2], d[4*g+3] };
        for (int c = 0; c < 4; c++) {
            const int ch = 4*b + c;
            vuc t[MR][2];
            for (int r = 0; r < MR; r++) q2_codes32(lo[r], hi[r], c, t[r]);
            vui rows4[4];
            for (int g = 0; g < 4; g++)
                for (int h = 0; h < 2; h++) {
                    for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                    mma_transpose4(rows4, &P->v[ch][16*h + g], 4);
                }
        }
    }
}

// Activations: no flip; also builds per-chunk scales and the separable
// correction accumulators E.
static void pack_B8(const block_q8_0 * B, int64_t ldb, int64_t cols,
                    int64_t q8blk0, int64_t nblk, bpack_t * P) {
    for (int64_t b = 0; b < nblk; b++) {
        vfl E0 = vec_splats(0.0f), E1 = vec_splats(0.0f);
        for (int c = 0; c < 4; c++) {
            const int64_t ch = 4*b + c;
            const block_q8_0 * yb[NR];
            float dB[NR], S[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = j < cols ? j : cols - 1;
                yb[j] = &B[jj*ldb + q8blk0 + ch];
                dB[j] = fp16_to_fp32(yb[j]->d);
            }
            vui rows4[4];
            for (int a = 0; a < 2; a++) {
                vuc q[4][2];
                for (int j = 0; j < 4; j++) {
                    q[j][0] = vec_xl(0,  (const unsigned char *)yb[4*a + j]->qs);
                    q[j][1] = vec_xl(16, (const unsigned char *)yb[4*a + j]->qs);
                    vsi z = vec_splats(0);
                    vsi s = vec_sum4s((vsc)q[j][0], z);
                    s = vec_sum4s((vsc)q[j][1], s);
                    S[4*a + j] = (float)(s[0] + s[1] + s[2] + s[3]);
                }
                for (int h = 0; h < 2; h++) {
                    for (int j = 0; j < 4; j++) rows4[j] = (vui)q[j][h];
                    mma_transpose4(rows4, &P->v[ch][8*h + a], 2);
                }
            }
            vfl dB0 = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            vfl dB1 = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            P->dB[ch][0] = dB0;
            P->dB[ch][1] = dB1;
            E0 = vec_madd(dB0, (vfl){ S[0], S[1], S[2], S[3] }, E0);
            E1 = vec_madd(dB1, (vfl){ S[4], S[5], S[6], S[7] }, E1);
        }
        P->E[b][0] = E0;
        P->E[b][1] = E1;
    }
}

// ---- 16x8 microkernel on all 8 accumulators ----
// fin[j][g]: activation col j (0..7), weight row group g (0..3).
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
            // fixup: fin += alpha*dB_j * dA_g * P
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
    // separable correction: fin[j][g] -= sum_b dA[b][g] * E(j,b)
    for (int64_t b = 0; b < nblk; b++) {
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
}

// ---- driver, templated over weight type ----
template <typename BLK, void (*PACK_A)(const BLK *, int64_t, int64_t, int64_t, int64_t, apack_t *), int ALPHA>
static void gemm_qbit(int64_t m, int64_t n, int64_t k,
                      const BLK * A, int64_t lda,
                      const block_q8_0 * B, int64_t ldb,
                      float * C, int64_t ldc, int ith, int nth) {
    assert(k % 128 == 0);
    const int64_t kb  = k / 128;
    const int64_t mt  = (m + MR - 1) / MR;
    const int64_t tpt = (mt + nth - 1) / nth;
    const int64_t i0  = MR * (ith * tpt);
    const int64_t i1  = m < MR * ((ith + 1) * tpt) ? m : MR * ((ith + 1) * tpt);
    if (i0 >= m) return;

    const int64_t njt = (n + NR - 1) / NR;
    apack_t * PA = (apack_t*)aligned_alloc(64, sizeof(apack_t));
    bpack_t * PB = (bpack_t*)aligned_alloc(64, njt * sizeof(bpack_t));
    vfl (*fin)[NR][4] = (vfl(*)[NR][4])aligned_alloc(64, njt * sizeof(vfl[NR][4]));

    for (int64_t i = i0; i < i1; i += MR) {
        const int64_t rows = (i1 - i) < MR ? (i1 - i) : MR;
        for (int64_t jt = 0; jt < njt; jt++)
            for (int j = 0; j < NR; j++)
                for (int g = 0; g < 4; g++)
                    fin[jt][j][g] = vec_splats(0.0f);

        for (int64_t blk0 = 0; blk0 < kb; blk0 += KC_BLKS) {
            const int64_t nblk = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
            PACK_A(A + i*lda, lda, rows, blk0, nblk, PA);
            for (int64_t jt = 0; jt < njt; jt++) {
                const int64_t j    = NR*jt;
                const int64_t cols = (n - j) < NR ? (n - j) : NR;
                if (i == i0 || kb > KC_BLKS)
                    pack_B8(B + j*ldb, ldb, cols, 4*blk0, nblk, &PB[jt]);
                kernel_16x8(PA, &PB[jt], nblk, (float)ALPHA, fin[jt]);
            }
        }

        for (int64_t jt = 0; jt < njt; jt++) {
            const int64_t j    = NR*jt;
            const int64_t cols = (n - j) < NR ? (n - j) : NR;
            for (int64_t cj = 0; cj < cols; cj++) {
                float * dst = C + i + (j + cj)*ldc;
                if (rows == MR) {
                    for (int g = 0; g < 4; g++)
                        vec_xst(fin[jt][cj][g], 16*g, dst);
                } else {
                    for (int64_t r = 0; r < rows; r++)
                        dst[r] = fin[jt][cj][r >> 2][r & 3];
                }
            }
        }
    }
    free(PA); free(PB); free(fin);
}

extern "C" void gemm_q1_0_q8_0_ppc_v3(int64_t m, int64_t n, int64_t k,
        const block_q1_0 * A, int64_t lda, const block_q8_0 * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    gemm_qbit<block_q1_0, pack_A16_q1, 2>(m, n, k, A, lda, B, ldb, C, ldc, ith, nth);
}

extern "C" void gemm_q2_0_q8_0_ppc_v3(int64_t m, int64_t n, int64_t k,
        const block_q2_0 * A, int64_t lda, const block_q8_0 * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    gemm_qbit<block_q2_0, pack_A16_q2, 1>(m, n, k, A, lda, B, ldb, C, ldc, ith, nth);
}

#endif // __MMA__

// ---------------------------------------------------------------------------
#ifdef QBIT_TEST
#include <cstdio>
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

// exact double references
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

template <typename BLK>
static void fill_A(BLK * A, int64_t nblk, int mod);
template <> void fill_A<block_q1_0>(block_q1_0 * A, int64_t nblk, int) {
    for (int64_t i = 0; i < nblk; i++) {
        A[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
        for (int b = 0; b < 16; b++) A[i].qs[b] = (uint8_t)(xr() & 0xff);
    }
}
template <> void fill_A<block_q2_0>(block_q2_0 * A, int64_t nblk, int mod) {
    for (int64_t i = 0; i < nblk; i++) {
        A[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
        for (int b = 0; b < 32; b++) {
            uint8_t byte = 0;
            for (int s = 0; s < 4; s++) byte |= (uint8_t)(xr() % mod) << (2*s);
            A[i].qs[b] = byte;
        }
    }
}

template <typename BLK,
          void (*GEMM)(int64_t,int64_t,int64_t,const BLK*,int64_t,
                       const block_q8_0*,int64_t,float*,int64_t,int,int),
          double (*DREF)(int64_t,const BLK*,const block_q8_0*)>
static int run_cases(const char * name, int mod) {
    struct { int64_t m, n, k; } cases[] = {
        { 16, 8, 128 }, { 16, 8, 256 }, { 32, 16, 512 },
        { 13, 7, 384 }, { 32, 1, 1024 }, { 64, 16, 1152 },
        { 40, 24, 4096 }, { 9, 3, 2176 }, { 17, 9, 2048 },
    };
    int fails = 0;
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        const int64_t lda = k/128, ldb = k/QK8_0, ldc = m;
        BLK * A = (BLK*)aligned_alloc(64, m*lda*sizeof(BLK));
        block_q8_0 * Bm = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
        float * C = (float*)aligned_alloc(64, m*n*sizeof(float));
        fill_A<BLK>(A, m*lda, mod);
        for (int64_t i = 0; i < n*ldb; i++) {
            Bm[i].d = f32_to_f16_approx(0.001f + (xr()%1000)/50000.0f);
            for (int b = 0; b < QK8_0; b++) Bm[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
        }
        GEMM(m, n, k, A, lda, Bm, ldb, C, ldc, 0, 2);
        GEMM(m, n, k, A, lda, Bm, ldb, C, ldc, 1, 2);
        double emax = 0, scale = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double ref = DREF(k, A + i*lda, Bm + j*ldb);
                scale += fabs(ref);
                double e = fabs((double)C[i + j*ldc] - ref);
                if (e > emax) emax = e;
            }
        scale = scale/(m*n) + 1e-30;
        emax /= scale;
        bool ok = emax < 5e-6;   // float rounding scale over up to 4096 depth
        printf("%s m=%3lld n=%3lld k=%5lld  err=%.3g  %s\n",
               name, (long long)m, (long long)n, (long long)k, emax, ok ? "OK":"FAIL");
        if (!ok) fails++;
        free(A); free(Bm); free(C);
    }
    return fails;
}

int main() {
    int fails = 0;
    fails += run_cases<block_q1_0, gemm_q1_0_q8_0_ppc_v3, dref_q1>("q1", 0);
    fails += run_cases<block_q2_0, gemm_q2_0_q8_0_ppc_v3, dref_q2>("q2(3)", 3);
    fails += run_cases<block_q2_0, gemm_q2_0_q8_0_ppc_v3, dref_q2>("q2(4)", 4);
    printf(fails ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fails ? 1 : 0;
}
#endif

#ifdef QBIT_BENCH
#include <cstdio>
#include <ctime>
int main() {
    const int64_t m = 512, n = 64, k = 2048, iters = 20;
    const int64_t lda = k/128, ldb = k/QK8_0, ldc = m;
    block_q1_0 * A = (block_q1_0*)aligned_alloc(64, m*lda*sizeof(block_q1_0));
    block_q8_0 * B = (block_q8_0*)aligned_alloc(64, n*ldb*sizeof(block_q8_0));
    float * C = (float*)aligned_alloc(64, m*n*sizeof(float));
    memset(A, 0x5a, m*lda*sizeof(block_q1_0));
    memset(B, 0x11, n*ldb*sizeof(block_q8_0));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int64_t it = 0; it < iters; it++)
        gemm_q1_0_q8_0_ppc_v3(m, n, k, A, lda, B, ldb, C, ldc, 0, 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);
    printf("v3(q1): %.3f s for %lld iters (m=%lld n=%lld k=%lld), C[0]=%g\n",
           s, (long long)iters, (long long)m, (long long)n, (long long)k, (double)C[0]);
    return 0;
}
#endif
