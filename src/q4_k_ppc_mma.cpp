// q4_k_ppc_mma.cpp
//
// POWER10/POWER11 MMA GEMM for Q4_K x Q8_K -- the first K-quant kernel,
// extending the same architecture as qbit_ppc_mma_v3/v4 to superblock
// quantization.  Q4_K is the most widely used GGUF weight format, and
// like Q1_0/Q2_0 it has no Power MMA path in mainline ggml.
//
// Format (ggml-common.h):
//   Q4_K: superblock of 256 = 8 sub-blocks of 32.
//         value = d*sc[sub]*q - dmin*m[sub],  q in 0..15 (unsigned
//         nibbles; sub 2g uses low nibbles of qs[32g..32g+31], sub 2g+1
//         the high nibbles), 6-bit sc/m packed in scales[12].
//   Q8_K: 256 int8 + one float d + bsums[16] (per-16 sums, precomputed
//         by ggml's quantizer).
//
// Mapping onto the v3 architecture:
//   * q is natively unsigned -> unsigned GER operand, signed activations
//     on the signed operand; no flips, no per-row corrections.
//   * per-sub-block scale d*sc replaces v3's per-block dA in the chunk
//     fixup -- structurally identical, just indexed per chunk.
//   * the mins term is the separable correction:
//         corr(i,j) = sum_sub (dmin_i*m_sub) * (dB_j * S_sub(j)),
//     where S_sub = bsums[2*sub] + bsums[2*sub+1] comes FREE from Q8_K.
//     dB*S is precomputed per (col, sub) in the B pack; the kernel
//     applies one vec_nmsub per (col, rowgroup, sub) at superblock end.
//   * 16x8 tile on all 8 accumulators, identical microkernel.
//
// Conventions: A m x k row-major in superblocks (lda in superblocks),
// B n x k row-major in Q8_K superblocks (ldb in superblocks), C
// column-major float; k % 256 == 0.  KC slab = 8 superblocks (2048).
//
// Tests: -DQ4K_TEST (vs exact double reference, random superblocks with
// random 6-bit scales/mins).  This kernel is standalone-verified; it is
// NOT yet wired into the llama.cpp dispatch patch.

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define QK_K 256
#define K_SCALE_SIZE 12

typedef uint16_t ggml_half;

typedef struct {
    union { struct { ggml_half d; ggml_half dmin; }; uint32_t dm; };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "bad q4_K");
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + (QK_K/16)*sizeof(int16_t), "bad q8_K");

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

// ggml's 6-bit scale/min decode
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
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


#define KC_SB     8                       // superblocks per K slab (2048 elems)
#define KC_CHUNKS (KC_SB * 8)             // 32-element chunks per slab
#define MR        16
#define NR        8

typedef struct {
    vuc v[KC_CHUNKS][32];                 // MMA-layout unsigned nibbles
    vfl dsc[KC_CHUNKS][4];                // d*sc per chunk, 4 rows per vfl
    vfl dm [KC_CHUNKS][4];                // dmin*m per chunk
} a4k_t;

typedef struct {
    vuc v[KC_CHUNKS][16];                 // signed activations
    vfl dB[KC_SB][2];                     // Q8_K d per column, per superblock
    vfl TS[KC_CHUNKS][2];                 // dB * S_sub per (col, chunk)
} b4k_t;

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

// nibble codes of sub-block `sub` (0..7) of one superblock
static inline void q4k_codes32(const uint8_t * qs, int sub, vuc v[2]) {
    const int g = sub >> 1;
    vuc lo = load16u((const uint8_t *)(qs) + (32*g));
    vuc hi = load16u((const uint8_t *)(qs) + (32*g + 16));
    if ((sub & 1) == 0) {
        const vuc mF = vec_splats((unsigned char)0xF);
        v[0] = vec_and(lo, mF);
        v[1] = vec_and(hi, mF);
    } else {
        v[0] = vec_sr(lo, vec_splats((unsigned char)4));
        v[1] = vec_sr(hi, vec_splats((unsigned char)4));
    }
}

// ---- one-time packing ----

static inline int64_t rt4k(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct4k(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl4k(int64_t k) { return (k/QK_K + KC_SB - 1) / KC_SB; }

extern "C" size_t q4k_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt4k(m) * sl4k(k)) * sizeof(a4k_t)) + 63) & ~(size_t)63;
}
extern "C" size_t q4k_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct4k(n) * sl4k(k)) * sizeof(b4k_t)) + 63) & ~(size_t)63;
}

extern "C" void q4k_repack_a(const block_q4_K * A, int64_t lda,
                             int64_t m, int64_t k, void * packed) {
    a4k_t * P = (a4k_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl4k(k);
    for (int64_t it = 0; it < rt4k(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        a4k_t * T = &P[it*ns + s];
        const int64_t sb0 = s*KC_SB;
        const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
        for (int64_t b = 0; b < nsl; b++) {
            const block_q4_K * bp[MR];
            float dsc[MR][8], dm[MR][8];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                bp[r] = &A[rr*lda + sb0 + b];
                const float d    = fp16_to_fp32(bp[r]->d);
                const float dmin = fp16_to_fp32(bp[r]->dmin);
                for (int sub = 0; sub < 8; sub++) {
                    uint8_t sc, mn;
                    get_scale_min_k4(sub, bp[r]->scales, &sc, &mn);
                    dsc[r][sub] = d * sc;
                    dm [r][sub] = dmin * mn;
                }
            }
            for (int sub = 0; sub < 8; sub++) {
                const int64_t ch = 8*b + sub;
                for (int g = 0; g < 4; g++) {
                    T->dsc[ch][g] = (vfl){ dsc[4*g][sub], dsc[4*g+1][sub],
                                           dsc[4*g+2][sub], dsc[4*g+3][sub] };
                    T->dm [ch][g] = (vfl){ dm[4*g][sub], dm[4*g+1][sub],
                                           dm[4*g+2][sub], dm[4*g+3][sub] };
                }
                vuc t[MR][2];
                for (int r = 0; r < MR; r++) q4k_codes32(bp[r]->qs, sub, t[r]);
                vui rows4[4];
                for (int g = 0; g < 4; g++)
                    for (int h = 0; h < 2; h++) {
                        for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                        mma_transpose4(rows4, &T->v[ch][16*h + g], 4);
                    }
            }
        }
    }
}

extern "C" void q4k_pack_b(const block_q8_K * B, int64_t ldb,
                           int64_t n, int64_t k, void * packed) {
    b4k_t * P = (b4k_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl4k(k);
    for (int64_t jt = 0; jt < ct4k(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        b4k_t * T = &P[jt*ns + s];
        const int64_t sb0 = s*KC_SB;
        const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
        for (int64_t b = 0; b < nsl; b++) {
            const block_q8_K * yb[NR];
            float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + sb0 + b];
                dB[j] = yb[j]->d;
            }
            T->dB[b][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            T->dB[b][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            for (int sub = 0; sub < 8; sub++) {
                const int64_t ch = 8*b + sub;
                float TS[NR];
                for (int j = 0; j < NR; j++)
                    TS[j] = dB[j] * (float)(yb[j]->bsums[2*sub] + yb[j]->bsums[2*sub + 1]);
                T->TS[ch][0] = (vfl){ TS[0], TS[1], TS[2], TS[3] };
                T->TS[ch][1] = (vfl){ TS[4], TS[5], TS[6], TS[7] };
                vui rows4[4];
                for (int a = 0; a < 2; a++) {
                    vuc q[4][2];
                    for (int j = 0; j < 4; j++) {
                        q[j][0] = load16u((const uint8_t *)(yb[4*a + j]->qs) + (32*sub));
                        q[j][1] = load16u((const uint8_t *)(yb[4*a + j]->qs) + (32*sub + 16));
                    }
                    for (int h = 0; h < 2; h++) {
                        for (int j = 0; j < 4; j++) rows4[j] = (vui)q[j][h];
                        mma_transpose4(rows4, &T->v[ch][8*h + a], 2);
                    }
                }
            }
        }
    }
}

// ---- 16x8 microkernel ----
static void kernel4k_16x8(const a4k_t * PA, const b4k_t * PB,
                          int64_t nsl, vfl fin[NR][4]) {
    for (int64_t b = 0; b < nsl; b++) {
        for (int sub = 0; sub < 8; sub++) {
            const int64_t ch = 8*b + sub;
            const vuc * a = PA->v[ch];
            const vuc * y = PB->v[ch];
            if (ch + 1 < 8*nsl) {
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
            // Stage all accumulators to the stack first (frees VSRs 0-31
            // for the fixup; see DESIGN.md register-pressure note).
            vsi pr[2][4][4];
            for (int i = 0; i < 2; i++)
                for (int g = 0; g < 4; g++)
                    __builtin_mma_disassemble_acc(pr[i][g], &acc[i][g]);
            // fin += P * (dB_j * dsc_g)  ;  then fin -= TS_j * dm_g
            const vfl dB0 = PB->dB[b][0], dB1 = PB->dB[b][1];
            for (int i = 0; i < 2; i++) {
                const vfl dB = i ? dB1 : dB0;
                const vfl TS = PB->TS[ch][i];
                for (int g = 0; g < 4; g++) {
                    const vsi * rowsP = pr[i][g];
                    const vfl dsc = PA->dsc[ch][g];
                    const vfl dm  = PA->dm [ch][g];
                    const vfl s0 = vec_mul(vec_splat(dB, 0), dsc);
                    const vfl s1 = vec_mul(vec_splat(dB, 1), dsc);
                    const vfl s2 = vec_mul(vec_splat(dB, 2), dsc);
                    const vfl s3 = vec_mul(vec_splat(dB, 3), dsc);
                    fin[4*i+0][g] = vec_madd(vec_ctf(rowsP[0],0), s0, fin[4*i+0][g]);
                    fin[4*i+1][g] = vec_madd(vec_ctf(rowsP[1],0), s1, fin[4*i+1][g]);
                    fin[4*i+2][g] = vec_madd(vec_ctf(rowsP[2],0), s2, fin[4*i+2][g]);
                    fin[4*i+3][g] = vec_madd(vec_ctf(rowsP[3],0), s3, fin[4*i+3][g]);
                    fin[4*i+0][g] = vec_nmsub(dm, vec_splat(TS, 0), fin[4*i+0][g]);
                    fin[4*i+1][g] = vec_nmsub(dm, vec_splat(TS, 1), fin[4*i+1][g]);
                    fin[4*i+2][g] = vec_nmsub(dm, vec_splat(TS, 2), fin[4*i+2][g]);
                    fin[4*i+3][g] = vec_nmsub(dm, vec_splat(TS, 3), fin[4*i+3][g]);
                }
            }
        }
    }
}

extern "C" void q4k_gemm_packed(int64_t m, int64_t n, int64_t k,
                                const void * packedA, const void * packedB,
                                float * C, int64_t ldc, int ith, int nth) {
    const a4k_t * PA = (const a4k_t *)packedA;
    const b4k_t * PB = (const b4k_t *)packedB;
    const int64_t nsb = k/QK_K, ns = sl4k(k), mt = rt4k(m), njt = ct4k(n);
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
                const int64_t sb0 = s*KC_SB;
                const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
                kernel4k_16x8(&PA[it*ns + s], &PB[jt*ns + s], nsl, fin);
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

#endif // __MMA__

// ---------------------------------------------------------------------------
// ---- two-row GEMV, n = 1 (the near-tie squeeze) -----------------------
//
// Generation cannot use the MMA tiles profitably (the packed operand
// rides at 1.8x native bytes; VALIDATION-POWER10.md D3), and ggml's
// vec_dot keeps one weight stream per thread alive -- measured at 49%
// of the machine's bandwidth on a 27B (s9.1).  This kernel walks two
// rows per pass: twice the memory-level parallelism on the weight
// side, and every activation load shared between the rows.  Per-row
// arithmetic order is copied exactly from ggml's POWER9 vec_dot, so
// each row's result is bit-identical to the single-row path -- the
// speedup must come from the memory system alone, and correctness
// comparisons stay trivial.
extern "C" void gemv2_q4_K_q8_K_ppc(int64_t m, int64_t k,
                                    const void * Av, int64_t lda,
                                    const void * Bv, float * C,
                                    int ith, int nth) {
    const block_q4_K * A = (const block_q4_K *)Av;
    const block_q8_K * y = (const block_q8_K *)Bv;
    const int64_t nb = k / QK_K;

    const vector signed char lowMask  = vec_splats((signed char)0xF);
    const vector signed char lowMask1 = vec_splats((int8_t)0x3f);
    const vector signed char lowMask2 = vec_splats((int8_t)0x30);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v2 = vec_splats((uint8_t)2);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    const int64_t rpt = (m + nth - 1) / nth;
    const int64_t lo = (int64_t)ith * rpt;
    const int64_t hi = (lo + rpt) < m ? (lo + rpt) : m;

    for (int64_t r = lo; r < hi; r += 2) {
        const int two = (r + 1 < hi);
        const block_q4_K * xa = A + r*lda;
        const block_q4_K * xb = A + (two ? (r + 1)*lda : r*lda);

        vector float sfa0 = vec_splats(0.0f), sfa1 = vec_splats(0.0f);
        vector float sfa2 = vec_splats(0.0f), sfa3 = vec_splats(0.0f);
        vector float sfb0 = vec_splats(0.0f), sfb1 = vec_splats(0.0f);
        vector float sfb2 = vec_splats(0.0f), sfb3 = vec_splats(0.0f);

        for (int64_t i = 0; i < nb; ++i) {
            const vector float vyd = vec_splats(y[i].d);
            const vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
            const vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

            // ---- per-row scale/min setup (identical to ggml's vec_dot) ----
            vector float vda, vdmina, vdb, vdminb;
            vector signed short vscalesa, vscalesb;
            vector signed short q4xmins0a, q4xmins1a, q4xmins0b, q4xmins1b;
            {
                const vector float vxd = vec_splats(fp16_to_fp32(xa[i].d));
                vda = vec_mul(vxd, vyd);
                const vector float vxmin = vec_splats(fp16_to_fp32(xa[i].dmin));
                vdmina = vec_mul(vxmin, vyd);
                vector signed char u0 = (vector signed char)vec_xl_len((unsigned char *)xa[i].scales, 8);
                vector signed char u1 = vec_and(vec_sr(u0, v2), lowMask2);
                vector signed char u2 = (vector signed char)vec_xl_len((unsigned char *)xa[i].scales + 8, 4);
                vector signed char u3 = vec_sr(u2, v4);
                vector signed char u30 = u1;
                vector signed char u31 = (vector signed char)vec_mergeh((vector signed int)vec_and(u2, lowMask), (vector signed int)u3);
                u1 = vec_and(u0, lowMask1);
                u2 = vec_or(u30, u31);
                vector signed char utmps = (vector signed char)vec_mergeh((vector signed int)u1, (vector signed int)u2);
                vscalesa = vec_unpackh(utmps);
                vector signed short q4xmins = vec_unpackl(utmps);
                q4xmins0a = vec_mergeh(q4xmins, q4xmins);
                q4xmins1a = vec_mergel(q4xmins, q4xmins);
            }
            {
                const vector float vxd = vec_splats(fp16_to_fp32(xb[i].d));
                vdb = vec_mul(vxd, vyd);
                const vector float vxmin = vec_splats(fp16_to_fp32(xb[i].dmin));
                vdminb = vec_mul(vxmin, vyd);
                vector signed char u0 = (vector signed char)vec_xl_len((unsigned char *)xb[i].scales, 8);
                vector signed char u1 = vec_and(vec_sr(u0, v2), lowMask2);
                vector signed char u2 = (vector signed char)vec_xl_len((unsigned char *)xb[i].scales + 8, 4);
                vector signed char u3 = vec_sr(u2, v4);
                vector signed char u30 = u1;
                vector signed char u31 = (vector signed char)vec_mergeh((vector signed int)vec_and(u2, lowMask), (vector signed int)u3);
                u1 = vec_and(u0, lowMask1);
                u2 = vec_or(u30, u31);
                vector signed char utmps = (vector signed char)vec_mergeh((vector signed int)u1, (vector signed int)u2);
                vscalesb = vec_unpackh(utmps);
                vector signed short q4xmins = vec_unpackl(utmps);
                q4xmins0b = vec_mergeh(q4xmins, q4xmins);
                q4xmins1b = vec_mergel(q4xmins, q4xmins);
            }

            {
                vector signed int prod0 = vec_mule(q4xmins0a, q8ysums0);
                vector signed int prod1 = vec_mule(q4xmins1a, q8ysums1);
                vector signed int prod2 = vec_mulo(q4xmins0a, q8ysums0);
                vector signed int prod3 = vec_mulo(q4xmins1a, q8ysums1);
                sfa0 = vec_nmsub(vec_ctf(prod0, 0), vdmina, sfa0);
                sfa1 = vec_nmsub(vec_ctf(prod1, 0), vdmina, sfa1);
                sfa2 = vec_nmsub(vec_ctf(prod2, 0), vdmina, sfa2);
                sfa3 = vec_nmsub(vec_ctf(prod3, 0), vdmina, sfa3);
            }
            {
                vector signed int prod0 = vec_mule(q4xmins0b, q8ysums0);
                vector signed int prod1 = vec_mule(q4xmins1b, q8ysums1);
                vector signed int prod2 = vec_mulo(q4xmins0b, q8ysums0);
                vector signed int prod3 = vec_mulo(q4xmins1b, q8ysums1);
                sfb0 = vec_nmsub(vec_ctf(prod0, 0), vdminb, sfb0);
                sfb1 = vec_nmsub(vec_ctf(prod1, 0), vdminb, sfb1);
                sfb2 = vec_nmsub(vec_ctf(prod2, 0), vdminb, sfb2);
                sfb3 = vec_nmsub(vec_ctf(prod3, 0), vdminb, sfb3);
            }

            vector signed int sia0 = v0, sia1 = v0, sia2 = v0, sia3 = v0;
            vector signed int sib0 = v0, sib1 = v0, sib2 = v0, sib3 = v0;

            const uint8_t * q4a = xa[i].qs;
            const uint8_t * q4b = xb[i].qs;
            const int8_t  * q8  = y[i].qs;

            for (int j = 0; j < QK_K/64; j += 2) {
                const vector signed char qxa0 = (vector signed char)vec_xl( 0, q4a);
                const vector signed char qxa1 = (vector signed char)vec_xl(16, q4a);
                const vector signed char qxa2 = (vector signed char)vec_xl(32, q4a);
                const vector signed char qxa3 = (vector signed char)vec_xl(48, q4a);
                q4a += 64;
                const vector signed char qxb0 = (vector signed char)vec_xl( 0, q4b);
                const vector signed char qxb1 = (vector signed char)vec_xl(16, q4b);
                const vector signed char qxb2 = (vector signed char)vec_xl(32, q4b);
                const vector signed char qxb3 = (vector signed char)vec_xl(48, q4b);
                q4b += 64;

                const vector signed char q8y00 = vec_xl(  0, q8);
                const vector signed char q8y10 = vec_xl( 16, q8);
                const vector signed char q8y01 = vec_xl( 32, q8);
                const vector signed char q8y11 = vec_xl( 48, q8);
                const vector signed char q8y20 = vec_xl( 64, q8);
                const vector signed char q8y30 = vec_xl( 80, q8);
                const vector signed char q8y21 = vec_xl( 96, q8);
                const vector signed char q8y31 = vec_xl(112, q8);
                q8 += 128;

                {
                    vector unsigned char x00 = (vector unsigned char)vec_and(qxa0, lowMask);
                    vector unsigned char x01 = (vector unsigned char)vec_sr(qxa0, v4);
                    vector unsigned char x10 = (vector unsigned char)vec_and(qxa1, lowMask);
                    vector unsigned char x11 = (vector unsigned char)vec_sr(qxa1, v4);
                    vector unsigned char x20 = (vector unsigned char)vec_and(qxa2, lowMask);
                    vector unsigned char x21 = (vector unsigned char)vec_sr(qxa2, v4);
                    vector unsigned char x30 = (vector unsigned char)vec_and(qxa3, lowMask);
                    vector unsigned char x31 = (vector unsigned char)vec_sr(qxa3, v4);
                    vector signed int qv00 = vec_msum(q8y00, x00, v0);
                    vector signed int qv01 = vec_msum(q8y01, x01, v0);
                    vector signed int qv10 = vec_msum(q8y10, x10, v0);
                    vector signed int qv11 = vec_msum(q8y11, x11, v0);
                    vector signed int qv20 = vec_msum(q8y20, x20, v0);
                    vector signed int qv21 = vec_msum(q8y21, x21, v0);
                    vector signed int qv30 = vec_msum(q8y30, x30, v0);
                    vector signed int qv31 = vec_msum(q8y31, x31, v0);
                    vector signed int vscales_h = vec_unpackh(vscalesa);
                    vector signed int vs0 = vec_splat(vscales_h, 0);
                    vector signed int vs1 = vec_splat(vscales_h, 1);
                    vector signed int vs2 = vec_splat(vscales_h, 2);
                    vector signed int vs3 = vec_splat(vscales_h, 3);
                    vscalesa = vec_sld(vscalesa, vscalesa, 8);
                    sia0 = vec_add(vec_mul(qv00, vs0), sia0);
                    sia1 = vec_add(vec_mul(qv01, vs1), sia1);
                    sia2 = vec_add(vec_mul(qv20, vs2), sia2);
                    sia3 = vec_add(vec_mul(qv21, vs3), sia3);
                    sia0 = vec_add(vec_mul(qv10, vs0), sia0);
                    sia1 = vec_add(vec_mul(qv11, vs1), sia1);
                    sia2 = vec_add(vec_mul(qv30, vs2), sia2);
                    sia3 = vec_add(vec_mul(qv31, vs3), sia3);
                }
                {
                    vector unsigned char x00 = (vector unsigned char)vec_and(qxb0, lowMask);
                    vector unsigned char x01 = (vector unsigned char)vec_sr(qxb0, v4);
                    vector unsigned char x10 = (vector unsigned char)vec_and(qxb1, lowMask);
                    vector unsigned char x11 = (vector unsigned char)vec_sr(qxb1, v4);
                    vector unsigned char x20 = (vector unsigned char)vec_and(qxb2, lowMask);
                    vector unsigned char x21 = (vector unsigned char)vec_sr(qxb2, v4);
                    vector unsigned char x30 = (vector unsigned char)vec_and(qxb3, lowMask);
                    vector unsigned char x31 = (vector unsigned char)vec_sr(qxb3, v4);
                    vector signed int qv00 = vec_msum(q8y00, x00, v0);
                    vector signed int qv01 = vec_msum(q8y01, x01, v0);
                    vector signed int qv10 = vec_msum(q8y10, x10, v0);
                    vector signed int qv11 = vec_msum(q8y11, x11, v0);
                    vector signed int qv20 = vec_msum(q8y20, x20, v0);
                    vector signed int qv21 = vec_msum(q8y21, x21, v0);
                    vector signed int qv30 = vec_msum(q8y30, x30, v0);
                    vector signed int qv31 = vec_msum(q8y31, x31, v0);
                    vector signed int vscales_h = vec_unpackh(vscalesb);
                    vector signed int vs0 = vec_splat(vscales_h, 0);
                    vector signed int vs1 = vec_splat(vscales_h, 1);
                    vector signed int vs2 = vec_splat(vscales_h, 2);
                    vector signed int vs3 = vec_splat(vscales_h, 3);
                    vscalesb = vec_sld(vscalesb, vscalesb, 8);
                    sib0 = vec_add(vec_mul(qv00, vs0), sib0);
                    sib1 = vec_add(vec_mul(qv01, vs1), sib1);
                    sib2 = vec_add(vec_mul(qv20, vs2), sib2);
                    sib3 = vec_add(vec_mul(qv21, vs3), sib3);
                    sib0 = vec_add(vec_mul(qv10, vs0), sib0);
                    sib1 = vec_add(vec_mul(qv11, vs1), sib1);
                    sib2 = vec_add(vec_mul(qv30, vs2), sib2);
                    sib3 = vec_add(vec_mul(qv31, vs3), sib3);
                }
            }

            sfa0 = vec_madd(vec_ctf(sia0, 0), vda, sfa0);
            sfa1 = vec_madd(vec_ctf(sia1, 0), vda, sfa1);
            sfa2 = vec_madd(vec_ctf(sia2, 0), vda, sfa2);
            sfa3 = vec_madd(vec_ctf(sia3, 0), vda, sfa3);
            sfb0 = vec_madd(vec_ctf(sib0, 0), vdb, sfb0);
            sfb1 = vec_madd(vec_ctf(sib1, 0), vdb, sfb1);
            sfb2 = vec_madd(vec_ctf(sib2, 0), vdb, sfb2);
            sfb3 = vec_madd(vec_ctf(sib3, 0), vdb, sfb3);
        }

        sfa0 = vec_add(sfa0, sfa2);
        sfa1 = vec_add(sfa1, sfa3);
        sfa0 = vec_add(sfa0, sfa1);
        sfa0 = vec_add(sfa0, vec_sld(sfa0, sfa0, 4));
        sfa0 = vec_add(sfa0, vec_sld(sfa0, sfa0, 8));
        C[r] = vec_extract(sfa0, 0);

        if (two) {
            sfb0 = vec_add(sfb0, sfb2);
            sfb1 = vec_add(sfb1, sfb3);
            sfb0 = vec_add(sfb0, sfb1);
            sfb0 = vec_add(sfb0, vec_sld(sfb0, sfb0, 4));
            sfb0 = vec_add(sfb0, vec_sld(sfb0, sfb0, 8));
            C[r + 1] = vec_extract(sfb0, 0);
        }
    }
}

#ifdef Q4K_TEST
#include <cstdio>
#include <cmath>

static uint32_t rng = 0x1234abcd;
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

// exact double reference from the dequant formula
static double dref(int64_t k, const block_q4_K * x, const block_q8_K * y) {
    double acc = 0;
    for (int64_t sb = 0; sb < k/QK_K; sb++) {
        const double d    = fp16_to_fp32(x[sb].d);
        const double dmin = fp16_to_fp32(x[sb].dmin);
        for (int sub = 0; sub < 8; sub++) {
            uint8_t sc, mn;
            get_scale_min_k4(sub, x[sb].scales, &sc, &mn);
            const int g = sub >> 1;
            long P = 0, S = 0;
            for (int l = 0; l < 32; l++) {
                const uint8_t byte = x[sb].qs[32*g + l];
                const int q = (sub & 1) ? (byte >> 4) : (byte & 0xF);
                const int yv = y[sb].qs[32*sub + l];
                P += q * yv; S += yv;
            }
            acc += (double)y[sb].d * (d*sc*(double)P - dmin*mn*(double)S);
        }
    }
    return acc;
}

int main() {
    struct { int64_t m, n, k; } cases[] = {
        { 16, 8, 256 }, { 32, 16, 512 }, { 13, 7, 768 },
        { 64, 16, 2048 }, { 40, 24, 4096 }, { 17, 9, 2304 },
        { 32, 1, 2048 }, { 9, 3, 4352 },
    };
    int fails = 0;
    for (auto & tc : cases) {
        const int64_t m = tc.m, n = tc.n, k = tc.k;
        const int64_t lda = k/QK_K, ldb = k/QK_K, ldc = m;
        block_q4_K * A = (block_q4_K*)aligned_alloc(64, m*lda*sizeof(block_q4_K));
        block_q8_K * B = (block_q8_K*)aligned_alloc(64, n*ldb*sizeof(block_q8_K));
        float * C = (float*)aligned_alloc(64, m*n*sizeof(float));

        for (int64_t i = 0; i < m*lda; i++) {
            A[i].d    = f32_to_f16_approx(0.0005f + (xr()%1000)/400000.0f);
            A[i].dmin = f32_to_f16_approx(0.0005f + (xr()%1000)/400000.0f);
            for (int b = 0; b < K_SCALE_SIZE; b++) A[i].scales[b] = (uint8_t)(xr() & 0xff);
            for (int b = 0; b < QK_K/2; b++)       A[i].qs[b]     = (uint8_t)(xr() & 0xff);
        }
        for (int64_t i = 0; i < n*ldb; i++) {
            B[i].d = 0.001f + (xr()%1000)/500000.0f;
            for (int b = 0; b < QK_K; b++) B[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
            for (int g = 0; g < QK_K/16; g++) {           // correct bsums
                int s = 0;
                for (int l = 0; l < 16; l++) s += B[i].qs[16*g + l];
                B[i].bsums[g] = (int16_t)s;
            }
        }

        void * PA = aligned_alloc(64, q4k_apack_size(m, k));
        void * PB = aligned_alloc(64, q4k_bpack_size(n, k));
        q4k_repack_a(A, lda, m, k, PA);
        q4k_pack_b(B, ldb, n, k, PB);
        q4k_gemm_packed(m, n, k, PA, PB, C, ldc, 0, 2);
        q4k_gemm_packed(m, n, k, PA, PB, C, ldc, 1, 2);

        double emax = 0, scale = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++) {
                double ref = dref(k, A + i*lda, B + j*ldb);
                scale += fabs(ref);
                double e = fabs((double)C[i + j*ldc] - ref);
                if (e > emax) emax = e;
            }
        scale = scale/(m*n) + 1e-30;
        emax /= scale;
        bool ok = emax < 1e-5;
        printf("q4_K m=%3lld n=%3lld k=%5lld  err=%.3g  %s\n",
               (long long)m, (long long)n, (long long)k, emax, ok ? "OK":"FAIL");
        if (!ok) fails++;
        free(A); free(B); free(C); free(PA); free(PB);
    }

    // ---- two-row GEMV (n = 1) vs the same float64 reference ----
    {
        struct { int64_t m, k; } gcases[] = {
            { 16, 256 }, { 13, 768 }, { 64, 2048 }, { 17, 4352 }, { 1, 512 },
        };
        for (auto & gc : gcases) {
            const int64_t m = gc.m, k = gc.k, lda = k/QK_K;
            block_q4_K * A = (block_q4_K*)aligned_alloc(64, ((m*lda*sizeof(block_q4_K))+63)&~63ul);
            block_q8_K * B = (block_q8_K*)aligned_alloc(64, ((lda*sizeof(block_q8_K))+63)&~63ul);
            float * C = (float*)aligned_alloc(64, ((m*sizeof(float))+63)&~63ul);
            for (int64_t i = 0; i < m*lda; i++) {
                A[i].d    = f32_to_f16_approx(0.001f + (xr()%1000)/400000.0f);
                A[i].dmin = f32_to_f16_approx(0.0005f + (xr()%1000)/800000.0f);
                for (int b = 0; b < K_SCALE_SIZE; b++) A[i].scales[b] = (uint8_t)(xr() & 0xff);
                for (int b = 0; b < QK_K/2; b++)       A[i].qs[b]     = (uint8_t)(xr() & 0xff);
            }
            for (int64_t i = 0; i < lda; i++) {
                B[i].d = 0.001f + (xr()%1000)/500000.0f;
                for (int b = 0; b < QK_K; b++) B[i].qs[b] = (int8_t)((int)(xr()%255) - 127);
                for (int g = 0; g < QK_K/16; g++) {
                    int s = 0;
                    for (int l = 0; l < 16; l++) s += B[i].qs[16*g + l];
                    B[i].bsums[g] = (int16_t)s;
                }
            }
            gemv2_q4_K_q8_K_ppc(m, k, A, lda, B, C, 0, 3);
            gemv2_q4_K_q8_K_ppc(m, k, A, lda, B, C, 1, 3);
            gemv2_q4_K_q8_K_ppc(m, k, A, lda, B, C, 2, 3);
            double emax = 0, scale = 0;
            for (int64_t i = 0; i < m; i++) {
                double ref = dref(k, A + i*lda, B);
                scale += fabs(ref);
                double e = fabs((double)C[i] - ref);
                if (e > emax) emax = e;
            }
            scale = scale/m + 1e-30;
            emax /= scale;
            bool ok = emax < 1e-5;
            printf("q4_K gemv2 m=%3lld k=%5lld  err=%.3g  %s\n",
                   (long long)m, (long long)k, emax, ok ? "OK":"FAIL");
            if (!ok) fails++;
            free(A); free(B); free(C);
        }
    }
    printf(fails ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return fails ? 1 : 0;
}
#endif
