// xcheck_grid.cpp — independent decoder cross-check, grid/ternary/NVFP4
// families (N1 in VALIDATION-POWER10.md).
//
// Compares this repo's production scalar decoders (dec_* / dec16_* in
// iq_grid_ppc_mma.cpp — the functions repack actually runs) against
// ggml's own dequantize_row_* implementations, vendored verbatim from
// the pinned fork into xcheck_ggml_ref.h.  Two layers:
//
//   1. EXHAUSTIVE table comparison: every entry of every grid/sign
//      table, memcmp against the vendored originals.  This is the
//      transcription-error class REVIEW.md flagged, closed completely.
//   2. Block-level dequant equivalence on random blocks: repo decode
//      (sc[group] * code[i]) vs ggml dequantize_row, all formats.
//      Bit-exact float agreement is expected for these decoders (the
//      expressions differ only by sign commutation and exact
//      power-of-two scalings); tolerance 1e-6 relative is the failure
//      line, and the bit-exact count is reported so a regression from
//      "identical" to "merely close" is visible.
//
// Closure argument: this file proves decode == ggml; the existing
// iq_grid suite proves kernel == decode-based reference.  Together:
// kernel == ggml.

#include "iq_grid_ppc_mma.cpp"   // repo decoders + iq_grids.h tables
#include "xcheck_ggml_ref.h"     // namespace ggmlref: vendored originals

#include <cstdio>
#include <cstdlib>
#include <cmath>

static uint32_t xrng = 0x5eed1234;
static uint32_t xrnd() { xrng ^= xrng<<13; xrng ^= xrng>>17; xrng ^= xrng<<5; return xrng; }

static int g_fails = 0;

static void table_check(const char * name, const void * repo, const void * ref,
                        size_t repo_sz, size_t ref_sz) {
    if (repo_sz != ref_sz) {
        printf("%-14s SIZE MISMATCH repo=%zu ref=%zu\n", name, repo_sz, ref_sz);
        g_fails++;
        return;
    }
    if (memcmp(repo, ref, repo_sz) != 0) {
        printf("%-14s ENTRY MISMATCH (%zu bytes)\n", name, repo_sz);
        g_fails++;
        return;
    }
    printf("%-14s identical (%zu bytes, exhaustive)\n", name, repo_sz);
}

// Decode one random block both ways and compare.  gs = elements per
// scale group; qk = elements per block.  Returns via counters.
template <typename BLK, typename REFBLK, int QK, int GS, int NSC>
static void block_check(const char * name,
                        void (*dec)(const BLK *, int8_t *, float *),
                        void (*refdeq)(const REFBLK *, float *, int64_t),
                        int trials) {
    static_assert(sizeof(BLK) == sizeof(REFBLK), "layout divergence");
    int8_t code[QK]; float sc[NSC];
    float yref[QK];
    uint8_t raw[sizeof(BLK)];
    double maxrel = 0;
    long bitexact = 0, total = 0, skipped = 0;
    for (int t = 0; t < trials; t++) {
        for (size_t i = 0; i < sizeof raw; i++) raw[i] = (uint8_t)(xrnd() & 0xff);
        BLK b; memcpy(&b, raw, sizeof b);
        REFBLK rb; memcpy(&rb, raw, sizeof rb);
        refdeq(&rb, yref, QK);
        bool finite = true;
        for (int i = 0; i < QK; i++) if (!isfinite(yref[i])) { finite = false; break; }
        if (!finite) { skipped++; continue; }   // NaN/Inf fp16 scale in random bytes
        dec(&b, code, sc);
        double bmax = 0;
        for (int i = 0; i < QK; i++) if (fabsf(yref[i]) > bmax) bmax = fabsf(yref[i]);
        for (int i = 0; i < QK; i++) {
            float yrepo = sc[i/GS] * (float)code[i];
            if (yrepo == yref[i] || (yrepo == 0.0f && yref[i] == 0.0f)) bitexact++;
            double rel = fabs((double)yrepo - (double)yref[i]) / (bmax + 1e-20);
            if (rel > maxrel) maxrel = rel;
            total++;
        }
    }
    const bool ok = maxrel < 1e-6;
    printf("%-10s %s  maxrel=%.3g  bitexact=%ld/%ld  (skipped %ld non-finite)\n",
           name, ok ? "OK  " : "FAIL", maxrel, bitexact, total, skipped);
    if (!ok) g_fails++;
}

int main() {
    printf("== exhaustive grid/sign table comparison vs vendored ggml\n");
    table_check("iq2xxs_grid", iq2xxs_grid, ggmlref::iq2xxs_grid, sizeof iq2xxs_grid, sizeof ggmlref::iq2xxs_grid);
    table_check("iq2xs_grid",  iq2xs_grid,  ggmlref::iq2xs_grid,  sizeof iq2xs_grid,  sizeof ggmlref::iq2xs_grid);
    table_check("iq2s_grid",   iq2s_grid,   ggmlref::iq2s_grid,   sizeof iq2s_grid,   sizeof ggmlref::iq2s_grid);
    table_check("iq3xxs_grid", iq3xxs_grid, ggmlref::iq3xxs_grid, sizeof iq3xxs_grid, sizeof ggmlref::iq3xxs_grid);
    table_check("iq3s_grid",   iq3s_grid,   ggmlref::iq3s_grid,   sizeof iq3s_grid,   sizeof ggmlref::iq3s_grid);
    table_check("iq1s_grid",   iq1s_grid,   ggmlref::iq1s_grid,   sizeof iq1s_grid,   sizeof ggmlref::iq1s_grid);
    table_check("ksigns_iq2xs", ksigns_iq2xs, ggmlref::ksigns_iq2xs, sizeof ksigns_iq2xs, sizeof ggmlref::ksigns_iq2xs);
    table_check("kmask_iq2xs", kmask_iq2xs, ggmlref::kmask_iq2xs, sizeof kmask_iq2xs, sizeof ggmlref::kmask_iq2xs);
    table_check("nvfp4_kvalues", nvfp4_kvalues, ggmlref::kvalues_mxfp4, sizeof nvfp4_kvalues, sizeof ggmlref::kvalues_mxfp4);

    printf("== block dequant equivalence, random blocks\n");
    const int R = 4096;
    block_check<block_tq2_0,   ggmlref::block_tq2_0,   QK_K, 32, 8 >("tq2_0",   dec_tq2_0,   ggmlref::dequantize_row_tq2_0,   R);
    block_check<block_tq1_0,   ggmlref::block_tq1_0,   QK_K, 32, 8 >("tq1_0",   dec_tq1_0,   ggmlref::dequantize_row_tq1_0,   R);
    block_check<block_iq2_xxs, ggmlref::block_iq2_xxs, QK_K, 32, 8 >("iq2_xxs", dec_iq2_xxs, ggmlref::dequantize_row_iq2_xxs, R);
    block_check<block_iq3_xxs, ggmlref::block_iq3_xxs, QK_K, 32, 8 >("iq3_xxs", dec_iq3_xxs, ggmlref::dequantize_row_iq3_xxs, R);
    block_check<block_iq3_s,   ggmlref::block_iq3_s,   QK_K, 32, 8 >("iq3_s",   dec_iq3_s,   ggmlref::dequantize_row_iq3_s,   R);
    block_check<block_iq1_s,   ggmlref::block_iq1_s,   QK_K, 32, 8 >("iq1_s",   dec_iq1_s,   ggmlref::dequantize_row_iq1_s,   R);
    block_check<block_iq2_xs,  ggmlref::block_iq2_xs,  QK_K, 16, 16>("iq2_xs",  dec16_iq2_xs, ggmlref::dequantize_row_iq2_xs, R);
    block_check<block_iq2_s,   ggmlref::block_iq2_s,   QK_K, 16, 16>("iq2_s",   dec16_iq2_s,  ggmlref::dequantize_row_iq2_s,  R);
    block_check<block_iq1_m,   ggmlref::block_iq1_m,   QK_K, 16, 16>("iq1_m",   dec16_iq1_m,  ggmlref::dequantize_row_iq1_m,  R);
    block_check<block_nvfp4,   ggmlref::block_nvfp4,   QK_NVFP4, 16, 4>("nvfp4", dec_nvfp4,   ggmlref::dequantize_row_nvfp4,  R);

    if (g_fails) { printf("XCHECK FAILED (%d)\n", g_fails); return 1; }
    printf("ALL DECODER CROSS-CHECKS PASSED\n");
    return 0;
}
