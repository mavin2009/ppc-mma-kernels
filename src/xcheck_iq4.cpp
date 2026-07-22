// xcheck_iq4.cpp — independent decoder cross-check, IQ4_NL / IQ4_XS /
// MXFP4 (N1 in VALIDATION-POWER10.md).
//
// These kernels decode inside their repack functions, so the scalar
// statement of their decode lives in the suite's own float64 dot
// references (dref_nl, dref_xs, and the inline MXFP4 ref).  A one-hot
// activation extracts single dequantized elements from those
// references: dref(block, e_j) = scale * value_j exactly, in double.
// Compared against ggml's dequantize_row_* (vendored verbatim).
//
// Closure argument: this file proves reference == ggml element-wise;
// the existing IQ4 suite proves kernel == reference.  Together:
// kernel == ggml.
//
// Comparison policy: results are cast to float and compared for
// bit-identity where the expression trees allow it (single-scale
// formats); the failure line is 1e-6 relative, far below any real
// decode defect (a wrong table entry or shift shows up at ~1e-1).

#include "xcheck_ggml_ref.h"

#define IQ4_TEST
#define main iq4_suite_main
#include "iq4_ppc_mma.cpp"
#undef main

static uint32_t xrng2 = 0xc0ffee11;
static uint32_t xrnd2() { xrng2 ^= xrng2<<13; xrng2 ^= xrng2>>17; xrng2 ^= xrng2<<5; return xrng2; }

static int g_fails = 0;

static void report(const char * name, double maxrel, long bitexact, long total, long skipped) {
    const bool ok = maxrel < 1e-6;
    printf("%-10s %s  maxrel=%.3g  bitexact=%ld/%ld  (skipped %ld non-finite)\n",
           name, ok ? "OK  " : "FAIL", maxrel, bitexact, total, skipped);
    if (!ok) g_fails++;
}

int main() {
    printf("== table comparison vs vendored ggml\n");
    if (sizeof kvalues_iq4nl != sizeof ggmlref::kvalues_iq4nl ||
        memcmp(kvalues_iq4nl, ggmlref::kvalues_iq4nl, sizeof kvalues_iq4nl) != 0) {
        printf("kvalues_iq4nl MISMATCH\n"); g_fails++;
    } else printf("kvalues_iq4nl  identical (exhaustive)\n");
    if (sizeof kvalues_mxfp4_ != sizeof ggmlref::kvalues_mxfp4 ||
        memcmp(kvalues_mxfp4_, ggmlref::kvalues_mxfp4, sizeof kvalues_mxfp4_) != 0) {
        printf("kvalues_mxfp4 MISMATCH\n"); g_fails++;
    } else printf("kvalues_mxfp4  identical (exhaustive)\n");

    printf("== one-hot element equivalence, random blocks\n");
    const int R = 2048;

    // ---- IQ4_NL: 32-element blocks, fp16 scale ----
    {
        double maxrel = 0; long bitexact = 0, total = 0, skipped = 0;
        for (int t = 0; t < R; t++) {
            block_iq4_nl b;
            uint8_t * raw = (uint8_t *)&b;
            for (size_t i = 0; i < sizeof b; i++) raw[i] = (uint8_t)(xrnd2() & 0xff);
            ggmlref::block_iq4_nl rb; memcpy(&rb, &b, sizeof b);
            float yref[QK4_NL];
            ggmlref::dequantize_row_iq4_nl(&rb, yref, QK4_NL);
            bool finite = true;
            for (int i = 0; i < QK4_NL; i++) if (!isfinite(yref[i])) { finite = false; break; }
            if (!finite) { skipped++; continue; }
            double bmax = 0;
            for (int i = 0; i < QK4_NL; i++) if (fabsf(yref[i]) > bmax) bmax = fabsf(yref[i]);
            block_q8_0 oh; oh.d = (ggml_half)0x3C00;   // 1.0f
            for (int j = 0; j < QK4_NL; j++) {
                memset(oh.qs, 0, sizeof oh.qs);
                oh.qs[j] = 1;
                float yrepo = (float)dref_nl(32, &b, &oh);
                if (yrepo == yref[j]) bitexact++;
                double rel = fabs((double)yrepo - (double)yref[j]) / (bmax + 1e-20);
                if (rel > maxrel) maxrel = rel;
                total++;
            }
        }
        report("iq4_nl", maxrel, bitexact, total, skipped);
    }

    // ---- MXFP4: 32-element blocks, E8M0 half scale ----
    {
        double maxrel = 0; long bitexact = 0, total = 0, skipped = 0;
        for (int t = 0; t < R; t++) {
            block_mxfp4 b;
            uint8_t * raw = (uint8_t *)&b;
            for (size_t i = 0; i < sizeof b; i++) raw[i] = (uint8_t)(xrnd2() & 0xff);
            ggmlref::block_mxfp4 rb; memcpy(&rb, &b, sizeof b);
            float yref[QK_MXFP4];
            ggmlref::dequantize_row_mxfp4(&rb, yref, QK_MXFP4);
            bool finite = true;
            for (int i = 0; i < QK_MXFP4; i++) if (!isfinite(yref[i])) { finite = false; break; }
            if (!finite) { skipped++; continue; }
            double bmax = 0;
            for (int i = 0; i < QK_MXFP4; i++) if (fabsf(yref[i]) > bmax) bmax = fabsf(yref[i]);
            // repo decode pieces, mapping as in the suite's MXFP4 reference:
            // element j   <- kvalues_mxfp4_[qs[j] & 0xF]
            // element j+16 <- kvalues_mxfp4_[qs[j] >> 4], scale e8m0_half
            const float d = e8m0_half_to_fp32(b.e);
            for (int j = 0; j < 16; j++) {
                float lo = d * (float)kvalues_mxfp4_[b.qs[j] & 0xF];
                float hi = d * (float)kvalues_mxfp4_[b.qs[j] >>  4];
                const float pairs[2] = { lo, hi };
                const int   idx[2]   = { j, j + 16 };
                for (int p = 0; p < 2; p++) {
                    if (pairs[p] == yref[idx[p]]) bitexact++;
                    double rel = fabs((double)pairs[p] - (double)yref[idx[p]]) / (bmax + 1e-20);
                    if (rel > maxrel) maxrel = rel;
                    total++;
                }
            }
        }
        report("mxfp4", maxrel, bitexact, total, skipped);
    }

    // ---- IQ4_XS: 256-element superblocks, fp16 super scale + 6-bit sub-scales ----
    {
        double maxrel = 0; long bitexact = 0, total = 0, skipped = 0;
        static block_q8_K oh;
        for (int t = 0; t < R/4; t++) {
            block_iq4_xs b;
            uint8_t * raw = (uint8_t *)&b;
            for (size_t i = 0; i < sizeof b; i++) raw[i] = (uint8_t)(xrnd2() & 0xff);
            ggmlref::block_iq4_xs rb; memcpy(&rb, &b, sizeof b);
            float yref[QK_K];
            ggmlref::dequantize_row_iq4_xs(&rb, yref, QK_K);
            bool finite = true;
            for (int i = 0; i < QK_K; i++) if (!isfinite(yref[i])) { finite = false; break; }
            if (!finite) { skipped++; continue; }
            double bmax = 0;
            for (int i = 0; i < QK_K; i++) if (fabsf(yref[i]) > bmax) bmax = fabsf(yref[i]);
            memset(&oh, 0, sizeof oh);
            oh.d = 1.0f;
            for (int j = 0; j < QK_K; j++) {
                oh.qs[j] = 1;
                float yrepo = (float)dref_xs(QK_K, &b, &oh);
                oh.qs[j] = 0;
                if (yrepo == yref[j]) bitexact++;
                double rel = fabs((double)yrepo - (double)yref[j]) / (bmax + 1e-20);
                if (rel > maxrel) maxrel = rel;
                total++;
            }
        }
        report("iq4_xs", maxrel, bitexact, total, skipped);
    }

    if (g_fails) { printf("XCHECK FAILED (%d)\n", g_fails); return 1; }
    printf("ALL DECODER CROSS-CHECKS PASSED\n");
    return 0;
}
