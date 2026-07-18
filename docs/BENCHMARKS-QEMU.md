# Virtualized (qemu) performance snapshot

Everything on this page was measured under **qemu-user TCG emulation**
(`qemu-ppc64le -cpu power10`) on an x86-64 host. Read the caveat first.

## What emulated numbers can and cannot tell you

qemu-user translates each guest instruction; wall time is therefore a
rough **instruction-count proxy**. It models **no** cache hierarchy, no
memory latency, no SMT, no MMA pipe throughput, and runs single-threaded.
Two consequences:

- **Valid**: relative comparisons of instruction volume on the *same
  workload* (e.g. kernel v2 vs v3, GEMM vs GEMV path).
- **Invalid**: absolute throughput, anything memory-bound, thread
  scaling, and cross-format comparisons (each suite's workload differs).

Hardware numbers (`llama-bench` on Power10/11) supersede everything
here on arrival; see DEPLOY.md step 5.

## Same-workload comparisons (the meaningful part)

`make bench`, m=512 n=64 k=2048, 20 iterations, Q1_0×Q8_0:

| kernel | qemu wall | vs v2 |
|---|---|---|
| v2 (8×8 tile, 4 accumulators) | 5.759 s | — |
| v3 (16×8 tile, 8 accumulators, separable correction) | 5.533 s | −3.9 % instructions |

v3's static analysis predicts a much larger *hardware* gap (non-GER
vector work per GER drops 3.81 → 2.16; DESIGN.md): qemu prices every
instruction equally, so it understates the value of shifting work off
the VSU-bound path. This is the clearest illustration of the proxy's
blindness — and why the repo does not publish absolute claims.

Production v4 API, n=1 token-generation shape (m=4096, k=2048):

| path | qemu per call |
|---|---|
| packed GEMM | 0.086 s |
| no-packing GEMV (mask-select + vsum4s) | 0.051 s (−41 %) |

On hardware the GEMV path's advantage should widen further: it reads
weights at 1–2 bits/element instead of the 8-bit packed form, and n=1
inference is bandwidth-bound.

## Verification-workload runtimes (not cross-comparable)

Full `make test-production` under qemu — each suite's random-matrix
workload differs, so these are provided only as a scale reference for
CI budgeting:

| suite | formats | qemu wall |
|---|---|---|
| qbit_test | Q1_0, Q2_0 | 0.39 s |
| q2_k_test | Q2_K | 1.58 s |
| q3_k_test | Q3_K | 1.21 s |
| q4_k_test | Q4_K | 0.18 s |
| q5_k_test | Q5_K | 0.20 s |
| q6_k_test | Q6_K | 1.05 s |
| iq4_test | IQ4_NL, IQ4_XS, MXFP4 | 0.32 s |
| legacy_test | Q4_1, Q5_0, Q5_1 | 5.23 s |
| iq_grid_test | TQ1_0/TQ2_0, IQ2/IQ3 grids, IQ1_S/M, NVFP4 | 1.04 s |

(≈12 s total for the whole production matrix — cheap enough to run on
every commit.)

## Static instruction accounting (compiler-level, exact)

Per 2048-element-slab chunk iteration of the compiled signed kernels
(GCC 14, `-O3 -mcpu=power10`, objdump):

| kernel | insns/chunk | GERs/chunk | non-GER : GER |
|---|---|---|---|
| 32-deep signed (grids, IQ4, MXFP4) | 276 | 32 | 7.6 : 1 |
| 16-deep signed (per-16 scales, NVFP4) | 232 | 16 | 13.5 : 1 |

These ratios bound how much of the machine's MMA throughput each
kernel family can expose and are the primary hypotheses to test on
silicon.
