# First hardware validation: POWER10

This is the run that BENCHMARKS-QEMU.md promised would supersede it —
the first execution of this repository on real Power silicon, closing
the gap the README labels "what is not verified." Everything below was
produced by the unmodified tree at commit `d8882ba` on 2026-07-22; the
validation itself is the stock `scripts/validate-on-power.sh`, no
hand-editing, no retries.

## The machine

IBM 9105-42A (Power S1022) LPAR, POWER10 "architected" mode, 8
hardware threads visible, 123 GiB RAM. RHEL 9.7, kernel
5.14.0-611.42.1.el9_7.ppc64le, GCC 11.5.0. Being an LPAR, the
absolute throughput numbers are specific to this partition's
entitlement; the MMA-vs-reference ratios were measured on the same
partition minutes apart and are the meaningful part, per house rules.

One incidental data point: the docs list GCC 14 as the tested
toolchain with a claimed floor of 10.2. GCC 11.5 built the kernels,
the patched fork, and both validation variants without a single source
change, so the floor holds in practice, not just in the manual.

## What ran

Three layers, in increasing order of integration:

1. **Kernel suites, native.** `make CROSS= QEMU= test` — all 14
   suites (13 plus the `IQGRID_PINGPONG` variant) against the exact
   float64 references, compiled `-mcpu=power10`, executed on the MMA
   engine itself rather than qemu's model of it. All passed;
   normalized errors sit in the same 1e-7 to 2e-6 band the emulated
   runs produce.
2. **Patched fork build.** `scripts/build-bonsai-power.sh` — pinned
   llama.cpp checkout, all nine patches applied in sequence with zero
   rejects, `llama-cli`/`llama-server`/`llama-bench` built and linked.
3. **The temperature-0 gate.** `scripts/validate-on-power.sh` with
   `prism-ml/Ternary-Bonsai-1.7B-Q2_0.gguf` (qwen3-arch, 1.72 B
   params, 436 MiB) — the full build-twice, generate-twice,
   bench-twice protocol. Fork build `79697f23a (9595)`.

## Correctness: PASS, tier 1

The MMA-native build and the `-mcpu=power9` reference build produced
**token-identical output** — 32 greedy tokens, seed 42, temperature 0,
same prompt. The gate's tier-2 near-tie certification never came into
play; this is the strong verdict.

What that one comparison establishes, all at once: `xvi8ger4pp`
behaves on silicon as the kernels assume (operand signedness, lane
layout, accumulation); the decode-at-repack decoders agree with ggml's
own dequantization; and the dispatch actually routes Q2_0 mat-muls
through the MMA path end to end. Scope worth stating precisely: this
run exercises the Q2_0 ternary kernels and the model's unquantized
layers through the full stack. The other 24 formats are
silicon-verified at kernel level (layer 1 above) but have not yet been
driven end-to-end through a model.

Because the fork's startup banner turned out not to print an `MMA =`
flag (see field notes), MMA presence was verified at the binary level
instead: `objdump -d build-mma/bin/libggml-cpu.so` contains **496**
`xvi8ger4`-family instructions; the reference build contains **zero**.

## Throughput

`llama-bench`, same model, same partition, thread ladder 2/4/8.
The reference build is the identical source tree with `-mcpu=power9`,
which compiles out every `#if __MMA__` region — i.e. exactly the
scalar/VSX vec_dot path a non-MMA build of this fork runs for Q2_0.

| threads | test | MMA t/s | reference t/s | ratio |
|--:|---|--:|--:|--:|
| 2 | pp128 | 116.65 ± 0.05 | 2.18 ± 0.00 | 53.5× |
| 4 | pp128 | 219.95 ± 0.01 | 4.36 ± 0.00 | 50.4× |
| 8 | pp128 | 417.72 ± 0.39 | 8.69 ± 0.01 | 48.1× |
| 2 | tg32  | 8.50 ± 0.00   | 1.74 ± 0.00 | 4.9× |
| 4 | tg32  | 16.69 ± 0.01  | 3.46 ± 0.00 | 4.8× |
| 8 | tg32  | 30.97 ± 0.03  | 6.70 ± 0.13 | 4.6× |

Two shapes in this table were predicted by the emulation-era analysis
and are now confirmed by an instrument that can actually see them:

- **Prompt processing vs generation.** ~50× on pp against ~5× on tg is
  the bandwidth story BENCHMARKS-QEMU.md said qemu could not price:
  batch mat-muls are compute-bound and the MMA engine eats them;
  single-token generation is memory-bound, so the kernels win by less
  and the remaining gap belongs to weight traffic, not arithmetic.
- **Thread scaling.** The reference build scales perfectly linearly
  (2.18 → 4.36 → 8.69, 3.99× over 2→8) because scalar code is
  compute-starved and nowhere near the memory system's limits. The
  MMA build scales at 3.58× over the same ladder — already brushing
  bandwidth at 8 threads. Linear scaling in the slow build is not a
  virtue; it is the signature of leaving the machine idle.

## What this closes in REVIEW.md, and what it does not

Three rows of the verification matrix were waiting on hardware:

- *End-to-end inference numerics through patched dispatch* — *verified*
  for the Q2_0 path by the tier-1 token identity above.
- *Decoders vs ggml's dequantization* — the same run covers the Q2_0
  decoder independently; the grid/codebook decoders remain
  consistency-checked only, pending end-to-end runs on models that use
  them.
- *Performance on silicon* — first real datapoint, this page.

Still open: end-to-end runs for the remaining formats (a K-quant model
and an IQ-grid model would cover most of the surface), UBSan on
silicon (emulation-only so far), and the `IQGRID_PINGPONG` A/B that
this repo has been explicitly saving for hardware — the bench above
does not exercise the grid kernels at all.

## Field notes

Honest small findings from the first contact, in the spirit of the
defect log:

- The script's "MMA active in native build" check cannot report
  absence: `grep … | head -1 || echo …` never triggers the fallback
  because `head` exits 0 regardless, and this fork's `system_info`
  line has no `MMA =` token to find in the first place (it prints
  `VSX = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1`). The objdump
  count above is the robust replacement and should probably become the
  scripted check.
- The report's `Repo:` field renders empty when the tree is deployed
  without `.git` (as it was here, via tar). Cosmetic, but the field
  exists to pin provenance, so it failed at its one job.
- The correctness-gate servers ran single-threaded (the fork's
  `llama-server` default on this box). Irrelevant to the verdict —
  token identity does not depend on thread count — but it explains why
  each generation took ~a minute despite the bench numbers above.
- `/tmp` on the test box still held outputs from the 2026-07-21 field
  session (the one that produced the "field FAIL" fixes in the git
  log). The script truncates and rewrites its files, so stale state
  cannot leak into the gate — verified by timestamp before trusting
  the run.

## Reproducing

```
scripts/build-bonsai-power.sh
scripts/validate-on-power.sh /path/to/Ternary-Bonsai-1.7B-Q2_0.gguf
```

Twenty minutes on this partition, most of it the two builds. The
script emits `validation-report.md`; this page is that report with its
context filled in.
