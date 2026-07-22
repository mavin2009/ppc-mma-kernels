# Hardware validation: POWER10

First contact between this repository and real Power silicon: an IBM
9105-42A (Power S1022) LPAR, POWER10 architected mode, 8 hardware
threads, 123 GiB RAM, RHEL 9.7, GCC 11.5.0, on 2026-07-22. The
kernels and patches ran unmodified at commit `d8882ba`; the
*validation harness* did not survive the day unmodified, and the
first half of this page explains why before any result is claimed.
An earlier revision of this page reported a hardware PASS whose
evidence was void — see the incident below. This revision replaces
it; nothing here rests on the invalidated runs.

Being an LPAR, absolute throughput is specific to this partition's
entitlement. MMA-vs-reference ratios were measured on the same
partition minutes apart and are the numbers that travel.

## The incident: a forged PASS, and what it voided

The temperature-0 gate works by starting `llama-server` twice on a
fixed port — once per build — and comparing greedy outputs. On this
machine, an orphaned `llama-server` from the *previous day's* field
session (started Jul 21 17:08, serving an unrelated 27B Q8_0 model)
still held that port. Every subsequent gate run — the field session's
later attempts and the first three runs of this validation — loaded
its model, failed to bind, and died, while the health check and both
completion requests were answered by the squatter. Two identical
answers from one server: a PASS that validated nothing.

Three details made it hard to see. The squatter kept writing its log
into the same file each new server truncated, NUL-padding it into
something `grep` calls binary. The generation text was a plausible
completion of the gate's prompt. And `run_gen`'s `kill` — the thing
that should have prevented the orphan from existing — had already
succeeded in every observed run; the orphan came from a run that died
between spawn and kill. The failure mode was invisible precisely
where the script was looking.

The harness now refuses to run if anything answers on its port,
verifies after the health check that its own child process is the
answerer (and, where `/props` exposes `model_path`, that the served
model is the one under test), and verifies after each kill that the
port actually died with the server. A squatted port is a loud FATAL,
not a forged PASS. The banner check, which grepped the server log
for an `MMA = 1` token this fork never prints (behind an `|| echo`
fallback that could never fire), was replaced by counting GER
instructions in the binaries: 496 `xvi8ger4`-family instructions in
the native `libggml-cpu.so`, zero in the reference build.

## What ran, clean

Kernel level first, then end to end. All native, GCC 11.5 (docs
previously claimed GCC 14 tested; the 10.2 floor holds in practice).

1. **All 14 kernel suites** (`make CROSS= QEMU= test`) against exact
   float64 references, on the MMA engine itself. All pass; errors in
   the same 1e-7 to 2e-6 band as under emulation.
2. **UBSan on silicon**: the qbit, q4_K, iq_grid, iq_grid_pp and
   legacy suites rebuilt with `-fsanitize=undefined
   -fno-sanitize-recover=all` and run natively. Clean. (Previously
   emulation-only.)
3. **The patch series**: applies in sequence with zero rejects at the
   pinned commit; `llama-cli`/`llama-server`/`llama-bench` build.
   Fork build `79697f23a (9595)`.
4. **End-to-end gates** on four models chosen to cover the format
   surface, run sequentially on an otherwise idle box:

| model | formats inside (tensor census) | gate verdict |
|---|---|---|
| Ternary-Bonsai-1.7B Q2_0, 436 MiB | Q2_0 (qbit) | **PASS** — token-identical |
| qwen3.5 27B Q4_K-Medium, 15.65 GiB | Q4_K + Q6_K | **PASS** — token-identical |
| Qwen2.5-1.5B IQ3_XS, 692 MiB | IQ3_XXS ×98, IQ3_S ×70, Q4_K ×28, Q6_K embed | **PASS** — token-identical |
| Qwen2.5-1.5B IQ2_M, 568 MiB | IQ2_S ×137, IQ3_S ×31, Q4_K ×28, Q5_K embed | tier-2 divergence → **certified benign** (below) |

## The IQ2_M divergence, and what it actually was

IQ2_M diverged at token 5 with chosen-vs-alternative gaps of
0.125/0.127 logprob against the gate's 0.10 near-tie tolerance — by
the gate's stated rule, a numerical defect. Before accepting that,
two experiments:

**A pure-Q5_K probe.** The failing file's one non-IQ suspect was its
Q5_K tied embedding — the output projection, m=151936, a shape no
kernel suite reaches. A pure-Q5_K requant of the same base model
(zero IQ tensors) *also* diverged (token 15, gaps 0.282/0.082). So
either two unrelated format families broke the same way, or the gate
was measuring something other than kernels.

**A control with no kernels in it.** The same tree built twice more,
`-mcpu=power8` vs `-mcpu=power9` — vanilla ggml both, no MMA, none of
this repository's code — generating the same 64 greedy tokens on the
same IQ2_M file:

| pair | diverges at | median \|Δlogprob\| over shared prefix |
|---|---|---|
| MMA ↔ power9 reference | token 5 | 0.122 |
| **power9 ↔ power8 (no repo code)** | **token 7** | **0.110** |
| MMA ↔ power8 | token 5 | 0.201 |

Two builds containing none of this project's code flip a token and
carry the same ~0.11 median drift as the MMA build. The divergence is
this machine-pair's *cross-codegen rounding envelope* — compiler
vectorization differences amplified through 28 transformer layers —
and the MMA build sits inside it. The passing IQ3_XS run drifts just
as much (median 0.113); it simply met no near-tie in 32 tokens. The
27B and Bonsai passes are sharper-distribution models: fewer
near-ties, so token identity survives the same drift.

Conclusion, stated carefully: **no evidence of a kernel defect; the
gate's fixed tolerance was below the machine's own envelope, so its
"real numerical defect" verdict overclaimed.** The gate now has a
third tier: on a tier-2 divergence it builds the power8 control,
measures the envelope, and confirms FAIL only if the MMA build's
drift exceeds twice what two MMA-free builds do to each other.
Re-running IQ2_M under the three-tier gate: **PASS (codegen
envelope)** — MMA-vs-reference median drift 0.114 against an MMA-free
control envelope of 0.072, with the control pair itself flipping a
token.

What tier 3 does *not* do is independently verify the grid decoders —
REVIEW.md's matrix row for that stays open, and an exhaustive
decoder-vs-`dequantize_row` cross-check remains the right follow-up.

## Throughput: hardware picked two winners and one loser

`llama-bench`, MMA vs power9 reference, 8-thread rows (full 2/4/8
ladders in the archived reports; ratios hold across the ladder):

| model | pp128 MMA | pp128 ref | pp ratio | tg32 MMA | tg32 ref | tg ratio |
|---|--:|--:|--:|--:|--:|--:|
| Q2_0 1.7B (qbit) | 403.7 | 8.7 | **46×** | 30.3 | 6.8 | **4.4×** |
| Q4_K 27B | 22.3 | 5.9 | **3.8×** | 1.41 | 3.90 | **0.36×** |
| IQ2_M 1.5B | 172.8 | 41.4 | **4.2×** | 4.18 | 32.6 | **0.13×** |
| IQ3_XS 1.5B | 228.4 | 32.7 | **7.0×** | 4.28 | 26.5 | **0.16×** |

Prompt processing: the MMA engine wins everywhere, 4× to 46×. Token
generation splits into two regimes:

- **qbit formats win tg** (4.4×): v4 has a dedicated no-packing GEMV
  path that reads weights at their native 1–2 bits per element.
- **Packed-cache formats lose tg**, 2.8× (Q4_K) to 7.8× (IQ2_M),
  against ggml's own VSX vec_dot. The signature is diagnostic: MMA
  tg is ~1.1/2.2/4.2 t/s at 2/4/8 threads for *both* IQ models
  regardless of their size — linear thread scaling, far below
  bandwidth (4.18 t/s × 568 MiB ≈ 2.4 GB/s) — i.e. compute-bound on
  per-token pack/decode work in the n=1 path, not memory-bound
  inference. The qbit GEMV design is the proven in-repo fix pattern;
  extending it to the K-quant and grid kernels is now the top
  performance work item, and no deployment serving generation-heavy
  traffic should ship the MMA build for these formats until it lands.

This is exactly the class of result emulation could not price, and it
cuts both ways: the 46× pp win and the 7.8× tg loss were equally
invisible to instruction counting.

## IQGRID_PINGPONG: hardware picked the winner, barely

The A/B this repo saved for silicon (`-DIQGRID_PINGPONG`, patch
0010): both variants built (xxmfacc count 165 vs 197 — the double
accumulator set is really there) and benched on both IQ models.
Result: ping-pong gains 3–4% in generation on IQ3_XS (4.38 vs 4.22
t/s at 8t, outside noise), costs 1–5% in prompt processing (61.1 vs
64.5 t/s at 2t), and moves IQ2_M by less than ±1% anywhere. The 3%
static-instruction cost predicted in DESIGN.md shows up in pp almost
exactly as counted; the engine-idle removal is real but small at this
scale. **Standard stays the default.** The A/B should be repeated
once the GEMV work lands, since tg is where ping-pong helps and tg is
currently dominated by the pack overhead above.

## Field notes

- The orphaned-server incident above is the headline entry; the
  repo's own prior "field FAIL" analysis from Jul 21 was reasoning
  about outputs that the squatter generated. Conclusions drawn from
  that session should be re-examined.
- `validate-on-power.sh` exits 0 even when the gate fails; the runner
  driving sequential validations had to grep verdicts out of logs.
  An exit code would be kinder to CI.
- The report's `Repo:` field is empty when the tree is deployed
  without `.git`. It exists to pin provenance; it failed at its one
  job and should fall back to something.
- `/tmp` artifact files are overwritten by every run; the failing
  IQ2_M evidence was lost to the next run's truncation and had to be
  regenerated. Runs should archive their artifacts under a per-run
  directory.
- The 16.8 GB file named `qwen3.6-q4.gguf` on the test box is
  actually qwen3.5 27B Q4_K-Medium per its own metadata. Filenames
  lie; tensor censuses do not.

## Reproducing

```
scripts/build-bonsai-power.sh
scripts/validate-on-power.sh /path/to/model.gguf
```

The gate is now three-tier: token identity, then near-tie
certification, then the power8 codegen-envelope control (built on
demand, a few extra minutes, only on divergence). A squatted port or
a GER-free "MMA" build aborts loudly. Twenty minutes per model on
this partition, most of it the two builds.
