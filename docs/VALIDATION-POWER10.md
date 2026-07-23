# Hardware validation report: POWER10

**Date:** 2026-07-22 · **Tree:** `d8882ba` at start; harness and
integration fixes landed same day (three-tier gate, patch 0015) ·
**Verdict: PASS.** Correctness holds at kernel level and end to end
for every format exercised; the one apparent numerical failure was
proven to be gate miscalibration, not a kernel defect. Three defects
were found — two in the validation harness, one in the fork
integration — all root-caused, fixed, and re-verified on silicon the
same day. This validation is the 1.0.0 gate.

An earlier revision of this page reported a PASS whose evidence was
void (defect D1). Nothing below rests on the invalidated runs.

## 1. Environment

| | |
|---|---|
| Machine | IBM 9105-42A (Power S1022) LPAR, POWER10 architected, 8 hw threads, 123 GiB RAM |
| OS / toolchain | RHEL 9.7, kernel 5.14.0-611.42.1.el9_7.ppc64le, GCC 11.5.0, cmake 3.31 |
| Fork | pinned commit `79697f23a (9595)`, patches 0001–0016 applied, zero rejects |
| Builds | `build-mma` (GGML_NATIVE), `build-ref` (`-mcpu=power9`, MMA compiled out), `build-ref8` (`-mcpu=power8`, gate control) |
| MMA presence | objdump: 496 `xvi8ger4`-family instructions in native `libggml-cpu.so`, 0 in reference |

GCC 11.5 is below the documented tested toolchain (GCC 14) and above
the 10.2 floor; everything built without a source change. Being an
LPAR, absolute throughput is partition-specific; MMA-vs-reference
ratios were measured minutes apart on the same partition and are the
numbers that travel.

## 2. Results summary

All numbers from the post-0015 build, 8 threads, `llama-bench`
(`-p 128 -n 32 -r 2`); reference = same tree, `-mcpu=power9`.

| model | formats (tensor census) | correctness gate | pp128 MMA / ref | pp | tg32 MMA / ref | tg |
|---|---|---|--:|--:|--:|--:|
| Ternary-Bonsai 1.7B Q2_0 | Q2_0 (qbit) | token-identical | 416.4 / 8.7 | **47.9×** | 31.7 / 6.8 | **4.6×** |
| qwen3.5 27B Q4_K-M | Q4_K, Q6_K | token-identical | 21.9 / 5.9 | **3.7×** | 3.80 / 3.90 | 0.97× |
| Qwen2.5 1.5B IQ2_M | IQ2_S ×137, IQ3_S ×31, Q4_K ×28, Q5_K embed | near-tie PASS (gaps 0.082 ≤ 0.10) | 199.7 / 41.4 | **4.8×** | 35.7 / 32.6 | **1.10×** |
| Qwen2.5 1.5B IQ3_XS | IQ3_XXS ×98, IQ3_S ×70, Q4_K ×28, Q6_K embed | token-identical | 316.4 / 32.7 | **9.7×** | 27.6 / 26.5 | 1.04× |

Kernel level, native: all 14 suites pass against exact float64
references (errors 1e-7…2e-6, same band as emulation); UBSan
(`-fsanitize=undefined -fno-sanitize-recover=all`) clean on the qbit,
q4_K, iq_grid, iq_grid_pp and legacy suites.

Interpretation: prompt processing wins 3.7–48× everywhere. Token
generation wins 4.6× where ggml's fallback is scalar (qbit) and sits
at parity or slightly better elsewhere — parity is the physical
ceiling there, because generation is bandwidth-bound and, post-0015,
both builds read native-width weights through vec_dot at n < 8 (the
MMA build's edge is power10 codegen). See D3.

## 3. Correctness methodology: the three-tier gate

`validate-on-power.sh` builds MMA and no-MMA variants from the same
tree and compares 32 greedy tokens (temp 0, fixed seed and prompt):

1. **Token identity** — strong PASS; validates GER semantics, the
   decode-at-repack decoders, and dispatch end to end in one test.
2. **Near-tie certification** — bit-identity against a different
   rounding tree is unachievable by construction, so a divergence is
   acceptable iff the road-not-taken token sits within 0.10 logprob
   of the chosen one in both builds.
3. **Codegen-envelope control** (added this session, defect D2) — on
   a tier-2 failure, build the same tree at `-mcpu=power8` (no MMA,
   no repo code) and measure drift between the two MMA-free builds.
   FAIL is confirmed only if MMA-vs-reference drift exceeds twice
   that envelope; if two builds containing none of this repository's
   code behave the same way, the divergence cannot indict the
   kernels.

Tier 3 exists because the data demanded it. The evidence, measured on
the IQ2_M divergence (64-token captures, median |Δlogprob| over the
shared prefix):

| pair | diverges at | median drift |
|---|---|--:|
| MMA ↔ power9 reference | token 5 | 0.122 |
| power9 ↔ power8 (no repo code) | token 7 | 0.110 |
| MMA ↔ power8 | token 5 | 0.201 |

Two vanilla ggml builds flip a token against each other and carry the
same drift as the MMA build. A pure-Q5_K probe model reproduced the
same pattern (drift 0.135 median) with zero IQ tensors in it,
independently clearing the IQ decoders of suspicion. Passing runs
carry the same drift (IQ3_XS: 0.113) — they simply met no near-tie.
The envelope is a property of compiler codegen variation amplified
through 28 transformer layers, not of these kernels.

What tier 3 does **not** establish by itself: independent
verification of the grid decoders against ggml's `dequantize_row`
family. That gap is now closed separately by the N1 cross-check
(`make test-xcheck`), which proves the decoders bit-exact at unit
level.

## 4. Defects found, fixed, verified

### D1 — Validation harness could forge a PASS (severity: critical)

- **Symptom:** temp-0 gate PASSed for every model, including runs
  later shown to produce different tokens.
- **Root cause:** the gate compares outputs of two servers started on
  a fixed port. An orphaned llama-server from the previous day's
  session held that port; each new server loaded its model, failed to
  bind, and died, while the health check and both completion requests
  were answered by the squatter — two identical answers from one
  unrelated server. Every gate verdict from Jul 21 17:08 to discovery
  was void, including this repo's first published hardware PASS.
- **Fix:** squatted port is a preflight FATAL; after the health check
  the script proves its own child is the answerer and (where `/props`
  exposes `model_path`) that it serves the model under test; each
  kill is verified to release the port, escalating to `kill -9`; the
  MMA-presence check is an objdump GER count (fatal at zero) instead
  of a banner grep that could not fire.
- **Verified:** all four models re-gated clean on an idle box; the
  guards demonstrably execute in every subsequent log.

### D2 — Gate tolerance overclaimed FAIL (severity: high)

- **Symptom:** IQ2_M and a pure-Q5_K probe reported "real numerical
  defect, do NOT deploy" at divergence gaps of 0.125–0.28 logprob.
- **Root cause:** the 0.10 near-tie tolerance sits below the
  machine's own cross-codegen envelope (§3): rounding drift shared by
  MMA-free builds was being attributed to the kernels.
- **Fix:** tier 3 (§3). Verdicts are now "PASS (codegen envelope)" or
  "FAIL (control-confirmed)", with the measured numbers inline.
- **Verified:** IQ2_M re-run under the three-tier gate certifies at
  tier 3 (0.114 vs 0.072 envelope, control flips a token); after D3's
  fix the same model passes at tier 2 outright.

### D3 — Token generation 2.8–7.8× slower than the no-MMA reference (severity: high)

- **Symptom:** tg32 at 8t: Q4_K-27B 1.41 vs 3.90 t/s; IQ2_M 4.18 vs
  32.6; IQ3_XS 4.28 vs 26.5. Diagnostic signature: identical MMA tg
  (~1.1/2.2/4.2 t/s at 2/4/8t) for both IQ models regardless of size
  — linear thread scaling far below bandwidth ⇒ compute-bound
  per-token work, not memory-bound inference.
- **Root cause (two stacked defects, isolated by one experiment
  each):** (a) the pack cache's 128 admission-only slots lose to the
  ~197 quantized tensors of a 28-layer model — a third of the weights
  re-decoded and re-packed on every call, invisible under qemu where
  a repack and a GER cost the same; (b) even at full hit rate the
  n=1 path reads int8-expanded packs (1.8–3× native weight bytes)
  and loses the bandwidth race to vec_dot by construction.
- **Fix (patch 0015):** slots 128 → 1024; below one column tile
  (`n < 8`) the dispatch declines and vec_dot keeps generation —
  the tree's own Q8_0/Q4_0 precedent from patch 0011.
- **Verified:** slot change alone doubled IQ2_M tg (4.18 → 8.45,
  thrash quantified); full fix restores the table in §2 — tg at
  parity or better everywhere, pp improved up to +38% (the overflow
  tensors were re-decoding per pp call too). Temp-0 gate re-passed on
  the fixed build. Patch verified to apply after 0014 and reproduce
  the silicon-validated tree byte-for-byte.

## 5. IQGRID_PINGPONG A/B

The accumulator ping-pong variant (patch 0010) was benched against
standard on both IQ models (xxmfacc 165 vs 197 — the second
accumulator set is really in the binary). Ping-pong: +3–4% tg on
IQ3_XS, −1–5% pp, ±1% on IQ2_M. DESIGN.md's counted 3% static cost
shows up in pp almost exactly. **Standard stays the default.**
Post-0015 the grid kernels no longer run at n < 8, so only the pp
side of this A/B is live; revisit under N4.

## 6. Limitations

- ~~End-to-end format coverage is partial~~ Closed by the N2 sweep
  (§8): every accelerated format is silicon-verified end-to-end
  except MXFP4 and NVFP4, which cannot be produced by requantization
  in this fork and remain kernel-level + N1-decode-verified.
- ~~Decoders not independently cross-checked~~ Closed by N1: all 13
  grid/ternary/codebook decoders are now bit-exact against ggml's
  `dequantize_row_*` at unit level (`make test-xcheck`).
- **One machine, one compiler.** The envelope constants (0.10
  tolerance, 2× control factor) are calibrated to this LPAR/GCC-11.5
  pair; a second machine should confirm they generalize. Power11 is
  untested. SMT beyond 8 threads is unmeasured.
- **tg parity is a ceiling, not a shortfall:** generation is
  bandwidth-bound and both paths now read native-width weights; an
  MMA tg *win* is only available where the fallback is compute-bound
  (as with qbit's scalar fallback, won 4.6×).

## 7. Next steps

| # | action | why | done when |
|---|---|---|---|
| N1 | ~~Exhaustive decoder cross-check~~ **DONE 2026-07-22**: `make test-xcheck` — every grid/sign/value table memcmp'd against ggml's originals (vendored verbatim from the pinned fork, `scripts/extract-xcheck-ref.py`); block dequant bit-exact across all 13 formats (~7.6M elements, maxrel = 0); mutation test proves the gate can fail | Closes the last open correctness row in REVIEW.md | ✅ In `make test`; all formats bit-exact, run natively on POWER10 |
| N2 | ~~End-to-end gates for uncovered families~~ **DONE 2026-07-22**: 14 requant probes, all PASS across tiers 1–3, zero kernel defects (§8); MXFP4/NVFP4 unreachable by requant, documented | §6 coverage gap | ✅ §8 table |
| N3 | ~~Cache hardening~~ **DONE 2026-07-22** (patch 0016): slot table grows — slot count can never bind; refusals counted and loud; stats API; canonical copy lives in src/ppc_pack_cache.cpp with xcheck_cache driving the N > capacity regime in `make test` | D3's lesson — a test, not a review, catches the third sequel | ✅ Test in `make test`; silence is structurally impossible |
| N4 | ~~Measure, then decide on low-bit MMA GEMV~~ **DONE 2026-07-22**: triad ceiling 133 GB/s vs per-format effective bandwidth (§9). Decision: build direct GEMV for IQ1_S/M + TQ1_0 (5–10x plausible), consider IQ2/IQ3/TQ2_0 (2–4x), skip the rest | build nothing on a hunch | ✅ §9 table; GEMV implementation is the one open engineering item |
| N5 | ~~Harness polish~~ **DONE 2026-07-22**: exit 0/1/2 mirrors the gate; per-run artifact dirs under validation-runs/; `Repo:` falls back to a source sha without `.git`; tier 3 can now certify token-0 divergences from position-0 candidate drift | CI-usability + D2's diagnosis friction | ✅ Smoke-tested on silicon (exit 0, artifacts archived) |
| N6 | Second machine: Power11 (or a differently-entitled POWER10 LPAR) full protocol run | §6 single-machine limitation; README solicits reports | A second report alongside this one |

## 8. Format-coverage sweep (N2, 2026-07-22 evening)

Every format family the fork can produce was driven through the
three-tier gate as a requant probe (quality irrelevant — the gate
tests determinism). Sources: the 1.5B IQ3_XS for well-conditioned
targets; the Bonsai F16 with a fresh imatrix for the fussy low-bit
ones, after `--pure` requants of a requant crashed IQ quantization
itself (`ggml_quantize_chunk` abort — a probe-construction lesson,
not a kernel issue).

| probe | gate verdict | tier |
|---|---|---|
| Q2_K | PASS token-identical | 1 |
| Q3_K_M (Q3_K + mixture) | PASS | 3 (envelope) |
| Q4_1 | PASS | 3 (envelope) |
| Q5_0, Q5_1 | PASS token-identical | 1 |
| IQ4_NL, IQ4_XS | PASS | 3 (envelope) |
| TQ1_0, TQ2_0 | PASS | 3 (envelope) |
| IQ1_S | PASS near-tie (0.0000 gap) | 2 |
| IQ1_M | PASS near-tie (0.0000 gap) | 2 |
| IQ2_XXS, IQ2_XS | PASS token-identical | 1 |
| Q8_0 (via MXFP4_MOE fallback) | PASS near-tie (0.014) | 2 |

Combined with §2, every quantized format this project accelerates is
now silicon-verified end-to-end **except MXFP4 and NVFP4**, which the
fork cannot produce by requantization (MXFP4_MOE falls back to Q8_0
on dense models; NVFP4 has no quantize target). Those two remain
covered at kernel level and by the N1 bit-exact decoder cross-check;
an externally converted model would close them. Zero kernel defects
in the sweep. Probe GGUFs, censuses, and per-run artifacts are
archived on the test machine (`~/models/probe`, `~/n2-reports`).

## 9. Generation headroom: the GEMV decision (N4)

Measured ceiling (pthread triad, best of 5): **74 / 99 / 133 GB/s at
2/4/8 threads**. Against the reference build's effective tg
bandwidth (model bytes x tg rate, 8 threads):

| formats | eff. GB/s | % of ceiling | GEMV verdict |
|---|--:|--:|---|
| IQ1_M 2.6, TQ1_0 3.4 | 2.6–3.4 | 2–3% | **build it** — vec_dot is compute-starved ~40x; 5–10x tg plausible |
| IQ2/IQ3 grids | ~18 | 14% | worthwhile — 2–4x, Amdahl-bounded at 1.5B scale |
| TQ2_0 | 21 | 16% | worthwhile, same bound |
| Q2_K–Q5_1, IQ4_NL/XS (1.5B) | 29–51 | 25–40% | skip — per-token overhead dominates at this scale |
| Q4_K 27B | ~65 | 49% | marginal — ≤1.5x even at ceiling; large models only |

The 0015 dispatch fallback stays the right default everywhere (it
never loses; post-guard MMA tg is within ±5% of reference on every
probe, +16% on TQ2_0 from power10 codegen alone). The follow-up with
real payoff is a qbit-style direct GEMV for the IQ1/TQ1 family
specifically — the same design that already wins 4.6x on Q2_0.

## 10. Reproducing

```
scripts/build-bonsai-power.sh
scripts/validate-on-power.sh /path/to/model.gguf
```

~20 minutes per model on this partition, most of it the two builds;
the power8 control adds a third build only when a divergence needs
certifying. A squatted port or a GER-free "MMA" build aborts loudly.
Archived per-model reports and raw gate JSONs from this validation
live on the test machine under `~/report-clean-*.md` and
`~/iq2m-fail/`.
