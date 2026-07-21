# Review guide and verification matrix

This document exists so a skeptical senior reviewer can spend their
skepticism efficiently. It states exactly what has been
machine-verified and by what method, what has not been verified and
why, and what this project got wrong along the way — because a repo
that has never found a defect in itself hasn't looked.

## Verification matrix

| Property | Method | Status |
|---|---|---|
| MMA instruction semantics (`xvi8ger4pp` operand signedness/layout) | Empirical probes under `qemu -cpu power10` | Verified (emulation) |
| Numerical correctness, all 28 format pairings | Exact float64 references, random data, ragged/multi-slab/n=1 shapes; 13 self-checking suites, `make test` | Verified (max normalized err ~4e-6, at float-reference rounding level) |
| Warnings hygiene | `-Wall -Wextra -Werror`, GCC 14 | Clean |
| Undefined behavior | `-fsanitize=undefined -fno-sanitize-recover=all` under qemu (v4, q4_K, iq_grid, legacy suites) | Clean (one alignment-model finding fixed repo-wide via `load16u`) |
| `aligned_alloc` C11 conformance | Pack-size functions round to alignment | Fixed |
| Allocation-failure behavior (in-tree drivers) | `GGML_ABORT` instead of silent skip | Fixed (patch 0006) |
| Out-of-bounds loads at block-array tails | Manual audit of every load against struct extents; one real OOB found and fixed in the v1 kernels | Audited |
| Integer overflow in GER accumulation | Bounds analysis: max chunk dot ≪ 2³¹ for every format | Verified by analysis |
| Patch-series integrity (0001–0014) | Sequential `git apply` gate on a pristine checkout of the pinned base; result diffed against the build-verified tree | Verified, byte-identical |
| Fork integration compiles + links | ppc64le cross-build, GCC 14, all 10 kernel TUs in ggml-cpu; `llama-cli` executes under qemu | Verified |
| End-to-end inference numerics through patched dispatch | Requires model weights + hardware | **NOT verified** — DEPLOY.md step 5 is mandatory before production |
| Grid/ternary/codebook decoders vs ggml's dequantization | Test refs share the decoders (consistency only); decoders are line-by-line ports | **NOT independently verified** — covered by the same step 5 |
| Performance on silicon | qemu is an instruction-count proxy; static accounting is exact but prices nothing | **NOT verified** — every performance statement is labeled with its instrument |

## Defects this project found in itself

Disclosed here deliberately; each one changed a process, not just a line.

- **Patch-series overlap (rollout-blocking).** Patches 0007/0008 were
  once generated from the same baseline and could not apply in
  sequence — the one-command build script was broken and nothing
  caught it. Fixed by regenerating from proper per-patch baselines,
  and by adding the gate that should have existed from day one: every
  push now proves the full series applies on a pristine tree with a
  byte-identical result.
- **Test-aggregation hole.** A Makefile editing error pointed the
  iq_grid test target at the legacy source: `make test` ran the legacy
  suite twice and the grid suite not at all. The grid suite had only
  ever run via direct compiler invocations (where it passed, including
  under UBSan) — correctness was never wrong, but the matrix a
  reviewer would run didn't cover what it claimed. Fixed; targets now
  declare their true sources.
- **Cache design flaw.** The pack cache's original round-robin
  eviction would, for any working set above the cap, re-pack every
  tensor every token — worse than no cache. Replaced with
  admission-without-eviction, making the cache's effect monotonic.
- **Silent-failure path.** Early one-shot drivers returned quietly on
  allocation failure while the dispatch reported success. Now aborts
  loudly (patch 0006).
- Two git recovery commits from process mistakes remain in history
  with honest messages; the test matrix, not the history, is the
  arbiter of code state.

## Pre-rollout audit (silicon-architect perspective)

**Fixed in patch 0008 (rollout hardening):** cache thrash → admission
policy; stale packs across model reload → FNV-1a content fingerprint
in the key plus `ppc_apack_cache_clear()`; nth-fold per-thread
activation-pack duplication → column-partitioned packing on cache hits
(total pack work now equals exactly one pack).

**Known, quantified, awaiting silicon:**
- 16-deep chunk kernels run 13.5 non-GER instructions per GER vs 7.6
  for the 32-deep kernels — the per-16-scale formats' inherent fixup
  tax, forced by their scale granularity. Hardware prices it; we only
  counted it.
- First touch of a tensor packs the full matrix on one thread while
  peers wait — cold-start latency, amortized over the model lifetime.
- `xvi8ger4pp` semantics were established under qemu; silicon
  agreement is expected, and confirming it is precisely what the
  temperature-0 check is for.
- fp32 output accumulation matches ggml's own vec_dot baseline; no
  extended-precision claim is made for very large k.
- n = 1 takes the no-packing GEMV path for `Q1_0`/`Q2_0` only; other
  formats run padded GEMM tiles at n = 1.
- The `lxvp` vector-pair experiment (DESIGN.md) regressed the static
  count via pair-split register moves that are near-free on real
  hardware — the one case where our own instrument is suspect, filed
  as hardware-decidable with the diff recorded.

## Engineering debt, stated plainly

- The pack cache delivers load-time-repack *behavior* at the dispatch
  layer; the eventual correct home is ggml's `repack.cpp` at model
  load. The packed APIs were designed for that move.
- Cache assumptions (weight immutability, pointer+fingerprint keying)
  are sound for llama.cpp inference and stated in the source header;
  training-style mutation would need explicit invalidation.
- The 16×8/8-acc vs 8×8/4-acc tile question is a register-pressure vs
  chain-parallelism tradeoff only hardware can settle.

## Suggested review order

1. `docs/DESIGN.md` — the algebra, the microarchitectural argument,
   and the measured decisions (including the ones that lost).
2. `src/reference/qbit_ppc_mma_v3.cpp` — the core formulation in its
   smallest complete form.
3. `src/qbit_ppc_mma_v4.cpp` — the production API shape.
4. One K-quant (`q4_k`) and one signed-codebook kernel (`iq4`) — the
   two operand orientations.
5. `patches/` in order — the series is gated, so what you read is what
   builds.

Reproduce everything: `scripts/setup_cross.sh && make CXX=powerpc64le-linux-gnu-g++-14 test` — about twelve seconds to re-verify all 26 formats.
