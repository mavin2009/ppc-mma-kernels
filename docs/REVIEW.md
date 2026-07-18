# Review guide and verification matrix

For reviewers assessing whether these kernels are trustworthy: this
document states exactly what has been machine-verified, by what method,
and what has not.

## Verification matrix

| Property | Method | Status |
|---|---|---|
| MMA instruction semantics (`xvi8ger4pp` operand signedness/layout) | Empirical probes under `qemu -cpu power10` | Verified |
| Numerical correctness, all 26 formats | Exact float64 references, random data, ragged/multi-slab/n=1 shapes; 13 self-checking suites, `make test` | Verified (max normalized err ~4e-6, matching float-reference rounding) |
| Warnings hygiene | `-Wall -Wextra -Werror`, GCC 14 | Clean |
| Undefined behavior | `-fsanitize=undefined -fno-sanitize-recover=all` under qemu (v4, q4_K, iq_grid, legacy suites) | Clean (one alignment-model finding fixed repo-wide via `load16u`) |
| `aligned_alloc` C11 conformance | Pack-size functions round to alignment | Fixed |
| Allocation-failure behavior (in-tree drivers) | `GGML_ABORT` instead of silent skip | Fixed (patch 0006) |
| Out-of-bounds loads at block-array tails | Manual audit of every `vec_xl`/`load16u` against struct extents; one real OOB found and fixed in v1 kernels early on | Audited |
| Integer overflow in GER accumulation | Bounds analysis: max chunk dot ≪ 2^31 for every format | Verified by analysis |
| Fork integration compiles + links (patches 0001–0008) | ppc64le cross-build, GCC 14, all 9 kernel TUs in ggml-cpu; `llama-cli` executes under qemu | Verified |
| End-to-end inference numerics through patched dispatch | Requires model weights + hardware | **NOT verified** — DEPLOY.md step 5 is mandatory before production |
| Grid/ternary decoders vs ggml's dequantization | Test refs share the decoders (consistency only); decoders are line-by-line ports | **NOT independently verified** — covered by the same step 5 |
| Performance on silicon | qemu is an instruction-count proxy only | **NOT verified** — all perf claims are static analysis |

## Known engineering debt (documented, not hidden)

- One-shot dispatch drivers pack per call and duplicate the activation
  pack per thread; the load-time `repack.cpp` integration is designed
  (packed APIs exist) but not landed inside ggml.
- 16×8/8-acc vs 8×8/4-acc tile choice for per-16-scale formats is a
  register-pressure vs chain-parallelism tradeoff only hardware can
  settle (DESIGN.md).
- Pack cache (patch 0007) assumes weight-tensor immutability and
  pointer-identity keying — sound for llama.cpp inference, stated in
  the source; training or reload-at-same-address needs invalidation.
- A Makefile defect found during build-chain review: an editing error
  had pointed the iq_grid test target at the legacy source, so `make
  test` ran the legacy suite twice and the grid suite not at all (the
  grid suite had only ever run via direct compiler invocations, where
  it passed, including under UBSan). Fixed; `make test` now runs all
  13 real suites. Lesson encoded: test targets now list their true
  source dependencies individually.
- Git history contains two recovery commits from process mistakes
  (documented in their messages); code state is what the test matrix
  verifies.

## Suggested review order

1. `docs/DESIGN.md` — the algebra and the microarchitectural argument.
2. `src/reference/qbit_ppc_mma_v3.cpp` — the core formulation, smallest complete kernel.
3. `src/qbit_ppc_mma_v4.cpp` — the production API shape.
4. One K-quant (`q4_k`) and one signed-codebook kernel (`iq4`) for the
   two operand orientations.
5. `patches/` in order — each is independently buildable on top of the
   previous.

Reproduce everything: `scripts/setup_cross.sh && make CXX=powerpc64le-linux-gnu-g++-14 test`.
