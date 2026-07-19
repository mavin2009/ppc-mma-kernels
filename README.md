# ppc-mma-kernels

POWER10/POWER11 MMA (Matrix-Multiply Assist) GEMM kernels for
quantized LLM inference — **every quantized GGUF format, 26 in all**,
from 1-bit and ternary through the K-quants, the IQ grid codebooks,
MXFP4, and NVFP4.

The story in one paragraph: mainline llama.cpp drives Power's MMA unit
for `Q4_0`/`Q8_0` and floats, and lets everything else fall back to
scalar code — surrendering the machine's best trick on precisely the
low-bit formats where Power's memory bandwidth should shine. This repo
started as a fix for the [PrismML fork's](https://github.com/PrismML-Eng/llama.cpp)
1-bit/ternary "Bonsai" formats and kept going until nothing was left
on the scalar path. Along the way it acquired a production packing
API, a tensor-keyed weight-pack cache, a GEMV fast path for token
generation, and a paper trail of measured decisions — including the
experiments that *lost*.

**Status**: `v0.9.0` — emulation-verified, pre-silicon. Correctness is
proven against exact float64 references under qemu-power10; every
performance statement is static analysis or an instruction-count
proxy, clearly labeled. `v1.0.0` is reserved for the first
hardware-validated release. No absolute performance claims appear
anywhere in this repository, which is exactly how you can tell the
relative ones are honest.

**Topics**: `power10` `power11` `ppc64le` `ibm-power` `mma` `matrix-multiply-assist` `gemm` `simd-kernels` `vsx` `altivec` `llama-cpp` `ggml` `gguf` `quantization` `low-bit-inference` `llm-inference` `bitnet` `ternary-networks` `cross-compilation` `qemu`

## Layout

Production kernels live in `src/` — these are what the nine patches in
`patches/` integrate into llama.cpp. The v1→v3 design evolution lives
in `src/reference/` for anyone following the derivation in
[docs/DESIGN.md](docs/DESIGN.md); nothing integrates those, but they
are how the v4 design earned its shape. `make help` lists every build
target; the Makefile header maps each production file to its formats
and its patch.

## Kernels

| File | What it is |
|---|---|
| `src/qbit_ppc_mma_v4.cpp` | `Q1_0`/`Q2_0` (PrismML 1-bit/ternary) × `Q8_0` — production API: one-time weight repack, shared activation pack, raw-weight GEMV for n = 1 |
| `src/q4_k_ppc_mma.cpp` | `Q4_K` × `Q8_K`: unsigned nibbles, separable mins correction via `bsums`, 16×8 tile on all 8 accumulators |
| `src/q5_k_ppc_mma.cpp` | `Q5_K` × `Q8_K`: Q4_K structure + 5th bit from `qh` |
| `src/q6_k_ppc_mma.cpp` | `Q6_K` × `Q8_K`: 16-deep chunks (per-16 scales); offset correction folds into the main FMA |
| `src/q2_k_ppc_mma.cpp` | `Q2_K` × `Q8_K`: 16-deep chunks, per-16 scale+min pairs |
| `src/q3_k_ppc_mma.cpp` | `Q3_K` × `Q8_K`: q′ = code∣(hbit≪2), folded offset (TS = 4·dB·bsums) |
| `src/legacy_ppc_mma.cpp` | `Q4_1`/`Q5_1` × `Q8_1` (the min term is one FMA — `Q8_1` already stores dB·Σy), `Q5_0` × `Q8_0` |
| `src/iq4_ppc_mma.cpp` | `IQ4_NL`/`MXFP4` × `Q8_0`, `IQ4_XS` × `Q8_K`: signed 16-entry codebooks resolved by a single `vec_perm`, flipped operand orientation with pre-folded −128·W correction |
| `src/iq_grid_ppc_mma.cpp` | `TQ1_0`/`TQ2_0` (BitNet), `IQ2_XXS/XS/S`, `IQ3_XXS/S`, `IQ1_S/M` × `Q8_K`, `NVFP4` × `Q8_0`: decode-at-repack to signed int8, two shared signed kernels (32- and 16-deep chunks) |
| `src/reference/…` | the v1 → v2 → v3 lineage, kept because the negative space of a design is part of the design |

## Quick start

Cross + qemu (e.g. x86 Ubuntu 24.04):

```sh
sudo apt-get install g++-14-powerpc64le-linux-gnu qemu-user
make CXX=powerpc64le-linux-gnu-g++-14 test        # 13 suites, ~12 s
```

On real Power10/Power11:

```sh
make CROSS= QEMU= test bench
```

## Deploy Bonsai (or any GGUF model) on Power

```sh
./scripts/build-bonsai-power.sh
```

clones the PrismML fork at a pinned commit, applies all nine patches
in sequence (the series is gated: sequential application on a pristine
checkout reproduces the build-verified tree byte-for-byte), and builds
`llama-cli` / `llama-server` / `llama-bench` with all ten kernel
translation units in place. Model download and the mandatory first
hardware check live in [docs/DEPLOY.md](docs/DEPLOY.md).

## Performance, virtualized — with its epistemics attached

[docs/BENCHMARKS-QEMU.md](docs/BENCHMARKS-QEMU.md) publishes measured
qemu numbers (v3 vs v2, GEMV vs GEMM at n = 1, suite runtimes, exact
per-chunk instruction accounting) under an explicit statement of what
an instruction-count proxy can and cannot show. The micro-optimization
history in [docs/DESIGN.md](docs/DESIGN.md) includes measured negative
results and a full assessment of the remaining POWER10/11 ISA surface
(`lxvp`, masked GERs, bf16 scale folding, stream prefetch) — each with
an adopt / reject / hardware-decidable verdict.

## For reviewers

[docs/REVIEW.md](docs/REVIEW.md) is the verification matrix: what has
been machine-verified and by what method, what has *not* been verified
and why hardware is the only remaining instrument, plus the findings
of a silicon-architect-perspective pre-rollout audit — defects found,
fixed, and disclosed.

## License

MIT
