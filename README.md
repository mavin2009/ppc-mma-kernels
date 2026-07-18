# ppc-mma-kernels

POWER10/POWER11 MMA (Matrix-Multiply Assist) GEMM kernels for low-bit
quantized LLM inference, targeting the [PrismML llama.cpp fork's](https://github.com/PrismML-Eng/llama.cpp)
1-bit (`Q1_0`) and ternary (`Q2_0`) weight formats against `Q8_0`-quantized
activations.

Mainline llama.cpp accelerates `Q4_0`/`Q8_0` matmuls on Power10 via MMA
(`tinyBLAS_Q0_PPC`), but the 1-bit/ternary formats fall back to scalar
code — throwing away Power's main trick on exactly the layers that
dominate compute. These kernels close that gap.

## Kernels

| File | What it is |
|---|---|
| `src/reference/q1_0_ppc_mma.cpp` | (reference) v1: 4x4 tile, single accumulator, `Q1_0` × `Q8_0` |
| `src/reference/q2_0_ppc_mma.cpp` | (reference) v1: same structure for ternary `Q2_0` × `Q8_0` |
| `src/reference/q1_0_ppc_mma_v2.cpp` | (reference) v2: hoisted packing, K-slab blocking, 8x8 tile on 4 accumulators |
| `src/reference/qbit_ppc_mma_v3.cpp` | (reference) v3: unified `Q1_0`+`Q2_0`, unsigned-code formulation with separable correction, 16x8 tile on all 8 accumulators |
| `src/qbit_ppc_mma_v4.cpp` | v4: production API — one-time weight repack, shared activation pack, raw-weight GEMV path for token generation |
| `src/q4_k_ppc_mma.cpp` | `Q4_K` × `Q8_K` (first K-quant kernel): unsigned nibbles + separable mins correction via `bsums`, same 16x8/8-acc architecture |
| `src/q5_k_ppc_mma.cpp` | `Q5_K` × `Q8_K`: Q4_K structure + 5th bit from `qh` |
| `src/q6_k_ppc_mma.cpp` | `Q6_K` × `Q8_K`: 16-deep chunks (per-16 scales); offset correction folds into the main FMA |
| `src/q2_k_ppc_mma.cpp` | `Q2_K` × `Q8_K`: 16-deep chunks, per-16 scale+min pairs |
| `src/q3_k_ppc_mma.cpp` | `Q3_K` × `Q8_K`: 16-deep chunks; q′ = code∣(hbit≪2), offset correction folded (TS = 4·dB·bsums) |
| `src/legacy_ppc_mma.cpp` | `Q4_1`/`Q5_1` × `Q8_1` (min term = one FMA using Q8_1's precomputed `s`), `Q5_0` × `Q8_0` (folded offset) |
| `src/iq_grid_ppc_mma.cpp` | `TQ1_0`/`TQ2_0` (BitNet ternary), `IQ2_XXS`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`, plus per-16-scale `IQ2_XS`, `IQ2_S`, `IQ1_M` × `Q8_K` and `NVFP4` × `Q8_0`: decode-at-repack to signed int8, shared signed kernels (32- and 16-deep chunk variants) |
| `src/iq4_ppc_mma.cpp` | `IQ4_NL`/`MXFP4` × `Q8_0` and `IQ4_XS` × `Q8_K`: signed 16-entry codebook via one `vec_perm`, v2-orientation with pre-folded −128·W correction, 8×8/4-acc |

See [docs/DESIGN.md](docs/DESIGN.md) for the derivation and the
microarchitectural rationale, [docs/INTEGRATION.md](docs/INTEGRATION.md)
for how the wiring works, and [docs/DEPLOY.md](docs/DEPLOY.md) for the
end-to-end path to running Bonsai on a Power machine.

## Deploy Bonsai on Power

```sh
./scripts/build-bonsai-power.sh
```

clones the PrismML fork at a pinned commit, applies both integration patches -- Q1_0/Q2_0 and the full K-quant
family (verified: the patched fork cross-compiles for ppc64le with all
five kernel TUs and `llama-cli` runs under qemu-power10) --
and builds `llama-cli` / `llama-server` / `llama-bench`. Details and
model download in [docs/DEPLOY.md](docs/DEPLOY.md).

## Virtualized performance snapshot

Measured qemu-emulation numbers — same-workload kernel comparisons,
suite runtimes, and exact static instruction accounting — live in
[docs/BENCHMARKS-QEMU.md](docs/BENCHMARKS-QEMU.md), with an explicit
statement of what emulated numbers can and cannot show. No absolute
hardware claims are made anywhere in this repo.

## For reviewers

See [docs/REVIEW.md](docs/REVIEW.md) for the verification matrix: what
is machine-verified (numerics vs float64 references, -Werror, UBSan,
cross-build of all six integration patches) and what still requires
hardware (end-to-end inference, all performance claims).

## Correctness

Every kernel ships with a self-checking test (`-DQ*_TEST`) that compares
against an exact double-precision reference over random data, covering
ragged tile edges, single-column (token generation) shapes, multi-slab K,
and the full ternary code range. Verified via cross-compilation and
`qemu-ppc64le -cpu power10` emulation; max normalized error is at the
same level as the scalar float reference (~1e-6).

## Quick start (cross + qemu, e.g. on x86 Ubuntu 24.04)

```sh
sudo apt-get install g++-14-powerpc64le-linux-gnu qemu-user
make CROSS=powerpc64le-linux-gnu- CXX=powerpc64le-linux-gnu-g++-14 test
```

On real Power10/Power11 hardware:

```sh
make CROSS= QEMU= test bench
```

## Status / honest caveats

- Correctness is machine-verified (qemu). **Performance is not**: qemu
  prices every instruction equally, so it can serve only as a rough
  dynamic-instruction-count proxy. v2 measured ~23% fewer emulated
  instructions than v1; v3's advantages (8 independent accumulator
  chains, prefetch, halved per-output companion work) are exactly the
  ones emulation cannot see. Benchmark on silicon before trusting any
  ranking between v2 and v3.
- The GEMM-side packing here is per-thread; in a real ggml integration
  the weight pack belongs in `repack.cpp` at model load and the
  activation pack next to activation quantization.

## License

MIT
