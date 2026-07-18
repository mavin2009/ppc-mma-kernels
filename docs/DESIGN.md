# Design notes

## Problem

The PrismML llama.cpp fork stores 1-bit weights (`Q1_0`: 128 weights per
block, one FP16 scale, value = 2·bit − 1) and ternary weights (`Q2_0`:
2-bit codes, value = code − 1). Its CPU paths for these types are
generic C. On Power10, ggml's MMA acceleration (`tinyBLAS_Q0_PPC`)
covers only `Q4_0`/`Q8_0`, so the low-bit matmuls — essentially all the
language-model compute — run scalar.

The activations are quantized to `Q8_0` (32 int8 + FP16 scale), and one
weight block spans exactly four activation blocks, so integer
accumulation can run 32 elements deep before a float scale must be
applied.

## The MMA primitive

`xvi8ger4pp(acc, va, vb)` performs a rank-4 outer-product update of a
4×4 int32 accumulator: `acc[i][j] += Σ_{b=0..3} va[4i+b] · vb[4j+b]`,
where **va is treated as signed int8 and vb as unsigned** (verified
empirically under `qemu -cpu power10` in this project's development;
operand layout is row-major 4 lanes × 4 depth bytes).

## v1/v2 formulation (signed weights, flipped activations)

Mirror `tinyBLAS_Q0_PPC`: signed weight values t on va, activations
XORed with 0x80 (y + 128, unsigned) on vb, then correct
`Σ t(y+128) = Σ ty + 128·Σt` by subtracting `128·Σt` per (row, chunk).
For ±1 weights `Σt = 2·popcount − 32`, so the correction is nearly free
at pack time.

v2 adds the standard GEMM engineering: packing hoisted out of the
compute loops with K blocked into 2048-element slabs (packed A slab
sized against the 32 KiB L1D), an 8×8 microkernel on 4 accumulators,
and scales converted from FP16 once at pack time.

## v3 formulation (unsigned codes, separable correction)

Both formats decompose as **t = α·code − 1** with α = 2 (`Q1_0`) or
α = 1 (`Q2_0`), giving

```
Σ t·y = α·Σ code·y − Σ y
```

Putting the raw unsigned codes on vb and untouched signed activations on
va eliminates the XOR flip, the per-row corrections, and the per-chunk
correction adds. The `−Σy` term is *separable in the scales*:

```
corr(i,j) = Σ_blk dA(i,blk) · E(j,blk),   E(j,blk) = Σ_{chunks c} dB(j,c) · Σy_c
```

a rank-1-per-block outer product applied once per tile at slab end
instead of on every chunk of the inner loop. E is computed during the
activation pack, whose cost is amortized over all weight rows — the
favorable side, since output dimension ≫ token count in LLM layers. The
unified formulation is also why one kernel serves both formats
(template parameter α plus a per-format unpack).

Side effect: with operands swapped, accumulator rows correspond to
activation columns, so the finished 16×8 tile stores to column-major C
with contiguous unaligned vector stores instead of lane extraction.

## Microarchitecture

Power10/Power11 cores have one MMA engine per SMT4 core sustaining one
512-bit accumulator GER per cycle; `xvi8ger4pp` is 64 int8 MACs, so the
GER stream is the throughput floor and all companion work (unpack,
transpose, scale fixup) must hide in the four 128-bit VSU slices running
alongside. GER results carry multi-cycle latency into the same
accumulator, so consecutive updates of one accumulator serialize;
saturating the engine requires many independent accumulator chains. v3
therefore runs a 16×8 tile on **all 8 accumulators**. ISA 3.1
accumulators ACC0–7 alias VSRs 0–31, leaving VSRs 32–63 for the six
operand vectors per depth step plus loop state — which is why 16×8 fits
and a larger tile would spill.

Static analysis of GCC 14 output (`-O3 -mcpu=power10`) for the hot
loops:

| | GERs per chunk-iter | non-GER insns | non-GER per output |
|---|---|---|---|
| v2 (8×8, 4 acc) | 32 | ~244 | 3.81 |
| v3 (16×8, 8 acc) | 64 | ~277 | 2.16 |

v2 is companion-work-bound (~61 VSU cycle-equivalents against 32 GER
cycles); v3 roughly halves companion work per output and doubles the
independent chains, moving the kernel to near MMA-bound. GCC emits the
64 GERs straight-line with the previous chunk's fixup interleaved and
essentially no spills.

Power11 retains the same MMA ISA and per-core engine (more cores,
higher clocks); build with `-mcpu=power10` and the same binary is the
right one there.

## Verification methodology

All kernels are compared against an exact double-precision reference
over random data (and against the fork's float-ordering scalar
reference), across shapes covering ragged tile edges, n = 1 token
generation, multi-slab K, and the full ternary code range including the
+2 code. Emulation caveat: qemu validates **correctness only**; its
timings are at best a dynamic-instruction-count proxy (v2 measured ~23%
below v1 on it), and are structurally blind to v3's latency-hiding and
prefetch benefits. Rank v2 vs v3 on silicon.

## Known limitations / future work

- Weight packing should move to ggml's `repack.cpp` (once, at model
  load); activation packing should be shared across threads next to
  activation quantization rather than duplicated per thread.
- K tails (k not a multiple of 128) are unsupported, matching the
  formats' block size.
- An alternative m = 1 (token generation) path using select/`vsum4s`
  (no multiplies, exploiting ±1 weights) may beat outer-product MMA for
  single-column GEMV; unbenchmarked.
