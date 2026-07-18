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

## v4: production API and GEMV

v4 restructures the same kernels behind a production-shaped API,
retiring the debt items above: `qbit_repack_q1/q2()` performs the
bit-unpack + MMA-layout transform once for the whole weight matrix (in
ggml this runs at model load in `repack.cpp`); `qbit_pack_b()` packs
activations once, shared across threads; `qbit_gemm_packed()` consumes
both. For token generation (n <= 2), `qbit_gemv_q1/q2()` switch to a
select-and-sum path that reads the RAW 1/2-bit weights — GEMV is
memory-bandwidth-bound, and raw bits are 8×/4× less traffic than
unpacked int8. It exploits the same algebra with masks instead of
multiplies (`2·Σ_{bit=1}y − Σy`; ternary via the two code-bit masks),
with per-chunk activation sums shared across all rows: the inner loop
contains no multiply instructions at all. Under the qemu
instruction-count proxy the GEMV path is already ~1.6× cheaper than the
packed GEMM at n = 1 (m = 4096, k = 2048), before its bandwidth
advantage — invisible to emulation — applies.

## Q4_K × Q8_K: extending to K-quants

`Q4_K` (superblocks of 256 = 8 sub-blocks of 32; value = d·sc_sub·q −
dmin·m_sub, unsigned nibbles) maps onto the identical architecture:

- q ∈ 0..15 is natively unsigned → unsigned GER operand, exactly like
  the v3 code formulation; no flips, no per-row corrections;
- the per-sub-block scale d·sc replaces v3's per-block dA in the chunk
  fixup — structurally identical, indexed per chunk;
- the mins term **is** the separable correction:
  `corr(i,j) = Σ_sub (dmin_i·m_sub)·(dB_j·S_sub(j))`, and S_sub comes
  free from `Q8_K`'s precomputed `bsums`. dB·S is folded per (col,
  chunk) in the activation pack; the kernel applies one `vec_nmsub` per
  (col, rowgroup, chunk).

The nibble unpack (and 0xF / shift 4) is cheaper than the 1/2-bit
unpacks. Verified against an exact double reference across the same
shape matrix (max normalized error ~2e-6). The full standard family is now implemented on the same recipe:
Q5_K adds the 5th bit from `qh` to the Q4_K nibble unpack; Q6_K and
Q2_K have per-16 scales and use a 16-deep chunk variant (4 depth steps
per accumulator round — the same reason the AVX/NEON paths work per
16). Q6_K's offset form (t = q − 32, same d·sc factor on both terms)
lets its correction fold into the main FMA as a `vec_msub`; Q2_K keeps
the two-term mins fixup. All are verified against exact double
references built directly from the ggml dequantization semantics.

**Q3_K** completes the standard family: q′ = code | (hbit << 2) ∈ 0..7
with t = q′ − 4 shares the d·(sc−32) factor on both terms, so it takes
the Q6_K folded-correction form with TS = 4·dB·bsums; the 6-bit scale
packing is decoded once at repack with ggml's aux-mask trick.

**IQ4_NL / IQ4_XS: the codebook technique.** These formats map nibbles
through a signed 16-entry int8 codebook. The lookup is a single
`vec_perm` per 16 codes — but signed weight values cannot ride the
unsigned GER operand, so the operand orientation flips back to the
v1/v2 scheme: codebook values on the signed operand, activations
XOR-0x80 flipped, and the bias correction uses per-(row, chunk)
codebook-value sums W computed at repack, pre-folded as 128·W·scale so
the fixup is one `vec_msub` + one `vec_madd`. These kernels use an 8×8
tile on 4 accumulators (fully register-resident — one side of the
tradeoff flagged below, and the natural fit for the flipped
orientation's write-back). The same technique extends to the wider IQ
family (IQ2/IQ3 grids are larger codebooks with sign masks); those
remain future work.

**Legacy quants (Q4_1/Q5_0/Q5_1).** All unsigned codes, v3
orientation. Q5_0 is offset form (t = q − 16) with the correction
folded as usual. Q4_1/Q5_1 are affine (value = d·q + m) against Q8_1 —
whose block stores s = dB·Σy precomputed by ggml's quantizer, so the
min term reduces to `fin += mA · s` — one vec_madd with no sum computed
anywhere in this project's code.

**Grid codebooks and ggml ternary: decode-at-repack.** The packed-API
architecture makes exotic formats cheap to support: weight decode runs
once at repack, so TQ2_0/TQ1_0 (BitNet, incl. the base-243 digit
extraction), IQ2_XXS/IQ3_XXS/IQ3_S (grid tables + sign masks) and
IQ1_S all reduce to "signed int8 codes + one scale per 32-chunk" and
share a single signed-operand 8×8 kernel. IQ1_S's fractional delta is
made exact with codes = 8·g ± 1 and scale = dl/8. The decoders are
direct scalar ports of ggml's dequantize_row semantics; note the test
references run *through the same decoders* (they verify GEMM/pack
consistency, not decoder-vs-ggml — the temperature-0 check in
DEPLOY.md is the decoder's independent verification). Still deferred:
IQ2_XS, IQ2_S, IQ1_M, whose per-16 scales need a 16-deep signed-chunk
kernel variant.

**Register-pressure note (K-quants).** Static analysis of the K-quant
hot loops shows ~50–85 stack vector ops per chunk iteration, versus
~0–8 for v3. This is structural, not a codegen accident: the fixup
keeps fin (32 vectors) live while 8 accumulators alias VSRs 0–31, and
per-chunk scale vectors push past the 64-register file. The kernels
stage all accumulator disassembly to a stack buffer before the fixup,
converting scattered spills into a predictable store/load stream. The
untested alternative is an 8×8 tile for the per-16 formats (fits the
register file entirely, at half the independent GER chains) — a
tradeoff only hardware can settle; both variants are worth
benchmarking on silicon.

## Known limitations / future work

- (retired in v4) Weight packing moved behind a one-time repack API;
  activation packing shared. Remaining: land these inside ggml itself.
- K tails (k not a multiple of 128) are unsupported, matching the
  formats' block size.
- (retired in v4) GEMV path implemented; needs hardware benchmarking
  against the GEMM path to set the crossover point.
