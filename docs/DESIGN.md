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
DEPLOY.md is the decoder's independent verification). A 16-deep signed-chunk
variant (4 GER steps per accumulator round) now covers the per-16-scale
stragglers IQ2_XS, IQ2_S, and IQ1_M — the last of these reassembling
its fp16 super-scale from scale-nibble high bits, with per-8 deltas
folded into exact integer codes like IQ1_S. NVFP4 completes the fp4 pair on the
16-deep framework (64-element blocks, four UE4M3-scaled sub-blocks;
the ×0.5 inside the UE4M3 conversion compensates the pre-doubled
codebook). MXFP4 (gpt-oss) rides the
IQ4 codebook kernel: its e2m1 values are exact integers once doubled
({0,1,2,3,4,6,8,12}±), with the E8M0 half-scale supplying the ÷2.

**Tensor-keyed pack cache (patch 0007).** Repacking is deterministic
and depends only on immutable weights, so it belongs at model load.
`ppc_pack_cache.cpp` delivers that behavior at the dispatch layer: the
first call for a tensor packs the whole matrix once (a mutex+condvar
publish protocol lets other threads wait rather than duplicate work);
every later call reuses it. Keyed on (data pointer, m, k, variant);
capacity-bounded (PPC_MMA_PACK_CACHE_MB, default 2048, 0 disables);
on any miss-with-full-cache or allocation failure callers fall back to
the per-call path — never wrong, only slower. The immutability
assumption and its limits are stated in the source header for review.

**Micro-optimization pass (measured, including a negative result).**
Static instruction accounting of the compiled signed kernels (GCC 14,
-O3 -mcpu=power10) per 2048-element-slab chunk iteration: the 32-deep
kernel runs 272 instructions for 32 GERs (7.5 non-GER per GER); the
16-deep kernel runs 228 for 16 (13.25 per GER) — quantifying the
per-16-scale formats' inherent fixup overhead. Of the non-GER work,
~28 stores + a similar share of the 68 loads are GCC's
disassemble-to-stack bounce and `fin` spill under accumulator aliasing
pressure. Attempting to hoist the per-rowgroup scale splats out of the
column-group loop — both as arrays and fully unrolled into named
registers — *regressed* both kernels to 304/256 instructions: the
eight extra live values displace other state into spills, and GCC's
baseline (recomputing splats per column group) is the local optimum.
The change kept from this pass is prefetch widening: each chunk's
packed panel spans two 128B lines and both are now touched (+4
instructions, memory-latency benefit only hardware can price). The
spill traffic itself is an accumulator-file constraint, not a codegen
bug; attacking it means smaller fin footprints (4-row tiles) — a
hardware-measurable tradeoff, not a static one.

**POWER10/11 feature assessment (beyond what the kernels already use).**
Each remaining ISA 3.1 capability, with status:

*lxvp/stxvp vector-pair memory ops* — experiment run on the 32-deep
signed kernel: GER-feed loads became 24 lxvp + spill traffic, but GCC's
pair disassembly inserts xxlor moves, regressing the static count 276 →
308. Unlike the splat-hoist negative result, this proxy verdict is
suspect: xxlor is near-free on silicon and lxvp consumes one dispatch
slot across both LSU halves, so hardware may invert the result. Status:
hardware-decidable; the experiment is a 10-line diff reproducible from
this note.

*pmxvi8ger4pp (prefixed masked GER)* — XMSK/YMSK/PMSK masking could
replace the row-clamping edge handling and depth-tail logic with
masked tiles. Expected perf-neutral (edges are rare); a code-clarity
and correctness-surface win. Status: good first hardware-era cleanup.

*xvbf16ger2pp scale folding* — folding quantization scales into bf16
weights at repack would delete the entire float fixup (no ctf/msub/
madd, no dB application): the accumulator would hold final floats.
Cost: 2-deep GERs double the GER count, activations widen to 16-bit
(2x B traffic), and — decisively — results would no longer be
bit-comparable with ggml's integer-exact path, breaking the
temperature-0 verification story. Status: rejected for numerical
parity; the math is recorded here should a fast-mode variant ever be
wanted.

*xvi16ger2pp* — no fit: activations are already int8 and 4-deep int8
GERs dominate 2-deep int16 for this workload.

*dcbt stream variants (TH field)* — one stream-setup touch per slab
could replace the per-chunk software prefetch and engage the hardware
stream prefetcher explicitly. Status: hardware-tunable; the current
per-chunk dcbt is the conservative baseline.

*dcbz on C tiles* — avoids read-for-ownership on output stores; C
traffic is small relative to packed panels. Status: marginal, untried.

*Prefixed loads/pcrel* — already emitted by GCC under -mcpu=power10.

*POWER11* — same user-level ISA (3.1 family); its gains for these
kernels come from frequency and memory bandwidth, which benefits the
bandwidth-bound GEMV path most. No P11-specific code paths are
warranted.

*OS-level* — the pack cache's large stable allocations are natural
transparent-hugepage candidates on radix MMU systems; worth checking
THP engagement during hardware bring-up.

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

## Landing the pack cache inside ggml's repack layer (design; deferred with reasons)

The tensor-keyed pack cache lives at the dispatch layer
(`llamafile_sgemm`) because that is where it could be built without
touching ggml's buffer machinery. Its eventual home is the
`repack.cpp` buffer-type mechanism ggml already uses on aarch64:
weights repacked once at model load into a dedicated CPU buffer type,
owned by the model, freed with it.

What the migration buys: pack lifetime tied to the model instead of
the process (no manual `ppc_apack_cache_clear()`, no cap heuristics —
the pack is simply the tensor's storage); load-time packing that
overlaps model I/O; and an end to double residency, since today the
int8 pack (1.8–4.6× native weight bytes) lives alongside the mmap'd
original.

What blocks it, stated honestly: the repack buffer type *replaces*
the original layout, and this project's small-n policy (patches
0015/0017) deliberately routes generation through ggml's vec_dot,
which needs the original blocks. Migrating a format therefore
requires either keeping both layouts resident (recreating the
double-residency problem inside the buffer) or providing a vec_dot
over the packed layout — which is exactly the direct low-bit GEMV
identified as the one open engineering item (VALIDATION-POWER10.md
§9). The 21 accelerated types also share tile kernels across formats,
while repack traits are declared per type; the trait surface is wide
but mechanical.

Sequencing decision: migrate family-by-family only where a
packed-layout n=1 story exists — today that is qbit (GEMV wins 4.6×)
and IQ1_M/TQ1_0 (cached-pack path, patch 0017). For the vectorized
vec_dot formats the GEMV experiment measured a loss
(VALIDATION-POWER10.md §9.2), so their originals must stay resident
and migration would mean double residency; they stay at the dispatch
layer by evidence, not by default. Until then the dispatch-layer cache carries the load, and the
pressures that motivated an early migration have been removed on
silicon: first-touch packing is slice-parallel across the op's
threads (patch 0019: 4.4 s → 1.05 s cold start on a 1.5B IQ2_M, with
a free +6% steady-state pp from NUMA-friendly first touch), the slot
table grows (0016), and a capacity refusal is loud instead of silent.
