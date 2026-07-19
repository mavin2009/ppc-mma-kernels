# ppc-mma-kernels

POWER10 and POWER11 MMA GEMM kernels for quantized LLM inference. All
26 quantized GGUF formats, from 1-bit and ternary weights through the
K-quants, the IQ grid codebooks, MXFP4 and NVFP4, with integration
patches for llama.cpp.

## Why I wrote this

Mainline llama.cpp drives Power's Matrix-Multiply Assist unit for
Q4_0, Q8_0 and the float types, and quietly lets every other format
fall back to scalar code. That bothered me more than it probably
should have. The formats that fall back are the low-bit ones, which
is to say the formats where a machine with Power's memory bandwidth
ought to be embarrassing everyone else. We ship an outer-product
engine that can retire 64 int8 multiply-accumulates per cycle per
SMT4 slice, and then we run ternary weights through a scalar loop.

This started as a fix for the PrismML fork's 1-bit and ternary Bonsai
formats and did not stop until the scalar path was empty. It picked
up a production packing API, a tensor-keyed weight cache, a GEMV fast
path for token generation, and a paper trail of measured decisions,
including the experiments that lost. I kept those on purpose. The
negative space of a design is part of the design.

## The engineering, briefly

The MMA GER instructions multiply a signed int8 operand against an
unsigned one. Quantized formats hand you neither cleanly, so the whole
project is really a study in operand orientation. For formats with
unsigned codes (the K-quants, Q5_0, the ternary types) I put the raw
codes on the unsigned operand and untouched int8 activations on the
signed one, then account for the offset algebraically: the correction
term is either separable (a rank-one update applied per slab, using
sums the activation quantizer already computed) or, when the offset
shares the block's scale factor, folded into the main FMA for free.
Q4_1 and Q5_1 are my favorite case: their min term reduces to a single
multiply-add because ggml's Q8_1 blocks already store the scaled
activation sum. Nobody computes a sum anywhere. It felt like finding
money in a coat pocket.

Signed codebooks (IQ4, MXFP4) will not ride the unsigned operand, so
the orientation flips: codebook values on the signed side, activations
XOR-flipped onto the unsigned side, and a per-row correction of 128
times the codebook sums, precomputed at repack. The lookup itself is
one vec_perm per sixteen codes.

The grid formats and NVFP4 exposed the architectural payoff of doing
packing once per tensor rather than per call: when decode runs at
repack time, its cost is off the hot path entirely, and nine exotic
formats collapse into two shared signed kernels (32-deep and 16-deep
chunks) plus a page of scalar decoders apiece. The 16-deep variant
exists because some formats carry a scale every 16 elements, and that
granularity has a real price: 13.5 non-GER instructions per GER versus
7.6 for the 32-deep kernel. I counted rather than guessed, and the
number is in the design notes for whoever gets silicon time.

One more thing worth knowing before you read the kernels. On Power10
the accumulators physically live inside the MMA engine; the VSR
aliasing in the ISA is a polite fiction. Draining an accumulator
serializes against its GER stream, so a kernel that computes, drains,
fixes up and only then starts the next chunk is idling the engine on
every fixup. There is a compile-flag variant (IQGRID_PINGPONG) that
alternates two accumulator sets so the next chunk's GERs issue before
the previous chunk drains. It costs a measured three percent in static
instructions and buys engine-idle removal that no instruction count
can see. Both variants are in the test matrix. Hardware will pick the
winner, not me.

## What is verified, and what is not

Correctness is proven the boring way: thirteen kernel suites (fourteen
counting the ping-pong variant) against exact float64 references on
random data, ragged shapes, multi-slab depths and n=1, all run under
qemu-power10, built with -Wall -Wextra -Werror, with UBSan clean on
the gnarliest suites. The nine integration patches are gated: every
push proves the series applies in sequence on a pristine checkout of
the pinned fork commit and reproduces the build-verified tree byte for
byte. I added that gate after shipping a broken series once. The
defect log in docs/REVIEW.md is honest about that and about the other
things I got wrong, because a repo that has never found a defect in
itself has not looked.

What is not verified is anything requiring silicon. Emulation prices
every instruction identically, models no cache and no engine, and so
every performance statement in this repository is either exact static
accounting or a clearly labeled instruction proxy. There are no
absolute performance claims here, which is how you can tell the
relative ones are honest. Version 0.9.0 marks the emulation-verified
state; 1.0.0 is reserved for the first hardware validation. If you
have a Power10 or Power11 machine, scripts/validate-on-power.sh will
build MMA and no-MMA variants, run a temperature-0 token-identity
comparison (one test that simultaneously validates the GER semantics,
all the decoders and the dispatch), benchmark both, and hand you a
report to paste into an issue. Twenty minutes of your LPAR closes the
one gap I cannot.

## Layout and use

Production kernels live in src/; the v1 through v3 design lineage
lives in src/reference/ for anyone following the derivation in
docs/DESIGN.md. The Makefile header maps every file to its formats and
its patch. `make help` explains the rest.

Cross-compile and test from an x86 box:

    sudo apt-get install g++-14-powerpc64le-linux-gnu qemu-user
    make CXX=powerpc64le-linux-gnu-g++-14 test    # ~12 seconds

Native on Power:

    make CROSS= QEMU= test bench

Deploy the patched fork (clones at a pinned commit, applies the gated
series, builds llama-cli, llama-server and llama-bench):

    ./scripts/build-bonsai-power.sh
    JOBS=2 ./scripts/build-bonsai-power.sh   # memory-constrained guests

Details, model links and the mandatory first hardware check are in
docs/DEPLOY.md. Emulated performance numbers, with an explicit
statement of what they can and cannot show, are in
docs/BENCHMARKS-QEMU.md. Reviewers should start at docs/REVIEW.md; I
wrote it for the skeptical reader and I mean that as a compliment.

## License

MIT
