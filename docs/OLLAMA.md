# Ollama on Power, and what IBM actually shipped there

Ollama has no inference engine of its own; it vendors llama.cpp/ggml
under its ml/backend tree. IBM's publicly visible Ollama contributions
(PRs #9469 and #9538, amritahs-ibm) add -mcpu=power10 to the vendored
llamafile folder's build, behind go build tags ppc64le.power10 and
ppc64le.power9. In other words: build-flag enablement of the upstream
ggml MMA path, not new kernels in Ollama itself. The kernel substance
is the same tinyBLAS_Q0_PPC code this repo analyzed in patch 0011's
commit message: per-call repacking of immutable weights, a scalar
double comparray fixup inside the hot loop, no GEMV path for n = 1,
and shallow accumulator interleave. Ollama on Power inherits all of
that through vendoring, which is likely why it underwhelms in practice
despite the flags being present.

## Supporting both runtimes

Because the target is the vendored ggml, this repo's series ports to
Ollama by applying against ollama's vendored ggml-cpu tree instead of
a llama.cpp checkout. Practical notes for whoever does it:

1. Patch 0001 (Q1_0/Q2_0) is PrismML-specific; Ollama's vendored ggml
   lacks those types. Skip it and drop the two dispatch hunks that
   reference them from later patches, or carry the prism type patches
   first.
2. Patches 0002 through 0011 target standard ggml files
   (ggml-cpu/llamafile/*, CMakeLists) and apply with a path prefix
   adjustment (git apply --directory pointing at the vendored root).
   Expect context drift: Ollama pins its own ggml snapshot, so hunks
   around the dispatch switch may need a trivial rebase.
3. Ollama's Go build must compile the new .cpp files; its cgo/cmake
   glue for the llamafile folder picks up the directory contents, but
   verify the added translation units land in the ppc64le.power10
   build tag path.
4. Verification carries over unchanged: the standalone suites in this
   repo are runtime-agnostic, and a temperature-0 comparison between a
   patched and stock Ollama binary on the same model is the
   acceptance gate, same as DEPLOY.md step 5.

This document records findings as of July 2026; check the Ollama PRs
above for anything newer before quoting the state of their tree.
