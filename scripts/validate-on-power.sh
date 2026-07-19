#!/bin/sh
# validate-on-power.sh — the missing half of this repo's verification.
#
# Runs ON a Power10/Power11 machine. Builds the patched fork twice from
# the same source tree — once with MMA (native), once with -mcpu=power9
# so every #if __MMA__ path compiles out and ggml falls back to its
# scalar/VSX vec_dot — then:
#
#   1. TEMPERATURE-0 CHECK: identical prompt, seed 42, temp 0, greedy
#      sampling on both binaries; outputs must match token-for-token.
#      This single test simultaneously validates xvi8ger4pp silicon
#      semantics, all decode-at-repack decoders against ggml's own
#      dequantization, and the end-to-end dispatch.
#   2. llama-bench on both, prompt-processing and generation.
#
# Emits validation-report.md, ready to paste into a "hardware result"
# issue at github.com/mavin2009/ppc-mma-kernels.
#
# Usage:  ./scripts/validate-on-power.sh /path/to/model.gguf [builddir]
set -e

MODEL="$1"
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "usage: $0 /path/to/model.gguf [builddir]"; exit 1; }
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${2:-$REPO_DIR/bonsai-build}"
SRC="$WORK/llama.cpp"
[ -d "$SRC" ] || { echo "run scripts/build-bonsai-power.sh first (source tree: $SRC)"; exit 1; }
cd "$SRC"

PROMPT="The three most important properties of a matrix multiplication kernel are"
COMMON="-m $MODEL -p \"$PROMPT\" -n 64 --temp 0 --seed 42 --no-display-prompt"

echo "== building MMA (native) variant"
cmake -B build-mma -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build build-mma -j"${JOBS:-$(nproc)}" --target llama-cli llama-bench > /dev/null

echo "== building no-MMA reference (-mcpu=power9)"
cmake -B build-ref -DGGML_NATIVE=OFF -DCMAKE_C_FLAGS=-mcpu=power9 -DCMAKE_CXX_FLAGS=-mcpu=power9 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build build-ref -j"${JOBS:-$(nproc)}" --target llama-cli llama-bench > /dev/null

echo "== confirming MMA is active in the native build"
build-mma/bin/llama-cli -m "$MODEL" -p "x" -n 1 --temp 0 2>&1 | grep -o "MMA = [01]" | head -1 || true

echo "== temperature-0 comparison"
eval build-mma/bin/llama-cli $COMMON  2>/dev/null > /tmp/out-mma.txt
eval build-ref/bin/llama-cli $COMMON  2>/dev/null > /tmp/out-ref.txt
if diff -q /tmp/out-mma.txt /tmp/out-ref.txt > /dev/null; then
    T0="**PASS** — MMA and scalar outputs are token-identical"
else
    T0="**FAIL** — outputs diverge (see diff below); do NOT deploy; please attach both outputs to an issue"
fi
echo "$T0"

echo "== llama-bench (this takes a few minutes)"
BM=$(build-mma/bin/llama-bench -m "$MODEL" 2>/dev/null | tail -8)
BR=$(build-ref/bin/llama-bench -m "$MODEL" 2>/dev/null | tail -8)

cat > "$REPO_DIR/validation-report.md" << REPORT
# Hardware validation report

- Machine: $(grep -m1 model /proc/cpuinfo 2>/dev/null || uname -m), $(nproc) threads
- Kernel/OS: $(uname -sr)
- Model: $(basename "$MODEL") ($(du -h "$MODEL" | cut -f1))
- Repo: $(cd "$REPO_DIR" && git describe --tags --always 2>/dev/null)
- Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Temperature-0 check
$T0

## llama-bench — MMA (native)
\`\`\`
$BM
\`\`\`

## llama-bench — reference (-mcpu=power9, no MMA)
\`\`\`
$BR
\`\`\`
REPORT
diff /tmp/out-mma.txt /tmp/out-ref.txt >> "$REPO_DIR/validation-report.md" 2>/dev/null || true

echo
echo "Report written: $REPO_DIR/validation-report.md"
echo "Please share it: https://github.com/mavin2009/ppc-mma-kernels/issues/new?template=hardware-result.md"
