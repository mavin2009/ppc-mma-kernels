#!/bin/sh
# Build the PrismML llama.cpp fork with POWER10/POWER11 MMA kernels for
# Q1_0/Q2_0, ready to run Bonsai models on ppc64le.
#
# Usage (on a Power10/Power11 machine):
#   ./scripts/build-bonsai-power.sh [builddir]
#
# Cross + qemu development loop (x86 host):
#   CROSS=1 ./scripts/build-bonsai-power.sh
#
# The patch was generated against fork commit
# 79697f23a2c8f3aa2ccb2fd7406095a8dbfbb454 (branch `prism`); if the fork
# has moved and a patch no longer applies cleanly, pin that commit or
# rebase the patch (it touches 2 files and adds 2).
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$REPO_DIR/bonsai-build}"
FORK_COMMIT=79697f23a2c8f3aa2ccb2fd7406095a8dbfbb454

mkdir -p "$WORK" && cd "$WORK"
if [ ! -d llama.cpp ]; then
    git clone --branch prism https://github.com/PrismML-Eng/llama.cpp llama.cpp
fi
cd llama.cpp
git fetch origin "$FORK_COMMIT" 2>/dev/null || true
git checkout "$FORK_COMMIT" 2>/dev/null || echo "warning: building fork tip instead of pinned commit"
for P in 0001-power-mma-q1-q2-sgemm 0002-power-mma-kquants-sgemm 0003-power-mma-q3k-iq4-sgemm 0004-power-mma-legacy-sgemm 0005-power-mma-grids-ternary-sgemm 0006-power-mma-robustness 0007-power-mma-formats-packcache 0008-power-mma-rollout-hardening; do
    git apply --check "$REPO_DIR/patches/$P.patch"
    git apply "$REPO_DIR/patches/$P.patch"
done

if [ -n "$CROSS" ]; then
    cat > /tmp/ppc64le-toolchain.cmake << 'TC'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)
set(CMAKE_C_COMPILER powerpc64le-linux-gnu-gcc-14)
set(CMAKE_CXX_COMPILER powerpc64le-linux-gnu-g++-14)
set(CMAKE_FIND_ROOT_PATH /usr/powerpc64le-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
TC
    EXTRA="-DCMAKE_TOOLCHAIN_FILE=/tmp/ppc64le-toolchain.cmake -DGGML_NATIVE=OFF -DCMAKE_C_FLAGS=-mcpu=power10 -DCMAKE_CXX_FLAGS=-mcpu=power10"
else
    # native Power build: -mcpu=native picks power10/power11 features incl. MMA
    EXTRA="-DGGML_NATIVE=ON"
fi

cmake -B build $EXTRA -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j"$(nproc)" --target llama-cli llama-server llama-bench

echo
echo "Built: $PWD/build/bin/{llama-cli,llama-server,llama-bench}"
echo "Verify the MMA path is active: the startup system-info line should show MMA = 1."
echo "Fetch a Bonsai GGUF (Q1_0/Q2_0) + mmproj from https://huggingface.co/PrismML"
echo "then e.g.: build/bin/llama-server -m bonsai-q2_0.gguf --mmproj mmproj.gguf"
