# ppc-mma-kernels — build & test
#
# Cross-compiles the kernels for ppc64le and runs correctness tests under
# qemu-user (POWER10).  On real Power10/Power11 hardware, set CROSS= and
# QEMU= empty to build and run natively:
#
#     make CROSS= QEMU= test
#
# Layout for practitioners:
#   src/*.cpp            PRODUCTION kernels — these are what the patches
#                        in patches/ integrate into llama.cpp/ggml.
#   src/reference/*.cpp  the v1→v3 design evolution, kept for review and
#                        for the derivation in docs/DESIGN.md.  Nothing
#                        integrates these.
#
# Production file → formats covered → integration patch:
#   qbit_ppc_mma_v4.cpp   Q1_0, Q2_0 (PrismML)         0001 (+0008 v4 upgrade)
#   q4_k_ppc_mma.cpp      Q4_K                          0002
#   q5_k_ppc_mma.cpp      Q5_K                          0002
#   q6_k_ppc_mma.cpp      Q6_K                          0002
#   q2_k_ppc_mma.cpp      Q2_K                          0002
#   q3_k_ppc_mma.cpp      Q3_K                          0003
#   iq4_ppc_mma.cpp       IQ4_NL, IQ4_XS, MXFP4         0003, 0007
#   legacy_ppc_mma.cpp    Q4_1, Q5_0, Q5_1              0004
#   iq_grid_ppc_mma.cpp   TQ1_0, TQ2_0, IQ2_XXS/XS/S,   0005, 0007, 0008
#                         IQ3_XXS/S, IQ1_S/M, NVFP4
#
# Targets: make help

CROSS   ?= powerpc64le-linux-gnu-
CXX     ?= $(CROSS)g++
SYSROOT ?= /usr/powerpc64le-linux-gnu
QEMU    ?= qemu-ppc64le -cpu power10 -L $(SYSROOT)
CXXFLAGS ?= -O3 -mcpu=power10 -Wall -Wextra

BUILD := build

# ---- production test suites (format-complete, patch-backed) ----
PROD_TESTS := \
	$(BUILD)/qbit_test \
	$(BUILD)/q2_k_test $(BUILD)/q3_k_test $(BUILD)/q4_k_test \
	$(BUILD)/q5_k_test $(BUILD)/q6_k_test \
	$(BUILD)/iq4_test $(BUILD)/legacy_test $(BUILD)/iq_grid_test

# ---- reference (design-history) suites ----
REF_TESTS := \
	$(BUILD)/ref_q1_v1_test $(BUILD)/ref_q2_v1_test \
	$(BUILD)/ref_q1_v2_test $(BUILD)/ref_qbit_v3_test

BENCH := $(BUILD)/qbit_bench $(BUILD)/ref_q1_v2_bench $(BUILD)/ref_qbit_v3_bench

all: $(PROD_TESTS) $(REF_TESTS) $(BENCH)

$(BUILD):
	mkdir -p $(BUILD)

# ---- production ----
$(BUILD)/qbit_test: src/qbit_ppc_mma_v4.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQBIT4_TEST $< -o $@

$(BUILD)/q2_k_test: src/q2_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ2K_TEST $< -o $@

$(BUILD)/q3_k_test: src/q3_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ3K_TEST $< -o $@

$(BUILD)/q4_k_test: src/q4_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ4K_TEST $< -o $@

$(BUILD)/q5_k_test: src/q5_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ5K_TEST $< -o $@

$(BUILD)/q6_k_test: src/q6_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ6K_TEST $< -o $@

$(BUILD)/iq4_test: src/iq4_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DIQ4_TEST $< -o $@

$(BUILD)/legacy_test: src/legacy_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DLEGACY_TEST $< -o $@

$(BUILD)/iq_grid_test: src/iq_grid_ppc_mma.cpp src/iq_grids.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -DIQGRID_TEST $< -o $@

$(BUILD)/qbit_bench: src/qbit_ppc_mma_v4.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQBIT4_BENCH $< -o $@

# ---- reference (design history; see docs/DESIGN.md) ----
$(BUILD)/ref_q1_v1_test: src/reference/q1_0_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ1MMA_TEST $< -o $@

$(BUILD)/ref_q2_v1_test: src/reference/q2_0_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ2MMA_TEST $< -o $@

$(BUILD)/ref_q1_v2_test: src/reference/q1_0_ppc_mma_v2.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ1MMA_TEST $< -o $@

$(BUILD)/ref_qbit_v3_test: src/reference/qbit_ppc_mma_v3.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQBIT_TEST $< -o $@

$(BUILD)/ref_q1_v2_bench: src/reference/q1_0_ppc_mma_v2.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ1MMA_BENCH $< -o $@

$(BUILD)/ref_qbit_v3_bench: src/reference/qbit_ppc_mma_v3.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQBIT_BENCH $< -o $@

# ---- entry points ----
test: $(PROD_TESTS) $(REF_TESTS)
	@for t in $(PROD_TESTS) $(REF_TESTS); do echo "== $$t"; $(QEMU) ./$$t || exit 1; done
	@echo "ALL SUITES PASSED"

test-production: $(PROD_TESTS)
	@for t in $(PROD_TESTS); do echo "== $$t"; $(QEMU) ./$$t || exit 1; done
	@echo "ALL PRODUCTION SUITES PASSED"

test-reference: $(REF_TESTS)
	@for t in $(REF_TESTS); do echo "== $$t"; $(QEMU) ./$$t || exit 1; done
	@echo "ALL REFERENCE SUITES PASSED"

bench: $(BENCH)
	@for b in $(BENCH); do echo "== $$b"; $(QEMU) ./$$b; done
	@echo "(qemu timings are an instruction-count proxy only; benchmark on hardware)"

help:
	@echo "ppc-mma-kernels targets:"
	@echo "  make test              build + run all correctness suites (production + reference)"
	@echo "  make test-production   only the 9 production suites (what the patches integrate)"
	@echo "  make test-reference    only the v1-v3 design-history suites"
	@echo "  make bench             qemu instruction-count proxy benchmarks"
	@echo "  make all               build everything without running"
	@echo "  make clean             remove build/"
	@echo ""
	@echo "Native Power10/11:       make CROSS= QEMU= test"
	@echo "Versioned compiler:      make CXX=powerpc64le-linux-gnu-g++-14 test"
	@echo "Strict build:            make CXXFLAGS='-O3 -mcpu=power10 -Wall -Wextra -Werror' test"
	@echo ""
	@echo "Integration into llama.cpp: see patches/ + docs/INTEGRATION.md,"
	@echo "or run scripts/build-bonsai-power.sh for the one-command build."

clean:
	rm -rf $(BUILD)

.PHONY: all test test-production test-reference bench help clean
