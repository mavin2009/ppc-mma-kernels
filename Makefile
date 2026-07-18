# ppc-mma-kernels
#
# Cross-compiles the kernels for ppc64le and runs correctness tests under
# qemu-user (POWER10). On real Power10/Power11 hardware, set CROSS= and
# QEMU= empty to build and run natively.

CROSS   ?= powerpc64le-linux-gnu-
CXX     := $(CROSS)g++
SYSROOT ?= /usr/powerpc64le-linux-gnu
QEMU    ?= qemu-ppc64le -cpu power10 -L $(SYSROOT)
CXXFLAGS := -O3 -mcpu=power10 -Wall -Wextra

BUILD := build

TESTS := $(BUILD)/q1_v1_test $(BUILD)/q2_v1_test $(BUILD)/q1_v2_test $(BUILD)/qbit_v3_test $(BUILD)/qbit_v4_test $(BUILD)/q4_k_test $(BUILD)/q5_k_test $(BUILD)/q6_k_test $(BUILD)/q2_k_test $(BUILD)/q3_k_test $(BUILD)/iq4_test $(BUILD)/legacy_test $(BUILD)/iq_grid_test
BENCH := $(BUILD)/q1_v2_bench $(BUILD)/qbit_v3_bench $(BUILD)/qbit_v4_bench

all: $(TESTS) $(BENCH)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/q1_v1_test: src/q1_0_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ1MMA_TEST $< -o $@

$(BUILD)/q2_v1_test: src/q2_0_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ2MMA_TEST $< -o $@

$(BUILD)/q1_v2_test: src/q1_0_ppc_mma_v2.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ1MMA_TEST $< -o $@

$(BUILD)/qbit_v3_test: src/qbit_ppc_mma_v3.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQBIT_TEST $< -o $@

$(BUILD)/q1_v2_bench: src/q1_0_ppc_mma_v2.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ1MMA_BENCH $< -o $@

$(BUILD)/qbit_v3_bench: src/qbit_ppc_mma_v3.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQBIT_BENCH $< -o $@

$(BUILD)/qbit_v4_test: src/qbit_ppc_mma_v4.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQBIT4_TEST $< -o $@

$(BUILD)/q4_k_test: src/q4_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ4K_TEST $< -o $@

$(BUILD)/q5_k_test: src/q5_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ5K_TEST $< -o $@

$(BUILD)/q6_k_test: src/q6_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ6K_TEST $< -o $@

$(BUILD)/q2_k_test: src/q2_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ2K_TEST $< -o $@

$(BUILD)/q3_k_test: src/q3_k_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQ3K_TEST $< -o $@

$(BUILD)/iq4_test: src/iq4_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DIQ4_TEST $< -o $@

$(BUILD)/legacy_test $(BUILD)/iq_grid_test: src/legacy_ppc_mma.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DLEGACY_TEST $< -o $@

$(BUILD)/qbit_v4_bench: src/qbit_ppc_mma_v4.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DQBIT4_BENCH $< -o $@

test: $(TESTS)
	@for t in $(TESTS); do echo "== $$t"; $(QEMU) ./$$t || exit 1; done
	@echo "ALL SUITES PASSED"

bench: $(BENCH)
	@for b in $(BENCH); do echo "== $$b"; $(QEMU) ./$$b; done
	@echo "(qemu timings are an instruction-count proxy only; benchmark on hardware)"

clean:
	rm -rf $(BUILD)

.PHONY: all test bench clean
