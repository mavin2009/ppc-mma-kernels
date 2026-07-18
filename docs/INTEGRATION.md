# Integrating into the PrismML llama.cpp fork

These kernels use llamafile-sgemm conventions on purpose so the wiring
is mechanical. **This integration is described but not machine-tested**
(building the full fork for ppc64le was out of scope); treat it as a
map, not a patch.

## Where to hook

`ggml/src/ggml-cpu/llamafile/sgemm.cpp`, in `llamafile_sgemm()`'s type
dispatch (the same switch that routes `GGML_TYPE_Q8_0`/`GGML_TYPE_Q4_0`
to `tinyBLAS_Q0_PPC` under `#if defined(__MMA__)`):

```cpp
case GGML_TYPE_Q1_0:
    if (Btype != GGML_TYPE_Q8_0) return false;
    gemm_q1_0_q8_0_ppc_v3(m, n, k,
        (const block_q1_0 *)A, lda,
        (const block_q8_0 *)B, ldb,
        (float *)C, ldc, ith, nth);
    return true;
case GGML_TYPE_Q2_0:
    if (Btype != GGML_TYPE_Q8_0) return false;
    gemm_q2_0_q8_0_ppc_v3(...);
    return true;
```

Notes:

- `k` (row length in elements) must satisfy `k % 128 == 0`; return
  `false` otherwise so ggml falls back to `vec_dot`. All PrismML
  release models satisfy this, but guard it anyway.
- The kernels' block structs must be replaced by the ggml ones
  (`ggml-common.h`) — they are layout-identical (static asserts
  included), so delete the local definitions and include the header.
- Replace the local `fp16_to_fp32` with `GGML_FP16_TO_FP32`.
- C layout in llamafile sgemm is column-major `C[i + j*ldc]` — already
  matched.
- Thread partitioning: the kernels take `(ith, nth)` and split rows in
  MR-sized tiles, matching how `llamafile_sgemm` is invoked from
  `ggml_compute_forward_mul_mat`.

## Longer-term (performance-correct) integration

- Move the weight pack (`pack_A16_*`) into `ggml-cpu/repack.cpp` as a
  `Q1_0x16`/`Q2_0x16` repacked type computed once at model load; the
  GEMM then consumes pre-packed weights and the per-slab A pack
  disappears from the hot path.
- Move the activation pack (`pack_B8`, including the separable
  correction accumulators `E`) next to the `Q8_0` activation
  quantization so it is computed once and shared across threads,
  instead of per-thread as in the standalone kernels.
- Register a `vec_dot`-compatible n=1 path or accept the GEMM path for
  token generation; benchmark both (see DESIGN.md future work).

## Sanity checks after wiring

1. `llama-cli` perplexity run on a small Q1_0/Q2_0 model vs the scalar
   build: outputs should match to float-rounding level.
2. `llama-bench` prompt-processing (pp) and token-generation (tg)
   throughput vs the scalar baseline and vs a Q4_0 model on the MMA
   path, on real Power10/Power11 hardware.
