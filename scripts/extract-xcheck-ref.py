#!/usr/bin/env python3
"""Regenerate src/xcheck_ggml_ref.h from a pinned llama.cpp fork checkout.

Vendors, verbatim, the reference dequantization surface this repo's
decoders are cross-checked against: block structs, grid/sign/value
tables and dequantize_row_* functions from ggml-common.h /
ggml-quants.c / ggml-impl.h, wrapped in `namespace ggmlref`. The only
edits are mechanical: GGML_RESTRICT is defined away and everything is
namespaced. Anything else would defeat the point of the cross-check.

Usage: scripts/extract-xcheck-ref.py /path/to/llama.cpp > src/xcheck_ggml_ref.h
"""
import re, sys, os

SRC = sys.argv[1]

def read(p):
    return open(os.path.join(SRC, p)).read()

common = read("ggml/src/ggml-common.h")
quants = read("ggml/src/ggml-quants.c")
impl   = read("ggml/src/ggml-impl.h")

out = []

def emit(s):
    out.append(s.rstrip() + "\n")

def slice_braced(text, start_pat, what):
    """Extract from start_pat match through the balanced closing brace + ';' (if present)."""
    m = re.search(start_pat, text)
    if not m:
        sys.exit(f"extract-xcheck-ref: cannot find {what}")
    i = text.index("{", m.start())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                if end < len(text) and text[end] == ";":
                    end += 1
                return text[m.start():end]
    sys.exit(f"extract-xcheck-ref: unbalanced braces for {what}")

def struct_def(name):
    """typedef struct { ... } name;  — located by its terminator."""
    term = "} " + name + ";"
    e = common.find(term)
    if e < 0:
        sys.exit(f"extract-xcheck-ref: no struct {name}")
    s = common.rfind("typedef", 0, e)
    return common[s:e + len(term)]

def define_of(name, text):
    m = re.search(r"#define %s[(\s].*" % re.escape(name), text)
    if not m:
        sys.exit(f"extract-xcheck-ref: no #define {name}")
    return m.group(0)

fork_commit = os.popen(f"git -C {SRC} rev-parse --short HEAD 2>/dev/null").read().strip() or "unknown"

emit("// xcheck_ggml_ref.h — reference dequantization vendored VERBATIM from")
emit(f"// the pinned llama.cpp fork (commit {fork_commit}): ggml-common.h,")
emit("// ggml-quants.c, ggml-impl.h (MIT, ggml authors).  Wrapped in")
emit("// namespace ggmlref with GGML_RESTRICT defined away; no other edits.")
emit("// Regenerate: scripts/extract-xcheck-ref.py /path/to/llama.cpp")
emit("#pragma once")
emit("#include <stdint.h>")
emit("#include <string.h>")
emit("#include <math.h>")
emit("#include <assert.h>")
emit("")
emit("namespace ggmlref {")
emit("")
emit("#define GGML_RESTRICT")
emit("typedef uint16_t ggml_half;")
emit("typedef uint16_t ggml_fp16_t;")
emit("")

for d in ["QK_K", "QK4_NL", "QK_MXFP4", "QK_NVFP4", "QK_NVFP4_SUB"]:
    emit(define_of(d, common))
emit(define_of("IQ3S_N_SCALE", common))
emit(define_of("NGRID_IQ1S", common))
emit(define_of("IQ1S_DELTA", common))
emit(define_of("IQ1M_DELTA", common))
emit("")

for s in ["block_tq1_0", "block_tq2_0", "block_iq2_xxs", "block_iq2_xs",
          "block_iq2_s", "block_iq3_xxs", "block_iq3_s", "block_iq1_s",
          "block_iq1_m", "iq1m_scale_t", "block_iq4_nl", "block_iq4_xs",
          "block_mxfp4", "block_nvfp4"]:
    emit(struct_def(s))
    emit("")

# Tables appear either as plain arrays or as
# GGML_TABLE_BEGIN(type, name, size) ... GGML_TABLE_END(); translate the
# macro form to a plain array declaration, values untouched.
for t in ["iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid",
          "iq3s_grid", "iq1s_grid", "ksigns_iq2xs", "kmask_iq2xs",
          "kvalues_iq4nl", "kvalues_mxfp4"]:
    m = re.search(r"GGML_TABLE_BEGIN\((\w+),\s*%s,\s*(\w+)\)" % t, common)
    if m:
        end = common.index("GGML_TABLE_END()", m.end())
        body = common[m.end():end]
        emit(f"static const {m.group(1)} {t}[{m.group(2)}] = {{{body}}};")
    else:
        emit(slice_braced(common, r"static const \w+ %s\[" % t, t))
    emit("")

# fp16/e8m0/ue4m3 conversion helpers from ggml-impl.h
emit(slice_braced(impl, r"static inline uint32_t fp32_to_bits\(", "fp32_to_bits"))
emit("")
for f in ["fp32_from_bits", "ggml_compute_fp16_to_fp32", "ggml_e8m0_to_fp32_half", "ggml_ue4m3_to_fp32"]:
    emit(slice_braced(impl, r"static inline float %s\(" % f, f))
    emit("")
    emit("")
emit("#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)")
emit(define_of("GGML_E8M0_TO_FP32_HALF", impl))
emit("")

for f in ["tq1_0", "tq2_0", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs",
          "iq3_s", "iq1_s", "iq1_m", "iq4_nl", "iq4_xs", "mxfp4", "nvfp4"]:
    emit(slice_braced(quants, r"void dequantize_row_%s\(" % f, f"dequantize_row_{f}"))
    emit("")

emit("} // namespace ggmlref")
sys.stdout.write("".join(out))
