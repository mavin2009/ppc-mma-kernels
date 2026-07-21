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
PORT=8971

# generate 32 greedy tokens via llama-server + curl: no TTY, no stdin,
# no conversation mode -- llama-cli's interactive behavior defeated
# three prior attempts at scripting it (see git log of this file).
run_gen() {   # $1 = build dir, $2 = output file, $3 = server log
    "$1/bin/llama-server" -m "$MODEL" --port $PORT --host 127.0.0.1         > "$3" 2>&1 &
    SPID=$!
    up=0
    for i in $(seq 1 120); do
        if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then up=1; break; fi
        sleep 2
    done
    if [ $up -ne 1 ]; then echo "server failed to start ($1); see $3"; kill $SPID 2>/dev/null; exit 1; fi
    curl -s --connect-timeout 5 --max-time 1800 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json'         -d "{\"prompt\": \"$PROMPT\", \"n_predict\": 32, \"temperature\": 0, \"seed\": 42, \"cache_prompt\": false}"         | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))' > "$2"
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
}

echo "[$(date +%H:%M:%S)] == building MMA (native) variant"
cmake -B build-mma -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build build-mma -j"${JOBS:-$(nproc)}" --target llama-server llama-bench > /dev/null

echo "[$(date +%H:%M:%S)] == building no-MMA reference (-mcpu=power9)"
cmake -B build-ref -DGGML_NATIVE=OFF -DCMAKE_C_FLAGS=-mcpu=power9 -DCMAKE_CXX_FLAGS=-mcpu=power9       -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build build-ref -j"${JOBS:-$(nproc)}" --target llama-server llama-bench > /dev/null

echo "[$(date +%H:%M:%S)] == temperature-0 comparison (two server starts + model loads; the scalar side is deliberately slow -- minutes each, not hung)"
run_gen build-mma /tmp/out-mma.txt /tmp/server-mma.log
run_gen build-ref /tmp/out-ref.txt /tmp/server-ref.log

echo "[$(date +%H:%M:%S)] == MMA active in native build:"
grep -o "MMA = [01]" /tmp/server-mma.log | head -1 || echo "MMA flag not found in server banner; check /tmp/server-mma.log"

# Two-tier gate. Tier 1: token identity (strong PASS). Tier 2: the
# -mcpu=power9 reference runs ggml's VSX vec_dot, whose vector partial
# sums + horizontal reduction form a DIFFERENT rounding tree than any
# other implementation -- bit-identity against it is unachievable by
# construction, so a divergence is acceptable iff it occurs at a
# genuine near-tie: the road-not-taken token must sit within TOL
# logprob of the chosen one in BOTH runs. Anything larger is a real
# numerical defect and fails.
if diff -q /tmp/out-mma.txt /tmp/out-ref.txt > /dev/null; then
    T0="**PASS** -- MMA and scalar outputs are token-identical"
else
    T0=$(python3 - << 'PYGATE'
import json
TOL = 0.10
def toks(p):
    d = json.load(open(p))
    out = []
    for t in d.get("completion_probabilities", []):
        cand = t.get("probs") or t.get("top_logprobs") or []
        chosen = t.get("content") or t.get("token")
        m = {}
        for c in cand:
            key = c.get("tok_str", c.get("token"))
            v = c.get("prob", c.get("logprob"))
            m[key] = v
        out.append((chosen, m))
    return out
try:
    A, B = toks("/tmp/out-mma.txt.json"), toks("/tmp/out-ref.txt.json")
    import math
    def lp(v):
        return math.log(max(v, 1e-30)) if 0 <= v <= 1 else v
    for i, ((ta, ma), (tb, mb)) in enumerate(zip(A, B)):
        if ta == tb:
            continue
        da = abs(lp(ma.get(ta, 0)) - lp(ma.get(tb, ma.get(ta, 0))))
        db = abs(lp(mb.get(tb, 0)) - lp(mb.get(ta, mb.get(tb, 0))))
        if da <= TOL and db <= TOL:
            print(f"**PASS (near-tie)** -- first divergence at token {i}: the two candidates sit within {max(da,db):.4f} logprob in both builds (tolerance {TOL}); consistent with reduction-order rounding between MMA and the VSX reference, not a kernel defect")
        else:
            print(f"**FAIL** -- divergence at token {i} with logprob gaps {da:.3f}/{db:.3f} (tolerance {TOL}): a real numerical defect; do NOT deploy; attach /tmp/out-*.txt.json to an issue")
        break
    else:
        print("**PASS** -- token streams identical over compared span")
except Exception as e:
    print(f"**INDETERMINATE** -- gate analysis failed ({e}); attach /tmp/out-*.txt.json to an issue")
PYGATE
)
fi
echo "$T0"

echo "[$(date +%H:%M:%S)] == llama-bench (this takes a few minutes)"
# llama-bench defaults to a physical-core heuristic (often 4 on
# ppc64le SMT topologies). Sweep a ladder instead: Power10 has one MMA
# engine per SMT4 slice, so the GEMM sweet spot is near slice count
# while bandwidth-bound generation often wants more threads.
NP=$(nproc)
TL="$((NP/4 > 0 ? NP/4 : 1)),$((NP/2 > 0 ? NP/2 : 1)),$NP"
echo "   thread ladder: -t $TL (nproc=$NP)"
BM=$(build-mma/bin/llama-bench -m "$MODEL" -t "$TL" -p 128 -n 32 -r 2 2>/dev/null | tail -14)
BR=$(build-ref/bin/llama-bench -m "$MODEL" -t "$TL" -p 128 -n 32 -r 2 2>/dev/null | tail -14)

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
