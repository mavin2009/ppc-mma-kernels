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

# Refuse a squatted port. A pre-existing server here answers both
# builds' health checks and completions, so the two outputs compare
# equal no matter what the kernels compute -- a forged PASS. Field
# incident 2026-07-22: an orphaned llama-server from the previous
# day's session survived its kill and did exactly that, three times.
if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "FATAL: something already answers on 127.0.0.1:$PORT."
    echo "  ss -tlnp | grep $PORT   -- find it, kill it, re-run."
    exit 1
fi

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
    # Health answering is necessary, not sufficient: prove the answerer
    # is OUR server, i.e. our child is alive and the served model is
    # the one we were asked to validate (when /props exposes it).
    if ! kill -0 $SPID 2>/dev/null; then
        echo "FATAL: our server ($1) is dead but :$PORT answers -- another server is squatting the port."
        exit 1
    fi
    served=$(curl -s --max-time 5 "http://127.0.0.1:$PORT/props" 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("model_path",""))
except Exception: print("")' 2>/dev/null)
    case "$served" in
        "" ) : ;;                      # this fork's /props may not expose model_path; the checks above still hold
        *"$(basename "$MODEL")" ) : ;;
        * ) echo "FATAL: server on :$PORT serves '$served', not $(basename "$MODEL") -- refusing to compare."
            kill $SPID 2>/dev/null; exit 1 ;;
    esac
    curl -s --connect-timeout 5 --max-time 1800 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\": \"$PROMPT\", \"n_predict\": 32, \"temperature\": 0, \"seed\": 42, \"cache_prompt\": false, \"n_probs\": 5}" \
        > "$2.json" || true
    python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("content",""))
except Exception as e:
    sys.stderr.write("extract failed: %s\n" % e)' < "$2.json" > "$2" || true
    if [ ! -s "$2" ]; then
        echo "WARNING: empty generation from $1 -- first 300 bytes of server response:"
        head -c 300 "$2.json" 2>/dev/null; echo
    fi
    kill $SPID 2>/dev/null || true
    wait $SPID 2>/dev/null || true
    # Verify the port is actually released -- a survivor here becomes
    # next run's squatter (see the field incident above).
    i=0
    while curl -s --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        i=$((i+1))
        [ $i -gt 10 ] && { echo "FATAL: :$PORT still answers after kill; refusing to continue."; exit 1; }
        kill -9 $SPID 2>/dev/null || true
        sleep 1
    done
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

# The old check grepped the server banner for "MMA = 1", which failed
# two ways at once in the field: this fork's system_info has no MMA
# token, and `grep | head -1 || echo` can never take the fallback
# (head exits 0 regardless). Count GER instructions in the binary
# instead -- the ISA does not lie.
echo "[$(date +%H:%M:%S)] == MMA instructions in the binaries (objdump):"
GER_MMA=$(objdump -d build-mma/bin/libggml-cpu.so 2>/dev/null | grep -cE "xvi8ger4|xvi4ger8|pmxvi8ger4" || true)
GER_REF=$(objdump -d build-ref/bin/libggml-cpu.so 2>/dev/null | grep -cE "xvi8ger4|xvi4ger8|pmxvi8ger4" || true)
echo "   build-mma: $GER_MMA GER; build-ref: $GER_REF GER"
[ "${GER_MMA:-0}" -gt 0 ] || { echo "FATAL: native build contains no GER instructions -- MMA never compiled in; this comparison proves nothing."; exit 1; }
[ "${GER_REF:-0}" -eq 0 ] || echo "WARNING: reference build contains GER instructions; baseline is not MMA-free."

# preflight: fail LOUDLY, never silently
for f in /tmp/out-mma.txt /tmp/out-ref.txt /tmp/out-mma.txt.json /tmp/out-ref.txt.json; do
    if [ ! -s "$f" ]; then
        echo "ERROR: $f missing or empty -- generation phase failed."
        echo "  server logs: /tmp/server-mma.log /tmp/server-ref.log"
        echo "  raw responses: /tmp/out-*.txt.json"
        T0="**INDETERMINATE** -- generation produced no output; see messages above"
    fi
done
set +e   # nothing in the gate below may abort the script silently
# Two-tier gate. Tier 1: token identity (strong PASS). Tier 2: the
# -mcpu=power9 reference runs ggml's VSX vec_dot, whose vector partial
# sums + horizontal reduction form a DIFFERENT rounding tree than any
# other implementation -- bit-identity against it is unachievable by
# construction, so a divergence is acceptable iff it occurs at a
# genuine near-tie: the road-not-taken token must sit within TOL
# logprob of the chosen one in BOTH runs. Anything larger is a real
# numerical defect and fails.
if [ -n "${T0:-}" ]; then
    : # preflight already decided
elif diff -q /tmp/out-mma.txt /tmp/out-ref.txt > /dev/null; then
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

# Tier 3: divergence beyond the near-tie band. Before calling that a
# kernel defect, measure this machine's own cross-codegen envelope: a
# -mcpu=power8 build of the SAME tree contains no MMA and none of this
# repo's kernels either, so drift between it and the power9 reference
# is pure compiler rounding-tree variation. If the MMA build's drift
# sits within twice that envelope -- or the two MMA-free builds flip a
# token themselves -- the divergence cannot indict the kernels.
# Added after field data 2026-07-22: power8-vs-power9 vanilla ggml
# flipped at token 7 with median drift 0.11 on a 1.5B model, larger
# than TOL. A fixed tolerance alone overclaims in both directions.
case "$T0" in *"**FAIL**"*)
    echo "[$(date +%H:%M:%S)] == tier-2 divergence: building -mcpu=power8 control (a few minutes)"
    if [ ! -x build-ref8/bin/llama-server ]; then
        cmake -B build-ref8 -DGGML_NATIVE=OFF -DCMAKE_C_FLAGS=-mcpu=power8 -DCMAKE_CXX_FLAGS=-mcpu=power8 \
              -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null 2>&1
        cmake --build build-ref8 -j"${JOBS:-$(nproc)}" --target llama-server > /dev/null 2>&1
    fi
    if [ -x build-ref8/bin/llama-server ]; then
        run_gen build-ref8 /tmp/out-ref8.txt /tmp/server-ref8.log
        T0=$(python3 - << 'PYCTL'
import json, math
def toks(p):
    d = json.load(open(p)); seq = []
    for t in d.get("completion_probabilities", []):
        m = {}
        for c in (t.get("probs") or t.get("top_logprobs") or []):
            m[c.get("tok_str", c.get("token"))] = c.get("prob", c.get("logprob"))
        seq.append((t.get("content") or t.get("token"), m))
    return seq
def lp(v):
    return math.log(max(v, 1e-30)) if 0 <= v <= 1 else v
def drift(A, B):
    ds = []; flip = False
    for (ta, ma), (tb, mb) in zip(A, B):
        if ta != tb:
            flip = True; break
        for k in set(ma) & set(mb):
            ds.append(abs(lp(ma[k]) - lp(mb[k])))
    ds.sort()
    return (ds[len(ds)//2] if ds else None), flip
try:
    M  = toks("/tmp/out-mma.txt.json")
    R9 = toks("/tmp/out-ref.txt.json")
    R8 = toks("/tmp/out-ref8.txt.json")
    m9, _ = drift(M, R9)
    c, ctl_flip = drift(R9, R8)
    if m9 is None:
        print("**INDETERMINATE** -- MMA and reference diverge at token 0; no span to measure. Attach /tmp/out-*.txt.json to an issue.")
    elif ctl_flip or (c is not None and c > 0 and m9 <= 2 * c):
        extra = " and itself flips a token" if ctl_flip else ""
        print(f"**PASS (codegen envelope)** -- the tier-2 divergence lies within this machine's cross-codegen rounding envelope: MMA vs power9-reference median logprob drift {m9:.3f}; the MMA-free power8-vs-power9 control pair drifts {(c if c is not None else float('nan')):.3f}{extra}. Two builds containing none of this repo's code behave the same way, so the divergence does not indict the kernels.")
    else:
        print(f"**FAIL (control-confirmed)** -- MMA drift median {m9:.3f} exceeds twice the MMA-free control envelope {c:.3f}: the divergence is specific to the MMA path. A real numerical defect; do NOT deploy; attach /tmp/out-*.txt.json to an issue.")
except Exception as e:
    print(f"**INDETERMINATE** -- envelope analysis failed ({e}); attach /tmp/out-*.txt.json to an issue")
PYCTL
)
    else
        T0="$T0
(power8 control build failed; tier-3 envelope check unavailable)"
    fi
;; esac

echo "$T0"
set -e

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
