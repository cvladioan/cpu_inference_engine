#!/usr/bin/env bash
# Tessera's regression test, on any machine (no GPU, no download): a tiny random MoE model goes through the real
# scripts - calibrate.sh builds a profile from a routing trace, serve.sh runs the tiers - and every configuration
# must answer like the plain engine.
#
#   tessera/tests/run_tests.sh                    # needs an engine build (scripts/build.sh; CUDA=0 is enough)
#   INSTALL_DIR=/path tessera/tests/run_tests.sh
#
# On a CPU-only build the "hot" copies land in RAM, so both halves of the split run on the CPU: this checks the
# split's logic (which experts go where, the remapped ids, the sum), not GPU speed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tessera="$(dirname "$here")"
source "$tessera/scripts/lib.sh"
load_config

work=$(mktemp -d)
pid=""
cleanup() { [[ -n "$pid" ]] && kill "$pid" 2>/dev/null; rm -rf "$work"; }
trap cleanup EXIT

export MODEL_FILE="$work/moe.gguf" PROFILE="$work/moe.profile" PORT=${TEST_PORT:-18290} HOST=127.0.0.1 API_KEY=
export CTX=1024 THREADS=${THREADS:-4} EXTRA_ARGS=""
# GPU-style arguments even on a CPU-only build: the hot copies then land in RAM, which tests the split's logic
export CPU_ONLY=0
python3 "$here/make_tiny_gguf.py" --embd 256 --ff 128 --experts 16 --used 4 --layers 4 --heads 4 "$MODEL_FILE" >/dev/null

echo "== gguf_info / plan"
python3 "$tessera/tools/gguf_info.py" "$MODEL_FILE"
python3 "$tessera/tools/plan.py" "$MODEL_FILE" --vram-gib 12 --ram-gib 26 --ctx 1024 --estimate

echo "== calibrate.sh (routing trace -> profile)"
printf 'Hello there\nWrite a list of colors\nCount to ten\n' >"$work/prompts.txt"
PROMPTS="$work/prompts.txt" TOKENS=24 CALIBRATE_PORT=$((PORT + 1)) INSTALL_DIR="$INSTALL_DIR" \
    "$tessera/scripts/calibrate.sh"
[[ -s "$PROFILE" ]] || { echo "FAIL: no profile"; exit 1; }
grep -c '^[0-9]' "$PROFILE" | xargs echo "profile pairs:"

run() {   # name, env... ; collects greedy outputs into $work/<name>.json
    local name=$1; shift
    env "$@" "$tessera/scripts/serve.sh" ${SERVE_ARGS:-} >"$work/$name.log" 2>&1 &
    pid=$!
    for _ in $(seq 120); do
        curl -sf "http://$HOST:$PORT/health" >/dev/null && break
        kill -0 "$pid" 2>/dev/null || { cat "$work/$name.log"; echo "FAIL: $name: server exited"; exit 1; }
        sleep 0.5
    done
    python3 "$here/greedy.py" collect "http://$HOST:$PORT" >"$work/$name.json"
    kill "$pid"; wait "$pid" 2>/dev/null || true
    pid=""
}

echo "== baseline: every expert on the CPU"
SERVE_ARGS=--no-hot run base HOT_MIB=0 CACHE_MIB=0
fail=0
check() {   # name, env...
    local name=$1; shift
    run "$name" "$@"
    grep -h "hot experts (\|hot experts disabled\|expert cache (final)" "$work/$name.log" | sed 's/^/    /' || true
    printf '%-26s' "$name"
    python3 "$here/greedy.py" compare "$work/base.json" "$work/$name.json" || fail=1
}
check hot-experts HOT_MIB=2 CACHE_MIB=0
check hot-all-experts HOT_MIB=64 CACHE_MIB=0
check hot+ssd-tier HOT_MIB=2 CACHE_MIB=1
check ssd-tier-only HOT_MIB=0 CACHE_MIB=1
check cpu-only CPU_ONLY=1 HOT_MIB=2 CACHE_MIB=0
check cpu-only+ssd-tier CPU_ONLY=1 HOT_MIB=2 CACHE_MIB=1

(( fail == 0 )) || { echo "FAIL"; exit 1; }
echo "PASS"
