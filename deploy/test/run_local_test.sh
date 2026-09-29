#!/usr/bin/env bash
# End-to-end test of the deploy scripts with a tiny random model: builds nothing,
# downloads nothing. Needs deploy/build.sh to have run (same INSTALL_DIR).
#
#   INSTALL_DIR=/opt/deepseek-cpu deploy/test/run_local_test.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
deploy="$(dirname "$here")"

work=$(mktemp -d)
server_pid=""
cleanup() {
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null && wait "$server_pid" 2>/dev/null
    rm -rf "$work"
}
trap cleanup EXIT

python3 "$here/make_tiny_gguf.py" "$work/tiny.gguf"

export MODEL_FILE="$work/tiny.gguf" MODEL_ALIAS=tiny NUMA_MODE=none
export HOST=127.0.0.1 PORT="${TEST_PORT:-18080}" API_KEY=test-key API_KEY_FILE=''
export PARALLEL=2 CTX_PER_SLOT=1024 CACHE_RAM_MIB=256 MLOCK=0 SPEC_TYPE='' EXTRA_ARGS=''
export THREADS="${THREADS:-2}"

echo "== serve.sh --plan"
"$deploy/serve.sh" --plan

echo "== start server"
"$deploy/serve.sh" >"$work/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 60); do
    if curl -sf -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/health" >/dev/null; then
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || { cat "$work/server.log"; echo "FAIL: server exited"; exit 1; }
    sleep 1
done

echo "== reject requests without the API key"
# /health and /v1/models are public by design; generation endpoints are not.
code=$(curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}' \
    "http://$HOST:$PORT/v1/chat/completions")
[[ "$code" == 401 ]] || { echo "FAIL: expected 401 without API key, got $code"; exit 1; }
echo "401 as expected"

echo "== smoke test"
python3 "$deploy/smoke_test.py" --url "http://$HOST:$PORT" --api-key "$API_KEY" \
    --concurrency 2 --max-tokens 32 --fixed-length

kill "$server_pid"
wait "$server_pid" 2>/dev/null || true
server_pid=""

echo "== tune.py"
python3 "$deploy/tune.py" --quick --threads "$THREADS" --max-tokens 16 --target 1 \
    --port "$(( PORT + 1 ))" --results-dir "$work/tune"
ls "$work"/tune/tune-*.json >/dev/null || { echo "FAIL: tune.py wrote no results"; exit 1; }

echo "== bench.sh"
BENCH_CTX=1024 BENCH_UBATCH=256 RESULTS_DIR="$work/results" "$deploy/bench.sh"
grep -q '^| *[0-9]' "$work"/results/sweep-*.txt || { echo "FAIL: no benchmark rows"; exit 1; }

ik_bin=$(bash -c 'source "$1/lib.sh"; load_config; echo "$IK_BIN"' _ "$deploy")
ik_help=$("$ik_bin/llama-server" --help 2>&1 || true)
if [[ "$ik_help" == *--expert-cache* ]]; then
    echo "== expert cache: outputs must match exactly, even with a budget far below one token's experts"
    python3 "$here/make_tiny_gguf.py" --embd 256 --ff 256 --experts 16 --used 4 --layers 4 --heads 4 "$work/moe.gguf"
    for variant in baseline cache; do
        extra=""
        [[ "$variant" == cache ]] && extra="--defer-experts --expert-cache 4"
        MODEL_FILE="$work/moe.gguf" API_KEY='' PORT=$(( PORT + 2 )) EXPERT_STREAMING=off EXPERT_CACHE_MIB=0 \
            EXTRA_ARGS="$extra" "$deploy/serve.sh" >"$work/$variant.log" 2>&1 &
        server_pid=$!
        for _ in $(seq 60); do
            curl -sf "http://$HOST:$(( PORT + 2 ))/health" >/dev/null && break
            kill -0 "$server_pid" 2>/dev/null || { cat "$work/$variant.log"; echo "FAIL: server exited"; exit 1; }
            sleep 1
        done
        python3 "$here/greedy_outputs.py" "http://$HOST:$(( PORT + 2 ))" >"$work/$variant.json"
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    done
    grep -h "expert cache (final)" "$work/cache.log" || true
    python3 "$here/greedy_outputs.py" --compare "$work/baseline.json" "$work/cache.json" \
        || { echo "FAIL: expert cache changed the outputs"; exit 1; }
else
    echo "== expert cache: skipped (llama-server built without engine/patches)"
fi

echo "PASS"
