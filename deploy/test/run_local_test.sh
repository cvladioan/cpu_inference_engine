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

echo "== bench.sh"
BENCH_CTX=1024 BENCH_UBATCH=256 RESULTS_DIR="$work/results" "$deploy/bench.sh"
grep -q '^| *[0-9]' "$work"/results/sweep-*.txt || { echo "FAIL: no benchmark rows"; exit 1; }

echo "PASS"
