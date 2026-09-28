#!/usr/bin/env bash
# Benchmark prompt processing (PP) and generation (TG) speed at growing context
# depth with llama-sweep-bench, using the same model, threads and NUMA placement
# as deploy/serve.sh. Stop the server first: both need the full memory bandwidth.
#
#   deploy/bench.sh                         default sweep
#   BENCH_CTX=32768 deploy/bench.sh         deeper sweep
#   EXTRA_ARGS="-rtr" deploy/bench.sh       compare an option before enabling it
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

RESULTS_DIR=${RESULTS_DIR:-$INSTALL_DIR/results}

[[ -x "$IK_BIN/llama-sweep-bench" ]] || die "llama-sweep-bench not found in $IK_BIN (run deploy/build.sh)"
model=$(resolve_model)
streaming=$(resolve_streaming)
BENCH_CTX=${BENCH_CTX:-16384}
# Match serve.sh: streaming uses 2048-token prompt batches.
BENCH_UBATCH=${BENCH_UBATCH:-$([[ "$streaming" == on ]] && echo 2048 || echo 1024)}
mode=$(resolve_numa_mode "$streaming")
# One instance is what a single server process gets: node 0 in per-node mode.
placement "$mode" 0 "$streaming"

extra=()
read -r -a extra <<<"$EXTRA_ARGS"
args=(
    -m "$model"
    -t "$RUN_THREADS" -tb "$RUN_THREADS"
    -c "$BENCH_CTX" -ub "$BENCH_UBATCH" -b "$BENCH_UBATCH"
    "${PLACEMENT_ARGS[@]}" "${extra[@]}"
)

mkdir -p "$RESULTS_DIR"
out="$RESULTS_DIR/sweep-$(hostname -s)-$(basename "$model" .gguf)-$(date +%Y%m%d-%H%M%S).txt"
{
    echo "# host: $(hostname) | cpu: $(lscpu | awk -F: '/Model name/ { gsub(/^ +/, "", $2); print $2; exit }')"
    echo "# ik_llama.cpp: $(git -C "$INSTALL_DIR/ik_llama.cpp" log -1 --format='%h %cs' 2>/dev/null || echo unknown)"
    echo "# numa mode: $mode | expert streaming: $streaming | threads: $RUN_THREADS | extra: ${EXTRA_ARGS:-none}"
    echo "# cmd: ${PLACEMENT_PREFIX[*]} llama-sweep-bench ${args[*]}"
} | tee "$out"

full_log="${out%.txt}.log"
if ! "${PLACEMENT_PREFIX[@]}" "$IK_BIN/llama-sweep-bench" "${args[@]}" 2>&1 \
    | tee "$full_log" | grep -E '^\|' | tee -a "$out"; then
    die "benchmark failed; full output in $full_log"
fi

log "saved $out (full output: $full_log)"
log "S_PP / S_TG columns are prompt and generation tokens/s at context depth N_KV"
