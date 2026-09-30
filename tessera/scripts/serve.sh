#!/usr/bin/env bash
# Start the OpenAI-compatible server with Tessera's tiers: the dense part and the hot experts on the GPU, the other
# experts on the CPU from RAM, and an SSD tier when the experts do not fit in RAM.
#
#   tessera/scripts/serve.sh                  # foreground; Ctrl+C stops it
#   tessera/scripts/serve.sh --plan           # print the plan and the command, then exit
#   tessera/scripts/serve.sh --no-hot         # every routed expert on the CPU (the baseline; calibrate.sh uses it)
#
# Without a GPU (CPU_ONLY=1, or a CPU-only build) everything runs on the CPU from RAM, plus the SSD tier when the
# model does not fit.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

plan_only=0 no_hot=0
while (( $# )); do
    case "$1" in
        --plan) plan_only=1 ;;
        --no-hot) no_hot=1 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
    shift
done

model=$(resolve_model)
profile=$(resolve_profile "$model")
cpu_only=$(cpu_only_mode)
plan_args=(--ctx "$CTX")
(( cpu_only )) && plan_args+=(--cpu-only)
planned=$(python "$TESSERA_DIR/tools/plan.py" "$model" "${plan_args[@]}" --env)
planned_hot=$(sed -n 's/^HOT_MIB=//p' <<<"$planned")
planned_cache=$(sed -n 's/^CACHE_MIB=//p' <<<"$planned")
hot=$HOT_MIB; [[ "$hot" == auto ]] && hot=$planned_hot
cache=$CACHE_MIB; [[ "$cache" == auto ]] && cache=$planned_cache

if (( cpu_only )); then
    args=(-m "$model" -c "$CTX" -np 1 -t "$THREADS" -tb "$THREADS"
          -ctk q8_0 -ctv q8_0 -ub 512 --jinja --host "$HOST" --port "$PORT")
    hot=0
else
    args=(-m "$model" -ngl 999 --cpu-moe -c "$CTX" -np 1 -t "$THREADS" -tb "$THREADS"
          -ctk q8_0 -ctv q8_0 -ub 1024 --jinja --host "$HOST" --port "$PORT")
fi
if (( no_hot || cpu_only )); then
    hot=0
elif [[ ! -f "$profile" ]]; then   # (a GPU run without a profile)
    log "no hot-expert profile at $profile: every expert runs on the CPU. Run scripts/calibrate.sh once for the GPU tier."
    hot=0
fi
if (( hot > 0 )); then
    args+=(--hot-experts "$profile" --hot-experts-mib "$hot")
fi
if (( cache > 0 )); then
    args+=(--defer-experts --expert-cache "$cache")
fi
[[ -n "$API_KEY" ]] && args+=(--api-key "$API_KEY")
extra=()
read -r -a extra <<<"${EXTRA_ARGS:-}"
args+=("${extra[@]}")

log "model $(basename "$model"), ctx $CTX, threads $THREADS$( (( cpu_only )) && echo ", CPU only")"
log "tiers: hot experts ${hot} MiB in VRAM$( (( hot > 0 )) && echo " ($profile)"), SSD tier $( (( cache > 0 )) && echo "on, RAM cache ${cache} MiB" || echo off)"
if (( plan_only )); then
    python "$TESSERA_DIR/tools/plan.py" "$model" "${plan_args[@]}" --estimate
    echo "$BIN/llama-server ${args[*]}"
    exit 0
fi
[[ -x "$BIN/llama-server" ]] || die "no engine at $BIN: run scripts/build.sh"
exec "$BIN/llama-server" "${args[@]}"
