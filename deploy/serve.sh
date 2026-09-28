#!/usr/bin/env bash
# Start the OpenAI-compatible DeepSeek-V4-Flash server (ik_llama.cpp llama-server).
#
#   deploy/serve.sh            start every instance this machine needs (foreground)
#   deploy/serve.sh --node N   start only the instance for NUMA node N (used by systemd)
#   deploy/serve.sh --plan     print the resolved NUMA mode and instances, then exit
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

only_node=""
plan=0
while (( $# )); do
    case "$1" in
        --node) only_node="$2"; shift 2 ;;
        --plan) plan=1; shift ;;
        -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

model=$(resolve_model)
streaming=$(resolve_streaming)
mode=$(resolve_numa_mode "$streaming")
if [[ "$mode" == per-node ]]; then
    mapfile -t instances < <(numa_nodes)
else
    instances=(0)
fi

if (( plan )); then
    echo "NUMA_MODE=$mode"
    echo "EXPERT_STREAMING=$streaming"
    echo "INSTANCES=${instances[*]}"
    exit 0
fi

[[ -x "$IK_BIN/llama-server" ]] || die "llama-server not found in $IK_BIN (run deploy/build.sh)"

run_instance() {
    local node="$1" port=$(( PORT + $1 ))
    placement "$mode" "$node" "$streaming"

    local args=(
        -m "$model" -a "$MODEL_ALIAS"
        --host "$HOST" --port "$port"
        -t "$RUN_THREADS" -tb "$RUN_THREADS"
        -c $(( PARALLEL * CTX_PER_SLOT )) -np "$PARALLEL"
        --jinja --reasoning-format deepseek
        -cram "$CACHE_RAM_MIB"
        "${PLACEMENT_ARGS[@]}"
    )
    [[ -n "$API_KEY_FILE" ]] && args+=(--api-key-file "$API_KEY_FILE")
    if [[ "$streaming" == on ]]; then
        log "streaming experts from SSD (EXPERT_STREAMING): expect much lower speed than all-in-RAM"
        # Each prompt batch touches nearly every expert, i.e. one pass over the SSD;
        # bigger batches spread that pass over more tokens (5x faster prompts at 1024 vs 128).
        args+=(-b 2048 -ub 2048)
    elif [[ "$MLOCK" == 1 ]]; then
        if [[ "$(ulimit -l)" == unlimited ]]; then
            args+=(--mlock)
        else
            log "memlock limit is $(ulimit -l) KB: skipping --mlock (the systemd unit sets LimitMEMLOCK=infinity)"
        fi
    fi
    if [[ -n "$SPEC_TYPE" ]]; then
        args+=(--spec-type "$SPEC_TYPE")
        # Stages that run a separate draft model need -md.
        if [[ "$SPEC_TYPE" =~ ^(dspark|dflash|draft|mtp) ]]; then
            local draft
            draft=$(resolve_draft)
            if [[ -n "$draft" ]]; then
                args+=(-md "$draft")
            elif [[ ! "$SPEC_TYPE" =~ ^mtp ]]; then
                die "SPEC_TYPE=$SPEC_TYPE needs a draft model: set DRAFT_REPO (and run download_model.sh) or DRAFT_FILE"
            fi
        fi
    fi
    local extra=()
    read -r -a extra <<<"$EXTRA_ARGS"
    args+=("${extra[@]}")

    log "instance $node: mode=$mode streaming=$streaming threads=$RUN_THREADS port=$port model=$(basename "$model")"
    log "exec: ${PLACEMENT_PREFIX[*]} $IK_BIN/llama-server ${args[*]}"
    if [[ -n "$API_KEY" && -z "$API_KEY_FILE" ]]; then
        # Pass the key through a pipe so it never shows up in `ps` or on disk.
        exec "${PLACEMENT_PREFIX[@]}" "$IK_BIN/llama-server" "${args[@]}" \
            --api-key-file <(printf '%s\n' "$API_KEY")
    fi
    exec "${PLACEMENT_PREFIX[@]}" "$IK_BIN/llama-server" "${args[@]}"
}

if [[ -n "$only_node" ]]; then
    [[ " ${instances[*]} " == *" $only_node "* ]] \
        || die "node $only_node is not an instance in NUMA_MODE=$mode (instances: ${instances[*]})"
    run_instance "$only_node"
fi

if (( ${#instances[@]} == 1 )); then
    run_instance "${instances[0]}"
fi

# Several per-node instances: run them all and stop them together.
pids=()
trap 'kill "${pids[@]}" 2>/dev/null; wait' EXIT
trap 'log "stopping all instances"; exit 0' INT TERM
for n in "${instances[@]}"; do
    ( run_instance "$n" ) &
    pids+=($!)
done
wait -n "${pids[@]}"
die "an instance exited; stopping the others"
