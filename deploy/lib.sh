# shellcheck shell=bash
# Shared helpers for the deploy scripts. Source it; do not execute it.
# Variables set here (IK_BIN, PLACEMENT_*, RUN_THREADS) are read by the callers.
# shellcheck disable=SC2034

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load settings: defaults from config.env.example, then config.env overrides.
# Variables already set in the environment win over both files.
load_config() {
    local file line key
    for file in "$DEPLOY_DIR/config.env.example" "$DEPLOY_DIR/config.env"; do
        [[ -f "$file" ]] || continue
        while IFS= read -r line || [[ -n "$line" ]]; do
            [[ "$line" =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)= ]] || continue
            key="${BASH_REMATCH[1]}"
            # Environment overrides are captured once, before any file is read.
            if [[ -n "${__ENV_OVERRIDES[$key]+x}" ]]; then
                continue
            fi
            eval "$line"
        done <"$file"
    done
    for key in "${!__ENV_OVERRIDES[@]}"; do
        export "$key=${__ENV_OVERRIDES[$key]}"
    done
    IK_BIN="$INSTALL_DIR/ik_llama.cpp/build/bin"
}

declare -A __ENV_OVERRIDES=()
for __var in INSTALL_DIR IK_LLAMA_REPO IK_LLAMA_COMMIT HF_REPO QUANT MODEL_DIR MODEL_FILE \
    MODEL_ALIAS HOST PORT API_KEY API_KEY_FILE NUMA_MODE THREADS PARALLEL CTX_PER_SLOT CACHE_RAM_MIB \
    MLOCK SPEC_TYPE EXTRA_ARGS; do
    if [[ -n "${!__var+x}" ]]; then
        __ENV_OVERRIDES[$__var]="${!__var}"
    fi
done
unset __var

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# Path of the model file to load (first shard of a split GGUF).
resolve_model() {
    if [[ -n "$MODEL_FILE" ]]; then
        [[ -f "$MODEL_FILE" ]] || die "MODEL_FILE=$MODEL_FILE does not exist"
        echo "$MODEL_FILE"
        return
    fi
    local dir="$MODEL_DIR/$QUANT" found
    [[ -d "$dir" ]] || dir="$MODEL_DIR"
    found=$(find "$dir" -maxdepth 2 -name '*.gguf' \( -name '*-00001-of-*' -o ! -name '*-of-*' \) \
        -path "*${QUANT}*" 2>/dev/null | sort | head -n1)
    [[ -n "$found" ]] || die "no ${QUANT} .gguf found under $MODEL_DIR (run deploy/download_model.sh or set MODEL_FILE)"
    echo "$found"
}

numa_nodes() {
    if command -v lscpu >/dev/null; then
        lscpu -p=NODE | grep -v '^#' | sort -un | grep -E '^[0-9]+$' || echo 0
    else
        echo 0
    fi
}

# Physical cores, optionally restricted to one NUMA node.
physical_cores() {
    local node="${1:-}"
    lscpu -p=CORE,SOCKET,NODE | grep -v '^#' | awk -F, -v n="$node" \
        'n == "" || $3 == n { print $2 ":" $1 }' | sort -u | wc -l
}

# Available memory in MiB on a NUMA node (from /sys), or system-wide if no node given.
mem_available_mib() {
    local node="${1:-}"
    if [[ -n "$node" && -r /sys/devices/system/node/node$node/meminfo ]]; then
        # Node meminfo has no MemAvailable; free + file pages is a fair estimate.
        awk '/MemFree|FilePages/ { s += $4 } END { print int(s / 1024) }' \
            "/sys/devices/system/node/node$node/meminfo"
    else
        awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo
    fi
}

model_size_mib() {
    local first="$1" dir base
    dir=$(dirname "$first")
    base=$(basename "$first")
    if [[ "$base" =~ ^(.*)-00001-of-([0-9]+)\.gguf$ ]]; then
        du -cm "$dir/${BASH_REMATCH[1]}"-*-of-"${BASH_REMATCH[2]}".gguf | tail -n1 | cut -f1
    else
        du -m "$first" | cut -f1
    fi
}

# Resolve NUMA_MODE=auto into none, interleave or per-node for this machine.
resolve_numa_mode() {
    local mode="$NUMA_MODE" nodes model_mib n
    nodes=$(numa_nodes | wc -l)
    if [[ "$mode" == auto ]]; then
        if (( nodes <= 1 )); then
            mode=none
        else
            mode=per-node
            model_mib=$(model_size_mib "$(resolve_model)")
            for n in $(numa_nodes); do
                # One full copy per node plus KV/prompt cache headroom.
                if (( $(mem_available_mib "$n") < model_mib * 11 / 10 + CACHE_RAM_MIB + 8192 )); then
                    mode=interleave
                fi
            done
        fi
    fi
    case "$mode" in
        none|interleave|per-node) ;;
        *) die "NUMA_MODE must be auto, none, interleave or per-node (got '$mode')" ;;
    esac
    if [[ "$mode" != none ]] && (( nodes > 1 )) && ! command -v numactl >/dev/null; then
        die "NUMA_MODE=$mode needs numactl (apt install numactl / dnf install numactl)"
    fi
    echo "$mode"
}

# Fill PLACEMENT_PREFIX (command prefix) and PLACEMENT_ARGS (llama.cpp args) for
# running on NUMA node $2 under mode $1, plus RUN_THREADS.
placement() {
    local mode="$1" node="${2:-0}"
    PLACEMENT_PREFIX=()
    PLACEMENT_ARGS=()
    case "$mode" in
        none)
            RUN_THREADS=${THREADS:-$(physical_cores)}
            ;;
        interleave)
            # Anonymous memory so the interleave policy decides page placement.
            PLACEMENT_PREFIX=(numactl --interleave=all)
            PLACEMENT_ARGS=(--numa distribute --no-mmap)
            RUN_THREADS=${THREADS:-$(physical_cores)}
            ;;
        per-node)
            # --no-mmap gives each node its own local copy instead of sharing
            # one page-cache copy that lives on a single node.
            PLACEMENT_PREFIX=(numactl --cpunodebind="$node" --membind="$node")
            PLACEMENT_ARGS=(--numa numactl --no-mmap)
            RUN_THREADS=${THREADS:-$(physical_cores "$node")}
            ;;
    esac
}
