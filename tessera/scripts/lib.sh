# shellcheck shell=bash
# Shared by the Tessera scripts: configuration and paths.  Sourced, not run.

TESSERA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# tessera.env.example, then tessera.env; a variable already set in the environment wins over both.
load_config() {
    local file line key
    declare -gA __TESSERA_ENV=()
    while IFS='=' read -r key _; do
        [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] && __TESSERA_ENV[$key]=1
    done < <(env)
    for file in "$TESSERA_DIR/tessera.env.example" "$TESSERA_DIR/tessera.env"; do
        [[ -f "$file" ]] || continue
        while IFS= read -r line || [[ -n "$line" ]]; do
            line="${line%%#*}"
            [[ "$line" =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)= ]] || continue
            key="${BASH_REMATCH[1]}"
            [[ -n "${__TESSERA_ENV[$key]+x}" ]] && continue
            eval "export $line"
        done <"$file"
    done
    ENGINE_DIR="$INSTALL_DIR/ik_llama.cpp"
    # shellcheck disable=SC2034  # used by the scripts that source this file
    BIN="$ENGINE_DIR/build/bin"
    MODELS_DIR="$INSTALL_DIR/models"
    PROFILES_DIR="$INSTALL_DIR/profiles"
    VENV="$INSTALL_DIR/venv"
}

# The model's .gguf (first shard of a split model): MODEL_FILE, or the download for MODEL_REPO / MODEL_PATTERN.
resolve_model() {
    if [[ -n "$MODEL_FILE" ]]; then
        [[ -f "$MODEL_FILE" ]] || die "MODEL_FILE not found: $MODEL_FILE"
        echo "$MODEL_FILE"
        return
    fi
    local dir="$MODELS_DIR/${MODEL_REPO//\//__}" f
    f=$(find "$dir" -name "*${MODEL_PATTERN}*.gguf" \( -name '*-00001-of-*' -o ! -name '*-of-*' \) 2>/dev/null |
        grep -v mmproj | sort | head -1)
    [[ -n "$f" ]] || die "no *${MODEL_PATTERN}*.gguf under $dir: run scripts/download.sh (or set MODEL_FILE)"
    echo "$f"
}

model_tag() { basename "$1" .gguf | sed -E 's/-[0-9]{5}-of-[0-9]{5}$//'; }

resolve_profile() {
    local model="$1"
    if [[ -n "$PROFILE" ]]; then echo "$PROFILE"; return; fi
    echo "$PROFILES_DIR/$(model_tag "$model").profile"
}

nvidia_smi() {
    if command -v nvidia-smi >/dev/null; then nvidia-smi "$@"
    elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then /usr/lib/wsl/lib/nvidia-smi "$@"
    else return 127; fi
}

is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

# 1 when the model runs on the CPU alone: CPU_ONLY=1, or (auto) the engine was built without CUDA or no NVIDIA
# GPU is visible. Then everything lives in RAM (and the SSD tier), and there are no hot experts.
cpu_only_mode() {
    case "${CPU_ONLY:-auto}" in
        1|yes|true) echo 1; return ;;
        0|no|false) echo 0; return ;;
    esac
    if grep -q '^GGML_CUDA:BOOL=ON' "$ENGINE_DIR/build/CMakeCache.txt" 2>/dev/null && nvidia_smi -L >/dev/null 2>&1; then
        echo 0
    else
        echo 1
    fi
}

python() { if [[ -x "$VENV/bin/python" ]]; then "$VENV/bin/python" "$@"; else python3 "$@"; fi; }
