#!/usr/bin/env bash
# Build the Tessera engine: ik_llama.cpp at the pinned commit with tessera/engine/patches, for this PC's GPU.
#
#   tessera/scripts/build.sh            # CUDA if a CUDA compiler is found, else CPU only
#   CUDA=0 tessera/scripts/build.sh     # CPU only (tests)
#
# The first build takes 20-60 minutes (the CUDA kernels); later builds only recompile what changed.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

src="$ENGINE_DIR"
mkdir -p "$INSTALL_DIR"
if [[ ! -d "$src/.git" ]]; then
    log "cloning $IK_LLAMA_REPO"
    git clone "$IK_LLAMA_REPO" "$src"
fi
if ! git -C "$src" cat-file -e "${IK_LLAMA_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$src" fetch origin
fi

# Patches apply as a stack recorded in a stamp: when the set changes, the tree is reset to the pinned commit and
# every patch is applied again (later patches edit files earlier ones add). Local edits under $src are lost then.
patches=()
for p in "$TESSERA_DIR"/engine/patches/*.patch; do [[ -e "$p" ]] && patches+=("$p"); done
stamp="$src/.tessera-patches"
want="$IK_LLAMA_COMMIT"
for p in "${patches[@]}"; do want+=$'\n'"$(basename "$p") $(sha256sum <"$p" | cut -d' ' -f1)"; done
if [[ "$(cat "$stamp" 2>/dev/null)" == "$want" ]]; then
    log "engine patches already applied: ${#patches[@]}"
else
    git -C "$src" -c advice.detachedHead=false checkout --quiet --force "$IK_LLAMA_COMMIT"
    git -C "$src" reset --quiet --hard "$IK_LLAMA_COMMIT"
    git -C "$src" clean -fdq
    for p in "${patches[@]}"; do
        git -C "$src" apply "$p" || die "patch $(basename "$p") does not apply to $IK_LLAMA_COMMIT"
        log "applied $(basename "$p")"
    done
    printf '%s\n' "$want" >"$stamp"
fi

nvcc=$(command -v nvcc || true)
[[ -z "$nvcc" && -x /usr/local/cuda/bin/nvcc ]] && nvcc=/usr/local/cuda/bin/nvcc
cuda=${CUDA:-auto}
[[ "$cuda" == auto ]] && cuda=$([[ -n "$nvcc" ]] && echo 1 || echo 0)
args=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON)
if [[ "$cuda" == 1 ]]; then
    [[ -n "$nvcc" ]] || die "CUDA=1 but no nvcc: run scripts/setup-wsl.sh"
    arch=$(nvidia_smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    arch=${CUDA_ARCH:-${arch:-89}}
    log "CUDA build for sm_$arch with $nvcc"
    args+=(-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="$arch")
else
    log "CPU-only build (no CUDA compiler, or CUDA=0)"
    args+=(-DGGML_CUDA=OFF)
fi
cmake -S "$src" -B "$src/build" "${args[@]}"
cmake --build "$src/build" --config Release -j"$(nproc)" --target llama-server llama-cli llama-bench

# capture first: with pipefail, `--help | grep -q` fails when grep exits early
help=$("$BIN/llama-server" --help 2>&1 || true)
[[ "$help" == *--hot-experts* ]] || die "the build lacks the Tessera patches"
log "built: $BIN/llama-server ($( [[ "$cuda" == 1 ]] && echo CUDA || echo CPU ))"
