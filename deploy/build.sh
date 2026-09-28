#!/usr/bin/env bash
# Build ik_llama.cpp (CPU only) at the pinned commit, tuned for this machine.
# Run it on the server that will serve the model: -march=native targets its CPU.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

src="$INSTALL_DIR/ik_llama.cpp"
mkdir -p "$INSTALL_DIR"

if [[ ! -d "$src/.git" ]]; then
    log "cloning $IK_LLAMA_REPO into $src"
    git clone "$IK_LLAMA_REPO" "$src"
fi
if ! git -C "$src" cat-file -e "${IK_LLAMA_COMMIT}^{commit}" 2>/dev/null; then
    log "fetching commit $IK_LLAMA_COMMIT"
    git -C "$src" fetch origin
fi
git -C "$src" -c advice.detachedHead=false checkout --quiet "$IK_LLAMA_COMMIT"
log "ik_llama.cpp at $(git -C "$src" log -1 --format='%h %cs %s')"

# GGML_NATIVE=ON compiles for this CPU (-march=native): AVX-512/VNNI/BF16 and AMX
# are enabled when present. Do not copy the binaries to a different CPU model.
cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF \
    -DLLAMA_CURL=OFF
cmake --build "$src/build" --config Release -j"$(nproc)" \
    --target llama-server llama-sweep-bench llama-bench llama-cli

# Confirm the fast quantized kernels were compiled in (see ik_llama.cpp docs/build.md).
if grep -qw -E 'avx512_vnni|avx_vnni' /proc/cpuinfo; then
    # The kernels live in the ggml shared library (or in the binary for static builds).
    lib=$(find "$src/build" -name 'libggml.so' -print -quit)
    n=$(objdump -d "${lib:-$src/build/bin/llama-server}" 2>/dev/null | grep -c vpdpbusd || true)
    if [[ "${n:-0}" -lt 20 ]]; then
        log "WARNING: CPU has VNNI but only ${n:-0} vpdpbusd instructions in ggml;"
        log "         the quantized kernels may have fallen back to plain AVX2 (check compiler version)."
    else
        log "VNNI kernels present ($n vpdpbusd instructions)"
    fi
fi

log "built: $IK_BIN/llama-server"
"$IK_BIN/llama-server" --version 2>&1 | tail -n2 || true
