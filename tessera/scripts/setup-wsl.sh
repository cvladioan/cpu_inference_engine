#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu (also works on a native Ubuntu with an NVIDIA driver installed):
# build tools, the CUDA compiler, and a Python environment with the Hugging Face downloader.
#
#   tessera/scripts/setup-wsl.sh
#
# The CUDA toolkit comes from NVIDIA's WSL-Ubuntu repository on purpose: Ubuntu's own nvidia-cuda-toolkit package
# pulls in a Linux copy of the driver library (libnvidia-compute), which in WSL hides the GPU driver that Windows
# provides. The WSL repository's cuda-toolkit has no driver in it.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

CUDA_PKG=${CUDA_PKG:-cuda-toolkit-12-8}   # 12.8: also builds for RTX 50 (sm_120)
cpu_only=0
[[ "${CPU_ONLY:-auto}" =~ ^(1|yes|true)$ ]] && cpu_only=1

if (( cpu_only )); then
    log "CPU_ONLY=1: no GPU check, no CUDA toolkit"
elif ! { log "checking the GPU"; nvidia_smi -L; }; then
    if is_wsl; then
        die "WSL cannot see an NVIDIA GPU: install or update the NVIDIA driver in Windows (not inside WSL), then run 'wsl --shutdown' in PowerShell"
    fi
    die "no NVIDIA GPU driver found (nvidia-smi)"
fi

log "installing build tools"
sudo apt-get update -q
sudo apt-get install -y -q build-essential cmake git python3-venv curl wget ca-certificates

if (( cpu_only )); then
    :
elif [[ -x /usr/local/cuda/bin/nvcc ]] || command -v nvcc >/dev/null; then
    log "CUDA compiler already installed: $( (command -v nvcc || echo /usr/local/cuda/bin/nvcc) | head -1)"
else
    log "installing $CUDA_PKG from NVIDIA's repository"
    . /etc/os-release
    if is_wsl; then
        repo="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64"
    else
        repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64"
    fi
    tmp=$(mktemp -d)
    wget -q -O "$tmp/cuda-keyring.deb" "$repo/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "$tmp/cuda-keyring.deb"
    rm -rf "$tmp"
    sudo apt-get update -q
    sudo apt-get install -y -q "$CUDA_PKG"
fi
if dpkg -l 2>/dev/null | grep -qE '^ii +libnvidia-compute-' && is_wsl; then
    log "WARNING: a Linux NVIDIA driver library (libnvidia-compute) is installed; in WSL it can hide the Windows"
    log "         driver. If CUDA fails later: sudo apt-get remove 'libnvidia-compute-*' nvidia-cuda-toolkit"
fi

log "Python environment for the downloader: $VENV"
mkdir -p "$INSTALL_DIR"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip "huggingface_hub[cli,hf_xet]"

log "done. Next: tessera/scripts/build.sh"
