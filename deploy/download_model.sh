#!/usr/bin/env bash
# Download the chosen GGUF quant of DeepSeek-V4-Flash from Hugging Face (resumable).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

if command -v hf >/dev/null; then
    dl=(hf download)
elif command -v huggingface-cli >/dev/null; then
    dl=(huggingface-cli download)
else
    die "Hugging Face CLI not found: pip install -U 'huggingface_hub[cli,hf_xet]'"
fi

mkdir -p "$MODEL_DIR"
log "downloading ${HF_REPO} (${QUANT}/*) into ${MODEL_DIR}; this is ~160 GB and resumes if interrupted"
"${dl[@]}" "$HF_REPO" --include "${QUANT}/*" --local-dir "$MODEL_DIR"

model=$(resolve_model)
touch "$MODEL_DIR/.download-complete-$QUANT"
log "done: $model ($(( $(model_size_mib "$model") / 1024 )) GB)"
