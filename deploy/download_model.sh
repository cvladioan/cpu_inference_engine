#!/usr/bin/env bash
# Download the chosen GGUF quant (HF_REPO, QUANT) from Hugging Face (resumable).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

# Large models are split into a QUANT/ folder; small ones are one file named *QUANT*.gguf.
if command -v hf >/dev/null; then
    dl=(hf download "$HF_REPO" --include "${QUANT}/*" --include "*${QUANT}*.gguf")
elif command -v huggingface-cli >/dev/null; then
    dl=(huggingface-cli download "$HF_REPO" --include "${QUANT}/*" "*${QUANT}*.gguf")
else
    die "Hugging Face CLI not found: pip install -U 'huggingface_hub[cli,hf_xet]'"
fi

mkdir -p "$MODEL_DIR"
log "downloading ${HF_REPO} (${QUANT}) into ${MODEL_DIR}; resumes if interrupted"
"${dl[@]}" --local-dir "$MODEL_DIR"

model=$(resolve_model)
touch "$MODEL_DIR/.download-complete-$QUANT"
log "done: $model ($(( $(model_size_mib "$model") / 1024 )) GB)"
