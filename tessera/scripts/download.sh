#!/usr/bin/env bash
# Download the model named by MODEL_REPO and MODEL_PATTERN (resumable: run it again after an interruption).
#
#   tessera/scripts/download.sh            # download
#   tessera/scripts/download.sh --list     # the repository's .gguf files and sizes, to choose MODEL_PATTERN
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

[[ -x "$VENV/bin/hf" ]] || die "no Hugging Face CLI in $VENV: run scripts/setup-wsl.sh"
if [[ "${1:-}" == --list ]]; then
    "$VENV/bin/python" - "$MODEL_REPO" <<'EOF'
import sys
from huggingface_hub import HfApi
info = HfApi().model_info(sys.argv[1], files_metadata=True)
groups = {}
for s in info.siblings:
    if s.rfilename.endswith(".gguf"):
        key = s.rfilename.split("/")[0] if "/" in s.rfilename else s.rfilename
        groups[key] = groups.get(key, 0) + (s.size or 0)
for k, v in sorted(groups.items(), key=lambda kv: kv[1]):
    print(f"{v / 2**30:7.1f} GiB  {k}")
EOF
    exit 0
fi
dir="$MODELS_DIR/${MODEL_REPO//\//__}"
mkdir -p "$dir"
log "downloading $MODEL_REPO ($MODEL_PATTERN) into $dir"
"$VENV/bin/hf" download "$MODEL_REPO" --include "*${MODEL_PATTERN}*.gguf" --include "${MODEL_PATTERN}/*" --local-dir "$dir"
log "model: $(resolve_model)"
