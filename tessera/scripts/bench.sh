#!/usr/bin/env bash
# Measure decode/prompt speed per configuration on this PC (tools/bench.py; stop the server first).
#
#   tessera/scripts/bench.sh                           # every expert on the CPU vs hot experts on the GPU
#   tessera/scripts/bench.sh --modes hot --threads 6,8,10
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config
python "$TESSERA_DIR/tools/bench.py" "$@"
