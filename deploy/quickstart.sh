#!/usr/bin/env bash
# From a fresh Linux server to a tuned DeepSeek-V4-Flash API, in one command.
#
#   deploy/quickstart.sh                          check, build, download, tune for 20 tok/s
#   deploy/quickstart.sh --target 25 --service deepseek
#                                                 ... then install and start the systemd service
#
# Each step is skipped or resumed when already done, so rerunning is safe.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

target=20
service_user=""
while (( $# )); do
    case "$1" in
        --target) target="$2"; shift 2 ;;
        --service) service_user="$2"; shift 2 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

log "1/5 checking the host"
"$DEPLOY_DIR/check_host.sh" || die "fix the [FAIL] items above, then rerun"

log "2/5 building ik_llama.cpp"
if [[ -x "$IK_BIN/llama-server" ]] \
    && [[ "$(git -C "$INSTALL_DIR/ik_llama.cpp" rev-parse HEAD 2>/dev/null)" == "$IK_LLAMA_COMMIT"* ]]; then
    log "already built at ${IK_LLAMA_COMMIT:0:10}"
else
    "$DEPLOY_DIR/build.sh"
fi

log "3/5 downloading the model"
if model=$(resolve_model 2>/dev/null) && [[ -f "$MODEL_DIR/.download-complete-$QUANT" || -n "$MODEL_FILE" ]] \
    && { [[ -z "$DRAFT_REPO" ]] || [[ -n "$(resolve_draft)" ]]; }; then
    log "already present: $model"
else
    "$DEPLOY_DIR/download_model.sh"
fi

log "4/5 tuning for >= ${target} tok/s (this loads the model several times)"
rc=0
python3 "$DEPLOY_DIR/tune.py" --target "$target" --apply || rc=$?
case "$rc" in
    0) ;;
    1) log "WARNING: below ${target} tok/s on this machine; the best configuration was still applied" ;;
    *) die "tuning failed; see the server logs listed above" ;;
esac

log "5/5 starting"
if [[ -n "$service_user" ]]; then
    sudo "$DEPLOY_DIR/install_service.sh" "$service_user"
else
    log "run deploy/serve.sh (foreground) or: sudo deploy/install_service.sh <user>"
fi
