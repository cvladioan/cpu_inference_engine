#!/usr/bin/env bash
# Install and start the systemd service(s). Run as root from the repository.
#
#   sudo deploy/install_service.sh [service-user]
#
# The NUMA mode is resolved once here and frozen into a drop-in, so every
# instance agrees on it across restarts.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

[[ $EUID -eq 0 ]] || die "run as root (sudo)"
user="${1:-${SUDO_USER:-root}}"
repo_dir="$(cd "$DEPLOY_DIR/.." && pwd)"

plan=$(sudo -u "$user" env NUMA_MODE="$NUMA_MODE" EXPERT_STREAMING="$EXPERT_STREAMING" EXPERT_CACHE_MIB="$EXPERT_CACHE_MIB" \
    "$DEPLOY_DIR/serve.sh" --plan)
mode=$(sed -n 's/^NUMA_MODE=//p' <<<"$plan")
streaming=$(sed -n 's/^EXPERT_STREAMING=//p' <<<"$plan")
cache_mib=$(sed -n 's/^EXPERT_CACHE_MIB=//p' <<<"$plan")
read -r -a instances <<<"$(sed -n 's/^INSTANCES=//p' <<<"$plan")"
log "NUMA mode: $mode, expert streaming: $streaming, expert cache: ${cache_mib} MiB, instances: ${instances[*]}"

unit=/etc/systemd/system/deepseek-cpu@.service
sed -e "s|@REPO_DIR@|$repo_dir|g" -e "s|@USER@|$user|g" \
    "$DEPLOY_DIR/systemd/deepseek-cpu@.service" >"$unit"
mkdir -p /etc/systemd/system/deepseek-cpu@.service.d
printf '[Service]\nEnvironment=NUMA_MODE=%s\nEnvironment=EXPERT_STREAMING=%s\nEnvironment=EXPERT_CACHE_MIB=%s\n' \
    "$mode" "$streaming" "$cache_mib" \
    >/etc/systemd/system/deepseek-cpu@.service.d/numa.conf
systemctl daemon-reload

for i in "${instances[@]}"; do
    systemctl enable --now "deepseek-cpu@$i.service"
    log "started deepseek-cpu@$i on port $(( PORT + i ))"
done
log "follow the logs with: journalctl -fu 'deepseek-cpu@*'"
log "loading takes a few minutes; then run: python3 deploy/smoke_test.py --url http://127.0.0.1:$PORT"
