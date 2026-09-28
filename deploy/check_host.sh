#!/usr/bin/env bash
# Report whether this server can run DeepSeek-V4-Flash CPU-only, and roughly how fast.
# Run as root to also see the DIMM population (dmidecode). Exit code 1 = blocking problem.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
load_config

fail=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=1; }

# Approximate file sizes (GB) of the Unsloth quants of DeepSeek-V4-Flash-0731.
case "$QUANT" in
    UD-Q8_K_XL) model_gb=162 ;;
    UD-Q4_K_XL) model_gb=155 ;;
    *)          model_gb=170 ;;
esac

echo "== CPU"
lscpu | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|NUMA node\(s\))' | sed 's/^/  /'
flags=$(grep -m1 '^flags' /proc/cpuinfo)
has() { [[ " $flags " == *" $1 "* ]]; }
has avx2 && ok "AVX2" || bad "no AVX2: ik_llama.cpp needs at least AVX2"
if has avx512f && has avx512_vnni; then ok "AVX-512 + VNNI (fast quantized kernels)"
else warn "no AVX-512 VNNI: expect noticeably slower prompt processing"; fi
has avx512_bf16 && ok "AVX512-BF16" || warn "no AVX512-BF16"
has amx_int8 && ok "AMX (Intel Sapphire Rapids or newer)" || echo "  [info] no AMX (AMD, or Intel before Sapphire Rapids)"

echo "== Memory"
total_gb=$(awk '/MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
avail_gb=$(( $(mem_available_mib) / 1024 ))
echo "  total ${total_gb} GB, available ${avail_gb} GB; ${QUANT} needs ~${model_gb} GB + KV cache + prompt cache"
need_gb=$(( model_gb + 16 + CACHE_RAM_MIB / 1024 ))
streaming_needed=0
if (( avail_gb >= need_gb )); then ok "enough RAM for one copy (~${need_gb} GB)"
elif (( avail_gb >= model_gb + 8 )); then warn "tight: lower CACHE_RAM_MIB / PARALLEL * CTX_PER_SLOT"
elif [[ "$EXPERT_STREAMING" == off ]]; then
    bad "not enough RAM for ${QUANT} (~${model_gb} GB) and EXPERT_STREAMING=off"
elif (( avail_gb < 24 )); then
    bad "only ${avail_gb} GB available: too little even for streaming experts from SSD (need 24+ GB)"
else
    streaming_needed=1
    warn "model is larger than RAM: experts will stream from SSD (EXPERT_STREAMING), expect a few tok/s"
    echo "         (~$(( (avail_gb - 12) * 100 / model_gb ))% of the model fits in the page cache; see docs/PLAN.md section 12)"
fi

nodes=$(numa_nodes | wc -l)
if (( nodes > 1 )); then
    for n in $(numa_nodes); do
        echo "  node $n: $(physical_cores "$n") physical cores, $(( $(mem_available_mib "$n") / 1024 )) GB available"
    done
fi

peak_gbs=""
if [[ $EUID -eq 0 ]] && command -v dmidecode >/dev/null; then
    dimms=$(dmidecode -t memory | awk -F: '/^\tSize:/ && $2 !~ /No Module/ { n++ } END { print n + 0 }')
    slots=$(dmidecode -t memory | grep -c $'^\tSize:')
    speed=$(dmidecode -t memory | awk -F: '/Configured Memory Speed:/ && $2 ~ /[0-9]/ { gsub(/[^0-9]/, "", $2); print $2; exit }')
    echo "  DIMMs: ${dimms} populated of ${slots} slots, configured speed ${speed:-unknown} MT/s"
    if [[ -n "$speed" ]] && (( dimms > 0 )); then
        # Upper bound: assumes one DIMM per channel. Two DIMMs per channel share a channel.
        peak_gbs=$(( dimms * speed * 8 / 1000 ))
        echo "  peak bandwidth if 1 DIMM per channel: ~${peak_gbs} GB/s"
    fi
    (( slots > dimms )) && warn "empty DIMM slots: fill every memory channel, decode speed scales with channels"
elif [[ $EUID -eq 0 ]]; then
    echo "  [info] install dmidecode to read DIMM population and speed"
else
    echo "  [info] run as root to read DIMM population and speed (dmidecode)"
fi

echo "== OS"
thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unknown)
[[ "$thp" == *"[always]"* || "$thp" == *"[madvise]"* ]] && ok "transparent hugepages: $thp" \
    || warn "transparent hugepages disabled ($thp)"
swap_kb=$(awk '/SwapTotal/ { print $2 }' /proc/meminfo)
(( swap_kb > 0 )) && warn "swap enabled: keep MLOCK=1 so the model is never paged out" || ok "no swap"
if (( nodes > 1 )); then
    command -v numactl >/dev/null && ok "numactl installed" || bad "numactl missing (apt install numactl / dnf install numactl)"
fi
memlock=$(ulimit -l)
[[ "$memlock" == unlimited ]] && ok "memlock unlimited" \
    || echo "  [info] memlock limit ${memlock} KB in this shell; the systemd unit sets LimitMEMLOCK=infinity"

echo "== Disk"
mkdir -p "$INSTALL_DIR" 2>/dev/null || true
dev=$(df --output=source "$INSTALL_DIR" 2>/dev/null | tail -n1)
if command -v lsblk >/dev/null && [[ "$dev" == /dev/* ]]; then
    # Bus and rotational flag live on the whole disk, not on the partition.
    disk="$dev"
    parent=$(lsblk -ndo PKNAME "$dev" 2>/dev/null || true)
    [[ -n "$parent" ]] && disk="/dev/$parent"
    rota="" tran=""
    read -r rota tran < <(lsblk -ndo ROTA,TRAN "$disk" 2>/dev/null || true)
    [[ -z "$tran" && "$disk" == /dev/vd* ]] && tran=virtio
    echo "  $INSTALL_DIR is on $dev (bus: ${tran:-unknown}, rotational: ${rota:-unknown})"
    if (( streaming_needed )); then
        if [[ "$tran" == nvme ]]; then ok "NVMe SSD for expert streaming"
        elif [[ "$tran" == virtio ]]; then warn "virtual disk: expert streaming speed depends on the host storage"
        elif [[ "$rota" == 1 ]]; then bad "expert streaming needs an SSD; this is a spinning disk"
        else warn "not NVMe: expert streaming from SATA (~0.5 GB/s) will be very slow"; fi
    fi
fi
disk_gb=$(df -BG --output=avail "$INSTALL_DIR" 2>/dev/null | tail -n1 | tr -dc 0-9)
if [[ -f "$MODEL_DIR/.download-complete-$QUANT" ]]; then ok "model ${QUANT} already downloaded"
elif (( ${disk_gb:-0} >= model_gb + 10 )); then ok "${disk_gb} GB free in $INSTALL_DIR"
else bad "${disk_gb:-?} GB free in $INSTALL_DIR, need ~$(( model_gb + 10 )) GB for ${QUANT}"; fi

echo "== Rough decode estimate (single user, one socket)"
if [[ -n "$peak_gbs" ]]; then
    per_socket=$(( peak_gbs / $(lscpu | awk -F: '/^Socket\(s\)/ { print $2 + 0 }') ))
    # ~9.6 GB read per token for native-precision V4-Flash; 45-75% of 78% of peak bandwidth.
    awk -v bw="$per_socket" 'BEGIN { printf "  ~%d-%d tok/s (see docs/PLAN.md section 5)\n", bw * 0.78 * 0.45 / 9.6, bw * 0.78 * 0.75 / 9.6 }'
else
    echo "  needs DIMM data (run as root); see the table in docs/PLAN.md section 5"
fi

echo
(( fail )) && { echo "RESULT: blocking problems above"; exit 1; }
echo "RESULT: ready"
