#!/usr/bin/env bash
# Build the hot-expert profile for this model: run it once with every expert on the CPU while it answers the
# calibration prompts, record which experts each layer routes to, and rank them.
#
#   tessera/scripts/calibrate.sh                         # prompts/calibration.txt, 160 tokens each
#   PROMPTS=my-prompts.txt TOKENS=300 tessera/scripts/calibrate.sh
#
# Use prompts like your own use (code, chat, languages): the profile decides which experts live in VRAM.
# Takes 5-20 minutes (every expert runs on the CPU meanwhile). Writes $INSTALL_DIR/profiles/<model>.profile.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

model=$(resolve_model)
profile=$(resolve_profile "$model")
if (( $(cpu_only_mode) )); then
    log "CPU only: the profile chooses the experts kept in VRAM, so there is nothing to calibrate without a GPU"
    exit 0
fi
prompts=${PROMPTS:-$TESSERA_DIR/prompts/calibration.txt}
tokens=${TOKENS:-160}
port=${CALIBRATE_PORT:-18181}
mkdir -p "$PROFILES_DIR"
trace="$PROFILES_DIR/$(model_tag "$model").trace"
rm -f "$trace"

log "calibrating $(basename "$model") on $(grep -c . "$prompts") prompts"
TESSERA_ROUTING_TRACE="$trace" HOST=127.0.0.1 PORT=$port API_KEY='' "$TESSERA_DIR/scripts/serve.sh" --no-hot \
    >"$PROFILES_DIR/calibrate-server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 900); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
    kill -0 $pid 2>/dev/null || { tail -20 "$PROFILES_DIR/calibrate-server.log"; die "the server exited"; }
    sleep 2
done
python - "$port" "$prompts" "$tokens" <<'EOF'
import json, sys, time, urllib.error, urllib.request
port, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
done = 0
for i, p in enumerate(prompts, 1):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=3600))
    except urllib.error.HTTPError as e:
        print(f"  {i}/{len(prompts)}: rejected ({e.code}), skipped", flush=True)
        continue
    done += 1
    u = r.get("usage", {})
    print(f"  {i}/{len(prompts)}: {u.get('completion_tokens', '?')} tokens in {time.time() - t:.0f} s", flush=True)
sys.exit(0 if done else "no prompt was answered")
EOF
kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true
trap - EXIT
python "$TESSERA_DIR/tools/make_profile.py" "$trace" --out "$profile"
log "profile: $profile (restart scripts/serve.sh to use it)"
