# Run DeepSeek-V4-Flash on a CPU server

This directory serves DeepSeek-V4-Flash (284B parameters in total, 13B used
per token) from CPU-only servers. It uses [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp),
the fastest open-source CPU engine for this model family. You get an
OpenAI-compatible API with streaming, tool calling and reasoning output. See
[`docs/PLAN.md`](../docs/PLAN.md) for why this works and what speed to expect.

| Script | What it does |
|---|---|
| `check_host.sh` | Checks CPU features, RAM, memory channels, NUMA and disk, and prints a rough speed estimate |
| `build.sh` | Builds ik_llama.cpp at the pinned commit, tuned for this CPU |
| `download_model.sh` | Downloads the GGUF model from Hugging Face; resumes if interrupted |
| `serve.sh` | Starts the API server with the right threads and NUMA placement |
| `smoke_test.py` | Checks a running server and measures time to first token and tokens/s |
| `bench.sh` | Measures prompt and generation speed at growing context lengths |
| `install_service.sh` | Installs the systemd service, which restarts on failure and starts at boot |
| `nginx.conf.example` | Optional load balancer in front of several instances |
| `test/run_local_test.sh` | Tests all of the above in about a minute with a tiny fake model |

## Requirements

- **CPU:** x86-64 with AVX2.
  - AVX-512 with VNNI is strongly recommended: Intel Xeon Sapphire Rapids or
    newer (AMX), or AMD EPYC Zen 4/5.
  - Decode speed scales with memory bandwidth, so populate every memory
    channel.
- **RAM:** at least 192 GB for the default quant (162 GB model); 256 GB or
  more is comfortable.
  - On a 2-socket server, the fastest mode (`per-node`) keeps a full copy on
    each socket, so each socket needs about 180 GB.
  - With less RAM, experts are streamed from an NVMe SSD instead (32 GB of
    RAM minimum, 64 GB or more recommended). This is much slower; see
    [Test on a desktop PC](#test-on-a-desktop-pc).
- **Disk:** about 175 GB free for the model. Use NVMe: load time is disk
  bound.
- **Software:** Linux, git, cmake ≥ 3.14, gcc or clang with C++17, and
  python3. Also:
  - `numactl` on multi-socket servers.
  - The Hugging Face CLI: `pip install -U "huggingface_hub[cli,hf_xet]"`.

## Quick start

Run every command from the root of this repository, on the server itself.
`build.sh` compiles for the local CPU.

```bash
cp deploy/config.env.example deploy/config.env   # edit INSTALL_DIR and the rest if needed
sudo deploy/check_host.sh                          # sudo only to read DIMM info
deploy/build.sh                                    # 10-20 minutes
deploy/test/run_local_test.sh                      # optional: validates the build and scripts in ~1 minute
deploy/download_model.sh                           # ~160 GB
deploy/serve.sh                                    # foreground; wait for "server is listening"
```

In a second terminal:

```bash
python3 deploy/smoke_test.py --fixed-length                   # one request
python3 deploy/smoke_test.py --fixed-length --concurrency 4   # four users at once
```

Add `--api-key <key>` once you have set `API_KEY`.

Loading takes a few minutes: 160 GB is read from disk into RAM.

## Test on a desktop PC

A desktop can run the full model by streaming experts from the SSD. It is
good for trying things out, not for serving users.

**What you need:**
- Linux, or Windows with WSL2.
- 32 GB of RAM or more; 64 GB or more is much better.
- An NVMe SSD with about 170 GB free.

**What to expect:** roughly 2-4 tok/s once warm; the first request is slow
while the cache fills. Even with everything in RAM, a 2-channel desktop tops
out around 5 tok/s on this model (details in `docs/PLAN.md` section 12).
Long prompts are slow because every prompt batch reads most experts from the
SSD.

**Settings.** Put these in `deploy/config.env`:

```bash
INSTALL_DIR=$HOME/deepseek-cpu   # on the NVMe drive
QUANT=UD-Q4_K_XL                 # fewer bytes per token than UD-Q8_K_XL
PARALLEL=1
CTX_PER_SLOT=16384
CACHE_RAM_MIB=2048
EXPERT_STREAMING=auto            # turns on by itself when the model is larger than RAM
```

Then follow the Quick start. `check_host.sh` warns instead of failing when
the model is larger than RAM, and checks that the model directory is on
NVMe. `serve.sh --plan` shows `EXPERT_STREAMING=on`.

**Check streaming before the 160 GB download.** Generate a fake MoE model
somewhat larger than your RAM and serve it:

```bash
python3 deploy/test/make_tiny_gguf.py --size-gb 80 ~/moe-test.gguf   # e.g. 80 GB for a 64 GB PC
MODEL_FILE=~/moe-test.gguf deploy/serve.sh
```

In a second terminal:

```bash
python3 deploy/smoke_test.py --fixed-length   # run twice: cold, then warm
```

Delete `~/moe-test.gguf` afterwards.

**WSL2 notes:**
- Keep `INSTALL_DIR` on the Linux filesystem (for example `~/deepseek-cpu`),
  not under `/mnt/c`: Windows drives are mounted through a slow file-sharing
  layer.
- WSL2 gets only half of your RAM by default. Raise the limit with
  `memory=` in `%UserProfile%\.wslconfig`, then run `wsl --shutdown`.

**macOS:** these scripts are Linux-only. On a Mac, use mainline llama.cpp,
which has a Metal GPU backend.

## Run it as a service

```bash
sudo deploy/install_service.sh <service-user>
journalctl -fu 'deepseek-cpu@*'
```

On a multi-socket server, `NUMA_MODE=auto` resolves to one of two modes, and
the choice is frozen when you install the service:

- **`per-node`**: one server per socket, on ports 8080, 8081 and so on. It
  needs enough RAM for one copy per socket. Use `nginx.conf.example` to expose
  them as a single endpoint.
- **`interleave`**: one server spread over all sockets.

`per-node` gives the most total throughput. `interleave` needs half the RAM.
To choose for your hardware, run `bench.sh` in both modes.

## Use the API

The server speaks the OpenAI API, so any OpenAI client works:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="<API_KEY or anything>")
reply = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```

Output format:

- The model's reasoning is returned in `reasoning_content` (because of
  `--reasoning-format deepseek`).
- Tool calls use the standard `tools` / `tool_calls` fields (because of
  `--jinja`).
- The built-in web UI is served at `http://<host>:8080/`.

**Security:**
- The server binds to `127.0.0.1` by default.
- Before setting `HOST=0.0.0.0`, set `API_KEY` (or `API_KEY_FILE`) and
  firewall the port. The key is passed to the server through a pipe, so it
  does not show up in `ps`.
- There is no TLS; put nginx or another proxy with TLS in front of it for
  remote clients.

## Tuning

All settings live in `deploy/config.env`. Environment variables override it
for a single run, for example `PARALLEL=8 deploy/serve.sh`.

| Setting | Default | When to change it |
|---|---|---|
| `QUANT` | `UD-Q8_K_XL` | `UD-Q4_K_XL` reads fewer bytes per token, so decode is faster, at a small quality cost. Measure both. |
| `PARALLEL` | `4` | Concurrent requests per server. More gives more total throughput but a slower stream per user (table in `docs/PLAN.md` section 8). |
| `CTX_PER_SLOT` | `32768` | Longest conversation per request. The total context is `PARALLEL x CTX_PER_SLOT`. |
| `CACHE_RAM_MIB` | `32768` | RAM for reusing earlier prompts, which lowers time to first token for chats and agents. Raise it if RAM allows. |
| `EXPERT_STREAMING` | `auto` | `auto` streams experts from SSD only when the model does not fit in RAM. `off` always loads the whole model into RAM, which fails or swaps if it does not fit; `on` forces streaming. |
| `THREADS` | physical cores | Leave empty unless benchmarks say otherwise; never count hyperthreads. |
| `SPEC_TYPE` | off | Try `mtp:n_max=1` (speculative decoding with the model's built-in draft head). Keep it only if `smoke_test.py` shows higher tok/s. |
| `EXTRA_ARGS` | none | Any extra `llama-server` flag, such as `-rtr` (repack weights at load; may speed up prompt processing). Measure with `bench.sh` first. |

To compare an option before rolling it out, stop the server and run:

```bash
deploy/bench.sh
EXTRA_ARGS="-rtr" deploy/bench.sh
QUANT=UD-Q4_K_XL deploy/bench.sh
```

Results are saved in `$INSTALL_DIR/results/`.

## What to expect

Rough single-user generation speed per socket, from `docs/PLAN.md`:

| Server | Tokens/s |
|---|---|
| EPYC 9005 Turin, 12ch DDR5-6000 | 21-35 |
| Xeon 6 6900P, 12ch DDR5-6400 | 22-37 |
| Xeon 6 6900P, 12ch MRDIMM-8800 | 31-51 |

- Prompt processing is much slower than on a GPU: expect a few hundred
  tokens/s. A 20K-token prompt can take a minute to the first token.
- The prompt cache (`CACHE_RAM_MIB`) avoids reprocessing a prompt prefix the
  server has already seen, such as the same system prompt, tools or
  conversation history.

## Upgrading ik_llama.cpp

DeepSeek-V4 support in ik_llama.cpp is recent and changes often, so upgrade
deliberately:

1. Set `IK_LLAMA_COMMIT` in `deploy/config.env` to the new commit.
2. Rebuild and re-test:
   ```bash
   deploy/build.sh
   deploy/test/run_local_test.sh
   deploy/bench.sh
   ```
3. Run `smoke_test.py` against a test instance, then restart the service:
   ```bash
   sudo systemctl restart 'deepseek-cpu@*'
   ```

## Troubleshooting

- **Slower than expected:**
  - Run `sudo deploy/check_host.sh`. Empty memory channels are the most common
    cause.
  - `build.sh` warns if the AVX-512 kernels were not compiled in.
  - Make sure nothing else is using memory bandwidth, such as another server
    instance or a benchmark.
- **Out of memory at load:** lower `CACHE_RAM_MIB` or `PARALLEL x
  CTX_PER_SLOT`, switch `NUMA_MODE` from `per-node` to `interleave`, or use a
  smaller quant.
- **`skipping --mlock`:** the shell's memlock limit is too low. The systemd
  service sets it to unlimited. For manual runs, use `ulimit -l unlimited` as
  root or set `MLOCK=0`.
- **"server system prompts are unsupported for DeepSeek4":** this refers to
  llama-server's `--system-prompt-file` option, which ik_llama.cpp does not
  support for this model. Send system messages in each request instead;
  those work normally.
