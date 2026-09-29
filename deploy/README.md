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
| `tune.py` | Finds the fastest configuration on this machine and checks it against a target such as 20 tok/s |
| `quickstart.sh` | Runs everything, from a fresh server to a tuned setup, in one command |
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

## Reaching 20 tok/s

Generation speed on a CPU is set by memory bandwidth: every token reads about
7 GB of weights (4-bit quant). So 20 tok/s for a single user needs about
400 GB/s of peak memory bandwidth per socket with today's engine efficiency
(`python3 tools/roofline.py --target-tps 20`).

That means a server CPU with 12 memory channels, all populated:

| Machine | Memory | Expected single-user tok/s (4-bit) |
|---|---|---|
| AMD EPYC 9004 Genoa (e.g. Hetzner AX162-R, AWS r7a/m7a) | 12ch DDR5-4800, 460 GB/s | ~23-38 |
| AMD EPYC 9005 Turin | 12ch DDR5-6000, 576 GB/s | ~29-48 |
| Intel Xeon 6 6900P | 12ch DDR5-6400 / MRDIMM-8800 | ~31-70 |
| Desktop PC (for comparison) | 2ch DDR5, ~90 GB/s | ~1-2 when streaming from SSD; ~5 if it all fit in RAM |

The same server roughly doubles its total throughput when several users are
served at once (`docs/PLAN.md` section 8).

**Renting one:**
- **Hetzner AX162-R** (EPYC 9454P, about €200-240/month). Order a memory
  option that populates all 12 channels, for example 12 x 32 GB = 384 GB. An
  8-DIMM configuration loses a third of the bandwidth.
- **AWS `r7a.48xlarge` / `m7a.48xlarge`** (2 x Genoa, billed per hour). Run
  with `NUMA_MODE=per-node`: one instance per socket, each above 20 tok/s.

Either way, confirm the DIMM count with `sudo deploy/check_host.sh`.

**One command** does everything on the server: host check, build, model
download, tuning against the target, and applying the best settings.

```bash
cp deploy/config.env.example deploy/config.env
echo 'QUANT=UD-Q4_K_XL' >> deploy/config.env     # 7.0 GB/token instead of 9.6: ~35% faster
deploy/quickstart.sh --target 20                  # add --service <user> to install the systemd service
```

`tune.py` loads the model once per candidate and measures real generation
speed on code, prose and JSON extraction. It tries:
- thread counts
- weight repacking at load (`-rtr`)
- n-gram self-speculation
- DSpark speculative decoding, if a draft model is configured

It prints a ranked table, exits with an error if the best result is below
`--target`, and with `--apply` writes the winner to `deploy/config.env`.

**Two optional speed-ups, both measured before use:**
- **DSpark drafts.** DeepSeek-V4-Flash-0731 ships a DSpark draft instead of
  MTP. Reported gains are 1.5-1.9x on GPU, but only +10-15% when the experts
  run on CPU, and it depends on the text (code benefits most). To let the
  tuner try it, set these in `deploy/config.env` and run `download_model.sh`
  again (about 11 GB):
  ```bash
  DRAFT_REPO=singulared/DeepSeek-V4-Flash-0731-DSpark-GGUF
  DRAFT_INCLUDE=*Q8_0*.gguf
  ```
- **Fewer experts per token** (`python3 deploy/tune.py --allow-expert-reduction`).
  `-ser 5,1` or `-ser 4,1` uses 5 or 4 of the 6 routed experts. That reads
  up to a third fewer expert bytes, but it changes the model's answers, so
  compare outputs before keeping it.

## Test on a desktop PC

A desktop can run the full model by streaming experts from the SSD. It is
good for trying things out, not for serving users.

**What you need:**
- Linux, or Windows 10/11 with WSL2.
- 32 GB of RAM or more; 64 GB or more is much better.
- An NVMe SSD with about 170 GB free, plus about 15 GB for a small test
  model.

**What to expect on a 32 GB desktop:**
- A small model that fits in RAM (step 1 below) runs at 10-20 tok/s.
- DeepSeek-V4-Flash streams from the SSD at about 1-2 tok/s once warm. The
  first request after a start is slower, and long prompts take minutes.
- Even with everything in RAM, a 2-channel desktop would top out around
  5 tok/s on this model (`docs/PLAN.md` section 12).

### Windows: set up WSL2

1. **Install Ubuntu.** In PowerShell (as administrator), run the command
   below. Restart if asked, then open "Ubuntu" from the Start menu and create
   a user.
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
2. **Give WSL2 more RAM and keep its cache.** Create
   `C:\Users\<you>\.wslconfig` with:
   ```ini
   [wsl2]
   memory=26GB

   [experimental]
   autoMemoryReclaim=disabled
   ```
   Why these settings:
   - By default WSL2 gets only half of your RAM.
   - By default it also empties its file cache after a few idle minutes,
     which throws away the cached experts.

   Run `wsl --shutdown` in PowerShell, then reopen Ubuntu. `free -g` should
   show about 25 GB. 26GB leaves about 6 GB for Windows on a 32 GB PC, so
   close large apps while testing.
3. **Install the tools** in Ubuntu:
   ```bash
   sudo apt update && sudo apt install -y build-essential cmake git python3-venv
   python3 -m venv ~/hf && ~/hf/bin/pip install -U "huggingface_hub[cli,hf_xet]"
   echo 'export PATH="$HOME/hf/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
   ```
4. **Clone this repository** in your Linux home directory (for example
   `~/cpu_inference_engine`), not under `/mnt/c`. Windows drives are mounted
   through a slow file-sharing layer. Keep `INSTALL_DIR` on the Linux side
   too.

Notes:
- The Linux disk is a file on your C: drive that grows as models are
  downloaded and does not shrink by itself. After deleting models, reclaim the
  space with `wsl --manage Ubuntu-24.04 --set-sparse true`, or by compacting
  the disk.
- `check_host.sh` reads your `.wslconfig` and warns about missing settings.

### Step 1: a small model that fits in RAM

Start with gpt-oss-20b (12 GB). It checks the build, server and API at a
usable speed. Put this in `deploy/config.env`:

```bash
INSTALL_DIR=$HOME/deepseek-cpu
HF_REPO=ggml-org/gpt-oss-20b-GGUF
QUANT=mxfp4
MODEL_DIR=$HOME/deepseek-cpu/models/gpt-oss-20b
MODEL_ALIAS=gpt-oss-20b
THREADS=6            # P-cores only on Intel 12th-14th gen (6 on a Core i5-13xxx)
PARALLEL=1
CTX_PER_SLOT=16384
CACHE_RAM_MIB=2048
```

Then:

```bash
deploy/check_host.sh
deploy/build.sh
deploy/download_model.sh
deploy/serve.sh
```

In a second terminal:

```bash
python3 deploy/smoke_test.py --fixed-length
```

On hybrid Intel CPUs, the efficiency cores slow everyone down when the
threads have to wait for each other. So compare `THREADS=6` against 8 and 10
with `deploy/bench.sh` (stop the server first) and keep the fastest.

### Step 2: DeepSeek-V4-Flash with SSD streaming

**Rebuild first.** Streaming uses this repository's explicit expert cache
(`engine/patches/`, applied by `deploy/build.sh`). It measured 4-6x faster
than leaving experts to the OS page cache. `serve.sh --plan` shows
`EXPERT_CACHE_MIB` > 0 when it is active.

**Optional: check streaming first, without the big download.**
1. Generate a fake MoE model larger than WSL's RAM (the file is about 40 GB):
   ```bash
   python3 deploy/test/make_tiny_gguf.py --size-gb 40 ~/moe-test.gguf
   ```
2. Serve it:
   ```bash
   MODEL_FILE=~/moe-test.gguf deploy/serve.sh
   ```
   `serve.sh` should log `streaming=on`.
3. In a second terminal, run the smoke test twice (cold, then warm):
   ```bash
   python3 deploy/smoke_test.py --fixed-length
   ```
4. Delete `~/moe-test.gguf` afterwards.

**Measure your SSD's read speed from inside WSL.** It bounds the streaming
speed:

```bash
dd if=/dev/zero of=~/ddtest bs=4M count=1024 oflag=direct status=progress   # writes 4 GB
dd if=~/ddtest of=/dev/null bs=4M iflag=direct status=progress              # read speed
rm ~/ddtest
```

**Switch the model lines in `deploy/config.env`**, then run
`download_model.sh` (about 155 GB) and `serve.sh`:

```bash
HF_REPO=unsloth/DeepSeek-V4-Flash-0731-GGUF
QUANT=UD-Q4_K_XL     # ~155 GB. A 2-bit quant (~91-97 GB) reads fewer bytes per token:
                     # faster, but lower quality. Check the repo's file list for names.
MODEL_DIR=$HOME/deepseek-cpu/models/DeepSeek-V4-Flash-0731-GGUF
MODEL_ALIAS=deepseek-v4-flash
EXTRA_ARGS="--reasoning-budget 0"   # skip the thinking phase; at 1-2 tok/s it takes minutes
```

`EXPERT_STREAMING=auto` turns streaming on by itself because the model is
larger than RAM. `serve.sh --plan` shows it, and `check_host.sh` warns instead
of failing.

### Linux desktops and macOS

- **Linux desktop:** the same steps without the WSL2 setup.
- **macOS:** these scripts are Linux-only. On a Mac, use mainline llama.cpp,
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
| `EXPERT_CACHE_MIB` | `auto` | When streaming, the engine-managed expert cache (docs/EXPERT_CACHE.md), 4-6x faster than the OS page cache. `auto` sizes it from free RAM; `0` switches back to the page cache. The OS cannot drop its memory like page cache (and with swap on it may swap it out, so turn swap off when streaming), so lower `EXPERT_CACHE_HEADROOM_MIB` (default 2048) only if you know the machine's memory well. |
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

Rough single-user generation speed per socket (`python3 tools/roofline.py`).
The lower number is today's engine efficiency, the upper one a well-tuned
setup:

| Server | UD-Q8_K_XL | UD-Q4_K_XL |
|---|---|---|
| EPYC 9004 Genoa, 12ch DDR5-4800 | 17-28 | 23-38 |
| EPYC 9005 Turin, 12ch DDR5-6000 | 21-35 | 29-48 |
| Xeon 6 6900P, 12ch DDR5-6400 | 22-37 | 31-51 |
| Xeon 6 6900P, 12ch MRDIMM-8800 | 31-51 | 42-70 |

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
