# Home tests: DeepSeek-V4-Flash on a 32 GB Windows PC (WSL2)

The first measurement of the expert cache on the real model, on the PC this
project is aimed at: Windows 11, 32 GB RAM, Core i5 13th gen, NVMe SSD with
about 500 GB.

**What the tests answer:**
- Are outputs identical with the expert cache on the real model?
- What hit rate does real routing give? The estimate for uniform routing is
  about 12% at an 18 GB budget (`docs/EXPERT_CACHE.md` section 5).
- How fast is decoding with the cache compared to the OS page cache? The
  estimate is about 1.3 against 0.3-0.5 tok/s.
- How does the hit rate grow with the budget? That tells how much RAM buys
  how much speed.

Use WSL2, not native Windows: the expert cache is Linux-only. A native Windows
build runs, but without the cache.

Total time is about 2 hours plus the download (155 GB: about 25 minutes at
1 Gbit/s, 3.5 hours at 100 Mbit/s). Start the download early (step 3) and
build and test while it runs.

## 1. Windows (once)

In PowerShell:

```powershell
wsl --version                            # needs WSL 2.x; if missing: wsl --install -d Ubuntu-24.04
Get-PSDrive C | Select-Object Used,Free  # need about 200 GB free: the WSL disk lives on C:
notepad $env:USERPROFILE\.wslconfig
```

Put this in `.wslconfig`, save, then run `wsl --shutdown`:

```ini
[wsl2]
memory=26GB
swap=0

[experimental]
autoMemoryReclaim=disabled
```

- `memory=26GB` gives WSL2 most of the RAM (the default is half) and leaves
  about 6 GB for Windows.
- `swap=0`: the expert cache must stay in RAM. With swap on, WSL2 could move
  it to its swap file, which is slower than reading the experts again.
- `autoMemoryReclaim=disabled` stops WSL2 from emptying its memory after a
  few idle minutes.

**Before measuring:**
- Plug the PC in and choose the "Best performance" power mode.
- Close the browser and other large apps. Every GB they use is a GB less of
  expert cache.
- Pause OneDrive sync and avoid a Defender full scan during the runs: both
  compete for the SSD.

## 2. WSL (once)

Open Ubuntu from the Start menu:

```bash
sudo apt update && sudo apt install -y build-essential cmake git python3-venv fio
python3 -m venv ~/hf && ~/hf/bin/pip install -U "huggingface_hub[cli,hf_xet]"
echo 'export PATH="$HOME/hf/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

git clone https://github.com/cvladioan/cpu_inference_engine.git ~/cpu_inference_engine
cd ~/cpu_inference_engine && git checkout claude/modest-goodall-ryj2y7
```

Keep everything under your Linux home (`~`), never under `/mnt/c`: Windows
drives are reached through a slow file-sharing layer.

Create `deploy/config.env`:

```bash
INSTALL_DIR=$HOME/deepseek-cpu
HF_REPO=unsloth/DeepSeek-V4-Flash-0731-GGUF
QUANT=UD-Q4_K_XL
MODEL_DIR=$HOME/deepseek-cpu/models/DeepSeek-V4-Flash-0731-GGUF
MODEL_ALIAS=deepseek-v4-flash
THREADS=6            # performance cores only; step 6 checks 8 and 10
PARALLEL=1
CTX_PER_SLOT=8192
CACHE_RAM_MIB=1024
EXTRA_ARGS="--reasoning-budget 0"
```

## 3. Start the download

In its own terminal (it resumes if interrupted; just run it again):

```bash
cd ~/cpu_inference_engine && deploy/download_model.sh
```

## 4. Baselines and build (while it downloads)

```bash
cd ~/cpu_inference_engine
deploy/check_host.sh                  # should show the .wslconfig settings with no warnings
gcc -O3 -march=native -fopenmp tools/membw.c -o ~/membw && ~/membw 4 12
fio --name=r --filename=$HOME/fio.test --size=8G --rw=randread --bs=2M --direct=1 \
    --ioengine=libaio --iodepth=16 --numjobs=4 --runtime=20 --time_based --group_reporting | grep READ:
rm ~/fio.test
deploy/build.sh
deploy/test/run_local_test.sh         # must end with PASS, including "expert cache: outputs must match"
```

- **`membw`:** a 13th-gen i5 with dual-channel DDR5 should reach 60-80 GB/s,
  or about half that with DDR4.
- **`fio`:** the SSD's speed for the cache's kind of reads. A good NVMe drive
  gives 2-5 GB/s inside WSL2. It sets the limit: tok/s is at most about
  SSD speed divided by the bytes missed per token.

## 5. The main test: page cache against expert cache

Once the download is complete:

```bash
python3 tools/cache_ab.py --modes page,cache --tokens 32
```

About 30-40 minutes. For each mode, the script:
- empties the page cache (it asks for your sudo password when needed);
- starts the server and warms it up;
- generates 32 greedy tokens for each of 4 prompts;
- measures decode speed, disk reads per token and the cache's hit rate;
- checks that both modes produce the same text and the same top-5 token
  probabilities.

It prints a table and saves it to `~/deepseek-cpu/results/cache-ab-*/summary.md`.

If the engine had to lower the cache budget to fit in memory, the table says
so ("lowered from ..."). That is the memory guard doing its job; the run is
still valid.

## 6. Follow-up tests

**How the hit rate grows with RAM** (about 30 minutes). The auto budget
is about 16-18 GB:

```bash
python3 tools/cache_ab.py --modes cache:6000,cache:12000,cache --tokens 32
```

**Threads** (about 20 minutes). On hybrid Intel CPUs, efficiency cores can
slow everyone down; with the SSD as the limit they may not matter:

```bash
python3 tools/cache_ab.py --modes cache --threads 8 --tokens 32
python3 tools/cache_ab.py --modes cache --threads 10 --tokens 32
```

**A real conversation.** This is what using it feels like:

```bash
deploy/serve.sh                                      # terminal 1
python3 deploy/smoke_test.py --max-tokens 200        # terminal 2
```

**Routing traces for cache-aware routing (R2).** Record traces if the engine
has a trace option by then (`docs/EXPERT_CACHE.md`). Pass it through with
`--env NAME=value`.

## 7. Save the results

```bash
mkdir -p docs/results
for d in ~/deepseek-cpu/results/cache-ab-*; do cp "$d/summary.md" "docs/results/$(basename "$d").md"; done
git add docs/results && git commit -m "Home PC results: expert cache on DeepSeek-V4-Flash" && git push
```

Also note, for the write-up:
- the `membw` and `fio` numbers;
- the exact CPU (`lscpu | grep 'Model name'`) and RAM type/speed;
- whether the PC stayed responsive during the runs.

## 8. If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `cache` row says "expert cache not active" | The build lacks the patches. Run `deploy/build.sh` again; `llama-server --help` must list `--expert-cache`. |
| WSL closes, or "the Windows Subsystem for Linux instance has terminated" | Out of memory. Close apps, or pass a smaller budget (`--modes page,cache:12000`). |
| Disk reads well below the `fio` result | The model is under `/mnt/c`, or Windows is using the SSD (Defender, OneDrive, Windows Update). |
| Page mode takes hours | The expected case on a slow SSD. Use `--tokens 16`. |
| `page cache not emptied` in the table | sudo was not available, e.g. when run by Claude Code. Allow just this command without a password: `echo "$USER ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" \| sudo tee /etc/sudoers.d/drop-caches` |

## 9. Continuing with Claude Code in WSL

```bash
curl -fsSL https://claude.ai/install.sh | bash && exec bash -l
cd ~/cpu_inference_engine && claude
```

Prompt:

> Read docs/HANDOFF.md, docs/EXPERT_CACHE.md and docs/HOME_TESTS.md. This is
> the 32 GB Windows PC with WSL2. Run the home tests in docs/HOME_TESTS.md
> from step 4 on, starting whatever does not need the model while it
> downloads. Then write the results into docs/EXPERT_CACHE.md (real-model
> section) and commit and push to branch claude/modest-goodall-ryj2y7.
