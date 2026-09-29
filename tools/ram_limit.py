#!/usr/bin/env python3
"""Hold RAM so only a chosen amount stays available: emulates a smaller-RAM PC.

Allocates and touches anonymous memory until MemAvailable drops to --leave GiB,
then sleeps until killed. Without swap the kernel cannot evict this memory, so
the page cache (and with it the cache of SSD-streamed experts) is squeezed
into what is left. Standard library only.

    python3 tools/ram_limit.py --leave 3 &     # leave 3 GiB available
    ...run the experiment...
    kill %1
"""

import argparse
import signal
import sys
import time

CHUNK = 256 << 20  # allocate in 256 MiB steps


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    raise RuntimeError("MemAvailable not found")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leave", type=float, required=True, help="GiB of MemAvailable to leave free")
    args = ap.parse_args()

    held = []
    page = b"\x01" * CHUNK
    while mem_available_gib() - CHUNK / (1 << 30) > args.leave:
        buf = bytearray(CHUNK)
        buf[:] = page  # touch every page so it is really allocated
        held.append(buf)
    print(f"holding {len(held) * CHUNK / (1 << 30):.1f} GiB; MemAvailable now {mem_available_gib():.1f} GiB",
          flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
