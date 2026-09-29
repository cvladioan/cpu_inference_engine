// Sustained DRAM read bandwidth: the ceiling for CPU decode speed.
//
// Streams a buffer much larger than the caches with 1..N threads and reports
// GB/s, like STREAM's read kernel. Decode on CPU reads every active weight
// once per token, so tok/s <= bandwidth / bytes-per-token.
//
//   gcc -O3 -march=native -fopenmp tools/membw.c -o membw && ./membw [GiB] [max_threads]
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Sum 64-bit words; the compiler vectorizes this into wide loads.
static uint64_t read_pass(const uint64_t *buf, size_t n, int threads) {
    uint64_t total = 0;
#pragma omp parallel for num_threads(threads) reduction(+ : total) schedule(static)
    for (size_t i = 0; i < n; i++) total += buf[i];
    return total;
}

int main(int argc, char **argv) {
    double gib = argc > 1 ? atof(argv[1]) : 2.0;
    int max_threads = argc > 2 ? atoi(argv[2]) : omp_get_num_procs();
    size_t bytes = (size_t)(gib * (1u << 30));
    size_t n = bytes / sizeof(uint64_t);
    uint64_t *buf = aligned_alloc(4096, n * sizeof(uint64_t));
    if (!buf) { fprintf(stderr, "allocation of %.1f GiB failed\n", gib); return 1; }

    // First touch in parallel so pages spread like they would under a threaded engine.
#pragma omp parallel for num_threads(max_threads) schedule(static)
    for (size_t i = 0; i < n; i++) buf[i] = i;

    printf("buffer %.1f GiB, %d max threads\n", gib, max_threads);
    printf("threads  read GB/s\n");
    uint64_t sink = 0;
    for (int t = 1; t <= max_threads; t++) {
        double best = 0;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = now();
            sink += read_pass(buf, n, t);
            double gbs = bytes / (now() - t0) / 1e9;
            if (gbs > best) best = gbs;
        }
        printf("%7d  %9.1f\n", t, best);
    }
    free(buf);
    return sink == 42;  // keep the reads from being optimized away
}
