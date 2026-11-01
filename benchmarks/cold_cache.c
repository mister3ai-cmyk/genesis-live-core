/*
 * benchmarks/cold_cache.c
 *
 * Answers one question: what does a query cost when the code index does not
 * fit in page cache and the scan has to fault pages in from disk?
 *
 * The published numbers (~4-5 us for the mmap read, storage ~2% of the 0.3 ms
 * budget) were measured on a 33k-vector corpus that sits in page cache whole.
 * That is honest for that corpus and says nothing about a larger one. This
 * harness builds a file-backed index of arbitrary size, drops it out of cache,
 * and measures the first scan against the steady state.
 *
 * Usage: cold_cache <index-file> <queries> [pin]
 *   `pin` calls shm_ring_pin() on the mapping before the first query, so the
 *   two runs show what prefaulting buys and what it costs up front.
 *
 * Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
 */
#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../src/shm_ring.h"
#include "../src/srp_lsh.h"

#define TOPK 10

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static long major_faults(void) {
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_majflt;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <index-file> <queries> [pin]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int queries = atoi(argv[2]);
    int want_pin = (argc > 3 && strcmp(argv[3], "pin") == 0);
    if (queries < 2) {
        fprintf(stderr, "need at least 2 queries\n");
        return 2;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }

    size_t bytes = (size_t)st.st_size;
    size_t n = bytes / sizeof(srp_code);
    if (n == 0) { fprintf(stderr, "index file too small\n"); close(fd); return 1; }

    const srp_code *db = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (db == MAP_FAILED) { perror("mmap"); return 1; }

    printf("corpus      : %zu codes, %.1f MiB on disk, %d bits per code\n",
           n, bytes / 1048576.0, SRP_BITS);

    double pin_ms = 0.0;
    if (want_pin) {
        int err = 0;
        double t0 = now_ms();
        shm_pin_result r = shm_ring_pin((void *)db, bytes, &err);
        pin_ms = now_ms() - t0;
        printf("pin         : %s", shm_pin_describe(r));
        if (err)
            printf("  [mlock refused: %s]", strerror(err));
        printf("\n");
        printf("pin cost    : %.1f ms up front\n", pin_ms);
    } else {
        printf("pin         : not attempted\n");
    }

    double *lat = malloc((size_t)queries * sizeof(double));
    if (!lat) { fprintf(stderr, "oom\n"); return 1; }
    uint32_t out[TOPK];
    long f0 = major_faults();

    for (int i = 0; i < queries; i++) {
        srp_code q;
        for (int w = 0; w < SRP_WORDS; w++)
            q.w[w] = rng_next();
        double t0 = now_ms();
        size_t got = srp_scan_topk(&q, db, n, TOPK, out);
        lat[i] = now_ms() - t0;
        if (got == 0) { fprintf(stderr, "scan returned nothing\n"); return 1; }
    }
    long faults = major_faults() - f0;

    /* The first query carries the fault cost; the rest show steady state. */
    double first = lat[0];
    double *rest = lat + 1;
    int nrest = queries - 1;
    qsort(rest, (size_t)nrest, sizeof(double), cmp_double);
    double p50 = rest[nrest / 2];
    double p99 = rest[(int)((nrest - 1) * 0.99)];

    printf("first query : %.2f ms   <- pays for whatever was not resident\n", first);
    printf("steady p50  : %.2f ms\n", p50);
    printf("steady p99  : %.2f ms\n", p99);
    /* ru_majflt undercounts: readahead pulls in many pages per accounted
     * fault, so a per-fault average is meaningless. It answers one thing,
     * whether the scan touched disk at all. */
    printf("major faults: %ld accounted (readahead coalesces)\n", faults);
    if (first > p50)
        printf("read rate   : %.0f MiB/s over the %.0f ms of extra time\n",
               (bytes / 1048576.0) / ((first - p50) / 1e3), first - p50);
    printf("penalty     : first query is %.1fx the steady median\n", first / p50);
    if (want_pin)
        printf("total cold  : %.2f ms including the %.1f ms prefault\n",
               pin_ms + first, pin_ms);

    free(lat);
    munmap((void *)db, bytes);
    return 0;
}
