/*
 * benchmarks/run_benchmark.c
 * Genesis L0-Sidecar benchmark: SRP-LSH memory/QPS/recall + cross-process
 * POSIX SHM handoff latency. No external dependencies.
 *
 * Licensed under the Apache License, Version 2.0.
 */
#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#define HAVE_TSC 1
#endif

#include "../src/shm_ring.h"
#include "../src/srp_lsh.h"

#define N_VECS     50000
#define DIM        512
#define N_CLUSTERS 1000
#define NOISE      1.0f
#define N_QUERY    200
#define K_TRUE     10
#ifndef K_CAND
#define K_CAND     128
#endif
#define QPS_ROUNDS 20
#define IPC_WARMUP 20000
#define IPC_ITERS  200000
#define SHM_NAME   "/genesis_l0_bench"

/* ── timing ───────────────────────────────────────────────────────────── */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#ifdef HAVE_TSC
static double tsc_per_ns;

static void calibrate_tsc(void) {
    double t0 = now_s();
    uint64_t c0 = __rdtsc();
    while (now_s() - t0 < 0.2) {}
    tsc_per_ns = (double)(__rdtsc() - c0) / ((now_s() - t0) * 1e9);
}

static inline uint64_t ticks(void) { _mm_lfence(); return __rdtsc(); }
static inline double ticks_to_ns(uint64_t t) { return t / tsc_per_ns; }
#define TIMER_NAME "rdtsc"
#else
static void calibrate_tsc(void) {}
static inline uint64_t ticks(void) { return (uint64_t)(now_s() * 1e9); }
static inline double ticks_to_ns(uint64_t t) { return (double)t; }
#define TIMER_NAME "clock_gettime"
#endif

/* ── synthetic clustered dataset ──────────────────────────────────────── */

static uint64_t rng_state = 42;

static uint64_t rand_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float randn(void) {
    uint64_t z = rand_u64();
    /* Irwin-Hall(4) approximation: fast, mean 0, variance 1 */
    float s = 0.0f;
    for (int i = 0; i < 4; i++)
        s += (float)((z >> (16 * i)) & 0xFFFF) / 65535.0f;
    return (s - 2.0f) * 1.7320508f;
}

static void normalize(float *x) {
    double n = 0.0;
    for (int j = 0; j < DIM; j++) n += (double)x[j] * x[j];
    float inv = (float)(1.0 / (n > 0 ? __builtin_sqrt(n) : 1.0));
    for (int j = 0; j < DIM; j++) x[j] *= inv;
}

/* Each vector = cluster center + NOISE * gaussian, unit-normalized. */
static void gen_point(const float *centers, float *x) {
    const float *c = centers + (size_t)(rand_u64() % N_CLUSTERS) * DIM;
    for (int j = 0; j < DIM; j++) x[j] = c[j] + NOISE * randn();
    normalize(x);
}

/* ── percentile helper ────────────────────────────────────────────────── */

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static void pin_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

/* ── cross-process SHM ping-pong ──────────────────────────────────────── */

static int run_ipc(double *p50, double *p99, double *p999) {
    shm_mailbox *mb = shm_ring_open(SHM_NAME, sizeof(shm_mailbox));
    if (!mb) { perror("shm_ring_open"); return -1; }
    memset((void *)mb, 0, sizeof(*mb));

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); shm_ring_close(SHM_NAME, mb, sizeof(*mb)); return -1; }
    if (pid == 0) {                         /* responder process */
        pin_cpu(1);
        uint64_t seen = 0;
        for (;;) {
            uint64_t r;
            while ((r = __atomic_load_n(&mb->req, __ATOMIC_ACQUIRE)) == seen) {}
            if (r == UINT64_MAX) _exit(0);
            mb->payload[1] = mb->payload[0] ^ 0xA5A5A5A5u;  /* touch payload */
            __atomic_store_n(&mb->ack, r, __ATOMIC_RELEASE);
            seen = r;
        }
    }

    pin_cpu(0);
    uint64_t *lat = malloc(IPC_ITERS * sizeof(uint64_t));
    if (!lat) { kill(pid, SIGKILL); waitpid(pid, NULL, 0); return -1; }

    for (uint64_t i = 1; i <= IPC_WARMUP + IPC_ITERS; i++) {
        uint64_t t0 = ticks();
        mb->payload[0] = (uint32_t)i;
        __atomic_store_n(&mb->req, i, __ATOMIC_RELEASE);
        while (__atomic_load_n(&mb->ack, __ATOMIC_ACQUIRE) != i) {}
        uint64_t t1 = ticks();
        if (i > IPC_WARMUP) lat[i - IPC_WARMUP - 1] = t1 - t0;
    }
    __atomic_store_n(&mb->req, UINT64_MAX, __ATOMIC_RELEASE);
    waitpid(pid, NULL, 0);
    shm_ring_close(SHM_NAME, mb, sizeof(*mb));

    qsort(lat, IPC_ITERS, sizeof(uint64_t), cmp_u64);
    /* one-way handoff = round trip / 2 */
    *p50  = ticks_to_ns(lat[IPC_ITERS / 2]) / 2.0;
    *p99  = ticks_to_ns(lat[(size_t)(IPC_ITERS * 0.99)]) / 2.0;
    *p999 = ticks_to_ns(lat[(size_t)(IPC_ITERS * 0.999)]) / 2.0;
    free(lat);
    return 0;
}

static void cpu_model(char *buf, size_t n) {
    snprintf(buf, n, "unknown");
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *p = strchr(line, ':');
            if (p) { snprintf(buf, n, "%s", p + 2); buf[strcspn(buf, "\n")] = 0; }
            break;
        }
    }
    fclose(f);
}

int main(void) {
    const char *rule = "============================================================";
    const char *thin = "------------------------------------------------------------";
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    char cpu[128];
    cpu_model(cpu, sizeof cpu);
    calibrate_tsc();

    float *centers = malloc((size_t)N_CLUSTERS * DIM * sizeof(float));
    float *db      = malloc((size_t)N_VECS * DIM * sizeof(float));
    float *qs      = malloc((size_t)N_QUERY * DIM * sizeof(float));
    float *planes  = malloc((size_t)SRP_BITS * DIM * sizeof(float));
    srp_code *codes    = malloc(N_VECS * sizeof(srp_code));
    uint32_t *all_idx  = malloc(N_VECS * sizeof(uint32_t));
    uint32_t *truth    = malloc((size_t)N_QUERY * K_TRUE * sizeof(uint32_t));
    uint32_t *cand     = malloc(K_CAND * sizeof(uint32_t));
    uint64_t *qlat     = malloc((size_t)QPS_ROUNDS * N_QUERY * sizeof(uint64_t));
    uint32_t top[K_TRUE];
    if (!centers || !db || !qs || !planes || !codes || !all_idx || !truth || !cand || !qlat) {
        fputs("out of memory\n", stderr);
        return 1;
    }

    for (size_t i = 0; i < (size_t)N_CLUSTERS * DIM; i++) centers[i] = randn();
    for (size_t i = 0; i < N_VECS; i++)  gen_point(centers, db + i * DIM);
    for (size_t i = 0; i < N_QUERY; i++) gen_point(centers, qs + i * DIM);

    srp_make_planes(planes, DIM, 7);
    for (size_t i = 0; i < N_VECS; i++) {
        codes[i] = srp_encode(planes, DIM, db + i * DIM);
        all_idx[i] = (uint32_t)i;
    }

    /* exact ground truth + baseline QPS: re-rank of the whole database */
    double t0 = now_s();
    for (int q = 0; q < N_QUERY; q++)
        srp_rerank_topk(db, DIM, qs + (size_t)q * DIM, all_idx, N_VECS, K_TRUE, truth + q * K_TRUE);
    double exact_qps = N_QUERY / (now_s() - t0);

    /* cascade: stage 1 = Hamming top-K_CAND, stage 2 = exact top-10.
     * Every query is timed end to end (encode + scan + re-rank). */
    size_t hit_cand = 0, hit_final = 0;
    t0 = now_s();
    for (int r = 0; r < QPS_ROUNDS; r++)
        for (int q = 0; q < N_QUERY; q++) {
            const float *qv = qs + (size_t)q * DIM;
            uint64_t c0 = ticks();
            srp_code qc = srp_encode(planes, DIM, qv);
            size_t m = srp_scan_topk(&qc, codes, N_VECS, K_CAND, cand);
            size_t f = srp_rerank_topk(db, DIM, qv, cand, m, K_TRUE, top);
            qlat[(size_t)r * N_QUERY + q] = ticks() - c0;
            if (r > 0) continue;
            for (int t = 0; t < K_TRUE; t++) {
                uint32_t want = truth[q * K_TRUE + t];
                for (size_t c = 0; c < m; c++) if (cand[c] == want) { hit_cand++;  break; }
                for (size_t c = 0; c < f; c++) if (top[c] == want)  { hit_final++; break; }
            }
        }
    double cascade_qps = (double)QPS_ROUNDS * N_QUERY / (now_s() - t0);
    qsort(qlat, (size_t)QPS_ROUNDS * N_QUERY, sizeof(uint64_t), cmp_u64);
    double q_p50 = ticks_to_ns(qlat[QPS_ROUNDS * N_QUERY / 2]) / 1000.0;
    double q_p99 = ticks_to_ns(qlat[(size_t)(QPS_ROUNDS * N_QUERY * 0.99)]) / 1000.0;

    double p50 = 0, p99 = 0, p999 = 0;
    int ipc_ok = ncpu >= 2 ? run_ipc(&p50, &p99, &p999) : 1;

    double raw_mb  = (double)N_VECS * DIM * sizeof(float) / 1e6;
    double code_mb = (double)N_VECS * sizeof(srp_code) / 1e6;

    puts(rule);
    puts("GENESIS L0-SIDECAR BENCHMARK (C99 POSIX SHM)");
    puts(rule);
    printf("Host:                   %s (%ld CPUs)\n", cpu, ncpu);
    printf("Dataset:                %d vectors (%d-dim, %d clusters, synthetic)\n",
           N_VECS, DIM, N_CLUSTERS);
    printf("Quantization:           SRP-LSH, %d random hyperplanes -> %d-bit code\n",
           SRP_BITS, SRP_BITS);
    printf("Index Footprint:        %.2f MB (vs %.1f MB raw float32)\n", code_mb, raw_mb);
    printf("Compression Ratio:      %.0fx\n", raw_mb / code_mb);
    puts(thin);
    if (ipc_ok == 0) {
        printf("SHM Handoff, cross-process, one-way (%s, %d iters):\n", TIMER_NAME, IPC_ITERS);
        printf("  p50:                  %.0f ns\n", p50);
        printf("  p99:                  %.0f ns\n", p99);
        printf("  p99.9:                %.0f ns\n", p999);
        puts("Syscalls in hot loop:   0 (mmap'd mailbox, atomic load/store only)");
    } else {
        puts("SHM Handoff:            skipped (needs >= 2 CPUs)");
    }
    puts(thin);
    printf("Two-stage cascade: Hamming top-%d -> exact re-rank -> top-%d\n", K_CAND, K_TRUE);
    printf("  Throughput:           %.0f QPS (single core)\n", cascade_qps);
    printf("  Query latency p50:    %.1f us\n", q_p50);
    printf("  Query latency p99:    %.1f us\n", q_p99);
    printf("  Recall 10@10:         %.1f %%  (final top-10 vs exact top-10)\n",
           100.0 * hit_final / (N_QUERY * K_TRUE));
    printf("  Candidate recall:     %.1f %%  (exact top-10 within Hamming top-%d)\n",
           100.0 * hit_cand / (N_QUERY * K_TRUE), K_CAND);
    printf("Exact f32 brute force:  %.0f QPS (single core, recall 100 %%)\n", exact_qps);
    puts(rule);
    free(centers); free(db); free(qs); free(planes);
    free(codes); free(all_idx); free(truth); free(cand); free(qlat);
    return 0;
}
