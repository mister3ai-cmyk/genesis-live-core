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
#define K_CAND     100
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

static float randn(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
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
    const float *c = centers + (size_t)(rng_state % N_CLUSTERS) * DIM;
    for (int j = 0; j < DIM; j++) x[j] = c[j] + NOISE * randn();
    normalize(x);
}

/* ── exact baseline: float32 inner-product scan ───────────────────────── */

static float dot(const float *a, const float *b) {
    float acc[16] = {0};
    for (int j = 0; j < DIM; j += 16)
        for (int l = 0; l < 16; l++) acc[l] += a[j + l] * b[j + l];
    float s = 0.0f;
    for (int l = 0; l < 16; l++) s += acc[l];
    return s;
}

static void exact_topk(const float *db, const float *q, uint32_t *out) {
    float best[K_TRUE];
    for (int i = 0; i < K_TRUE; i++) { best[i] = -2.0f; out[i] = 0; }
    for (uint32_t i = 0; i < N_VECS; i++) {
        float s = dot(db + (size_t)i * DIM, q);
        if (s <= best[K_TRUE - 1]) continue;
        int p = K_TRUE - 1;
        while (p > 0 && best[p - 1] < s) { best[p] = best[p - 1]; out[p] = out[p - 1]; p--; }
        best[p] = s; out[p] = i;
    }
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
    uint32_t *codes  = malloc(N_VECS * sizeof(uint32_t));
    uint32_t *qcodes = malloc(N_QUERY * sizeof(uint32_t));
    uint32_t *truth  = malloc((size_t)N_QUERY * K_TRUE * sizeof(uint32_t));
    uint32_t cand[K_CAND];
    if (!centers || !db || !qs || !planes || !codes || !qcodes || !truth) {
        fputs("out of memory\n", stderr);
        return 1;
    }

    for (size_t i = 0; i < (size_t)N_CLUSTERS * DIM; i++) centers[i] = randn();
    for (size_t i = 0; i < N_VECS; i++)  gen_point(centers, db + i * DIM);
    for (size_t i = 0; i < N_QUERY; i++) gen_point(centers, qs + i * DIM);

    srp_make_planes(planes, DIM, 7);
    for (size_t i = 0; i < N_VECS; i++) codes[i] = srp_encode(planes, DIM, db + i * DIM);

    /* exact ground truth + baseline QPS */
    double t0 = now_s();
    for (int q = 0; q < N_QUERY; q++) exact_topk(db, qs + (size_t)q * DIM, truth + q * K_TRUE);
    double exact_qps = N_QUERY / (now_s() - t0);

    /* recall: exact top-10 found among SRP top-10 / top-100 candidates */
    size_t hit10 = 0, hit100 = 0;
    for (int q = 0; q < N_QUERY; q++) {
        qcodes[q] = srp_encode(planes, DIM, qs + (size_t)q * DIM);
        size_t m = srp_scan_topk(qcodes[q], codes, N_VECS, K_CAND, cand);
        uint32_t c10[K_TRUE];
        size_t m10 = srp_scan_topk(qcodes[q], codes, N_VECS, K_TRUE, c10);
        for (int t = 0; t < K_TRUE; t++) {
            uint32_t want = truth[q * K_TRUE + t];
            for (size_t c = 0; c < m; c++)   if (cand[c] == want) { hit100++; break; }
            for (size_t c = 0; c < m10; c++) if (c10[c] == want)  { hit10++;  break; }
        }
    }

    /* SRP QPS: encode query + Hamming top-100 scan, single thread */
    volatile uint32_t sink = 0;
    t0 = now_s();
    for (int r = 0; r < QPS_ROUNDS; r++)
        for (int q = 0; q < N_QUERY; q++) {
            uint32_t c = srp_encode(planes, DIM, qs + (size_t)q * DIM);
            srp_scan_topk(c, codes, N_VECS, K_CAND, cand);
            sink ^= cand[0];
        }
    double srp_qps = (double)QPS_ROUNDS * N_QUERY / (now_s() - t0);

    double p50 = 0, p99 = 0, p999 = 0;
    int ipc_ok = ncpu >= 2 ? run_ipc(&p50, &p99, &p999) : 1;

    double raw_mb  = (double)N_VECS * DIM * sizeof(float) / 1e6;
    double code_mb = (double)N_VECS * sizeof(uint32_t) / 1e6;

    puts(rule);
    puts("GENESIS L0-SIDECAR BENCHMARK (C99 POSIX SHM)");
    puts(rule);
    printf("Host:                   %s (%ld CPUs)\n", cpu, ncpu);
    printf("Dataset:                %d vectors (%d-dim, %d clusters, synthetic)\n",
           N_VECS, DIM, N_CLUSTERS);
    printf("Quantization:           SRP-LSH, %d random hyperplanes -> uint32\n", SRP_BITS);
    printf("Memory Footprint:       %.2f MB (vs %.1f MB raw float32)\n", code_mb, raw_mb);
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
    printf("Throughput, SRP top-%d: %.0f QPS (single core, POPCNT)\n", K_CAND, srp_qps);
    printf("Throughput, exact f32:  %.0f QPS (single core, brute-force scan)\n", exact_qps);
    printf("Recall 10@10:           %.1f %%  (SRP top-10 vs exact top-10)\n",
           100.0 * hit10 / (N_QUERY * K_TRUE));
    printf("Recall 10@%d:          %.1f %%  (exact top-10 within SRP top-%d)\n",
           K_CAND, 100.0 * hit100 / (N_QUERY * K_TRUE), K_CAND);
    puts(rule);
    (void)sink;
    free(centers); free(db); free(qs); free(planes);
    free(codes); free(qcodes); free(truth);
    return 0;
}
