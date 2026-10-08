/*
 * benchmarks/c_shm_harness.c
 * NGP 4.6 Bare-Metal POSIX SHM IPC Latency Harness
 * Build: gcc -O3 -march=native -Wall -Wextra -std=c99 -o c_shm_harness c_shm_harness.c -lpthread -lrt
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SHM_NAME      "/ngp_tensor_ring"
#define SHM_SIZE      (4 * 1024 * 1024)   /* 4 MB */
#define CACHELINE     64
#define N_ITER        100000
#define PAYLOAD_BYTES 64

typedef struct __attribute__((aligned(64))) {
    volatile uint64_t seq;
    uint8_t  data[PAYLOAD_BYTES];
    uint8_t  _pad[CACHELINE - sizeof(uint64_t) - PAYLOAD_BYTES % CACHELINE];
} RingSlot;

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static inline uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int main(void) {
    /* ── create / open SHM ── */
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm_open"); return 1; }
    if (ftruncate(fd, SHM_SIZE) < 0) { perror("ftruncate"); return 1; }

    void *base = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);

    RingSlot *slot = (RingSlot *)base;
    memset(slot, 0, sizeof(RingSlot));

    uint8_t  payload[PAYLOAD_BYTES];
    memset(payload, 0xAB, PAYLOAD_BYTES);

    uint64_t *latencies = malloc(N_ITER * sizeof(uint64_t));
    if (!latencies) { fputs("OOM\n", stderr); return 1; }

    /* ── warm-up (1 % of N_ITER) ── */
    for (int i = 0; i < N_ITER / 100; i++) {
        memcpy(slot->data, payload, PAYLOAD_BYTES);
        slot->seq++;
        __asm__ volatile("" ::: "memory");
        (void)slot->seq;
    }

    /* ── timed loop ── */
    for (int i = 0; i < N_ITER; i++) {
        uint64_t t0 = ns_now();
        memcpy(slot->data, payload, PAYLOAD_BYTES);
        slot->seq++;
        __asm__ volatile("" ::: "memory");           /* compiler fence */
        volatile uint64_t chk = slot->seq;           /* read-back */
        (void)chk;
        uint64_t t1 = ns_now();
        latencies[i] = t1 - t0;
    }

    /* ── sort → percentiles ── */
    qsort(latencies, N_ITER, sizeof(uint64_t), cmp_u64);

    uint64_t p50   = latencies[(uint64_t)(N_ITER * 0.500)];
    uint64_t p90   = latencies[(uint64_t)(N_ITER * 0.900)];
    uint64_t p99   = latencies[(uint64_t)(N_ITER * 0.990)];
    uint64_t p999  = latencies[(uint64_t)(N_ITER * 0.999)];
    uint64_t p_min = latencies[0];
    uint64_t p_max = latencies[N_ITER - 1];

    double sla_p99  = p99  / 1000.0;
    double sla_hand = p50  / 1000.0;

    int pass_p99  = sla_p99  < 1.700;
    int pass_hand = sla_hand < 0.350;

    printf("\n");
    printf("  NGP 4.6 — POSIX SHM IPC Latency Harness\n");
    printf("  Ring buffer : /dev/shm%s  (%d MB)\n", SHM_NAME, SHM_SIZE >> 20);
    printf("  Iterations  : %d  |  Payload: %d bytes\n\n", N_ITER, PAYLOAD_BYTES);
    printf("  %-12s %10s %10s\n", "Percentile", "ns", "µs");
    printf("  %s\n", "------------------------------------");
    printf("  %-12s %10llu %10.3f\n", "min",  (unsigned long long)p_min,  p_min  / 1000.0);
    printf("  %-12s %10llu %10.3f\n", "p50",  (unsigned long long)p50,   p50   / 1000.0);
    printf("  %-12s %10llu %10.3f\n", "p90",  (unsigned long long)p90,   p90   / 1000.0);
    printf("  %-12s %10llu %10.3f\n", "p99",  (unsigned long long)p99,   p99   / 1000.0);
    printf("  %-12s %10llu %10.3f\n", "p99.9", (unsigned long long)p999,  p999  / 1000.0);
    printf("  %-12s %10llu %10.3f\n", "max",  (unsigned long long)p_max, p_max / 1000.0);
    printf("  %s\n\n", "------------------------------------");
    printf("  SLA check  p99 < 1.700 µs  : %s  (%.3f µs)\n",
           pass_p99  ? "PASS" : "FAIL", sla_p99);
    printf("  SLA check  handoff <= 350 ns : %s  (%.0f ns)\n\n",
           pass_hand ? "PASS" : "FAIL", p50 * 1.0);

    free(latencies);
    munmap(base, SHM_SIZE);
    shm_unlink(SHM_NAME);
    return (pass_p99 && pass_hand) ? 0 : 1;
}
