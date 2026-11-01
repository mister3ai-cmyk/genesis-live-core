/*
 * src/shm_ring.h
 * POSIX shared-memory mailbox for zero-copy handoff between processes
 * on the same host. After shm_ring_open(), reads and writes are plain
 * loads/stores into the mapping: no syscalls on the hot path.
 *
 * Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
 */
#ifndef GENESIS_SHM_RING_H
#define GENESIS_SHM_RING_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#define SHM_CACHELINE 64

/* Producer and consumer counters live on separate cache lines
 * to avoid false sharing. They are C11 atomics: `volatile` alone would
 * stop the compiler caching the value in a register but would not order
 * the surrounding payload accesses, which is what the acquire/release
 * pairing in the ping-pong loop relies on. */
typedef struct {
    _Atomic uint64_t req;
    uint8_t _pad0[SHM_CACHELINE - sizeof(uint64_t)];
    _Atomic uint64_t ack;
    uint8_t _pad1[SHM_CACHELINE - sizeof(uint64_t)];
    uint32_t payload[SHM_CACHELINE / sizeof(uint32_t)];
} shm_mailbox;

/* Outcome of shm_ring_pin(). The distinction matters: only SHM_PIN_LOCKED
 * keeps the mapping out of the page reclaim path, and an unprivileged
 * container usually cannot get there. */
typedef enum {
    SHM_PIN_LOCKED  = 0,  /* mlock() held: no page fault on the hot path   */
    SHM_PIN_ADVISED = 1,  /* mlock() refused, kernel hinted only (see below) */
    SHM_PIN_NONE    = 2   /* neither worked: expect faults under pressure  */
} shm_pin_result;

/* Create (or open) a named segment of `size` bytes under /dev/shm.
 * Returns the mapping or NULL on error. */
void *shm_ring_open(const char *name, size_t size);

/* Ask the kernel to keep the mapping resident.
 *
 * mlock() is capped by RLIMIT_MEMLOCK, which most container runtimes leave
 * at 64 KB, so the call fails with EPERM or ENOMEM on a segment of any real
 * size. Failing hard there would make the library unusable in exactly the
 * environment it targets, and failing silently would hand the caller a
 * guarantee it does not have. So the attempt degrades one step and reports
 * which step it reached; `out_errno` (optional) carries why mlock() refused.
 *
 * Be clear about what the fallback is worth: POSIX_MADV_WILLNEED is advice,
 * not a synchronous read. Measured on a 1 GiB index, it returns in under a
 * millisecond and the first query still pays the full cold-start cost, while
 * a successful mlock() pays that cost once, up front, outside the query path.
 * SHM_PIN_ADVISED therefore means "no guarantee", not "almost as good".
 *
 * Raising the ceiling is a deployment matter, not a code one: see the
 * container prerequisites in README.md. */
shm_pin_result shm_ring_pin(void *base, size_t size, int *out_errno);

/* One-line description of a pin result, for diagnostics. */
const char *shm_pin_describe(shm_pin_result r);

/* Unmap and unlink the segment. */
void shm_ring_close(const char *name, void *base, size_t size);

#endif
