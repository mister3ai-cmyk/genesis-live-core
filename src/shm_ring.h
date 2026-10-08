/*
 * src/shm_ring.h
 * POSIX shared-memory mailbox for zero-copy handoff between processes
 * on the same host. After shm_ring_open(), reads and writes are plain
 * loads/stores into the mapping: no syscalls on the hot path.
 *
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef GENESIS_SHM_RING_H
#define GENESIS_SHM_RING_H

#include <stddef.h>
#include <stdint.h>

#define SHM_CACHELINE 64

/* Producer and consumer counters live on separate cache lines
 * to avoid false sharing. */
typedef struct {
    volatile uint64_t req;
    uint8_t _pad0[SHM_CACHELINE - sizeof(uint64_t)];
    volatile uint64_t ack;
    uint8_t _pad1[SHM_CACHELINE - sizeof(uint64_t)];
    uint32_t payload[SHM_CACHELINE / sizeof(uint32_t)];
} shm_mailbox;

/* Create (or open) a named segment of `size` bytes under /dev/shm.
 * Returns the mapping or NULL on error. */
void *shm_ring_open(const char *name, size_t size);

/* Unmap and unlink the segment. */
void shm_ring_close(const char *name, void *base, size_t size);

#endif
