/*
 * src/shm_ring.c
 * Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
 */
#define _POSIX_C_SOURCE 200809L
#include "shm_ring.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

void *shm_ring_open(const char *name, size_t size) {
    int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (fd < 0)
        return NULL;
    if (ftruncate(fd, (off_t)size) < 0) {
        close(fd);
        return NULL;
    }
    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return base == MAP_FAILED ? NULL : base;
}

shm_pin_result shm_ring_pin(void *base, size_t size, int *out_errno) {
    if (out_errno)
        *out_errno = 0;
    if (!base || size == 0)
        return SHM_PIN_NONE;

    if (mlock(base, size) == 0)
        return SHM_PIN_LOCKED;

    /* Typically EPERM (no CAP_IPC_LOCK) or ENOMEM (RLIMIT_MEMLOCK too low).
     * Keep it for the caller: the two cases call for different fixes. */
    if (out_errno)
        *out_errno = errno;

    /* posix_madvise is the portable spelling of madvise(MADV_WILLNEED); it
     * returns the error directly rather than through errno. Pages get read
     * in now, but stay eligible for reclaim later. */
    if (posix_madvise(base, size, POSIX_MADV_WILLNEED) == 0)
        return SHM_PIN_ADVISED;

    return SHM_PIN_NONE;
}

const char *shm_pin_describe(shm_pin_result r) {
    switch (r) {
    case SHM_PIN_LOCKED:  return "locked (mlock held, no reclaim)";
    case SHM_PIN_ADVISED: return "advised (prefaulted, reclaimable)";
    default:              return "none (faults expected under pressure)";
    }
}

void shm_ring_close(const char *name, void *base, size_t size) {
    if (base)
        munmap(base, size);
    shm_unlink(name);
}
