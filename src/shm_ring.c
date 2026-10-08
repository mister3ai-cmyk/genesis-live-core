/*
 * src/shm_ring.c
 * Licensed under the Apache License, Version 2.0.
 */
#define _POSIX_C_SOURCE 200809L
#include "shm_ring.h"

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

void shm_ring_close(const char *name, void *base, size_t size) {
    if (base)
        munmap(base, size);
    shm_unlink(name);
}
