/*
 * src/srp_lsh.h
 * Sign Random Projection LSH (Charikar 2002): float vector -> 32-bit code,
 * Hamming-distance candidate scan via hardware POPCNT.
 *
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef GENESIS_SRP_LSH_H
#define GENESIS_SRP_LSH_H

#include <stddef.h>
#include <stdint.h>

#define SRP_BITS 32

/* Fill planes[SRP_BITS * dim] with N(0,1) hyperplane normals (deterministic). */
void srp_make_planes(float *planes, int dim, uint64_t seed);

/* Encode one vector: bit b = sign(<planes_b, x>). */
uint32_t srp_encode(const float *planes, int dim, const float *x);

/*
 * Return the k database codes closest to q in Hamming distance.
 * Writes up to k indices into out_idx (unordered), returns the count.
 * O(n) counting select; no allocation.
 */
size_t srp_scan_topk(uint32_t q, const uint32_t *db, size_t n,
                     size_t k, uint32_t *out_idx);

#endif
