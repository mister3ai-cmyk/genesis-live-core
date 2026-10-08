/*
 * src/srp_lsh.h
 * Two-stage cascade search:
 *   1. Sign Random Projection LSH (Charikar 2002): float vector -> 128-bit code,
 *      Hamming-distance candidate scan via hardware POPCNT.
 *   2. Exact inner-product re-rank of the candidates.
 *
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef GENESIS_SRP_LSH_H
#define GENESIS_SRP_LSH_H

#include <stddef.h>
#include <stdint.h>

/* Code length; override at build time, e.g. make CPPFLAGS=-DSRP_BITS=256 */
#ifndef SRP_BITS
#define SRP_BITS       128
#endif
#if SRP_BITS <= 0 || SRP_BITS % 64 != 0
#error "SRP_BITS must be a positive multiple of 64"
#endif
#define SRP_WORDS      (SRP_BITS / 64)
#define SRP_RERANK_MAX 64   /* largest k accepted by srp_rerank_topk */

typedef struct {
    uint64_t w[SRP_WORDS];
} srp_code;

/* Fill planes[SRP_BITS * dim] with N(0,1) hyperplane normals (deterministic). */
void srp_make_planes(float *planes, int dim, uint64_t seed);

/*
 * Optional mean-centering against anisotropic embeddings:
 * offsets[b] = <planes_b, mean>, so sign(<p, x - mean>) costs nothing extra.
 */
void srp_make_offsets(const float *planes, int dim, const float *mean, float *offsets);

/* Encode one vector: bit b = sign(<planes_b, x> - offsets[b]).
 * offsets may be NULL (no centering). */
srp_code srp_encode(const float *planes, int dim, const float *x, const float *offsets);

static inline int srp_hamming(const srp_code *a, const srp_code *b) {
    int d = 0;
    for (int i = 0; i < SRP_WORDS; i++)
        d += __builtin_popcountll(a->w[i] ^ b->w[i]);
    return d;
}

/*
 * Stage 1: return the k database codes closest to q in Hamming distance.
 * Writes up to k indices into out_idx (unordered), returns the count.
 * O(n) counting select; no allocation.
 */
size_t srp_scan_topk(const srp_code *q, const srp_code *db, size_t n,
                     size_t k, uint32_t *out_idx);

/* Inner product; dim must be a multiple of 16. */
float srp_dot(const float *a, const float *b, int dim);

/*
 * Stage 2: exact re-rank. Scores the m candidate rows of db against q and
 * writes the best k (k <= SRP_RERANK_MAX) into out_idx, best first.
 * Returns the count.
 */
size_t srp_rerank_topk(const float *db, int dim, const float *q,
                       const uint32_t *cand, size_t m,
                       size_t k, uint32_t *out_idx);

#endif
