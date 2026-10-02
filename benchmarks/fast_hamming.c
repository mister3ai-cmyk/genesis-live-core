/*
 * benchmarks/fast_hamming.c
 * NGP 4.6 — bare-metal Hamming distance kernel (C99, __builtin_popcount)
 * Compiled as a shared library, called from Python via ctypes.
 *
 * Build (done by Makefile):
 *   gcc -O3 -march=native -std=c99 -shared -fPIC \
 *       -o fast_hamming.so fast_hamming.c
 */
#include <stdint.h>
#include <stddef.h>

/*
 * hamming_batch_u32
 * -----------------
 * For each of the `n_queries` query codes, computes Hamming distance
 * to every one of the `n_db` database codes and writes the nearest-
 * neighbour index into `out_indices[q]`.
 *
 * Uses __builtin_popcount which maps to the hardware POPCNT instruction
 * when compiled with -march=native.
 */
void hamming_batch_u32(
        const uint32_t * __restrict__ q_codes,   /* (n_queries,) */
        const uint32_t * __restrict__ db_codes,  /* (n_db,)      */
        int32_t        * __restrict__ out_indices,/* (n_queries,) */
        size_t n_queries,
        size_t n_db)
{
    for (size_t q = 0; q < n_queries; q++) {
        uint32_t qc   = q_codes[q];
        int      best = 33;          /* max possible = 32 + 1 sentinel */
        int32_t  best_idx = 0;
        for (size_t d = 0; d < n_db; d++) {
            int dist = __builtin_popcount(qc ^ db_codes[d]);
            if (dist < best) {
                best     = dist;
                best_idx = (int32_t)d;
            }
        }
        out_indices[q] = best_idx;
    }
}

/*
 * hamming_batch_u32_distances
 * ----------------------------
 * Same, but writes the minimum Hamming distance (not index) into out_dist.
 * Used for QPS measurement without result-array penalty.
 */
void hamming_batch_u32_distances(
        const uint32_t * __restrict__ q_codes,
        const uint32_t * __restrict__ db_codes,
        int32_t        * __restrict__ out_dist,
        size_t n_queries,
        size_t n_db)
{
    for (size_t q = 0; q < n_queries; q++) {
        uint32_t qc   = q_codes[q];
        int      best = 33;
        for (size_t d = 0; d < n_db; d++) {
            int dist = __builtin_popcount(qc ^ db_codes[d]);
            if (dist < best) best = dist;
        }
        out_dist[q] = (int32_t)best;
    }
}
