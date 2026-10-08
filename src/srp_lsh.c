/*
 * src/srp_lsh.c
 * Licensed under the Apache License, Version 2.0.
 */
#include "srp_lsh.h"

#include <math.h>

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float gauss(uint64_t *s) {
    double u1 = ((splitmix64(s) >> 11) + 1.0) / 9007199254740993.0;
    double u2 = (splitmix64(s) >> 11) / 9007199254740992.0;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

void srp_make_planes(float *planes, int dim, uint64_t seed) {
    for (int i = 0; i < SRP_BITS * dim; i++)
        planes[i] = gauss(&seed);
}

uint32_t srp_encode(const float *planes, int dim, const float *x) {
    uint32_t code = 0;
    for (int b = 0; b < SRP_BITS; b++) {
        const float *p = planes + (size_t)b * dim;
        float dot = 0.0f;
        for (int j = 0; j < dim; j++)
            dot += p[j] * x[j];
        code |= (uint32_t)(dot >= 0.0f) << b;
    }
    return code;
}

size_t srp_scan_topk(uint32_t q, const uint32_t *db, size_t n,
                     size_t k, uint32_t *out_idx) {
    /* Pass 1: histogram of distances 0..32. */
    size_t hist[SRP_BITS + 1] = {0};
    for (size_t i = 0; i < n; i++)
        hist[__builtin_popcount(q ^ db[i])]++;

    /* Largest distance threshold t such that count(dist < t) <= k. */
    int t = 0;
    size_t below = 0;
    while (t <= SRP_BITS && below + hist[t] <= k)
        below += hist[t++];

    /* Pass 2: take everything below t, fill the rest with dist == t. */
    size_t m = 0, tie_room = k - below;
    for (size_t i = 0; i < n && m < k; i++) {
        int d = __builtin_popcount(q ^ db[i]);
        if (d < t) {
            out_idx[m++] = (uint32_t)i;
        } else if (d == t && tie_room > 0) {
            out_idx[m++] = (uint32_t)i;
            tie_room--;
        }
    }
    return m;
}
