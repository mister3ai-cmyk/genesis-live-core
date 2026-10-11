/*
 * l0_api.c — shared-library facade over genesis-live-core src/srp_lsh.c
 * Same cascade as benchmarks/run_benchmark.c: planes seed 7, optional
 * mean-centering, Hamming top-pool -> exact re-rank top-k.
 */
#define _POSIX_C_SOURCE 199309L
#include "srp_lsh.h"

#include <stdlib.h>
#include <time.h>

static float    *g_planes;
static float     g_offsets[SRP_BITS];
static int       g_centered;
static srp_code *g_codes;
static const float *g_db;      /* owned by caller, must outlive queries */
static size_t    g_n;
static int       g_dim;
static uint32_t *g_cand;
static size_t    g_pool;

int l0_bits(void) { return SRP_BITS; }

int l0_build(const float *db, size_t n, int dim, uint64_t seed, int center, size_t pool) {
    if (dim % 16 != 0 || n == 0 || pool == 0) return -1;
    free(g_planes); free(g_codes); free(g_cand);
    g_planes = malloc((size_t)SRP_BITS * dim * sizeof(float));
    g_codes  = malloc(n * sizeof(srp_code));
    g_cand   = malloc(pool * sizeof(uint32_t));
    if (!g_planes || !g_codes || !g_cand) return -2;

    srp_make_planes(g_planes, dim, seed);
    g_centered = 0;
    if (center) {
        double *acc = calloc(dim, sizeof(double));
        float *mean = malloc(dim * sizeof(float));
        if (!acc || !mean) { free(acc); free(mean); return -2; }
        for (size_t i = 0; i < n; i++)
            for (int j = 0; j < dim; j++) acc[j] += db[i * dim + j];
        for (int j = 0; j < dim; j++) mean[j] = (float)(acc[j] / n);
        srp_make_offsets(g_planes, dim, mean, g_offsets);
        g_centered = 1;
        free(acc); free(mean);
    }
    for (size_t i = 0; i < n; i++)
        g_codes[i] = srp_encode(g_planes, dim, db + i * dim, g_centered ? g_offsets : NULL);

    g_db = db; g_n = n; g_dim = dim; g_pool = pool < n ? pool : n;
    return 0;
}

/* Top-k glyph_ids (best first). *ns_inside = time spent inside C. */
size_t l0_query(const float *q, size_t k, uint32_t *out, double *ns_inside) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    srp_code qc = srp_encode(g_planes, g_dim, q, g_centered ? g_offsets : NULL);
    size_t m = srp_scan_topk(&qc, g_codes, g_n, g_pool, g_cand);
    size_t r = srp_rerank_topk(g_db, g_dim, q, g_cand, m, k, out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (ns_inside)
        *ns_inside = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    return r;
}

size_t l0_index_bytes(void) { return g_n * sizeof(srp_code); }
