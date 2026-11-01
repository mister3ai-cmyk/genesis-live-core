/*
 * tests/test_srp_lsh.c
 * Unit tests for the two-stage cascade in src/srp_lsh.c.
 *
 * Builds standalone: cc -O2 -std=c99 -I../src test_srp_lsh.c ../src/srp_lsh.c -lm
 *
 * Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
 */
#include "srp_lsh.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 64 /* srp_dot requires a multiple of 16 */

/* M_PI is not part of strict C99, and the build uses -std=c99. */
#define PI 3.14159265358979323846

static int failures = 0;
static int checks = 0;

static void check(int cond, const char *what) {
    checks++;
    if (!cond) {
        failures++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_near(double got, double want, double tol, const char *what) {
    checks++;
    if (!(fabs(got - want) <= tol)) {
        failures++;
        printf("  FAIL  %s (got %.6f, want %.6f +/- %.6f)\n", what, got, want, tol);
    }
}

/* Deterministic pseudo-random source, independent of the one under test. */
static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static float frand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)(rng_state >> 11) / 9007199254740992.0 * 2.0 - 1.0);
}

static void fill_random(float *v, int n) {
    for (int i = 0; i < n; i++) v[i] = frand();
}

/* ---------------------------------------------------------------- srp_dot */

static void test_dot(void) {
    printf("srp_dot\n");

    float a[DIM], b[DIM];

    /* Orthogonal basis vectors have zero inner product. */
    memset(a, 0, sizeof a);
    memset(b, 0, sizeof b);
    a[0] = 1.0f;
    b[1] = 1.0f;
    check_near(srp_dot(a, b, DIM), 0.0, 1e-6, "orthogonal unit vectors give 0");

    /* <x, x> is the squared norm. */
    for (int i = 0; i < DIM; i++) a[i] = 2.0f;
    check_near(srp_dot(a, a, DIM), 4.0 * DIM, 1e-3, "<x,x> equals squared norm");

    /* Symmetry and linearity in one scalar. */
    fill_random(a, DIM);
    fill_random(b, DIM);
    float ab = srp_dot(a, b, DIM);
    check_near(srp_dot(b, a, DIM), ab, 1e-5, "symmetric in its arguments");

    float a3[DIM];
    for (int i = 0; i < DIM; i++) a3[i] = 3.0f * a[i];
    check_near(srp_dot(a3, b, DIM), 3.0 * ab, 1e-4, "scales linearly");

    /* Against a straightforward reference loop: the 16-accumulator layout
     * exists so the compiler can vectorise, and must not change the result
     * beyond float rounding. */
    double ref = 0.0;
    for (int i = 0; i < DIM; i++) ref += (double)a[i] * (double)b[i];
    check_near(ab, ref, 1e-4, "matches a naive reference sum");
}

/* -------------------------------------------------------- srp_make_planes */

static void test_planes_deterministic(void) {
    printf("srp_make_planes\n");

    static float p1[SRP_BITS * DIM], p2[SRP_BITS * DIM], p3[SRP_BITS * DIM];

    srp_make_planes(p1, DIM, 12345);
    srp_make_planes(p2, DIM, 12345);
    check(memcmp(p1, p2, sizeof p1) == 0, "same seed reproduces the planes bit for bit");

    srp_make_planes(p3, DIM, 12346);
    check(memcmp(p1, p3, sizeof p1) != 0, "a different seed gives different planes");

    /* Normals are drawn from N(0,1): the sample mean should sit near zero and
     * the sample deviation near one. Loose bounds — this guards against a
     * broken generator, not against a slightly skewed one. */
    double sum = 0.0, sq = 0.0;
    const int n = SRP_BITS * DIM;
    for (int i = 0; i < n; i++) {
        sum += p1[i];
        sq += (double)p1[i] * p1[i];
    }
    double mean = sum / n;
    double sd = sqrt(sq / n - mean * mean);
    check_near(mean, 0.0, 0.05, "plane coefficients are centred on zero");
    check_near(sd, 1.0, 0.10, "plane coefficients have unit deviation");
}

/* -------------------------------------------------------------- srp_encode */

static void test_encode(void) {
    printf("srp_encode\n");

    static float planes[SRP_BITS * DIM];
    srp_make_planes(planes, DIM, 777);

    float x[DIM], scaled[DIM], negated[DIM];
    fill_random(x, DIM);
    for (int i = 0; i < DIM; i++) {
        scaled[i] = 5.0f * x[i];
        negated[i] = -x[i];
    }

    srp_code cx = srp_encode(planes, DIM, x, NULL);
    srp_code cs = srp_encode(planes, DIM, scaled, NULL);
    srp_code cn = srp_encode(planes, DIM, negated, NULL);

    check(srp_hamming(&cx, &cx) == 0, "a code is at distance 0 from itself");

    /* The code records sign(<p,x>), so a positive scalar cannot change it. */
    check(srp_hamming(&cx, &cs) == 0, "positive scaling leaves the code unchanged");

    /* Negation flips every sign, so the code is the exact complement. */
    check(srp_hamming(&cx, &cn) == SRP_BITS, "negation flips every bit");

    /* Bit b must equal sign(<planes_b, x>) with no offsets. */
    int mismatches = 0;
    for (int b = 0; b < SRP_BITS; b++) {
        int bit = (int)((cx.w[b >> 6] >> (b & 63)) & 1u);
        int want = srp_dot(planes + (size_t)b * DIM, x, DIM) >= 0.0f;
        if (bit != want) mismatches++;
    }
    check(mismatches == 0, "every bit equals the sign of its projection");
}

static void test_encode_offsets(void) {
    printf("srp_make_offsets\n");

    static float planes[SRP_BITS * DIM];
    srp_make_planes(planes, DIM, 4242);

    float mean[DIM], x[DIM], centred[DIM], offsets[SRP_BITS];
    fill_random(mean, DIM);
    fill_random(x, DIM);
    for (int i = 0; i < DIM; i++) centred[i] = x[i] - mean[i];

    srp_make_offsets(planes, DIM, mean, offsets);

    /* Encoding x against the offsets must equal encoding (x - mean) without
     * them: that equivalence is the whole point of precomputing offsets. */
    srp_code with_offsets = srp_encode(planes, DIM, x, offsets);
    srp_code pre_centred = srp_encode(planes, DIM, centred, NULL);
    check(srp_hamming(&with_offsets, &pre_centred) == 0,
          "offsets are equivalent to subtracting the mean first");

    /* Boundary of the comparison. The contract is bit = (dot >= thr), so a
     * projection landing exactly on the threshold must set the bit, not clear
     * it. Random data never produces an exact tie, which lets a `>` slip
     * through unnoticed; using x itself as the mean makes dot == thr hold for
     * every bit at once. */
    float self_offsets[SRP_BITS];
    srp_make_offsets(planes, DIM, x, self_offsets);
    srp_code on_threshold = srp_encode(planes, DIM, x, self_offsets);

    int bits_set = 0;
    for (int b = 0; b < SRP_BITS; b++)
        bits_set += (int)((on_threshold.w[b >> 6] >> (b & 63)) & 1u);
    check(bits_set == SRP_BITS, "a projection exactly on the threshold sets the bit");
}

/* ------------------------------------------------------ Charikar property */

static void test_charikar_angle(void) {
    printf("Charikar collision probability\n");

    /* For sign random projections the probability that two vectors differ in
     * a given bit is theta/pi. Build a pair at a known 60 degree angle and
     * check the measured Hamming fraction against 1/3. */
    static float planes[SRP_BITS * DIM];
    srp_make_planes(planes, DIM, 2024);

    const double theta = PI / 3.0; /* 60 degrees */
    double total = 0.0;
    const int trials = 200;

    for (int t = 0; t < trials; t++) {
        float u[DIM], w[DIM], v[DIM];
        fill_random(u, DIM);
        fill_random(w, DIM);

        /* Gram-Schmidt: make w orthogonal to u, then normalise both. */
        float nu = sqrtf(srp_dot(u, u, DIM));
        for (int i = 0; i < DIM; i++) u[i] /= nu;
        float proj = srp_dot(w, u, DIM);
        for (int i = 0; i < DIM; i++) w[i] -= proj * u[i];
        float nw = sqrtf(srp_dot(w, w, DIM));
        for (int i = 0; i < DIM; i++) w[i] /= nw;

        /* v sits at exactly theta from u in the plane spanned by u and w. */
        for (int i = 0; i < DIM; i++)
            v[i] = (float)(cos(theta) * u[i] + sin(theta) * w[i]);

        srp_code cu = srp_encode(planes, DIM, u, NULL);
        srp_code cv = srp_encode(planes, DIM, v, NULL);
        total += (double)srp_hamming(&cu, &cv) / SRP_BITS;
    }

    double measured = total / trials;
    check_near(measured, theta / PI, 0.03,
               "bit disagreement rate matches theta/pi");
}

/* --------------------------------------------------------- srp_scan_topk */

static int hamming_ref(const srp_code *a, const srp_code *b) {
    int d = 0;
    for (int i = 0; i < SRP_WORDS; i++) {
        uint64_t x = a->w[i] ^ b->w[i];
        while (x) { d += (int)(x & 1u); x >>= 1; }
    }
    return d;
}

static void test_scan_topk(void) {
    printf("srp_scan_topk\n");

    enum { N = 500, K = 16 };
    static float planes[SRP_BITS * DIM];
    static srp_code db[N];
    static uint32_t out[K];
    srp_make_planes(planes, DIM, 31337);

    float v[DIM];
    for (size_t i = 0; i < N; i++) {
        fill_random(v, DIM);
        db[i] = srp_encode(planes, DIM, v, NULL);
    }

    /* Query equal to a database entry: that entry must come back. */
    srp_code q = db[123];
    size_t m = srp_scan_topk(&q, db, N, K, out);
    check(m == K, "returns exactly k indices when n > k");

    int found_exact = 0, out_of_range = 0, duplicates = 0;
    for (size_t i = 0; i < m; i++) {
        if (out[i] == 123) found_exact = 1;
        if (out[i] >= N) out_of_range = 1;
        for (size_t j = i + 1; j < m; j++)
            if (out[i] == out[j]) duplicates = 1;
    }
    check(found_exact, "an exact match is among the candidates");
    check(!out_of_range, "every index is within the database");
    check(!duplicates, "no index is returned twice");

    /* The returned set must be the true k nearest: compare the largest
     * returned distance against the k-th smallest distance overall. */
    int worst_returned = 0;
    for (size_t i = 0; i < m; i++) {
        int d = hamming_ref(&q, &db[out[i]]);
        if (d > worst_returned) worst_returned = d;
    }
    int all[N];
    for (size_t i = 0; i < N; i++) all[i] = hamming_ref(&q, &db[i]);
    for (size_t i = 0; i < N; i++)
        for (size_t j = i + 1; j < N; j++)
            if (all[j] < all[i]) { int t = all[i]; all[i] = all[j]; all[j] = t; }
    check(worst_returned <= all[K - 1], "candidates are the true k nearest");

    /* Degenerate sizes must not overrun. */
    m = srp_scan_topk(&q, db, 5, K, out);
    check(m == 5, "returns n when n < k");
    m = srp_scan_topk(&q, db, N, 0, out);
    check(m == 0, "k = 0 returns nothing");
}

/* ------------------------------------------------------- srp_rerank_topk */

static void test_rerank_topk(void) {
    printf("srp_rerank_topk\n");

    enum { N = 200, M = 40, K = 10 };
    static float db[N * DIM];
    static uint32_t cand[M], out[SRP_RERANK_MAX];
    float q[DIM];

    fill_random(db, N * DIM);
    fill_random(q, DIM);
    for (size_t i = 0; i < M; i++) cand[i] = (uint32_t)(i * 3);

    size_t m = srp_rerank_topk(db, DIM, q, cand, M, K, out);
    check(m == K, "returns exactly k results");

    /* Output must be ordered best first. */
    int ordered = 1;
    for (size_t i = 1; i < m; i++)
        if (srp_dot(db + (size_t)out[i] * DIM, q, DIM) >
            srp_dot(db + (size_t)out[i - 1] * DIM, q, DIM))
            ordered = 0;
    check(ordered, "results are sorted by descending inner product");

    /* The single best candidate must be first. */
    uint32_t best = cand[0];
    float best_score = srp_dot(db + (size_t)cand[0] * DIM, q, DIM);
    for (size_t i = 1; i < M; i++) {
        float s = srp_dot(db + (size_t)cand[i] * DIM, q, DIM);
        if (s > best_score) { best_score = s; best = cand[i]; }
    }
    check(out[0] == best, "the top result is the true maximum");

    /* Every result must come from the candidate list. */
    int foreign = 0;
    for (size_t i = 0; i < m; i++) {
        int seen = 0;
        for (size_t j = 0; j < M; j++) if (out[i] == cand[j]) seen = 1;
        if (!seen) foreign = 1;
    }
    check(!foreign, "results are drawn only from the candidate list");

    /* Boundaries: k above the cap, k above m, and the empty case. */
    m = srp_rerank_topk(db, DIM, q, cand, M, SRP_RERANK_MAX + 50, out);
    check(m <= SRP_RERANK_MAX, "k is clamped to SRP_RERANK_MAX");
    m = srp_rerank_topk(db, DIM, q, cand, 3, K, out);
    check(m == 3, "returns m when m < k");
    m = srp_rerank_topk(db, DIM, q, cand, 0, K, out);
    check(m == 0, "an empty candidate list returns nothing");
}

/* -------------------------------------------------------- end-to-end pass */

static void test_cascade_recall(void) {
    printf("cascade end to end\n");

    /* The cascade must retrieve the exact nearest neighbour of a query that
     * is a small perturbation of a database vector. */
    enum { N = 400, KC = 64 };
    static float planes[SRP_BITS * DIM];
    static float db[N * DIM];
    static srp_code codes[N];
    static uint32_t cand[KC], out[8];

    srp_make_planes(planes, DIM, 555);
    fill_random(db, N * DIM);
    for (size_t i = 0; i < N; i++)
        codes[i] = srp_encode(planes, DIM, db + i * DIM, NULL);

    const size_t target = 77;
    float q[DIM];
    for (int i = 0; i < DIM; i++)
        q[i] = db[target * DIM + i] * 1.02f + 0.001f * frand();

    srp_code qc = srp_encode(planes, DIM, q, NULL);
    size_t m = srp_scan_topk(&qc, codes, N, KC, cand);
    check(m > 0, "stage 1 produced candidates");

    size_t r = srp_rerank_topk(db, DIM, q, cand, m, 8, out);
    check(r > 0, "stage 2 produced results");

    int hit = 0;
    for (size_t i = 0; i < r; i++) if (out[i] == target) hit = 1;
    check(hit, "a near-duplicate query retrieves its source vector");
}

int main(void) {
    printf("SRP-LSH unit tests (SRP_BITS=%d, dim=%d)\n\n", SRP_BITS, DIM);

    test_dot();
    test_planes_deterministic();
    test_encode();
    test_encode_offsets();
    test_charikar_angle();
    test_scan_topk();
    test_rerank_topk();
    test_cascade_recall();

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
