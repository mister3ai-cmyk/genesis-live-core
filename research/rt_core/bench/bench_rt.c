/* Business Source License 1.1 - см. research/LICENSE
 * Латентность rt_core: CLOCK_MONOTONIC вокруг каждого вызова, 100 000 раз.
 * Печатает среднее, p50, p99, максимум. Не WCET: обычная ОС, не бортовое железо. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rt_grassmann.h"
#include "rt_kinematics.h"
#include "rt_xcorr.h"

#define N_RUNS 100000

static double samples[N_RUNS];
static volatile double sink;

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void report(const char *name, double overhead)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < N_RUNS; i++) {
        samples[i] -= overhead;
        if (samples[i] < 0)
            samples[i] = 0;
        sum += samples[i];
    }
    qsort(samples, N_RUNS, sizeof(double), cmp);
    printf("%-46s mean %8.1f ns  p50 %8.1f  p99 %8.1f  max %9.1f\n", name, sum / N_RUNS,
           samples[N_RUNS / 2], samples[N_RUNS * 99 / 100], samples[N_RUNS - 1]);
}

static unsigned long long rs = 88172645463325252ULL;
static double urand(void)
{
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (double)(rs >> 11) / 4503599627370496.0 - 1.0;
}

int main(void)
{
    static rt_context_t c;
    static double A[64 * 4], U[64 * 4], V[64 * 4];
    double overhead, claim[3], mine[3], t0;
    int i, a, j;
    char model[128] = "?";
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "model name", 10)) {
                const char *p = strchr(line, ':');
                size_t k = 0;
                p = p ? p + 2 : line;
                while (p[k] && p[k] != '\n' && k + 1 < sizeof model) {
                    model[k] = p[k];
                    k++;
                }
                model[k] = 0;
                break;
            }
        fclose(f);
    }
    printf("CPU: %s\n", model);
#ifdef __VERSION__
    printf("compiler: %s, flags: %s\n", __VERSION__, RT_BENCH_FLAGS);
#endif

    for (i = 0; i < N_RUNS; i++) {               /* цена самого замера */
        t0 = now_ns();
        samples[i] = now_ns() - t0;
    }
    qsort(samples, N_RUNS, sizeof(double), cmp);
    overhead = samples[N_RUNS / 2];
    printf("накладные clock_gettime (вычитаются): %.1f ns\n\n", overhead);

    /* кольцо заполнено: 4 автора x 8 отсчётов за 4 с, дуга с разгоном */
    rt_init(&c, NULL);
    for (j = 0; j < 8; j++)
        for (a = 0; a < 4; a++) {
            double t = 0.05 + 0.5 * j + 0.11 * a, p[3];
            p[0] = 10000 + 400 * t + 30 * t * t;
            p[1] = 20000 - 250 * t + 12 * t * t;
            p[2] = 500 + 20 * t;
            rt_kin_push(&c, 0, a, (int64_t)(t * 1e6), (int64_t)(t * 1e6), p);
        }
    mine[0] = 10000 + 400 * 4.0 + 30 * 16.0;
    mine[1] = 20000 - 250 * 4.0 + 12 * 16.0;
    mine[2] = 500 + 80.0;

    claim[0] = mine[0] + 20; claim[1] = mine[1]; claim[2] = mine[2];
    for (i = 0; i < N_RUNS; i++) {
        t0 = now_ns();
        sink = rt_kin_gate(&c, 0, 5, 4000000, 4000500, claim, mine, NULL);
        samples[i] = now_ns() - t0;
    }
    report("rt_kin_gate, быстрый путь (спора нет)", overhead);

    claim[0] = mine[0] + 1100; claim[1] = mine[1] - 900;
    for (i = 0; i < N_RUNS; i++) {
        t0 = now_ns();
        sink = rt_kin_gate(&c, 0, 5, 4000000, 4000500, claim, mine, NULL);
        samples[i] = now_ns() - t0;
    }
    report("rt_kin_gate, прогноз, окно честное", overhead);

    {   /* худший случай: в окне лжец (автор 0 сдвинут на 1200), нужны все кандидаты */
        static rt_context_t c2;
        rt_init(&c2, NULL);
        for (j = 0; j < 8; j++)
            for (a = 0; a < 4; a++) {
                double t = 0.05 + 0.5 * j + 0.11 * a, p[3];
                p[0] = 10000 + 400 * t + 30 * t * t + (a == 0 ? 1200 : 0);
                p[1] = 20000 - 250 * t + 12 * t * t;
                p[2] = 500 + 20 * t;
                rt_kin_push(&c2, 0, a, (int64_t)(t * 1e6), (int64_t)(t * 1e6), p);
            }
        for (i = 0; i < N_RUNS; i++) {
            t0 = now_ns();
            sink = rt_kin_gate(&c2, 0, 5, 4000000, 4000500, claim, mine, NULL);
            samples[i] = now_ns() - t0;
        }
        report("rt_kin_gate, прогноз, лжец в окне", overhead);
        printf("  (вердикт с лжецом в окне: %s)\n",
               rt_kin_gate(&c2, 0, 5, 4000000, 4000500, claim, mine, NULL) == RT_REFUTE ? "REFUTE" : "другой");
    }

    for (i = 0; i < 64 * 4; i++)
        A[i] = urand();
    rt_gr_orth_mgs(64, 4, A, U);
    for (i = 0; i < 64 * 4; i++)
        A[i] = urand();
    rt_gr_orth_mgs(64, 4, A, V);
    for (i = 0; i < N_RUNS; i++) {
        t0 = now_ns();
        sink = rt_gr_chordal2(64, 4, U, V);
        samples[i] = now_ns() - t0;
    }
    report("rt_gr_chordal2, Gr(4, 64)", overhead);
    for (i = 0; i < N_RUNS; i++) {
        t0 = now_ns();
        sink = rt_gr_orth_mgs(64, 4, A, V);
        samples[i] = now_ns() - t0;
    }
    report("rt_gr_orth_mgs, 64 x 4", overhead);
    {   /* взаимная корреляция: окно 2 с при 200 Гц, 3 канала, сдвиги +-20 */
        static double xa[3 * 400], xb[3 * 400];
        double lag, peak;
        int n_x = 2000;
        for (i = 0; i < 3 * 400; i++) {
            xa[i] = urand();
            xb[i] = xa[i] + 0.1 * urand();
        }
        for (i = 0; i < n_x; i++) {
            t0 = now_ns();
            sink = rt_xcorr_delay_multi(xa, xb, 400, 3, 20, 1e-6, &lag, &peak);
            samples[i] = now_ns() - t0;
        }
        qsort(samples, n_x, sizeof(double), cmp);
        printf("%-46s p50 %8.1f мкс  p99 %8.1f мкс (%d вызовов)\n", "rt_xcorr_delay_multi, 400 x 3, +-20",
               (samples[n_x / 2] - overhead) / 1e3, (samples[n_x * 99 / 100] - overhead) / 1e3, n_x);
    }

    /* пачкой: один интервал на N_RUNS вызовов - без дрожания таймера на вызов */
    printf("\nпачкой (%d вызовов за один замер, среднее):\n", N_RUNS);
    t0 = now_ns();
    for (i = 0; i < N_RUNS; i++)
        sink = rt_kin_gate(&c, 0, 5, 4000000, 4000500, claim, mine, NULL);
    printf("  rt_kin_gate, прогноз           %8.1f ns\n", (now_ns() - t0) / N_RUNS);
    t0 = now_ns();
    for (i = 0; i < N_RUNS; i++) {
        U[i & 255] += 1e-18;               /* не даём вынести вызов из цикла */
        sink = rt_gr_chordal2(64, 4, U, V);
    }
    printf("  rt_gr_chordal2, Gr(4, 64)      %8.1f ns\n", (now_ns() - t0) / N_RUNS);
    t0 = now_ns();
    for (i = 0; i < N_RUNS; i++)
        sink = rt_gr_orth_mgs(64, 4, A, V);
    printf("  rt_gr_orth_mgs, 64 x 4         %8.1f ns\n", (now_ns() - t0) / N_RUNS);
    printf("\nсчётчики шлюза: pass %u refute %u failsafe %u\n", c.n_pass, c.n_refute, c.n_failsafe);
    return 0;
}
