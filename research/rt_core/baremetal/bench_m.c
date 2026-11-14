/* Business Source License 1.1 - см. research/LICENSE
 *
 * Цена вызовов rt_core в ИНСТРУКЦИЯХ на голом Cortex-M под qemu-system-arm
 * с -icount: виртуальное время растёт строго на 2^shift нс за инструкцию,
 * SysTick тикает от этого времени. Тиков на инструкцию - калибровка на блоке
 * из известного числа инструкций. Это число выполненных инструкций (включая
 * программную double на M4), а не такты кремния: ожидания флеш-памяти,
 * конвейер и кэш qemu не моделирует.
 *
 * Сценарии: фиксированные (как bench/bench_rt.c) и случайные потоки
 * (авторы 2..8, лжецы, грубые сбои, вытеснения, перебазирование,
 * горизонты 0.5 и 4 с) - худший найденный случай и перцентили.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rt_kinematics.h"

#define SYST_CSR (*(volatile uint32_t *)0xE000E010)
#define SYST_RVR (*(volatile uint32_t *)0xE000E014)
#define SYST_CVR (*(volatile uint32_t *)0xE000E018)

static double tick_per_insn, overhead_ticks;

static inline uint32_t ticks(void)
{
    return SYST_CVR;                      /* считает вниз, 24 бита */
}

static double elapsed(uint32_t a, uint32_t b)
{
    return (double)((a - b) & 0xFFFFFFu);
}

/* 1000 инструкций add подряд (ровно 1000: .rept разворачивает) */
static void __attribute__((noinline)) block1000(void)
{
    __asm__ volatile(".rept 1000\n\tadds r0, r0, #1\n\t.endr" ::: "r0", "cc");
}

static void calibrate(void)
{
    uint32_t a, b;
    double o = 1e30, k = 1e30;
    int i;
    for (i = 0; i < 50; i++) {
        a = ticks();
        b = ticks();
        if (elapsed(a, b) < o)
            o = elapsed(a, b);
        a = ticks();
        block1000();
        b = ticks();
        if (elapsed(a, b) < k)
            k = elapsed(a, b);
    }
    overhead_ticks = o;
    tick_per_insn = (k - o) / 1000.0;     /* вызов/возврат block1000 - пара инструкций, < 0.5 % */
}

static double insn(uint32_t a, uint32_t b)
{
    double t = elapsed(a, b) - overhead_ticks;
    return t > 0 ? t / tick_per_insn : 0.0;
}

#define N_MAX 40000
static float samples[N_MAX];
static int n_s;

static int cmpf(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static void report(const char *name)
{
    qsort(samples, (size_t)n_s, sizeof(float), cmpf);
    printf("%-44s n %6d  p50 %7.0f  p99 %7.0f  max %7.0f инстр.\n", name, n_s,
           (double)samples[n_s / 2], (double)samples[n_s * 99 / 100], (double)samples[n_s - 1]);
    n_s = 0;
}

static unsigned long long rs = 88172645463325252ULL;
static double urand(void)                 /* [0, 1) */
{
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (double)(rs >> 11) / 9007199254740992.0;
}
static double gauss(void)
{
    double u = urand() + 1e-12, v = urand();
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static void fixed_scenarios(void)
{
    static rt_context_t c, c2;
    double claim[3], mine[3];
    uint32_t a0, b0;
    int i, j, a;
    rt_init(&c, NULL);
    rt_init(&c2, NULL);
    for (j = 0; j < 8; j++)
        for (a = 0; a < 4; a++) {
            double t = 0.05 + 0.5 * j + 0.11 * a, p[3];
            p[0] = 10000 + 400 * t + 30 * t * t;
            p[1] = 20000 - 250 * t + 12 * t * t;
            p[2] = 500 + 20 * t;
            rt_kin_push(&c, 0, a, (int64_t)(t * 1e6), (int64_t)(t * 1e6), p);
            p[0] += a == 0 ? 1200 : 0;
            rt_kin_push(&c2, 0, a, (int64_t)(t * 1e6), (int64_t)(t * 1e6), p);
        }
    mine[0] = 10000 + 400 * 4.0 + 30 * 16.0;
    mine[1] = 20000 - 250 * 4.0 + 12 * 16.0;
    mine[2] = 500 + 80.0;
    claim[0] = mine[0] + 20; claim[1] = mine[1]; claim[2] = mine[2];
    for (i = 0; i < 200; i++) {
        a0 = ticks();
        rt_kin_gate(&c, 0, 5, 4000000, 4000500, claim, mine, NULL);
        b0 = ticks();
        samples[n_s++] = (float)insn(a0, b0);
    }
    report("gate: быстрый путь (спора нет)");
    claim[0] = mine[0] + 1100; claim[1] = mine[1] - 900;
    for (i = 0; i < 200; i++) {
        a0 = ticks();
        rt_kin_gate(&c, 0, 5, 4000000, 4000500, claim, mine, NULL);
        b0 = ticks();
        samples[n_s++] = (float)insn(a0, b0);
    }
    report("gate: прогноз, окно честное (4 автора)");
    for (i = 0; i < 200; i++) {
        a0 = ticks();
        rt_kin_gate(&c2, 0, 5, 4000000, 4000500, claim, mine, NULL);
        b0 = ticks();
        samples[n_s++] = (float)insn(a0, b0);
    }
    report("gate: прогноз, лжец в окне (4 автора)");
    {   /* максимум авторов: 8 x 4 точки = полное кольцо, автор 0 лжёт */
        static rt_context_t c3;
        rt_init(&c3, NULL);
        for (j = 0; j < 4; j++)
            for (a = 0; a < RT_MAX_AUTHORS; a++) {
                double t = 0.05 + 0.9 * j + 0.11 * a, p[3];
                p[0] = 10000 + 400 * t + 30 * t * t + (a == 0 ? 1200 : 0);
                p[1] = 20000 - 250 * t + 12 * t * t;
                p[2] = 500 + 20 * t;
                rt_kin_push(&c3, 0, a, (int64_t)(t * 1e6), (int64_t)(t * 1e6), p);
            }
        for (i = 0; i < 200; i++) {
            a0 = ticks();
            rt_kin_gate(&c3, 0, 1, 4000000, 4000500, claim, mine, NULL);
            b0 = ticks();
            samples[n_s++] = (float)insn(a0, b0);
        }
        report("gate: прогноз, лжец в окне (8 авторов)");
    }
}

/* Случайные потоки: на каждом шаге push и gate, оба замеряются. */
static void random_streams(int n_streams, int n_steps)
{
    static rt_context_t c;
    static float push_s[N_MAX];
    int n_push = 0, k, step;
    double worst_gate = 0, worst_push = 0;
    for (k = 0; k < n_streams; k++) {
        rt_cfg_t cfg;
        int n_auth = 2 + (int)(urand() * 7), liar = (int)(urand() * n_auth);
        int has_liar = urand() < 0.6;
        double tol = k % 2 ? 0.4 : 400.0, cf[9], noise = tol * 0.05;
        int64_t t = (int64_t)(urand() * 1e9), dt;
        static const int64_t dts[4] = {20000, 40000, 150000, 400000};
        rt_cfg_default(&cfg);
        cfg.horizon_us = k % 2 ? 500000 : 4000000;
        cfg.drop_tol = tol * 0.7;
        cfg.thr = tol * 1.4;
        cfg.pos_tol = tol * 0.7;
        rt_init(&c, &cfg);
        for (step = 0; step < 9; step++)
            cf[step] = (urand() - 0.5) * 100.0;
        dt = dts[(int)(urand() * 4)];
        for (step = 0; step < n_steps; step++) {
            double x[3], tq_s, claim[3], mine[3];
            int a = (int)(urand() * n_auth), ax;
            int64_t tq;
            uint32_t a0, b0;
            t += dt / 2 + (int64_t)(urand() * (double)dt);
            tq_s = (double)(t % 1000000000) / 1e6;
            for (ax = 0; ax < 3; ax++)
                x[ax] = cf[3 * ax] + cf[3 * ax + 1] * tq_s + cf[3 * ax + 2] * tq_s * tq_s * 0.1 + gauss() * noise;
            if (has_liar && a == liar && step > n_steps / 3)
                x[(int)(urand() * 3)] += (urand() < 0.5 ? -1 : 1) * tol * (1.5 + 2.5 * urand());
            if (urand() < 0.03)
                x[(int)(urand() * 3)] += tol * 10;
            a0 = ticks();
            rt_kin_push(&c, 0, a, t, t + 1000, x);
            b0 = ticks();
            if (n_push < N_MAX)
                push_s[n_push++] = (float)insn(a0, b0);
            tq = t + (int64_t)((urand() - 0.5) * 2.0 * (double)dt);
            tq_s = (double)(tq % 1000000000) / 1e6;
            for (ax = 0; ax < 3; ax++) {
                double tr = cf[3 * ax] + cf[3 * ax + 1] * tq_s + cf[3 * ax + 2] * tq_s * tq_s * 0.1;
                claim[ax] = tr + gauss() * tol * 0.3;
                mine[ax] = tr + gauss() * noise;
            }
            a0 = ticks();
            rt_kin_gate(&c, 0, (int)(urand() * n_auth), tq, tq + 2000, claim, mine, NULL);
            b0 = ticks();
            if (n_s < N_MAX)
                samples[n_s++] = (float)insn(a0, b0);
        }
    }
    for (k = 0; k < n_s; k++)
        if (samples[k] > worst_gate)
            worst_gate = samples[k];
    for (k = 0; k < n_push; k++)
        if (push_s[k] > worst_push)
            worst_push = push_s[k];
    report("gate: случайные потоки (авторы 2..8, лжецы)");
    for (k = 0; k < n_push; k++)
        samples[k] = push_s[k];
    n_s = n_push;
    report("push: те же потоки (с перебазированием)");
    (void)worst_gate;
    (void)worst_push;
}

int main(void)
{
    SYST_RVR = 0xFFFFFF;
    SYST_CVR = 0;
    SYST_CSR = 5;                         /* включён, тактирование от процессора */
    calibrate();
    printf("калибровка: %.4f тика SysTick на инструкцию, накладные %.0f тиков\n",
           tick_per_insn, overhead_ticks);
    if (tick_per_insn < 0.5) {
        printf("FAIL: мало тиков на инструкцию - запускать с -icount shift=7\n");
        return 1;
    }
    fixed_scenarios();
    random_streams(200, 160);
    printf("bench-m: ok\n");
    return 0;
}
