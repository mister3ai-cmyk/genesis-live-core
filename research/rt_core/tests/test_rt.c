/* Business Source License 1.1 - см. research/LICENSE
 * Юнит-тесты rt_core без фреймворка: make check. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rt_grassmann.h"
#include "rt_kinematics.h"
#include "rt_port.h"
#include "rt_xcorr.h"

static int n_ok, n_fail;

#define CHECK(cond) do { \
    if (cond) n_ok++; \
    else { n_fail++; printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

static unsigned long long rs = 88172645463325252ULL;

static double urand(void)              /* xorshift64, [-1, 1) */
{
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return (double)(rs >> 11) / 4503599627370496.0 - 1.0;
}

static void rand_mat(int n, int k, double *A)
{
    int i;
    for (i = 0; i < n * k; i++)
        A[i] = urand();
}

static double orth_err(int n, int k, const double *Q)
{
    double e = 0.0;
    int a, b, i;
    for (a = 0; a < k; a++)
        for (b = 0; b < k; b++) {
            double s = 0.0;
            for (i = 0; i < n; i++)
                s += Q[a * n + i] * Q[b * n + i];
            s -= a == b ? 1.0 : 0.0;
            if (fabs(s) > e)
                e = fabs(s);
        }
    return e;
}

static void test_grassmann(void)
{
    static double A[128 * 4], Q[128 * 4], V[128 * 4], R[4 * 4], QR[128 * 4];
    int n, i, a, b, c;
    for (n = 64; n <= 128; n += 64) {
        rand_mat(n, 4, A);
        CHECK(rt_gr_orth_mgs(n, 4, A, Q) == RT_OK);
        CHECK(orth_err(n, 4, Q) < 1e-12);
        CHECK(fabs(rt_gr_chordal2(n, 4, Q, Q)) < 1e-12);
        /* базис не важен: Q -> Q R, R - ортогональная 4 x 4 */
        rand_mat(4, 4, A);
        CHECK(rt_gr_orth_mgs(4, 4, A, R) == RT_OK);
        for (a = 0; a < 4; a++)
            for (i = 0; i < n; i++) {
                double s = 0.0;
                for (c = 0; c < 4; c++)
                    s += Q[c * n + i] * R[a * 4 + c];
                QR[a * n + i] = s;
            }
        rand_mat(n, 4, A);
        CHECK(rt_gr_orth_mgs(n, 4, A, V) == RT_OK);
        CHECK(fabs(rt_gr_chordal2(n, 4, Q, V) - rt_gr_chordal2(n, 4, QR, V)) < 1e-12);
        CHECK(rt_gr_chordal2(n, 4, Q, V) >= 0.0 && rt_gr_chordal2(n, 4, Q, V) <= 4.0);
    }
    /* крайние значения: e0..e3 против самих себя и против e4..e7 */
    memset(A, 0, sizeof(A));
    memset(V, 0, sizeof(V));
    for (b = 0; b < 4; b++) {
        A[b * 64 + b] = 1.0;
        V[b * 64 + 4 + b] = 1.0;
    }
    CHECK(fabs(rt_gr_chordal2(64, 4, A, A)) < 1e-15);
    CHECK(fabs(rt_gr_chordal2(64, 4, A, V) - 4.0) < 1e-15);
    /* вырожденный вход и границы */
    rand_mat(64, 4, A);
    memcpy(A + 3 * 64, A + 1 * 64, 64 * sizeof(double));
    CHECK(rt_gr_orth_mgs(64, 4, A, Q) == RT_ERR_RANK);
    CHECK(rt_gr_orth_mgs(3, 4, A, Q) == RT_ERR_ARG);
    CHECK(rt_gr_orth_mgs(RT_GR_MAX_N + 1, 4, A, Q) == RT_ERR_ARG);
    /* на месте */
    rand_mat(64, 4, A);
    CHECK(rt_gr_orth_mgs(64, 4, A, A) == RT_OK && orth_err(64, 4, A) < 1e-12);
}

/* --------------------------------------------------------- кинематика */

static void line(double t_s, double out[3])          /* 400, -250, 30 ед/с */
{
    out[0] = 10000.0 + 400.0 * t_s;
    out[1] = 20000.0 - 250.0 * t_s;
    out[2] = 500.0 + 30.0 * t_s;
}

static void parab(double t_s, double out[3])
{
    out[0] = 1000.0 + 300.0 * t_s + 40.0 * t_s * t_s;
    out[1] = 2000.0 - 100.0 * t_s + 25.0 * t_s * t_s;
    out[2] = 300.0 + 5.0 * t_s * t_s;
}

/* 4 автора, отсчёты в 1.0 + 0.23 a + 0.9 j секунд (до 4 с); лжец - автор liar */
static void feed(rt_context_t *c, void (*f)(double, double *), int liar)
{
    int a, j;
    for (j = 0; j < 4; j++)
        for (a = 0; a < 4; a++) {
            double t = 0.2 + 0.23 * a + 0.9 * j, p[3];
            f(t, p);
            if (a == liar) {
                p[0] += 1200.0;
                p[1] -= 900.0;
            }
            CHECK(rt_kin_push(c, 0, a, (int64_t)(t * 1e6), (int64_t)(t * 1e6), p) == RT_OK);
        }
}

static void test_kinematics(void)
{
    static rt_context_t c;
    double p[3], q[3], m[3], rc;
    int i;
    int64_t T = 4000000;

    rt_init(&c, NULL);
    feed(&c, line, -1);
    CHECK(rt_kin_predict(&c, 0, T, -1, p) == RT_OK);
    line(4.0, q);
    CHECK(fabs(p[0] - q[0]) < 1e-6 && fabs(p[1] - q[1]) < 1e-6 && fabs(p[2] - q[2]) < 1e-6);

    rt_init(&c, NULL);
    feed(&c, parab, -1);
    CHECK(rt_kin_predict(&c, 0, T, -1, p) == RT_OK);
    parab(4.0, q);
    CHECK(fabs(p[0] - q[0]) < 1e-6 && fabs(p[1] - q[1]) < 1e-6 && fabs(p[2] - q[2]) < 1e-6);

    /* лжец в окне не сбивает прогноз */
    rt_init(&c, NULL);
    feed(&c, line, 0);
    CHECK(rt_kin_predict(&c, 0, T, -1, p) == RT_OK);
    line(4.0, q);
    CHECK(fabs(p[0] - q[0]) < 1e-6 && fabs(p[1] - q[1]) < 1e-6);

    /* исключённый автор и мало авторов */
    CHECK(rt_kin_predict(&c, 0, T, 1, p) == RT_OK);
    rt_init(&c, NULL);
    for (i = 0; i < 6; i++) {
        line(0.5 * i, p);
        rt_kin_push(&c, 0, 2, (int64_t)(0.5e6 * i) + 1, (int64_t)(0.5e6 * i) + 1, p);
    }
    CHECK(rt_kin_predict(&c, 0, T, -1, p) == RT_NO_PRED);    /* один автор */
    CHECK(rt_kin_predict(&c, 1, T, -1, p) == RT_NO_PRED);    /* пустая цель */

    /* шлюз */
    rt_init(&c, NULL);
    feed(&c, line, -1);
    line(4.0, m);
    q[0] = m[0] + 1100.0; q[1] = m[1] - 900.0; q[2] = m[2];  /* T8 */
    CHECK(rt_kin_gate(&c, 0, 3, T, T + 2000, q, m, &rc) == RT_REFUTE && rc > 800.0);
    q[0] = m[0] + 30.0; q[1] = m[1] - 20.0; q[2] = m[2];     /* честная */
    CHECK(rt_kin_gate(&c, 0, 3, T, T + 2000, q, m, &rc) == RT_PASS && rc < 0.0);
    {   /* мой грубый сбой: далёк и от заявки, и от прогноза */
        double bad[3];
        bad[0] = m[0] + 4000.0; bad[1] = m[1] + 3500.0; bad[2] = m[2];
        q[0] = m[0] + 30.0; q[1] = m[1]; q[2] = m[2];
        CHECK(rt_kin_gate(&c, 0, 3, T, T + 2000, q, bad, NULL) == RT_FAILSAFE);
    }
    q[0] = m[0] + 1100.0; q[1] = m[1] - 900.0; q[2] = m[2];
    CHECK(rt_kin_gate(&c, 0, 3, T, T - 60000, q, m, NULL) == RT_FAILSAFE);   /* метка из будущего */
    CHECK(rt_kin_gate(&c, 0, 3, T, T + 900000, q, m, NULL) == RT_REFUTE);    /* поздняя доставка */
    CHECK(rt_kin_gate(&c, 5, 3, T, T, q, m, NULL) == RT_FAILSAFE);           /* нет прогноза */
    CHECK(c.n_refute == 2 && c.n_pass == 1 && c.n_failsafe == 3);

    /* время и кольцо */
    rt_init(&c, NULL);
    line(1.0, p);
    CHECK(rt_kin_push(&c, 0, 1, 1000000, 1000000, p) == RT_OK);
    CHECK(rt_kin_push(&c, 0, 1, 1000000, 1000000, p) == RT_ERR_TIME);       /* не монотонно */
    CHECK(rt_kin_push(&c, 0, 1, 1100000, 1000000, p) == RT_ERR_TIME);       /* из будущего */
    CHECK(rt_kin_push(&c, 0, 2, 1000000, 1000000, p) == RT_OK);             /* другой автор */
    CHECK(rt_kin_push(&c, RT_MAX_TARGETS, 1, 2000000, 2000000, p) == RT_ERR_ARG);
    CHECK(rt_kin_push(&c, 0, RT_MAX_AUTHORS, 2000000, 2000000, p) == RT_ERR_ARG);
    for (i = 0; i < RT_RING + 5; i++)
        rt_kin_push(&c, 0, 3, 2000000 + i, 2000000 + i, p);
    CHECK(c.ring[0].count == RT_RING);
    CHECK(c.ring[0].obs[(c.ring[0].head + RT_RING - 1) % RT_RING].t_us == 2000000 + RT_RING + 4);
    CHECK(rt_ctx_size() == sizeof(rt_context_t));

    /* карантин: лжец забыт - окно и суммы как будто его не было */
    {
        static rt_context_t clean;
        double pc[3];
        rt_init(&c, NULL);
        feed(&c, line, 0);                            /* автор 0 лжёт */
        rt_init(&clean, NULL);
        feed(&clean, line, -1);
        CHECK(rt_kin_forget_author(&c, 0, 0) == 4);
        CHECK(c.ring[0].count == 12 && rt_kin_sums_check(&c, 0) < 1e-12);
        CHECK(rt_kin_predict(&c, 0, T, -1, p) == RT_OK);
        CHECK(rt_kin_forget_author(&clean, 0, 0) == 4);
        CHECK(rt_kin_predict(&clean, 0, T, -1, pc) == RT_OK);
        CHECK(fabs(p[0] - pc[0]) < 1e-9 && fabs(p[1] - pc[1]) < 1e-9);
        line(1.0, q);
        CHECK(rt_kin_push(&c, 0, 1, 9000000, 9000000, q) == RT_OK);   /* запись после удаления */
        CHECK(c.ring[0].count == 13 && rt_kin_sums_check(&c, 0) < 1e-12);
        CHECK(rt_kin_forget_author(&c, 0, RT_MAX_AUTHORS) == RT_ERR_ARG);
    }
}

/* Дрейф инкрементальных сумм: 10^5 записей подряд, 4 автора, координаты
 * ~1e6-1e7, 8 отсчётов/с (кольцо 32 = 4 с); на каждой 1000-й - суммы против
 * прямого пересчёта и прогноз против контекста, собранного заново из кольца. */
static void ring_traj(double t, double p[3])
{
    p[0] = 5e6 + 400.0 * t;
    p[1] = 3e6 - 250.0 * t + 1000.0 * sin(0.3 * t);
    p[2] = 1e4 + 50.0 * sin(0.7 * t);
}

static void test_incremental_drift(void)
{
    static rt_context_t c, fresh;
    double worst_sum = 0.0, worst_pred = 0.0;
    long tick;
    int i, ok_pred = 1, n_cmp = 0;
    rt_init(&c, NULL);
    for (tick = 1; tick <= 100000; tick++) {
        int64_t t = tick * 125000;
        double p[3];
        ring_traj((double)t / 1e6, p);
        if (rt_kin_push(&c, 0, (int)(tick % 4), t, t, p) != RT_OK)
            ok_pred = 0;
        if (tick % 1000 == 0) {
            double e = rt_kin_sums_check(&c, 0), a[3], b[3];
            const rt_ring_t *r = &c.ring[0];
            if (e > worst_sum)
                worst_sum = e;
            rt_init(&fresh, NULL);
            for (i = 0; i < r->count; i++) {        /* по времени: от старейшего */
                const rt_obs_t *o = &r->obs[(r->head + i) % RT_RING];
                rt_kin_push(&fresh, 0, o->author, o->t_us, o->t_us, o->p);
            }
            if (rt_kin_predict(&c, 0, t + 60000, -1, a) == RT_OK
                    && rt_kin_predict(&fresh, 0, t + 60000, -1, b) == RT_OK) {
                int ax;
                n_cmp++;
                for (ax = 0; ax < 3; ax++) {
                    double d = fabs(a[ax] - b[ax]) / (fabs(b[ax]) + 1.0);
                    if (d > worst_pred)
                        worst_pred = d;
                }
            } else
                ok_pred = 0;
        }
    }
    printf("дрейф за 10^5 записей: суммы %.2e, прогноз %.2e (отн.), перебазирований %u\n",
           worst_sum, worst_pred, c.ring[0].n_rebase);
    CHECK(ok_pred && n_cmp == 100);
    CHECK(worst_sum < 1e-9);
    CHECK(worst_pred < 1e-9);
    CHECK(c.ring[0].n_rebase > 0);
}

/* Взаимная корреляция: сигнал с известной задержкой (целой и дробной),
 * шум, сигнал без возбуждения. */
static double sig(double t)
{
    return sin(1.3 * t) + 0.6 * sin(3.7 * t + 0.4) + 0.3 * sin(9.1 * t + 1.1);
}

static void test_xcorr(void)
{
    static double a[400], b[400];
    double lag, peak, shifts[4] = {0.0, 4.0, -7.0, 2.5};
    int i, s;
    for (s = 0; s < 4; s++) {
        for (i = 0; i < 400; i++) {                   /* 200 Гц, b отстаёт на shifts[s] */
            a[i] = sig(i / 200.0);
            b[i] = sig((i - shifts[s]) / 200.0);
        }
        CHECK(rt_xcorr_delay(a, b, 400, 20, 1e-6, &lag, &peak) == RT_OK);
        CHECK(fabs(lag - shifts[s]) < 0.05 && peak > 0.999);
        for (i = 0; i < 400; i++)
            b[i] += 0.1 * urand();                    /* шум 10 % */
        CHECK(rt_xcorr_delay(a, b, 400, 20, 1e-6, &lag, &peak) == RT_OK);
        CHECK(fabs(lag - shifts[s]) < 0.5);
    }
    for (i = 0; i < 400; i++)
        b[i] = 0.3 + 1e-5 * urand();                  /* висение: почти константа */
    CHECK(rt_xcorr_delay(a, b, 400, 20, 1e-4, &lag, &peak) == RT_ERR_LOW_EXCITATION);
    CHECK(rt_xcorr_delay(a, a, 400, 101, 1e-6, &lag, &peak) == RT_ERR_ARG);   /* > n/4 */
    CHECK(rt_xcorr_delay(a, a, 400, 0, 1e-6, &lag, &peak) == RT_ERR_ARG);
    {   /* 3 канала: один постоянный (вращение вокруг оси без изменения), два несут сдвиг 6 */
        static double A3[3 * 400], B3[3 * 400];
        for (i = 0; i < 400; i++) {
            A3[i] = 1.57;
            B3[i] = 1.57;
            A3[400 + i] = sig(i / 200.0);
            B3[400 + i] = sig((i - 6.0) / 200.0);
            A3[800 + i] = sig(0.7 * i / 200.0 + 2.0);
            B3[800 + i] = sig(0.7 * (i - 6.0) / 200.0 + 2.0);
        }
        CHECK(rt_xcorr_delay_multi(A3, B3, 400, 3, 20, 1e-6, &lag, &peak) == RT_OK);
        CHECK(fabs(lag - 6.0) < 0.05 && peak > 0.999);
        CHECK(rt_xcorr_delay(A3, B3, 400, 20, 1e-6, &lag, &peak) == RT_ERR_LOW_EXCITATION);
    }
}

static void test_sqrt_newton(void)
{
    double worst = 0.0, x = 1e-300;
    while (x < 1e300) {
        double e = fabs(rt_sqrt_newton(x) - sqrt(x)) / sqrt(x);
        if (e > worst)
            worst = e;
        x *= 1.37;
    }
    CHECK(worst < 4.5e-16);                          /* ~2 ulp */
    CHECK(rt_sqrt_newton(0.0) == 0.0 && rt_sqrt_newton(-4.0) == 0.0);
    CHECK(rt_sqrt_newton(4.0) == 2.0);
}

int main(void)
{
    test_grassmann();
    test_kinematics();
    test_incremental_drift();
    test_xcorr();
    test_sqrt_newton();
    printf("rt_core: %d passed, %d failed\n", n_ok, n_fail);
    return n_fail ? 1 : 0;
}
