/* Business Source License 1.1 - см. research/LICENSE */
#include "rt_xcorr.h"
#include "rt_port.h"

static double variance(const double *x, int n)
{
    double m = 0.0, v = 0.0;
    int i;
    for (i = 0; i < n; i++)
        m += x[i];
    m /= n;
    for (i = 0; i < n; i++)
        v += (x[i] - m) * (x[i] - m);
    return v / n;
}

/* Многоканальный Пирсон для сдвига k: ковариации и дисперсии каналов по
 * перекрытию (a_c[i], b_c[i + k]) суммируются, затем нормируются. */
static double pearson(const double *a, const double *b, int n, int ch, int k)
{
    int lo = k < 0 ? -k : 0, hi = k > 0 ? n - k : n, m = hi - lo, c, i;
    double ca = 0, cb = 0, cab = 0;
    for (c = 0; c < ch; c++) {
        const double *x = a + (size_t)c * n, *y = b + (size_t)c * n;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (i = lo; i < hi; i++) {
            double u = x[i], v = y[i + k];
            sa += u;
            sb += v;
            saa += u * u;
            sbb += v * v;
            sab += u * v;
        }
        ca += saa - sa * sa / m;
        cb += sbb - sb * sb / m;
        cab += sab - sa * sb / m;
    }
    if (ca <= 0.0 || cb <= 0.0)
        return 0.0;
    return cab / RT_SQRT(ca * cb);
}

int rt_xcorr_delay_multi(const double *a, const double *b, int n, int channels, int max_lag,
                         double min_var, double *lag, double *peak)
{
    double best = -2.0, va = 0.0, vb = 0.0, rm, rp, den;
    int k, kb = 0, c;
    if (!a || !b || !lag || !peak || n < 8 || channels < 1 || max_lag < 1 || max_lag > n / 4)
        return RT_ERR_ARG;
    for (c = 0; c < channels; c++) {
        va += variance(a + (size_t)c * n, n);
        vb += variance(b + (size_t)c * n, n);
    }
    if (va < min_var || vb < min_var)
        return RT_ERR_LOW_EXCITATION;
    for (k = -max_lag; k <= max_lag; k++) {
        double r = pearson(a, b, n, channels, k);
        if (r > best) {
            best = r;
            kb = k;
        }
    }
    *lag = (double)kb;
    *peak = best;
    if (kb > -max_lag && kb < max_lag) {         /* парабола по трём точкам */
        rm = pearson(a, b, n, channels, kb - 1);
        rp = pearson(a, b, n, channels, kb + 1);
        den = rm - 2.0 * best + rp;
        if (den < 0.0) {
            double d = 0.5 * (rm - rp) / den;
            if (d > -1.0 && d < 1.0)
                *lag = kb + d;
        }
    }
    return RT_OK;
}

int rt_xcorr_delay(const double *a, const double *b, int n, int max_lag,
                   double min_var, double *lag, double *peak)
{
    return rt_xcorr_delay_multi(a, b, n, 1, max_lag, min_var, lag, peak);
}
