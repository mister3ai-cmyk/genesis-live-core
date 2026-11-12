/* Business Source License 1.1 - см. research/LICENSE */

#include "rt_grassmann.h"
#include "rt_port.h"

static double dot(int n, const double *a, const double *b)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

int rt_gr_orth_mgs(int n, int k, const double *A, double *Q)
{
    int i, j, pass;
    if (n <= 0 || k <= 0 || k > n || n > RT_GR_MAX_N || !A || !Q)
        return RT_ERR_ARG;
    for (j = 0; j < k; j++) {
        double *q = Q + (size_t)j * n;
        double a_norm, nrm;
        if (q != A + (size_t)j * n)
            for (i = 0; i < n; i++)
                q[i] = A[(size_t)j * n + i];
        a_norm = RT_SQRT(dot(n, q, q));
        if (a_norm == 0.0)
            return RT_ERR_RANK;
        /* два прохода: второй снимает потерю ортогональности MGS */
        for (pass = 0; pass < 2; pass++) {
            int p;
            for (p = 0; p < j; p++) {
                const double *qp = Q + (size_t)p * n;
                double r = dot(n, qp, q);
                for (i = 0; i < n; i++)
                    q[i] -= r * qp[i];
            }
        }
        nrm = RT_SQRT(dot(n, q, q));
        if (nrm <= 1e-12 * a_norm)
            return RT_ERR_RANK;
        for (i = 0; i < n; i++)
            q[i] /= nrm;
    }
    return RT_OK;
}

double rt_gr_chordal2(int n, int k, const double *U, const double *V)
{
    double s = 0.0;
    int a, b;
    if (k == RT_GR_K) {
        /* M = U^T V построчно: 16 независимых накопителей вместо 16
           цепочек зависимых сложений - упор в пропускную способность, а не
           в задержку сложения */
        double M[RT_GR_K][RT_GR_K];
        int i;
        rt_zero(M, sizeof(M));
        for (i = 0; i < n; i++) {
            double u0 = U[i], u1 = U[n + i], u2 = U[2 * n + i], u3 = U[3 * n + i];
            for (b = 0; b < RT_GR_K; b++) {
                double v = V[(size_t)b * n + i];
                M[0][b] += u0 * v;
                M[1][b] += u1 * v;
                M[2][b] += u2 * v;
                M[3][b] += u3 * v;
            }
        }
        for (a = 0; a < RT_GR_K; a++)
            for (b = 0; b < RT_GR_K; b++)
                s += M[a][b] * M[a][b];
        return (double)k - s;
    }
    for (a = 0; a < k; a++)
        for (b = 0; b < k; b++) {
            double d = dot(n, U + (size_t)a * n, V + (size_t)b * n);
            s += d * d;
        }
    return (double)k - s;
}
