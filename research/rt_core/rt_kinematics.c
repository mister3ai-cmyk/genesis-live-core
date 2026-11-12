/* Business Source License 1.1 - см. research/LICENSE */
#include <math.h>
#include <string.h>

#include "rt_kinematics.h"

#define RT_T_NONE INT64_MIN

size_t rt_ctx_size(void)
{
    return sizeof(rt_context_t);
}

void rt_cfg_default(rt_cfg_t *cfg)
{
    cfg->horizon_us = 4000000;
    cfg->max_skew_us = 50000;
    cfg->drop_tol = 400.0;
    cfg->thr = 800.0;
    cfg->pos_tol = 300.0;
    cfg->min_pts = 4;
}

int rt_init(rt_context_t *ctx, const rt_cfg_t *cfg)
{
    int t, a;
    if (!ctx)
        return RT_ERR_ARG;
    memset(ctx, 0, sizeof(*ctx));
    if (cfg)
        ctx->cfg = *cfg;
    else
        rt_cfg_default(&ctx->cfg);
    for (t = 0; t < RT_MAX_TARGETS; t++)
        for (a = 0; a < RT_MAX_AUTHORS; a++)
            ctx->ring[t].last_t[a] = RT_T_NONE;
    return RT_OK;
}

int rt_kin_push(rt_context_t *ctx, int target, int author,
                int64_t t_us, int64_t now_us, const double p[3])
{
    rt_ring_t *r;
    rt_obs_t *o;
    if (!ctx || !p || target < 0 || target >= RT_MAX_TARGETS
            || author < 0 || author >= RT_MAX_AUTHORS)
        return RT_ERR_ARG;
    r = &ctx->ring[target];
    if (t_us <= r->last_t[author] || t_us > now_us + ctx->cfg.max_skew_us) {
        ctx->n_reject_time++;
        return RT_ERR_TIME;
    }
    r->last_t[author] = t_us;
    o = &r->obs[r->head];
    o->t_us = t_us;
    o->author = (uint8_t)author;
    o->p[0] = p[0];
    o->p[1] = p[1];
    o->p[2] = p[2];
    r->head = (r->head + 1) % RT_RING;
    if (r->count < RT_RING)
        r->count++;
    ctx->n_push++;
    return RT_OK;
}

/* ------------------------------------------------------------ подгонка */

typedef struct {
    double tau[RT_RING];       /* секунды до момента прогноза (<= 0) */
    double p[RT_RING][3];
    int    author[RT_RING];
    int    n;
} pts_t;

/* Решение m x m (m <= 3) с тремя правыми частями методом Гаусса с выбором
 * главного элемента. A и B портятся. 0 - вырождена. */
static int solve3(int m, double A[3][3], double B[3][3], double X[3][3])
{
    int c, r, k, piv;
    for (c = 0; c < m; c++) {
        piv = c;
        for (r = c + 1; r < m; r++)
            if (fabs(A[r][c]) > fabs(A[piv][c]))
                piv = r;
        if (fabs(A[piv][c]) < 1e-12)
            return 0;
        if (piv != c) {
            for (k = 0; k < m; k++) {
                double t = A[c][k]; A[c][k] = A[piv][k]; A[piv][k] = t;
            }
            for (k = 0; k < 3; k++) {
                double t = B[c][k]; B[c][k] = B[piv][k]; B[piv][k] = t;
            }
        }
        for (r = 0; r < m; r++) {
            double f;
            if (r == c)
                continue;
            f = A[r][c] / A[c][c];
            for (k = c; k < m; k++)
                A[r][k] -= f * A[c][k];
            for (k = 0; k < 3; k++)
                B[r][k] -= f * B[c][k];
        }
    }
    for (r = 0; r < m; r++)
        for (k = 0; k < 3; k++)
            X[r][k] = B[r][k] / A[r][r];
    return 1;
}

/* Суммы нормальных уравнений многочлена 2-й степени: A[a][b] = sum tau^(a+b),
 * B[a][ax] = sum tau^a p_ax. Для 1-й степени берётся левый верхний угол 2 x 2. */
typedef struct {
    double A[3][3], B[3][3];
    int    n;
} sums_t;

static void sums_add(sums_t *S, double tau, const double *p, double sign)
{
    double pw[3];
    int a, b;
    pw[0] = 1.0;
    pw[1] = tau;
    pw[2] = tau * tau;
    for (a = 0; a < 3; a++) {
        for (b = 0; b < 3; b++)
            S->A[a][b] += sign * pw[a] * pw[b];
        for (b = 0; b < 3; b++)
            S->B[a][b] += sign * pw[a] * p[b];
    }
    S->n += sign > 0 ? 1 : -1;
}

/* Подгонка по готовым суммам; степень по числу точек. 0 - не удалось. */
static int fit_sums(const sums_t *S, double coef[3][3], int *deg)
{
    double A[3][3], B[3][3];
    int m;
    *deg = S->n >= 6 ? 2 : 1;
    m = *deg + 1;
    memcpy(A, S->A, sizeof(A));
    memcpy(B, S->B, sizeof(B));
    memset(coef, 0, sizeof(double) * 9);
    return solve3(m, A, B, coef);
}

/* Квадрат невязки точки против подгонки. */
static double resid2(double coef[3][3], int deg, double tau, const double *p)
{
    double s = 0.0;
    int ax;
    for (ax = 0; ax < 3; ax++) {
        double v = coef[0][ax] + coef[1][ax] * tau
                   + (deg == 2 ? coef[2][ax] * tau * tau : 0.0);
        s += (p[ax] - v) * (p[ax] - v);
    }
    return s;
}

static int usable(const pts_t *P, const int *use, int min_pts)
{
    int i, n = 0, first = -1, two = 0;
    for (i = 0; i < P->n; i++) {
        if (!use[i])
            continue;
        n++;
        if (first < 0)
            first = P->author[i];
        else if (P->author[i] != first)
            two = 1;
    }
    return n >= min_pts && two;
}

int rt_kin_predict(const rt_context_t *ctx, int target, int64_t t_us,
                   int exclude_author, double out[3])
{
    const rt_ring_t *r;
    pts_t P;
    int use[RT_RING], best[RT_RING], cand[RT_RING];
    int authors[RT_MAX_AUTHORS], n_auth = 0;
    int i, s, best_n = -1, deg;
    double coef[3][3], tol2;
    sums_t total, by_auth[RT_MAX_AUTHORS], S;
    if (!ctx || !out || target < 0 || target >= RT_MAX_TARGETS)
        return RT_ERR_ARG;
    r = &ctx->ring[target];
    P.n = 0;
    for (i = 0; i < r->count; i++) {
        const rt_obs_t *o = &r->obs[i];
        if (o->author == exclude_author || o->t_us >= t_us
                || o->t_us < t_us - ctx->cfg.horizon_us)
            continue;
        P.tau[P.n] = (double)(o->t_us - t_us) / 1e6;
        P.p[P.n][0] = o->p[0];
        P.p[P.n][1] = o->p[1];
        P.p[P.n][2] = o->p[2];
        P.author[P.n] = o->author;
        P.n++;
    }
    for (i = 0; i < P.n; i++)
        use[i] = 1;
    if (!usable(&P, use, ctx->cfg.min_pts))
        return RT_NO_PRED;
    for (i = 0; i < P.n; i++) {
        int a = P.author[i], k, seen = 0;
        for (k = 0; k < n_auth; k++)
            seen |= authors[k] == a;
        if (!seen)
            authors[n_auth++] = a;
    }
    /* суммы нормальных уравнений: по каждому автору один раз; кандидат
       «без автора a» = итог - суммы a (O(авторов), а не O(точек)) */
    memset(&total, 0, sizeof(total));
    memset(by_auth, 0, sizeof(by_auth));
    for (i = 0; i < P.n; i++) {
        int k = 0;
        while (authors[k] != P.author[i])
            k++;
        sums_add(&by_auth[k], P.tau[i], P.p[i], 1.0);
    }
    for (s = 0; s < n_auth; s++) {
        int a, b;
        for (a = 0; a < 3; a++)
            for (b = 0; b < 3; b++) {
                total.A[a][b] += by_auth[s].A[a][b];
                total.B[a][b] += by_auth[s].B[a][b];
            }
        total.n += by_auth[s].n;
    }
    tol2 = ctx->cfg.drop_tol * ctx->cfg.drop_tol;
    /* кандидаты: s = -1 - все авторы; s >= 0 - все, кроме authors[s] */
    for (s = -1; s < (n_auth > 2 ? n_auth : 0); s++) {
        int n_in = 0;
        S = total;
        if (s >= 0) {
            int a, b;
            for (a = 0; a < 3; a++)
                for (b = 0; b < 3; b++) {
                    S.A[a][b] -= by_auth[s].A[a][b];
                    S.B[a][b] -= by_auth[s].B[a][b];
                }
            S.n -= by_auth[s].n;
        }
        for (i = 0; i < P.n; i++)
            cand[i] = s < 0 || P.author[i] != authors[s];
        if (!usable(&P, cand, ctx->cfg.min_pts) || !fit_sums(&S, coef, &deg))
            continue;
        for (i = 0; i < P.n; i++) {
            use[i] = resid2(coef, deg, P.tau[i], P.p[i]) <= tol2;
            n_in += use[i];
        }
        if (n_in > best_n) {
            best_n = n_in;
            memcpy(best, use, sizeof(int) * (size_t)P.n);
        }
        if (best_n == P.n)      /* все точки согласны: больше не набрать, итог тот же */
            break;
    }
    if (best_n < 0 || !usable(&P, best, ctx->cfg.min_pts))
        return RT_NO_PRED;
    memset(&S, 0, sizeof(S));
    for (i = 0; i < P.n; i++)
        if (best[i])
            sums_add(&S, P.tau[i], P.p[i], 1.0);
    if (!fit_sums(&S, coef, &deg))
        return RT_NO_PRED;
    out[0] = coef[0][0];
    out[1] = coef[0][1];
    out[2] = coef[0][2];
    return RT_OK;
}

static double dist3(const double *a, const double *b)
{
    return sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
                + (a[2] - b[2]) * (a[2] - b[2]));
}

int rt_kin_gate(rt_context_t *ctx, int target, int author,
                int64_t t_claim_us, int64_t now_us,
                const double claim[3], const double mine[3], double *res_claim)
{
    double pred[3], dmax = 0.0, rc;
    int ax, v;
    int64_t skew;
    if (res_claim)
        *res_claim = -1.0;
    if (!ctx || !claim || !mine)
        return RT_ERR_ARG;
    for (ax = 0; ax < 3; ax++) {
        double d = fabs(claim[ax] - mine[ax]);
        if (d > dmax)
            dmax = d;
    }
    if (dmax <= ctx->cfg.pos_tol) {             /* быстрый путь: спора нет */
        ctx->n_pass++;
        return RT_PASS;
    }
    /* метка из будущего - часы автора убежали; задержка доставки (метка в
       прошлом) - нормальна и шлюз не выключает */
    skew = t_claim_us - now_us;
    if (skew > ctx->cfg.max_skew_us
            || rt_kin_predict(ctx, target, t_claim_us, author, pred) != RT_OK
            || dist3(mine, pred) > ctx->cfg.thr) {
        ctx->n_failsafe++;
        return RT_FAILSAFE;
    }
    rc = dist3(claim, pred);
    if (res_claim)
        *res_claim = rc;
    v = rc > ctx->cfg.thr ? RT_REFUTE : RT_PASS;
    if (v == RT_REFUTE)
        ctx->n_refute++;
    else
        ctx->n_pass++;
    return v;
}
