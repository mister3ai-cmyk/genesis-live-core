/* Business Source License 1.1 - см. research/LICENSE */

#include "rt_kinematics.h"
#include "rt_port.h"

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
    rt_zero(ctx, sizeof(*ctx));
    if (cfg)
        rt_copy(&ctx->cfg, cfg, sizeof(ctx->cfg));
    else
        rt_cfg_default(&ctx->cfg);
    for (t = 0; t < RT_MAX_TARGETS; t++)
        for (a = 0; a < RT_MAX_AUTHORS; a++)
            ctx->ring[t].last_t[a] = RT_T_NONE;
    return RT_OK;
}

/* ------------------------------------------------- суммы в локальном времени */

static void raw_add(rt_sums_t *S, double s, const double *p, double sign)
{
    double s2 = s * s;
    int ax;
    S->S[0] += sign;
    S->S[1] += sign * s;
    S->S[2] += sign * s2;
    S->S[3] += sign * s2 * s;
    S->S[4] += sign * s2 * s2;
    for (ax = 0; ax < 3; ax++) {
        S->B[0][ax] += sign * p[ax];
        S->B[1][ax] += sign * s * p[ax];
        S->B[2][ax] += sign * s2 * p[ax];
    }
    S->n += sign > 0 ? 1 : -1;
}

static double loc_s(const rt_ring_t *r, int64_t t_us)
{
    return (double)(t_us - r->t0_us) / 1e6;
}

/* Перебазирование: t0 = t_us, суммы всех авторов заново из кольца. */
static void rebase(rt_ring_t *r, int64_t t_us)
{
    int i;
    r->t0_us = t_us;
    rt_zero(r->sums, sizeof(r->sums));
    for (i = 0; i < r->count; i++)
        raw_add(&r->sums[r->obs[i].author], loc_s(r, r->obs[i].t_us), r->obs[i].p, 1.0);
    r->n_evict = 0;
    r->n_rebase++;
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
    if (r->count == 0)
        r->t0_us = t_us;
    o = &r->obs[r->head];
    if (r->count == RT_RING) {                   /* вытесняем старейший */
        raw_add(&r->sums[o->author], loc_s(r, o->t_us), o->p, -1.0);
        r->n_evict++;
    }
    o->t_us = t_us;
    o->author = (uint8_t)author;
    o->p[0] = p[0];
    o->p[1] = p[1];
    o->p[2] = p[2];
    r->head = (r->head + 1) % RT_RING;
    if (r->count < RT_RING)
        r->count++;
    if (RT_FABS(loc_s(r, t_us)) > RT_REBASE_S || r->n_evict >= RT_REBASE_EVICT)
        rebase(r, t_us);
    else
        raw_add(&r->sums[author], loc_s(r, t_us), p, 1.0);
    ctx->n_push++;
    return RT_OK;
}

int rt_kin_forget_author(rt_context_t *ctx, int target, int author)
{
    rt_ring_t *r;
    rt_obs_t tmp[RT_RING];
    int i, n = 0, removed;
    if (!ctx || target < 0 || target >= RT_MAX_TARGETS || author < 0 || author >= RT_MAX_AUTHORS)
        return RT_ERR_ARG;
    r = &ctx->ring[target];
    for (i = 0; i < r->count; i++) {             /* от старейшего к новейшему */
        const rt_obs_t *o = &r->obs[(r->head - r->count + i + RT_RING) % RT_RING];
        if (o->author != author)
            rt_copy(&tmp[n++], o, sizeof(*o));
    }
    removed = r->count - n;
    for (i = 0; i < n; i++)
        rt_copy(&r->obs[i], &tmp[i], sizeof(tmp[i]));
    r->count = n;
    r->head = n % RT_RING;
    rt_zero(&r->sums[author], sizeof(r->sums[author]));
    return removed;
}

double rt_kin_sums_check(const rt_context_t *ctx, int target)
{
    static const double one = 1.0;
    const rt_ring_t *r;
    rt_sums_t D[RT_MAX_AUTHORS];
    double worst = 0.0;
    int i, a, k, ax;
    if (!ctx || target < 0 || target >= RT_MAX_TARGETS)
        return -1.0;
    r = &ctx->ring[target];
    rt_zero(D, sizeof(D));
    for (i = 0; i < r->count; i++)
        raw_add(&D[r->obs[i].author], loc_s(r, r->obs[i].t_us), r->obs[i].p, one);
    for (a = 0; a < RT_MAX_AUTHORS; a++) {
        for (k = 0; k < 5; k++) {
            double e = RT_FABS(r->sums[a].S[k] - D[a].S[k]) / (RT_FABS(D[a].S[k]) + 1.0);
            if (e > worst)
                worst = e;
        }
        for (k = 0; k < 3; k++)
            for (ax = 0; ax < 3; ax++) {
                double e = RT_FABS(r->sums[a].B[k][ax] - D[a].B[k][ax]) / (RT_FABS(D[a].B[k][ax]) + 1.0);
                if (e > worst)
                    worst = e;
            }
        if (r->sums[a].n != D[a].n)
            return 1e300;
    }
    return worst;
}

/* ------------------------------------------------------------ подгонка */

/* Точки окна не копируются: они читаются прямо из кольца, набор задаётся
 * битовой маской по индексам кольца (S5.2: кадр стека rt_kin_predict был
 * 4.6 КБ из-за копии окна и сумм по авторам). Порядок обхода и формулы те
 * же, что у копии, поэтому результат совпадает бит в бит. */
#if RT_RING > 32
#error "маски окна - uint32_t: RT_RING <= 32"
#endif
typedef uint32_t rt_mask_t;
#define RT_BIT(i) ((rt_mask_t)1u << (i))

/* Решение m x m (m <= 3) с тремя правыми частями методом Гаусса с выбором
 * главного элемента. A и B портятся. 0 - вырождена. */
static int solve3(int m, double A[3][3], double B[3][3], double X[3][3])
{
    int c, r, k, piv;
    for (c = 0; c < m; c++) {
        piv = c;
        for (r = c + 1; r < m; r++)
            if (RT_FABS(A[r][c]) > RT_FABS(A[piv][c]))
                piv = r;
        if (RT_FABS(A[piv][c]) < 1e-12)
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

/* Суммы нормальных уравнений многочлена 2-й степени: T[k] = sum tau^k
 * (матрица A[a][b] = T[a + b] - ганкелева, хранится 5 чисел вместо 9),
 * B[a][ax] = sum tau^a p_ax. Для 1-й степени берётся левый верхний угол. */
typedef struct {
    double T[5], B[3][3];
    int    n;
} sums_t;

static void sums_add(sums_t *S, double tau, const double *p, double sign)
{
    double pw[3];
    int a, b;
    pw[0] = 1.0;
    pw[1] = tau;
    pw[2] = tau * tau;
    /* те же произведения, что у полной матрицы: (sign * pw[a]) * pw[b] */
    S->T[0] += sign * pw[0] * pw[0];
    S->T[1] += sign * pw[0] * pw[1];
    S->T[2] += sign * pw[0] * pw[2];
    S->T[3] += sign * pw[1] * pw[2];
    S->T[4] += sign * pw[2] * pw[2];
    for (a = 0; a < 3; a++)
        for (b = 0; b < 3; b++)
            S->B[a][b] += sign * pw[a] * p[b];
    S->n += sign > 0 ? 1 : -1;
}

static void sums_sub(sums_t *S, const sums_t *X)
{
    int k, ax;
    for (k = 0; k < 5; k++)
        S->T[k] -= X->T[k];
    for (k = 0; k < 3; k++)
        for (ax = 0; ax < 3; ax++)
            S->B[k][ax] -= X->B[k][ax];
    S->n -= X->n;
}

/* Степени (-s_q)^k, k = 0..4: общие для всех сдвигов одного прогноза. */
static void shift_pows(double s_q, double pw[5])
{
    int m;
    pw[0] = 1.0;
    for (m = 1; m < 5; m++)
        pw[m] = pw[m - 1] * -s_q;
}

/* Сдвиг сумм из локального времени кольца s к tau = s - s_q:
 * sum tau^m = sum_j C(m, j) S_j (-s_q)^(m-j); то же для B. pw - shift_pows. */
static void shift(const rt_sums_t *R, const double pw[5], sums_t *out)
{
    static const double C[5][5] = {{1, 0, 0, 0, 0}, {1, 1, 0, 0, 0}, {1, 2, 1, 0, 0},
                                   {1, 3, 3, 1, 0}, {1, 4, 6, 4, 1}};
    int m, j, ax;
    for (m = 0; m < 5; m++) {
        out->T[m] = 0.0;
        for (j = 0; j <= m; j++)
            out->T[m] += C[m][j] * R->S[j] * pw[m - j];
    }
    for (m = 0; m < 3; m++)
        for (ax = 0; ax < 3; ax++) {
            out->B[m][ax] = 0.0;
            for (j = 0; j <= m; j++)
                out->B[m][ax] += C[m][j] * R->B[j][ax] * pw[m - j];
        }
    out->n = R->n;
}

/* Суммы автора a в окне: веденные при записи минус его точки вне окна
 * (маска out - только точки автора a, обычно 0-2), со сдвигом к моменту
 * прогноза. Биты обходятся по возрастанию - порядок сложения как у копии. */
static void author_sums(const rt_ring_t *r, int a, rt_mask_t out, const double pw[5], sums_t *res)
{
    rt_sums_t w, x;
    int k, ax;
    if (!out) {                           /* частый случай: вычитать нечего */
        shift(&r->sums[a], pw, res);
        return;
    }
    rt_zero(&x, sizeof(x));
    for (; out; out &= out - 1) {
        const rt_obs_t *o = &r->obs[RT_CTZ32(out)];
        raw_add(&x, loc_s(r, o->t_us), o->p, 1.0);
    }
    for (k = 0; k < 5; k++)
        w.S[k] = r->sums[a].S[k] - x.S[k];
    for (k = 0; k < 3; k++)
        for (ax = 0; ax < 3; ax++)
            w.B[k][ax] = r->sums[a].B[k][ax] - x.B[k][ax];
    w.n = r->sums[a].n - x.n;
    shift(&w, pw, res);
}

/* Подгонка по готовым суммам; степень по числу точек. 0 - не удалось. */
static int fit_sums(const sums_t *S, double coef[3][3], int *deg)
{
    double A[3][3], B[3][3];
    int m, a, b;
    *deg = S->n >= 6 ? 2 : 1;
    m = *deg + 1;
    for (a = 0; a < 3; a++)
        for (b = 0; b < 3; b++)
            A[a][b] = S->T[a + b];
    rt_copy(B, S->B, sizeof(B));
    rt_zero(coef, sizeof(double) * 9);
    return solve3(m, A, B, coef);
}

static double tau_of(const rt_obs_t *o, int64_t t_us)
{
    return (double)(o->t_us - t_us) / 1e6;
}

/* Точка согласна с подгонкой: квадрат невязки <= tol2. У подгонки 1-й
 * степени coef[2] = 0 (fit_sums обнуляет), поэтому формула одна. Выход, как
 * только частичная сумма превысила порог, - ответ тот же. */
static int agrees(double coef[3][3], double tau, double tau2, const double *p, double tol2)
{
    double s = 0.0;
    int ax;
    for (ax = 0; ax < 3; ax++) {
        double d = p[ax] - (coef[0][ax] + coef[1][ax] * tau + coef[2][ax] * tau2);
        s += d * d;
        if (s > tol2)
            return 0;
    }
    return 1;
}

static int usable(const rt_ring_t *r, rt_mask_t use, int min_pts)
{
    int n = 0, first = -1, two = 0;
    for (; use; use &= use - 1) {
        int a = r->obs[RT_CTZ32(use)].author;
        n++;
        if (first < 0)
            first = a;
        else if (a != first)
            two = 1;
    }
    return n >= min_pts && two;
}

int rt_kin_predict(const rt_context_t *ctx, int target, int64_t t_us,
                   int exclude_author, double out[3])
{
    const rt_ring_t *r;
    rt_mask_t win = 0, outside[RT_MAX_AUTHORS], use, best = 0, m;
    int authors[RT_MAX_AUTHORS], n_auth = 0, n_pts = 0;
    int i, s, best_n = -1, best_exact = 0, best_s = -2, deg;
    double coef[3][3], tol2, pw[5];
    sums_t total, S, X;
    if (!ctx || !out || target < 0 || target >= RT_MAX_TARGETS)
        return RT_ERR_ARG;
    r = &ctx->ring[target];
    rt_zero(outside, sizeof(outside));
    for (i = 0; i < r->count; i++) {
        const rt_obs_t *o = &r->obs[i];
        if (o->author == exclude_author)
            continue;
        if (o->t_us >= t_us || o->t_us < t_us - ctx->cfg.horizon_us) {
            outside[o->author] |= RT_BIT(i);   /* вне окна: вычесть из сумм автора */
            continue;
        }
        win |= RT_BIT(i);
        n_pts++;
    }
    if (!usable(r, win, ctx->cfg.min_pts))
        return RT_NO_PRED;
    for (m = win; m; m &= m - 1) {
        int a = r->obs[RT_CTZ32(m)].author, k, seen = 0;
        for (k = 0; k < n_auth; k++)
            seen |= authors[k] == a;
        if (!seen)
            authors[n_auth++] = a;
    }
    /* итог = сумма сумм авторов в окне; кандидат «без автора a» = итог -
       суммы a: O(авторов), а не O(точек). Суммы автора пересчитываются из
       кольца, а не хранятся (стек) - те же операции, тот же результат */
    rt_zero(&total, sizeof(total));
    shift_pows(loc_s(r, t_us), pw);
    for (s = 0; s < n_auth; s++) {
        int k, ax;
        author_sums(r, authors[s], outside[authors[s]], pw, &X);
        for (k = 0; k < 5; k++)
            total.T[k] += X.T[k];
        for (k = 0; k < 3; k++)
            for (ax = 0; ax < 3; ax++)
                total.B[k][ax] += X.B[k][ax];
        total.n += X.n;
    }
    tol2 = ctx->cfg.drop_tol * ctx->cfg.drop_tol;
    /* кандидаты: s = -1 - все авторы; s >= 0 - все, кроме authors[s] */
    for (s = -1; s < (n_auth > 2 ? n_auth : 0); s++) {
        int n_in = 0, j = 0;
        rt_copy(&S, &total, sizeof(S));
        if (s >= 0) {
            author_sums(r, authors[s], outside[authors[s]], pw, &X);
            sums_sub(&S, &X);
        }
        /* годность кандидата по счётчикам: точек >= min_pts, авторов >= 2
           (без одного из > 2 авторов их остаётся >= 2) */
        if (S.n < ctx->cfg.min_pts || !fit_sums(&S, coef, &deg))
            continue;
        {
            int exact = 1, beaten = 0, skip = s < 0 ? -1 : authors[s];
            use = 0;
            for (m = win; m; m &= m - 1) {
                const rt_obs_t *o = &r->obs[RT_CTZ32(m)];
                double tau = tau_of(o, t_us);
                int ok = agrees(coef, tau, tau * tau, o->p, tol2);
                use |= ok ? (m & (~m + 1)) : 0;  /* младший бит m - эта точка */
                n_in += ok;
                exact &= ok == (o->author != skip);
                if (n_in + (n_pts - j - 1) <= best_n) {   /* лучшего уже не превзойти */
                    beaten = 1;
                    break;
                }
                j++;
            }
            if (!beaten && n_in > best_n) {
                best_n = n_in;
                best_exact = exact;
                best_s = s;                   /* суммы кандидата восстановимы по s */
                best = use;
            }
        }
        if (best_n == n_pts)    /* все точки согласны: больше не набрать, итог тот же */
            break;
    }
    if (best_n < 0 || !usable(r, best, ctx->cfg.min_pts))
        return RT_NO_PRED;
    if (!best_exact) {
        /* согласны ровно все точки, кроме точек одного автора? - тогда суммы
           готовы: итог минус этот автор (частый случай: лжец в окне) */
        int in_k[RT_MAX_AUTHORS], all_k[RT_MAX_AUTHORS], k, out_k = -1, ok = 1;
        rt_zero(in_k, sizeof(in_k));
        rt_zero(all_k, sizeof(all_k));
        for (m = win; m; m &= m - 1) {
            i = RT_CTZ32(m);
            for (k = 0; authors[k] != r->obs[i].author; k++)
                ;
            all_k[k]++;
            in_k[k] += (best & RT_BIT(i)) != 0;
        }
        for (k = 0; k < n_auth && ok; k++) {
            if (in_k[k] == 0 && out_k < 0)
                out_k = k;
            else if (in_k[k] != all_k[k])
                ok = 0;
        }
        if (ok && out_k >= 0) {
            best_s = out_k;
            best_exact = 1;
        }
    }
    if (best_exact) {
        /* суммы набора согласных точек: итог или итог минус один автор */
        rt_copy(&S, &total, sizeof(S));
        if (best_s >= 0) {
            author_sums(r, authors[best_s], outside[authors[best_s]], pw, &X);
            sums_sub(&S, &X);
        }
    } else {
        rt_zero(&S, sizeof(S));
        for (m = best; m; m &= m - 1) {
            const rt_obs_t *o = &r->obs[RT_CTZ32(m)];
            sums_add(&S, tau_of(o, t_us), o->p, 1.0);
        }
    }
    if (!fit_sums(&S, coef, &deg))
        return RT_NO_PRED;
    out[0] = coef[0][0];
    out[1] = coef[0][1];
    out[2] = coef[0][2];
    return RT_OK;
}

static double dist3(const double *a, const double *b)
{
    return RT_SQRT((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
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
        double d = RT_FABS(claim[ax] - mine[ax]);
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
