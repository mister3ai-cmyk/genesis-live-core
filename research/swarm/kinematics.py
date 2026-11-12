# Business Source License 1.1 - см. research/LICENSE
"""v4.0: кинематический предиктор для шлюза --traj-gate.

Положение цели в момент t предсказывается по заявкам других авторов за
последние HORIZON_MS: подгонка x(tau), y(tau) многочленом (квадратичным при
>= 6 точках, иначе линейным), tau - секунды до t. Устойчивость к лжецу в
окне: после каждой подгонки отбрасываются точки с невязкой > DROP_TOL и
подгонка повторяется. Нужно >= MIN_PTS точек от >= 2 авторов, иначе None.

Почему не подпись Ганкеля: на том же окне линейная невязка разделяет честную
заявку и ложь T8 с 1.9 % ложных тревог, подпись Ганкеля - с 31.7 %
(SWARM_FINDINGS, Часть XI, этап B).
"""
import math

HORIZON_MS = 4000
MIN_PTS = 4
DROP_TOL = 400


def _fit(pts, deg):
    """Наименьшие квадраты по нормальным уравнениям; pts - (tau, x, y)."""
    n = deg + 1
    A = [[0.0] * n for _ in range(n)]
    bx, by = [0.0] * n, [0.0] * n
    for tau, x, y in pts:
        p = [tau ** k for k in range(n)]
        for i in range(n):
            bx[i] += p[i] * x
            by[i] += p[i] * y
            for j in range(n):
                A[i][j] += p[i] * p[j]
    return _solve(A, bx), _solve(A, by)


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _at(coef, tau):
    return sum(c * tau ** k for k, c in enumerate(coef))


def _usable(pts):
    return len(pts) >= MIN_PTS and len({p[3] for p in pts}) >= 2


def _fit_pts(pts):
    cx, cy = _fit([(p[0], p[1], p[2]) for p in pts], 2 if len(pts) >= 6 else 1)
    return None if cx is None or cy is None else (cx, cy)


def _inliers(fit, pts):
    cx, cy = fit
    return [p for p in pts
            if math.hypot(p[1] - _at(cx, p[0]), p[2] - _at(cy, p[0])) <= DROP_TOL]


def predict(claims, t_ms, horizon_ms=HORIZON_MS):
    """claims - [(t_ms, author, (x, y))] уже без проверяемого автора и
    исключённых узлов. Возвращает (x, y) в момент t_ms или None.

    Консенсус в духе RANSAC с допуском одного лжеца: кандидаты - подгонка по
    всем авторам и по всем без одного; выигрывает кандидат, с которым согласно
    (невязка <= DROP_TOL) больше всего точек; итог - подгонка по его точкам
    согласия."""
    pts = [((tc - t_ms) / 1000.0, float(v[0]), float(v[1]), a)
           for tc, a, v in claims if t_ms - horizon_ms <= tc < t_ms]
    if not _usable(pts):
        return None
    authors = sorted({p[3] for p in pts})
    subsets = [pts] + ([[p for p in pts if p[3] != a] for a in authors] if len(authors) > 2 else [])
    best = None
    for sub in subsets:
        if not _usable(sub):
            continue
        fit = _fit_pts(sub)
        if fit is None:
            continue
        inl = _inliers(fit, pts)
        if best is None or len(inl) > len(best):
            best = inl
    if best is None or not _usable(best):
        return None
    fit = _fit_pts(best)
    if fit is None:
        return None
    return _at(fit[0], 0.0), _at(fit[1], 0.0)


def residual(value, pred):
    return math.hypot(value[0] - pred[0], value[1] - pred[1])
