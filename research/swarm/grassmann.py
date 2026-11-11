# Business Source License 1.1 - см. research/LICENSE
"""v4.0, разведка: подпространства Gr(k, d) для состояния узла (k = 4, d = 128).

Изолированный модуль: симулятор его пока не вызывает. Здесь только геометрия
и честные замеры обратимости; встраивать ли его в протокол - решение по цифрам.

- Y (d x k) - ортонормированный базис, P = Y Y^T (d x d) - проектор; P по сети
  не передаётся, расстояния считаются через Y_a^T Y_b (k x k).
- d^2 = k - ||Y_a^T Y_b||_F^2 = sum sin^2(theta_i), 0 <= d^2 <= k.
- embed_point: точка -> подпространство через случайные фурье-признаки. Это
  кодирует только расстояние между точками в масштабе `scale`; дальше
  масштаба подпространства почти независимы (d^2 ~ k - k^2/d).
- srp256: 256 бит по проектору P (не зависят от выбора базиса Y), через
  ранг-1 гиперплоскости <u u^T, P> = ||Y^T u||^2 с порогом k/d * ||u||^2.
- srp_point / decode_srp_point: прямой SRP точки со сдвигами порогов и
  согласованное восстановление (проекции на полуплоскости).
"""
import numpy as np

K, D = 4, 128
FIELD = 200_000


def orth(Y):
    """Ортонормированный базис того же подпространства; знак столбцов фиксирован."""
    Q, R = np.linalg.qr(np.asarray(Y, dtype=float))
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s


def proj(Y):
    return Y @ Y.T


def principal_angles(Ya, Yb):
    s = np.linalg.svd(Ya.T @ Yb, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def chordal2(Ya, Yb):
    M = Ya.T @ Yb
    return float(Ya.shape[1] - np.sum(M * M))


# --------------------------------------------------------------- кодирование

class Embed:
    """Точка (x, y) -> Y в Gr(K, D): K групп по D случайных фурье-признаков."""

    def __init__(self, scale=500.0, seed=0, k=K, d=D):
        rng = np.random.default_rng(seed)
        self.k, self.d, self.scale = k, d, float(scale)
        self.W = rng.normal(size=(k, d, 2)) / self.scale
        self.b = rng.uniform(0, 2 * np.pi, size=(k, d))

    def __call__(self, xy):
        x = np.asarray(xy, dtype=float)
        F = np.cos(self.W @ x + self.b) * np.sqrt(2.0 / self.d)   # (k, d)
        return orth(F.T)


class SRP:
    """256 бит по проектору P: бит i = [||Y^T u_i||^2 > k/d * ||u_i||^2]."""

    def __init__(self, bits=256, seed=1, k=K, d=D):
        rng = np.random.default_rng(seed)
        self.U = rng.normal(size=(d, bits))
        self.tau = k / d * np.sum(self.U * self.U, axis=0)

    def __call__(self, Y):
        z = Y.T @ self.U
        return np.packbits(np.sum(z * z, axis=0) > self.tau)


def hamming(a, b):
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


# ------------------------------------------------------------- кодбук 512

def grid_centers(m=512, field=FIELD):
    """Регулярная сетка центров: 16 x 32 = 512 на поле field x field."""
    nx = 2 ** (int(np.log2(m)) // 2)
    ny = m // nx
    xs = (np.arange(nx) + 0.5) * field / nx
    ys = (np.arange(ny) + 0.5) * field / ny
    return np.array([(x, y) for x in xs for y in ys]), (field / nx, field / ny)


def nearest_center(xy, centers):
    return int(np.argmin(np.sum((centers - np.asarray(xy, float)) ** 2, axis=1)))


class Codebook:
    """512 центров, их подпространства и подписи; декодер - ближайший по Хэммингу."""

    def __init__(self, embed, srp, m=512, field=FIELD):
        self.centers, self.cell = grid_centers(m, field)
        self.sigs = np.array([srp(embed(c)) for c in self.centers])

    def decode(self, bits):
        x = np.bitwise_xor(self.sigs, bits)
        dist = np.unpackbits(x, axis=1).sum(axis=1)
        return self.centers[int(np.argmin(dist))]


# ------------------------------------------------- прямой SRP точки (dithered)

class PointSRP:
    """b_i = [a_i . x > t_i], a_i - случайные единичные направления на плоскости,
    t_i - равномерные сдвиги в диапазоне проекций поля."""

    def __init__(self, bits=256, seed=2, field=FIELD):
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0, np.pi, size=bits)
        self.A = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        lo = np.minimum(0, self.A @ np.array([field, 0.0])) + np.minimum(0, self.A @ np.array([0.0, field]))
        hi = np.maximum(0, self.A @ np.array([field, 0.0])) + np.maximum(0, self.A @ np.array([0.0, field]))
        self.t = rng.uniform(lo, hi)
        self.field = field

    def __call__(self, xy):
        return self.A @ np.asarray(xy, float) > self.t

    def decode(self, bits, iters=5000, margin=1e-3):
        """Согласованное восстановление: циклические проекции на полуплоскости
        (POCS) с малым запасом внутрь, старт из центра поля. Возвращает точку,
        согласную со всеми битами, если такая найдена за iters проходов."""
        s = np.where(bits, 1.0, -1.0)
        x = np.array([self.field / 2, self.field / 2])
        for _ in range(iters):
            viol = s * (self.A @ x - self.t) <= 0
            if not viol.any():
                break
            for i in np.flatnonzero(viol):
                g = s[i] * (self.A[i] @ x - self.t[i])
                if g <= 0:
                    x = x + (margin - g) * s[i] * self.A[i]
        return x


# ------------------------------------------------------- среднее Кархера

def log_map(Y, X):
    """Касательный вектор в Y, указывающий на X (Грассман, канонич. метрика)."""
    YtX = Y.T @ X
    T = (X - Y @ YtX) @ np.linalg.inv(YtX)
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    return U @ np.diag(np.arctan(s)) @ Vt


def exp_map(Y, H):
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    return orth(Y @ Vt.T @ np.diag(np.cos(s)) @ Vt + U @ np.diag(np.sin(s)) @ Vt)


def karcher_mean(Ys, iters=50, median=False, tol=1e-10):
    """Среднее (или геометрическая медиана, Вайсфельд) подпространств."""
    Y = Ys[0]
    for _ in range(iters):
        Hs = [log_map(Y, X) for X in Ys]
        if median:
            w = np.array([1.0 / max(np.linalg.norm(H), 1e-9) for H in Hs])
            w /= w.sum()
        else:
            w = np.full(len(Hs), 1.0 / len(Hs))
        step = sum(wi * H for wi, H in zip(w, Hs))
        if np.linalg.norm(step) < tol:
            break
        Y = exp_map(Y, step)
    return Y
