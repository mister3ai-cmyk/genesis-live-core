# Business Source License 1.1 - см. research/LICENSE
"""v4.0: подпись траектории - подпространство блочной матрицы Ганкеля на Gr(k, d).

Изолированный модуль: симулятор его пока не вызывает.

Окно из W наблюдений y_t, ..., y_{t+W-1} (точки на плоскости). Каждое
наблюдение вкладывается случайными фурье-признаками phi(y) в R^m, m = d / L.
Блочная матрица Ганкеля задержек (L блоков по строкам, W - L + 1 столбцов):

    H = [ phi(y_t)       phi(y_{t+1})   ...  phi(y_{t+W-L})  ]
        [ phi(y_{t+1})   phi(y_{t+2})   ...  phi(y_{t+W-L+1}) ]
        [ ...                                                 ]
        [ phi(y_{t+L-1}) phi(y_{t+L})   ...  phi(y_{t+W-1})   ]

Это аналог матрицы наблюдаемости [C; CA; ...; CA^{L-1}] линейной системы
x_{t+1} = A x_t, y_t = C x_t. Подпись - первые k левых сингулярных векторов
H: базис Y (d x k), точка Gr(k, d).

Стрела времени: для обратного хода H_rev = J H_fwd P, где J переставляет
блоки строк в обратном порядке, а P - столбцы. Поэтому span(Y_rev) =
J span(Y_fwd); подпространства совпадают, только если span(Y_fwd)
инвариантно относительно J. Для движения с разгоном или поворотом это не так.

relative=True (по умолчанию): точки берутся относительно первой точки окна, и
подпись описывает форму движения, а не место. Место передаётся явными
координатами (varint, Марзулло) - подпись его не заменяет (SWARM_FINDINGS,
Часть IX, v3.1-5).

Порог аномалии THETA = 0.3 по d^2 (решение 11.11.2026 вместо >= 2.5 из ТЗ).
d^2 >= 2.5 для частичной атаки недостижим по построению: атака портит часть
окна, а неповреждённые столбцы H остаются общими с честной траекторией, и
подпространства не могут стать почти ортогональными. Замер на 300 траекториях
(W = 8, L = 4, tests/test_hankel_grassmann.py): худший шум +-5 % шага -
0.063, слабейшая атака (телепорт, разворот скорости, скачок ускорения) -
0.67; при THETA = 0.3 ложных тревог и пропусков 0.
"""
import numpy as np

THETA = 0.3


class HankelTrajectoryEmbedder:
    def __init__(self, window_size=8, lag=4, k=4, d=128, scale=1000.0, seed=0, relative=True):
        if d % lag:
            raise ValueError("d должно делиться на lag")
        if not 1 <= lag <= window_size:
            raise ValueError("нужно 1 <= lag <= window_size")
        if k > window_size - lag + 1:
            raise ValueError("k больше числа столбцов H (window_size - lag + 1)")
        self.W, self.L, self.k, self.d = window_size, lag, k, d
        self.m = d // lag
        self.relative = relative
        rng = np.random.default_rng(seed)
        self.Wf = rng.normal(size=(self.m, 2)) / float(scale)
        self.b = rng.uniform(0, 2 * np.pi, size=self.m)

    def phi(self, pts):
        """(n, 2) -> (n, m) случайные фурье-признаки гауссова ядра масштаба scale."""
        return np.cos(np.asarray(pts, float) @ self.Wf.T + self.b) * np.sqrt(2.0 / self.m)

    def hankel(self, pts):
        P = np.asarray(pts, float)
        if P.shape != (self.W, 2):
            raise ValueError("ожидается окно (%d, 2)" % self.W)
        if self.relative:
            P = P - P[0]
        F = self.phi(P)
        n = self.W - self.L + 1
        return np.vstack([F[i:i + n].T for i in range(self.L)])   # (L*m, n), блок i = phi(y_{i+j})

    def __call__(self, pts):
        U, _, _ = np.linalg.svd(self.hankel(pts), full_matrices=False)
        return U[:, :self.k]


def fill_gaps(pts):
    """Пропущенные отсчёты окна (строки с NaN) - линейной интерполяцией по
    номеру отсчёта; на краях - ближайшим известным. Нужно >= 2 известных."""
    P = np.array(pts, float)
    ok = ~np.isnan(P).any(axis=1)
    if ok.sum() < 2:
        raise ValueError("меньше двух известных отсчётов")
    t = np.arange(len(P))
    for j in range(P.shape[1]):
        P[~ok, j] = np.interp(t[~ok], t[ok], P[ok, j])
    return P
