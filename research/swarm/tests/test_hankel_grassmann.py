# Business Source License 1.1 - см. research/LICENSE
"""v4.0: подпись траектории блочной матрицей Ганкеля на Gr(4, 128).
Числа печатаются (pytest -s); проверяется только то, что держится с запасом
на 300 траекториях. Порог аномалии THETA = 0.3 (вместо >= 2.5 из ТЗ, см.
hankel_grassmann.py); доля атак с d^2 >= 2.5 печатается для сравнения."""
import os
import sys
import time

import pytest

np = pytest.importorskip("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM = os.path.dirname(HERE)
if SWARM not in sys.path:
    sys.path.insert(0, SWARM)

import grassmann as gm                                       # noqa: E402
from hankel_grassmann import THETA, HankelTrajectoryEmbedder, fill_gaps   # noqa: E402

pytestmark = pytest.mark.concept

W, L = 8, 4
N_TRAJ = 300


def trajectory(rng, w=W):
    """Разгон, дуга, торможение. Шаг 300..1100, поворот до 0.25 рад за шаг."""
    p = rng.uniform(20_000, 180_000, size=2)
    sp, h = rng.uniform(300, 600), rng.uniform(0, 2 * np.pi)
    om, acc = rng.uniform(-0.25, 0.25), rng.uniform(80, 150)
    out = [p.copy()]
    for t in range(1, w):
        sp = sp + acc if t < w // 2 else max(200.0, sp - acc)
        h += om
        p = p + sp * np.array([np.cos(h), np.sin(h)])
        out.append(p.copy())
    return np.array(out)


def step_of(T):
    return float(np.mean(np.linalg.norm(np.diff(T, axis=0), axis=1)))


def noisy(rng, T, frac=0.05):
    """Шум датчика: равномерный +-frac от среднего шага по каждой оси."""
    return T + rng.uniform(-1, 1, T.shape) * frac * step_of(T)


def teleport(rng, T):
    A = T.copy()
    A[rng.integers(1, W)] += rng.uniform(3000, 5000, 2) * rng.choice((-1, 1), 2)
    return A


def flip180(T, at=4):
    B = T.copy()
    B[at:] = T[at - 1] - (T[at:] - T[at - 1])
    return B


def jerk(rng, T, at=5, a=400.0):
    ang = rng.uniform(0, 2 * np.pi)
    C = T.copy()
    C[at:] += np.outer(np.arange(1, W - at + 1) ** 2, a * np.array([np.cos(ang), np.sin(ang)]))
    return C


def stats(a):
    a = np.asarray(a)
    return "медиана %.3f  p5 %.3f  p95 %.3f  мин %.3f  макс %.3f" % (
        np.median(a), np.percentile(a, 5), np.percentile(a, 95), a.min(), a.max())


# --- устройство матрицы ------------------------------------------------------

def test_hankel_block_structure_and_orthonormal_basis():
    emb = HankelTrajectoryEmbedder(window_size=W, lag=L)
    T = trajectory(np.random.default_rng(0))
    H = emb.hankel(T)
    m, n = emb.m, W - L + 1
    assert H.shape == (L * m, n) == (128, 5)
    F = emb.phi(T - T[0])
    for i in range(L):
        for j in range(n):
            assert np.allclose(H[i * m:(i + 1) * m, j], F[i + j])    # блок i столбца j = phi(y_{i+j})
    Y = emb(T)
    assert Y.shape == (128, 4)
    assert np.allclose(Y.T @ Y, np.eye(4), atol=1e-12)


def test_reverse_is_block_reflection_of_forward():
    """H_rev = J H_fwd P: переставлены блоки строк и столбцы. Без relative это
    тождество точное; оно и делает стрелу времени видимой."""
    emb = HankelTrajectoryEmbedder(window_size=W, lag=L, relative=False)
    T = trajectory(np.random.default_rng(1))
    H, Hr = emb.hankel(T), emb.hankel(T[::-1])
    m = emb.m
    J = np.vstack([H[(L - 1 - i) * m:(L - i) * m] for i in range(L)])
    assert np.allclose(Hr, J[:, ::-1])


# --- стрела времени, шум, атаки ---------------------------------------------

def _measure():
    emb = HankelTrajectoryEmbedder(window_size=W, lag=L)
    rng = np.random.default_rng(2)
    out = {k: [] for k in ("reverse", "noise", "teleport", "flip180", "jerk")}
    for _ in range(N_TRAJ):
        T = trajectory(rng)
        Y = emb(T)
        out["reverse"].append(gm.chordal2(Y, emb(T[::-1])))
        out["noise"].append(gm.chordal2(Y, emb(noisy(rng, T))))
        out["teleport"].append(gm.chordal2(Y, emb(teleport(rng, T))))
        out["flip180"].append(gm.chordal2(Y, emb(flip180(T))))
        out["jerk"].append(gm.chordal2(Y, emb(jerk(rng, T))))
    return {k: np.array(v) for k, v in out.items()}


@pytest.fixture(scope="module")
def measured():
    return _measure()


def test_time_reversal_asymmetry(measured):
    rev, noise = measured["reverse"], measured["noise"]
    print("\nW=%d L=%d, %d траекторий, d^2 (максимум 4):" % (W, L, N_TRAJ))
    print("  прямой против обратного хода   " + stats(rev))
    print("  прямой против себя с шумом 5 % " + stats(noise))
    assert rev.min() > 20 * np.percentile(noise, 95)


def test_noise_vs_kinematic_attacks(measured):
    noise = measured["noise"]
    print("\nd^2 к честной траектории (ТЗ: шум <= 0.05, атака >= 2.5):")
    print("  шум +-5 % шага        " + stats(noise))
    for k, name in (("teleport", "телепорт 3000-5000 "), ("flip180", "разворот скорости  "),
                    ("jerk", "скачок ускорения   ")):
        a = measured[k]
        print("  %s   %s   >= 2.5: %5.1f %%" % (name, stats(a), 100 * np.mean(a >= 2.5)))
    print("  порог THETA = %.1f: ложных тревог %.1f %%, пропущено атак %.1f %%" % (
        THETA, 100 * np.mean(noise >= THETA),
        100 * np.mean(np.concatenate([measured[k] for k in ("teleport", "flip180", "jerk")]) < THETA)))
    assert np.percentile(noise, 95) <= 0.05
    assert noise.max() < THETA                       # ложных тревог 0
    for k in ("teleport", "flip180", "jerk"):
        assert measured[k].min() > THETA             # пропусков 0


# --- рой: медиана Кархера ----------------------------------------------------

def test_karcher_median_rejects_one_liar_of_five():
    """5 бортов видят одну цель: шум 5 %, у каждого 0-2 пропуска внутри окна
    (интерполяция). Борт-лжец шлёт развёрнутую траекторию."""
    emb = HankelTrajectoryEmbedder(window_size=W, lag=L)
    rng = np.random.default_rng(3)
    med, avg, single, liar_d = [], [], [], []
    for _ in range(100):
        T = trajectory(rng)
        Yt = emb(T)
        Ys = []
        for _node in range(4):
            O = noisy(rng, T)
            O[rng.choice(np.arange(1, W - 1), size=rng.integers(0, 3), replace=False)] = np.nan
            Ys.append(emb(fill_gaps(O)))
        Yl = emb(flip180(noisy(rng, T)))
        Ys.insert(rng.integers(0, 5), Yl)
        med.append(gm.chordal2(gm.karcher_mean(Ys, median=True), Yt))
        avg.append(gm.chordal2(gm.karcher_mean(Ys), Yt))
        single.append(np.median([gm.chordal2(Y, Yt) for Y in Ys if Y is not Yl]))
        liar_d.append(gm.chordal2(Yl, Yt))
    med, avg = np.array(med), np.array(avg)
    print("\nРой 5 бортов, 1 лжец (разворот), 100 испытаний, d^2 до истины:")
    print("  медиана Кархера  " + stats(med))
    print("  среднее Кархера  " + stats(avg))
    print("  честный борт     " + stats(single))
    print("  лжец             " + stats(liar_d))
    print("  медиана ближе среднего: %d из 100" % np.sum(med < avg))
    assert np.median(med) < np.median(avg)
    assert med.max() < np.min(liar_d)


# --- цена ----------------------------------------------------------------------

def test_cost_print():
    emb = HankelTrajectoryEmbedder(window_size=W, lag=L)
    T = trajectory(np.random.default_rng(4))
    srp = gm.SRP(seed=5)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        H = emb.hankel(T)
    t1 = time.perf_counter()
    for _ in range(n):
        np.linalg.svd(H, full_matrices=False)
    t2 = time.perf_counter()
    for _ in range(n):
        srp(emb(T))
    t3 = time.perf_counter()
    sig = srp(emb(T))
    print("\nЦена окна W=%d L=%d (128 x 5), на одном ядре этой машины:" % (W, L))
    print("  матрица Ганкеля %.1f мкс, SVD %.1f мкс, всё + SRP-256 %.1f мкс"
          % ((t1 - t0) / n * 1e6, (t2 - t1) / n * 1e6, (t3 - t2) / n * 1e6))
    print("  подпись SRP-256: %d B; сами 8 точек окна varint: 8 x 6 = 48 B" % sig.nbytes)
    assert sig.nbytes == 32
