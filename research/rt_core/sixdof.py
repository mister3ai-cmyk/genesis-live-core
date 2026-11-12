# Business Source License 1.1 - см. research/LICENSE
"""6-DoF генератор твёрдого тела и эксперимент «рассинхрон IMU и позы».

Истина (1 кГц): позиция - сумма синусоид + дрейф; ориентация - углы ZYX
(рысканье psi, тангаж theta, крен phi), режимы:
  straight - малые колебания; turn - разворот 0.5 рад/с с креном;
  barrel   - «бочка»: крен 2 pi за 4 с и винтовое смещение.
Угловая скорость в связанной СК - из производных углов Эйлера, угловое
ускорение - численно по истине. IMU (200 Гц) стоит в точке r_b от центра
масс: удельная сила f_b = R^T (a + alpha x r + omega x (omega x r) - g),
гироскоп omega_b; шум и постоянные смещения. Поза (100 Гц): позиция +-2 см,
ориентация 0.3 град.

Рассинхрон: поток IMU (весь или одна ось акселерометра) отстаёт на delta мс.
Два детектора на окне 64 отсчёта IMU (0.32 с):
  L - невязка простого предиктора: ускорение и угловая скорость из
      квадратичной подгонки позы (+-0.1 с) против IMU, приведённого в мир;
  G - Gr(4, 64): подпространство 4 главных направлений матрицы 64 x 6
      (каналы ускорения и угловой скорости, нормированы) из позы против того
      же из IMU; хордальное расстояние считает C-ядро (rt_gr_chordal2).
Числа синтетические (S2)."""
import numpy as np

G = np.array([0.0, 0.0, -9.81])
R_ARM = np.array([0.10, 0.03, 0.02])        # плечо IMU от центра масс, м
REGIMES = ("straight", "turn", "barrel")


def euler_R(psi, th, ph):
    """R = Rz(psi) Ry(theta) Rx(phi), массивы углов -> (n, 3, 3)."""
    cps, sps, cth, sth, cph, sph = np.cos(psi), np.sin(psi), np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
    R = np.empty(psi.shape + (3, 3))
    R[..., 0, 0] = cps * cth
    R[..., 0, 1] = cps * sth * sph - sps * cph
    R[..., 0, 2] = cps * sth * cph + sps * sph
    R[..., 1, 0] = sps * cth
    R[..., 1, 1] = sps * sth * sph + cps * cph
    R[..., 1, 2] = sps * sth * cph - cps * sph
    R[..., 2, 0] = -sth
    R[..., 2, 1] = cth * sph
    R[..., 2, 2] = cth * cph
    return R


def _sines(rng, t, n, amp, wmin, wmax):
    A = rng.uniform(-amp, amp, n)
    w = rng.uniform(wmin, wmax, n)
    f = rng.uniform(0, 2 * np.pi, n)
    x = (A[:, None] * np.sin(w[:, None] * t + f[:, None])).sum(0)
    dx = (A[:, None] * w[:, None] * np.cos(w[:, None] * t + f[:, None])).sum(0)
    ddx = -(A[:, None] * w[:, None] ** 2 * np.sin(w[:, None] * t + f[:, None])).sum(0)
    return x, dx, ddx


def truth(rng, regime, T=12.0, dt=0.001):
    t = np.arange(0.0, T, dt)
    pos, acc = np.zeros((len(t), 3)), np.zeros((len(t), 3))
    v0 = rng.uniform(-3, 3, 3)
    for ax in range(3):
        x, _, ddx = _sines(rng, t, 3, 1.5 if ax < 2 else 0.5, 0.3, 2.0)
        pos[:, ax] = v0[ax] * t + x
        acc[:, ax] = ddx
    psi, dpsi, _ = _sines(rng, t, 2, 0.2, 0.3, 1.5)
    th, dth, _ = _sines(rng, t, 2, 0.1, 0.3, 1.5)
    ph, dph, _ = _sines(rng, t, 2, 0.1, 0.3, 1.5)
    if regime == "turn":
        psi, dpsi, ph = psi + 0.5 * t, dpsi + 0.5, ph + 0.3
    elif regime == "barrel":
        rate = 2 * np.pi / 4.0
        ph, dph = ph + rate * t, dph + rate
        rad = 2.0                                  # винт: смещение по y, z с кренов
        pos[:, 1] += rad * np.cos(rate * t)
        pos[:, 2] += rad * np.sin(rate * t)
        acc[:, 1] += -rad * rate ** 2 * np.cos(rate * t)
        acc[:, 2] += -rad * rate ** 2 * np.sin(rate * t)
    R = euler_R(psi, th, ph)
    wb = np.stack([dph - dpsi * np.sin(th),
                   dth * np.cos(ph) + dpsi * np.cos(th) * np.sin(ph),
                   -dth * np.sin(ph) + dpsi * np.cos(th) * np.cos(ph)], axis=1)
    alb = np.gradient(wb, dt, axis=0)
    return dict(t=t, pos=pos, acc=acc, R=R, wb=wb, alb=alb, dt=dt)


def imu(rng, tr, delay_s=0.0, axis=None, rate=200.0):
    """Отсчёты IMU в моменты k / rate; показания - истина в момент
    t - delay_s (весь поток или только ось акселерометра axis)."""
    tk = np.arange(0.5, tr["t"][-1] - 0.5, 1.0 / rate)

    def at(tq):
        i = np.clip(np.round(tq / tr["dt"]).astype(int), 0, len(tr["t"]) - 1)
        R, wb, alb, a = tr["R"][i], tr["wb"][i], tr["alb"][i], tr["acc"][i]
        rw = np.einsum("nij,j->ni", R, R_ARM)
        ww = np.einsum("nij,nj->ni", R, wb)
        aw = np.einsum("nij,nj->ni", R, alb)
        a_s = a + np.cross(aw, rw) + np.cross(ww, np.cross(ww, rw))
        f = np.einsum("nji,nj->ni", R, a_s - G)
        return f, wb

    f, w = at(tk)
    if delay_s:
        fd, wd = at(tk - delay_s)
        if axis is None:
            f, w = fd, wd
        else:
            f = f.copy()
            f[:, axis] = fd[:, axis]
    f = f + rng.normal(0, 0.08, f.shape) + rng.uniform(-0.05, 0.05, 3)
    w = w + rng.normal(0, 0.004, w.shape) + rng.uniform(-0.002, 0.002, 3)
    return tk, f, w


def pose(rng, tr, rate=100.0, sp=0.02, sa_deg=0.3):
    """Поза с шумом: позиция sp м, ориентация sa_deg град (по умолчанию -
    визуальная одометрия; Vicon в EuRoC - около 0.001 м и 0.1 град)."""
    tp = np.arange(0.0, tr["t"][-1], 1.0 / rate)
    i = np.round(tp / tr["dt"]).astype(int)
    p = tr["pos"][i] + rng.normal(0, sp, (len(i), 3))
    e = rng.normal(0, np.radians(sa_deg), (len(i), 3))
    R = np.einsum("nij,njk->nik", tr["R"][i], euler_R(e[:, 0], e[:, 1], e[:, 2]))
    return tp, p, R


def _rotvec(R):
    """log SO(3) -> вектор поворота, (n, 3, 3) -> (n, 3)."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(ang)
    k = np.where(s > 1e-9, ang / (2 * np.where(s > 1e-9, s, 1)), 0.5)
    return v * k[:, None]


def pose_derivs(tp, p, R, tq, half=0.1):
    """В моменты tq: ускорение (мир), угловая скорость (связанная СК) и R
    из квадратичной подгонки позы по +-half с."""
    a_out, w_out, R_out = np.empty((len(tq), 3)), np.empty((len(tq), 3)), np.empty((len(tq), 3, 3))
    for n, t0 in enumerate(tq):
        lo, hi = np.searchsorted(tp, [t0 - half, t0 + half])
        tau = tp[lo:hi] - t0
        X = np.stack([np.ones_like(tau), tau, tau * tau], 1)
        cp = np.linalg.lstsq(X, p[lo:hi], rcond=None)[0]
        a_out[n] = 2 * cp[2]
        j = lo + int(np.argmin(np.abs(tau)))
        Rj = R[j]
        phi = _rotvec(np.einsum("ji,njk->nik", Rj, R[lo:hi]))      # R_j^T R
        cw = np.linalg.lstsq(X, phi, rcond=None)[0]
        tj = tp[j] - t0
        w_out[n] = cw[1] + 2 * cw[2] * tj
        R_out[n] = Rj
    return a_out, w_out, R_out


def window_stats(lib, rt, tk, f, w, tp, p, R, start, W=64):
    """Детекторы L и G на окне IMU [start, start + W)."""
    sl = slice(start, start + W)
    a_p, w_p, R_p = pose_derivs(tp, p, R, tk[sl])
    rw = np.einsum("nij,j->ni", R_p, R_ARM)
    ww = np.einsum("nij,nj->ni", R_p, w[sl])
    a_imu = np.einsum("nij,nj->ni", R_p, f[sl]) + G - np.cross(ww, np.cross(ww, rw))
    L = np.mean(np.linalg.norm(a_p - a_imu, axis=1) / 1.0 + np.linalg.norm(w_p - w[sl], axis=1) / 0.05)

    def sub(X):
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        U = np.linalg.svd(X, full_matrices=False)[0][:, :4]
        return rt.orth(lib, U)

    Gd = rt.chordal2(lib, sub(np.hstack([a_p, w_p])), sub(np.hstack([a_imu, w[sl]])))
    return L, Gd


def best_split(h, a):
    """Порог с минимумом ошибок: h - без рассинхрона (ниже), a - с ним."""
    vals = np.sort(np.concatenate([h, a]))
    best = (len(h) + len(a), None)
    for v in vals:
        e = int(np.sum(h >= v) + np.sum(a < v))
        if e < best[0]:
            best = (e, v)
    return 100.0 * best[0] / (len(h) + len(a))


def experiment(lib, rt, n_traj=12, n_win=5, delays_ms=(0, 5, 10, 20, 50), seed=7,
               sp=0.02, sa_deg=0.3):
    """Ошибка лучшего порога (%) для L и G по режимам, вариантам рассинхрона и
    delta. 50 % - угадывание, 0 % - разделяет полностью."""
    out = {}
    rng = np.random.default_rng(seed)
    for regime in REGIMES:
        stats = {}
        for j in range(n_traj):
            tr = truth(rng, regime)
            tp, p, R = pose(rng, tr, sp=sp, sa_deg=sa_deg)
            starts = rng.integers(10, 2000 - 80, n_win)
            for variant, axis in (("весь IMU", None), ("ось y акселерометра", 1)):
                for d in delays_ms:
                    tk, f, w = imu(np.random.default_rng(1000 * j + d), tr, d / 1000.0, axis)
                    for s in starts:
                        L, Gd = window_stats(lib, rt, tk, f, w, tp, p, R, int(s))
                        stats.setdefault((variant, d), []).append((L, Gd))
        for (variant, d), v in stats.items():
            if d == 0:
                continue
            h = np.array(stats[(variant, 0)])
            a = np.array(v)
            out[(regime, variant, d)] = (best_split(h[:, 0], a[:, 0]), best_split(h[:, 1], a[:, 1]),
                                         float(np.median(h[:, 0])), float(np.median(a[:, 0])),
                                         float(np.median(h[:, 1])), float(np.median(a[:, 1])))
    return out
