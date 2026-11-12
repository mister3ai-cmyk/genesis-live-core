# Business Source License 1.1 - см. research/LICENSE
"""Точность rt_xcorr_delay_multi: задержка IMU относительно позы по трём
компонентам угловой скорости в связанной СК (гироскоп против первой
производной ориентации). Модуль |w| не годится: при постоянном вращении
(«бочка») он почти константа - так было в первой версии этой проверки.

Сетка 200 Гц (1 отсчёт = 5 мс), окно 2 с (400 отсчётов), поиск +-100 мс
(+-20 отсчётов), порог возбуждения - суммарная дисперсия компонент не меньше
(0.05 рад/с)^2. Угловая скорость позы - вектор поворота R[k-2]^T R[k+2],
делённый на интервал (первая производная; вторая - шумная).
EuRoC: истина выровнена с IMU не идеально, поэтому восстановление вводимого
сдвига меряется как (оценка при d) - (оценка при d = 0) в том же окне.

  python3 xcorr_check.py [data/euroc/V1_02_medium ...]   # без аргументов - только синтетика
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import rt_core                                               # noqa: E402

FS, WIN, LAG = 200.0, 400, 20
MIN_VAR = 0.05 ** 2
MIN_PEAK = 0.9          # ниже - окно отбраковывается вызывающим (код LOW_PEAK)
LOW_PEAK = -100
DELAYS_MS = (0, 5, 10, 20, 50)


def rate_from_R(t, R, span=2):
    """w в связанной СК по ориентациям: вектор поворота R[k-span]^T R[k+span]
    / интервал, (n, 3)."""
    import sixdof as sd
    Rr = np.einsum("nji,njk->nik", R[:-2 * span], R[2 * span:])
    return t[span:-span], sd._rotvec(Rr) / (t[2 * span:] - t[:-2 * span])[:, None]


def quat_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def windows(lib, t_pose, w_pose, t_imu, w_imu, t_lo, t_hi, step_s=1.0):
    """Оценки задержки по окнам 2 с: список (код, задержка в мс)."""
    g = np.arange(t_lo, t_hi, 1.0 / FS)
    a = np.stack([np.interp(g, t_pose, w_pose[:, c]) for c in range(3)], 1)
    b = np.stack([np.interp(g, t_imu, w_imu[:, c]) for c in range(3)], 1)
    out = []
    for s in range(0, len(g) - WIN, int(step_s * FS)):
        rc, lag, peak = rt_core.xcorr(lib, a[s:s + WIN], b[s:s + WIN], LAG, MIN_VAR)
        if rc == rt_core.OK and peak < MIN_PEAK:
            rc = LOW_PEAK                                    # пик слабый: оценке не доверять
        out.append((rc, lag * 1000.0 / FS))
    return out


def summarize(est, delay_ms):
    ok = [e for rc, e in est if rc == rt_core.OK]
    low = sum(rc in (rt_core.ERR_LOW_EXCITATION, LOW_PEAK) for rc, _ in est)
    if not ok:
        return dict(n=len(est), low=low, med=float("nan"), p95=float("nan"), within=float("nan"))
    err = np.abs(np.array(ok) - delay_ms)
    return dict(n=len(est), low=low, med=float(np.median(err)), p95=float(np.percentile(err, 95)),
                within=100.0 * float(np.mean(err <= 1000.0 / FS)))


def synthetic(lib, n_traj=8, seed=11):
    import sixdof as sd
    res = {}
    for label, sp, sa in (("VIO 2 см / 0.3°", 0.02, 0.3), ("Vicon 1 мм / 0.1°", 0.001, 0.1)):
        for regime in sd.REGIMES:
            for dms in DELAYS_MS:
                rng = np.random.default_rng(seed)
                est = []
                for j in range(n_traj):
                    tr = sd.truth(rng, regime, T=14.0)
                    tp, _, Rp = sd.pose(rng, tr, sp=sp, sa_deg=sa)
                    tk, _, wk = sd.imu(np.random.default_rng(100 + j), tr, dms / 1000.0)
                    t_r, w_r = rate_from_R(tp, Rp)
                    est += windows(lib, t_r, w_r, tk, wk, 1.0, 13.0)
                res[(label, regime, dms)] = summarize(est, dms)
    return res


def euroc(lib, seq_dir):
    import euroc_bench as eb
    d = eb.load(seq_dir)
    gt = np.loadtxt(os.path.join(seq_dir, "state_groundtruth_estimate0.csv"), delimiter=",", comments="#")
    t_r, w_r = rate_from_R(d["t"], quat_R(gt[:, 4:8]))
    res = {}
    base = windows(lib, t_r, w_r, d["ti"], d["w"], 1.0, d["t"][-1] - 1.0)
    ok0 = [e for rc, e in base if rc == rt_core.OK]
    res["natural"] = float(np.median(ok0)) if ok0 else float("nan")
    res["natural_p"] = (float(np.percentile(ok0, 5)), float(np.percentile(ok0, 95))) if ok0 else None
    for dms in DELAYS_MS:
        est = windows(lib, t_r, w_r, d["ti"] + dms / 1000.0, d["w"], 1.0, d["t"][-1] - 1.0)
        rel = [(rc if rc0 == rt_core.OK else rc0, e - e0) for (rc, e), (rc0, e0) in zip(est, base)]
        res[dms] = summarize(rel, dms)
    return d["name"], res


def main():
    lib = rt_core.load()
    print("Синтетика 6-DoF: ошибка оценки задержки IMU, мс (1 отсчёт = 5 мс)")
    print("  %-18s %-9s %5s | %6s %6s %8s %10s" % ("шум позы", "режим", "d, мс", "медиана", "p95", "<=5 мс", "отбраковано"))
    for (label, regime, dms), r in sorted(synthetic(lib).items()):
        print("  %-18s %-9s %5d | %6.2f %6.2f %7.1f %% %6d / %d" % (
            label, regime, dms, r["med"], r["p95"], r["within"], r["low"], r["n"]))
    for p in sys.argv[1:]:
        name, res = euroc(lib, p)
        print("\nEuRoC %s: собственное смещение IMU относительно истины: медиана %.2f мс (5-95 %%: %.1f..%.1f)"
              % (name, res["natural"], res["natural_p"][0], res["natural_p"][1]))
        print("  восстановление вводимого сдвига (оценка при d минус оценка при 0 в том же окне):")
        print("  %5s | %6s %6s %8s %10s" % ("d, мс", "медиана", "p95", "<=5 мс", "отбраковано"))
        for dms in DELAYS_MS:
            r = res[dms]
            print("  %5d | %6.2f %6.2f %7.1f %% %6d / %d" % (dms, r["med"], r["p95"], r["within"], r["low"], r["n"]))


if __name__ == "__main__":
    main()
