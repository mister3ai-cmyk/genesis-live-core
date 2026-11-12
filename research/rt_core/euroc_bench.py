# Business Source License 1.1 - см. research/LICENSE
"""EuRoC MAV: проверка rt_core на реальной полётной траектории.

A. Шлюз (rt_kin_gate) на живой траектории. 4 виртуальных борта видят дрон
   (истина state_groundtruth_estimate0) на 20 Гц со своей фазой, шум sigma
   = 2 см; пятый борт - атакующий. Пороги перенесены из симулятора в единицах
   sigma: pos_tol 10s, drop_tol 14s, thr 28s. Горизонт подгонки и thr
   калибруются на первой половине первой сессии, цифры - на второй половине
   и на второй сессии.
B. Gr(4, 64) на сравнении сегментов движения. Каналы (50 Гц): ускорение и
   гироскоп IMU, горизонтальная и вертикальная скорость - нормированы по всему
   полёту; блочная матрица Ганкеля 8 каналов x 8 задержек = 64 строки, окно
   2 с; подпись - 4 главных левых сингулярных вектора; расстояние
   rt_gr_chordal2. Базовая линия - векторы признаков (среднее, разброс,
   мощность в 3 полосах по каналу).

  python3 euroc_bench.py data/euroc/V1_02_medium data/euroc/V2_02_medium
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import rt_core                                               # noqa: E402

SIGMA = 0.02


# ------------------------------------------------------------------ данные

def load(seq_dir):
    gt = np.loadtxt(os.path.join(seq_dir, "state_groundtruth_estimate0.csv"), delimiter=",", comments="#")
    imu = np.loadtxt(os.path.join(seq_dir, "imu0.csv"), delimiter=",", comments="#")
    d = dict(name=os.path.basename(os.path.normpath(seq_dir)),
             t=gt[:, 0] * 1e-9, p=gt[:, 1:4], q=gt[:, 4:8], v=gt[:, 8:11],
             ti=imu[:, 0] * 1e-9, w=imu[:, 1:4], a=imu[:, 4:7])
    t0 = d["t"][0]
    d["t"], d["ti"] = d["t"] - t0, d["ti"] - t0
    assert np.all(np.diff(d["t"]) > 0) and np.all(np.diff(d["ti"]) > 0)
    return d


def _interp(t, ts, X):
    return np.stack([np.interp(t, ts, X[:, k]) for k in range(X.shape[1])], -1)


# --------------------------------------------------------------- задача A

def gate_run(lib, d, t_lo, t_hi, horizon_s, thr, attack, seed=0, n_auth=4, rate=20.0,
             sigma=SIGMA, band=(20, 50)):
    """Поток заявок по времени; каждая проверяется шлюзом и затем пишется в
    кольцо. attack: 'T8' - борт 4 всегда лжёт на 20-50 sigma по оси;
    'jump' - борт 4 честен, кроме скачка 1.5-3 м в моменты верхних 10 %
    угловой скорости. Возвращает словарь долей."""
    rng = np.random.default_rng(seed)
    kin = rt_core.Kin(lib, horizon_us=int(horizon_s * 1e6), max_skew_us=50_000,
                      drop_tol=14 * sigma, thr=thr, pos_tol=10 * sigma, min_pts=4)
    wnorm = np.linalg.norm(d["w"], axis=1)
    w_hi = np.quantile(wnorm, 0.9)
    acc = np.linalg.norm(np.gradient(d["v"], d["t"], axis=0), axis=1)
    sharp_a, sharp_w = np.quantile(acc, 0.9), w_hi
    ev = []
    for a in range(n_auth + 1):
        ph = a / (rate * (n_auth + 1))
        for t in np.arange(t_lo + ph, t_hi, 1.0 / rate):
            ev.append((t, a))
    ev.sort()
    res = {"hon": 0, "hon_ref": 0, "hon_fs": 0, "hon_sharp": 0, "hon_sharp_ref": 0,
           "att": 0, "att_ref": 0, "att_fs": 0, "own_fault": 0, "own_fault_ref": 0, "lat": []}
    for t, a in ev:
        truth = _interp(np.array([t]), d["t"], d["p"])[0]
        claim = truth + rng.normal(0, sigma, 3)
        is_att = False
        if a == n_auth:
            if attack == "T8":
                claim = truth + rng.choice((-1, 1), 3) * rng.uniform(band[0] * sigma, band[1] * sigma, 3)
                is_att = True
            elif attack == "jump":
                wi = np.interp(t, d["ti"], wnorm)
                if wi >= w_hi:
                    claim = truth + rng.choice((-1, 1), 3) * rng.uniform(1.5, 3.0, 3)
                    is_att = True
        fault = rng.random() < 0.05
        mine = truth + (rng.choice((-1, 1), 3) * rng.uniform(60 * sigma, 100 * sigma, 3) if fault
                        else rng.normal(0, sigma, 3))
        t_us = int(t * 1e6)
        c0 = time.perf_counter()
        v, _ = kin.gate(0, a, t_us, t_us + 10_000, claim, mine)
        res["lat"].append(time.perf_counter() - c0)
        ref, fs = v == rt_core.REFUTE, v == rt_core.FAILSAFE
        if fault:
            res["own_fault"] += 1
            res["own_fault_ref"] += ref and not is_att
        elif is_att:
            res["att"] += 1
            res["att_ref"] += ref
            res["att_fs"] += fs
        else:
            res["hon"] += 1
            res["hon_ref"] += ref
            res["hon_fs"] += fs
            wi = np.interp(t, d["ti"], wnorm)
            ai = np.interp(t, d["t"], acc)
            if wi >= sharp_w or ai >= sharp_a:
                res["hon_sharp"] += 1
                res["hon_sharp_ref"] += ref
        kin.push(0, a, t_us, claim, t_us + 10_000)
    return res


def _pct(a, b):
    return 100.0 * a / b if b else float("nan")


def task_a(lib, seqs):
    d0 = seqs[0]
    mid = d0["t"][-1] / 2
    print("\n=== A. Шлюз на траектории EuRoC (sigma %.0f мм; pos_tol %.2f, drop_tol %.2f м) ===" % (
        SIGMA * 1e3, 10 * SIGMA, 14 * SIGMA))
    print("Калибровка на %s, первая половина (0-%.0f с):" % (d0["name"], mid))
    print("  %-8s %-6s | %-22s | %-14s | %-10s" % ("горизонт", "thr", "ложные тревоги, честн.", "T8 опровергнуто", "fail-safe T8"))
    best = None
    for hz in (0.5, 1.0, 2.0, 4.0):
        for thr_s in (20, 28, 40):
            thr = thr_s * SIGMA
            rh = gate_run(lib, d0, 1.0, mid, hz, thr, "T8")
            fp, det = _pct(rh["hon_ref"], rh["hon"]), _pct(rh["att_ref"], rh["att"])
            print("  %-8.1f %-6.2f | %20.2f %% | %12.1f %% | %8.1f %%" % (
                hz, thr, fp, det, _pct(rh["att_fs"], rh["att"])))
            score = fp + (100 - det)
            if best is None or score < best[0]:
                best = (score, hz, thr)
    _, hz, thr = best
    print("Выбрано на калибровке: горизонт %.1f с, thr %.2f м (минимум ложных тревог + пропусков T8)" % (hz, thr))
    print("\nПроверка (на калибровку не влияла):")
    rows = [(d0["name"] + ", 2-я половина", d0, mid, d0["t"][-1] - 1)]
    for d in seqs[1:]:
        rows.append((d["name"] + ", целиком", d, 1.0, d["t"][-1] - 1))
    for name, d, lo, hi in rows:
        r8 = gate_run(lib, d, lo, hi, hz, thr, "T8", seed=1)
        rj = gate_run(lib, d, lo, hi, hz, thr, "jump", seed=2)
        lat = np.array(r8["lat"] + rj["lat"]) * 1e6
        print("  %s:" % name)
        print("    честные заявки: %d, ложные тревоги %.2f %%, на резких 10 %% моментов %.2f %% (из %d), fail-safe %.1f %%"
              % (r8["hon"], _pct(r8["hon_ref"], r8["hon"]), _pct(r8["hon_sharp_ref"], r8["hon_sharp"]),
                 r8["hon_sharp"], _pct(r8["hon_fs"], r8["hon"])))
        print("    T8: опровергнуто %.1f %% из %d, fail-safe %.1f %%" % (
            _pct(r8["att_ref"], r8["att"]), r8["att"], _pct(r8["att_fs"], r8["att"])))
        print("    скачок 1.5-3 м на вираже: опровергнуто %.1f %% из %d" % (_pct(rj["att_ref"], rj["att"]), rj["att"]))
        print("    свой грубый сбой: ошибочных опровержений %d из %d (fail-safe ожидается)" % (
            r8["own_fault_ref"] + rj["own_fault_ref"], r8["own_fault"] + rj["own_fault"]))
        print("    rt_kin_gate через ctypes: p50 %.1f мкс, p99 %.1f мкс (вместе с накладными ctypes)"
              % (np.percentile(lat, 50), np.percentile(lat, 99)))

    # Стресс: где шлюз ломается. Пороги - в тех же единицах sigma, что выбраны
    # на калибровке; ложь ближе к шуму. Последняя сессия, целиком.
    d = seqs[-1]
    k_thr = thr / SIGMA
    print("\nСтресс на %s: горизонт %.1f с, thr %.0f sigma, pos_tol 10 sigma; ложь T8 - полоса сдвига по оси" % (
        d["name"], hz, k_thr))
    print("  %-8s %-10s | %-14s | %-14s | %-10s" % ("шум", "сдвиг", "ложные тревоги", "T8 опровергнуто", "fail-safe T8"))
    for sg in (0.02, 0.05, 0.10):
        for band in ((5, 10), (10, 20), (20, 50)):
            r = gate_run(lib, d, 1.0, d["t"][-1] - 1, hz, k_thr * sg, "T8", seed=4, sigma=sg, band=band)
            print("  %4.0f мм  %3d-%-3d s  | %12.2f %% | %12.1f %% | %8.1f %%" % (
                sg * 1e3, band[0], band[1], _pct(r["hon_ref"], r["hon"]), _pct(r["att_ref"], r["att"]),
                _pct(r["att_fs"], r["att"])))
    return hz, thr


# --------------------------------------------------------------- задача B

FS = 50.0
WIN = 100          # 2 с
LAGS = 8


def channels(d, delay_axis=None, delay_s=0.0, swap_xy=False):
    """8 каналов на 50 Гц: a_xyz, w_xyz (IMU, среднее по 4 отсчётам),
    |v_xy|, v_z (истина). Порча: задержка оси акселерометра, обмен x и y."""
    a, w = d["a"].copy(), d["w"].copy()
    if delay_axis is not None and delay_s:
        a[:, delay_axis] = np.interp(d["ti"] - delay_s, d["ti"], d["a"][:, delay_axis])
    if swap_xy:
        a[:, [0, 1]] = a[:, [1, 0]]
        w[:, [0, 1]] = w[:, [1, 0]]
    tg = np.arange(d["ti"][0] + 0.1, d["ti"][-1] - 0.1, 1.0 / FS)
    idx = np.searchsorted(d["ti"], tg)
    A = np.stack([a[np.clip(idx + k, 0, len(a) - 1)] for k in range(-2, 2)]).mean(0)
    W = np.stack([w[np.clip(idx + k, 0, len(w) - 1)] for k in range(-2, 2)]).mean(0)
    v = _interp(tg, d["t"], d["v"])
    X = np.hstack([A, W, np.linalg.norm(v[:, :2], axis=1, keepdims=True), v[:, 2:3]])
    return tg, X


def hankel_sig(lib, Xn, s):
    seg = Xn[s:s + WIN]
    cols = WIN - LAGS + 1
    H = np.vstack([seg[k:k + cols].T for k in range(LAGS)])          # 64 x cols
    U = np.linalg.svd(H, full_matrices=False)[0][:, :4]
    return rt_core.orth(lib, U)


def feats(Xn, s):
    seg = Xn[s:s + WIN]
    P = np.abs(np.fft.rfft(seg - seg.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(WIN, 1 / FS)
    bands = [P[(f > lo) & (f <= hi)].sum(0) for lo, hi in ((0, 1), (1, 4), (4, 25))]
    return np.concatenate([seg.mean(0), seg.std(0)] + [np.log1p(b) for b in bands])


def labels(d, tg, starts):
    v = _interp(tg, d["t"], d["v"])
    acc = np.linalg.norm(_interp(tg, d["t"], np.gradient(d["v"], d["t"], axis=0)), axis=1)
    w = _interp(tg, d["ti"], d["w"])
    sp, wz, wn = np.linalg.norm(v, axis=1), np.abs(w[:, 2]), np.linalg.norm(w, axis=1)
    m = lambda x, s: x[s:s + WIN].mean()
    S = np.array([[m(sp, s), m(wz, s), m(wn, s), m(acc, s)] for s in starts])
    q = lambda k, p: np.quantile(S[:, k], p)
    lab = np.full(len(starts), "прочее", dtype=object)
    lab[(S[:, 0] > q(0, 0.6)) & (S[:, 1] < q(1, 0.5))] = "пролёт"
    lab[S[:, 1] > q(1, 0.8)] = "разворот"
    lab[S[:, 3] > q(3, 0.85)] = "агрессивный"
    lab[(S[:, 0] < q(0, 0.15)) & (S[:, 2] < q(2, 0.3))] = "висение"
    return lab


def task_b(lib, seqs):
    print("\n=== B. Gr(4, 64): сравнение сегментов движения (Ганкель 8 каналов x 8 задержек, окно 2 с) ===")
    sigs, F, lab, tm, sq = [], [], [], [], []
    per = {}
    for qi, d in enumerate(seqs):
        tg, X = channels(d)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xn = (X - mu) / sd
        starts = np.arange(0, len(tg) - WIN, int(0.5 * FS))
        L = labels(d, tg, starts)
        per[d["name"]] = (tg, Xn, mu, sd, starts)
        for s, l in zip(starts, L):
            sigs.append(hankel_sig(lib, Xn, s))
            F.append(feats(Xn, s))
            lab.append(l)
            tm.append(tg[s])
            sq.append(qi)
    F = np.array(F)
    f_mu, f_sd = F.mean(0), F.std(0) + 1e-9
    F = (F - f_mu) / f_sd
    fz = lambda x: (x - f_mu) / f_sd                       # та же нормировка для копий
    lab, tm, sq = np.array(lab), np.array(tm), np.array(sq)
    n = len(sigs)
    DG = np.array([[rt_core.chordal2(lib, sigs[i], sigs[j]) for j in range(n)] for i in range(n)])
    DF = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    print("Сегментов: %d (%s)" % (n, ", ".join("%s %d" % (c, np.sum(lab == c)) for c in np.unique(lab))))

    # 1. сдвиг во времени
    d0 = seqs[0]
    tg, Xn, mu, sd, starts = per[d0["name"]]
    rng = np.random.default_rng(3)
    pick = rng.choice(starts[starts < len(tg) - WIN - int(FS)], 60, replace=False)
    print("\n1. Сдвиг во времени: d^2 сегмента до себя со сдвигом против до случайного другого (медианы)")
    other = np.median([DG[i, j] for i in range(min(n, 200)) for j in rng.choice(n, 5)])
    otherF = np.median([DF[i, j] for i in range(min(n, 200)) for j in rng.choice(n, 5)])
    for ds in (0.1, 0.25, 0.5, 1.0):
        k = int(round(ds * FS))
        g = np.median([rt_core.chordal2(lib, hankel_sig(lib, Xn, s), hankel_sig(lib, Xn, s + k)) for s in pick])
        f = np.median([np.linalg.norm(fz(feats(Xn, s)) - fz(feats(Xn, s + k))) for s in pick])
        print("   сдвиг %.2f с: Gr d^2 %.3f (до другого %.3f, отношение %.2f) | признаки %.2f (до другого %.2f, отношение %.2f)"
              % (ds, g, other, g / other, f, otherF, f / otherF))

    # 2. классы
    cls = [c for c in ("висение", "пролёт", "разворот", "агрессивный") if np.sum(lab == c) >= 5]
    print("\n2. Средние d^2 Gr(4,64) класс x класс (диагональ - однотипные):")
    print("   %-12s" % "" + "".join("%12s" % c for c in cls))
    for a in cls:
        row = []
        for b in cls:
            M = DG[np.ix_(lab == a, lab == b)]
            if a == b:
                M = M[~np.eye(len(M), dtype=bool)]
            row.append(M.mean())
        print("   %-12s" % a + "".join("%12.3f" % x for x in row))

    def knn(D):
        ok = tot = 0
        for i in range(n):
            if lab[i] not in cls:
                continue
            far = (sq != sq[i]) | (np.abs(tm - tm[i]) > 3.0)
            far &= np.isin(lab, cls)
            if not far.any():
                continue
            j = np.argmin(np.where(far, D[i], np.inf))
            ok += lab[j] == lab[i]
            tot += 1
        return 100.0 * ok / max(tot, 1), tot
    (ag, nt), (af, _) = knn(DG), knn(DF)
    base = 100.0 * max(np.mean(lab[np.isin(lab, cls)] == c) for c in cls)
    print("   1-NN (сосед не ближе 3 с по времени): Gr(4,64) %.1f %%, признаки %.1f %%, доля крупнейшего класса %.1f %% (n = %d)"
          % (ag, af, base, nt))

    # 3. нарушение связности осей
    print("\n3. Нарушение связности: расстояние до испорченной копии / медиана расстояния до однотипных")
    for name, kw in (("задержка оси x акселерометра 20 мс", dict(delay_axis=0, delay_s=0.02)),
                     ("задержка оси x акселерометра 50 мс", dict(delay_axis=0, delay_s=0.05)),
                     ("обмен осей x и y (IMU)", dict(swap_xy=True))):
        tgc, Xc = channels(d0, **kw)
        Xcn = (Xc - mu) / sd
        rg, rf = [], []
        idx0 = np.where(sq == 0)[0]
        for ii, s in enumerate(starts[:len(idx0)]):
            i = idx0[ii]
            if lab[i] not in cls:
                continue
            same = np.where((lab == lab[i]) & (np.abs(tm - tm[i]) > 3.0))[0]
            if len(same) < 3:
                continue
            g = rt_core.chordal2(lib, sigs[i], hankel_sig(lib, Xcn, s))
            rg.append(g / np.median(DG[i, same]))
            rf.append(np.linalg.norm(fz(feats(Xcn, s)) - F[i]) / np.median(DF[i, same]))
        print("   %-38s Gr: медиана %.3f, > 1 у %.0f %% | признаки: медиана %.3f, > 1 у %.0f %%"
              % (name, np.median(rg), 100 * np.mean(np.array(rg) > 1), np.median(rf), 100 * np.mean(np.array(rf) > 1)))


def main():
    lib = rt_core.load()
    seqs = [load(p) for p in sys.argv[1:]]
    for d in seqs:
        print("%s: %.0f с, истина %d отсчётов (%.0f Гц), IMU %d (%.0f Гц), скорость до %.2f м/с, |w| до %.2f рад/с"
              % (d["name"], d["t"][-1], len(d["t"]), 1 / np.median(np.diff(d["t"])), len(d["ti"]),
                 1 / np.median(np.diff(d["ti"])), np.linalg.norm(d["v"], axis=1).max(),
                 np.linalg.norm(d["w"], axis=1).max()))
    task_a(lib, seqs)
    task_b(lib, seqs)


if __name__ == "__main__":
    main()
