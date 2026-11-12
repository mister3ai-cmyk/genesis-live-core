# Business Source License 1.1 - см. research/LICENSE
"""Этап B v4.0: калибровка кинематического шлюза вне сети.
Окно: сетка dt = 1 с, 8 точек; точки 0..6 - интерполяция заявок 4 других авторов
(шум +-50, ~0.5 заявки/с на автора), точка 7 - проверяемая x.
Статистики: Ганкель S_H = d2(Y_prev, Y_x); линейная r1 = |x - (2w6 - w5)|;
квадратичная r2 = |x - (3w6 - 3w5 + w4)|.
argv: папка swarm"""
import math, random, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
import grassmann as gm
import swarm_p2p_simulation as sim
from hankel_grassmann import HankelTrajectoryEmbedder, THETA

DT, W = 1000, 8
N_TRACKS = 150
RATE = 0.5


def truth_fn(tt, vv):
    T = np.array(tt, float); V = np.array(vv, float)
    return lambda t: np.array([np.interp(t, T, V[:, 0]), np.interp(t, T, V[:, 1])])


def claims(rng, tr, t0, t1, liar=False):
    out, t = [], t0
    while True:
        t += rng.expovariate(RATE) * 1000
        if t > t1:
            return out
        p = tr(t) + np.array([rng.randint(-50, 50), rng.randint(-50, 50)])
        if liar:
            p = p + np.array([rng.choice((-1, 1)) * rng.randint(600, 1500) for _ in (0, 1)])
        out.append((t, p))


def grid(cl, times):
    cl = sorted(cl, key=lambda c: c[0])
    ts = np.array([c[0] for c in cl]); P = np.array([c[1] for c in cl])
    return np.stack([np.interp(times, ts, P[:, 0]), np.interp(times, ts, P[:, 1])], axis=1)


def stats_for(emb, authors, t, x):
    """authors: список списков заявок. Возвращает S_H (медиана Кархера по окнам
    без одного автора, если авторов > 2), r1, r2 (медиана по тем же окнам)."""
    g_prev = t - DT * np.arange(W, 0, -1)          # t-8..t-1
    g_cur = t - DT * np.arange(W - 1, 0, -1)       # t-7..t-1, + x
    sets = [sum(authors[:i] + authors[i + 1:], []) for i in range(len(authors))] if len(authors) > 2 else [sum(authors, [])]
    Yp, Yx, r1, r2 = [], [], [], []
    for cl in sets:
        cl = [c for c in cl if c[0] <= t]
        wp = grid(cl, g_prev)
        wc = np.vstack([grid(cl, g_cur), x])
        Yp.append(emb(wp)); Yx.append(emb(wc))
        w = wc[:-1]
        r1.append(np.linalg.norm(x - (2 * w[-1] - w[-2])))
        r2.append(np.linalg.norm(x - (3 * w[-1] - 3 * w[-2] + w[-3])))
    if len(sets) > 1:
        yp, yx = gm.karcher_mean(Yp, median=True), gm.karcher_mean(Yx, median=True)
    else:
        yp, yx = Yp[0], Yx[0]
    return gm.chordal2(yp, yx), float(np.median(r1)), float(np.median(r2))


def best_split(h, a):
    """Порог с минимумом FP+FN (FP - честные выше порога, FN - атаки ниже)."""
    vals = np.sort(np.concatenate([h, a]))
    best = (1e9, None)
    for v in vals:
        e = np.sum(h >= v) + np.sum(a < v)
        if e < best[0]:
            best = (e, v)
    thr = best[1]
    return thr, 100 * np.mean(h >= thr), 100 * np.mean(a < thr)


def run(scale, contaminate):
    emb = HankelTrajectoryEmbedder(window_size=W, lag=4, scale=scale)
    rng = random.Random(7)
    print("\n=== scale %d, %s ===" % (scale, "1 из 4 других авторов - лжец T8" if contaminate else "другие авторы честные"))
    print("%-9s %-4s | %-30s | %-30s | %-30s" % ("режим", "", "Ганкель d2 (THETA 0.3)", "линейная |x-(2w6-w5)|", "квадратичная"))
    tot = {k: {"h": [], "a": [], "g": []} for k in ("H", "r1", "r2")}
    for mode in sim.KIN_MODES:
        res = {k: {"h": [], "a": [], "g": []} for k in ("H", "r1", "r2")}
        for _ in range(N_TRACKS):
            tt, vv, _ = sim.kinematic_track(rng, 30_000, modes=(mode,))
            tr = truth_fn(tt, vv)
            t = rng.uniform(12_000, 26_000)
            authors = [claims(rng, tr, 0, t, liar=(contaminate and i == 0)) for i in range(4)]
            if any(len([c for c in a if c[0] <= t]) < 2 for a in authors):
                continue
            truth = tr(t)
            xs = {"h": truth + np.array([rng.randint(-50, 50), rng.randint(-50, 50)]),
                  "a": truth + np.array([rng.choice((-1, 1)) * rng.randint(600, 1500) for _ in (0, 1)]),
                  "g": truth + np.array([rng.choice((-1, 1)) * rng.randint(3000, 5000) for _ in (0, 1)])}
            for kx, x in xs.items():
                sH, s1, s2 = stats_for(emb, authors, t, x)
                res["H"][kx].append(sH); res["r1"][kx].append(s1); res["r2"][kx].append(s2)
        line = []
        for k in ("H", "r1", "r2"):
            h, a = np.array(res[k]["h"]), np.array(res[k]["a"])
            thr, fp, fn = best_split(h, a)
            extra = ""
            if k == "H":
                extra = " @0.3 FP%.0f FN%.0f" % (100 * np.mean(h >= THETA), 100 * np.mean(a < THETA))
            line.append("ч.max %.3g ат.min %.3g FP%.0f FN%.0f%s" % (h.max(), a.min(), fp, fn, extra) if k == "H"
                        else "ч.max %.0f ат.min %.0f FP%.0f FN%.0f" % (h.max(), a.min(), fp, fn))
            for kx in ("h", "a", "g"):
                tot[k][kx] += res[k][kx]
        print("%-9s %-4d | %s" % (mode, len(res["H"]["h"]), " | ".join(line)))
    print("  итого, лучший порог по всем режимам сразу:")
    for k, name in (("H", "Ганкель"), ("r1", "линейная"), ("r2", "квадратичная")):
        h, a, g = (np.array(tot[k][x]) for x in ("h", "a", "g"))
        thr, fp, fn = best_split(h, a)
        print("   %-12s порог %.3g: FP %.1f %%, FN %.1f %%; честн. p95 %.3g, атака p5 %.3g, грубый сбой p5 %.3g"
              % (name, thr, fp, fn, np.percentile(h, 95), np.percentile(a, 5), np.percentile(g, 5)))
    hh, aa = np.array(tot["H"]["h"]), np.array(tot["H"]["a"])
    print("   Ганкель при THETA=0.3: FP %.1f %%, FN %.1f %%" % (100 * np.mean(hh >= THETA), 100 * np.mean(aa < THETA)))


def timing():
    emb = HankelTrajectoryEmbedder(window_size=W, lag=4, scale=1000)
    rng = random.Random(1)
    tt, vv, _ = sim.kinematic_track(rng, 30_000)
    tr = truth_fn(tt, vv)
    authors = [claims(rng, tr, 0, 20_000) for _ in range(4)]
    x = tr(20_000)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        stats_for(emb, authors, 20_000, x)
    t1 = time.perf_counter()
    w = grid(sum(authors, []), 20_000 - DT * np.arange(7, 0, -1))
    t2 = time.perf_counter()
    for _ in range(100_000):
        np.linalg.norm(x - (2 * w[-1] - w[-2]))
    t3 = time.perf_counter()
    print("\nВремя: полная проверка (4 окна без одного автора x 2 SVD + 2 медианы Кархера + линейные) %.0f мкс; "
          "одна линейная невязка %.2f мкс" % ((t1 - t0) / n * 1e6, (t3 - t2) / 100_000 * 1e6))


for sc in (500, 1000, 2000):
    run(sc, False)
run(1000, True)
timing()
