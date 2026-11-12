# Business Source License 1.1 - см. research/LICENSE
"""Калибровка kinematics.predict: невязки честной заявки, T8, грубого сбоя
своего датчика; окно 4 других автора по 0.5 заявки/с, шум +-50; без лжеца и с
лжецом среди 4. argv: папка swarm"""
import random, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
import swarm_p2p_simulation as sim
import kinematics as kin


def run(liar, jitter_ms):
    rng = random.Random(11)
    res = {m: {"h": [], "a": [], "g": [], "none": 0} for m in sim.KIN_MODES}
    for mode in sim.KIN_MODES:
        for _ in range(300):
            tt, vv, _ = sim.kinematic_track(rng, 30_000, modes=(mode,))
            T = np.array(tt, float); V = np.array(vv, float)
            tr = lambda t: np.array([np.interp(t, T, V[:, 0]), np.interp(t, T, V[:, 1])])
            t = rng.uniform(8_000, 26_000)
            cl = []
            for a in range(4):
                tc = t - 6000
                while True:
                    tc += rng.expovariate(0.5) * 1000
                    if tc >= t:
                        break
                    p = tr(tc) + np.array([rng.randint(-50, 50), rng.randint(-50, 50)])
                    if liar and a == 0:
                        p = p + np.array([rng.choice((-1, 1)) * rng.randint(600, 1500) for _ in (0, 1)])
                    cl.append((tc + rng.uniform(-jitter_ms, jitter_ms), a, p))
            pred = kin.predict(cl, t)
            if pred is None:
                res[mode]["none"] += 1
                continue
            x = tr(t)
            res[mode]["h"].append(kin.residual(x + np.array([rng.randint(-50, 50), rng.randint(-50, 50)]), pred))
            res[mode]["a"].append(kin.residual(x + np.array([rng.choice((-1, 1)) * rng.randint(600, 1500) for _ in (0, 1)]), pred))
            res[mode]["g"].append(kin.residual(x + np.array([rng.choice((-1, 1)) * rng.randint(3000, 5000) for _ in (0, 1)]), pred))
    print("\n=== лжец в окне: %s, разброс меток времени +-%d мс ===" % (liar, jitter_ms))
    H, A, G = [], [], []
    for m, r in res.items():
        h, a = np.array(r["h"]), np.array(r["a"])
        H += r["h"]; A += r["a"]; G += r["g"]
        print("  %-8s n=%d нет прогноза %d | честн. p95 %4.0f макс %4.0f | T8 мин %4.0f p5 %4.0f"
              % (m, len(h), r["none"], np.percentile(h, 95), h.max(), a.min(), np.percentile(a, 5)))
    H, A, G = map(np.array, (H, A, G))
    for thr in (500, 600, 700, 800):
        print("  порог %d: ложных тревог %.1f %%, пропусков T8 %.1f %%, свой грубый сбой ниже порога %.1f %%"
              % (thr, 100 * np.mean(H > thr), 100 * np.mean(A <= thr), 100 * np.mean(G <= thr)))


for liar in (False, True):
    for j in (20,):
        run(liar, j)
run(False, 500)

cl = [(i * 400.0, i % 4, (1000 + i * 150, 2000 - i * 90)) for i in range(10)]
n = 20000
t0 = time.perf_counter()
for _ in range(n):
    kin.predict(cl, 4000.0)
print("\npredict: %.1f мкс на вызов (10 заявок, чистый Python)" % ((time.perf_counter() - t0) / n * 1e6))
