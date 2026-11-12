# Business Source License 1.1 - см. research/LICENSE
"""S5.2: rt_kin_predict без копии окна (стек 4.6 КБ -> < 1.5 КБ) обязан
давать БИТ В БИТ то же, что версия S5.1 (main 3a7eb41).

Детерминированные потоки (random.Random, без numpy): 8 авторов, лжецы,
грубые сбои, вытеснения из кольца, перебазирование (> 16 с), забывание
автора, исключение автора в прогнозе; горизонты 0.5 и 4 с. Каждый ответ
predict и gate упаковывается в байты, сравнивается SHA-256 с эталоном,
снятым на версии S5.1. Если задан RT_REF_LIB (путь к librt_core.so
версии S5.1) - ещё и поэлементное сравнение с ней."""
import hashlib
import os
import random
import struct
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
if RT not in sys.path:
    sys.path.insert(0, RT)

import rt_core                                               # noqa: E402

try:
    LIB = rt_core.load()
except OSError:
    pytest.skip("librt_core.so не собран или не грузится на этой платформе", allow_module_level=True)

pytestmark = pytest.mark.concept

# SHA-256 ответов версии S5.1 (main 3a7eb41), см. _records()
REF_DIGEST = "38bde2efdd9f66907b4449f79d17ccf249a4c88e7e84325a15b59f3a57e0b319"
N_STREAMS, N_STEPS = 120, 160


def _records(lib):
    rng = random.Random(52)
    out = []
    for k in range(N_STREAMS):
        horizon = 500_000 if k % 2 else 4_000_000
        tol = 0.4 if k % 2 else 400.0
        kin = rt_core.Kin(lib, horizon_us=horizon, max_skew_us=50_000, drop_tol=tol * 0.7,
                          thr=tol * 1.4, pos_tol=tol * 0.7, min_pts=4)
        n_auth = rng.randint(2, 8)
        liars = set(rng.sample(range(n_auth), rng.randint(0, max(0, n_auth // 3))))
        c = [rng.uniform(-50, 50) for _ in range(9)]
        tr = lambda t: tuple(c[3 * a] + c[3 * a + 1] * t + c[3 * a + 2] * t * t * 0.1 for a in range(3))
        t = rng.randint(0, 10**9)
        dt = rng.choice((20_000, 40_000, 150_000, 400_000))   # 400 мс: > 16 с за 40 шагов
        for step in range(N_STEPS):
            t += rng.randint(dt // 2, dt * 3 // 2)
            a = rng.randrange(n_auth)
            ts = (t - 7 * 10**8) / 1e6
            x = list(tr(ts))
            noise = tol * 0.05
            x = [v + rng.gauss(0, noise) for v in x]
            if a in liars and step > N_STEPS // 3:
                x[rng.randrange(3)] += rng.choice((-1, 1)) * tol * rng.uniform(1.5, 4)
            if rng.random() < 0.03:
                x[rng.randrange(3)] += tol * 10
            kin.push(0, a, t, x, t + 1000)
            if rng.random() < 0.01:
                out.append(struct.pack(">i", lib.rt_kin_forget_author(kin.buf, 0, rng.randrange(n_auth))))
            tq = t + rng.randint(-dt, dt)
            ex = rng.choice((-1, -1, rng.randrange(n_auth)))
            p = kin.predict(0, tq, exclude=ex)
            out.append(b"N" if p is None else struct.pack(">ddd", *p))
            claim = [v + rng.gauss(0, tol * 0.3) for v in tr((tq - 7 * 10**8) / 1e6)]
            mine = [v + rng.gauss(0, noise) for v in tr((tq - 7 * 10**8) / 1e6)]
            v, res = kin.gate(0, rng.randrange(n_auth), tq, tq + 2000, claim, mine)
            out.append(struct.pack(">id", v, res))
    return out


def test_predict_and_gate_bit_identical_to_s51():
    recs = _records(LIB)
    n_pred = sum(r != b"N" and len(r) == 24 for r in recs)
    assert n_pred > N_STREAMS * N_STEPS // 2                  # прогнозы реально считались
    ref = os.environ.get("RT_REF_LIB")
    if ref:
        rec_ref = _records(rt_core.load(ref))
        diff = [i for i, (x, y) in enumerate(zip(recs, rec_ref)) if x != y]
        assert len(recs) == len(rec_ref) and diff == [], diff[:5]
    assert hashlib.sha256(b"".join(recs)).hexdigest() == REF_DIGEST
