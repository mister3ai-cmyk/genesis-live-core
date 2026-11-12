# Business Source License 1.1 - см. research/LICENSE
"""Сверка C-ядра с Python-эталоном (research/swarm: grassmann.py,
kinematics.py) на одних данных. Пропускается, если librt_core.so не собран
или не грузится (Windows): make -C research/rt_core all."""
import os
import random
import sys

import pytest

np = pytest.importorskip("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
SWARM = os.path.join(os.path.dirname(RT), "swarm")
for p in (RT, SWARM):
    if p not in sys.path:
        sys.path.insert(0, p)

import rt_core                                               # noqa: E402

try:
    LIB = rt_core.load()
except OSError:
    pytest.skip("librt_core.so не собран или не грузится на этой платформе", allow_module_level=True)

import grassmann as gm                                       # noqa: E402
import kinematics as kin                                     # noqa: E402

pytestmark = pytest.mark.concept


def test_orth_and_chordal_match_numpy_reference():
    rng = np.random.default_rng(0)
    for n in (64, 128):
        for _ in range(20):
            Q = rt_core.orth(LIB, rng.normal(size=(n, 4)))
            assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-12)
            Qn = gm.orth(rng.normal(size=(n, 4)))
            V = rt_core.orth(LIB, rng.normal(size=(n, 4)))
            assert abs(rt_core.chordal2(LIB, Q, V) - gm.chordal2(Q, V)) < 1e-12
            th = gm.principal_angles(Q, Qn)
            assert abs(rt_core.chordal2(LIB, Q, Qn) - np.sum(np.sin(th) ** 2)) < 1e-10


@pytest.mark.parametrize("q_back_ms", [0, 1500], ids=["query-now", "late-delivery"])
def test_predictor_matches_python_reference_with_and_without_liar(q_back_ms):
    """2D-данные (z = 0): C-предиктор против kinematics.predict. late-delivery:
    запрос на 1.5 с раньше последних точек - часть кольца «в будущем»."""
    rng = random.Random(1)
    n_cmp = n_none = 0
    for trial in range(300):
        v = (rng.uniform(-600, 600), rng.uniform(-600, 600))
        acc = (rng.uniform(-80, 80), rng.uniform(-80, 80))
        p0 = (rng.uniform(20_000, 180_000), rng.uniform(20_000, 180_000))
        liar = trial % 2 == 1
        T = 10_000                                            # мс
        claims = []
        for j in range(8):                                    # раунды по 4 автора, по времени
            for a in range(4):
                tc = T - 4000 + 450 * j + 97 * a + rng.randint(0, 40)
                if tc >= T:
                    continue
                s = (tc - 6000) / 1000.0
                x = p0[0] + v[0] * s + acc[0] * s * s + rng.randint(-50, 50)
                y = p0[1] + v[1] * s + acc[1] * s * s + rng.randint(-50, 50)
                if liar and a == 0:
                    x, y = x + 1100, y - 900
                claims.append((tc, a, (x, y)))
        claims.sort(key=lambda c: c[0])
        c = rt_core.Kin(LIB)
        for tc, a, p in claims:
            assert c.push(0, a, tc * 1000, p) == rt_core.OK
        Tq = T - q_back_ms
        ref = kin.predict(claims, Tq)
        got = c.predict(0, Tq * 1000)
        if ref is None:
            n_none += 1
            assert got is None
            continue
        n_cmp += 1
        assert got is not None
        assert abs(got[0] - ref[0]) < 1e-6 and abs(got[1] - ref[1]) < 1e-6 and got[2] == 0.0
    print("\nсверка предиктора: %d совпали, %d - прогноза нет у обоих" % (n_cmp, n_none))
    assert n_cmp >= 200


def test_gate_verdicts_through_ctypes():
    c = rt_core.Kin(LIB)
    for j in range(4):
        for a in range(4):
            t = 0.2 + 0.23 * a + 0.9 * j
            c.push(0, a, t * 1e6, (10_000 + 400 * t, 20_000 - 250 * t, 500 + 30 * t))
    T = 4_000_000
    mine = (10_000 + 1600, 20_000 - 1000, 620)
    v, r = c.gate(0, 3, T, T + 1000, (mine[0] + 1100, mine[1] - 900, mine[2]), mine)
    assert v == rt_core.REFUTE and r > 800
    assert c.gate(0, 3, T, T + 1000, (mine[0] + 30, mine[1], mine[2]), mine)[0] == rt_core.PASS
    assert c.gate(0, 3, T, T - 100_000, (mine[0] + 1100, mine[1], mine[2]), mine)[0] == rt_core.FAILSAFE
