# Business Source License 1.1 - см. research/LICENSE
"""v4.0, разведка: геометрия Gr(4, 128) и честный замер обратимости 256-битных
подписей. Числа ошибок печатаются (pytest -s); проверяется только то, что
верно по построению."""
import os
import sys

import pytest

np = pytest.importorskip("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM = os.path.dirname(HERE)
if SWARM not in sys.path:
    sys.path.insert(0, SWARM)

import grassmann as gm                                       # noqa: E402

pytestmark = pytest.mark.concept


def _rand_Y(rng, k=4, d=128):
    return gm.orth(rng.normal(size=(d, k)))


# --- геометрия ------------------------------------------------------------

def test_orth_projector_and_trace():
    rng = np.random.default_rng(0)
    Y = _rand_Y(rng)
    assert np.allclose(Y.T @ Y, np.eye(4), atol=1e-12)
    P = gm.proj(Y)
    assert P.shape == (128, 128)
    assert np.allclose(P @ P, P, atol=1e-12)
    assert abs(np.trace(P) - 4) < 1e-12


def test_chordal_equals_sum_sin2_and_basis_invariant():
    rng = np.random.default_rng(1)
    for _ in range(50):
        Ya, Yb = _rand_Y(rng), _rand_Y(rng)
        d2 = gm.chordal2(Ya, Yb)
        th = gm.principal_angles(Ya, Yb)
        assert abs(d2 - np.sum(np.sin(th) ** 2)) < 1e-10
        assert abs(d2 - (4 - np.trace(gm.proj(Ya) @ gm.proj(Yb)))) < 1e-9
        Q = gm.orth(rng.normal(size=(4, 4)))           # Y -> Y Q, Q в O(4)
        assert abs(gm.chordal2(Ya @ Q, Yb) - d2) < 1e-10
        assert 0 <= d2 <= 4 + 1e-12


def test_chordal_extremes():
    E = np.eye(128)
    assert abs(gm.chordal2(E[:, :4], E[:, :4])) < 1e-12
    assert abs(gm.chordal2(E[:, :4], E[:, 4:8]) - 4) < 1e-12   # максимум d^2 = 4, d = 2


# --- шум и ложь в embed_point ---------------------------------------------

def test_noise_small_lie_large_and_print():
    emb = gm.Embed(scale=500.0, seed=3)
    rng = np.random.default_rng(4)
    noise, lie = [], []
    for _ in range(200):
        p = rng.uniform(0, gm.FIELD, size=2)
        Y = emb(p)
        noise.append(gm.chordal2(Y, emb(p + rng.integers(-50, 51, size=2))))
        sh = rng.integers(3000, 5001, size=2) * rng.choice((-1, 1), size=2)
        lie.append(gm.chordal2(Y, emb(p + sh)))
    print("\nembed scale 500: d^2 шум +-50 медиана %.4f p95 %.4f | ложь 3000-5000 мин %.3f медиана %.3f"
          % (np.median(noise), np.percentile(noise, 95), min(lie), np.median(lie)))
    assert np.percentile(noise, 95) < min(lie)


def test_srp_hamming_tracks_chordal():
    emb, srp = gm.Embed(scale=500.0, seed=3), gm.SRP(seed=5)
    rng = np.random.default_rng(6)
    p = rng.uniform(0, gm.FIELD, size=2)
    Y, s0 = emb(p), srp(emb(p))
    d2, h = [], []
    for r in np.linspace(0, 500, 60):               # в пределах масштаба: дальше d^2 насыщен
        a = rng.uniform(0, 2 * np.pi)
        q = p + r * np.array([np.cos(a), np.sin(a)])
        d2.append(gm.chordal2(Y, emb(q)))
        h.append(gm.hamming(s0, srp(emb(q))))
    rd, rh = np.argsort(np.argsort(d2)), np.argsort(np.argsort(h))
    rho = np.corrcoef(rd, rh)[0, 1]
    print("\nSRP-256 по проектору: Спирмен(d^2, Хэмминг) = %.3f" % rho)
    assert rho > 0.9


def test_srp_does_not_depend_on_basis_choice():
    srp, rng = gm.SRP(seed=5), np.random.default_rng(7)
    Y = _rand_Y(rng)
    Q = gm.orth(rng.normal(size=(4, 4)))
    assert gm.hamming(srp(Y), srp(Y @ Q)) == 0


# --- обратимость: а) кодбук 512, б) SRP-256 через Gr + кодбук, в) прямой SRP

def test_reversibility_table():
    rng = np.random.default_rng(8)
    pts = np.vstack([[[1500, 2300]], rng.uniform(0, gm.FIELD, size=(999, 2))])
    centers, cell = gm.grid_centers()
    half_diag = 0.5 * np.hypot(*cell)
    err_a = np.array([np.linalg.norm(centers[gm.nearest_center(p, centers)] - p) for p in pts])
    assert err_a.max() <= half_diag + 1e-9           # по построению

    rows = []
    for scale in (500.0, 8000.0):
        emb, srp = gm.Embed(scale=scale, seed=3), gm.SRP(seed=5)
        cb = gm.Codebook(emb, srp)
        err_b = np.array([np.linalg.norm(cb.decode(srp(emb(p))) - p) for p in pts[:300]])
        rows.append((scale, err_b))
        assert any(np.allclose(cb.decode(srp(emb(p))), c) for c in centers for p in pts[:1])

    ps = gm.PointSRP(seed=9)
    err_c, cons = [], 0
    for p in pts:
        b = ps(p)
        x = ps.decode(b)
        cons += bool(np.all(ps(x) == b))
        err_c.append(np.linalg.norm(x - p))
    err_c = np.array(err_c)

    def line(name, e):
        return "  %-46s медиана %8.0f  p95 %8.0f  макс %8.0f  в пределах +-50: %5.1f %%" % (
            name, np.median(e), np.percentile(e, 95), e.max(), 100 * np.mean(e <= 50 * np.sqrt(2)))
    print("\nОбратимость на поле %d x %d (ячейка кодбука %.0f x %.0f):" % (gm.FIELD, gm.FIELD, *cell))
    print(line("а) индекс кодбука 512 (9 бит, идеальный)", err_a))
    for scale, e in rows:
        print(line("б) SRP-256 через Gr(4,128) -> кодбук, scale %d" % scale, e))
    print(line("в) прямой SRP-256 точки со сдвигами", err_c))
    print("  точка [1500, -2300] -> поле сдвинуто в [1500, 2300]: а) %.0f  в) %.0f" % (err_a[0], err_c[0]))
    print("  в) декодер согласован со всеми 256 битами: %d из %d" % (cons, len(pts)))
    assert cons >= 0.95 * len(pts)


# --- слияние при выпадении источников ---------------------------------------

def test_karcher_mean_with_two_of_five_dropped_and_median_with_liar():
    emb, rng = gm.Embed(scale=500.0, seed=3), np.random.default_rng(10)
    truth = np.array([100_000.0, 80_000.0])
    Yt = emb(truth)
    obs = [emb(truth + rng.integers(-50, 51, size=2)) for _ in range(5)]
    keep = [obs[i] for i in sorted(rng.choice(5, size=3, replace=False))]
    Ym = gm.karcher_mean(keep)
    d_mean = gm.chordal2(Ym, Yt)
    d_single = np.median([gm.chordal2(Y, Yt) for Y in keep])
    liar = emb(truth + np.array([4000.0, -3500.0]))
    Ymed = gm.karcher_mean(obs[:4] + [liar], median=True)
    Yavg = gm.karcher_mean(obs[:4] + [liar])
    print("\nКархер: 3 из 5 честных d^2 до истины %.5f (одиночный %.5f); с лжецом 1 из 5: медиана %.5f, среднее %.5f"
          % (d_mean, d_single, gm.chordal2(Ymed, Yt), gm.chordal2(Yavg, Yt)))
    assert d_mean <= d_single + 1e-9
    assert gm.chordal2(Ymed, Yt) < gm.chordal2(Yavg, Yt)
