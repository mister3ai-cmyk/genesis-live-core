# Business Source License 1.1 - see research/LICENSE
"""test_encoder.py - synthetic checks of encoder_v1 and stats. 2026-11-21 · 5 tests.
    python -m pytest -q test_encoder.py
"""
import numpy as np

import encoder_v1 as enc
import stats as st


def test_g1_equals_projection_energy():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((200, 4, 128))
    F /= np.linalg.norm(F, axis=2, keepdims=True)
    Y, degen, err = enc.subspaces(F, np.arange(200))
    assert err <= 1e-5 and not degen.any()
    q = rng.standard_normal(128)
    q /= np.linalg.norm(q)
    g1, _ = enc.g1_scores(Y, q.astype(np.float32))
    for i in range(200):                                   # ||P q||^2 with P = projector onto span(F_i)
        Qm, _ = np.linalg.qr(F[i].T)
        ref = np.linalg.norm(Qm.T @ q) ** 2
        assert abs(g1[i] - ref) < 1e-5


def test_degenerate_faces_replaced_and_orthonormal():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(128)
    F = np.stack([np.stack([v, v, rng.standard_normal(128), np.zeros(128)])])
    Y, degen, err = enc.subspaces(F, np.array([7]))
    assert degen[0] and err <= 1e-5
    Y2, _, _ = enc.subspaces(F, np.array([7]))
    assert np.array_equal(Y, Y2)                           # deterministic (seed = node id)


def test_metrics_basic():
    s = np.array([0.9, 0.1, 0.8, 0.2])
    rel = np.array([False, False, True, True])
    m = st.metrics(s, rel)
    assert m["recall_any@1"] == 0 and m["recall_any@5"] == 1 and m["recall_all@5"] == 1
    assert abs(m["mrr"] - 0.5) < 1e-12


def test_bootstrap_and_mcnemar():
    a = np.array([1, 1, 1, 0, 1, 1, 0, 1] * 20, float)
    b = np.array([1, 0, 1, 0, 1, 0, 0, 1] * 20, float)
    d, lo, hi = st.paired_bootstrap(a, b, n=2000)
    assert lo > 0 and abs(d - 0.25) < 1e-12
    n01, n10, p = st.mcnemar_exact(a > 0, b > 0)
    assert (n01, n10) == (0, 40) and p < 1e-6


def test_prep_unit_norm_and_projection():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((1000, 384)) + 3.0
    pr = enc.Prep(X, d=128)
    Z = pr(X[:10])
    assert Z.shape == (10, 128) and np.allclose(np.linalg.norm(Z, axis=1), 1.0)
    assert pr(X[:3]).shape == (3, 128) and enc.Prep(X, d=384).P is None
