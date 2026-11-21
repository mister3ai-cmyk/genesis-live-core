# Business Source License 1.1 - see research/LICENSE
"""stats.py - retrieval metrics per question, paired bootstrap, exact McNemar.
2026-11-21 · 4 functions. Everything is aggregated per question (one value per
question per method), and the tests pair methods by question.
"""
import math

import numpy as np

KS = (1, 5, 10)


def metrics(scores, rel):
    """scores (n,), rel bool (n,) -> dict of recall_any@K, recall_all@K, ndcg@10, mrr."""
    order = np.argsort(-scores, kind="stable")
    r = rel[order]
    n_rel = int(rel.sum())
    out = {}
    for k in KS:
        hit = int(r[:k].sum())
        out["recall_any@%d" % k] = float(hit > 0)
        out["recall_all@%d" % k] = float(hit == n_rel)
    top = r[:10]
    gains = top / np.log2(np.arange(2, 2 + len(top)))
    ideal = (1.0 / np.log2(np.arange(2, 2 + min(n_rel, 10)))).sum()
    out["ndcg@10"] = float(gains.sum() / ideal) if ideal > 0 else 0.0
    first = np.flatnonzero(r)
    out["mrr"] = float(1.0 / (first[0] + 1)) if len(first) else 0.0
    return out


def session_scores(scores, session_idx, n_sessions):
    """Session score = max over its turns."""
    s = np.full(n_sessions, -np.inf)
    np.maximum.at(s, session_idx, scores)
    return s


def paired_bootstrap(a, b, n=10000, seed=0):
    """Mean of a - b over questions with a 95 % percentile CI (paired resampling)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mcnemar_exact(a_hit, b_hit):
    """Two-sided exact McNemar on binary hits; returns (b01, b10, p)."""
    a, b = np.asarray(a_hit, bool), np.asarray(b_hit, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    m = n01 + n10
    if m == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return n01, n10, min(1.0, 2 * p)
