#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/reconsolidation_experiment.py

Does memory that changes on read actually retrieve better?

That is the central claim of NGP 4.6 and it has never been measured. The engine
in modules/ngp46_reconsolidation_engine.py updates a unitary phase operator on
every query, and the specification asserts this maintains coherence where a
static index accumulates contradictions. Neither half has a number attached.

The measurement has one trap worth stating. Recall against cosine ground truth
cannot be the metric: any change to W moves retrieval away from plain cosine by
construction, so the adaptive arm would be guaranteed to lose. That would
measure fidelity to the baseline, not usefulness.

So relevance is defined by the workload instead. The corpus is clustered into
topics; a session draws its queries from one topic; a result counts as relevant
if it belongs to the query's topic. An operator that adapts to what it is being
asked should raise topic precision over a session. A static one cannot, and its
flat line is the control that proves the harness is not drifting on its own.

The experiment is falsifiable: if reconsolidation does nothing, the two lines
sit on top of each other.

Usage: python3 reconsolidation_experiment.py [--dim 256] [--topics 20]
Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/tmp/genesis_eval"
K = 10


def load_engine():
    path = os.path.join(HERE, "modules", "ngp46_reconsolidation_engine.py")
    spec = importlib.util.spec_from_file_location("ngp46", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_vectors(dim):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(CACHE, "train-*.parquet")))
    if not files:
        sys.exit("no parquet shards in %s — run benchmarks/eval_real_embeddings.py"
                 % CACHE)
    chunks = []
    for f in files:
        t = pq.read_table(f)
        col = next(c for c in t.column_names
                   if "embedding" in c.lower() or "vector" in c.lower())
        chunks.append(np.vstack(t.column(col).to_numpy(zero_copy_only=False))
                      .astype(np.float32))
    v = np.vstack(chunks)[:, :dim].copy()
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v)


def topic_precision(scores, labels, query_topic, k=K):
    """Share of the top-k that belongs to the topic the query came from."""
    top = np.argpartition(-scores, k)[:k]
    return float(np.mean(labels[top] == query_topic))


def make_rule(kind, corpus, eta, dim):
    """Три правила. Первые два — контроли, третий — исправленный."""
    I = np.eye(dim, dtype=np.complex128)
    from scipy.linalg import expm

    def static(W, q):
        return W

    def ngp46(W, q):
        """Как в modules/: цель = собственный выход. Измерено как тождество."""
        psi = W @ q
        psi = psi / (np.linalg.norm(psi) + 1e-12)
        Wt = W + eta * np.outer(psi, np.conj(q))
        n = np.linalg.norm(Wt - W, "fro")
        if n > 0.05:
            Wt = W + (Wt - W) * 0.05 / n
        return 0.5 * (Wt @ (3.0 * I - np.conj(Wt.T) @ Wt))

    def corrected(W, q):
        """Цель — центр того, что реально достали. Генератор косоэрмитов,
        поэтому унитарность сохраняется без Ньютона-Шульца."""
        psi = W @ q
        psi = psi / (np.linalg.norm(psi) + 1e-12)
        sc = corpus @ np.real(psi).astype(np.float32)
        top = np.argpartition(-sc, K)[:K]
        t = corpus[top].mean(axis=0).astype(np.complex128)
        n = np.linalg.norm(t)
        if n < 1e-9:
            return W
        t = t / n
        A = np.outer(t, np.conj(q)) - np.outer(q, np.conj(t))
        return W @ expm(eta * A)

    def temporal(W, q, nxt=None):
        """Цель — следующий запрос той же сессии. Это единственный источник,
        которого оператор не знает: структуру сессий ему никто не показывал.
        Цель, вычисленная из собственного выхода, информации не несёт."""
        if nxt is None:
            return W
        psi = W @ q
        psi = psi / (np.linalg.norm(psi) + 1e-12)
        t = nxt.astype(np.complex128)
        t = t / (np.linalg.norm(t) + 1e-12)
        A = np.outer(t, np.conj(q)) - np.outer(q, np.conj(t))
        return W @ expm(eta * A)

    return {"static": static, "ngp46": ngp46,
            "corrected": corrected, "temporal": temporal}[kind]


def run_arm(corpus, labels, sessions, kind, eta, dim):
    """Один прогон по одному и тому же потоку запросов."""
    rule = make_rule(kind, corpus, eta, dim)
    I = np.eye(dim, dtype=np.complex128)
    npos = max(len(s[1]) for s in sessions)
    per_position = [[] for _ in range(npos)]
    returned = set()
    W = I.copy()
    t0 = time.perf_counter()
    for topic, q_idx in sessions:
        for pos, qi in enumerate(q_idx):
            q = corpus[qi].astype(np.complex128)
            psi = W @ q
            probe = np.real(psi).astype(np.float32)
            n = np.linalg.norm(probe)
            probe = probe / n if n > 1e-9 else corpus[qi]
            scores = corpus @ probe
            scores[qi] = -np.inf
            top = np.argpartition(-scores, K)[:K]
            returned.update(top.tolist())
            per_position[pos].append(float(np.mean(labels[top] == topic)))
            if kind == "temporal":
                nxt = corpus[q_idx[pos + 1]] if pos + 1 < len(q_idx) else None
                W = rule(W, q, nxt)
            else:
                W = rule(W, q)
    dt = time.perf_counter() - t0
    curve = [float(np.mean(p)) if p else float("nan") for p in per_position]
    return curve, len(returned), float(np.linalg.norm(W - I, "fro")), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--topics", type=int, default=20)
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--per-session", type=int, default=25)
    ap.add_argument("--corpus", type=int, default=20000)
    args = ap.parse_args()

    for m in ("numpy", "pyarrow", "sklearn"):
        try:
            __import__(m)
        except ImportError:
            sys.exit("missing %s — pip install numpy pyarrow scikit-learn" % m)
    from sklearn.cluster import KMeans

    print("loading %d vectors at %d dims..." % (args.corpus, args.dim))
    corpus = load_vectors(args.dim)[:args.corpus]

    print("clustering into %d topics (this defines relevance)..." % args.topics)
    km = KMeans(n_clusters=args.topics, n_init=4, random_state=0)
    labels = km.fit_predict(corpus)

    rng = np.random.default_rng(13)
    sessions = []
    for _ in range(args.sessions):
        topic = int(rng.integers(args.topics))
        members = np.flatnonzero(labels == topic)
        if len(members) < args.per_session:
            continue
        sessions.append((topic, rng.choice(members, args.per_session, replace=False)))
    print("%d sessions x %d queries, drawn from one topic each\n"
          % (len(sessions), args.per_session))

    ngp = load_engine()
    engine = ngp.NGP46ReconsolidationEngineHPC(
        dim=args.dim, estimator=ngp.HeuristicBurstEstimator())

    n_q = len(sessions) * args.per_session
    results = {}
    for kind, eta in (("static", 0.0), ("ngp46", 0.015),
                      ("corrected", 0.02),
                      ("temporal", 0.005), ("temporal", 0.02), ("temporal", 0.05)):
        key = kind if eta == 0.0 else "%s eta=%.3f" % (kind, eta)
        curve, uniq, drift, dt = run_arm(corpus, labels, sessions, kind, eta, args.dim)
        results[key] = (curve, uniq, drift, dt)

    fifth = max(1, args.per_session // 5)
    print("%-20s %9s %9s %9s %10s %9s" %
          ("правило", "начало", "конец", "прирост", "уник.док", "||W-I||"))
    print("-" * 72)
    for key, (curve, uniq, drift, dt) in results.items():
        a = float(np.mean(curve[:fifth]))
        b = float(np.mean(curve[-fifth:]))
        print("%-20s %9.3f %9.3f %+9.3f %10d %9.3f"
              % (key, a, b, b - a, uniq, drift))
    print("-" * 72)
    print("начало/конец — точность попадания в тему в первой и последней пятой сессии")
    print("уник.док — сколько различных документов выдано за %d запросов;" % n_q)
    print("    обвал этого числа означает схлопывание оператора в одну область")
    print()
    print("static — контроль: по построению не меняется, его прирост есть")
    print("свойство самой нагрузки, и его надо вычитать из остальных.")


if __name__ == "__main__":
    main()
