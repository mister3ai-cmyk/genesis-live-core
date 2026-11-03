#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/hebbian_experiment.py

Does the Hebbian association layer improve retrieval, where the unitary phase
operator provably cannot?

The unitary reconsolidation of RFC-4.6-001 was measured to have zero associative
capacity: a unitary transform preserves every inner product, so rotating the
query against a fixed corpus can only misalign it. See the capacity check in
reconsolidation_experiment.py and the finding it produced.

octant1_hebbian_engine.py is a different construction and the objection does not
apply to it. It keeps a weighted edge table between nodes, updated by Oja's rule
with decay:

    delta = ETA * y_j * (x_i - y_j * w_ij) - DECAY * w_ij      (ETA 0.02, DECAY 0.005)

That is an associative store, not a rotation, so it can hold "these two belong
together". Whether holding it helps retrieval is the open question, and this
measures it.

Method, same as the unitary experiment so the two are comparable: the corpus is
clustered into topics, each session draws its queries from one topic, and a
result counts as relevant when it shares the query's topic. The association
layer is given no topic labels - it only sees which items get retrieved together.

Arms:
  cosine      plain vector retrieval, the control
  hebbian     score = cosine + beta * association evidence from the session so far

Usage: python3 hebbian_experiment.py [--beta 0.3]
Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

CACHE = "/tmp/genesis_eval"
K = 10
ETA = 0.02          # as in octant1_hebbian_engine.py
DECAY = 0.005
PRUNE = 0.01


def load_vectors(dim, n):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(CACHE, "train-*.parquet")))
    if not files:
        sys.exit("no parquet shards in %s" % CACHE)
    chunks, total = [], 0
    for f in files:
        t = pq.read_table(f)
        col = next(c for c in t.column_names
                   if "embedding" in c.lower() or "vector" in c.lower())
        a = np.vstack(t.column(col).to_numpy(zero_copy_only=False)).astype(np.float32)
        chunks.append(a)
        total += len(a)
        if total >= n:
            break
    v = np.vstack(chunks)[:n, :dim].copy()
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v)


def run(corpus, labels, sessions, beta, use_hebb, eta=ETA, prune=PRUNE):
    """One pass over the identical query stream."""
    n = len(corpus)
    W = np.zeros((n, n), dtype=np.float32) if use_hebb else None
    npos = max(len(s[1]) for s in sessions)
    per_position = [[] for _ in range(npos)]
    novel_position = [[] for _ in range(npos)]   # новые попадания в тему
    returned = set()
    edges_alive = 0
    t0 = time.perf_counter()

    for topic, q_idx in sessions:
        active = []                      # узлы, активированные в этой сессии
        seen = set()                     # что уже выдавалось в этой сессии
        for pos, qi in enumerate(q_idx):
            q = corpus[qi]
            score = corpus @ q
            if use_hebb and active:
                # Свидетельство связей от уже активированных в сессии узлов.
                evidence = W[active].sum(axis=0)
                m = evidence.max()
                if m > 1e-9:
                    score = score + beta * (evidence / m)
            score[qi] = -np.inf
            top = np.argpartition(-score, K)[:K]
            returned.update(top.tolist())
            per_position[pos].append(float(np.mean(labels[top] == topic)))
            # Кэш или память: сколько попаданий в тему ВПЕРВЫЕ за сессию.
            novel = [int(t) for t in top if labels[t] == topic and int(t) not in seen]
            novel_position[pos].append(len(novel))
            seen.update(int(t) for t in top)

            if use_hebb:
                # Правило Ойи между соизвлечёнными узлами, активация = косинус.
                act = np.clip(corpus[top] @ q, 0.0, None).astype(np.float32)
                sub = W[np.ix_(top, top)]
                x = act[:, None]
                y = act[None, :]
                sub += eta * y * (x - y * sub) - DECAY * sub
                np.fill_diagonal(sub, 0.0)
                sub[sub < prune] = 0.0
                W[np.ix_(top, top)] = sub
                active.extend(top.tolist())
                active = active[-40:]
    dt = time.perf_counter() - t0
    if use_hebb:
        edges_alive = int((W > 0).sum())
    curve = [float(np.mean(p)) if p else float("nan") for p in per_position]
    novel = [float(np.mean(p)) if p else float("nan") for p in novel_position]
    return curve, len(returned), edges_alive, dt, novel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--corpus", type=int, default=6000)
    ap.add_argument("--topics", type=int, default=20)
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--per-session", type=int, default=25)
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    print("loading %d vectors at %d dims..." % (args.corpus, args.dim))
    corpus = load_vectors(args.dim, args.corpus)
    print("clustering into %d topics (defines relevance, hidden from the model)..."
          % args.topics)
    labels = KMeans(n_clusters=args.topics, n_init=4, random_state=0).fit_predict(corpus)

    rng = np.random.default_rng(13)
    sessions = []
    for _ in range(args.sessions):
        topic = int(rng.integers(args.topics))
        members = np.flatnonzero(labels == topic)
        if len(members) < args.per_session:
            continue
        sessions.append((topic, rng.choice(members, args.per_session, replace=False)))
    print("%d sessions x %d queries\n" % (len(sessions), args.per_session))

    fifth = max(1, args.per_session // 5)
    print("%-22s %8s %8s %8s %9s %9s %9s" %
          ("правило", "нач.", "кон.", "прирост", "новых/зпр", "уник.док", "рёбер"))
    print("-" * 82)
    rows = [("cosine (контроль)", 0.0, False, ETA, PRUNE)]
    rows += [("eta=.02 prune=.01 b=%.1f" % b, b, True, 0.02, 0.01) for b in (0.3,)]
    rows += [("БЕЗ отсечения   b=%.1f" % b, b, True, 0.02, 0.0) for b in (0.3, 1.0)]
    rows += [("eta=.2 prune=.01 b=%.1f" % b, b, True, 0.2, 0.01) for b in (0.3, 1.0)]
    for name, beta, use, eta, prune in rows:
        curve, uniq, edges, dt, novel = run(corpus, labels, sessions, beta, use,
                                            eta, prune)
        a = float(np.mean(curve[:fifth]))
        z = float(np.mean(curve[-fifth:]))
        nov = float(np.nanmean(novel))
        print("%-22s %8.3f %8.3f %+8.3f %9.2f %9d %9d"
              % (name, a, z, z - a, nov, uniq, edges))
    print("-" * 82)
    print("новых/зпр — сколько попаданий в тему ВПЕРВЫЕ за сессию, на запрос.")
    print("    Если растёт только точность, а это число падает — слой работает")
    print("    кэшем: переотдаёт найденное, а не находит новое.")
    print("прирост считается внутри сессии: последняя пятая минус первая.")
    print("Контроль обязан быть плоским — его прирост есть свойство нагрузки,")
    print("и его надо вычитать из остальных строк.")


if __name__ == "__main__":
    main()
