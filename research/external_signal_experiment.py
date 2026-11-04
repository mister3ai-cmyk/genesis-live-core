#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/external_signal_experiment.py

Three attempts to make retrieval improve with use have failed, and measurement
showed all three failed for one reason: the learning signal was the system's own
output. The unitary rule reinforced "W maps q to Wq", a tautology. The centroid
target was a function of W. Hebbian co-retrieval reinforced whatever cosine had
already found, and acted as a session cache - precision up, new relevant items
down by half.

This gives the loop a signal from outside it: which of the returned results was
actually used. An agent citing three of ten sources produces exactly that, and
the retrieval layer cannot know it in advance.

The obvious objection is leakage: the simulated feedback is derived from the
same topic labels the metric scores, so precision is partly guaranteed to rise.
Two measurements guard against it.

  novel hits      items of the right topic returned for the first time in the
                  session. Memorising the feedback cannot raise this; only
                  generalisation can.
  held-out probe  after the session ends, fresh queries from the same topic with
                  no feedback at all. This is the honest test: did the layer
                  learn something transferable, or only cache what it was told.

Usage: python3 external_signal_experiment.py
Licensed under the Business Source License 1.1 (SPDX: BUSL-1.1); see research/LICENSE.
"""
import argparse
import glob
import os
import sys

import numpy as np

CACHE = "/tmp/genesis_eval"
K = 10
ETA = 0.2
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


def retrieve(corpus, W, active, q, qi, beta):
    score = corpus @ q
    if W is not None and active:
        ev = W[active].sum(axis=0)
        m = ev.max()
        if m > 1e-9:
            score = score + beta * (ev / m)
    score[qi] = -np.inf
    return np.argpartition(-score, K)[:K]


def run(corpus, labels, sessions, mode, beta):
    """mode: 'cosine' | 'self' (как раньше) | 'external' (сигнал извне)"""
    n = len(corpus)
    W = None if mode == "cosine" else np.zeros((n, n), dtype=np.float32)
    prec, novel, probe_prec, probe_novel = [], [], [], []

    for topic, q_idx, probe_idx in sessions:
        active, seen = [], set()
        for qi in q_idx:
            q = corpus[qi]
            top = retrieve(corpus, W, active, q, qi, beta)
            prec.append(float(np.mean(labels[top] == topic)))
            novel.append(sum(1 for t in top
                             if labels[t] == topic and int(t) not in seen))
            seen.update(int(t) for t in top)

            if W is not None:
                act = np.clip(corpus[top] @ q, 0.0, None).astype(np.float32)
                if mode == "external":
                    # Сигнал извне: подкрепляем только то, что «использовали»,
                    # остальное гасим. Система не знает разметки заранее.
                    used = (labels[top] == topic).astype(np.float32)
                    act = act * (2.0 * used - 1.0)      # +act нужным, -act прочим
                sub = W[np.ix_(top, top)]
                x, y = act[:, None], act[None, :]
                sub += ETA * y * (x - y * sub) - DECAY * sub
                np.fill_diagonal(sub, 0.0)
                sub[sub < PRUNE] = 0.0
                W[np.ix_(top, top)] = sub
                active = (active + top.tolist())[-40:]

        # Отложенная проверка: свежие запросы той же темы, обратной связи нет.
        pseen = set()
        for qi in probe_idx:
            top = retrieve(corpus, W, active, corpus[qi], qi, beta)
            probe_prec.append(float(np.mean(labels[top] == topic)))
            probe_novel.append(sum(1 for t in top
                                   if labels[t] == topic and int(t) not in pseen))
            pseen.update(int(t) for t in top)

    return (float(np.mean(prec)), float(np.mean(novel)),
            float(np.mean(probe_prec)), float(np.mean(probe_novel)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--corpus", type=int, default=6000)
    ap.add_argument("--topics", type=int, default=20)
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--per-session", type=int, default=25)
    ap.add_argument("--probe", type=int, default=8)
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    print("loading %d vectors..." % args.corpus)
    corpus = load_vectors(args.dim, args.corpus)
    labels = KMeans(n_clusters=args.topics, n_init=4,
                    random_state=0).fit_predict(corpus)

    rng = np.random.default_rng(13)
    sessions = []
    need = args.per_session + args.probe
    for _ in range(args.sessions):
        topic = int(rng.integers(args.topics))
        members = np.flatnonzero(labels == topic)
        if len(members) < need:
            continue
        pick = rng.choice(members, need, replace=False)
        sessions.append((topic, pick[:args.per_session], pick[args.per_session:]))
    print("%d сессий x %d запросов + %d отложенных\n"
          % (len(sessions), args.per_session, args.probe))

    print("%-34s %10s %10s %12s %12s" %
          ("режим", "точность", "новых", "отлож.точн", "отлож.новых"))
    print("-" * 82)
    base = None
    for name, mode, beta in (("косинус (контроль)", "cosine", 0.0),
                             ("свой выход, b=0.3", "self", 0.3),
                             ("ВНЕШНИЙ сигнал, b=0.3", "external", 0.3),
                             ("ВНЕШНИЙ сигнал, b=1.0", "external", 1.0)):
        p, nv, pp, pn = run(corpus, labels, sessions, mode, beta)
        if base is None:
            base = (p, nv, pp, pn)
        print("%-34s %10.3f %10.2f %12.3f %12.2f" % (name, p, nv, pp, pn))
    print("-" * 82)
    print("отлож. — свежие запросы той же темы ПОСЛЕ сессии, без обратной связи.")
    print("Обучение состоялось, только если выросли и отложенная точность,")
    print("и отложенное число новых попаданий. Рост одной точности при падении")
    print("новых означает запоминание подсказанного, а не обобщение.")


if __name__ == "__main__":
    main()
