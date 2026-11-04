#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/recovery_experiment.py

Can an association layer recover information that dimensional truncation threw
away?

The previous experiments defined relevance by topic cluster, and that was the
flaw in them. With 300 members per topic and k=10, precision saturates for any
system that returns a stable central core, so the metric rewarded exactly the
degenerate behaviour it was meant to detect. The novelty guard caught the
symptom; the design caused it.

Cosine ground truth is not usable either: it makes flat retrieval perfect by
definition, leaving nothing to improve.

This uses the gap between the two. Ground truth is the top-10 by cosine in the
**full 1536-dim** embedding. Retrieval runs on vectors **truncated to 256 dims**,
where that ground truth is no longer reachable - truncation measurably costs
about ten points of recall. The question is whether co-occurrence structure can
put some of it back.

The design has no leakage: the layer never sees the full-dimensional vectors,
and no configuration of it can reach recall 1.0 by narrowing, because every
query has its own distinct set of true neighbours.

  baseline     flat cosine in 256 dims. Its recall is the deficit to close.
  ceiling      flat cosine in 1536 dims, which is 1.0 by construction.
  association  256-dim retrieval blended with Hebbian edge evidence.

Usage: python3 recovery_experiment.py
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


def load_full(n):
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
    return np.vstack(chunks)[:n]


def norm(a):
    a = a.astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return np.ascontiguousarray(a)


def truth_topk(full, qi):
    s = full @ full[qi]
    s[qi] = -np.inf
    return set(np.argpartition(-s, K)[:K].tolist())


def run(short, truth, sessions, use_layer, beta, mode='session'):
    n = len(short)
    W = np.zeros((n, n), dtype=np.float32) if use_layer else None
    rec_train, rec_probe = [], []

    for q_idx, probe_idx in sessions:
        active = []
        for qi in q_idx:
            q = short[qi]
            base = short @ q
            score = base.copy()
            if W is not None:
                if mode == "session" and active:
                    # Одинаково для всех запросов сессии: константная добавка.
                    ev = W[active].sum(axis=0)
                elif mode == "query":
                    # Растекание активации от собственных кандидатов запроса:
                    # свидетельство зависит от q, а не от сессии.
                    seed = np.argpartition(-base, K)[:K]
                    wts = np.clip(base[seed], 0.0, None)
                    ev = (W[seed] * wts[:, None]).sum(axis=0)
                else:
                    ev = None
                if ev is not None:
                    m = ev.max()
                    if m > 1e-9:
                        score = score + beta * (ev / m)
            score[qi] = -np.inf
            top = np.argpartition(-score, K)[:K]
            rec_train.append(len(set(top.tolist()) & truth[qi]) / float(K))

            if W is not None:
                # Сигнал извне контура: какие из выданных оказались настоящими
                # соседями в полной размерности. Слой полных векторов не видит.
                act = np.clip(short[top] @ q, 0.0, None).astype(np.float32)
                good = np.array([1.0 if int(t) in truth[qi] else -1.0
                                 for t in top], dtype=np.float32)
                act = act * good
                sub = W[np.ix_(top, top)]
                x, y = act[:, None], act[None, :]
                sub += ETA * y * (x - y * sub) - DECAY * sub
                np.fill_diagonal(sub, 0.0)
                sub[sub < PRUNE] = 0.0
                W[np.ix_(top, top)] = sub
                active = (active + top.tolist())[-40:]

        for qi in probe_idx:          # отложенные: обратной связи нет
            q = short[qi]
            base = short @ q
            score = base.copy()
            if W is not None:
                if mode == "session" and active:
                    ev = W[active].sum(axis=0)
                elif mode == "query":
                    seed = np.argpartition(-base, K)[:K]
                    wts = np.clip(base[seed], 0.0, None)
                    ev = (W[seed] * wts[:, None]).sum(axis=0)
                else:
                    ev = None
                if ev is not None:
                    m = ev.max()
                    if m > 1e-9:
                        score = score + beta * (ev / m)
            score[qi] = -np.inf
            top = np.argpartition(-score, K)[:K]
            rec_probe.append(len(set(top.tolist()) & truth[qi]) / float(K))

    return float(np.mean(rec_train)), float(np.mean(rec_probe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-dim", type=int, default=256)
    ap.add_argument("--corpus", type=int, default=6000)
    ap.add_argument("--topics", type=int, default=20)
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--per-session", type=int, default=25)
    ap.add_argument("--probe", type=int, default=8)
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    print("loading %d vectors (full 1536 dims)..." % args.corpus)
    raw = load_full(args.corpus)
    full = norm(raw)
    short = norm(raw[:, :args.short_dim].copy())
    print("эталон — соседи в %d измерениях, поиск — в %d\n"
          % (full.shape[1], short.shape[1]))

    labels = KMeans(n_clusters=args.topics, n_init=4,
                    random_state=0).fit_predict(short)
    rng = np.random.default_rng(13)
    sessions, needed = [], set()
    need = args.per_session + args.probe
    for _ in range(args.sessions):
        t = int(rng.integers(args.topics))
        members = np.flatnonzero(labels == t)
        if len(members) < need:
            continue
        pick = rng.choice(members, need, replace=False)
        sessions.append((pick[:args.per_session], pick[args.per_session:]))
        needed.update(int(x) for x in pick)

    print("вычисляю эталон для %d запросов..." % len(needed))
    truth = {qi: truth_topk(full, qi) for qi in needed}

    print("\n%-32s %14s %14s" % ("режим", "recall обуч.", "recall отлож."))
    print("-" * 64)
    b_tr, b_pr = run(short, truth, sessions, False, 0.0)
    print("%-32s %14.3f %14.3f" % ("плоский косинус 256 (база)", b_tr, b_pr))
    for mode, tag in (("session", "от сессии"), ("query", "ОТ ЗАПРОСА")):
        for beta in (0.05, 0.1, 0.3):
            tr, pr = run(short, truth, sessions, True, beta, mode)
            print("%-32s %14.3f %14.3f   %+.3f отлож."
                  % ("%s, beta=%.2f" % (tag, beta), tr, pr, pr - b_pr))
    print("%-32s %14.3f %14.3f" % ("потолок: полные 1536 измерений", 1.0, 1.0))
    print("-" * 64)
    print("Дефицит усечения = 1.000 − база. Слой полезен, только если")
    print("закрывает его часть на ОТЛОЖЕННЫХ запросах, где подсказок не было.")


if __name__ == "__main__":
    main()
