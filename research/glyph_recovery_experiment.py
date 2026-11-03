#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/glyph_recovery_experiment.py

Can usage resolve what quantisation collapsed?

The previous six experiments ran on dense float vectors and all came out
negative. That was the wrong substrate. On floats there is nothing for an
association layer to add: cosine is already optimal for the information the
vectors still carry, so any extra term can only blur the ranking, which is
exactly what was measured.

Glyphs lose information differently. A 128-bit code does not merely approximate
a vector, it **collapses distinct documents onto the same or neighbouring
codes**, and to a Hamming scan those documents are indistinguishable in
principle. The measured deficit is large: recall 56-68 % at 128 bits against
82-89 % at 256, where the full-dimensional neighbours are the truth.

That loss is of a kind usage can address. If two documents share a code but are
retrieved in different contexts, co-occurrence separates them — and that
separation is not present in the codes, nor recoverable from the float vectors.
It comes from use. This is the one regime where "memory that changes on read"
has something to do that nothing else can do.

  baseline     Hamming scan over SRP-LSH codes, as the engine does it
  association  the same scan, blended with edge evidence from use
  ceiling      exact cosine in the full 1536 dims, recall 1.0 by construction

The association layer never sees a float vector. Held-out queries receive no
feedback at all.

Usage: python3 glyph_recovery_experiment.py [--bits 128]
Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
"""
import argparse
import ctypes
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
CACHE = "/tmp/genesis_eval"
K = 10
ETA = 0.2
DECAY = 0.005
PRUNE = 0.01
POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def load_core(bits):
    so = os.path.join(CACHE, "libsrp%d.so" % bits)
    if not os.path.exists(so):
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-std=c99", "-shared", "-fPIC",
             "-DSRP_BITS=%d" % bits, os.path.join(SRC, "srp_lsh.c"),
             "-o", so, "-lm"], check=True)

    class SrpCode(ctypes.Structure):
        _fields_ = [("w", ctypes.c_uint64 * (bits // 64))]

    lib = ctypes.CDLL(so)
    f = ctypes.POINTER(ctypes.c_float)
    lib.srp_make_planes.argtypes = [f, ctypes.c_int, ctypes.c_uint64]
    lib.srp_make_offsets.argtypes = [f, ctypes.c_int, f, f]
    lib.srp_encode.argtypes = [f, ctypes.c_int, f, f]
    lib.srp_encode.restype = SrpCode
    return lib, SrpCode


def fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


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


def encode_all(vecs, bits, dim):
    """Кодирование ровно тем же C-ядром, что и в движке, с центрированием."""
    lib, SrpCode = load_core(bits)
    planes = np.empty(bits * dim, dtype=np.float32)
    lib.srp_make_planes(fptr(planes), dim, 42)
    mean = np.ascontiguousarray(vecs.mean(axis=0))
    offs = np.empty(bits, dtype=np.float32)
    lib.srp_make_offsets(fptr(planes), dim, fptr(mean), fptr(offs))
    codes = (SrpCode * len(vecs))()
    for i in range(len(vecs)):
        codes[i] = lib.srp_encode(fptr(planes), dim, fptr(vecs[i]), fptr(offs))
    return np.frombuffer(bytes(codes), dtype=np.uint8).reshape(len(vecs), bits // 8)


def hamming_scores(packed, qrow):
    """Чем меньше расстояние, тем выше оценка. Тот же POPCNT, что в C."""
    d = POPCNT[np.bitwise_xor(packed, qrow)].sum(axis=1).astype(np.float32)
    return -d


def run(packed, truth, sessions, beta, use_layer):
    n = len(packed)
    W = np.zeros((n, n), dtype=np.float32) if use_layer else None
    tr, pr, collisions = [], [], []

    for q_idx, probe_idx in sessions:
        for qi in q_idx:
            base = hamming_scores(packed, packed[qi:qi + 1])
            score = base.copy()
            if W is not None:
                seed = np.argpartition(-base, K)[:K]
                ev = W[seed].sum(axis=0)
                m = ev.max()
                if m > 1e-9:
                    score = score + beta * (ev / m)
            score[qi] = -np.inf
            top = np.argpartition(-score, K)[:K]
            tr.append(len(set(top.tolist()) & truth[qi]) / float(K))
            # сколько кандидатов делят расстояние с десятым — мера неразличимости
            cut = np.partition(base, -K)[-K]
            collisions.append(int((base == cut).sum()))

            if W is not None:
                good = np.array([1.0 if int(t) in truth[qi] else -1.0
                                 for t in top], dtype=np.float32)
                act = good.astype(np.float32)
                sub = W[np.ix_(top, top)]
                x, y = act[:, None], act[None, :]
                sub += ETA * y * (x - y * sub) - DECAY * sub
                np.fill_diagonal(sub, 0.0)
                sub[sub < PRUNE] = 0.0
                W[np.ix_(top, top)] = sub

        for qi in probe_idx:
            base = hamming_scores(packed, packed[qi:qi + 1])
            score = base.copy()
            if W is not None:
                seed = np.argpartition(-base, K)[:K]
                ev = W[seed].sum(axis=0)
                m = ev.max()
                if m > 1e-9:
                    score = score + beta * (ev / m)
            score[qi] = -np.inf
            top = np.argpartition(-score, K)[:K]
            pr.append(len(set(top.tolist()) & truth[qi]) / float(K))

    return float(np.mean(tr)), float(np.mean(pr)), float(np.mean(collisions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--corpus", type=int, default=6000)
    ap.add_argument("--topics", type=int, default=20)
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--per-session", type=int, default=25)
    ap.add_argument("--probe", type=int, default=8)
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    print("загружаю %d векторов..." % args.corpus)
    raw = load_full(args.corpus)
    full = norm(raw)
    short = norm(raw[:, :args.dim].copy())

    print("кодирую в глифы: %d бит из %d измерений..." % (args.bits, args.dim))
    packed = encode_all(short, args.bits, args.dim)
    print("индекс глифов: %.2f МБ против %.1f МБ float32\n"
          % (packed.nbytes / 1e6, full.nbytes / 1e6))

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

    print("эталон — соседи в полных 1536 измерениях, %d запросов" % len(needed))
    truth = {}
    for qi in needed:
        s = full @ full[qi]
        s[qi] = -np.inf
        truth[qi] = set(np.argpartition(-s, K)[:K].tolist())

    print("\n%-30s %13s %13s %12s" %
          ("режим", "recall обуч.", "recall отлож.", "неразличим."))
    print("-" * 72)
    b_tr, b_pr, coll = run(packed, truth, sessions, 0.0, False)
    print("%-30s %13.3f %13.3f %12.1f" %
          ("скан по глифам (база)", b_tr, b_pr, coll))
    for beta in (0.5, 2.0, 8.0):
        tr, prv, _ = run(packed, truth, sessions, beta, True)
        print("%-30s %13.3f %13.3f %12s   %+.3f отлож."
              % ("+ ассоциации, beta=%.1f" % beta, tr, prv, "", prv - b_pr))
    print("%-30s %13.3f %13.3f" % ("потолок: 1536 измерений", 1.0, 1.0))
    print("-" * 72)
    print("неразличим. — сколько кандидатов делят расстояние с десятым:")
    print("    это и есть коллизии квантования, которые опыт может развести.")


if __name__ == "__main__":
    main()
