#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmarks/mixed_workload.py

Every ANN benchmark in common use measures a static corpus: build the index
once, then query it. That is the workload a search engine has. It is not the
workload an agent's memory has, where the corpus grows and changes while it is
being read.

This measures the other case. A stream of operations mixes queries with
insertions at a given write fraction, and both engines pay the real cost of
absorbing the writes:

  genesis-l0   encode one vector, append the code. O(1), no structure to repair.
  hnswlib      add_items, which links the new node into the graph.

The question is the write fraction at which the graph stops paying for itself.
Below it the graph wins, as every static benchmark shows. Above it the cost of
maintaining the structure exceeds what it saves on the queries.

Usage: python3 mixed_workload.py [--base 20000] [--ops 20000]
Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
"""
import argparse
import ctypes
import glob
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
CACHE = "/tmp/genesis_eval"
BITS = 256
K = 10
N_QUERY = 100
WRITE_FRACTIONS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)


def require(*mods):
    missing = [m for m in mods if not _importable(m)]
    if missing:
        sys.exit("missing: %s\n  pip install %s"
                 % (", ".join(missing), " ".join(missing)))


def _importable(m):
    try:
        __import__(m)
        return True
    except ImportError:
        return False


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
    lib.srp_scan_topk.argtypes = [ctypes.POINTER(SrpCode), ctypes.POINTER(SrpCode),
                                  ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_uint32)]
    lib.srp_scan_topk.restype = ctypes.c_size_t
    lib.srp_rerank_topk.argtypes = [f, ctypes.c_int, f, ctypes.POINTER(ctypes.c_uint32),
                                    ctypes.c_size_t, ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_uint32)]
    lib.srp_rerank_topk.restype = ctypes.c_size_t
    return lib, SrpCode


def fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def load_vectors(dim):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(CACHE, "train-*.parquet")))
    if not files:
        sys.exit("no parquet shards in %s" % CACHE)
    chunks = []
    for fpath in files:
        t = pq.read_table(fpath)
        col = next(c for c in t.column_names
                   if "embedding" in c.lower() or "vector" in c.lower())
        chunks.append(np.vstack(t.column(col).to_numpy(zero_copy_only=False))
                      .astype(np.float32))
    full = np.vstack(chunks)[:, :dim].copy()
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    return np.ascontiguousarray(full)


def make_stream(n_ops, write_fraction, seed=7):
    """Deterministic op stream so both engines face exactly the same sequence."""
    rng = np.random.default_rng(seed)
    return rng.random(n_ops) < write_fraction


def run_genesis(vectors, base, stream, queries, pool, dim):
    lib, SrpCode = load_core(BITS)
    planes = np.empty(BITS * dim, dtype=np.float32)
    lib.srp_make_planes(fptr(planes), dim, 42)
    mean = np.ascontiguousarray(vectors[:base].mean(axis=0))
    offsets = np.empty(BITS, dtype=np.float32)
    lib.srp_make_offsets(fptr(planes), dim, fptr(mean), fptr(offsets))
    off = fptr(offsets)

    cap = base + int(stream.sum()) + 1
    codes = (SrpCode * cap)()
    db = np.zeros((cap, dim), dtype=np.float32)
    db[:base] = vectors[:base]
    for i in range(base):
        codes[i] = lib.srp_encode(fptr(planes), dim, fptr(db[i]), off)

    cand = (ctypes.c_uint32 * pool)()
    out = (ctypes.c_uint32 * K)()
    n_live = base
    next_src = base
    qi = 0
    t0 = time.perf_counter()
    for is_write in stream:
        if is_write:
            db[n_live] = vectors[next_src]
            codes[n_live] = lib.srp_encode(fptr(planes), dim, fptr(db[n_live]), off)
            n_live += 1
            next_src += 1
        else:
            q = queries[qi % len(queries)]
            qi += 1
            qc = lib.srp_encode(fptr(planes), dim, fptr(q), off)
            m = lib.srp_scan_topk(ctypes.byref(qc), codes, n_live, pool, cand)
            lib.srp_rerank_topk(fptr(db), dim, fptr(q), cand, m, K, out)
    return time.perf_counter() - t0, n_live


def run_hnsw(vectors, base, stream, queries, dim, ef=64, M=16):
    import hnswlib
    cap = base + int(stream.sum()) + 1
    h = hnswlib.Index(space="ip", dim=dim)
    h.init_index(max_elements=cap, ef_construction=200, M=M)
    h.set_num_threads(1)
    h.add_items(vectors[:base], np.arange(base))
    h.set_ef(ef)

    n_live = base
    next_src = base
    qi = 0
    t0 = time.perf_counter()
    for is_write in stream:
        if is_write:
            h.add_items(vectors[next_src].reshape(1, -1), np.array([n_live]))
            n_live += 1
            next_src += 1
        else:
            q = queries[qi % len(queries)]
            qi += 1
            h.knn_query(q.reshape(1, -1), k=K)
    return time.perf_counter() - t0, n_live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=20000)
    ap.add_argument("--ops", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pool", type=int, default=512)
    args = ap.parse_args()
    require("numpy", "pyarrow", "hnswlib")

    vectors = load_vectors(args.dim)
    queries = vectors[-N_QUERY:]
    corpus = vectors[:-N_QUERY]
    need = args.base + args.ops
    if len(corpus) < need:
        sys.exit("need %d vectors, have %d" % (need, len(corpus)))

    print("base %d vectors x %d dims, %d operations per run, %d-bit codes, "
          "pool %d, single thread\n" % (args.base, args.dim, args.ops, BITS, args.pool))
    print("%-8s %10s %12s %12s %10s" %
          ("writes", "inserted", "genesis op/s", "hnswlib op/s", "ratio"))
    print("-" * 58)

    for wf in WRITE_FRACTIONS:
        stream = make_stream(args.ops, wf)
        tg, ng = run_genesis(corpus, args.base, stream, queries, args.pool, args.dim)
        th, nh = run_hnsw(corpus, args.base, stream, queries, args.dim)
        og, oh = args.ops / tg, args.ops / th
        print("%6.0f %% %10d %12.0f %12.0f %9.2fx"
              % (wf * 100, ng - args.base, og, oh, og / oh))

    print()
    print("ratio > 1 means genesis-l0 sustains more operations per second than the")
    print("graph does. A static benchmark only ever measures the first row.")


if __name__ == "__main__":
    main()
