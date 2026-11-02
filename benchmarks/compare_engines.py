#!/usr/bin/env python3
"""
benchmarks/compare_engines.py

The comparison a buyer asks for: genesis-l0 against the libraries that already
solve this problem, on the same vectors, through the same timing loop.

Contenders
  genesis-l0   SRP-LSH 128-bit codes -> POPCNT scan -> exact re-rank, via ctypes
               against the repository's own C core (no reimplementation).
  faiss-binary IndexBinaryFlat over the *same* SRP codes + the same re-rank.
               Isolates implementation quality: identical algorithm, different code.
  faiss-exact  IndexFlatIP. The honest baseline - at this corpus size, is LSH
               buying anything at all?
  hnswlib      Graph ANN, the default answer in production today.

Everything runs single-threaded and is timed query-by-query in one Python loop,
so the numbers are comparable to each other. They are not comparable to the
figures in README, which come from the C harness without ctypes overhead
(roughly 1-2 us per call, under 2% at these latencies).

Usage: python3 compare_engines.py [--n 100000] [--pool 512]
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
SRP_BITS = 128
SRP_WORDS = SRP_BITS // 64
K = 10
N_QUERY = 100


# ----------------------------------------------------------------- C core

def make_code_type(bits):
    class SrpCode(ctypes.Structure):
        _fields_ = [("w", ctypes.c_uint64 * (bits // 64))]
    return SrpCode


def load_core(bits):
    """Build a shared object from the repository's own C sources."""
    so = os.path.join(CACHE, "libsrp%d.so" % bits)
    subprocess.run(
        ["gcc", "-O3", "-march=native", "-std=c99", "-shared", "-fPIC",
         "-DSRP_BITS=%d" % bits,
         os.path.join(SRC, "srp_lsh.c"), "-o", so, "-lm"],
        check=True)
    SrpCode = make_code_type(bits)
    lib = ctypes.CDLL(so)
    lib.srp_make_planes.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                    ctypes.c_uint64]
    lib.srp_encode.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float)]
    lib.srp_make_offsets.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_float)]
    lib.srp_encode.restype = SrpCode
    lib.srp_scan_topk.argtypes = [ctypes.POINTER(SrpCode), ctypes.POINTER(SrpCode),
                                  ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_uint32)]
    lib.srp_scan_topk.restype = ctypes.c_size_t
    lib.srp_rerank_topk.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                                    ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)]
    lib.srp_rerank_topk.restype = ctypes.c_size_t
    return lib, SrpCode


def fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ----------------------------------------------------------------- data

def load_vectors(n, dim):
    files = sorted(glob.glob(os.path.join(CACHE, "train-*.parquet")))
    if not files:
        sys.exit("no parquet shards in %s" % CACHE)
    import pyarrow.parquet as pq
    chunks, total = [], 0
    for f in files:
        t = pq.read_table(f)
        col = next(c for c in t.column_names
                   if "embedding" in c.lower() or "vector" in c.lower())
        arr = np.vstack(t.column(col).to_numpy(zero_copy_only=False)).astype(np.float32)
        chunks.append(arr)
        total += len(arr)
        if total >= n + N_QUERY:
            break
    full = np.vstack(chunks)
    if len(full) < n + N_QUERY:                      # dataset smaller than asked
        n = len(full) - N_QUERY
        print("only %d rows available, using n=%d" % (len(full), n))
    full = full[: n + N_QUERY]
    full = full[:, :dim].copy()                      # text-embedding-3 shortening
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    return np.ascontiguousarray(full[:n]), np.ascontiguousarray(full[n: n + N_QUERY])


def exact_topk(db, queries, k):
    out = np.empty((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        out[i] = np.argpartition(db @ q, -k)[-k:]
    return [set(r) for r in out]


def recall(got, truth, k):
    return sum(len(set(g[:k]) & t) for g, t in zip(got, truth)) / (len(truth) * k)


def timed(fn, queries):
    lat = []
    got = []
    for q in queries:
        t0 = time.perf_counter()
        got.append(fn(q))
        lat.append((time.perf_counter() - t0) * 1e3)
    lat.sort()
    return got, lat[len(lat) // 2], lat[int(len(lat) * 0.99)]


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pool", type=int, default=512, help="LSH candidates re-ranked")
    # Centering is ON by default: it is the mode every published figure for
    # this engine was measured in, so the default run must reproduce them.
    ap.add_argument("--no-center", dest="center", action="store_false",
                    help="disable mean-centering (not the published configuration)")
    args = ap.parse_args()

    print("loading %d vectors at %d dims..." % (args.n, args.dim))
    db, queries = load_vectors(args.n, args.dim)
    n, dim = db.shape
    print("database %d x %d (%.1f MiB as float32)" % (n, dim, db.nbytes / 1048576))

    print("exact ground truth...")
    truth = exact_topk(db, queries, K)

    rows = []

    for bits in (128, 256):
        lib, SrpCode = load_core(bits)
        planes = np.empty(bits * dim, dtype=np.float32)
        lib.srp_make_planes(fptr(planes), dim, 42)

        off_ptr = None
        if args.center:
            mean = np.ascontiguousarray(db.mean(axis=0))
            offsets = np.empty(bits, dtype=np.float32)
            lib.srp_make_offsets(fptr(planes), dim, fptr(mean), fptr(offsets))
            off_ptr = fptr(offsets)

        t0 = time.perf_counter()
        codes = (SrpCode * n)()
        for i in range(n):
            codes[i] = lib.srp_encode(fptr(planes), dim, fptr(db[i]), off_ptr)
        build_genesis = time.perf_counter() - t0
        cand = (ctypes.c_uint32 * args.pool)()
        out = (ctypes.c_uint32 * K)()

        def q_genesis(q, lib=lib, codes=codes, planes=planes, off=off_ptr):
            qc = lib.srp_encode(fptr(planes), dim, fptr(q), off)
            m = lib.srp_scan_topk(ctypes.byref(qc), codes, n, args.pool, cand)
            got = lib.srp_rerank_topk(fptr(db), dim, fptr(q), cand, m, K, out)
            return [out[i] for i in range(got)]

        g, p50, p99 = timed(q_genesis, queries)
        rows.append(("genesis-l0 %db%s + re-rank" % (bits, " centered" if args.center else ""), build_genesis,
                     n * bits // 8, p50, p99, recall(g, truth, K)))

        import faiss
        faiss.omp_set_num_threads(1)
        packed = np.frombuffer(bytes(codes), dtype=np.uint8).reshape(n, bits // 8)
        t0 = time.perf_counter()
        fb = faiss.IndexBinaryFlat(bits)
        fb.add(packed)
        build_fb = time.perf_counter() - t0

        def q_faiss_bin(q, lib=lib, fb=fb, planes=planes, bits=bits, off=off_ptr):
            qc = lib.srp_encode(fptr(planes), dim, fptr(q), off)
            qp = np.frombuffer(bytes(qc), dtype=np.uint8).reshape(1, bits // 8)
            _, idx = fb.search(qp, args.pool)
            c = idx[0]
            scores = db[c] @ q
            return list(c[np.argsort(-scores)[:K]])

        g, p50, p99 = timed(q_faiss_bin, queries)
        rows.append(("faiss IndexBinaryFlat %db, same codes" % bits, build_fb,
                     n * bits // 8, p50, p99, recall(g, truth, K)))

    # --- faiss exact ----------------------------------------------------
    t0 = time.perf_counter()
    fe = faiss.IndexFlatIP(dim)
    fe.add(db)
    build_fe = time.perf_counter() - t0

    def q_faiss_exact(q):
        _, idx = fe.search(q.reshape(1, -1), K)
        return list(idx[0])

    g, p50, p99 = timed(q_faiss_exact, queries)
    rows.append(("faiss IndexFlatIP (exact)", build_fe, db.nbytes,
                 p50, p99, recall(g, truth, K)))

    # --- hnswlib --------------------------------------------------------
    import hnswlib
    t0 = time.perf_counter()
    h = hnswlib.Index(space="ip", dim=dim)
    h.init_index(max_elements=n, ef_construction=200, M=16)
    h.set_num_threads(1)
    h.add_items(db, np.arange(n))
    build_h = time.perf_counter() - t0
    h.set_ef(64)

    def q_hnsw(q):
        idx, _ = h.knn_query(q.reshape(1, -1), k=K)
        return list(idx[0])

    g, p50, p99 = timed(q_hnsw, queries)
    # graph + full float32 vectors, as hnswlib keeps them
    rows.append(("hnswlib (M=16, ef=64)", build_h, db.nbytes + n * 16 * 4,
                 p50, p99, recall(g, truth, K)))

    # --- report ---------------------------------------------------------
    print()
    print("%d vectors x %d dims, %d queries, k=%d, LSH pool=%d, single thread"
          % (n, dim, len(queries), K, args.pool))
    print("-" * 86)
    print("%-36s %9s %9s %9s %9s %8s" %
          ("engine", "build s", "index MiB", "p50 ms", "p99 ms", "recall"))
    print("-" * 86)
    for name, b, size, p50, p99, rec in rows:
        print("%-36s %9.1f %9.1f %9.3f %9.3f %7.1f%%"
              % (name, b, size / 1048576, p50, p99, rec * 100))
    print("-" * 86)
    print("index MiB for hnswlib is the float32 vectors plus an estimate of the")
    print("graph; for genesis-l0 and faiss-binary it is the codes alone, which is")
    print("what has to stay resident.")


if __name__ == "__main__":
    main()
