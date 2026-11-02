#!/usr/bin/env python3
"""
benchmarks/tie_analysis.py

Does code width, rather than the latency budget, set the ceiling?

A 128-bit code admits only 129 distances, binomially concentrated near 64, so as
the corpus grows the number of database codes sharing the distance at the top-k
boundary grows with it. Past some size the candidate pool is decided by ties
rather than by similarity, and recall falls while the clock says there is budget
left. This script measures that directly: the tie count at the k-th boundary, the
recall, and how the query budget splits between the Hamming scan and the re-rank.

Usage: python3 tie_analysis.py [--dim 512] [--pool 512]
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
K = 10
N_QUERY = 100
SIZES = (20000, 50000, 99900)

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


def load_vectors(n, dim):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(CACHE, "train-*.parquet")))
    if not files:
        sys.exit("no parquet shards in %s" % CACHE)
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
    n = min(n, len(full) - N_QUERY)
    full = full[: n + N_QUERY][:, :dim].copy()
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    return (np.ascontiguousarray(full[:n]),
            np.ascontiguousarray(full[n: n + N_QUERY]))


def hamming_all(packed_db, packed_q):
    """Distance from one query code to every database code."""
    return POPCNT[np.bitwise_xor(packed_db, packed_q)].sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pool", type=int, default=512)
    args = ap.parse_args()

    big_db, queries = load_vectors(max(SIZES), args.dim)
    print("%d x %d loaded, %d queries, k=%d, pool=%d\n"
          % (len(big_db), args.dim, len(queries), K, args.pool))

    print("%-6s %-9s %8s %10s %10s %9s %9s %9s" %
          ("bits", "n", "d@k", "ties@d", "tie/pool", "recall", "scan ms", "rank ms"))
    print("-" * 82)

    for bits in (128, 256):
        lib, SrpCode = load_core(bits)
        planes = np.empty(bits * args.dim, dtype=np.float32)
        lib.srp_make_planes(fptr(planes), args.dim, 42)
        nbytes = bits // 8

        for n in SIZES:
            if n > len(big_db):
                continue
            db = big_db[:n]
            codes = (SrpCode * n)()
            for i in range(n):
                codes[i] = lib.srp_encode(fptr(planes), args.dim, fptr(db[i]), None)
            packed = np.frombuffer(bytes(codes), dtype=np.uint8).reshape(n, nbytes)

            cand = (ctypes.c_uint32 * args.pool)()
            out = (ctypes.c_uint32 * K)()
            dk, ties, hits, t_scan, t_rank = [], [], 0, 0.0, 0.0

            for q in queries:
                qc = lib.srp_encode(fptr(planes), args.dim, fptr(q), None)
                qp = np.frombuffer(bytes(qc), dtype=np.uint8).reshape(1, nbytes)

                # the distance at the k-th boundary, and how many share it
                d = hamming_all(packed, qp)
                boundary = np.partition(d, K - 1)[K - 1]
                dk.append(int(boundary))
                ties.append(int((d == boundary).sum()))

                t0 = time.perf_counter()
                m = lib.srp_scan_topk(ctypes.byref(qc), codes, n, args.pool, cand)
                t1 = time.perf_counter()
                got = lib.srp_rerank_topk(fptr(db), args.dim, fptr(q), cand, m,
                                          K, out)
                t2 = time.perf_counter()
                t_scan += t1 - t0
                t_rank += t2 - t1

                truth = set(np.argpartition(db @ q, -K)[-K:].tolist())
                hits += len({out[i] for i in range(got)} & truth)

            nq = len(queries)
            print("%-6d %-9d %8.1f %10.1f %9.0f%% %8.1f%% %9.3f %9.3f" %
                  (bits, n, np.median(dk), np.median(ties),
                   100.0 * np.median(ties) / args.pool,
                   100.0 * hits / (nq * K),
                   1e3 * t_scan / nq, 1e3 * t_rank / nq))
        print()

    # Is recall bound by distance ties, or simply by the pool covering a
    # smaller share of a growing corpus? Scaling the pool separates the two.
    print("pool sweep at n=%d (same vectors, same codes)" % len(big_db))
    print("%-6s %-9s %9s %9s %9s" % ("bits", "pool", "recall", "scan ms", "rank ms"))
    print("-" * 48)
    for bits in (128, 256):
        lib, SrpCode = load_core(bits)
        planes = np.empty(bits * args.dim, dtype=np.float32)
        lib.srp_make_planes(fptr(planes), args.dim, 42)
        n = len(big_db)
        codes = (SrpCode * n)()
        for i in range(n):
            codes[i] = lib.srp_encode(fptr(planes), args.dim, fptr(big_db[i]), None)
        for pool in (512, 1024, 2048, 4096):
            cand = (ctypes.c_uint32 * pool)()
            out = (ctypes.c_uint32 * K)()
            hits, t_scan, t_rank = 0, 0.0, 0.0
            for q in queries:
                qc = lib.srp_encode(fptr(planes), args.dim, fptr(q), None)
                t0 = time.perf_counter()
                m = lib.srp_scan_topk(ctypes.byref(qc), codes, n, pool, cand)
                t1 = time.perf_counter()
                got = lib.srp_rerank_topk(fptr(big_db), args.dim, fptr(q), cand, m,
                                          K, out)
                t2 = time.perf_counter()
                t_scan += t1 - t0
                t_rank += t2 - t1
                truth = set(np.argpartition(big_db @ q, -K)[-K:].tolist())
                hits += len({out[i] for i in range(got)} & truth)
            nq = len(queries)
            print("%-6d %-9d %8.1f%% %9.3f %9.3f"
                  % (bits, pool, 100.0 * hits / (nq * K),
                     1e3 * t_scan / nq, 1e3 * t_rank / nq))
        print()

    print("d@k      median Hamming distance at the k-th nearest code")
    print("ties@d   median number of codes sharing that exact distance")
    print("tie/pool that tie set as a share of the candidate pool: once it passes")
    print("         100%, which rows enter the pool is decided by code collisions")
    print("         rather than by similarity, and recall stops tracking the clock.")


if __name__ == "__main__":
    main()
