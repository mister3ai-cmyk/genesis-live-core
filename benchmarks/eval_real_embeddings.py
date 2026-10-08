#!/usr/bin/env python3
"""
benchmarks/eval_real_embeddings.py
Recall of the two-stage cascade on real text embeddings.

Dataset: Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K
         (DBpedia entities embedded with OpenAI text-embedding-3-large),
         first parquet shard, ~318 MB, cached in /tmp/genesis_eval/.
Runs the C binary at full 1536 dims and at 512 dims (text-embedding-3
supports shortening: truncate + L2-normalize). Recall is computed by the
C binary and re-checked here against an independent numpy brute force.

Requirements: pip install numpy pyarrow
Usage:        python3 benchmarks/eval_real_embeddings.py [--center]
              (extra flags are passed through to run_benchmark)

Licensed under the Apache License, Version 2.0.
"""
import os
import re
import subprocess
import sys
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

URL = ("https://huggingface.co/datasets/Qdrant/"
       "dbpedia-entities-openai3-text-embedding-3-large-1536-100K/"
       "resolve/main/data/train-00000-of-00003.parquet")
CACHE = "/tmp/genesis_eval"
N_QUERY = 100
K = 10
DIMS = (1536, 512)

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "run_benchmark")


def download(url, path):
    if os.path.exists(path):
        return
    print(f"Downloading {url}")
    tmp = path + ".part"

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, progress)
    print()
    os.replace(tmp, path)


def load_embeddings(path):
    table = pq.read_table(path)
    for name in table.column_names:
        col_type = table.schema.field(name).type
        if pa.types.is_list(col_type) or pa.types.is_large_list(col_type) \
                or pa.types.is_fixed_size_list(col_type):
            col = table.column(name).combine_chunks()
            flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
            return name, flat.reshape(len(col), -1)
    sys.exit(f"no embedding column found in {table.column_names}")


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def write_fvecs(path, x):
    n, d = x.shape
    rec = np.empty((n, d + 1), dtype=np.float32)
    rec[:, 0] = np.array([d], dtype=np.int32).view(np.float32)[0]
    rec[:, 1:] = x
    rec.tofile(path)


def read_ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    k = raw[0]
    return raw.reshape(-1, k + 1)[:, 1:]


def main():
    os.makedirs(CACHE, exist_ok=True)
    parquet = os.path.join(CACHE, os.path.basename(URL))
    download(URL, parquet)

    if not os.path.exists(BINARY):
        subprocess.run(["make", "-C", HERE], check=True)

    column, full = load_embeddings(parquet)
    rng = np.random.default_rng(0)
    q_idx = rng.choice(len(full), N_QUERY, replace=False)
    mask = np.ones(len(full), dtype=bool)
    mask[q_idx] = False
    print(f"Column: {column}, {len(full)} vectors x {full.shape[1]} dims; "
          f"{N_QUERY} held-out queries, {mask.sum()} corpus vectors\n")

    summary = []
    for dim in DIMS:
        corpus = normalize(full[mask, :dim])
        queries = normalize(full[q_idx, :dim])
        d_path = os.path.join(CACHE, f"corpus_{dim}.fvecs")
        q_path = os.path.join(CACHE, f"queries_{dim}.fvecs")
        o_path = os.path.join(CACHE, f"top10_{dim}.ivecs")
        write_fvecs(d_path, corpus)
        write_fvecs(q_path, queries)

        out = subprocess.run([BINARY, "--data", d_path, "--queries", q_path, "--out", o_path]
                             + sys.argv[1:],
                             check=True, capture_output=True, text=True).stdout
        print(out)
        c_recall = float(re.search(r"Recall 10@10:\s+([\d.]+)", out).group(1))
        qps = float(re.search(r"Throughput:\s+([\d.]+)", out).group(1))

        # independent check: numpy brute force vs the cascade's top-10
        truth = np.argpartition(-(queries @ corpus.T), K, axis=1)[:, :K]
        found = read_ivecs(o_path)
        np_recall = 100.0 * np.mean([len(set(t) & set(f)) / K for t, f in zip(truth, found)])
        summary.append((dim, c_recall, np_recall, qps))

    print("=" * 60)
    print("REAL-EMBEDDING RECALL (OpenAI text-embedding-3-large, DBpedia)")
    print("=" * 60)
    for dim, c_recall, np_recall, qps in summary:
        print(f"{dim:>5}-dim  Recall 10@10: {c_recall:5.1f} %  "
              f"(numpy check {np_recall:5.1f} %)  {qps:,.0f} QPS")
    print("=" * 60)


if __name__ == "__main__":
    main()
