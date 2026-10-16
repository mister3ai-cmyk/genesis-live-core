#!/usr/bin/env python3
"""
benchmarks/bench_chroma.py
genesis-l0 vs ChromaDB (local, in-process) on the same corpus.

  pip install genesis-l0 chromadb matplotlib
  python3 eval_real_embeddings.py --center   # writes /tmp/genesis_eval/*_512.fvecs
  python3 bench_chroma.py --data /tmp/genesis_eval [--png out.png]

Both engines get the same vectors, the same payload per vector and the same
queries; each query returns the top-10 ids plus their payloads. Every query is
timed individually (perf_counter_ns) after one warm-up pass. Recall@10 is
against exact inner-product brute force. Numbers on the plot are the measured
ones from this run.

genesis-l0 is installed from PyPI and has its own license (BSL 1.1).

Licensed under the Business Source License 1.1 (SPDX: BUSL-1.1); see LICENSE.
"""
import argparse
import os
import tempfile
import time

import numpy as np

from genesis_l0 import GenesisL0Memory

K = 10


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].view(np.float32).copy()


def stats(v):
    v = np.asarray(v)
    return {"p50": np.percentile(v, 50), "p90": np.percentile(v, 90),
            "p99": np.percentile(v, 99), "mean": v.mean()}


def timed(fn, queries, passes):
    for q in queries:  # warm-up
        fn(q)
    lat = []
    for _ in range(passes):
        for q in queries:
            t0 = time.perf_counter_ns()
            fn(q)
            lat.append((time.perf_counter_ns() - t0) / 1e3)
    return lat


def recall(fn_ids, queries, truth):
    hits = sum(len(set(fn_ids(q)) & set(t.tolist())) for q, t in zip(queries, truth))
    return 100.0 * hits / (K * len(queries))


def plot(results, path, n, dim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    colors = {"ChromaDB (local)": "#8c8c8c", "genesis-l0": "#1f6feb"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    metrics = ["p50", "p90", "p99", "mean"]
    width = 0.38
    x = np.arange(len(metrics))
    for i, name in enumerate(names):
        vals = [results[name]["lat"][m] for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=colors.get(name))
        for b, val in zip(bars, vals):
            label = f"{val / 1000:.2f} ms" if val >= 1000 else f"{val:.0f} µs"
            ax.text(b.get_x() + b.get_width() / 2, val * 1.08, label, ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, metrics)
    ax.set_ylabel("query latency, µs (log scale)")
    ax.set_ylim(bottom=100, top=max(r["lat"]["p99"] for r in results.values()) * 4)

    g, c = results["genesis-l0"], results["ChromaDB (local)"]
    speed = c["lat"]["p50"] / g["lat"]["p50"]
    ax.set_title(f"Top-10 query latency, {n:,} OpenAI vectors ({dim}-dim)", fontsize=11)
    note = (f"ChromaDB (local): p50 {c['lat']['p50'] / 1000:.2f} ms, Recall@10 {c['recall']:.1f} %\n"
            f"genesis-l0 (L0 co-processor): p50 {g['lat']['p50'] / 1000:.2f} ms, Recall@10 {g['recall']:.1f} %\n"
            f"p50 speedup: {speed:.1f}x")
    ax.text(0.01, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    ax.legend(loc="upper right", fontsize=8.5)
    fig.text(0.99, 0.01, "Contabo VPS, AMD EPYC 2.0 GHz; same queries, ids + payload per query",
             ha="right", fontsize=7, color="#666666")
    fig.tight_layout()
    fig.savefig(path, dpi=300)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with corpus_512.fvecs and queries_512.fvecs")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--png", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "chroma_vs_genesis_latency.png"))
    args = ap.parse_args()

    import chromadb

    corpus = read_fvecs(os.path.join(args.data, "corpus_512.fvecs"))
    queries = read_fvecs(os.path.join(args.data, "queries_512.fvecs"))
    n, dim = corpus.shape
    truth = np.argsort(-(queries @ corpus.T), axis=1)[:, :K]
    payloads = [f"payload {i}" for i in range(n)]
    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        # ChromaDB: persistent local client, HNSW in inner-product space
        client = chromadb.PersistentClient(path=os.path.join(tmp, "chroma"))
        col = client.create_collection("bench", metadata={"hnsw:space": "ip"})
        step = client.get_max_batch_size()
        t = time.perf_counter()
        for s in range(0, n, step):
            col.add(ids=[str(i) for i in range(s, min(s + step, n))],
                    embeddings=corpus[s:s + step], documents=payloads[s:s + step])
        c_build = time.perf_counter() - t

        def c_query(q):
            return col.query(query_embeddings=[q], n_results=K, include=["documents"])

        c_rec = recall(lambda q: [int(i) for i in c_query(q)["ids"][0]], queries, truth)
        c_lat = timed(c_query, queries, args.passes)
        results["ChromaDB (local)"] = {"lat": stats(c_lat), "recall": c_rec, "build": c_build}

        # genesis-l0: defaults of the precision mode (pool 512, mean-centered)
        with GenesisL0Memory(os.path.join(tmp, "l0.db"), capacity=n, dim=dim,
                             mean=corpus.mean(axis=0)) as mem:
            t = time.perf_counter()
            mem.add_batch(np.arange(n), corpus,
                          [{"slug": f"g{i}", "faces": {"code": payloads[i]}} for i in range(n)])
            g_build = time.perf_counter() - t
            g_rec = recall(lambda q: mem.index.query(q, K)[0], queries, truth)
            g_lat = timed(lambda q: mem.query(q, top_k=K, face="code"), queries, args.passes)
        results["genesis-l0"] = {"lat": stats(g_lat), "recall": g_rec, "build": g_build}

    print(f"{n} x {dim} vectors, {len(queries)} queries x {args.passes} passes = "
          f"{len(queries) * args.passes} timed queries per engine, top-{K} ids + payload\n")
    hdr = f"{'engine':20}{'p50':>10}{'p90':>10}{'p99':>10}{'mean':>10}{'Recall@10':>11}{'build s':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    for name, r in results.items():
        l = r["lat"]
        print(f"{name:20}" + "".join(f"{l[m]:10.1f}" for m in ("p50", "p90", "p99", "mean"))
              + f"{r['recall']:10.1f}%{r['build']:9.1f}")
    print("(latency in µs)")
    g, c = results["genesis-l0"]["lat"], results["ChromaDB (local)"]["lat"]
    print(f"\nspeedup genesis-l0 vs ChromaDB: p50 {c['p50'] / g['p50']:.1f}x, p99 {c['p99'] / g['p99']:.1f}x")

    plot(results, args.png, n, dim)
    print(f"plot: {args.png}")


if __name__ == "__main__":
    main()
