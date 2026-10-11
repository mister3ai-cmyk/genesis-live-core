#!/usr/bin/env python3
"""genesis_bridge.py — C99 L0 index (libgenesis_l0.so via ctypes) -> Galoglyph face read (SQLite).

glyph_id == row position of the vector in the L0 index.
Demo DB is genesis_demo.db next to this script.

  python3 genesis_bridge.py init
  python3 genesis_bridge.py seed
  python3 genesis_bridge.py query --test --face code
  python3 genesis_bridge.py recall          # sanity: Recall@10 vs ground truth
"""
import argparse
import ctypes
import hashlib
import os
import sqlite3
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "genesis_demo.db")
LIB_PATH = os.path.join(HERE, "libgenesis_l0.so")
# created by benchmarks/eval_real_embeddings.py
EVAL_DIR = os.environ.get("GENESIS_EVAL_DIR", "/tmp/genesis_eval")
CORPUS = os.path.join(EVAL_DIR, "corpus_512.fvecs")
QUERIES = os.path.join(EVAL_DIR, "queries_512.fvecs")
TRUTH = os.path.join(EVAL_DIR, "top10_512.ivecs")
POOL, SEED, PASSES = 512, 7, 10

PRAGMAS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA mmap_size = 268435456",
    "PRAGMA cache_size = -131072",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA foreign_keys = ON",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS glyphs (
    glyph_id INTEGER PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS glyph_faces (
    glyph_id INTEGER NOT NULL,
    face TEXT NOT NULL CHECK(face IN ('code', 'research', 'business')),
    payload TEXT NOT NULL,
    PRIMARY KEY (glyph_id, face),
    FOREIGN KEY (glyph_id) REFERENCES glyphs(glyph_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_glyph_faces_lookup ON glyph_faces(glyph_id, face);
"""

SEED_GLYPH = {
    "glyph_id": 0,
    "slug": "genesis_v1_0_0",
    "code": (
        "Build: gcc -O3 -march=native -std=c99 -Wall -Wextra -fPIC -shared -DSRP_BITS=256 "
        "(Contabo CPU: AVX2 + POPCNT, no AVX-512).\n"
        "typedef struct { uint64_t w[SRP_WORDS]; } srp_code;\n"
        "static inline int srp_hamming(const srp_code *a, const srp_code *b) {\n"
        "    int d = 0;\n"
        "    for (int i = 0; i < SRP_WORDS; i++) d += __builtin_popcountll(a->w[i] ^ b->w[i]);\n"
        "    return d;\n"
        "}\n"
        "size_t srp_scan_topk(const srp_code *q, const srp_code *db, size_t n, size_t k, uint32_t *out_idx);"
    ),
    "research": (
        "OpenAI text-embedding-3-large, DBpedia, 512-dim, ~33k vectors, mean-centered. "
        "256-bit SRP, pool 512 -> exact re-rank: Recall@10 90.6 %, ~3,400 QPS, "
        "index 1.06 MB (fits L3). Measured on Contabo VPS."
    ),
    "business": (
        "Genesis Live Core v1.0.0. Concept DOI 10.5281/zenodo.22926046; "
        "release v1.0.0 DOI 10.5281/zenodo.23246581. "
        "License: Apache-2.0 (src/, benchmarks/), BSL-1.1 (research/)."
    ),
}
FACES = ("code", "research", "business")


def connect():
    conn = sqlite3.connect(DB_PATH)
    for p in PRAGMAS:
        conn.execute(p)
    return conn


def content_hash(code, research, business):
    # \x1f separator keeps ("ab","c") and ("a","bc") from hashing alike
    return hashlib.sha256("\x1f".join((code, research, business)).encode()).hexdigest()


def cmd_init(_):
    with connect() as conn:
        conn.executescript(SCHEMA)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    print(f"initialized {DB_PATH} (journal_mode={mode})")


def cmd_seed(_):
    g = SEED_GLYPH
    h = content_hash(g["code"], g["research"], g["business"])
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO glyphs (glyph_id, slug, content_hash) VALUES (?, ?, ?)",
                     (g["glyph_id"], g["slug"], h))
        conn.executemany("INSERT OR REPLACE INTO glyph_faces (glyph_id, face, payload) VALUES (?, ?, ?)",
                         [(g["glyph_id"], f, g[f]) for f in FACES])
    print(f"seeded {g['slug']} glyph_id={g['glyph_id']} faces={len(FACES)} content_hash={h}")


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].view(dtype).copy()


def load_l0():
    lib = ctypes.CDLL(LIB_PATH)
    f32p, u32p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)
    lib.l0_build.argtypes = [f32p, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_size_t]
    lib.l0_build.restype = ctypes.c_int
    lib.l0_query.argtypes = [f32p, ctypes.c_size_t, u32p, ctypes.POINTER(ctypes.c_double)]
    lib.l0_query.restype = ctypes.c_size_t
    lib.l0_index_bytes.restype = ctypes.c_size_t

    corpus = np.ascontiguousarray(read_vecs(CORPUS, np.float32))
    n, dim = corpus.shape
    t = time.perf_counter()
    rc = lib.l0_build(corpus.ctypes.data_as(f32p), n, dim, SEED, 1, POOL)
    if rc != 0:
        raise SystemExit(f"l0_build failed: {rc}")
    print(f"L0 index: {n} vectors x {dim}-dim, {lib.l0_bits()} bit, pool {POOL}, "
          f"codes {lib.l0_index_bytes() / 2**20:.2f} MB, built in {time.perf_counter() - t:.2f} s")
    return lib, corpus  # corpus must stay alive: C keeps a pointer for re-rank


def search(lib, q, k, out, ns):
    f32p, u32p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)
    m = lib.l0_query(q.ctypes.data_as(f32p), k, out.ctypes.data_as(u32p), ctypes.byref(ns))
    return out[:m]


def cmd_query(args):
    lib, corpus = load_l0()
    qs = read_vecs(QUERIES, np.float32)
    out, ns = np.zeros(args.k, dtype=np.uint32), ctypes.c_double()
    sql = "SELECT payload FROM glyph_faces WHERE glyph_id = ? AND face = ?"
    conn = connect()

    # 1. hit check: --test queries with vector 0, the seeded glyph
    q = np.ascontiguousarray(corpus[0]) if args.test else np.ascontiguousarray(qs[0])
    ids = [int(i) for i in search(lib, q, args.k, out, ns)]
    print(f"Top-{args.k} glyph_ids: {ids}")
    row = conn.execute(sql, (ids[0], args.face)).fetchone()
    if row is None:
        print(f"glyph_id {ids[0]}: face '{args.face}' not seeded")
    else:
        print(f"--- glyph_id {ids[0]} face '{args.face}' ---\n{row[0]}\n---")

    # 2. latency: every held-out query, PASSES times (one warm-up pass first).
    # Distinct queries touch distinct re-rank rows, so the float32 corpus is
    # read from RAM, not replayed from cache.
    qs = [np.ascontiguousarray(v) for v in qs]
    for v in qs:
        search(lib, v, args.k, out, ns)
    inside, wall = [], []
    for _ in range(PASSES):
        for v in qs:
            t = time.perf_counter_ns()
            search(lib, v, args.k, out, ns)
            wall.append(time.perf_counter_ns() - t)
            inside.append(ns.value)

    # SQLite: read the seeded face (a hit) as often as there were searches
    for _ in range(len(qs)):
        conn.execute(sql, (SEED_GLYPH["glyph_id"], args.face)).fetchone()
    reads = []
    for _ in range(len(wall)):
        t = time.perf_counter_ns()
        conn.execute(sql, (SEED_GLYPH["glyph_id"], args.face)).fetchone()
        reads.append(time.perf_counter_ns() - t)
    conn.close()

    def pct(v, p):
        return sorted(v)[min(len(v) - 1, int(p * len(v)))] / 1000

    l0_in, l0_wall, rd = ([pct(v, p) for p in (0.5, 0.99)] for v in (inside, wall, reads))
    print(f"{len(qs)} held-out queries x {PASSES} passes = {len(wall)} searches     p50 / p99")
    print(f"[L0 Search Time]                 {l0_wall[0]:8.1f} / {l0_wall[1]:6.1f} us  "
          f"(inside C {l0_in[0]:.1f} / {l0_in[1]:.1f}, ctypes ~{l0_wall[0] - l0_in[0]:.1f})")
    print(f"[SQLite mmap Face Read Time]     {rd[0]:8.1f} / {rd[1]:6.1f} us  (demo DB, 1 glyph)")
    print(f"[Total End-to-End Latency]       {l0_wall[0] + rd[0]:8.1f} / {l0_wall[1] + rd[1]:6.1f} us  (sum of stage percentiles)")


def cmd_recall(_):
    lib, corpus = load_l0()
    qs = read_vecs(QUERIES, np.float32)
    # ground truth: exact inner-product top-10 (as eval_real_embeddings.py)
    truth = np.argpartition(-(qs @ corpus.T), 10, axis=1)[:, :10]
    # TRUTH file is run_benchmark --out, i.e. the release binary's own answers
    bench = read_vecs(TRUTH, np.int32)
    out, ns, hits, same = np.zeros(10, dtype=np.uint32), ctypes.c_double(), 0, 0
    for q, t, b in zip(qs, truth, bench):
        ids = {int(i) for i in search(lib, np.ascontiguousarray(q), 10, out, ns)}
        hits += len(ids & {int(i) for i in t})
        same += len(ids & {int(i) for i in b})
    n = 10 * len(qs)
    print(f"Recall@10 vs exact brute force over {len(qs)} queries: {100.0 * hits / n:.1f} %")
    print(f"Overlap with release run_benchmark output: {100.0 * same / n:.1f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    sub.add_parser("seed").set_defaults(fn=cmd_seed)
    q = sub.add_parser("query")
    q.add_argument("--test", action="store_true", help="query with vector 0 (the seeded glyph)")
    q.add_argument("--face", choices=FACES, default="research")
    q.add_argument("-k", type=int, default=10)
    q.set_defaults(fn=cmd_query)
    sub.add_parser("recall").set_defaults(fn=cmd_recall)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
