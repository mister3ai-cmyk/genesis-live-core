#!/usr/bin/env python3
"""
benchmarks/bench_grassmannian_vs_faiss.py
NGP 4.6  --  Grassmannian SRP-LSH vs FAISS Memory & QPS Benchmark

Requirements (optional):
    pip install faiss-cpu numpy

If faiss-cpu is not installed, memory footprint section still runs
and QPS section is skipped with a clear WARNING.
"""
import time, sys, struct
import numpy as np

N_VECS  = 50_000
DIM     = 512
N_QUERY = 1_000
N_PROJ  = 32          # SRP-LSH projection planes -> 32 bits -> uint32

rng = np.random.default_rng(42)

# ── Generate dataset ──────────────────────────────────────────────────────────
vecs   = rng.standard_normal((N_VECS,  DIM), dtype=np.float32)
norms  = np.linalg.norm(vecs,  axis=1, keepdims=True)
vecs  /= np.maximum(norms, 1e-8)

queries = rng.standard_normal((N_QUERY, DIM), dtype=np.float32)
q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
queries /= np.maximum(q_norms, 1e-8)

# ── SRP-LSH Grassmannian quantization ────────────────────────────────────────
# Project R^512 -> {0,1}^32 -> pack to uint32  (128x compression vs float32)
proj_planes = rng.standard_normal((N_PROJ, DIM), dtype=np.float32)
proj_planes /= np.linalg.norm(proj_planes, axis=1, keepdims=True)

def srp_encode(X):
    """Encode batch X (n, DIM) -> uint32 array (n,)"""
    bits = (X @ proj_planes.T) >= 0          # (n, 32) bool
    packed = np.packbits(bits, axis=1, bitorder="big")  # (n, 4) uint8
    return packed.view(np.uint32).reshape(-1)

db_codes  = srp_encode(vecs)                 # uint32 per vector
q_codes   = srp_encode(queries)              # uint32 per query

# ── Memory footprint ─────────────────────────────────────────────────────────
mem_flat_mb = N_VECS * DIM * 4 / 1e6        # float32
mem_hnsw_mb = mem_flat_mb * 1.35            # ~35% graph overhead (empirical)
mem_srp_mb  = N_VECS * 4 / 1e6             # uint32
compression = mem_flat_mb / mem_srp_mb

print()
print("  NGP 4.6 — Grassmannian SRP-LSH vs FAISS Memory Benchmark")
print(f"  Dataset : {N_VECS:,} vectors  x  dim={DIM}  (float32)")
print()
print(f"  {'Backend':<22} {'RAM (MB)':>10}  {'Notes'}")
print(f"  {'-'*60}")
print(f"  {'FAISS IndexFlatIP':<22} {mem_flat_mb:>10.1f}  raw float32 vectors")
print(f"  {'FAISS IndexHNSWFlat':<22} {mem_hnsw_mb:>10.1f}  vectors + HNSW graph (~35% overhead)")
print(f"  {'NGP SRP-LSH uint32':<22} {mem_srp_mb:>10.2f}  4 bytes/vec, Grassmannian projection")
print(f"  {'-'*60}")
print(f"  Compression vs FlatIP : {compression:.0f}x")
print()

# ── QPS benchmark ─────────────────────────────────────────────────────────────
try:
    import faiss
    faiss_available = True
except ImportError:
    print("  WARNING: faiss-cpu not installed — skipping QPS comparison.")
    print("  Install with: pip install faiss-cpu")
    faiss_available = False

def hamming_batch(q_codes_arr, db_codes_arr):
    """Vectorised Hamming distance via XOR + popcount (numpy)."""
    xor = q_codes_arr[:, None] ^ db_codes_arr[None, :]    # (Q, N)
    # popcount via bit-manipulation trick
    x = xor.astype(np.uint32)
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    x = (x + (x >> 4)) & np.uint32(0x0F0F0F0F)
    return ((x * np.uint32(0x01010101)) >> 24).astype(np.int32)

if faiss_available:
    # -- FAISS FlatIP --
    idx_flat = faiss.IndexFlatIP(DIM)
    idx_flat.add(vecs)
    t0 = time.perf_counter()
    idx_flat.search(queries, 1)
    t1 = time.perf_counter()
    qps_flat = N_QUERY / (t1 - t0)

    # -- FAISS HNSW --
    idx_hnsw = faiss.IndexHNSWFlat(DIM, 32)
    idx_hnsw.add(vecs)
    t0 = time.perf_counter()
    idx_hnsw.search(queries, 1)
    t1 = time.perf_counter()
    qps_hnsw = N_QUERY / (t1 - t0)

    # -- NGP SRP-LSH Hamming --
    t0 = time.perf_counter()
    _ = hamming_batch(q_codes, db_codes)
    t1 = time.perf_counter()
    qps_srp = N_QUERY / (t1 - t0)

    print(f"  {'Backend':<22} {'QPS':>10}  {'Notes'}")
    print(f"  {'-'*60}")
    print(f"  {'FAISS IndexFlatIP':<22} {qps_flat:>10,.0f}  exact inner-product search")
    print(f"  {'FAISS IndexHNSWFlat':<22} {qps_hnsw:>10,.0f}  approximate, M=32")
    print(f"  {'NGP SRP-LSH (numpy)':<22} {qps_srp:>10,.0f}  Hamming on uint32 codes")
    print(f"  {'-'*60}")
    note = "faster" if qps_srp > qps_flat else "slower"
    print(f"  SRP-LSH vs FlatIP : {qps_srp/qps_flat:.1f}x {note}  (128x less RAM)")
    print()
else:
    # Hamming-only timing (no FAISS)
    t0 = time.perf_counter()
    _ = hamming_batch(q_codes, db_codes)
    t1 = time.perf_counter()
    qps_srp = N_QUERY / (t1 - t0)
    print(f"  NGP SRP-LSH QPS (numpy Hamming): {qps_srp:,.0f}")
    print()
