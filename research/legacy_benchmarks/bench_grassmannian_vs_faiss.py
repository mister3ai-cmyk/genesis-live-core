#!/usr/bin/env python3
"""
benchmarks/bench_grassmannian_vs_faiss.py
NGP 4.6 — Grassmannian SRP-LSH vs FAISS  (Memory + QPS)

Hamming distance path (in order of preference):
  1. fast_hamming.so  via ctypes  — C99 __builtin_popcount, POPCNT hw instruction
  2. numpy fallback               — pure Python, ~2k QPS (slow)

Requirements:
    pip install faiss-cpu numpy
"""
import ctypes, os, time, sys
import numpy as np

N_VECS  = 50_000
DIM     = 512
N_QUERY = 1_000
N_PROJ  = 32          # 32-bit SRP-LSH code per vector

rng = np.random.default_rng(42)

# ── Dataset ──────────────────────────────────────────────────────────────────
vecs    = rng.standard_normal((N_VECS,  DIM), dtype=np.float32)
vecs   /= np.maximum(np.linalg.norm(vecs,  axis=1, keepdims=True), 1e-8)
queries = rng.standard_normal((N_QUERY, DIM), dtype=np.float32)
queries/= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)

# ── SRP-LSH encode ────────────────────────────────────────────────────────────
proj = rng.standard_normal((N_PROJ, DIM), dtype=np.float32)
proj /= np.linalg.norm(proj, axis=1, keepdims=True)

def srp_encode(X):
    bits  = (X @ proj.T) >= 0
    packed = np.packbits(bits, axis=1, bitorder="big")
    return packed.view(np.uint32).reshape(-1)

db_codes = srp_encode(vecs)
q_codes  = srp_encode(queries)

# ── Load C popcount library ───────────────────────────────────────────────────
_lib = None
_LIB_PATH = os.path.join(os.path.dirname(__file__), "fast_hamming.so")

def _load_lib():
    global _lib
    if not os.path.exists(_LIB_PATH):
        return False
    try:
        _lib = ctypes.CDLL(_LIB_PATH)
        _lib.hamming_batch_u32_distances.restype  = None
        _lib.hamming_batch_u32_distances.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),  # q_codes
            ctypes.POINTER(ctypes.c_uint32),  # db_codes
            ctypes.POINTER(ctypes.c_int32),   # out_dist
            ctypes.c_size_t,                  # n_queries
            ctypes.c_size_t,                  # n_db
        ]
        return True
    except OSError:
        return False

_lib_ok = _load_lib()

def hamming_c(q, db):
    """Call C99 __builtin_popcount kernel via ctypes."""
    out = np.empty(len(q), dtype=np.int32)
    _lib.hamming_batch_u32_distances(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        db.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(len(q)),
        ctypes.c_size_t(len(db)),
    )
    return out

def hamming_numpy(q, db):
    """Pure-numpy Hamming (slow fallback)."""
    xor = q[:, None] ^ db[None, :]
    x = xor.astype(np.uint32)
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    x = (x + (x >> 4)) & np.uint32(0x0F0F0F0F)
    pop = ((x * np.uint32(0x01010101)) >> 24).astype(np.int32)
    return pop.min(axis=1)

# ── Memory footprint ──────────────────────────────────────────────────────────
mem_flat = N_VECS * DIM * 4 / 1e6
mem_hnsw = mem_flat * 1.35
mem_srp  = N_VECS * 4 / 1e6
compression = mem_flat / mem_srp

print()
print("  NGP 4.6 — Grassmannian SRP-LSH vs FAISS Memory Benchmark")
print(f"  Dataset : {N_VECS:,} vectors  x  dim={DIM}  (float32)")
print()
print(f"  {'Backend':<22} {'RAM (MB)':>10}  Notes")
print(f"  {'-'*60}")
print(f"  {'FAISS IndexFlatIP':<22} {mem_flat:>10.1f}  raw float32 vectors")
print(f"  {'FAISS IndexHNSWFlat':<22} {mem_hnsw:>10.1f}  vectors + HNSW graph (~35% overhead)")
print(f"  {'NGP SRP-LSH uint32':<22} {mem_srp:>10.2f}  4 bytes/vec, Grassmannian projection")
print(f"  {'-'*60}")
print(f"  Compression vs FlatIP : {compression:.0f}x")
print()

# ── QPS benchmark ─────────────────────────────────────────────────────────────
try:
    import faiss
    faiss_available = True
except ImportError:
    print("  WARNING: faiss-cpu not installed — skipping FAISS QPS.")
    faiss_available = False

hamming_label = f"NGP SRP-LSH C99 POPCNT" if _lib_ok else "NGP SRP-LSH (numpy)"
hamming_fn    = hamming_c if _lib_ok else lambda q, db: hamming_numpy(q, db)

if not _lib_ok:
    print("  WARNING: fast_hamming.so not found — using slow numpy fallback.")

if faiss_available:
    idx_flat = faiss.IndexFlatIP(DIM)
    idx_flat.add(vecs)
    t0 = time.perf_counter(); idx_flat.search(queries, 1); t1 = time.perf_counter()
    qps_flat = N_QUERY / (t1 - t0)

    idx_hnsw = faiss.IndexHNSWFlat(DIM, 32)
    idx_hnsw.add(vecs)
    t0 = time.perf_counter(); idx_hnsw.search(queries, 1); t1 = time.perf_counter()
    qps_hnsw = N_QUERY / (t1 - t0)

    # Warm-up
    _ = hamming_fn(q_codes, db_codes)
    t0 = time.perf_counter(); _ = hamming_fn(q_codes, db_codes); t1 = time.perf_counter()
    qps_srp = N_QUERY / (t1 - t0)

    print(f"  {'Backend':<26} {'QPS':>10}  Notes")
    print(f"  {'-'*64}")
    print(f"  {'FAISS IndexFlatIP':<26} {qps_flat:>10,.0f}  exact inner-product search")
    print(f"  {'FAISS IndexHNSWFlat':<26} {qps_hnsw:>10,.0f}  approximate, M=32")
    print(f"  {hamming_label:<26} {qps_srp:>10,.0f}  {'C99 __builtin_popcount / POPCNT' if _lib_ok else 'numpy XOR+popcount fallback'}")
    print(f"  {'-'*64}")
    ratio = qps_srp / qps_flat
    word  = "faster" if ratio >= 1 else "slower"
    print(f"  SRP-LSH vs FlatIP : {ratio:.1f}x {word}  ({compression:.0f}x less RAM)")
    print()
else:
    _ = hamming_fn(q_codes, db_codes)
    t0 = time.perf_counter(); _ = hamming_fn(q_codes, db_codes); t1 = time.perf_counter()
    qps_srp = N_QUERY / (t1 - t0)
    print(f"  {hamming_label} QPS: {qps_srp:,.0f}")
    print()
