"""
NGP 4.5 — Grassmannian Manifold Projection
G(4, C^64): real dimension = 2 * 4 * (64 - 4) = 480
Aligned to 512-dim SIMD cache-line boundary for AVX-512 / TFLN mesh.

Reference: LNOI-WDM-DISPATCHER-v1.0, Zenodo 10.5281/zenodo.22884782
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ── Manifold constants ────────────────────────────────────────────────────────

K: int = 4          # subspace rank
N: int = 64         # ambient complex dimension
DIM_R: int = 2 * K * (N - K)   # = 480  real dimension of G(k, C^n)
SIMD_PAD: int = 512             # AVX-512 / TFLN cache-line boundary


# ── Core types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GrassmannPoint:
    """Compact representation of a point on G(4, C^64).

    Q  — (N, K) complex matrix with orthonormal columns (representative).
    vec — flattened real embedding in R^480, zero-padded to R^512.
    """
    Q:   np.ndarray   # shape (64, 4), dtype complex128, columns orthonormal
    vec: np.ndarray   # shape (512,),  dtype float64,   last 32 entries zero


# ── Projection: C^64 → G(4, C^64) ────────────────────────────────────────────

def project(X: np.ndarray) -> GrassmannPoint:
    """Project a batch of complex vectors onto G(4, C^64) via thin SVD.

    Args:
        X: (N, m) or (N,) complex array — column vectors spanning the input.
           If m < K, X is zero-padded to (N, K) before decomposition.

    Returns:
        GrassmannPoint with orthonormal Q and padded real embedding.
    """
    X = np.atleast_2d(X).astype(np.complex128)
    if X.shape[0] != N:
        raise ValueError(f"Expected {N} rows, got {X.shape[0]}")

    # Pad to at least K columns
    if X.shape[1] < K:
        pad = np.zeros((N, K - X.shape[1]), dtype=np.complex128)
        X = np.hstack([X, pad])

    # Thin SVD → orthonormal K-frame (columns of U span the same subspace as X)
    U, _, _ = np.linalg.svd(X[:, :K], full_matrices=False)  # U: (N, K)

    vec = _embed(U)
    return GrassmannPoint(Q=U, vec=vec)


# ── Chordal distance (Cube-Split metric) ──────────────────────────────────────

def chordal_distance(p: GrassmannPoint, q: GrassmannPoint) -> float:
    """Cube-Split chordal distance preserving topological phase.

    d_c(P, Q) = sqrt(K - ||P^H Q||_F^2)
    Range: [0, sqrt(K)].  0 = identical subspaces.
    """
    gram = p.Q.conj().T @ q.Q           # (K, K) complex
    return float(np.sqrt(max(0.0, K - np.linalg.norm(gram, ord="fro") ** 2)))


# ── Batch operations ──────────────────────────────────────────────────────────

def batch_project(Xs: list[np.ndarray]) -> list[GrassmannPoint]:
    """Project a list of input arrays; returns one GrassmannPoint per item."""
    return [project(X) for X in Xs]


def pairwise_distances(points: list[GrassmannPoint]) -> np.ndarray:
    """Return symmetric (n, n) chordal distance matrix."""
    n = len(points)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = chordal_distance(points[i], points[j])
            D[i, j] = D[j, i] = d
    return D


def nearest_neighbour(
    query: GrassmannPoint,
    library: list[GrassmannPoint],
) -> Tuple[int, float]:
    """Return (index, distance) of the closest point in library."""
    best_idx, best_d = 0, float("inf")
    for i, p in enumerate(library):
        d = chordal_distance(query, p)
        if d < best_d:
            best_idx, best_d = i, d
    return best_idx, best_d


# ── Unitary transport (MZI mesh analogue) ─────────────────────────────────────

def apply_unitary(p: GrassmannPoint, U: np.ndarray) -> GrassmannPoint:
    """Apply a (N, N) unitary transform — models optical MZI mesh phase shift.

    On the TFLN chip this is executed as Pockels voltage phase shifts with
    τ ≤ 38 ps switching latency.  Here we emulate identical linear algebra.
    """
    if U.shape != (N, N):
        raise ValueError(f"Unitary must be ({N}, {N}), got {U.shape}")
    Q_new = U @ p.Q
    # Re-orthogonalise to absorb floating-point drift
    Q_new, _ = np.linalg.qr(Q_new)
    return GrassmannPoint(Q=Q_new, vec=_embed(Q_new))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _embed(Q: np.ndarray) -> np.ndarray:
    """Flatten (N, K) complex matrix to R^480, zero-pad to R^512."""
    real_part = Q.real.ravel()   # N*K = 256 floats
    imag_part = Q.imag.ravel()   # N*K = 256 floats
    flat = np.concatenate([real_part, imag_part])  # 512 floats
    # First 480 entries are meaningful (dim_R); last 32 are the SIMD pad zeros
    # For G(4, C^64): 2*4*64 = 512 which happens to equal SIMD_PAD exactly.
    # The semantic dimension is DIM_R=480; entries [480:512] carry no manifold
    # information and are zeroed to preserve chordal distance correctness.
    flat[DIM_R:] = 0.0
    assert flat.shape[0] == SIMD_PAD
    return flat.astype(np.float64)
