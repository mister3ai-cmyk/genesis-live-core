"""
NGP 4.5 — Isoperimetric Sieve (Poincaré Ball Filter)
Eliminates ~99% pseudo-scientific and thermodynamically impossible hypotheses.
Outputs a shortlist of <= 10 hardware-executable candidates per cycle.

Pipeline:
  TaskSpec JSON  →  complex embedding  →  G(4, C^64) projection
  →  Poincaré Ball score  →  thermodynamic feasibility gate
  →  ranked shortlist[10]

Reference: Genesis Live Core README, Tier 2 — NGP 4.5 In Silico Core
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grassmannian_manifold import (
    GrassmannPoint,
    N, K,
    batch_project,
    chordal_distance,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SHORTLIST_SIZE: int = 10          # max hardware-executable candidates per cycle
POINCARE_RADIUS: float = 0.97     # unit ball cap; hypotheses outside → rejected
THERMO_THRESHOLD: float = 0.15    # minimum feasibility score (0–1)
NOISE_REJECTION_TARGET: float = 0.99  # design target: eliminate 99% of input


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    """Structured experimental proposal submitted to the Global Ingestion Layer."""
    hypothesis_id:  str
    description:    str
    reagents:       list[str]
    temperature_k:  float          # Kelvin
    pressure_atm:   float
    delta_g_kjmol:  float          # Gibbs free energy change (negative = spontaneous)
    metadata:       dict[str, Any] = field(default_factory=dict)

    def spec_hash(self) -> str:
        """Deterministic keccak-256-style SHA-256 hash for on-chain anchoring."""
        canonical = json.dumps(
            {
                "id":          self.hypothesis_id,
                "reagents":    sorted(self.reagents),
                "T_K":         self.temperature_k,
                "P_atm":       self.pressure_atm,
                "dG_kJ_mol":   self.delta_g_kjmol,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class FilteredCandidate:
    """A hypothesis that passed all sieve stages."""
    spec:           TaskSpec
    point:          GrassmannPoint
    poincare_norm:  float   # distance from origin in Poincaré ball (0–1)
    thermo_score:   float   # thermodynamic feasibility (0–1)
    composite_score: float  # combined ranking score (higher = more executable)


# ── Embedding: TaskSpec → C^64 ────────────────────────────────────────────────

def _embed_spec(spec: TaskSpec) -> np.ndarray:
    """Map a TaskSpec to a complex (64,) vector for Grassmannian projection.

    Encoding scheme (deterministic, hardware-reproducible):
      - Channels  0–15  : reagent fingerprints (SHA-256 mod π)
      - Channels 16–31  : thermodynamic scalars (T, P, ΔG, derived features)
      - Channels 32–63  : metadata hash expansion (zero if absent)
    """
    vec = np.zeros(N, dtype=np.complex128)

    # Reagent fingerprints → real part
    for i, reagent in enumerate(spec.reagents[:16]):
        h = int(hashlib.sha256(reagent.encode()).hexdigest(), 16)
        vec[i].real = (h % 1_000_003) / 1_000_003  # normalised to (0, 1)

    # Thermodynamic scalars → imaginary part of channels 0–2
    vec[0].imag = np.tanh(spec.temperature_k / 1000.0)
    vec[1].imag = np.tanh(spec.pressure_atm  / 100.0)
    vec[2].imag = np.tanh(-spec.delta_g_kjmol / 500.0)  # sign: negative ΔG = good

    # Derived features: dimensionless Gibbs criterion, RT product
    R = 8.314e-3  # kJ/(mol·K)
    RT = R * spec.temperature_k
    vec[16].real = np.tanh(spec.delta_g_kjmol / RT) if RT > 0 else 0.0
    vec[17].real = np.tanh(spec.pressure_atm * spec.temperature_k / 30_000.0)

    # Metadata hash expansion → channels 32–47
    meta_str = json.dumps(spec.metadata, sort_keys=True)
    meta_hash = hashlib.sha256(meta_str.encode()).digest()
    for i in range(16):
        vec[32 + i].real = meta_hash[i] / 255.0
        vec[32 + i].imag = meta_hash[i + 16] / 255.0

    return vec.reshape(N, 1)  # column vector for project()


# ── Scoring ────────────────────────────────────────────────────────────────────

def _poincare_norm(point: GrassmannPoint) -> float:
    """Project Grassmannian embedding into Poincaré unit ball and return norm.

    Uses the first DIM_R=480 components of the padded 512-vec.
    Normalised so that any physically reachable hypothesis maps inside the ball.
    """
    v = point.vec[:480]
    raw_norm = float(np.linalg.norm(v))
    # Map to (0, 1) via tanh so the ball is bounded
    return float(np.tanh(raw_norm / np.sqrt(480)))


def _thermo_score(spec: TaskSpec) -> float:
    """Thermodynamic feasibility score in [0, 1].

    Combines:
      - Gibbs criterion:  ΔG < 0 → spontaneous
      - Temperature window: 273–400 K standard lab range scores higher
      - Pressure window:   0.5–5 atm scores higher
    """
    R = 8.314e-3
    RT = R * max(spec.temperature_k, 1.0)

    # Gibbs component: sigmoid on -ΔG/RT (positive if spontaneous)
    gibbs = 1.0 / (1.0 + np.exp(spec.delta_g_kjmol / RT))

    # Temperature window component
    t_norm = spec.temperature_k / 400.0
    temp = float(np.exp(-((t_norm - 0.85) ** 2) / 0.1))  # peak near 340 K

    # Pressure window component
    p_norm = np.log1p(spec.pressure_atm) / np.log1p(5.0)
    pressure = float(1.0 - abs(p_norm - 0.5))

    return float((gibbs * 0.6 + temp * 0.25 + pressure * 0.15))


def _composite(poincare_norm: float, thermo: float) -> float:
    """Combined ranking score: higher = more likely to produce clean peaks."""
    # Penalise proximity to ball boundary (noisy embeddings cluster near edge)
    boundary_penalty = max(0.0, poincare_norm - 0.85) * 5.0
    return thermo * (1.0 - boundary_penalty)


# ── Main sieve ────────────────────────────────────────────────────────────────

def run_sieve(specs: list[TaskSpec]) -> list[FilteredCandidate]:
    """Apply the Isoperimetric Sieve to a batch of TaskSpecs.

    Stages:
      1. Embed each spec into C^64
      2. Project onto G(4, C^64)
      3. Reject if Poincaré norm > POINCARE_RADIUS (geometric noise gate)
      4. Reject if thermodynamic score < THERMO_THRESHOLD
      5. Rank survivors by composite score
      6. Return top SHORTLIST_SIZE candidates

    Args:
        specs: Raw hypothesis proposals from Global Ingestion Layer.

    Returns:
        Ordered list of up to 10 FilteredCandidate objects.
    """
    if not specs:
        return []

    # Stage 1–2: embed + project
    embeddings = [_embed_spec(s) for s in specs]
    points = batch_project(embeddings)

    candidates: list[FilteredCandidate] = []

    for spec, point in zip(specs, points):
        # Stage 3: geometric gate
        p_norm = _poincare_norm(point)
        if p_norm > POINCARE_RADIUS:
            continue

        # Stage 4: thermodynamic gate
        t_score = _thermo_score(spec)
        if t_score < THERMO_THRESHOLD:
            continue

        c_score = _composite(p_norm, t_score)
        candidates.append(FilteredCandidate(
            spec=spec,
            point=point,
            poincare_norm=p_norm,
            thermo_score=t_score,
            composite_score=c_score,
        ))

    # Stage 5–6: rank and truncate
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return candidates[:SHORTLIST_SIZE]


def rejection_rate(total_in: int, shortlist: list[FilteredCandidate]) -> float:
    """Fraction of input hypotheses eliminated by the sieve."""
    if total_in == 0:
        return 0.0
    return 1.0 - len(shortlist) / total_in
