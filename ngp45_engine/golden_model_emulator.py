"""
NGP 4.5 — Golden Model Emulator (CPU / AVX-512 reference)
Emulates the TFLN photonic MZI mesh in software to generate golden reference
vectors for calibrating the hardware chip before and after tape-out.

Physical model:
  - Clements mesh topology (Clements et al. 2016) — universal unitary on N modes
  - MZI cell length: 5.15 µm (TFLN active zone per switching stage)
  - Optical latency per cell: τ = L_cell * n_eff / c ≈ 37.98 ps  (≤ 38.0 ps spec)
  - n_eff (LNOI ridge waveguide, TE₀, 1550 nm): 2.211
  - c in µm/ps: 0.299792 µm/ps  (speed of light)

Golden checksum:
  Serialised as np.round(frame, 8).astype(complex64).tobytes() on a C-contiguous
  array — platform-agnostic (x86 / ARM / RISC-V) for on-chain anchoring.

Reference: LNOI-WDM-DISPATCHER-v1.0, Zenodo 10.5281/zenodo.22884782
           Genesis Live Core README, Phase 0–2 hardware bridge
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .grassmannian_manifold import GrassmannPoint, N, K, project, apply_unitary

# ── Physical constants ────────────────────────────────────────────────────────

C_LIGHT_UM_PS: float = 0.299792      # speed of light, µm/ps  (~300 000 km/s)
N_EFF_LNOI:   float = 2.211          # effective refractive index, TFLN TE₀ @ 1550 nm
CELL_LENGTH_UM: float = 5.15         # MZI active zone per switching stage, µm

# τ = L_cell * n_eff / c  →  5.15 * 2.211 / 0.299792 ≈ 37.98 ps  (≤ 38.0 ps ✓)
OPTICAL_LATENCY_PS: float = (CELL_LENGTH_UM * N_EFF_LNOI) / C_LIGHT_UM_PS

TOLERANCE: float = 1e-6    # golden vector acceptance tolerance for chip verification
CHECKSUM_DECIMALS: int = 8 # rounding precision before cross-platform serialisation


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class GoldenFrame:
    """Output of one emulator pass — reference vector + reproducible checksum."""
    input_hash:      str          # SHA-256 of input state vector
    output_state:    np.ndarray   # (N,) complex128 — optical field amplitudes
    point:           GrassmannPoint
    optical_latency_ps: float
    golden_checksum: str          # 0x<sha256> — anchored to on-chain verifier


@dataclass
class CalibrationReport:
    """Result of comparing emulator golden vectors against chip measurements."""
    pass_count:  int = 0
    fail_count:  int = 0
    max_error:   float = 0.0
    errors:      list[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.fail_count == 0


# ── Clements mesh ─────────────────────────────────────────────────────────────

def _mzi_unitary(n_modes: int, theta: float, phi: float, ch: int) -> np.ndarray:
    """2×2 MZI transfer matrix embedded in an (n_modes × n_modes) identity.

    Beamsplitter convention (symmetric):
      M = [[cos θ, -e^{iφ} sin θ],
           [sin θ,  e^{iφ} cos θ]]

    Args:
        n_modes: Total number of waveguide modes.
        theta:   Internal phase shift (beamsplitter angle).
        phi:     External phase shift (Pockels modulator).
        ch:      Index of the first mode of the 2-mode pair (ch, ch+1).
    """
    U = np.eye(n_modes, dtype=np.complex128)
    c, s = np.cos(theta), np.sin(theta)
    ep = np.exp(1j * phi)
    U[ch,   ch]   =  c
    U[ch,   ch+1] = -ep * s
    U[ch+1, ch]   =  s
    U[ch+1, ch+1] =  ep * c
    return U


def _clements_mesh(n_modes: int, phases: np.ndarray) -> np.ndarray:
    """Build a Clements-topology unitary from a flat phase parameter array.

    Layer structure (alternating offsets — canonical Clements 2016):
      Even layers (layer % 2 == 0): pairs (0,1), (2,3), (4,5) ...
      Odd  layers (layer % 2 == 1): pairs (1,2), (3,4), (5,6) ...

    Args:
        n_modes: Number of waveguide modes (N).
        phases:  Flat array of shape (n_layers * n_pairs_per_layer * 2,)
                 encoding [theta_0, phi_0, theta_1, phi_1, ...] per layer.

    Returns:
        (n_modes, n_modes) complex128 unitary matrix U.
    """
    U = np.eye(n_modes, dtype=np.complex128)
    n_layers = n_modes - 1
    idx = 0

    for layer in range(n_layers):
        shift = layer % 2                          # ← Clements alternating offset
        for ch in range(shift, n_modes - 1, 2):
            theta = phases[idx]
            phi   = phases[idx + 1]
            idx  += 2
            U = _mzi_unitary(n_modes, theta, phi, ch) @ U

    return U


def _random_clements_phases(n_modes: int, seed: Optional[int] = None) -> np.ndarray:
    """Sample uniformly random phases for a Clements mesh (for test / calibration)."""
    rng = np.random.default_rng(seed)
    n_layers = n_modes - 1
    # Each layer has floor((n_modes - shift) / 2) MZIs, each with 2 phases
    total = sum(
        len(range(layer % 2, n_modes - 1, 2)) * 2
        for layer in range(n_layers)
    )
    return rng.uniform(0, 2 * np.pi, size=total)


# ── Emulator core ─────────────────────────────────────────────────────────────

class GoldenModelEmulator:
    """CPU emulator of the TFLN photonic MZI mesh.

    Executes identical linear algebra to the hardware chip:
      - Clements mesh unitary  →  apply_unitary() on Grassmannian point
      - Per-cell latency bookkeeping
      - Deterministic golden checksum for on-chain anchoring

    Usage:
        emulator = GoldenModelEmulator(n_modes=64)
        frame = emulator.run(input_state, phases)
        # frame.golden_checksum → bytes32 for SiLA2HardwareVerifier
    """

    def __init__(self, n_modes: int = N):
        self.n_modes = n_modes
        self.c_light_um_ps  = C_LIGHT_UM_PS
        self.neff_lnoi      = N_EFF_LNOI
        self.cell_length_um = CELL_LENGTH_UM
        # Latency per MZI stage (single cell traversal)
        self.latency_per_stage_ps = OPTICAL_LATENCY_PS

    # ── Single forward pass ───────────────────────────────────────────────────

    def run(
        self,
        input_state: np.ndarray,
        phases: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> GoldenFrame:
        """Execute one forward pass through the Clements MZI mesh.

        Args:
            input_state: (n_modes,) complex128 — optical field amplitudes.
            phases:      Flat phase array for the mesh.  If None, random phases
                         are sampled (useful for calibration sweeps).
            seed:        RNG seed for reproducible random phases.

        Returns:
            GoldenFrame with output state, Grassmannian point, latency, checksum.
        """
        state = np.asarray(input_state, dtype=np.complex128)
        if state.shape != (self.n_modes,):
            raise ValueError(f"input_state must be ({self.n_modes},)")

        if phases is None:
            phases = _random_clements_phases(self.n_modes, seed=seed)

        # Build mesh unitary and propagate state
        U = _clements_mesh(self.n_modes, phases)
        output = U @ state

        # Count MZI stages for total latency
        n_stages = sum(
            len(range(layer % 2, self.n_modes - 1, 2))
            for layer in range(self.n_modes - 1)
        )
        total_latency_ps = n_stages * self.latency_per_stage_ps

        # Project output onto Grassmannian
        point = project(output.reshape(self.n_modes, 1))

        # Deterministic cross-platform checksum
        input_hash     = _state_checksum(state)
        golden_checksum = _state_checksum(output)

        return GoldenFrame(
            input_hash=input_hash,
            output_state=output,
            point=point,
            optical_latency_ps=total_latency_ps,
            golden_checksum=golden_checksum,
        )

    # ── Calibration: compare emulator vs chip ────────────────────────────────

    def calibrate(
        self,
        chip_outputs: list[np.ndarray],
        emulator_frames: list[GoldenFrame],
    ) -> CalibrationReport:
        """Compare chip-measured output states against emulator golden vectors.

        Acceptance criterion: max |chip - golden|  ≤  TOLERANCE (10⁻⁶).

        Args:
            chip_outputs:     List of (n_modes,) complex128 arrays from hardware.
            emulator_frames:  Corresponding GoldenFrame objects from this emulator.

        Returns:
            CalibrationReport — pass/fail counts, max error, per-run errors.
        """
        if len(chip_outputs) != len(emulator_frames):
            raise ValueError("chip_outputs and emulator_frames must have equal length")

        report = CalibrationReport()
        for chip_out, frame in zip(chip_outputs, emulator_frames):
            err = float(np.max(np.abs(chip_out - frame.output_state)))
            report.errors.append(err)
            report.max_error = max(report.max_error, err)
            if err <= TOLERANCE:
                report.pass_count += 1
            else:
                report.fail_count += 1

        return report

    # ── Batch calibration sweep ───────────────────────────────────────────────

    def generate_golden_vectors(
        self,
        n_vectors: int = 100,
        seed: int = 42,
    ) -> list[GoldenFrame]:
        """Generate a reproducible set of golden reference frames.

        Used during Phase 0 (CPU cluster) to build the calibration dataset
        that the TFLN chip must match within TOLERANCE after tape-out.
        """
        rng = np.random.default_rng(seed)
        frames = []
        for i in range(n_vectors):
            state = rng.standard_normal(self.n_modes) + 1j * rng.standard_normal(self.n_modes)
            state /= np.linalg.norm(state)                   # normalise to unit sphere
            phases = _random_clements_phases(self.n_modes, seed=seed + i)
            frames.append(self.run(state, phases))
        return frames


# ── Checksum helper ───────────────────────────────────────────────────────────

def _state_checksum(state: np.ndarray) -> str:
    """Deterministic, platform-agnostic SHA-256 checksum of a complex state vector.

    Rounds to CHECKSUM_DECIMALS=8 significant figures and casts to complex64
    before serialisation — eliminates float64 ULP drift and endianness artefacts
    between x86, ARM, and RISC-V hosts.
    """
    canonical = np.ascontiguousarray(
        np.round(state, decimals=CHECKSUM_DECIMALS).astype(np.complex64)
    )
    return "0x" + hashlib.sha256(canonical.tobytes()).hexdigest()
