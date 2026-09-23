"""
SiLA 2 Bridge — EMCCD Spectrometer Stream
ARETUSA laser module (337.1 nm) + Hamamatsu EMCCD detector.

Responsibilities:
  - Stream spectral frames (optical density + fluorescence) in real time
  - Build Merkle tree over frame hashes for 4K broadcast and on-chain verification
  - Emit signed SpectralPacket compatible with SiLA2HardwareVerifier.sol
  - Support live viewer validation: raw sensor data visible per frame
"""
from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

LASER_WAVELENGTH_NM:  float = 337.1     # ARETUSA nitrogen laser excitation
DETECTOR_PIXELS:      int   = 1024      # Hamamatsu EMCCD horizontal pixels
SPECTRAL_RANGE_NM:    tuple = (200.0, 800.0)  # detection window
FRAME_RATE_HZ:        float = 25.0      # 4K broadcast frame rate
DEFAULT_EXPOSURE_MS:  float = 40.0      # exposure time per frame (= 1/25 s)
DARK_CURRENT_COUNTS:  float = 2.5       # EMCCD dark counts per pixel per frame
READ_NOISE_ELECTRONS: float = 1.2       # Hamamatsu EMCCD read noise spec


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SpectralFrame:
    """Single EMCCD acquisition frame."""
    frame_index:    int
    timestamp_ns:   int
    wavelengths_nm: np.ndarray   # (DETECTOR_PIXELS,) float64
    counts:         np.ndarray   # (DETECTOR_PIXELS,) float64 — raw ADU counts
    exposure_ms:    float = DEFAULT_EXPOSURE_MS

    def optical_density(self, reference: np.ndarray) -> np.ndarray:
        """OD = -log10(I / I_0).  Clips to avoid log(0)."""
        ratio = np.clip(self.counts / np.clip(reference, 1e-9, None), 1e-9, None)
        return -np.log10(ratio)

    def peak_wavelength(self) -> float:
        """Wavelength of maximum emission."""
        return float(self.wavelengths_nm[np.argmax(self.counts)])

    def fluorescence_integral(self, lo_nm: float = 350.0, hi_nm: float = 700.0) -> float:
        """Integrated fluorescence counts in [lo_nm, hi_nm] window."""
        mask = (self.wavelengths_nm >= lo_nm) & (self.wavelengths_nm <= hi_nm)
        return float(np.trapz(self.counts[mask], self.wavelengths_nm[mask]))

    def frame_hash(self) -> bytes:
        """Deterministic SHA-256 leaf for Merkle tree construction."""
        header = struct.pack(">Iqf", self.frame_index, self.timestamp_ns, self.exposure_ms)
        payload = header + self.counts.astype(np.float32).tobytes()
        return hashlib.sha256(payload).digest()


@dataclass
class SpectralSession:
    """Accumulated frames and Merkle tree for a full spectral run."""
    session_id:     str
    frames:         list[SpectralFrame] = field(default_factory=list)
    _leaf_hashes:   list[bytes]         = field(default_factory=list, repr=False)

    def add_frame(self, frame: SpectralFrame) -> None:
        self.frames.append(frame)
        self._leaf_hashes.append(frame.frame_hash())

    def merkle_root(self) -> bytes:
        """Incremental Merkle root over all frames received so far."""
        if not self._leaf_hashes:
            return b"\x00" * 32
        return _merkle_root(self._leaf_hashes)

    def merkle_proof(self, frame_index: int) -> tuple[bytes, list[bytes], int]:
        """Return (leaf_hash, proof_path, leaf_index) for on-chain verification."""
        if frame_index >= len(self._leaf_hashes):
            raise IndexError(f"Frame {frame_index} not in session")
        leaf = self._leaf_hashes[frame_index]
        proof = _merkle_proof(self._leaf_hashes, frame_index)
        return leaf, proof, frame_index


@dataclass
class SpectralPacket:
    """Signed spectral telemetry bundle for SiLA2HardwareVerifier.sol."""
    session:        SpectralSession
    merkle_root:    bytes
    device_address: str
    run_id:         int
    timestamp_ns:   int
    frame_count:    int
    signature_hex:  str


# ── Spectrometer stream ───────────────────────────────────────────────────────

class EMCCDSpectrometerStream:
    """Stream controller for ARETUSA 337.1 nm + Hamamatsu EMCCD detector.

    Production: wraps SiLA 2 gRPC ObservableProperty for continuous acquisition.
    Stub: generates synthetic fluorescence + Raman-shifted spectral frames.

    4K broadcast pipeline:
      acquire_frame() → add_frame() → merkle_root() updated per frame
      → SpectralPacket signed → on-chain anchor every N frames (configurable)
    """

    def __init__(
        self,
        device_private_key_hex: Optional[str] = None,
        anchor_every_n_frames: int = 25,   # anchor Merkle root every 1 second @ 25 fps
    ):
        self._private_key   = bytes.fromhex(device_private_key_hex) if device_private_key_hex \
            else bytes(32)
        self.device_address = _derive_address(self._private_key)
        self.anchor_every   = anchor_every_n_frames
        self._run_counter   = 0
        self._wavelengths   = np.linspace(*SPECTRAL_RANGE_NM, DETECTOR_PIXELS)

    # ── Reference acquisition ─────────────────────────────────────────────────

    def acquire_reference(self, seed: int = 0) -> np.ndarray:
        """Acquire dark-corrected reference spectrum (laser without sample)."""
        return _synthetic_laser_spectrum(self._wavelengths, seed=seed)

    # ── Streaming acquisition ─────────────────────────────────────────────────

    def stream(
        self,
        n_frames: int,
        session_id: Optional[str] = None,
        seed: int = 0,
    ) -> Generator[SpectralFrame, None, SpectralSession]:
        """Generator: yields SpectralFrame objects, returns completed SpectralSession.

        Usage:
            gen = emccd.stream(100, session_id="run_001")
            for frame in gen:
                display(frame)   # 4K broadcast hook
            session = gen.value  # after exhaustion
        """
        self._run_counter += 1
        sid = session_id or f"session_{self._run_counter:06d}"
        session = SpectralSession(session_id=sid)

        for i in range(n_frames):
            frame = self._acquire_frame(i, seed=seed + i)
            session.add_frame(frame)
            yield frame

        return session

    def acquire_and_package(
        self,
        n_frames: int,
        session_id: Optional[str] = None,
        seed: int = 0,
    ) -> SpectralPacket:
        """Acquire n_frames, build Merkle tree, sign, return SpectralPacket."""
        self._run_counter += 1
        sid = session_id or f"session_{self._run_counter:06d}"
        session = SpectralSession(session_id=sid)
        ts_start = time.monotonic_ns()

        for i in range(n_frames):
            frame = self._acquire_frame(i, seed=seed + i)
            session.add_frame(frame)

        root = session.merkle_root()
        run_id = self._run_counter
        sig = _sign(root, self._private_key, run_id)

        return SpectralPacket(
            session=session,
            merkle_root=root,
            device_address=self.device_address,
            run_id=run_id,
            timestamp_ns=ts_start,
            frame_count=len(session.frames),
            signature_hex=sig,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _acquire_frame(self, index: int, seed: int = 0) -> SpectralFrame:
        """Production: read from EMCCD via SiLA 2 gRPC. Stub: synthetic frame."""
        rng = np.random.default_rng(seed)
        counts = _synthetic_sample_spectrum(self._wavelengths, rng)
        return SpectralFrame(
            frame_index=index,
            timestamp_ns=time.monotonic_ns(),
            wavelengths_nm=self._wavelengths.copy(),
            counts=counts,
        )


# ── Synthetic spectra (stub) ──────────────────────────────────────────────────

def _synthetic_laser_spectrum(wavelengths: np.ndarray, seed: int = 0) -> np.ndarray:
    """Simulate ARETUSA 337.1 nm reference laser profile."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(DARK_CURRENT_COUNTS, READ_NOISE_ELECTRONS, len(wavelengths))
    # Rayleigh scatter peak at excitation wavelength
    signal += 50_000 * np.exp(-0.5 * ((wavelengths - LASER_WAVELENGTH_NM) / 2.0) ** 2)
    return np.clip(signal, 0, None).astype(np.float64)


def _synthetic_sample_spectrum(
    wavelengths: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate fluorescence emission + Raman shift from sample."""
    signal = rng.normal(DARK_CURRENT_COUNTS, READ_NOISE_ELECTRONS, len(wavelengths))
    # Broad fluorescence emission band (420–600 nm)
    signal += 8_000 * np.exp(-0.5 * ((wavelengths - 480.0) / 60.0) ** 2)
    # Raman-shifted peak (+~1450 cm⁻¹ from 337.1 nm → ~365 nm)
    signal += 2_000 * np.exp(-0.5 * ((wavelengths - 365.0) / 5.0) ** 2)
    # Laser scatter residual
    signal += 500  * np.exp(-0.5 * ((wavelengths - LASER_WAVELENGTH_NM) / 1.5) ** 2)
    return np.clip(signal, 0, None).astype(np.float64)


# ── Merkle helpers ────────────────────────────────────────────────────────────

def _merkle_root(leaves: list[bytes]) -> bytes:
    nodes = list(leaves)
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(nodes[i] + nodes[i + 1]).digest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0]


def _merkle_proof(leaves: list[bytes], leaf_index: int) -> list[bytes]:
    """Return sibling-hash proof path for a leaf (bottom-up)."""
    nodes = list(leaves)
    proof = []
    idx = leaf_index
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        sibling = idx ^ 1
        proof.append(nodes[sibling])
        nodes = [
            hashlib.sha256(nodes[i] + nodes[i + 1]).digest()
            for i in range(0, len(nodes), 2)
        ]
        idx //= 2
    return proof


# ── Crypto helpers ────────────────────────────────────────────────────────────

def _derive_address(private_key: bytes) -> str:
    pub  = hashlib.sha256(private_key).digest()
    addr = hashlib.sha256(pub).hexdigest()[-40:]
    return "0x" + addr


def _sign(root: bytes, private_key: bytes, run_id: int) -> str:
    """Stub signing — production: replace with secp256k1 ECDSA."""
    import hmac
    message = root + run_id.to_bytes(4, "big")
    sig = hmac.new(private_key, message, hashlib.sha256).digest()
    v = b"\x1b"
    return "0x" + (sig[:32] + sig[16:] + v).hex()
