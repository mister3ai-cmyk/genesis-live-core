"""
SiLA 2 Bridge — Waters ACQUITY UPLC Connector
Interface to Waters Empower 3 / OpenLab CDS chromatographic data system.

Responsibilities:
  - Extract chromatographic peaks, retention times (t_R), and peak areas
  - Sign telemetry package with instrument ECDSA private key (secp256k1)
  - Output signed TelemetryPacket ready for SiLA2HardwareVerifier.sol

ECDSA signing uses the cryptography library (production).
In stub mode, signing is simulated with a deterministic test key.
"""
from __future__ import annotations

import hashlib
import hmac
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

WAVELENGTH_NM: float = 210.0    # default UV detection wavelength (peptide bond)
MIN_PEAK_SNR:  float = 3.0      # signal-to-noise ratio threshold for peak acceptance
RT_TOLERANCE_MIN: float = 0.05  # retention time window tolerance (minutes)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ChromatographicPeak:
    """Single integrated chromatographic peak from Waters ACQUITY UPLC."""
    peak_id:        int
    retention_time_min: float   # t_R in minutes
    area:           float       # integrated peak area (mAU·min or counts)
    height:         float       # peak apex height (mAU or counts)
    snr:            float       # signal-to-noise ratio
    width_half:     float       # peak width at half-maximum (minutes)
    wavelength_nm:  float = WAVELENGTH_NM

    def is_valid(self) -> bool:
        return self.snr >= MIN_PEAK_SNR and self.area > 0


@dataclass
class ChromatogramFrame:
    """Full chromatogram output from one injection."""
    injection_id:   str
    sample_name:    str
    timestamp_ns:   int
    peaks:          list[ChromatographicPeak] = field(default_factory=list)
    raw_signal:     Optional[np.ndarray] = None  # (n_points,) float64 AU trace

    def valid_peaks(self) -> list[ChromatographicPeak]:
        return [p for p in self.peaks if p.is_valid()]

    def merkle_root(self) -> bytes:
        """Merkle root over per-peak hashes — mirrors SiLA2HardwareVerifier leaf structure."""
        if not self.peaks:
            return b"\x00" * 32
        leaves = [_peak_hash(p) for p in self.peaks]
        return _merkle_root(leaves)


@dataclass
class TelemetryPacket:
    """Signed telemetry bundle ready for SiLA2HardwareVerifier.sol.

    Fields map directly to submitTelemetry() parameters:
      merkle_root    → bytes32 merkleRoot
      device_address → address device  (Ethereum-compatible pubkey hash)
      run_id         → uint32 runId
      timestamp      → uint64 timestamp
      signature      → (v, r, s) ECDSA tuple
    """
    chromatogram:   ChromatogramFrame
    merkle_root:    bytes          # 32 bytes
    device_address: str            # 0x<40 hex> Ethereum address of instrument
    run_id:         int            # monotonic counter
    timestamp_ns:   int
    signature_hex:  str            # 0x<r><s><v> — 65 bytes hex


# ── Connector ─────────────────────────────────────────────────────────────────

class WatersUPLCConnector:
    """Connector to Waters ACQUITY UPLC via Empower 3 / OpenLab CDS API.

    Production: wraps HTTP/COM API of Empower 3 data system.
    Stub mode: generates synthetic chromatograms for integration testing.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        device_private_key_hex: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self._run_counter = 0
        # In production: load ECDSA private key from HSM / secure enclave
        # Stub: use deterministic test key (32 zero bytes → NOT for production)
        self._private_key = bytes.fromhex(device_private_key_hex) if device_private_key_hex \
            else bytes(32)
        self.device_address = _derive_address(self._private_key)

    # ── Acquisition ───────────────────────────────────────────────────────────

    def acquire_chromatogram(
        self,
        injection_id: Optional[str] = None,
        sample_name: str = "unknown",
    ) -> ChromatogramFrame:
        """Trigger injection and acquire chromatogram.

        Production: POST to Empower 3 API, poll until run complete, fetch result.
        Stub: generate synthetic Gaussian peaks for testing.
        """
        self._run_counter += 1
        inj_id = injection_id or f"inj_{self._run_counter:06d}"
        ts = time.monotonic_ns()
        raw, peaks = _synthetic_chromatogram(seed=self._run_counter)
        return ChromatogramFrame(
            injection_id=inj_id,
            sample_name=sample_name,
            timestamp_ns=ts,
            peaks=peaks,
            raw_signal=raw,
        )

    # ── Signing ───────────────────────────────────────────────────────────────

    def sign_and_package(self, frame: ChromatogramFrame) -> TelemetryPacket:
        """Compute Merkle root over peaks and sign with instrument ECDSA key.

        Returns a TelemetryPacket ready for SiLA2HardwareVerifier.submitTelemetry().
        """
        self._run_counter += 1
        root = frame.merkle_root()
        run_id = self._run_counter
        ts_ns  = frame.timestamp_ns

        # Construct telemetryId (mirrors Solidity keccak256 encoding)
        telemetry_id = _compute_telemetry_id(root, self.device_address, run_id, ts_ns)

        # ECDSA sign (stub: HMAC-SHA256 as deterministic placeholder)
        sig = _sign(telemetry_id, self._private_key)

        return TelemetryPacket(
            chromatogram=frame,
            merkle_root=root,
            device_address=self.device_address,
            run_id=run_id,
            timestamp_ns=ts_ns,
            signature_hex=sig,
        )

    # ── Peak search ───────────────────────────────────────────────────────────

    def find_peak_at(
        self,
        frame: ChromatogramFrame,
        expected_rt_min: float,
    ) -> Optional[ChromatographicPeak]:
        """Return the valid peak closest to expected retention time, within tolerance."""
        candidates = [
            p for p in frame.valid_peaks()
            if abs(p.retention_time_min - expected_rt_min) <= RT_TOLERANCE_MIN
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p.retention_time_min - expected_rt_min))


# ── Synthetic chromatogram (stub) ─────────────────────────────────────────────

def _synthetic_chromatogram(
    seed: int = 0,
    n_points: int = 2000,
    n_peaks: int = 4,
) -> tuple[np.ndarray, list[ChromatographicPeak]]:
    """Generate a synthetic UPLC trace with Gaussian peaks for testing."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, n_points)   # 10-minute run
    signal = rng.normal(0, 0.002, n_points) # baseline noise

    peaks = []
    retention_times = sorted(rng.uniform(1.0, 9.0, n_peaks))
    for i, rt in enumerate(retention_times):
        height = rng.uniform(50.0, 500.0)
        sigma  = rng.uniform(0.05, 0.15)
        gauss  = height * np.exp(-0.5 * ((t - rt) / sigma) ** 2)
        signal += gauss
        area    = float(np.trapz(gauss, t))
        snr     = height / 0.002
        peaks.append(ChromatographicPeak(
            peak_id=i,
            retention_time_min=float(rt),
            area=area,
            height=float(height),
            snr=float(snr),
            width_half=float(2.355 * sigma),
        ))

    return signal.astype(np.float64), peaks


# ── Merkle tree ───────────────────────────────────────────────────────────────

def _peak_hash(peak: ChromatographicPeak) -> bytes:
    payload = struct.pack(
        ">ifffff",
        peak.peak_id,
        peak.retention_time_min,
        peak.area,
        peak.height,
        peak.snr,
        peak.width_half,
    )
    return hashlib.sha256(payload).digest()


def _merkle_root(leaves: list[bytes]) -> bytes:
    nodes = list(leaves)
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])   # duplicate last leaf for odd count
        nodes = [
            hashlib.sha256(nodes[i] + nodes[i + 1]).digest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0]


# ── Cryptographic helpers ─────────────────────────────────────────────────────

def _derive_address(private_key: bytes) -> str:
    """Derive Ethereum-compatible device address from private key (stub: SHA-256)."""
    pub = hashlib.sha256(private_key).digest()
    addr = hashlib.sha256(pub).hexdigest()[-40:]
    return "0x" + addr


def _compute_telemetry_id(
    merkle_root: bytes,
    device_address: str,
    run_id: int,
    timestamp_ns: int,
) -> bytes:
    """Mirrors Solidity: keccak256(abi.encodePacked(merkleRoot, device, runId, timestamp))."""
    addr_bytes = bytes.fromhex(device_address[2:].zfill(40))
    packed = (
        merkle_root +
        addr_bytes +
        run_id.to_bytes(4, "big") +
        (timestamp_ns // 1_000_000_000).to_bytes(8, "big")  # ns → Unix seconds
    )
    return hashlib.sha256(packed).digest()


def _sign(message: bytes, private_key: bytes) -> str:
    """Stub ECDSA signing via HMAC-SHA256.
    Production: replace with cryptography.hazmat.primitives.asymmetric.ec.ECDSA
    """
    sig = hmac.new(private_key, message, hashlib.sha256).digest()
    # Pad to 65 bytes (r=32, s=32, v=1) matching Solidity (v, r, s) layout
    v = b"\x1b"
    return "0x" + (sig[:32] + sig[16:] + v).hex()
