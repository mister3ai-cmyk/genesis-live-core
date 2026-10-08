"""
test_sila2_latency.py — Hardware SLA validation
Covers:
  - Hamilton STARlet p99 < 50 ms, mean < 20 ms, compliance_ok == True
  - Waters UPLC Merkle root stability and peak validity
  - EMCCD incremental Merkle proof round-trip (leaf → root)
"""
import hashlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sila2_bridge.hamilton_starlet_driver import (
    HamiltonSTARletDriver, PipetteCommand, WellAddress, PlateFormat,
    P99_LATENCY_MS,
)
from sila2_bridge.waters_uplc_connector import WatersUPLCConnector
from sila2_bridge.emccd_spectrometer_stream import EMCCDSpectrometerStream


# ── Hamilton STARlet latency ──────────────────────────────────────────────────

class TestHamiltonLatency:

    def _make_plate_commands(self, n: int = 96) -> list[PipetteCommand]:
        commands = []
        for i in range(n):
            row, col = divmod(i, 12)
            commands.append(PipetteCommand(
                well=WellAddress(row=row, col=col, plate_format=PlateFormat.W96),
                volume_ul=10.0,
                aspirate=(i % 2 == 0),
                tip_index=i % 8,
            ))
        return commands

    def test_p99_under_50ms(self):
        with HamiltonSTARletDriver() as drv:
            telemetry = drv.run_plate(self._make_plate_commands(96))
        assert telemetry.p99_latency_ms < P99_LATENCY_MS, (
            f"p99 = {telemetry.p99_latency_ms:.2f} ms exceeds {P99_LATENCY_MS} ms SLA"
        )

    def test_mean_latency_under_20ms(self):
        with HamiltonSTARletDriver() as drv:
            telemetry = drv.run_plate(self._make_plate_commands(96))
        latencies = [r.latency_ms for r in telemetry.records]
        mean_ms = sum(latencies) / len(latencies)
        assert mean_ms < 20.0, f"Mean latency {mean_ms:.2f} ms exceeds 20 ms"

    def test_all_commands_compliant(self):
        with HamiltonSTARletDriver() as drv:
            telemetry = drv.run_plate(self._make_plate_commands(96))
        assert telemetry.compliance_ok, "Run compliance_ok must be True"
        non_compliant = [r for r in telemetry.records if not r.compliance_ok]
        assert not non_compliant, f"{len(non_compliant)} non-compliant commands"

    def test_telemetry_hash_is_deterministic(self):
        """Same records must produce same hash (no random salt)."""
        with HamiltonSTARletDriver() as drv:
            t = drv.run_plate(self._make_plate_commands(8))
        h1 = t.telemetry_hash()
        h2 = t.telemetry_hash()
        assert h1 == h2, "telemetry_hash() must be idempotent"

    def test_well_address_label(self):
        assert WellAddress(0, 0).label() == "A1"
        assert WellAddress(7, 11).label() == "H12"
        assert WellAddress(0, 0, PlateFormat.W384).label() == "A1"

    def test_invalid_volume_raises(self):
        with HamiltonSTARletDriver() as drv:
            with pytest.raises(ValueError, match="out of range"):
                drv.execute(PipetteCommand(
                    well=WellAddress(0, 0), volume_ul=0.0, aspirate=True
                ))


# ── Waters UPLC Merkle root ───────────────────────────────────────────────────

class TestWatersUPLC:

    def test_chromatogram_has_valid_peaks(self):
        conn = WatersUPLCConnector()
        frame = conn.acquire_chromatogram()
        assert len(frame.peaks) > 0
        assert all(p.snr >= 3.0 for p in frame.valid_peaks())

    def test_merkle_root_is_32_bytes(self):
        conn = WatersUPLCConnector()
        frame = conn.acquire_chromatogram()
        root = frame.merkle_root()
        assert isinstance(root, bytes) and len(root) == 32

    def test_merkle_root_stable_across_calls(self):
        """Root must not change between calls on the same frame."""
        conn = WatersUPLCConnector()
        frame = conn.acquire_chromatogram()
        assert frame.merkle_root() == frame.merkle_root()

    def test_telemetry_packet_fields(self):
        conn = WatersUPLCConnector()
        frame = conn.acquire_chromatogram()
        packet = conn.sign_and_package(frame)
        assert packet.device_address.startswith("0x")
        assert len(packet.device_address) == 42
        assert packet.signature_hex.startswith("0x")
        assert packet.run_id > 0
        assert packet.merkle_root == frame.merkle_root()

    def test_find_peak_within_tolerance(self):
        conn = WatersUPLCConnector()
        frame = conn.acquire_chromatogram()
        valid = frame.valid_peaks()
        if valid:
            target_rt = valid[0].retention_time_min
            found = conn.find_peak_at(frame, target_rt)
            assert found is not None
            assert abs(found.retention_time_min - target_rt) <= 0.05


# ── EMCCD Merkle proof round-trip ─────────────────────────────────────────────

class TestEMCCDMerkleProof:

    def test_frame_hash_is_32_bytes(self):
        emccd = EMCCDSpectrometerStream()
        packet = emccd.acquire_and_package(n_frames=4, seed=0)
        for frame in packet.session.frames:
            assert len(frame.frame_hash()) == 32

    def test_merkle_root_matches_manual(self):
        """Root computed by session must equal root recomputed from leaf hashes."""
        import hashlib as hl

        def manual_root(leaves):
            nodes = list(leaves)
            while len(nodes) > 1:
                if len(nodes) % 2 == 1:
                    nodes.append(nodes[-1])
                nodes = [hl.sha256(nodes[i] + nodes[i+1]).digest()
                         for i in range(0, len(nodes), 2)]
            return nodes[0]

        emccd = EMCCDSpectrometerStream()
        packet = emccd.acquire_and_package(n_frames=8, seed=1)
        leaves = [f.frame_hash() for f in packet.session.frames]
        assert packet.merkle_root == manual_root(leaves)

    def test_merkle_proof_verifies(self):
        """Proof path must reconstruct the Merkle root for every leaf."""
        emccd = EMCCDSpectrometerStream()
        packet = emccd.acquire_and_package(n_frames=8, seed=2)
        session = packet.session

        for idx in range(len(session.frames)):
            leaf, proof, leaf_idx = session.merkle_proof(idx)
            assert _verify_proof(leaf, proof, leaf_idx, session.merkle_root()), (
                f"Proof failed for frame {idx}"
            )

    def test_packet_frame_count(self):
        n = 25
        emccd = EMCCDSpectrometerStream()
        packet = emccd.acquire_and_package(n_frames=n, seed=3)
        assert packet.frame_count == n
        assert len(packet.session.frames) == n


def _verify_proof(leaf: bytes, proof: list[bytes], index: int, root: bytes) -> bool:
    computed = leaf
    for sibling in proof:
        if index % 2 == 0:
            computed = hashlib.sha256(computed + sibling).digest()
        else:
            computed = hashlib.sha256(sibling + computed).digest()
        index //= 2
    return computed == root
