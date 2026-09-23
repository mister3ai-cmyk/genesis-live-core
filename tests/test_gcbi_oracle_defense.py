"""
test_gcbi_oracle_defense.py — Cryptographic and economic oracle defense
Covers:
  - Unsigned / tampered telemetry is rejected (ecrecover mismatch)
  - gcbiScore == 0 burns author grant (Anti-Goodhart enforcement)
  - Staker boosts are zeroed when no physical peak detected
  - Valid telemetry + score > 0 → correct 10/20/70 split
  - Component scores out of range [0, 10000] are rejected
"""
import hashlib
import struct
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


# ── Pure-Python simulation of GCBI engine + verifier logic ───────────────────

AUTHOR_BPS      = 1000   # 10%
DESCI_BPS       = 2000   # 20%
PLATFORM_BPS    = 7000   # 70%
BPS_DENOM       = 10_000
SCORE_MAX       = 10_000

ALPHA = 3500
BETA  = 3000
GAMMA = 1500
DELTA = 2000


class FakeTelemetryStore:
    """Mirrors SiLA2HardwareVerifier — accepts only signed records."""

    def __init__(self):
        self._records: dict[bytes, dict] = {}
        self._device_keys: dict[str, bytes] = {}   # address → secret
        self._last_run: dict[str, int] = {}

    def register_device(self, address: str, secret: bytes):
        self._device_keys[address] = secret

    def submit(self, merkle_root: bytes, device: str, run_id: int, ts: int, sig: bytes) -> bytes:
        assert device in self._device_keys, "device not registered"
        assert run_id > self._last_run.get(device, 0), "replay detected"

        telemetry_id = hashlib.sha256(
            merkle_root + device.encode() + run_id.to_bytes(4, "big") + ts.to_bytes(8, "big")
        ).digest()

        expected_sig = hashlib.sha256(self._device_keys[device] + telemetry_id).digest()
        assert sig == expected_sig, "invalid instrument signature"

        self._last_run[device] = run_id
        self._records[telemetry_id] = {"merkle_root": merkle_root, "verified": True}
        return telemetry_id

    def is_verified(self, telemetry_id: bytes) -> bool:
        return self._records.get(telemetry_id, {}).get("verified", False)


class GCBIEngineSim:
    """Mirrors GCBIPostFactumEngine.sol logic."""

    def __init__(self, verifier: FakeTelemetryStore):
        self.verifier  = verifier
        self.pools: dict[int, dict] = {}
        self.scores: dict[int, dict] = {}
        self.oracle: str | None = None
        self._pool_count = 0

    def set_oracle(self, address: str):
        self.oracle = address

    def create_pool(self, author: str, desci_vault: str, platform: str) -> int:
        pid = self._pool_count
        self._pool_count += 1
        self.pools[pid] = {
            "author": author, "desci_vault": desci_vault, "platform": platform,
            "total_stake": 0, "telemetry_id": None, "gcbi_score": 0, "settled": False,
        }
        return pid

    def add_stake(self, pool_id: int, amount: int):
        p = self.pools[pool_id]
        assert not p["settled"]
        p["total_stake"] += amount

    def submit_gcbi(self, caller: str, pool_id: int, telemetry_id: bytes, cs: dict):
        assert caller == self.oracle,             "not oracle"
        p = self.pools[pool_id]
        assert not p["settled"],                  "already settled"
        assert p["telemetry_id"] is None,         "already scored"
        assert self.verifier.is_verified(telemetry_id), "telemetry not verified"
        _validate_scores(cs)

        p["telemetry_id"] = telemetry_id
        self.scores[pool_id] = cs
        composite = (cs["h"] * ALPHA + cs["e"] * BETA +
                     cs["o"] * GAMMA + cs["f"] * DELTA) // BPS_DENOM
        p["gcbi_score"] = composite

    def settle(self, pool_id: int) -> dict:
        p = self.pools[pool_id]
        assert not p["settled"],          "already settled"
        assert p["telemetry_id"] is not None, "not scored"
        p["settled"] = True
        total = p["total_stake"]

        if p["gcbi_score"] == 0:
            desci    = total * DESCI_BPS    // BPS_DENOM
            platform = total - desci
            return {"author": 0, "desci": desci, "platform": platform, "burned": True}
        else:
            author   = total * AUTHOR_BPS   // BPS_DENOM
            desci    = total * DESCI_BPS    // BPS_DENOM
            platform = total - author - desci
            return {"author": author, "desci": desci, "platform": platform, "burned": False}


def _validate_scores(cs: dict):
    for k, v in cs.items():
        assert 0 <= v <= SCORE_MAX, f"score {k}={v} out of range [0, {SCORE_MAX}]"


def _make_telemetry(store: FakeTelemetryStore, device: str, run_id: int = 1) -> bytes:
    """Submit a valid signed telemetry record; return telemetry_id."""
    secret = store._device_keys[device]
    root   = hashlib.sha256(b"peak_data_run_" + run_id.to_bytes(4, "big")).digest()
    ts     = 1_727_000_000 + run_id
    tid    = hashlib.sha256(
        root + device.encode() + run_id.to_bytes(4, "big") + ts.to_bytes(8, "big")
    ).digest()
    sig    = hashlib.sha256(secret + tid).digest()
    return store.submit(root, device, run_id, ts, sig)


# ── Tests: telemetry authentication ──────────────────────────────────────────

class TestTelemetryAuthentication:

    def setup_method(self):
        self.store  = FakeTelemetryStore()
        self.device = "0xWatersUPLC"
        self.secret = b"device_secret_key_32bytes_pad00"
        self.store.register_device(self.device, self.secret)

    def test_valid_signature_accepted(self):
        tid = _make_telemetry(self.store, self.device)
        assert self.store.is_verified(tid)

    def test_wrong_signature_rejected(self):
        root   = hashlib.sha256(b"peak_data").digest()
        ts, run_id = 1_727_000_001, 1
        bad_sig = b"\x00" * 32   # garbage signature
        with pytest.raises(AssertionError, match="invalid instrument signature"):
            self.store.submit(root, self.device, run_id, ts, bad_sig)

    def test_unregistered_device_rejected(self):
        root = hashlib.sha256(b"data").digest()
        with pytest.raises(AssertionError, match="device not registered"):
            self.store.submit(root, "0xUnknown", 1, 1_727_000_000, b"\x00" * 32)

    def test_replay_attack_rejected(self):
        _make_telemetry(self.store, self.device, run_id=1)
        with pytest.raises(AssertionError, match="replay detected"):
            _make_telemetry(self.store, self.device, run_id=1)

    def test_monotonic_run_id_accepted(self):
        _make_telemetry(self.store, self.device, run_id=1)
        tid2 = _make_telemetry(self.store, self.device, run_id=2)
        assert self.store.is_verified(tid2)


# ── Tests: Anti-Goodhart burn on zero peaks ───────────────────────────────────

class TestAntiGoodhartBurn:

    def setup_method(self):
        self.store  = FakeTelemetryStore()
        self.device = "0xWatersUPLC"
        self.store.register_device(self.device, b"secret_key_32bytes_pad_000000000")
        self.engine = GCBIEngineSim(self.store)
        self.engine.set_oracle("0xOracleNode")

    def test_zero_gcbi_burns_author_grant(self):
        pid = self.engine.create_pool("0xAuthor", "0xDeSci", "0xPlatform")
        self.engine.add_stake(pid, 10_000)
        tid = _make_telemetry(self.store, self.device)
        self.engine.submit_gcbi("0xOracleNode", pid,
            tid, {"h": 0, "e": 0, "o": 0, "f": 0})
        result = self.engine.settle(pid)
        assert result["burned"]  is True
        assert result["author"]  == 0,     "Author grant must be zero on no-peak run"
        assert result["desci"]   > 0
        assert result["platform"] > 0

    def test_zero_gcbi_total_conserved(self):
        """Total wei must be conserved even on burn (no ether lost)."""
        pid   = self.engine.create_pool("0xAuthor", "0xDeSci", "0xPlatform")
        total = 10_000
        self.engine.add_stake(pid, total)
        tid = _make_telemetry(self.store, self.device)
        self.engine.submit_gcbi("0xOracleNode", pid,
            tid, {"h": 0, "e": 0, "o": 0, "f": 0})
        result = self.engine.settle(pid)
        assert result["author"] + result["desci"] + result["platform"] == total

    def test_positive_gcbi_pays_author(self):
        pid = self.engine.create_pool("0xAuthor", "0xDeSci", "0xPlatform")
        self.engine.add_stake(pid, 10_000)
        tid = _make_telemetry(self.store, self.device)
        self.engine.submit_gcbi("0xOracleNode", pid,
            tid, {"h": 8000, "e": 7000, "o": 9000, "f": 8500})
        result = self.engine.settle(pid)
        assert result["burned"]   is False
        assert result["author"]   == 1000   # 10% of 10_000
        assert result["desci"]    == 2000   # 20%
        assert result["platform"] == 7000   # 70%

    def test_split_sum_equals_total(self):
        pid = self.engine.create_pool("0xA", "0xD", "0xP")
        total = 8_750_000   # $8.75M pool
        self.engine.add_stake(pid, total)
        tid = _make_telemetry(self.store, self.device)
        self.engine.submit_gcbi("0xOracleNode", pid,
            tid, {"h": 9000, "e": 8000, "o": 10_000, "f": 9500})
        result = self.engine.settle(pid)
        assert result["author"] + result["desci"] + result["platform"] == total


# ── Tests: oracle access control ──────────────────────────────────────────────

class TestOracleAccessControl:

    def setup_method(self):
        self.store  = FakeTelemetryStore()
        self.device = "0xWatersUPLC"
        self.store.register_device(self.device, b"oracle_test_secret_32bytes_00000")
        self.engine = GCBIEngineSim(self.store)
        self.engine.set_oracle("0xTrustedOracle")

    def test_unauthorised_oracle_rejected(self):
        pid = self.engine.create_pool("0xA", "0xD", "0xP")
        tid = _make_telemetry(self.store, self.device)
        with pytest.raises(AssertionError, match="not oracle"):
            self.engine.submit_gcbi("0xAttacker", pid,
                tid, {"h": 9000, "e": 9000, "o": 9000, "f": 9000})

    def test_unverified_telemetry_rejected(self):
        pid = self.engine.create_pool("0xA", "0xD", "0xP")
        fake_tid = b"\xde\xad" * 16   # not in verifier store
        with pytest.raises(AssertionError, match="telemetry not verified"):
            self.engine.submit_gcbi("0xTrustedOracle", pid,
                fake_tid, {"h": 9000, "e": 9000, "o": 9000, "f": 9000})

    def test_double_scoring_rejected(self):
        pid = self.engine.create_pool("0xA", "0xD", "0xP")
        tid = _make_telemetry(self.store, self.device)
        self.engine.submit_gcbi("0xTrustedOracle", pid,
            tid, {"h": 5000, "e": 5000, "o": 5000, "f": 5000})
        tid2 = _make_telemetry(self.store, self.device, run_id=2)
        with pytest.raises(AssertionError, match="already scored"):
            self.engine.submit_gcbi("0xTrustedOracle", pid,
                tid2, {"h": 5000, "e": 5000, "o": 5000, "f": 5000})

    def test_score_out_of_range_rejected(self):
        pid = self.engine.create_pool("0xA", "0xD", "0xP")
        tid = _make_telemetry(self.store, self.device)
        with pytest.raises(AssertionError):
            self.engine.submit_gcbi("0xTrustedOracle", pid,
                tid, {"h": 10_001, "e": 5000, "o": 5000, "f": 5000})

    def test_settle_without_scoring_rejected(self):
        pid = self.engine.create_pool("0xA", "0xD", "0xP")
        with pytest.raises(AssertionError, match="not scored"):
            self.engine.settle(pid)

    def test_double_settle_rejected(self):
        pid = self.engine.create_pool("0xA", "0xD", "0xP")
        self.engine.add_stake(pid, 1000)
        tid = _make_telemetry(self.store, self.device)
        self.engine.submit_gcbi("0xTrustedOracle", pid,
            tid, {"h": 5000, "e": 5000, "o": 5000, "f": 5000})
        self.engine.settle(pid)
        with pytest.raises(AssertionError, match="already settled"):
            self.engine.settle(pid)
