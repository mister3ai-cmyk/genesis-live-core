"""
SiLA 2 Bridge — Hamilton Microlab STARlet Driver
gRPC service stub for liquid handling: aspirate/dispense on 96/384-well plates.

Latency contract: p99 < 50 ms per command cycle.
All commands are logged with monotonic timestamps for SiLA2HardwareVerifier anchoring.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

P99_LATENCY_MS: float = 50.0      # SiLA 2 latency contract
PLATE_FORMATS  = {96: (8, 12), 384: (16, 24)}   # rows × cols
MIN_VOLUME_UL  = 0.5
MAX_VOLUME_UL  = 1000.0


# ── Data model ────────────────────────────────────────────────────────────────

class PlateFormat(Enum):
    W96  = 96
    W384 = 384


class CommandStatus(Enum):
    OK      = auto()
    ERROR   = auto()
    TIMEOUT = auto()


@dataclass
class WellAddress:
    row: int    # 0-based
    col: int    # 0-based
    plate_format: PlateFormat = PlateFormat.W96

    def label(self) -> str:
        rows, cols = PLATE_FORMATS[self.plate_format.value]
        if not (0 <= self.row < rows and 0 <= self.col < cols):
            raise ValueError(f"Address ({self.row},{self.col}) out of range for {self.plate_format}")
        return f"{chr(65 + self.row)}{self.col + 1}"


@dataclass
class PipetteCommand:
    well:         WellAddress
    volume_ul:    float
    aspirate:     bool          # True = aspirate, False = dispense
    tip_index:    int = 0       # physical tip channel (0–7 for STARlet 8-channel)


@dataclass
class CommandRecord:
    command_id:    str
    command:       PipetteCommand
    status:        CommandStatus
    latency_ms:    float
    timestamp_ns:  int          # monotonic nanoseconds
    compliance_ok: bool         # passive compliance check passed


@dataclass
class RunTelemetry:
    """Aggregated telemetry for a full pipetting run — hashed for on-chain anchoring."""
    run_id:        str
    records:       list[CommandRecord] = field(default_factory=list)
    p99_latency_ms: float = 0.0
    compliance_ok:  bool  = True

    def telemetry_hash(self) -> str:
        """Deterministic SHA-256 over all command records in run order."""
        payload = b"".join(
            r.command_id.encode() +
            r.status.name.encode() +
            r.timestamp_ns.to_bytes(8, "big")
            for r in self.records
        )
        return "0x" + hashlib.sha256(payload).hexdigest()


# ── Driver ────────────────────────────────────────────────────────────────────

class HamiltonSTARletDriver:
    """SiLA 2 gRPC service stub for Hamilton Microlab STARlet.

    In production this class wraps a real gRPC channel to the SiLA 2 server
    running on the instrument PC.  Here it provides a validated software
    interface with latency tracking and compliance checks — ready for
    integration tests and golden-model calibration runs.
    """

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self._connected = False
        self._run_counter = 0

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Establish gRPC channel to SiLA 2 server on instrument PC."""
        # Production: grpc.insecure_channel(f"{self.host}:{self.port}")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def __enter__(self) -> "HamiltonSTARletDriver":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Core command ──────────────────────────────────────────────────────────

    def execute(self, cmd: PipetteCommand) -> CommandRecord:
        """Execute a single aspirate or dispense command.

        Validates volume range, records wall-clock latency, checks p99 contract.
        In production: serialises to SiLA2 protobuf and sends over gRPC.
        """
        if not self._connected:
            raise RuntimeError("HamiltonSTARlet: not connected")

        _validate_volume(cmd.volume_ul)

        t0 = time.monotonic_ns()
        status = self._send_command(cmd)
        t1 = time.monotonic_ns()

        latency_ms = (t1 - t0) / 1e6
        compliance  = latency_ms < P99_LATENCY_MS and status == CommandStatus.OK

        return CommandRecord(
            command_id=_command_id(cmd, t0),
            command=cmd,
            status=status,
            latency_ms=latency_ms,
            timestamp_ns=t0,
            compliance_ok=compliance,
        )

    # ── Plate run ─────────────────────────────────────────────────────────────

    def run_plate(
        self,
        commands: list[PipetteCommand],
        run_id: Optional[str] = None,
    ) -> RunTelemetry:
        """Execute an ordered list of commands; aggregate into RunTelemetry."""
        self._run_counter += 1
        run_id = run_id or f"run_{self._run_counter:06d}"
        telemetry = RunTelemetry(run_id=run_id)

        for cmd in commands:
            record = self.execute(cmd)
            telemetry.records.append(record)
            if not record.compliance_ok:
                telemetry.compliance_ok = False

        if telemetry.records:
            latencies = sorted(r.latency_ms for r in telemetry.records)
            p99_idx   = max(0, int(len(latencies) * 0.99) - 1)
            telemetry.p99_latency_ms = latencies[p99_idx]

        return telemetry

    # ── Stub: simulates gRPC round-trip ──────────────────────────────────────

    def _send_command(self, cmd: PipetteCommand) -> CommandStatus:
        """Production: gRPC call. Stub: simulate instrument latency ~2–8 ms."""
        simulated_ms = np.random.uniform(2.0, 8.0)
        time.sleep(simulated_ms / 1000.0)
        return CommandStatus.OK


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_volume(volume_ul: float) -> None:
    if not (MIN_VOLUME_UL <= volume_ul <= MAX_VOLUME_UL):
        raise ValueError(f"Volume {volume_ul} µL out of range [{MIN_VOLUME_UL}, {MAX_VOLUME_UL}]")


def _command_id(cmd: PipetteCommand, timestamp_ns: int) -> str:
    payload = f"{cmd.well.label()}:{cmd.volume_ul}:{cmd.aspirate}:{timestamp_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
