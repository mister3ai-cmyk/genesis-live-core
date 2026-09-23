# Five-Tier Pipeline Topology
**Genesis Live Core — Architectural Specification**
*Records: 5 tiers · Date: 2026-09-24 · Status: canonical*

---

## Overview

Genesis Live transforms a researcher's hypothesis into a globally verifiable,
cryptographically anchored experimental result through five deterministic tiers.
No human intermediary exists between Tier 1 ingestion and Tier 5 DOI release.

```
[ Tier 1 ]         [ Tier 2 ]          [ Tier 3 ]           [ Tier 4 ]        [ Tier 5 ]
  Global      ──►   NGP 4.5        ──►  Rollover        ──►  3D Cube      ──►  DeSci
  Ingestion         In Silico           Escrow               MODR              DOI
  (TaskSpec)        (Sieve 99%)         (Micro-stake)        (SiLA 2)          (Zenodo)
```

---

## Tier 1 — Global Ingestion Layer

### Purpose
Open gateway accepting structured experimental proposals from any researcher,
institution, or citizen scientist worldwide.

### Input formats

**`TaskSpec`** — machine-readable hypothesis descriptor:
```json
{
  "hypothesis_id":  "string (UUID v4)",
  "description":    "string (max 2048 chars)",
  "reagents":       ["string", ...],
  "temperature_k":  "float  (Kelvin)",
  "pressure_atm":   "float  (atmospheres)",
  "delta_g_kjmol":  "float  (kJ/mol, negative = spontaneous)",
  "metadata":       {}
}
```

**`InstrumentCard`** — hardware execution constraints:
```json
{
  "task_spec_id":    "UUID ref to TaskSpec",
  "sila2_profile":   "string (SiLA 2 Feature URI)",
  "volume_ul":       "float",
  "plate_format":    "96 | 384",
  "uplc_method":     "string (Empower 3 method name)",
  "detection_nm":    "float (UV wavelength)",
  "run_duration_min":"float"
}
```

### Canonical hash
Each `TaskSpec` is hashed on submission:
```
spec_hash = SHA-256(
  JSON.stringify({id, reagents.sort(), T_K, P_atm, dG_kJ_mol}, sort_keys=true)
)
```
This `bytes32` value is stored in `GenesisRolloverEscrow.HypothesisSlot.taskSpecHash`
and serves as the permanent on-chain fingerprint of the proposal.

---

## Tier 2 — NGP 4.5 In Silico Core (Isoperimetric Sieve)

### Purpose
Automatic elimination of ≥ 99% pseudo-scientific noise and thermodynamically
impossible reactions. Outputs a ranked shortlist of ≤ 10 hardware-executable
hypotheses per cycle.

### Knowledge base
- 5,346 verified scientific sources
- Domains: non-Hermitian quantum mechanics, SIRT6/NAD⁺ biophysics, SiLA 2 / ICH Q14 standards

### Mathematical pipeline

**Step 1 — Complex embedding** (`ngp45_engine/isoperimetric_filter.py::_embed_spec`):
```
TaskSpec  →  C^64 vector
  channels  0–15 : reagent SHA-256 fingerprints (real part)
  channels  0– 2 : thermodynamic scalars T, P, ΔG (imaginary part)
  channels 16–17 : dimensionless Gibbs criterion, RT product
  channels 32–47 : metadata hash expansion
```

**Step 2 — Grassmannian projection** (`ngp45_engine/grassmannian_manifold.py::project`):
```
C^64  →  G(4, C^64)   via thin SVD
  dim_R = 2 × 4 × (64 − 4) = 480
  padded to 512D SIMD cache-line for AVX-512 / TFLN alignment
```

**Step 3 — Dual gate**:
```
Poincaré ball gate:   ‖v‖_tanh > 0.97   → REJECT  (geometric noise)
Thermodynamic gate:   thermo_score < 0.15 → REJECT  (infeasible ΔG / T / P)
```

**Step 4 — Ranking**:
```
composite_score = thermo_score × (1 − boundary_penalty)
shortlist = top_10(survivors, key=composite_score)
```

### Photonic acceleration
On Phase 2 hardware, Steps 1–3 are executed by the TFLN co-processor:
- Switching latency: τ ≤ 38.0 ps per MZI stage (cell length 5.15 µm)
- Throughput: ≥ 18 × 10⁹ combinatorial evaluations / second
- WDM: 64 active C-band carriers @ 1550 nm

---

## Tier 3 — Rollover Escrow Layer

### Purpose
Collect micro-stakes on each shortlisted hypothesis and manage the 10-slot
active queue across streams. Unexecuted hypotheses carry their stake into the
next stream without fee (Zero Slippage).

### On-chain contracts
```
GenesisRolloverEscrow.sol
  └── GCBIPostFactumEngine.sol
        └── SiLA2HardwareVerifier.sol
              └── (ECDSA device keys: Waters UPLC, Hamilton STARlet, Hamamatsu)
```

### Queue lifecycle
```
addHypothesis(slot, taskSpecHash, author, gcbiPoolId)   // 10 slots populated
stake(slotIndex)  { value: $1–$5 in ETH/USDC/L2 }      // micro-bets accumulate
markExecuted(winner, telemetryId, ComponentScores)       // hardware run complete
rollOverUnexecuted()                                      // 9 slots carry forward
```

### Stake routing
`stake()` immediately forwards `msg.value` to `GCBIPostFactumEngine.addStake()`.
No double custody. `GenesisRolloverEscrow` holds only the `stakerBalance` mapping.

---

## Tier 4 — In Vitro Execution (3D Cube MODR Cleanroom)

### Physical environment
| Parameter | Specification |
|-----------|---------------|
| Classification | ISO 5 (core) / ISO 7 (buffer) |
| Human access | Zero — fully automated under 4K camera array |
| Control protocol | SiLA 2 gRPC, p99 < 50 ms |
| RT-error | < 2% across full run |

### Instrument stack

**Hamilton Microlab STARlet** (`sila2_bridge/hamilton_starlet_driver.py`):
- 96 / 384-well plate handling, 0.5–1000 µL range
- Compliance check per command, p99 latency enforced in software
- `RunTelemetry.telemetry_hash()` → SHA-256 anchor

**Waters ACQUITY UPLC + Tandem MS** (`sila2_bridge/waters_uplc_connector.py`):
- Chromatographic peak extraction: t_R, area, height, SNR ≥ 3.0
- Merkle root over per-peak hashes → `TelemetryPacket`
- ECDSA-signed with instrument device key → `SiLA2HardwareVerifier.submitTelemetry()`

**ARETUSA 337.1 nm + Hamamatsu EMCCD** (`sila2_bridge/emccd_spectrometer_stream.py`):
- 1024-pixel spectral range 200–800 nm, 25 fps acquisition
- Incremental Merkle tree: root updated per frame
- `merkle_proof(frame_index)` → `SiLA2HardwareVerifier.verifyFrame()`

### SiLA 2 gRPC latency contract
```
p99_latency_ms < 50.0       enforced per CommandRecord
mean_latency_ms < 20.0      soft target for throughput
RT-error < 2%               validated by compliance_ok flag
```

---

## Tier 5 — Crystallization Layer (DeSci Prior Art)

### Purpose
Permanent anchoring of verified experimental results to the global scientific
record via Zenodo / OpenAIRE DOI.

### Pipeline
```
1. GCBI post-factum score computed from hardware telemetry
2. GCBIPostFactumEngine.settle() distributes stakes (10/20/70 split)
3. Merkle root of full run (Hamilton + Waters + EMCCD) committed on-chain
4. DOI generated: Zenodo REST API v1
   upload_type: "publication", publication_type: "preprint"
5. DOI badge embedded in repository README (permanent prior art anchor)
```

### DOI schema
```json
{
  "upload_type":       "publication",
  "publication_type":  "preprint",
  "title":             "<Hypothesis description>",
  "creators":          [{"name": "<author>"}],
  "description":       "<TaskSpec JSON + run summary>",
  "license":           "cc-by-4.0",
  "keywords":          ["DeSci", "SiLA2", "NGP4.5", "Genesis Live"]
}
```

### Immutability guarantee
Once published, the Zenodo DOI is immutable. The on-chain `telemetryId`
(stored in `GCBIPostFactumEngine.HypothesisPool.telemetryId`) cross-references
the DOI, creating a dual-layer prior art anchor:
- **Blockchain layer**: `telemetryId = keccak256(merkleRoot, device, runId, timestamp)`
- **Archive layer**: Zenodo DOI with MD5 checksum of raw data bundle
