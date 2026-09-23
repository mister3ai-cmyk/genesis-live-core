# GCBI Post-Factum Mathematics
**Genesis Live Core — Anti-Goodhart Oracle Specification**
*Records: 5 sections · Date: 2026-09-24 · Status: canonical*

---

## 1. Goodhart's Law and Why Pre-Factum Scoring Fails

> *"When a measure becomes a target, it ceases to be a good measure."*
> — Charles Goodhart, 1975

In standard prediction markets, reward scores are computed before or during
the experiment. This creates a single attack surface: any agent who can
influence the scoring function can capture rewards without producing science.

Genesis Live breaks this attack surface by enforcing a strict temporal
separation:

| Phase | Process | Scoring? |
|-------|---------|---------|
| **Pre-factum** | NGP 4.5 sieve, Grassmannian filter, Poincaré gate | ❌ No reward scoring |
| **In-vitro** | SiLA 2 gRPC execution in ISO 5 cleanroom | ❌ No reward scoring |
| **Post-factum** | Hardware telemetry verified on-chain | ✅ GCBI computed here only |

The sieve's job is **feasibility** (can this run in a robot?), not **value**
(how much is this worth?). These two questions are answered by different
systems at different times with different data sources.

---

## 2. GCBI Formula

$$\text{GCBI} = \frac{\alpha \cdot S_{\text{health}} + \beta \cdot S_{\text{energy}} + \gamma \cdot S_{\text{openAccess}} + \delta \cdot S_{\text{feasibility}}}{10{,}000}$$

### Component weights (basis points, sum = 10,000)

| Component | Symbol | Weight | Description |
|-----------|--------|--------|-------------|
| ΔHealthspan | $S_{\text{health}}$ | α = 3500 | Healthy life-years generated (SIRT6, LINE-1, DunedinPACE) |
| ΔEnergy | $S_{\text{energy}}$ | β = 3000 | Clean energy output or thermal savings (LENR, SWAC) |
| OpenAccess | $S_{\text{openAccess}}$ | γ = 1500 | Degree of knowledge decentralisation (Zenodo DOI, no patents) |
| HardwareFeasibility | $S_{\text{feasibility}}$ | δ = 2000 | SiLA 2 compliance, p99 latency, RT-error < 2% |

### Solidity implementation (`GCBIPostFactumEngine.sol::submitGCBI`):
```solidity
uint16 composite = uint16(
    (uint32(cs.healthspan)  * 3500 +
     uint32(cs.energy)      * 3000 +
     uint32(cs.openAccess)  * 1500 +
     uint32(cs.feasibility) * 2000) / 10_000
);
```

### Score bounds
- Each component: `uint16` in `[0, 10_000]` — enforced by `_validateScores()`
- Composite: `[0, 10_000]` — `0` triggers Anti-Goodhart burn

---

## 3. Hardware Oracle Chain

GCBI scores are not submitted by humans or NGP 4.5 predictions.
They are submitted by an authorised oracle **after** hardware verification.

```
Waters ACQUITY UPLC
  │  chromatographic peaks (t_R, area, SNR)
  │  ECDSA-signed with device key (secp256k1)
  ▼
SiLA2HardwareVerifier.submitTelemetry(merkleRoot, device, runId, ts, v, r, s)
  │  ecrecover(ethHash, v, r, s) == device   ← replay-protected by runId nonce
  ▼
telemetryId = keccak256(merkleRoot ‖ device ‖ runId ‖ timestamp)
  │  stored on-chain as immutable record
  ▼
GCBIPostFactumEngine.submitGCBI(poolId, telemetryId, ComponentScores)
  │  require(verifier.isVerified(telemetryId))  ← telemetry gate
  │  oracle computes S_health, S_energy, S_openAccess, S_feasibility
  │  from ECDSA-signed spectrometer output only
  ▼
GCBI composite score committed to pool
```

### Merkle frame verification
Each spectral frame from the Hamamatsu EMCCD is a Merkle leaf:
```python
frame_hash = SHA-256(struct.pack(">Iqf", frame_index, timestamp_ns, exposure_ms)
                     + counts.astype(float32).tobytes())
```
Any viewer can submit `verifyFrame(telemetryId, frameHash, proof, leafIndex)`
to the `SiLA2HardwareVerifier` contract and confirm that a specific 25-fps
frame was part of the verified run. This is the basis for live audience
validator consensus.

---

## 4. Anti-Goodhart Enforcement: Three Attack Vectors Closed

### Attack 1 — Plutocratic stake amplification
**Threat**: large staker artificially boosts a hypothesis's escrow pool,
forcing NGP 4.5 to rank it higher.

**Defense**: NGP 4.5 sieve operates on `TaskSpec` content only — `pledgedPool`
is not an input to `isoperimetric_filter.py::run_sieve()`. Stake accumulates
in `GenesisRolloverEscrow` independently of the filter ranking.

### Attack 2 — Oracle capture (fake peaks)
**Threat**: attacker bribes oracle to submit high GCBI scores without a
real experimental run.

**Defense**:
```solidity
require(verifier.isVerified(telemetryId), "GCBI: telemetry not verified");
```
`telemetryId` is only accepted if `SiLA2HardwareVerifier` has a record with
`verified = true`. That record requires an ECDSA signature from the instrument's
hardware key — a key stored in the cleanroom controller's HSM, not accessible
to external agents.

### Attack 3 — Fake peak submission
**Threat**: attacker generates synthetic chromatogram data and signs it with
a stolen device key.

**Defense**: device key revocation via `SiLA2HardwareVerifier.revokeDevice()`.
Additionally, Waters UPLC firmware signs each injection with a monotonic
`runId` — replay attacks are detected by `require(runId > lastRunId[device])`.

---

## 5. Settlement Logic: Burn on Zero Peaks

```
gcbiScore > 0  →  Normal settlement
  author   =  totalStake × 10%
  desci    =  totalStake × 20%
  platform =  totalStake × 70%

gcbiScore == 0  →  Anti-Goodhart burn
  author   =  0                          ← grant annulled
  desci    =  totalStake × 20%
  platform =  totalStake × 80%           ← absorbs author share
  emit BoostBurned(poolId, totalStake)
```

**Invariant**: `author + desci + platform == totalStake` in all branches —
no ether is destroyed, conservation of capital is maintained.

**Economic consequence**: a participant who stakes on a hypothesis with
`gcbiScore == 0` loses their relative upside (no author grant to amplify)
but their principal is redistributed within the system, not burned to
`address(0)`. This preserves DeFi composability while enforcing scientific integrity.
