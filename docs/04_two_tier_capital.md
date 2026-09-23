# Two-Tier Capital Structure
**Genesis Live Core — $7.5M Round Allocation & Regulatory Framing**
*Records: 5 sections · Date: 2026-09-24 · Status: canonical*

---

## 1. Round Overview

| Parameter | Value |
|-----------|-------|
| Total round size | $7,500,000 USD |
| Structure | Contour 1 (active) + Contour 2 (milestone reserve) |
| Instrument | SAFE / convertible, DeSci-native terms |
| Jurisdiction | DIFC / ADGM (UAE) or Switzerland |
| License | BSL 1.1 → Apache 2.0 (Change Date: 2029-09-24) |

---

## 2. Contour 1 — Active Committed Allocation ($5,000,000)

| Item | Amount | Rationale |
|------|--------|-----------|
| TFLN/LNOI photonic co-processor (fabless MPW, EDA, fiber packaging) | $2,000,000 | Core IP asset of the round; two MPW shuttle slots at LioniX / IMEC |
| Hardware lab: 3D Cube MODR (Hamilton STARlet, Waters UPLC, ARETUSA laser) | $1,400,000 | ISO 5/7 cleanroom build-out, instrument procurement, SiLA 2 commissioning |
| Team & Scientific Advisory Board (18 months) | $1,000,000 | Core engineering (photonics, robotics, smart contracts), 3 SAB members |
| Legal, compliance, infrastructure runway | $600,000 | BSL licensing, DIFC/Swiss entity formation, CDN, cloud compute |

### TFLN chip timeline
```
Month  0–2:   EDA layout + tape-out submission to foundry (LioniX MPW shuttle)
Month  3–8:   Wafer fabrication (parallel to Phase 0 operations — zero downtime)
Month  9–12:  Die packaging + fiber array attachment
Month 12–14:  Golden Model verification (tolerance 10⁻⁶ against CPU reference)
Month 15+:    TFLN chip operational — 18B states/sec photonic acceleration
```

---

## 3. Contour 2 — Milestone Reserve Buffer ($2,500,000)

Protected escrow released only on trigger events:

| Trigger | Max draw | Description |
|---------|----------|-------------|
| Tape-Out cycle 3 (yield failure) | $800,000 | Additional MPW shuttle if yield < 40% on cycle 2 |
| Spectrometer replacement | $350,000 | Hamamatsu EMCCD swap if dark-current drift exceeds spec |
| LC-MS column pre-financing | $400,000 | 24-month reagent and column stockpile for stream scaling |
| Optical path replacement | $250,000 | MZI array re-fibre if insertion loss > 3 dB after packaging |
| General hardware contingency | $700,000 | Unallocated buffer for unforeseen instrument failures |

**Unconsumed funds rule**: any Contour 2 balance not drawn by Month 18
auto-converts to reagent pre-financing for Streams 2–5, extending operational
runway without a follow-on raise.

---

## 4. Golden Model Bridge — Zero Pipeline Stall

The TFLN photonic chip is the **target asset** of the round, not a prerequisite
for starting operations. The software Golden Model (`ngp45_engine/golden_model_emulator.py`)
runs identical linear algebra on CPU/AVX-512 during chip fabrication.

```
Phase 0 (Months 0–8, CPU cluster):
  GoldenModelEmulator.generate_golden_vectors(n=100)
  → calibrates Hamilton robots and Waters UPLC against reference spectra
  → validates Grassmannian algebra and Clements mesh unitarity
  → generates reference vectors for chip acceptance test

Phase 1 (Months 3–8, parallel fab):
  TFLN wafer at foundry
  → zero operational downtime
  → first live streams can run on CPU emulator

Phase 2 (Months 12–15, integration):
  chip.output vs emulator.output: max|Δ| ≤ 10⁻⁶ (TOLERANCE)
  → if CalibrationReport.passed: activate photonic acceleration
  → if CalibrationReport.fail_count > 0: trigger Contour 2 Tape-Out draw
```

**Economic consequence**: the platform earns revenue from live streams
during Phases 0–1 (CPU mode) while the chip is being fabricated.
By the time the chip arrives, the Evergreen Queue has been validated
across 2–3 streams and the calibration dataset is production-ready.

---

## 5. Regulatory Status

| Dimension | Status |
|-----------|--------|
| Activity scope | In Vitro High-Throughput Screening (HTS) + fundamental analytical mapping |
| Human subjects | None — subjects are solutions, protein crystals, and cell biomarkers |
| FDA IND requirement | Not required (21 CFR Part 58 GLP, not clinical) |
| IRB requirement | Not required (no human or animal subjects) |
| Laboratory standard | GLP (Good Laboratory Practice) + ISO 17025 |
| Data classification | Pre-competitive scientific publication (open data from automated screenings) |
| Jurisdiction | DIFC / ADGM (UAE) or Switzerland |
| IP protection | BSL 1.1 (commercial use requires license) + Zenodo DOI prior art chain |

### Why no FDA IND / IRB
Genesis Live operates exclusively in the **in vitro analytical** domain:
- Waters UPLC measures chromatographic retention times of molecular solutions
- Hamamatsu EMCCD measures optical density and fluorescence spectra
- Hamilton STARlet dispenses reagents into 96/384-well plates

None of these activities involve human subjects, animal models, or clinical
decision-making. The regulatory category is equivalent to a university
analytical chemistry laboratory — governed by GLP and ISO 17025, not by
FDA IND or IRB oversight.

### Prior art chain
This repository is the 9th anchor in the DeepTech Proofs series.
Each DOI in the chain cross-references all previous anchors,
creating an immutable, globally verifiable intellectual property record:

| # | Title | DOI |
|---|-------|-----|
| 01 | Biorock Electrochemical Accretion | 10.5281/zenodo.22798026 |
| 02 | SWAC & Venturi Microclimate | 10.5281/zenodo.22800377 |
| 03 | Reversible Photonic-Spin Computing | 10.5281/zenodo.22816555 |
| 04 | THz-Driven SIRT6 & Waddington | 10.5281/zenodo.22816994 |
| 05 | SiLA 2 Closed-Loop DryLab | 10.5281/zenodo.22818737 |
| 06 | Non-Hermitian D(0) Energy Catalysis | 10.5281/zenodo.22819187 |
| 07 | Post-Quantum Swarm Intelligence | 10.5281/zenodo.22819532 |
| 08 | LNOI-WDM-DISPATCHER-v1.0 | 10.5281/zenodo.22884782 |
| **GL** | **Genesis Live Core (this repository)** | **10.5281/zenodo.22926047** |
