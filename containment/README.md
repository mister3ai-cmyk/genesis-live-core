# ngp-preemptive-ai-containment

**NGP 4.5 Module 6 — Deterministic Pre-emptive AI Containment**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22964910.svg)](https://doi.org/10.5281/zenodo.22964910)
[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](../../LICENSE)
[![CI](https://github.com/mister3ai-cmyk/genesis-live-core/actions/workflows/verify_containment.yml/badge.svg)](https://github.com/mister3ai-cmyk/genesis-live-core/actions/workflows/verify_containment.yml)

---

## License

**Business Source License 1.1 (BSL-1.1)**  
Licensor: Maksym Babych / Synapse Core Infrastructure  
Change Date: **2029-09-24** → Apache License 2.0  
Full text: [LICENSE](../../LICENSE)

Non-commercial, research, and educational use is free of charge.  
Commercial use requires a separate written license: research@syn.ai

---

## Summary

Solves the **Containment Latency Paradox** for physical AI agents operating high-speed tooling:

$$\tau_{\text{eval}} + \tau_{\text{intercept}} > \tau_{\text{actuation}}$$

**Solution:** Decouple into two strictly separated channels:
- **Fast path (O(1)):** `DeterministicKinematicOracle` — analytical linear extrapolation, `τ_eval ≤ 1 µs`
- **Slow path:** `TimesFMSidecarBridge` — 200M transformer for macro-trends (10 s horizon, outside safety loop)

Hardware intercept via `ORingPreemptiveBarrierGuard` suppresses stepper motor pulses **before** actuation when predicted trajectory exits the 3D Cube MODR boundary.

---

## Benchmark Results

| Metric | Value | Requirement |
|--------|-------|-------------|
| Oracle latency p99 | **0.000834 ms** | < 0.01 ms ✅ |
| Hazardous trajectory intercepted | **true** | before motor pulse ✅ |
| OOM predictive flush triggered | **1** | ≥ 1 ✅ |
| RAM limit (2GB edge node) | **750 MB** | configurable ✅ |

---

## DOI Chain

| Record | DOI |
|--------|-----|
| Master Concept | [10.5281/zenodo.22944521](https://doi.org/10.5281/zenodo.22944521) |
| This Version | [10.5281/zenodo.22964910](https://doi.org/10.5281/zenodo.22964910) |
| Genesis Live Core | [10.5281/zenodo.22926047](https://doi.org/10.5281/zenodo.22926047) |

---

## Quick Start

```bash
python preemptive_containment_sentinel_v2_1.py
```

Expected output: `"status": "ALL_SYSTEMS_VERIFIED"`

## Theory

See [THEORY_PREEMPTIVE_CONTAINMENT.md](THEORY_PREEMPTIVE_CONTAINMENT.md) for full mathematical formalization.
