# Negative OpEx Flywheel
**Genesis Live Core — Unit Economics & Rollover LTV Model**
*Records: 6 sections · Date: 2026-09-24 · Status: canonical*

---

## 1. Unit Economics — Single Stream

| Line Item | Value |
|-----------|-------|
| Participants per stream | 3,500,000 |
| Average stake | $2.50 USD |
| **Gross escrow inflow** | **$8,750,000 USD** |
| Reagents + columns + consumables | $28,000 |
| CDN + 4K streaming infrastructure | $31,000 |
| Compute (NGP 4.5 CPU cluster) | $18,000 |
| SiLA 2 instrument maintenance | $16,000 |
| **Total hard OpEx** | **~$93,000 USD** |
| **Coverage ratio** | **> 94×** |

The coverage ratio exceeds 90× by design — ensuring that platform economics
are robust to ±30% participant variance or OpEx overruns without approaching
break-even.

---

## 2. Liquidity Split

Every settled pool distributes as follows (enforced by `GenesisTokenomicsPool.sol`):

```
Total escrow pool
  ├── 10%  →  Hypothesis author grant
  ├── 20%  →  DeSci Foundation Vault (ecosystem treasury)
  └── 70%  →  Platform operational margin
                └── Distributed to Tier 1 Sovereign Foundry Pass holders
                    via accumulatedPlatformPerShare (MasterChef rewardDebt pattern)
```

**Anti-Goodhart enforcement**: if `gcbiScore == 0` (no physical chromatographic
peak detected), the 10% author grant is burned to the platform. Author grant
is computed post-factum exclusively from hardware telemetry — not from NGP 4.5
predictions or human evaluation.

---

## 3. Evergreen Queue — Compute Reuse

NGP 4.5 filter runs once per 10-hypothesis shortlist. Unexecuted hypotheses
carry forward at zero marginal compute cost:

```
Stream N:   NGP 4.5 runs on 1,000 submitted TaskSpecs
            → shortlist[10] generated
            → hypothesis #1 wins (executed in 3D Cube MODR)
            → hypotheses #2..#10 carry over (Zero Slippage)

Stream N+1: NGP 4.5 ingests new submissions only
            → 9 slots already filled from Stream N (no recompute)
            → 1 vacant slot filled from staging queue
```

Marginal compute cost per rollover slot: **$0** (algebra already done in Stream N).
This is the Evergreen Queue property: compute spent once, value compounded across streams.

---

## 4. Rollover LTV Model

**Near-Miss psychology** drives stake accumulation on carried hypotheses.
Each `rolloverCount` increment signals to participants that the hypothesis
has survived repeated rigorous sieving — increasing perceived credibility
and stake density.

### LTV formula

Let:
- $S_0$ = initial stake on a hypothesis at first appearance ($\bar{S}_0 \approx \$875{,}000$ for 10% of stream pool)
- $\alpha$ = stake growth factor per rollover (empirical range: 1.15 – 1.40)
- $n$ = number of rollover cycles before execution

$$\text{LTV}_n = S_0 \cdot \alpha^n$$

### LTV projection table

| Rollovers | α = 1.15 | α = 1.25 | α = 1.40 |
|-----------|----------|----------|----------|
| 0 (initial) | $875k | $875k | $875k |
| 2 | $1.16M | $1.37M | $1.72M |
| 4 | $1.53M | $2.14M | $3.37M |
| 6 | $2.03M | $3.34M | $6.59M |
| 8 | $2.69M | $5.22M | $12.9M |

At α = 1.25 and 5–7 rollover cycles, a single hypothesis pool reaches
**$3.3M – $5.2M** — bringing the top 10-slot portfolio to **$40M–$60M gross**
before the winning run executes.

### Pool LTV (10 slots, 5 streams)

```
Stream 1:  pool = 10 × $875k = $8.75M
Stream 2:  9 carried × 1.25 + 1 new = $9.84M
Stream 3:  9 carried × 1.25 + 1 new = $11.0M
Stream 4:  9 carried × 1.25 + 1 new = $12.3M
Stream 5:  9 carried × 1.25 + 1 new = $13.8M
                                       ────────
                             Running total: ~$55.7M across 5 streams
```

---

## 5. Bootstrap Path

### Phase 0 — Pilot ($250k target bounties)
- Source: DeSci syndicates, patient advocacy groups, pharma precompetitive consortia
- Mechanism: B2B fixed-price hypothesis bounties (not retail micro-stakes)
- Goal: validate full 5-tier pipeline end-to-end with 3–5 funded hypotheses
- OpEx covered by bounty pool; no retail participant acquisition cost

### Phase 1 — Soft Launch (retail scale, target 350k participants)
- Source: Zero-CAC media partnerships (science YouTube, DeSci communities)
- Mechanism: live-streamed experiment events, real-time GCBI leaderboard
- Escrow pool: ~$875k (10% of full-scale)
- Purpose: operational rehearsal, stream infrastructure validation

### Phase 2 — Full Scale (3.5M participants)
- TFLN chip operational (photonic acceleration active)
- Full 10-slot queue running at $8.75M escrow per stream
- Continuous Evergreen Queue cycling

---

## 6. Key Economic Invariants

| Invariant | Enforcement |
|-----------|-------------|
| No stake lost on non-execution | `GenesisRolloverEscrow.rollOverUnexecuted()` preserves `pledgedPool` |
| No fee on rollover | Zero gas burned on carry-forward (state update only) |
| OpEx covered at < 2% of inflow | Hard OpEx $93k / gross $8.75M = 1.06% |
| Author grant conditional on peaks | `GCBIPostFactumEngine.settle()` burns grant if `gcbiScore == 0` |
| Plutocratic stake cannot buy peaks | GCBI computed from ECDSA-signed spectrometer output only |
