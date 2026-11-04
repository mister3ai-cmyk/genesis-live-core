# Reconsolidation: what was measured

**2026-11-03. Every figure here comes from a script in this repository, named
beside it. Measured on a Contabo VPS, AMD EPYC, single core.**

RFC-4.6-001 specifies a phase operator that reconsolidates on read. This records
what that operator does when run, including the parts that did not work and were
fixed, because the specification asserted properties nobody had measured.

---

## 1. The update rule was an identity

`modules/ngp46_reconsolidation_engine.py`, before the fix:

```
||W - I|| = 5.4e-17   after 2,000 updates
```

The rule reinforced the mapping the operator already performed — target equal to
its own output — which is a tautology. What little remained was removed by the
Newton-Schulz projection: a Hermitian rank-one perturbation of a unitary matrix
has that same matrix as its unitary polar factor, so the projection annihilates
it exactly. One update of Frobenius norm 1.5e-02 measures 3.39e-04 after the
step.

Eighty tests in this repository passed throughout and none of them noticed.

**Fix.** `generate_context` now takes an external target. The update is Oja's
subspace rule, `dW = eta * (x y^H - y y^H W)`, which is not unitary and therefore
carries information; Newton-Schulz then projects back onto the Grassmannian. Oja
learns, Newton-Schulz cleans up, in that order.

| | `\|\|W - I\|\|` | non-unitarity |
|---|---|---|
| no target (previous path, kept) | 8.79e-07 | 1.76e-06 |
| external target | **3.62e-01** | 4.44e-06 |

Plasticity rises by five orders of magnitude; stability costs a factor of 2.5,
absorbed by the existing norm clipping and corrective step.

Covered by `research/tests/test_reconsolidation_plasticity.py`, written red:
three of its four tests failed against the old rule.

---

## 2. It is not a retrieval optimiser, and cannot be

A unitary operator preserves every inner product. Rotating both query and corpus
changes nothing — measured: top-10 identical, scores agree to 8.26e-16. Rotating
only the query merely misaligns it from the fixed corpus: 1 of 10 results
survives.

**There is no W that encodes "these items are related".** The associative
capacity of a unitary query transform is zero by construction, independent of
the update rule or the learning rate.

Seven experiments measuring retrieval recall found nothing, which is the correct
result. Recall belongs to the two-stage cascade, which reaches 0.906.

---

## 3. What it is for, measured

Retention of old associations while new ones arrive — the standard
continual-learning shape. `research/retention_experiment.py`, dim 256, twelve
old associations then thirty-six new, the old never revisited. Chance is 0.048.

| rounds | arm | retained after phase 1 | after phase 2 | forgetting | non-unitarity |
|---|---|---|---|---|---|
| 120 | naive accumulation | 0.862 | 0.716 | 0.146 | 9.6 |
| 120 | reconsolidation | 0.656 | 0.568 | **0.088** | 5.8e-07 |
| 300 | naive accumulation | 0.930 | 0.614 | 0.316 | 17.0 |
| 300 | reconsolidation | 0.702 | 0.581 | **0.121** | 5.9e-07 |
| 800 | naive accumulation | 0.841 | **0.426** | 0.415 | 29.0 |
| 800 | reconsolidation | 0.703 | **0.581** | **0.122** | 5.9e-07 |

At 75 % of capacity — forty-eight old and one hundred forty-four new — forgetting
is 0.114 for naive accumulation against **0.054** for reconsolidation.

### The result is convergence, not a smaller number

Reconsolidation reaches a fixed point: its figures at 300 and 800 rounds agree to
three decimals, so further training changes nothing. Naive accumulation never
settles — retention falls 0.716, 0.614, 0.426 while non-unitarity climbs 9.6, 17,
29. It destroys what it learned by continuing to learn.

The crossover sits between 300 and 800 rounds, after which reconsolidation leads
on absolute retention too.

### The cost, stated

Reconsolidation peaks lower: 0.70 against 0.93. Naive accumulation is
unconstrained linear fitting and is strictly more expressive than a rotation, so
it should fit better. It does. It does not keep what it fits, and it leaves the
manifold to get there — at non-unitarity 29 the operator is no longer a rotation
and no longer the same object.

---

## 4. Figures withdrawn

**`coherence 99.85 %`** in `NGP_4_6_RECONSOLIDATION_SPEC-v2` traces to
`modules/hyper_tetrapod_timesfm_sidecar.py:156`, a hardcoded constant for
dissipation suppression in the photonic module. The reconsolidation engine
computes no quality metric of any kind. The figure does not belong to this
component.

**`Static RAG accuracy: 0 %`** in the same section is a tautology: a static index
does not change on read by definition. The inference to worse accuracy was drawn
without measurement.

---

## 5. Reproducing all of it

```bash
python3 -m pytest research/tests/test_reconsolidation_plasticity.py -q
python3 research/retention_experiment.py --dim 256 --rounds 800
python3 research/reconsolidation_experiment.py     # capacity of the unitary path
python3 research/ngp_4_6_live_benchmark_stand.py   # 7 passed, 0 failed
```
