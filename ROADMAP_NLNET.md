# Genesis Live Core: NLnet NGI Zero Restack Roadmap

*Proposal #2026-11-322, submitted to NLnet NGI Zero Restack. Not yet funded.*
3 milestones · ~600 hours · €42,000 · last updated 2026-10-18

All deliverables are released under the Apache License 2.0 in `src/`, `benchmarks/` and new top-level directories. Nothing in this roadmap depends on `research/` (BSL 1.1).

Baseline figures come from the measurements published in [README.md](README.md): shared Contabo VPS, AMD EPYC 2.0 GHz, AVX2 (no AVX-512), single core.

---

## Work completed since submission, unfunded

Added 2026-11-05. None of this was part of the proposal and none of it is
claimed against its hours; it is recorded because a reviewer checking the
repository will find it and should know where it came from.

**Dynamic agent memory, measured against the obvious alternative.** A stand that
runs a real agent over 40+ linked turns with a retrieval tool and a write tool,
against a control that keeps the whole history in its prompt. Closed corpus of
this project's own documentation, so the model cannot answer from training data.
Both arms graded by a separate judge model that reads the assertion rather than
matching substrings. Code: `research/agent_memory_traffic.py`. Findings:
[`research/docs/AGENT_MEMORY_FINDINGS.md`](research/docs/AGENT_MEMORY_FINDINGS.md).

**Reconsolidation made non-inert.** The specified update rule was an identity —
its target was the operator's own output, which a Newton-Schulz projection
annihilates exactly: `||W - I|| = 5.4e-17` after 2,000 updates, with eighty
tests passing and none of them noticing. The target is now a state-error vector
carried by Oja's rule; plasticity rose from 8.79e-07 to 3.62e-01 with
non-unitarity at 2.5e-07. Covered by four regression tests written red.

**A result we did not expect and did not bury.** Claude Haiku 5.5 begins
refusing innocuous turns once a long history accumulates in its context — 34
refusals in 264 control turns, 12.9 %, against 0 for the memory arm, which
carries no history. Checked on a second model family: Grok 4.6 shows none, so
the effect is specific and is stated at that width. The same cross-model check
also showed a full-context agent answering every chain correctly at 42 turns,
which is a result against our own architecture and is published as such.

**Audit groundwork.** [`BENCHMARKS.md`](BENCHMARKS.md) now names the command
behind each figure, lists the figures withdrawn after re-measurement — including
a 67× memory advantage that was an accounting error and a 31 ns latency that was
a unittest runtime — and lists what is not measured at all. Eight such
corrections are recorded. The reviewer quick start is three commands needing
only GCC, Python and NumPy: 33 C assertions, 51 Python tests, and the
reconsolidation invariants printing their measured values.

This does not change the milestones below, which remain the proposal as
submitted.

## Milestone 1: Core Engine Optimization & AVX-512 SIMD Primitives

**€14,000 · ~200 hours**

**Baseline.** Portable C99 with scalar `POPCNT` for the Hamming scan and a scalar float32 inner product for the re-rank. 128-bit codes, pool of 128, synthetic 50k × 512: 6,300–6,900 QPS, Recall 10@10 88.0 %. 256-bit codes, pool of 512, OpenAI 512-dim: 3,435 QPS, 90.6 %.

**Deliverables**

| # | Deliverable |
| --- | --- |
| 1.1 | Runtime ISA dispatch via `cpuid`: scalar, AVX2 and AVX-512 (`VPOPCNTDQ`, `AVX512BW`) Hamming-scan kernels behind one C function signature. |
| 1.2 | SIMD exact re-rank kernels: AVX2 + FMA and AVX-512F float32 inner product. |
| 1.3 | Scalar fallback stays the reference implementation. The build stays pure C99 on non-x86 hosts. |
| 1.4 | Correctness tests: every kernel returns the same top-k ids and distances as the scalar reference on fixed seeds and on the OpenAI dataset. Run in CI with gcc and clang. |
| 1.5 | Benchmark matrix published in the repository: QPS, p50/p99 latency and recall per ISA, code length (128/256) and candidate pool (128/512), on at least one AVX2 host and one AVX-512 host. |

**Acceptance criteria**

* Top-k results of every SIMD kernel are identical to the scalar reference.
* Recall 10@10 is unchanged against the baseline table.
* On AVX2-only hosts there is no regression compared with the current release.
* Measured speedups are reported as measured. No target figure is assumed in advance.

**Risk.** The current test host has no AVX-512. An AVX-512 machine (bare-metal or cloud instance) is needed for 1.5.

---

## Milestone 2: Native SQLite Virtual Table Extension & ACID Persistence

**€16,000 · ~230 hours**

**Baseline.** The index exists only in memory. A ctypes demo that reads payloads from SQLite lives in `research/genesis-bridge/` (BSL 1.1, not reused). There is no native SQLite integration.

**Deliverables**

| # | Deliverable |
| --- | --- |
| 2.1 | Loadable SQLite extension in C99 registered with `sqlite3_create_module_v2`: `CREATE VIRTUAL TABLE v USING genesis(dim=512, bits=256);` |
| 2.2 | KNN query through `xBestIndex` / `xFilter`, e.g. `SELECT rowid, distance FROM v WHERE embedding MATCH ? ORDER BY distance LIMIT 10;`, joinable with ordinary tables. |
| 2.3 | Persistence in shadow tables of the same database file (codes, raw vectors, encoder parameters), so inserts, updates and deletes follow SQLite transactions, rollback and WAL. |
| 2.4 | Index load on open, with the in-memory stage-1 codes rebuilt from or mapped onto the persisted state. |
| 2.5 | Tests: transaction rollback, interrupted-write recovery, `PRAGMA integrity_check`, fuzzing of the SQL interface. |
| 2.6 | Benchmark of the virtual-table path against direct C calls on the same data. |

**Acceptance criteria**

* After a rolled-back or interrupted transaction, query results match the committed state.
* Recall through SQL equals recall of the C library on the same data and parameters.
* The virtual-table overhead per query is measured and published.
* The extension builds with the SQLite amalgamation alone and needs no other dependencies.

---

## Milestone 3: Local Runtime Bindings, Air-Gapped Tooling & Documentation

**€12,000 · ~170 hours**

**Baseline.** The C headers (`src/srp_lsh.h`, `src/shm_ring.h`) are the only interface. No bindings or CLI ship in this repository.

**Deliverables**

| # | Deliverable |
| --- | --- |
| 3.1 | Language bindings over the C ABI, Apache-2.0, in this repository: Python, Rust (`-sys` crate plus a thin safe wrapper), Go (cgo). |
| 3.2 | Offline CLI: `build` (from `.fvecs` / `.npy`), `query`, `verify` (recall against exact brute force) and `export`. No network access at any point. |
| 3.3 | Air-gapped build: release tarball with vendored dependencies (SQLite amalgamation), SHA-256 checksums, and build instructions for a machine without network access. |
| 3.4 | Documentation: architecture, C ABI reference, on-disk formats, benchmarking methodology, and privacy/threat-model notes (what data is stored where, and what never leaves the host). |

**Acceptance criteria**

* Full build and test suite pass on a host with networking disabled, starting from the release tarball.
* Each binding passes the same recall test as the C library.
* The documentation covers every public C function and on-disk structure.

---

## Summary

| Milestone | Budget | Effort |
| --- | --- | --- |
| 1. Core Engine Optimization & AVX-512 SIMD Primitives | €14,000 | ~200 h |
| 2. Native SQLite Virtual Table Extension & ACID Persistence | €16,000 | ~230 h |
| 3. Local Runtime Bindings, Air-Gapped Tooling & Documentation | €12,000 | ~170 h |
| **Total** | **€42,000** | **~600 h** |
