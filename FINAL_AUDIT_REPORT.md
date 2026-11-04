# Pre-flight audit report: genesis-live-core

**2026-11-05. Self-audit by the project, not a third-party certification. Covers
the commit tagged `v4.6-preflight-2026-11-05` (5 checklist blocks, 84 automated checks, 12 strict compiles).
NLnet proposal #2026-11-322, not yet funded.**

**Environment:** clean `git clone` on WSL2 (kernel 5.10), GCC 15.2, Python 3.14.4,
pytest 9.0.2, NumPy 2.3.5. This is not the Contabo host behind the published
latency figures, so latency was not compared across hosts.

## Result

| block | result |
|---|---|
| 1. Secrets and privacy | pass, with notes |
| 2. C static analysis | pass, one analyzer warning reviewed as a false positive |
| 3. Automated tests | pass: 33 C checks and 51 Python tests, 0 failures |
| 4. Documentation against code | pass after corrections; figures from LLM runs not re-run |
| 5. Report and tag | this file; tag `v4.6-preflight-2026-11-05` |

## 1. Secrets and privacy

- Patterns for API keys, GitHub, Telegram, Slack, xAI, Google and AWS tokens and
  private keys, searched over the working tree and `git log --all -p`: 0 matches.
- This is a pattern search, not a dedicated scanner such as gitleaks.
- `.gitignore` now also covers `tests/test_srp_lsh`, `benchmarks/cold_cache`,
  `venv/`, `.venv/`, `*.dylib` and `*.log`. After a full build and test run,
  `git status` shows no untracked files.
- `/tmp/genesis_eval` is a documented cache path and is left as is.
  `research/agent_memory_traffic.py:66` names the private corpus on the
  measurement host; it is not a credential, but see the limits in section 4.
- Nine investor and tokenomics documents under `research/` were replaced by
  notices at the same paths. `GCBIPostFactumEngine.sol` and
  `GenesisRolloverEscrow.sol` remain.
  `research/docs/01_five_tier_topology.md` and `03_gcbi_post_factum_math.md`
  still describe that mechanism.

## 2. C static analysis

Flags: `-Wall -Wextra -pedantic -std=c11 -O3`, default build and
`-DSRP_BITS=256 -DK_CAND=512`, over `src/srp_lsh.c`, `src/shm_ring.c`,
`tests/test_srp_lsh.c`, `benchmarks/run_benchmark.c`, `benchmarks/cold_cache.c`
and `research/genesis-bridge/l0_api.c`.

| check | result |
|---|---|
| strict compile, 12 builds | 0 warnings |
| ASan + UBSan + leak check, C tests, both configurations | 33 checks, 0 reports |
| ASan + UBSan, `run_benchmark` including the SHM mailbox | exit 0, 0 reports |
| `gcc -fanalyzer`, 5 non-test sources | 1 warning, reviewed below |

`src/srp_lsh.c:95`, uninitialized `best[p - 1]`: false positive. `p` never exceeds
`filled`, and `best[0..filled-1]` is written before it is read; the analyzer does
not track that invariant. The code was not changed. ASan does not detect
uninitialized reads, and neither MSan nor Valgrind was run.

Known precondition: `srp_dot` reads 16 values at a time, so `dim` must be a
multiple of 16. This is documented at `src/srp_lsh.h:58`, and `run_benchmark` and
`l0_api.c` enforce it, but `srp_dot` itself does not check it.

## 3. Automated tests

| command | result |
|---|---|
| `cd tests && make check` | 33 checks, 0 failures |
| `python3 -m pytest` (repository root) | 51 passed |
| `python3 research/tests/test_reconsolidation_plasticity.py` | exit 0 |
| `cd benchmarks && make && ./run_benchmark` | recall 10@10 88.0 %, index 0.80 MB |

The invariant script prints `||W - I|| = 3.623e-01` (threshold > 5e-02),
`||W^H W - I|| = 4.444e-06` (threshold < 1e-04) and, for the inert control,
`||W - I|| = 8.781e-07`.

CI (`.github/workflows/benchmark.yml`) runs the C tests and the benchmark and now
also the Python tests and the invariant script. Those commands were checked in a
clean virtual environment; the new CI job first runs on push, so its result is not
part of this audit.

Of the 84 checks, the 33 C checks cover the Apache-2.0 core. The 51 Python tests
cover code under `research/`, which is BSL 1.1 and outside the NLnet deliverables.

## 4. Documentation against code

| figure | status | where |
|---|---|---|
| non-unitarity, stress test: 4.44e-06 | **re-run** | `BENCHMARKS.md:30` |
| non-unitarity, sessions: 9.9e-08 – 4.1e-07 | not re-run; files agree | `README.md:359`, `BENCHMARKS.md:124` |
| memory share of the loop: 0.354 %, range 0.16–0.36 % | not re-run; files agree | `README.md:353`, `BENCHMARKS.md:61` |
| prompt growth: +25 tokens/turn against +2,170 | not re-run; files agree | `README.md:354`, `BENCHMARKS.md:59` |
| false refusals: 0 of 264 turns against 34 of 264 | not re-run; files agree | `README.md:358`, `research/docs/AGENT_MEMORY_FINDINGS.md:414` |
| read through `Memory.search`: 580–605 µs | not re-run; printed by `--mode independent` | `README.md:352` |
| `Glyphs.search` alone: 271 µs | **no command prints it** | `README.md:351` |

Limits on these figures:

- **"Not re-run"** means the files were checked against each other, not reproduced.
  `research/agent_memory_traffic.py` needs paid model APIs and a private corpus, so
  an outside reviewer cannot reproduce those figures from this repository alone. On
  a clean host it also expects `/tmp/genesis_eval` to exist already.
- **271 µs** was not traced to a stored run. A separate timing of `Glyphs.search`
  on the audit machine (4,000 random 384-dim vectors, 1,000 queries) gave p50
  306–314 µs alone and 359–361 µs with weights and 20 slots. This supports the order
  of magnitude only. That path is NumPy; only encoding is C.
- **Scope:** +25 tokens/turn holds for documents per turn, at 40 % against 38 %
  accuracy, with no advantage at 16 turns. With facts per turn, full context is
  cheaper. Zero false refusals was measured against 34 on Haiku 5.5; on Grok 4.6 the
  control also refused 0 times.

Corrected in this audit:

- Non-unitarity has three thresholds for three quantities, now labelled: 1e-04
  stress test, 1e-05 agent session, 1e-06 gate on the second corrective step. The
  code agrees (`research/tests/test_reconsolidation_plasticity.py:49`,
  `research/agent_memory_traffic.py:1168`,
  `research/modules/ngp46_reconsolidation_engine.py:131`).
- Stale values in the plasticity test notes and
  `research/docs/RECONSOLIDATION_FINDINGS.md`: 3.60e-01 and 2.81e-06 are now
  3.62e-01 and 4.44e-06; the stability cost is now a factor of 2.5, not 1.6.
- Latency rows now name the code path each figure measures.

**Licensing.** The repository has two parts:

- `src/`, `tests/` and `benchmarks/` are Apache-2.0. The NLnet proposal builds on
  this part (`ROADMAP_NLNET.md`).
- `research/` is BSL 1.1 (`research/LICENSE`): an experimental benchmarking stand
  that also holds internal material, outside the NLnet deliverables. BSL 1.1 is not
  an open-source license. The memory stand behind the section 4 figures lives here.

Eight files under `research/` carried Apache-2.0 headers and now follow
`research/LICENSE`; versions already published under those headers remain available
under Apache-2.0. `benchmarks/bench_chroma.py` is Apache-2.0 but compares against
`genesis-l0` from PyPI, which is licensed separately under BSL 1.1 (`README.md:219`).
`benchmarks/scan_sweep.sh` gained its missing SPDX line.

**Unverified claims.** Older material under `research/`, including `synvision/`,
`sila2_bridge/` and `containment/`, was searched for claims the project has
withdrawn but was not corrected. It is not covered by this audit.

## 5. Reproduce

```
cd tests && make check && cd ..
python3 -m pytest
python3 research/tests/test_reconsolidation_plasticity.py
```
