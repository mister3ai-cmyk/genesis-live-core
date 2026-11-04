# Benchmarks: every figure, its command, and what it does not prove

**Updated 2026-11-04. 84 automated checks: 33 C assertions, 51 Python tests.
Measurement host for every figure below: shared Contabo VPS, AMD EPYC (with
IBPB), 6 vCPU, 11 GB RAM, Debian 12, GCC 13.3, Python 3.11, single core unless
stated.**

This file exists so that no number in this repository has to be taken on trust.
Each row names the command that produces it. Where a figure has a caveat, the
caveat is in the same row — not in a footnote.

---

## 1. Verifiable without an API key or network access

These are the three commands a reviewer should run. They need GCC, Python 3 and
NumPy, nothing else.

```bash
cd tests && make check && cd ..                              # 33 assertions
python3 -m pytest research/tests/ -q                         # 51 tests
python3 research/tests/test_reconsolidation_plasticity.py    # prints the values
```

| figure | value | command |
|---|---|---|
| C core assertions | 33 checks, 0 failures | `cd tests && make check` |
| Python tests | 51 passed | `python3 -m pytest research/tests/ -q` |
| plasticity, `\|\|W - I\|\|` after 500 updates | 3.62e-01 (threshold > 5e-02) | step 3 above |
| non-unitarity, `\|\|W^H W - I\|\|` | 4.44e-06 (threshold < 1e-04) | step 3 above |
| inert control (target = own output) | 8.78e-07 after 200 updates | step 3 above |
| static cascade: recall 10@10 | 88.0 % at 6,300–6,900 QPS/core | `cd benchmarks && make && ./run_benchmark` |
| index footprint | 0.80 MB against 102.4 MB float32 (128×) | same |
| cross-process SHM handoff, one-way | p50 30–290 ns across 8 runs | same |
| retention under continual learning | forgetting 0.122 against 0.415 naive | `python3 research/retention_experiment.py --dim 256 --rounds 800` |

The two mutation checks matter more than the count: flipping `>=` to `>` in
`srp_encode` and corrupting the accumulator in `srp_dot` are both caught.

## 2. Requires an LLM API key

The agent-memory stand calls a commercial API and downloads a sentence encoder
on first use, so it cannot be part of the offline set. Stated plainly rather
than presented as reproducible-for-free.

```bash
export ANTHROPIC_API_KEY=...
python3 research/agent_memory_traffic.py --mode chained --steps 42 --arm both
python3 research/agent_memory_traffic.py --arm both --turns 40    # document regime
```

A run of 42 turns with both arms costs roughly 0.5 M input tokens.

### Static corpus, documents per turn — Haiku 5.5, 40 turns

| | memory L0 | full context |
|---|---|---|
| tokens per turn | 16,318 | 46,633 |
| growth per later turn | **+25** | **+2,170** |
| accuracy | 16/40 (40 %) | 15/40 (38 %) |
| memory share of loop | 0.354 % | — |

2.86× fewer tokens at equal accuracy. From the slopes, the control passes the
memory arm at turn 9 and costs three times as much by turn 27. **At 16 turns
there is no advantage at all** — 0.97× and worse accuracy — because the memory
arm spends ~4.8 model calls per turn and the control's context has not yet
grown.

### Chained facts, revised mid-session — 42 turns, judged by assertion

| | Haiku memory | Haiku full | Grok memory | Grok full |
|---|---|---|---|---|
| correct | **6/6** | 2/6 | 5/6 | **6/6** |
| asserted a superseded value | 0/6 | 0/6 | 0/6 | 0/6 |
| refused the question | 0/6 | 2/6 | 0/6 | 0/6 |
| false refusals per run | 0 % | 18.2 % | 0 % | 0 % |
| tokens per turn | 7,335 | 3,799 | 13,275 | 5,593 |
| non-unitarity after writes | 2.5e-07 | — | 1.4e-07 | — |

**In this regime full context is cheaper**, because a fact is one line and the
control's context grows by only ~150–220 tokens per turn. The memory arm wins on
correctness on Haiku and on refusal resistance on both, not on cost.

### Latency, split honestly

| | value |
|---|---|
| Hamming scan + exact re-rank, 4,000 × 384-dim | **271 µs** |
| the same through the Python memory layer | 580–605 µs |
| embedding the query, MiniLM on CPU | **60,000 µs** |
| memory share of the LLM loop | 0.16–0.36 % |

The scan is 0.4 % of a search-by-text. Retrieval latency is not the product
argument and is not claimed as one.

### Working set

| | value |
|---|---|
| glyph index, 4,000 × 128-bit | 0.064 MB |
| float corpus held for exact re-ranking | 6.14 MB |
| reconsolidation operator, 384-dim complex | 2.36 MB |
| **engine total** | **8.6 MB** |
| peak RSS of the full stand | **2,006 MB** (PyTorch + encoder) |

A deployment that embeds elsewhere carries 8.6 MB. One that embeds locally with
this encoder needs 2 GB. Both are measured; quoting only the first would
misrepresent the requirement.

## 3. Figures withdrawn, and why

Kept here because an auditor is entitled to the history, not only the current
numbers.

| claim | what it actually was |
|---|---|
| 67× memory advantage over hnswlib | compared our codes-only against hnswlib's total. Honest: 65.9 against 66.9 MB, 1.5 % |
| 31 ns IPC latency p99 | `Ran 5 tests in 0.031s` — a unittest runtime, wrong by 10⁶ |
| coherence 99.85 % for reconsolidation | a hardcoded constant in an unrelated photonic module; the engine computes no quality metric |
| static RAG accuracy 0 % | a tautology about static indexes, inferred without measurement |
| 2.3× faster than HNSW | 1.6× on re-measurement with mean-centering enabled |
| AVX-512 POPCNT | the build uses AVX2 |
| 83 % accuracy on overwritten facts | substring scoring; 67 % when the assertion is judged, and both of the two failures were real |
| non-unitarity ≤ 9.85e-08 | the best single run; the honest range is 9.9e-08 – 4.1e-07 |

## 4. What is not measured

- **Worst-case latency.** Every figure is typical behaviour on a virtualised
  shared host. No real-time or determinism guarantee is claimed and no such
  measurement has been made.
- **Air-gapped operation end to end.** The C core issues no syscalls on the
  query path and needs no network. The Python stand downloads a sentence encoder
  on first use; an offline deployment must ship that encoder itself, which has
  not been tested here.
- **More than two model families.** Context-induced refusal appears on one and
  not the other. Nothing about a third is known.
- **Hardware safety or actuator behaviour.** No CAN, EtherCAT, ROS or serial
  code exists in this repository. Earlier correspondence implying otherwise was
  wrong and is withdrawn.
