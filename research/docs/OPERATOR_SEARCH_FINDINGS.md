# Does the reconsolidation operator change what the agent retrieves?

**2026-11-06, 249 lines. 8 arms × 10 seeds × 2 name modes, ~1,050 queries per
arm. Offline: no model API, embedder paraphrase-multilingual-MiniLM-L12-v2,
WSL Ubuntu, 12 cores. Every figure comes from one of the commands in §9.**

**Answer: no.** The operator does not reach the agent's search results, and
none of the five ways of putting it there beats the slot rules written in
Python — with exact names, where the slots leave nothing to win, and with
varying names, where they miss 27 % of rewrites and there is room. Stale-value
handling in agent memory comes from the slots, not from NGP 4.6. Two defects
found in the slots themselves were fixed (§5).

## 1. Who reads the operator

| file:line | reader | what it does with the result |
|---|---|---|
| `research/agent_memory_traffic.py`, `Memory.write` | memory | calls `generate_context(q, target=x_new−x_old)`, discards the return value, keeps `‖ΔW‖` in `self.plasticity` |
| `research/agent_memory_traffic.py`, `main` (chained report) | stand's report | prints non-unitarity, plasticity, `‖W−I‖` |
| `research/cost_comparison.py:327,358` | cost stand | constructs `Memory` with the engine, nothing else |
| `research/ngp_4_6_live_benchmark_stand.py:147–150` | benchmark stand | `‖W−I‖`, drift |
| `research/retention_experiment.py:90–91` | retention experiment | alignment on synthetic pairs (§7) |
| `research/reconsolidation_experiment.py:191` | recall experiment | — (README: found nothing) |
| `research/tests/test_reconsolidation_plasticity.py:74–92` | product tests | plasticity, drift |

`Memory.search` → `Glyphs.search` ranks by `vecs[cand] @ q` times
`Memory._weight` (decay and `DEPRECATED_W`). No line outside tests and metrics
turns the operator's output into system output. No C code, `ngp45_engine/` or
`.md` file consumes it either.

## 2. Null test

`research/tests/test_operator_null.py` (`concept`): one scenario through
`Memory.write`/`Memory.search` with `engine=load_engine(dim)` and with
`engine=None`; results must be byte-identical and the operator must have moved
(‖W−I‖ > 1e-3).

| layer | runs where | result |
|---|---|---|
| stub encoder: hash vectors, dim 64, exact and varying names, seeds 0–2 | CI (gcc + numpy only) | 6 passed |
| real encoder, seeds 0–2 | where sentence-transformers is installed | 3 passed, skipped otherwise |

The stand repeats the check on 10 seeds × 2 modes: `memory==noengine: True`
twenty times. Consequence for the DeepSeek runs (memory 12, 10 against noengine
6, 4, 9 on documents; 6, 4 against 3, 6, 4 on chains): model-side noise.

## 3. Scenario

`research/operator_search_experiment.py`, `build_scenario(seed, synonyms)`: 20
entities × 2–3 attributes, each rewritten 3–5 times (1,558 rewrites over 10
seeds), writes shuffled with per-slot order kept, a paraphrased question about
a current value after ~half the writes (1,051 queries, 692 with older versions
existing). Background: paragraph chunks of this repository's `.md` files (319
at the time of the run; the count moves as documents are added). Two name
modes:

- **exact** — the agent always writes `cooling loop` / `temperature`
- **synonyms** (`--synonyms`) — each write picks one of three spellings:
  `loop`, `cooling_loop`; `temp`, `temperature reading`; `fw`, `firmware
  version`. Questions use the canonical name. The alias list was written by the
  same person who wrote the slot fix in §5; read the synonym figures with that
  in mind.

Metrics: `stale@5` — a superseded value of the asked slot is in the top 5;
`fresh@5` — the current value is; search latency, `Memory.read_time / reads`.
Slot accounting: **slot miss** — a rewrite left its own previous value live;
**false supersession** — a rewrite retired a value of another slot.

## 4. Variants against the slots

Acceptance rule, fixed before the run: mean paired gain over noengine larger
than the noengine standard deviation across seeds, same sign on ≥ 8/10 seeds,
other metric not worse.

| arm | exact: stale@5 | exact: fresh@5 | synonyms: stale@5 | synonyms: fresh@5 |
|---|---|---|---|---|
| noengine (slots, no engine) | 0.000 | 0.980 ± 0.018 | 0.247 ± 0.052 | 0.918 ± 0.032 |
| nosupersede (`DEPRECATED_W = 1`) | 0.985 | 0.920 | 0.885 | 0.843 |
| memory (engine on, current code) | 0.000 | 0.980 | 0.247 | 0.918 |
| a) query rotation `Re⟨Wq, x⟩` | 0.000 | 0.980 | 0.247 | 0.920 |
| b) records rotated at write | 0.000 | 0.980 | 0.246 | 0.918 |
| c) detector `‖Wx−x‖` instead of slots | 0.485 | 0.531 | 0.447 | 0.510 |
| c) detector together with slots | 0.000 | 0.554 | **0.174** | 0.533 |
| g) query rotation, slots off | 0.980 | 0.926 | 0.882 | 0.849 |

Search latency (exact mode, three runs sharing the machine): noengine 769 µs,
memory 992, a) 2,309, b) 812, c) 4,587–4,658, g) 1,630. Relative order is
stable; absolute values move with load.

**a) Query rotation.** *Hypothesis:* Oja on input x_new with target
(x_new−x_old) moves Wx_new towards that difference, so for q near x_new the
margin ⟨Wq, x_new⟩ − ⟨Wq, x_old⟩ grows. *Measured:* the direction is right —
the margin grows in 93 % of queries (84 % with synonyms) — by **+3.1e-03**
(+2.1e-03), against gaps between candidates of order 1e-01. Best case fresh
+0.001, better on 1 seed of 10. **Not counted.** Search 3× slower.

**b) Records rotated at write.** *Hypothesis:* weak, the query stays unrotated.
*Measured:* identical to noengine except one query on one seed. **Not
counted.**

**c) Staleness detector.** *Hypothesis:* blind, since Oja's input is x_new and
the current record moves at least as much as the stale ones. *Measured:* AUC
P(dev_stale > dev_current) = **0.508** exact, **0.503** synonyms — chance.
With synonyms, detector + slots lowers stale by 0.073 on 9/10 seeds, which
alone would pass; it does so by suppressing half of all records
indiscriminately, and fresh falls by 0.386. **Not counted — harmful.**

**g) Query rotation without slots** (added after the first run): against
nosupersede, stale −0.005 / −0.003, fresh +0.006 / +0.006, better on 2–4
seeds of 10. **No effect.**

Why the ceiling is low by construction: η ≈ 0.02 with ‖ΔW‖_F clipped to 0.05
per update, one unitary operator shared by ~50 slots, and a target that is a
direction in embedding space, not a fact.

## 5. Two defects in the slots, fixed

**False supersession.** With the original `_find_slot` (ENTITY_MATCH 0.45, no
lexical step), on exact names: 304 rewrites retired a value of another slot,
and in **123 of 1,051 queries (11.7 %)** the current value was lost that way —
nearly all of the gap to fresh@5 = 1. Causes, measured on the scenario's
names: different entities reach 0.559 (fermenter / bakery oven), 0.552 (pump
station / water tank), so the first write of a new entity was absorbed into
an old one and kept its name; different attributes reach 0.651 (temperature /
humidity), while true variants fall to 0.286 (flow rate / flowrate) and 0.155
(firmware version / fw). **No embedding threshold separates short names.**

Fix (`agent_memory_traffic.py`, `_find_slot` and `_same_name`): compare
spellings before meanings — case, `_`/`-`, articles, joined words, word
subset, ≥4-letter prefix — and raise ENTITY_MATCH to 0.60, where no two
different entities in this set merge. ATTR_MATCH stays 0.50. Grid over six
policies, noengine, 10 seeds (scratch script, see the note in §9):

| policy | exact: stale / fresh | synonyms: stale / fresh |
|---|---|---|
| original | 0.000 / 0.871 | 0.330 / 0.730 |
| sorted only | 0.003 / 0.863 | 0.316 / 0.739 |
| **lexical, E 0.60, A 0.50 (applied)** | **0.000 / 0.980** | **0.247 / 0.918** |
| lexical, E 0.60, A 0.70 | 0.000 / 1.000 | 0.375 / 0.939 |

The applied policy is better on all four numbers in both modes. A 0.70 buys
fresh with stale and was rejected. After the fix: false supersessions 67
(exact) and 144 (synonyms), current value lost in 2.1 % and 3.7 % of queries;
slot misses 0 % and **27.0 %**. Synonyms without shared words (fw / firmware)
are still missed — that is a limit of name matching, not a tuning problem.
**Known cost:** `cooling_loop` / `synthesis_loop` (0.504, one contour in the
earlier chains) no longer merges by meaning. **The paid runs in README and
BENCHMARKS were made with the original policy.**

**Hash-seed dependence.** `_find_slot` iterated a set of strings, so a
borderline write landed in different slots from process to process (seed 6:
154/155/156 supersessions; embeddings checked bit-identical across processes).
Iteration is now sorted. Runs under `PYTHONHASHSEED=1` and `=2` give identical
stale/fresh on every seed and arm.

## 6. Conclusion for the paid long-history run

On this scenario the agent's memory quality is set entirely by the slots. A
paid run comparing memory against RAG would measure the slots and the cascade;
it would not show any contribution of NGP 4.6.

## 7. What the operator does do: retention, as `retention_experiment.py` measures it

Not agent memory: random complex pairs q → x in C^d, 20 old then 60 new, old
never revisited, alignment |⟨x, Wq/‖Wq‖⟩|. One seed (`default_rng(5)`).

| `--dim 256 --rounds 800` | retention after old | after new | forgetting | acquisition | non-unitarity |
|---|---|---|---|---|---|
| frozen | 0.047 | 0.047 | — | 0.057 | 0 |
| naive Oja | 0.773 | 0.301 | **0.472** | 0.682 | 3.5e+01 |
| reconsolidation | 0.690 | 0.490 | **0.200** | 0.643 | 6.7e-07 |

16 m 55 s on 12 cores. `BENCHMARKS.md` stated 0.122 against 0.415 for this
command; that did not reproduce and was corrected. Default run (`dim 64,
rounds 12`): forgetting 0.004 against 0.018, all alignments ≤ 0.23.

## 8. Real names: DeepSeek chained run on df984fa, and the twin guard

The synonym figures in §5 come from aliases written by the same hand as the
fix. To check against names a model chooses itself, the chained regime was run
on df984fa: deepseek-flash, 42 turns, 3 rounds, `cost_comparison.py
--writes-log`, 2026-11-06 04:25–05:20 UTC, off-peak, **$0.553** in all.

| arm | correct of 6 | stale asserted | earlier run | forced answers |
|---|---|---|---|---|
| full_cache | 6, 6, 6 | 0, 0, 0 | 6, 6, 5 | 0 |
| noengine_cache (slots) | 4, 5, 5 | 1, 1, 1 | 3, 6, 4 | 20, 18, 21 |
| notes_cache (no supersession) | 3, 4, 6 | 2, 0, 0 | 5, 4, 4 | 20, 18, 15 |

Slots 14/18 against plain notes 13/18 — within round-to-round spread. Full
context 18/18: on short facts it stays ahead.

The 205 writes (74 distinct (entity, attribute) pairs, filler turns excluded)
are kept as the offline benchmark for slot matching:
`research/tests/data/ds3_written_names.jsonl`, each row labelled with its
quantity by `slot_names_replay.quantity()`. The labels were written after
seeing the names; the judgement calls are listed in that function.

Replayed through `_find_slot` (`slot_names_replay.py`), 117 true quantities:

| version | slot misses | false merges | live slots |
|---|---|---|---|
| 151d2ae (original) | 16 | 34 | 113 |
| df984fa (spelling first, E 0.60) | 16 | 32 | 114 |
| **twin guard (current)** | 19 | **4** | 130 |

df984fa changed almost nothing on real names: the model writes clean
spellings. 28 of the 32 false merges were **twins** — `agent_beta.key` retiring
`agent_alpha.key` (14), `tile_2.phase_shift` retiring `tile_1.phase_shift` (7),
trust of `source_y` retiring trust of `source_x` (3): names that differ in one
index token, which the embedding reads as one entity. The guard
(`_siblings`) refuses a semantic entity merge when the names differ in exactly
one token and both tokens are indices — digits, single letters, letter+digit
(k1/k2) or Greek letter names. Lexical matches are unaffected.

The three extra misses were accidental hits before: `phi_1.value` used to
retire `tile_1.phase_shift` only because `tile_2` had already been merged into
`tile_1`; and `source_y` / `source_x` speed, which the labels count as one
quantity while the guard, by design, keeps the sources apart. The remaining
misses need knowledge, not spelling: `phi_1.value` is `tile_1.phase_shift`,
`ed25519_k1.key_status` is the state of `agent_alpha.key`.

Synthetic scenario with the guard: exact 0.000 / 0.980, synonyms 0.247 / 0.918
— identical to df984fa. Matching entities by spelling only (no semantic step)
gives the same 19 / 4 on real names but raises synonym stale@5 to 0.354, so
the guard keeps the semantic step and removes only the twins.

## 9. Commands

```bash
# Linux/WSL with gcc; sentence-transformers for the real-encoder parts
python3 -m pytest research/tests/test_operator_null.py -q             # 9 passed (6 in CI)
python3 research/operator_search_experiment.py --seeds 10             # §4 exact, §5 slot lines
python3 research/operator_search_experiment.py --seeds 10 --synonyms  # §4 synonyms
python3 -m pytest -m product research/tests/ -q                       # 4 passed, unchanged
python3 research/retention_experiment.py --dim 256 --rounds 800       # §7
python3 research/slot_names_replay.py research/tests/data/ds3_written_names.jsonl \
        --old df984fa --new worktree                                  # §8 replay
# §8 paid run (DeepSeek key needed): cost_comparison.py --model deepseek-flash
#   --regime chained --arms full_cache noengine_cache notes_cache --chain-turns 42
#   --rounds 3 --corpus 4000 --budget 0.8 --writes-log writes.jsonl
```

Running several stands at once: set `OMP_NUM_THREADS` so the sum stays within
the cores. Three runs at 12 threads each on 12 cores did not finish one seed
in 90 minutes; at 4 threads each all three finished in minutes.

The §5 grid was a scratch script over subclasses of `Memory`; its applied row
is reproduced by the two stand commands above (noengine rows).
