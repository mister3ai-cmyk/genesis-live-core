# Agent memory traffic: what was measured

**2026-11-03. Every figure comes from `research/agent_memory_traffic.py`, run on
a Contabo VPS, AMD EPYC, single core. Corpus: 4,000 paragraph chunks from this
project's own documentation — closed material no model has seen in training.**

The objection this answers was raised by a practitioner running agents on
pgvector and Qdrant: retrieval costs milliseconds against a model turn that
takes seconds, so shaving it to 0.3 ms is invisible. That is correct, and it is
not the claim. The claim is about what an agent's turn costs when it does not
have to carry its knowledge in the prompt.

---

## 1. The measurement

Two arms, identical tasks, identical retrieval ceiling.

**Memory arm.** Prompt holds the current turn and its tool results, nothing
else. Everything earlier must be retrieved through `search_memory` or recalled
from `write_observation`. Retrieval is the engine's own two-stage cascade: the
query is embedded, projected by `src/srp_lsh.c` into a 128-bit code, scanned by
POPCNT, then exactly re-ranked. Reconsolidation (Oja + Newton-Schulz) fires on
write only — a read is a scan and must stay microseconds.

**Control arm.** No memory. Every document it has been handed stays in the
prompt, and the history is never truncated.

Tasks are derived from the corpus so they cannot be answered from model weights:
the query is the opening of a target passage, and the answer is checked against
characteristic terms drawn from the **tail** of that same passage, which the
query never contains.

## 2. Result at 40 turns, Claude Haiku 5.5

| | memory L0 | full context |
|---|---|---|
| tokens per turn | **16,318** | **46,633** |
| turns 1–20 → 21–40 | 12,862 → 19,774 | 23,458 → 69,807 |
| model calls per turn | 4.5 | 1.0 |
| accuracy | 16/40 (40 %) | 15/40 (38 %) |
| memory share of loop | 0.354 % | — |

**2.86× fewer tokens per turn at equal accuracy.** The mechanism is the growth
rate, not the absolute figure: the control arm adds ~2,170 tokens to every
subsequent turn, the memory arm ~25. From those slopes the control passes the
memory arm at turn 9, costs twice as much by turn 18 and three times by turn 27.

### At 16 turns there is no advantage at all

| | memory L0 | full context |
|---|---|---|
| tokens per turn | 18,641 | 18,166 (0.97×) |
| accuracy | 3/16 (19 %) | 6/16 (38 %) |

This is the honest short-horizon result and it is worth stating plainly. The
memory arm spends 4.8 model calls per turn against the control's one, and each
call re-sends the instructions and the tool cycle. Until the control's context
has grown, that overhead eats the whole saving. **The 40-turn and 16-turn runs
also differ in step limit (9 against 6), so the accuracy recovery cannot be
attributed to horizon alone.**

## 3. Latency, measured

| operation | cost |
|---|---|
| read: scan 4,000 codes + exact re-rank | 605 µs |
| write: reconsolidation, dim 384 | 18.3 ms |
| write: reconsolidation, dim 256 | 5.3 ms |

Memory is 0.354 % of the loop. The objection holds completely on this axis:
retrieval latency is irrelevant at agent scale. What is not irrelevant is the
token bill, and that is where the measurement lands.

Write fraction is 21 %, well past the 5 % crossover measured in
`benchmarks/mixed_workload.py`, which is where a flat scan beats a graph index
under mixed load.

## 4. The saving is a property of the pair, not of the memory

The same memory arm, the same 40 tasks, on Grok 4.6:

| | Haiku 5.5 | Grok 4.6 |
|---|---|---|
| tokens per turn | 16,318 | **45,141** |
| reads per turn | 3.7 | **14.8** |
| writes | 39 | **0** |
| model calls per turn | 4.5 | 7.3 |
| accuracy | 40 % | 20 % |
| memory share of loop | 0.354 % | 0.042 % |

Grok searches four times as often and never writes, so its prompt fills with
retrieval results inside a single turn and the saving disappears — 45,141 tokens
per turn is the full-context arm's figure. **A 2.86× saving is a property of the
model-and-memory pair, not of the memory**, and quoting it without naming the
model would be the same error as quoting a benchmark without naming the host.

The zero writes are not an adapter fault: driven directly, Grok calls
`write_observation` correctly on the first opportunity. They are a fault in the
task set — see below.

## 5. A third defect, in the task set, not yet fixed

The 40 tasks are independent of one another. Nothing in turn 12 needs anything
from turn 3, so `write_observation` has no work to do: Haiku wrote 39 notes
because it was told to, Grok wrote none because nothing required it. The 21 %
write fraction is therefore an artefact of one model's disposition and should
not be read against the 5 % crossover in `benchmarks/mixed_workload.py`.

The architecture's own case is a swarm accumulating closed context over a
session, which means **chained tasks** where later turns depend on earlier
findings. Until the task set has chains, this stand measures retrieval traffic
and context cost honestly, and measures the value of writing not at all.

## 6. Two defects found in this stand, and fixed

**The search ignored the query.** The first version hashed the query text to a
random corpus index, so the agent was reacting to a broken tool. Every figure
from that version was withdrawn.

**The arms had different ceilings.** The control was handed the target passage
outright — reachable in 16 of 16 — while the memory arm had to find it, and
retrieval reaches 7 of 16. That compares retrieval recall against guaranteed
delivery, not the cost of context. After levelling, the memory arm's accuracy
fell from 38 % to 19 %: the first table flattered it. The old behaviour survives
behind `--fed-target`, labelled as unfair.

Also: `all-MiniLM-L6-v2` is an English encoder and this corpus is Russian — the
target was found 2 times in 16. Chunking by character count cut words in half
and the resulting fragments matched nothing. Both were corrected before any
figure above was taken.

## 7. The retrieval ceiling, stated

14 of 40 targets are reachable by a single query. Both arms are bounded by it,
so the accuracy figures are not to be read against 100 %.

## 8. Reproducing

```bash
python3 research/agent_memory_traffic.py --arm both --turns 40
python3 research/agent_memory_traffic.py --arm both --turns 16   # no advantage
python3 research/agent_memory_traffic.py --arm memory --turns 40 \
        --provider openai --model grok-4.6 --key-file <path>
```
