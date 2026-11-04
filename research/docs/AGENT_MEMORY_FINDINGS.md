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

## 5. Chained tasks: what writing is worth

The independent tasks above measure retrieval cost and nothing else — nothing in
turn 12 needs anything from turn 3, so `write_observation` has no work to do.
Haiku wrote 39 notes because the prompt told it to and Grok wrote none because
nothing required it, which is why the 21 % write fraction is a disposition and
not a workload.

`--mode chained` fixes that. Six chains introduce a fact, revoke or revise it
later, and ask about it twenty turns on: a calibration whose three figures are
each overwritten at a different turn, a revoked signing key, a cascaded
recomputation, a compliance rule that erases an identifier, a long-horizon cell
state, and two sources that contradict each other at different trust levels.
Accuracy counts only the current values; reporting a superseded one is counted
separately, because "phi_2 = 2.80 (was 2.50)" is a correct answer that mentions
an old figure.

### First run: the control collapsed, and it was our fault

Memory scored 67 % against the control's 0 %, which was a lie in our favour.
`stop_reason: refusal` — the first chain was written in synthesis terms
("exothermic shift, reduce the reagent, correct the pH") and the model declined
from turn 8 onward. The refusal then sat in the accumulating arm's history and
silenced every later turn, while the memory arm, which keeps no history, was
untouched. The chain tests overwriting of figures, not chemistry, so the same
three values are now loop calibration parameters. The stand counts refusals and
prints which turns produced them; a residual 2 of 42 still come from the
trust-conflict chain.

### Second run: the memory arm lost

| | memory L0 | full context |
|---|---|---|
| tokens per turn | 6,777 | **3,913** |
| slope | −6 | +189 |
| accuracy on overwritten state | 4/6 (67 %) | **5/6 (83 %)** |
| reported a superseded fact | **3/6 (50 %)** | 0/6 (0 %) |

The prediction that a full-context agent would lose the thread did not hold at
40 turns: it never once reported a stale value. The memory arm did, in half its
answers, and the mechanism is plain — **the index had no arrow of time.** A
cosine does not know that a fact was revoked, so the search returned the turn-1
note and the turn-8 note with comparable scores and the model merged them.

The token result is just as instructive. A fact is one line, so the control's
context grows by 189 tokens per turn rather than 2,170, and paying 2.8 model
calls per turn to avoid that is a loss. **What the memory buys is set by the
volume of what must be remembered, not by the length of the session.**

### The arrow of time, added

Two mechanisms, both in the search layer — the C kernel returns Hamming
distance, ranking policy belongs above it:

**Decay.** `score *= exp(-dt / tau)`, dt in turns, tau 12 by default, measured
from the most recent write rather than from the current turn. The first attempt
measured from the current turn and was wrong: only notes carry timestamps, so
only notes were penalised, and background corpus fragments at 0.38 displaced all
three notes out of the top 5.

**Entity tombstoning.** `write_observation` takes an `entity` key; writing the
same key again marks the previous note deprecated and multiplies its weight by
0.1, and the tool's reply says plainly what was superseded.

Measured on the calibration chain: the current note ranks first at 0.789, the
superseded note is out of the top 5 entirely.

### Reconsolidation was inert, and now is not

`non-unitarity 0.00e+00` after 18 writes was not perfect stability. The target
passed to the engine was the note itself, and a target equal to the operator's
own output cannot move it at any learning rate — a Hermitian rank-one
perturbation of a unitary matrix has that same matrix as its unitary polar
factor. The engine was running and learning nothing.

The target is now the state-error vector dE = x_new − x_old between successive
notes under the same entity key, normalised; the first write of a key has no
divergence and applies no update. Oja's rule carries it, Newton-Schulz projects
back.

| | measured | threshold |
|---|---|---|
| non-unitarity | 1.88e-06 | < 1e-05 |
| plasticity per update | 1.57e-02 | — |

Plasticity is per single update and the engine clips `||dW||_F <= 0.05`, so the
0.05 figure is reached over three to four updates rather than one. Over a whole
session drift still accumulated to 8.04e-04, because the engine's second
corrective step was gated at 1e-03 and never fired; the gate is now 1e-06 and
the figure is 9.85e-08. See the trade-off section below for what the arrow of
time did and did not buy.

### The arrow of time is a trade-off, not a fix

Three configurations of the same memory arm, same six chains, Haiku 5.5, 42
turns. The target was 0 % stale and over 85 % accuracy. No configuration
reaches both, and the two metrics move against each other.

| configuration | accuracy on overwritten state | reported a stale fact | tombstoned |
|---|---|---|---|
| no arrow of time | 4/6 (67 %) | 3/6 (50 %) | — |
| entity key + decay with floor | 4/6 (67 %) | 2/6 (33 %) | 2 |
| + auto-tombstone by cosine | 2/6 (33 %) | **0/6 (0 %)** | 10, of which 9 automatic |

**Why the entity key alone is weak.** The key is chosen by the model, and the
model does not reuse it: across 19 writes, tombstoning fired twice, because
related facts were filed as `loop.calibration` and `loop.temperature` — two
entities as far as the index is concerned.

**Why automatic tombstoning overshoots.** On pairs drawn from these chains the
two classes separate cleanly — versions of one entity score at least 0.469,
unrelated notes at most 0.330 — so a 0.45 threshold looked safe. In the run it
fired nine times and took live facts with it: stale output fell to zero because
the memory had been emptied of the material the question needed. Six pairs were
not enough to set that boundary, which was stated as a caveat before the run
and is now a measurement.

**What this says about the mechanism.** Similarity is not contradiction. Two
notes about the same quantity are similar whether one supersedes the other or
they describe different attributes of it — the turn-1 calibration holds the
temperature that stays valid until turn 18, so retiring the whole note throws
away a current fact along with two stale ones. A correct mechanism has to detect
a **conflicting assertion about the same attribute**, which is not a distance in
embedding space. Facts need fields, not just vectors.

### What the fixes did achieve

Reconsolidation now learns. The target is the state-error vector between
successive notes under one key, and the engine's drift guardrail fires at 1e-06
rather than the 1e-03 that never fired at all:

| | before | after |
|---|---|---|
| non-unitarity after 19 writes | 8.04e-04 | **9.85e-08** |
| offset from identity | 1.28e-02 | 2.76e-02 |

Plasticity is 9.1e-03 per update against the engine's own `||dW||_F <= 0.05`
clip, so the 0.05 figure in the specification is reached over several updates,
not one.

### The control arm is too noisy to rank against

Identical reruns of the full-context arm gave 83 %, 33 % and 50 % accuracy with
1, 6 and 2 refusals respectively. The refusals land on the 15 THz pulse chain
and the trust-conflict chain, and a refused turn silently removes a fact the
later probe depends on. Until the chains are phrased clear of that, differences
between the arms on this task set are inside the noise, and only the memory
arm's internal comparison above is safe to read.

### Atomic slots: entity, attribute, value

Retiring a note wholesale was too coarse — the turn-1 calibration carries a
temperature valid until turn 18, so tombstoning the note discarded a live fact
with two stale ones, and accuracy fell to 33 % while stale output fell to zero.
The unit of storage is now a triplet, and supersession acts inside one slot:
`(loop, temperature, 50 C)` retires `(loop, temperature, 42 C)` and leaves
`(loop, pH, 6.8)` alone.

Two further mechanisms were needed, both because the model does not reuse the
keys it invents.

**Attribute names.** 19 writes produced 2 tombstones: related facts arrived as
`loop.calibration` and `loop.temperature`. Attribute names are now matched
semantically inside one entity, and the record is stored under the existing
name — without that canonicalisation the slot merged correctly while the text
stayed `loop.temp` and could not be found by the word "temperature". Measured on
these chains: variants of one attribute score 0.485 and above, different
attributes of one entity 0.382 and below; threshold 0.50.

**Entity names.** The next run produced 26 slots from 29 writes and 3
tombstones, because one loop arrived as `loop`, `cooling_loop` and
`synthesis_loop`. Entity names are matched the same way: same entity 0.504 and
above, different 0.293 and below; threshold 0.45. After that, 12 to 14 slots are
retired per run, 8 to 9 of them through semantic matching.

**Working memory is searched separately.** A record reading `loop.temp = 50 C`
is five words, and its cosine loses to a paragraph of corpus prose: measured, no
record reached the top 5 at all even when one was the only correct answer. The
agent's own records are now scanned in full rather than filtered by the Hamming
stage, and three of five result slots are held for them. Superseded records are
not lifted by the quota — their weight is already extinguished.

### Result, Haiku 5.5, 42 turns

| | memory L0 | full context |
|---|---|---|
| accuracy on overwritten state | **5/6 (83 %)** | 2/6 (33 %) |
| reported a stale fact | 3/6 | 2/6 |
| tokens per turn | 8,411 | 4,041 |
| non-unitarity after 33 writes | **4.09e-07** | — |
| memory share of loop | 0.155 % | — |

83 % held across two runs, and the memory arm is now ahead of the control on
accuracy by a wide margin. The token result does not change: a fact is one line,
the control's context grows by 181 tokens per turn, and 2.9 model calls per turn
cost more than that.

### The leak metric measures the wrong thing, in both directions

Reading the actual answers settles it. Of three flagged leaks, two are correct
answers that cite history:

> The current cooling loop calibration: **Temperature: 50 °C.** This is a
> validator correction that supersedes the earlier baseline of 42 °C.

That is the behaviour we want, scored as a failure. And the reverse happens too:
the Waddington chain was scored **correct** while being wrong —

> **H3K9ac: 0.35.** [...] Don't confuse the H3K9ac figure with the separate
> value of 0.45 stored for the "auto-degron term" entity.

0.45 appears in the text, inside a denial, so substring matching counted it as
found. The real accuracy is 4 of 6, not 5. Substring scoring cannot tell
"asserted instead of" from "mentioned as superseded", and it will keep flattering
or punishing at random. **The next version of this metric has to read the
assertion, not the string.**

That failure also has a cause worth naming: `auto-degron term` reads as its own
entity, so the turn-24 update landed in a different slot from the turn-14 state.
The chain's own wording created the split — the phrasing came from clearing the
refusals, and it traded one artefact for another.

### Clearing the refusals did not work

Rewording chains 5 and 6 cleared refusals when each chain was tested alone, and
the control arm still produced 5 to 6 refusals per run — landing on `Validator
correction: temperature raised to 50 C`, on corpus fillers, and once on a probe.
So the trigger is not the wording of those two chains: with a long accumulated
history, the model begins refusing on innocuous turns. The memory arm, which
carries no history, sees none. Until this is understood, the control arm's
accuracy cannot be compared with anything.


### Scoring the assertion, not the string

Substring scoring was wrong in both directions, and two answers show why: one
that reads "Temperature: 50 °C, superseding the earlier baseline of 42 °C" was
counted as a leak though it is exactly the wanted behaviour, and one that reads
"H3K9ac: 0.35. Do not confuse it with the 0.45 stored for the auto-degron term"
was counted as correct though it asserts the superseded figure — 0.45 appeared
in the text, inside a denial.

A judge model now reads each answer against the current and superseded values
and returns one of four verdicts: `correct`, `stale_asserted`, `wrong`,
`refusal`, plus a `cites_history` flag, since naming a superseded value **as**
superseded is correct and must not be punished. The judge is always Anthropic
and always the same model regardless of which model is under test, so a Grok run
is not graded by Grok. It reproduced all five reference verdicts identically on
two consecutive passes. Caveat worth stating: the judge is from the same family
as one of the models under test and may be the gentler reader of its own
output.

### Result, Haiku 5.5, 42 turns, judged

| | memory L0 | full context |
|---|---|---|
| correct | **4/6 (67 %)** | 3/6 (50 %) |
| asserted a superseded value | **1/6 (17 %)** | 2/6 (33 %) |
| otherwise wrong | 1/6 | 0/6 |
| refused the question | **0/6** | 1/6 |
| cited history while correct | 2 | 2 |
| tokens per turn | 7,558 | 3,894 |

The honest accuracy is 67 %, not the 83 % the substring metric reported in
earlier runs. The memory arm still leads on both correctness and on asserting
stale values, and the gap is smaller and believable.

Its two failures are real. One is the Zero-PII chain, where it produced neither
the anonymous identifier nor a refusal. The other is the Waddington chain, where
`auto-degron term` reads as its own entity, so the turn-24 update landed in a
different slot from the turn-14 state — a defect in the chain's wording, which
came from clearing refusals and traded one artefact for another.

### Refusal resistance (superseded by the section below)

Rewording the sensitive chains did not stop the control arm refusing. Across
five runs of 44 turns each:

| | refusals | rate |
|---|---|---|
| memory L0 | **0 of 220 turns** | 0.0 % |
| full context | 26 of 220 turns | 11.8 % |

Per run the control produced 6, 6, 5, 6 and 3. They land on innocuous material —
`Validator correction: temperature raised to 50 C`, corpus fillers, and twice on
a control question itself — so the trigger is not the wording of any one chain.
With a long accumulated history the model begins declining turns it answers
without hesitation in isolation, and a refused turn silently removes a fact that
a later question depends on.

The memory arm never refuses because it never carries that history: its prompt
is the current turn and its tool results. **This is a second, independent reason
to keep the agent's knowledge outside the context window, and unlike the token
saving it does not depend on how much there is to remember.** It is also the
one finding here that favours the architecture without qualification — though
it is a property of this model's behaviour on long contexts, not a theorem, and
it should be re-measured on other models before being quoted as general.

## Context-induced refusal, and what removes it

The effect was found while trying to explain why the full-context arm kept
scoring badly, and it turned out not to be a scoring problem at all.

**The observation.** Claude Haiku 5.5 refuses turns it answers without
hesitation in isolation, once a long history has accumulated in its context.
`stop_reason` comes back `refusal` with no content blocks. Over six runs of 44
turns:

| arm | refusals | rate |
|---|---|---|
| memory L0 | **0 of 264 turns** | 0.0 % |
| full context | 34 of 264 turns | 12.9 % |

Per run the control produced 6, 6, 5, 6, 3 and 8.

**It is not the material.** The refusals land on `Validator correction:
temperature raised to 50 C`, on corpus fillers, on a transaction notice, and
repeatedly on control questions. Every chain was tested on its own, turn by
turn, and every one passed clean — including the two whose original phrasing did
trigger refusals and was rewritten for that reason. Rewriting them did not
reduce the rate; the highest count, 8, came after the rewrite. What correlates is
the accumulated history, not the sentence being asked.

**The consequence is not just a missing answer.** A refused turn silently drops
the fact it was carrying, so a question twenty turns later has nothing to answer
from. In the arm that refuses, accuracy and refusal are not independent
failures: one causes the other. In the run with 8 refusals the control answered
2 of 6, with two of those failures being refusals on the question itself.

**L0 memory removes it completely, and not by being clever.** The memory arm's
prompt is the current turn plus its tool results, so no history accumulates, and
it refused 0 of 264 turns across both models.

**It is a property of one model, not of long contexts.** The same control arm on
Grok 4.6 produced 0 refusals. Two models are not a survey, and the claim is
stated at that width: one exhibits the effect, one does not.

## Both models, both arms, 42 turns, judged by the same arbiter

| | Haiku memory | Haiku full | Grok memory | Grok full |
|---|---|---|---|---|
| correct | **6/6** | 2/6 | 5/6 | **6/6** |
| asserted superseded | 0/6 | 0/6 | 0/6 | 0/6 |
| refused the question | 0/6 | 2/6 | 0/6 | 0/6 |
| false refusals per run | 0 % | 18.2 % | 0 % | 0 % |
| tokens per turn | 7,335 | 3,799 | 13,275 | 5,593 |
| non-unitarity | 2.49e-07 | — | 1.37e-07 | — |

With the Waddington chain's entity naming fixed, **the memory arm answers every
chain correctly on Haiku and asserts a superseded value nowhere, on either
model.** The arrow of time, the atomic slots and the honest error vector do the
job they were built for.

Two readings of this table have to be stated together, because quoting either
alone would mislead:

**Where memory wins, it wins decisively.** On Haiku it is 6/6 against 2/6, and
the control's failure is driven by refusals it cannot avoid while carrying
history. Nothing about the memory arm's prompt provokes them.

**Where the model does not refuse, carrying the history simply worked.** Grok's
full-context arm answered all six chains correctly at 42 turns, with no
lost-in-the-middle effect, for 5,593 tokens per turn against the memory arm's
13,275. The prediction that a full-context agent would lose the thread by turn
15 to 20 did not hold on either model: on Haiku it failed through refusal, not
confusion, and on Grok it did not fail at all.

So on chained facts the case for L0 memory rests on refusal resistance and on
the shape of the token curve, not on accuracy per se — and the token curve
favours full context here, because a fact is one line and the control's context
grows by roughly 150 to 220 tokens per turn. The 2.86x saving measured earlier
belongs to the regime where each turn carries documents.

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
