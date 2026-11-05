#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/cost_comparison.py

Does L0 memory still save money when the control is not naive? The earlier stand
compared memory with an uncached full context and counted tokens. A real team
would cache the prompt or compact the history, and pays in dollars.

Arms, on identical tasks, model and retrieval:

  full           whole history in the prompt, no caching (the earlier control)
  full_cache     the same with prompt caching
  compact_cache  history summarised every --compact-every turns, with caching
  memory         L0 memory tools, no caching (the earlier memory arm)
  memory_cache   the same with prompt caching

Regimes: "documents" (documents per turn, keyword scoring) and "chained" (facts
revised mid-session, judged by a separate model). compact_cache runs in the
documents regime only, because the chained loop has no compaction step.

Prices are constants below and must be checked against the current price list.
Judge calls are test infrastructure and are not billed to any arm.

Usage:
  python3 cost_comparison.py --regime documents chained --turns 40 --rounds 3
Licensed under the Business Source License 1.1 (SPDX: BUSL-1.1); see research/LICENSE.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import agent_memory_traffic as a

# USD per million tokens, Claude API list price recorded 2026-10-06.
PRICES = {
    "claude-haiku-5-5": dict(input=0.10, output=0.50, long_input=0.50,
                             long_output=2.50, long_above=100_000),
    "claude-sonnet-5-5": dict(input=2.00, output=10.00, long_input=2.00,
                              long_output=10.00, long_above=10**12),
}
CACHE_WRITE = 1.25
CACHE_READ = 0.10


class BudgetExceeded(Exception):
    pass


class Meter:
    """Wraps client.messages so the existing arms run unchanged and every call is
    priced. input_tokens excludes cached tokens once caching is on, so token
    totals for an arm are read from here and not from the arm's own stats."""

    spent_total = 0.0

    def __init__(self, client, model, cache, budget):
        self.client, self.cache, self.budget = client, cache, budget
        self.price = PRICES[model]
        self.messages = self
        self.calls = self.fresh = self.written = self.read = self.output = 0
        self.refusals = self.long_calls = 0
        self.usd = 0.0

    def create(self, **kw):
        if Meter.spent_total >= self.budget:
            raise BudgetExceeded("budget $%.2f reached" % self.budget)
        if self.cache:
            kw["extra_body"] = {"cache_control": {"type": "ephemeral"}}
        r = self.client.messages.create(**kw)
        u = r.usage
        fresh = u.input_tokens or 0
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        read = getattr(u, "cache_read_input_tokens", 0) or 0
        out = u.output_tokens or 0
        p = self.price
        is_long = fresh + written + read > p["long_above"]
        rate_in = p["long_input"] if is_long else p["input"]
        rate_out = p["long_output"] if is_long else p["output"]
        usd = (fresh * rate_in + written * rate_in * CACHE_WRITE
               + read * rate_in * CACHE_READ + out * rate_out) / 1e6
        self.calls += 1
        self.fresh += fresh
        self.written += written
        self.read += read
        self.output += out
        self.long_calls += 1 if is_long else 0
        self.refusals += 1 if r.stop_reason == "refusal" else 0
        self.usd += usd
        Meter.spent_total += usd
        return r

    def row(self):
        return dict(calls=self.calls, fresh=self.fresh, cache_written=self.written,
                    cache_read=self.read, output=self.output, refusals=self.refusals,
                    long_price_calls=self.long_calls, usd=round(self.usd, 5))


def arm_compact(client, model, tasks, chunks, glyphs, encoder, every):
    """arm_full with the history replaced by a model-written summary every N turns."""
    correct, msgs = 0, []
    for n, t in enumerate(tasks):
        if n and n % every == 0:
            ask = msgs + [{"role": "user", "content":
                           "Summarise this session for your own later use. Keep "
                           "document ids, names, values and open items. At most "
                           "400 words."}]
            r = client.messages.create(model=model, max_tokens=1200,
                                       system=a.SYSTEM_FULL, messages=ask)
            text = " ".join(b.text for b in r.content if b.type == "text")
            msgs = [{"role": "user", "content": "Summary of the session so far:\n" + text},
                    {"role": "assistant", "content": "Understood."}]
        qv = encoder.encode([t["prompt"]],
                            normalize_embeddings=True)[0].astype(np.float32)
        docs = "\n\n".join("[%s | %s]\n%s" % (chunks[i]["id"], chunks[i]["src"],
                                              chunks[i]["text"])
                           for i, _ in glyphs.search(qv))
        msgs.append({"role": "user", "content": "Documents:\n%s\n\n%s" % (docs, t["prompt"])})
        r = client.messages.create(model=model, max_tokens=700,
                                   system=a.SYSTEM_FULL, messages=msgs)
        answer = " ".join(b.text for b in r.content if b.type == "text")
        msgs.append({"role": "assistant", "content": r.content})
        correct += 1 if a.scored(answer, t["terms"]) else 0
    return dict(correct=correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5", choices=sorted(PRICES))
    ap.add_argument("--regime", nargs="+", default=["documents", "chained"],
                    choices=["documents", "chained"])
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--chain-turns", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--corpus", type=int, default=4000)
    ap.add_argument("--compact-every", type=int, default=10)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--out", default="/tmp/cost_comparison.jsonl")
    ap.add_argument("--encoder", default="paraphrase-multilingual-MiniLM-L12-v2")
    args = ap.parse_args()

    import anthropic
    real = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    chunks = a.collect_chunks(a.CORPUS_ROOTS, args.corpus)
    vecs, encoder = a.embed([c["text"] for c in chunks], args.encoder)
    print("corpus %d, dim %d" % (len(chunks), vecs.shape[1]), flush=True)
    fresh_glyphs = lambda: a.Glyphs(vecs.copy(), list(chunks))

    def record(**row):
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    try:
        for rnd in range(args.rounds):
            if "documents" in args.regime:
                tasks = a.build_tasks(chunks, args.turns, np.random.default_rng(11))
                for arm in args.arms or ["full", "full_cache", "compact_cache",
                                         "memory", "memory_cache"]:
                    m = Meter(real, args.model, arm.endswith("_cache"), args.budget)
                    g = fresh_glyphs()
                    t0 = time.time()
                    if arm.startswith("memory"):
                        mem = a.Memory(g, encoder, a.load_engine(vecs.shape[1]))
                        s = a.arm_memory(m, args.model, tasks, mem)
                    elif arm.startswith("compact"):
                        s = arm_compact(m, args.model, tasks, chunks, g, encoder,
                                        args.compact_every)
                    else:
                        s = a.arm_full(m, args.model, tasks, chunks,
                                       np.random.default_rng(11), glyphs=g, encoder=encoder)
                    record(regime="documents", round=rnd, arm=arm, turns=len(tasks),
                           correct=s["correct"], seconds=round(time.time() - t0),
                           **m.row())
            if "chained" in args.regime:
                tasks = a.build_tasks(chunks, args.chain_turns, np.random.default_rng(11))
                sched = a.build_schedule(args.chain_turns, tasks)
                for arm in args.arms or ["full", "full_cache", "memory", "memory_cache"]:
                    if arm.startswith("compact"):
                        continue
                    m = Meter(real, args.model, arm.endswith("_cache"), args.budget)
                    t0 = time.time()
                    if arm.startswith("memory"):
                        mem = a.Memory(fresh_glyphs(), encoder,
                                       a.load_engine(vecs.shape[1]), tau=12.0)
                        kind = "memory"
                    else:
                        mem = a.Memory(fresh_glyphs(), encoder, None)
                        kind = "full"
                    s = a.run_chained(m, args.model, sched, mem, arm=kind,
                                      judge_client=real, judge_model="claude-haiku-5-5")
                    record(regime="chained", round=rnd, arm=arm, turns=len(sched),
                           correct=s.get("correct", 0), probes=s.get("probes", 0),
                           stale=s.get("stale_asserted", 0),
                           arm_refusals=s.get("refusals", 0),
                           seconds=round(time.time() - t0), **m.row())
    except BudgetExceeded as e:
        print("STOP:", e, flush=True)
    print("TOTAL_USD %.4f" % Meter.spent_total, flush=True)


if __name__ == "__main__":
    sys.exit(main())
