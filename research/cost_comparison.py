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

Ablation arms, all cached, to find which part produces any gain:

  rag_cache      search tool only: no writes, slots or operator
  notes_cache    writes are stored and searched; no superseding, no decay
  noengine_cache slots and decay, without the reconsolidation operator

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
import re
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
# USD per million tokens at peak; off-peak is half. Recorded 2026-11-05 from
# api-docs.deepseek.com/quick_start/pricing. Peak: 01-04 and 06-10 UTC, Mon-Fri.
DEEPSEEK = {"deepseek-flash": dict(hit=0.006, miss=0.30, output=1.20),
            "deepseek-v4-pro": dict(hit=0.044, miss=1.32, output=3.96)}
PRICES.update({m: dict(input=0, output=0, long_input=0, long_output=0,
                       long_above=10**12) for m in DEEPSEEK})

SECRET_PATTERNS = [
    r"\b\d{8,10}:[A-Za-z0-9_-]{30,}", r"sk-ant-[A-Za-z0-9_-]{10,}",
    r"\bsk-[A-Za-z0-9_-]{20,}", r"\bxai-[A-Za-z0-9]{20,}",
    r"\bgh[pos]_[A-Za-z0-9]{20,}", r"AIza[0-9A-Za-z_-]{30,}", r"PRIVATE KEY-----",
    r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{16,}",
]
SECRET_FILES = r"(?i)(api-keys|agent-keys|secret|credential|\.env)"


def sanitize(chunks):
    """Nothing with a credential leaves this machine. Chunks are dropped rather
    than redacted, because a partly redacted key file is still a key file."""
    kept, dropped = [], 0
    for c in chunks:
        text = c["text"]
        if re.search(SECRET_FILES, c["src"]) or any(re.search(p, text) for p in SECRET_PATTERNS):
            dropped += 1
            continue
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}", "<email>", text)
        text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<ip>", text)
        kept.append(dict(c, text=text))
    return kept, dropped


class Capture:
    """Keeps the raw response: the adapter drops DeepSeek's cache counters."""

    def __init__(self, client):
        self.real, self.chat, self.completions, self.last = client, self, self, None
        self.bad_tool_args = 0

    def create(self, **kw):
        self.last = self.real.chat.completions.create(**kw)
        # A malformed tool call is the model's failure and stays in the count;
        # it must not end the whole run.
        for tc in self.last.choices[0].message.tool_calls or []:
            try:
                json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                self.bad_tool_args += 1
                tc.function.arguments = "{}"
        return self.last


def is_peak(now=None):
    t = now or time.gmtime()
    return t.tm_wday < 5 and (1 <= t.tm_hour < 4 or 6 <= t.tm_hour < 10)


RAG_SYSTEM = (
    "You are an agent whose knowledge lives in an external document corpus, not "
    "in your context. The corpus is private project documentation you have never "
    "seen in training, so you cannot answer from your own knowledge - search it. "
    "Earlier turns are not kept in your context.")

TURN_LOG = []
_scored = a.scored


def logged_scored(answer, terms):
    ok = _scored(answer, terms)
    TURN_LOG.append(1 if ok else 0)
    return ok


a.scored = logged_scored


class PlainNotes(a.Memory):
    """Notes are stored and searched, but nothing is superseded and nothing
    decays: what the slots and the arrow of time add is measured against this."""

    def _find_slot(self, entity, attribute):
        return None, attribute, entity


def long_schedule(turns, filler_tasks, base=42, late_window=None):
    """The six chains of agent_memory_traffic.CHAINS stretched over `turns`
    (event turn t -> t * turns / base), and every probe asked a second time,
    spread over the last `late_window` turns (min(15, turns // 4): turns 185,
    188 ... 200 for a 200-turn run). Without the stretch all events sit in
    turns 1-40 and a 200-turn run is 160 turns of filler.
    Returns the schedule and, per item, whether it is a late repeat."""
    lw = late_window or min(15, turns // 4)
    n = len(a.CHAINS)
    slot = {}
    for i, ch in enumerate(a.CHAINS):
        for ev in ch["events"]:
            item = (ch["name"],) + tuple(ev[1:])
            if len(item) == 3:
                item = item + ([], [])
            t = max(1, round(ev[0] * turns / base))
            slot.setdefault(t, []).append((item, False))
            if ev[1] == "probe":
                at = turns - lw + round(i * lw / (n - 1))
                slot.setdefault(at, []).append((item, True))
    out, late, fi = [], [], 0
    for t in range(1, turns + 1):
        if t in slot:
            for item, is_late in slot[t]:
                out.append(item)
                late.append(is_late)
        elif fi < len(filler_tasks):
            out.append(("наполнитель", "filler", filler_tasks[fi]["prompt"], [], []))
            late.append(False)
            fi += 1
    return out, late


def log_writes(mem, path, arm, rnd, sched):
    """Every write of a chained run as one JSONL row: the names the model chose
    and the slot they landed in. Wraps this instance only - Memory is untouched,
    so the run behaves exactly as without logging."""
    find, write, last = mem._find_slot, mem.write, {}

    def _find(entity, attribute):
        prev, attr, ent = find(entity, attribute)
        last.update(canon_entity=ent, canon_attribute=attr,
                    slot_existed=prev is not None)
        return prev, attr, ent

    def _write(note, entity=None, attribute=None, value=None):
        last.clear()
        before = mem.superseded
        out = write(note, entity, attribute, value)
        t = mem.turn
        ev = sched[t - 1] if 0 < t <= len(sched) else (None, None)
        row = dict(arm=arm, round=rnd, turn=t, chain=ev[0], kind=ev[1],
                   entity=entity, attribute=attribute, value=value, note=note,
                   canon_entity=last.get("canon_entity"),
                   canon_attribute=last.get("canon_attribute"),
                   superseded=mem.superseded > before)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out

    mem._find_slot, mem.write = _find, _write


class BudgetExceeded(Exception):
    pass


class Meter:
    """Wraps client.messages so the existing arms run unchanged and every call is
    priced. input_tokens excludes cached tokens once caching is on, so token
    totals for an arm are read from here and not from the arm's own stats."""

    spent_total = 0.0

    def __init__(self, client, model, cache, budget, capture=None):
        self.client, self.cache, self.budget = client, cache, budget
        self.capture, self.model = capture, model
        self.price = PRICES[model]
        self.messages = self
        self.calls = self.fresh = self.written = self.read = self.output = 0
        self.refusals = self.long_calls = 0
        self.usd = 0.0
        self.force_at = None
        self.forced = 0

    def create(self, **kw):
        if Meter.spent_total >= self.budget:
            raise BudgetExceeded("budget $%.2f reached" % self.budget)
        if self.capture is not None:
            # deepseek-flash finds the target on step 1 and keeps searching until
            # the step limit, so the arm scored its looping and not the memory.
            # The last allowed step goes without tools and has to answer.
            if (self.force_at and kw.get("tools")
                    and len(kw["messages"]) >= 2 * self.force_at - 1):
                kw.pop("tools")
                self.forced += 1
            # DeepSeek caches on its own and reports it; there is no off switch.
            r = self.client.messages.create(**kw)
            raw = self.capture.last
            u = raw.usage
            read = getattr(u, "prompt_cache_hit_tokens", 0) or 0
            fresh = getattr(u, "prompt_cache_miss_tokens", u.prompt_tokens - read) or 0
            out = u.completion_tokens or 0
            p = DEEPSEEK[self.model]
            usd = (fresh * p["miss"] + read * p["hit"] + out * p["output"]) / 1e6
            usd *= 1.0 if is_peak() else 0.5
            self.calls += 1
            self.fresh += fresh
            self.read += read
            self.output += out
            self.refusals += 1 if raw.choices[0].finish_reason == "content_filter" else 0
            self.usd += usd
            Meter.spent_total += usd
            return r
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
                    long_price_calls=self.long_calls, usd=round(self.usd, 5),
                    forced_answers=self.forced,
                    bad_tool_args_total=self.capture.bad_tool_args if self.capture else 0)


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
    ap.add_argument("--writes-log", help="chained regime: one JSONL row per write")
    ap.add_argument("--stretch", action="store_true",
                    help="chained regime: spread the chains over --chain-turns "
                         "and repeat every probe at the end (long_schedule)")
    args = ap.parse_args()

    cap = None
    if args.model in DEEPSEEK:
        from openai import OpenAI
        cap = Capture(OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                             base_url="https://api.deepseek.com"))
        real = a.OpenAIAdapter(cap)
        judge_model = args.model
    else:
        import anthropic
        real = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        judge_model = "claude-haiku-5-5"

    chunks, dropped = sanitize(a.collect_chunks(a.CORPUS_ROOTS, args.corpus))
    left = sum(1 for c in chunks if any(re.search(p, c["text"]) for p in SECRET_PATTERNS))
    if left:
        sys.exit("sanitizer left %d chunks with credentials" % left)
    vecs, encoder = a.embed([c["text"] for c in chunks], args.encoder)
    print("corpus %d, dropped %d, dim %d" % (len(chunks), dropped, vecs.shape[1]), flush=True)
    fresh_glyphs = lambda: a.Glyphs(vecs.copy(), list(chunks))

    def record(**row):
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    try:
        for rnd in range(args.rounds):
            if "documents" in args.regime:
                tasks = a.build_tasks(chunks, args.turns, np.random.default_rng(11))
                base = fresh_glyphs()
                ceiling = 0
                for t in tasks:
                    qv = encoder.encode([t["prompt"]],
                                        normalize_embeddings=True)[0].astype(np.float32)
                    ceiling += 1 if t["chunk"] in [i for i, _ in base.search(qv)] else 0
                for arm in args.arms or ["full", "full_cache", "compact_cache",
                                         "memory", "memory_cache"]:
                    m = Meter(real, args.model, arm.endswith("_cache"), args.budget, cap)
                    m.force_at = 9 if cap else None     # arm_memory max_steps
                    g = fresh_glyphs()
                    t0 = time.time()
                    TURN_LOG.clear()
                    reads = None
                    if arm.startswith("rag"):
                        saved = a.TOOLS, a.SYSTEM_MEM
                        a.TOOLS, a.SYSTEM_MEM = [a.TOOLS[0]], RAG_SYSTEM
                        try:
                            mem = a.Memory(g, encoder, None)
                            s = a.arm_memory(m, args.model, tasks, mem)
                        finally:
                            a.TOOLS, a.SYSTEM_MEM = saved
                        reads = mem.reads
                    elif arm.startswith("notes"):
                        mem = PlainNotes(g, encoder, None, tau=1e9)
                        s = a.arm_memory(m, args.model, tasks, mem)
                        reads = mem.reads
                    elif arm.startswith("noengine"):
                        mem = a.Memory(g, encoder, None)
                        s = a.arm_memory(m, args.model, tasks, mem)
                        reads = mem.reads
                    elif arm.startswith("memory"):
                        mem = a.Memory(g, encoder, a.load_engine(vecs.shape[1]))
                        s = a.arm_memory(m, args.model, tasks, mem)
                        reads = mem.reads
                    elif arm.startswith("compact"):
                        s = arm_compact(m, args.model, tasks, chunks, g, encoder,
                                        args.compact_every)
                    else:
                        s = a.arm_full(m, args.model, tasks, chunks,
                                       np.random.default_rng(11), glyphs=g, encoder=encoder)
                    record(regime="documents", round=rnd, arm=arm, turns=len(tasks),
                           correct=s["correct"], seconds=round(time.time() - t0),
                           searches=reads, ceiling=ceiling,
                           per_turn="".join(map(str, TURN_LOG)), **m.row())
            if "chained" in args.regime:
                tasks = a.build_tasks(chunks, args.chain_turns, np.random.default_rng(11))
                if args.stretch:
                    sched, late = long_schedule(args.chain_turns, tasks)
                else:
                    sched = a.build_schedule(args.chain_turns, tasks)
                    late = [False] * len(sched)
                probe_late = [l for e, l in zip(sched, late) if e[1] == "probe"]
                for arm in args.arms or ["full", "full_cache", "memory", "memory_cache"]:
                    if arm.startswith(("compact", "rag")):
                        continue
                    m = Meter(real, args.model, arm.endswith("_cache"), args.budget, cap)
                    m.force_at = 7 if cap else None     # run_chained max_steps
                    judge = Meter(real, args.model, False, args.budget, cap)
                    t0 = time.time()
                    if arm.startswith("notes"):
                        mem = PlainNotes(fresh_glyphs(), encoder, None, tau=1e9)
                        kind = "memory"
                    elif arm.startswith("noengine"):
                        mem = a.Memory(fresh_glyphs(), encoder, None, tau=12.0)
                        kind = "memory"
                    elif arm.startswith("memory"):
                        mem = a.Memory(fresh_glyphs(), encoder,
                                       a.load_engine(vecs.shape[1]), tau=12.0)
                        kind = "memory"
                    else:
                        mem = a.Memory(fresh_glyphs(), encoder, None)
                        kind = "full"
                    if args.writes_log and kind == "memory":
                        log_writes(mem, args.writes_log, arm, rnd, sched)
                    s = a.run_chained(m, args.model, sched, mem, arm=kind,
                                      judge_client=judge if cap else real,
                                      judge_model=judge_model)
                    per_probe = [dict(chain=p["chain"], late=l, verdict=p["verdict"],
                                      cites_history=p.get("cites_history"),
                                      answer=(p.get("answer") or "")[:400])
                                 for p, l in zip(s.get("answers", []), probe_late)]
                    lp = [p for p in per_probe if p["late"]]
                    record(regime="chained", round=rnd, arm=arm, turns=len(sched),
                           judge_usd=round(judge.usd, 5),
                           correct=s.get("correct", 0), probes=s.get("probes", 0),
                           stale=s.get("stale_asserted", 0),
                           late_probes=len(lp),
                           late_correct=sum(p["verdict"] == "correct" for p in lp),
                           late_stale=sum(p["verdict"] == "stale_asserted" for p in lp),
                           per_probe=per_probe,
                           arm_refusals=s.get("refusals", 0),
                           seconds=round(time.time() - t0), **m.row())
    except BudgetExceeded as e:
        print("STOP:", e, flush=True)
    print("TOTAL_USD %.4f" % Meter.spent_total, flush=True)


if __name__ == "__main__":
    sys.exit(main())
