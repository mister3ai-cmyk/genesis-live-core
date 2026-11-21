# Business Source License 1.1 - see research/LICENSE
"""llm_f12.py - faces F1 (summary) and F2 (entities) with Claude Haiku 5.5.
2026-11-21 · CLI: pilot | submit | status | collect | forecast.

One call per unique turn text (dedup by normalised sha256), JSON {summary, entities}
via structured outputs, thinking disabled, effort low. Cache: data/llm_cache.jsonl,
one line per text hash, key sha256(model + prompt_version + role + text).
Sampling: Claude Haiku 5.5 rejects a non-default temperature (400), so the
reproducibility of F1/F2 comes from the disk cache, not from temperature 0.

  pilot    500 dev texts, random, stratified by role x length quartile, direct calls
  submit   Message Batches (-50 %) for all uncached dev texts (chunks of 20 000)
  status   batch processing status
  collect  stream finished batch results into the cache
  forecast cost of the remaining dev texts and of test texts not in dev
"""
import concurrent.futures as cf
import hashlib
import json
import os
import pickle
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(DATA, "llm_cache.jsonl")
BATCHES = os.path.join(DATA, "batches.json")
COST_LOG = os.path.join(HERE, "results", "llm_cost.json")
MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "f12-v1"
PRICE = {"in": 0.10e-6, "out": 0.50e-6}          # $/token, standard; batch = half
BUDGET = 7.17
SYSTEM = (
    "You extract retrieval keys from one message of a chat between a user and an AI assistant.\n"
    "summary: one sentence of at most 30 words with the key fact, request or answer of the message, "
    "naming the speaker (e.g. \"User says they graduated in Business Administration.\").\n"
    "entities: named entities, numbers, dates, quantities and key predicates (short verb phrases) from "
    "the message, separated by \"; \"; empty string if there are none.\n"
    "Use only information present in the message.")
SCHEMA = {"type": "object",
          "properties": {"summary": {"type": "string"}, "entities": {"type": "string"}},
          "required": ["summary", "entities"], "additionalProperties": False}


def client_():
    """ANTHROPIC_API_KEY from the environment, else from a file outside the repository
    (%USERPROFILE%/.anthropic/api_key or HG_KEY_FILE). The key is never printed or logged."""
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        path = os.environ.get("HG_KEY_FILE") or os.path.join(os.path.expanduser("~"), ".anthropic", "api_key")
        if os.path.exists(path):
            os.environ["ANTHROPIC_API_KEY"] = open(path, encoding="utf-8").read().strip()
    return anthropic.Anthropic()


def params(role, text):
    return dict(model=MODEL, max_tokens=400, system=SYSTEM, thinking={"type": "disabled"},
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": "Speaker: %s\nMessage:\n%s" % (role, text)}])


def cache_key(role, text):
    return hashlib.sha256((MODEL + PROMPT_VERSION + role + "\n" + text).encode("utf-8")).hexdigest()


def load_cache():
    out = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            r = json.loads(line)
            out[r["h"]] = r
    return out


def append_cache(rows):
    with open(CACHE, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_message(msg):
    """-> (ok, summary, entities, in_tok, out_tok, reason)."""
    u = msg.usage
    if msg.stop_reason == "refusal":
        return False, "", "", u.input_tokens, u.output_tokens, "refusal"
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        d = json.loads(text)
        s, e = str(d.get("summary", "")).strip(), str(d.get("entities", "")).strip()
        if not s:
            return False, "", "", u.input_tokens, u.output_tokens, "empty"
        return True, s, e, u.input_tokens, u.output_tokens, msg.stop_reason
    except (json.JSONDecodeError, AttributeError):
        return False, "", "", u.input_tokens, u.output_tokens, "bad_json"


def dev_texts(split="dev"):
    """Unique texts of a split: list of (h, role, text) in node order (first occurrence)."""
    p = pickle.load(open(os.path.join(DATA, "prep.pkl"), "rb"))
    seen, out = set(), []
    for q in sorted(p["questions"].values(), key=lambda q: q["qid"]):
        if q["split"] != split:
            continue
        for n in p["nodes"][q["node_range"][0]:q["node_range"][1]]:
            if n["h"] not in seen:
                seen.add(n["h"])
                out.append((n["h"], n["role"], p["texts"][n["h"]]))
    return out, p


def strata(items, edges=None):
    lens = np.array([len(t) for _, _, t in items])
    if edges is None:
        edges = np.quantile(lens, [0.25, 0.5, 0.75])
    q = np.searchsorted(edges, lens, side="right")
    return [(r, int(k)) for (_, r, _), k in zip(items, q)], edges


def pilot(n=500, seed=1):
    import anthropic
    items, _ = dev_texts()
    st, edges = strata(items)
    groups = {}
    for i, s in enumerate(st):
        groups.setdefault(s, []).append(i)
    rng = random.Random(seed)
    pick = []
    for s, ids in sorted(groups.items()):
        k = round(n * len(ids) / len(items))
        pick += rng.sample(ids, min(k, len(ids)))
    cache = load_cache()
    pick = [i for i in pick if items[i][0] not in cache]
    client = client_()

    def one(i):
        h, role, text = items[i]
        try:
            msg = client.messages.create(**params(role, text))
            ok, s, e, ti, to, why = parse_message(msg)
        except anthropic.APIStatusError as ex:
            ok, s, e, ti, to, why = False, "", "", 0, 0, "api_%d" % ex.status_code
        return dict(h=h, key=cache_key(role, text), role=role, ok=ok, summary=s, entities=e,
                    in_tok=ti, out_tok=to, reason=why, mode="direct", stratum=list(st[i]))

    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, pick))
    append_cache(rows)
    cost = sum(r["in_tok"] * PRICE["in"] + r["out_tok"] * PRICE["out"] for r in rows)
    print("pilot: %d calls, %.0f s, $%.4f, fails %d" % (len(rows), time.time() - t0, cost,
                                                        sum(not r["ok"] for r in rows)))
    json.dump({"edges": edges.tolist()}, open(os.path.join(DATA, "strata_edges.json"), "w"))
    log_cost("pilot", cost, rows)


def log_cost(stage, cost, rows):
    os.makedirs(os.path.dirname(COST_LOG), exist_ok=True)
    log = json.load(open(COST_LOG)) if os.path.exists(COST_LOG) else {"stages": []}
    log["stages"].append({"stage": stage, "calls": len(rows), "usd": round(cost, 4),
                          "in_tok": sum(r["in_tok"] for r in rows), "out_tok": sum(r["out_tok"] for r in rows),
                          "fails": sum(not r["ok"] for r in rows), "time": time.strftime("%Y-%m-%d %H:%M")})
    log["spent_usd"] = round(sum(s["usd"] for s in log["stages"]), 4)
    json.dump(log, open(COST_LOG, "w"), indent=1)


def forecast():
    """Per-stratum mean tokens from the cache (direct + batch) -> cost of what is left."""
    cache = load_cache()
    edges = np.array(json.load(open(os.path.join(DATA, "strata_edges.json")))["edges"])
    dev, _ = dev_texts("dev")
    test, _ = dev_texts("test")
    dev_h = {h for h, _, _ in dev}
    by = {}
    for r in cache.values():
        if r["in_tok"]:
            by.setdefault(tuple(r["stratum"]), []).append((r["in_tok"], r["out_tok"]))
    mean = {k: np.mean(v, axis=0) for k, v in by.items()}
    allm = np.mean([x for v in by.values() for x in v], axis=0)

    def cost(items, batch=True):
        st, _ = strata(items, edges)
        f = 0.5 if batch else 1.0
        tot = np.zeros(2)
        for s in st:
            tot += mean.get(s, allm)
        return f * (tot[0] * PRICE["in"] + tot[1] * PRICE["out"]), tot
    left_dev = [x for x in dev if x[0] not in cache]
    test_new = [x for x in test if x[0] not in dev_h]
    cd, td = cost(left_dev)
    ct, tt = cost(test_new)
    spent = json.load(open(COST_LOG))["spent_usd"] if os.path.exists(COST_LOG) else 0.0
    out = {"spent_usd": spent, "dev_left_texts": len(left_dev), "dev_left_usd_batch": round(cd, 3),
           "test_new_texts": len(test_new), "test_new_usd_batch": round(ct, 3),
           "total_projected_usd": round(spent + cd + ct, 3), "budget_usd": BUDGET,
           "mean_in_tok": float(allm[0]), "mean_out_tok": float(allm[1])}
    print(json.dumps(out, indent=1))
    return out


def submit(chunk=20000):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    items, _ = dev_texts()
    cache = load_cache()
    todo = [x for x in items if x[0] not in cache]
    edges = np.array(json.load(open(os.path.join(DATA, "strata_edges.json")))["edges"])
    st, _ = strata(todo, edges)
    client = client_()
    meta = json.load(open(BATCHES)) if os.path.exists(BATCHES) else {"batches": [], "stratum": {}}
    for a in range(0, len(todo), chunk):
        part = todo[a:a + chunk]
        reqs = [Request(custom_id=h[:64], params=MessageCreateParamsNonStreaming(**params(role, text)))
                for h, role, text in part]
        b = client.messages.batches.create(requests=reqs)
        meta["batches"].append({"id": b.id, "n": len(part), "collected": False})
        for (h, role, text), s in zip(part, st[a:a + chunk]):
            meta["stratum"][h[:64]] = [h, role, cache_key(role, text), list(s)]
        json.dump(meta, open(BATCHES, "w"))
        print("submitted", b.id, len(part), flush=True)


def status():
    import anthropic
    client = client_()
    meta = json.load(open(BATCHES))
    for b in meta["batches"]:
        r = client.messages.batches.retrieve(b["id"])
        print(b["id"], r.processing_status, r.request_counts, "collected" if b["collected"] else "")


def collect():
    import anthropic
    client = client_()
    meta = json.load(open(BATCHES))
    for b in meta["batches"]:
        if b["collected"]:
            continue
        r = client.messages.batches.retrieve(b["id"])
        if r.processing_status != "ended":
            print(b["id"], "not ended:", r.processing_status)
            continue
        rows = []
        for res in client.messages.batches.results(b["id"]):
            h, role, key, s = meta["stratum"][res.custom_id]
            if res.result.type == "succeeded":
                ok, su, en, ti, to, why = parse_message(res.result.message)
            else:
                ok, su, en, ti, to, why = False, "", "", 0, 0, res.result.type
            rows.append(dict(h=h, key=key, role=role, ok=ok, summary=su, entities=en, in_tok=ti,
                             out_tok=to, reason=why, mode="batch", stratum=s))
        append_cache(rows)
        cost = 0.5 * sum(x["in_tok"] * PRICE["in"] + x["out_tok"] * PRICE["out"] for x in rows)
        log_cost("batch " + b["id"], cost, rows)
        b["collected"] = True
        json.dump(meta, open(BATCHES, "w"))
        print(b["id"], "collected", len(rows), "fails", sum(not x["ok"] for x in rows), "$%.3f" % cost)


if __name__ == "__main__":
    {"pilot": pilot, "submit": submit, "status": status, "collect": collect, "forecast": forecast}[sys.argv[1]]()
