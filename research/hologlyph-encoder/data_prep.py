# Business Source License 1.1 - see research/LICENSE
"""data_prep.py - LongMemEval_S (cleaned) -> per-question turn nodes, split, dedup.
2026-11-21 · 1 public function (prepare), 1 CLI. Writes data/prep.pkl and prints
the dataset facts that go into ENCODER_EVAL_RESULTS.md.

Unit = one turn (reply). Relevant turns = has_answer == True; session level =
answer_session_ids. Split by question, stratified by question_type, seed 0:
dev 150 / test 350. _abs questions stay in their split but are excluded from
retrieval metrics.
"""
import collections
import hashlib
import json
import os
import pickle
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SRC = os.path.join(DATA, "longmemeval_s_cleaned.json")
SEED = 0
N_DEV = 150


def norm_hash(text):
    t = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def split_ids(questions):
    by_type = collections.defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q["question_id"])
    rng = random.Random(SEED)
    dev = []
    total = len(questions)
    types = sorted(by_type)
    quota = {t: round(N_DEV * len(by_type[t]) / total) for t in types}
    while sum(quota.values()) != N_DEV:                      # fix rounding
        t = max(types, key=lambda x: len(by_type[x]))
        quota[t] += 1 if sum(quota.values()) < N_DEV else -1
    for t in types:
        ids = sorted(by_type[t])
        rng.shuffle(ids)
        dev += ids[:quota[t]]
    dev = set(dev)
    return dev, {q["question_id"] for q in questions} - dev


def prepare():
    qs = json.load(open(SRC, encoding="utf-8"))
    dev, test = split_ids(qs)
    questions, nodes = {}, []                                 # node = dict per turn per question
    texts = {}                                                # hash -> text (unique turn texts)
    for q in qs:
        qid = q["question_id"]
        first = len(nodes)
        for si, (sid, sess) in enumerate(zip(q["haystack_session_ids"], q["haystack_sessions"])):
            for ti, turn in enumerate(sess):
                h = norm_hash(turn["content"])
                texts.setdefault(h, turn["content"])
                nodes.append(dict(qid=qid, si=si, ti=ti, sid=sid, role=turn["role"], h=h,
                                  rel=bool(turn.get("has_answer", False)),
                                  prev=norm_hash(sess[ti - 1]["content"]) if ti > 0 else None,
                                  next=norm_hash(sess[ti + 1]["content"]) if ti + 1 < len(sess) else None))
        questions[qid] = dict(qid=qid, type=q["question_type"], question=q["question"], answer=q["answer"],
                              answer_session_ids=list(q["answer_session_ids"]),
                              split="dev" if qid in dev else "test", abs=qid.endswith("_abs"),
                              node_range=(first, len(nodes)))
    out = dict(questions=questions, nodes=nodes, texts=texts)
    with open(os.path.join(DATA, "prep.pkl"), "wb") as f:
        pickle.dump(out, f)
    return out


def facts(p):
    qs, nodes = p["questions"], p["nodes"]
    f = {"questions": len(qs), "turns_total": len(nodes), "unique_turn_texts_all": len(p["texts"])}
    for sp in ("dev", "test"):
        ids = [q for q in qs.values() if q["split"] == sp]
        nn = [n for q in ids for n in nodes[q["node_range"][0]:q["node_range"][1]]]
        f[sp] = {"questions": len(ids), "abs": sum(q["abs"] for q in ids),
                 "types": dict(collections.Counter(q["type"] for q in ids)),
                 "turns": len(nn), "unique_texts": len({n["h"] for n in nn}),
                 "relevant_turns": sum(n["rel"] for n in nn),
                 "non_abs_without_rel_turn": sum(1 for q in ids if not q["abs"] and not any(
                     n["rel"] for n in nodes[q["node_range"][0]:q["node_range"][1]]))}
    dev_h = {n["h"] for q in qs.values() if q["split"] == "dev" for n in nodes[q["node_range"][0]:q["node_range"][1]]}
    test_h = {n["h"] for q in qs.values() if q["split"] == "test" for n in nodes[q["node_range"][0]:q["node_range"][1]]}
    f["test_unique_not_in_dev"] = len(test_h - dev_h)
    f["dev_test_text_overlap"] = len(dev_h & test_h)
    roles = collections.Counter(n["role"] for n in nodes)
    f["roles"] = dict(roles)
    return f


if __name__ == "__main__":
    p = prepare()
    print(json.dumps(facts(p), indent=1))
