# Business Source License 1.1 - see research/LICENSE
"""eval_longmemeval.py - LongMemEval_S turn-level retrieval: B0..B5, G1, H on one split.
2026-11-21 · CLI: embed-base | embed-llm | run. Results go to results/.

  python eval_longmemeval.py embed-base   # F0, F3, questions (no LLM needed)
  python eval_longmemeval.py embed-llm    # F1, F2, B2 concat (after llm_f12.py collect)
  python eval_longmemeval.py run          # all methods + metrics + stats on dev
"""
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encoder_v1 as enc  # noqa: E402

DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
SPLIT = os.environ.get("HG_SPLIT", "dev")


def load_prep():
    return pickle.load(open(os.path.join(DATA, "prep.pkl"), "rb"))


def split_nodes(p, split=SPLIT):
    qs = [q for q in p["questions"].values() if q["split"] == split]
    qs.sort(key=lambda q: q["qid"])
    idx = [i for q in qs for i in range(*q["node_range"])]
    return qs, idx


def llm_faces():
    """hash -> (summary, entities) or None on failure; from data/llm_cache.jsonl."""
    out = {}
    path = os.path.join(DATA, "llm_cache.jsonl")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r["h"]] = (r["summary"], r["entities"]) if r.get("ok") else None
    return out


def face_strings(p, idx, key, llm=None):
    """Per node: dict of face strings for model `key`; plus truncation counters."""
    texts, nodes = p["texts"], p["nodes"]
    maxlen = enc.model(key).max_seq_length
    rows, cut = [], {"F3_parts_cut": 0, "F3_parts": 0}
    for i in idx:
        n = nodes[i]
        f0 = texts[n["h"]]
        f3, c = enc.f3_text(key, texts[n["prev"]] if n["prev"] else "", texts[n["next"]] if n["next"] else "")
        cut["F3_parts_cut"] += c
        cut["F3_parts"] += (n["prev"] is not None) + (n["next"] is not None)
        r = {"F0": f0, "F3": f3 if f3 else f0}
        if llm is not None:
            s = llm.get(n["h"])
            r["F1"], r["F2"] = (s if s else (f0, f0))
            r["llm_fail"] = s is None
            r["B2"] = "\n".join([r["F1"], r["F2"], f0, r["F3"]])
        rows.append(r)
    return rows, cut, maxlen


def embed_base():
    p = load_prep()
    qs, idx = split_nodes(p)
    stats = {}
    for key in enc.MODELS:
        t0 = time.time()
        rows, cut, maxlen = face_strings(p, idx, key)
        uniq_f0 = sorted({r["F0"] for r in rows})
        ntok = enc.n_tokens(key, uniq_f0)
        stats[key] = {"max_seq_length": maxlen, "revision": enc.model_revision(key),
                      "F0_unique": len(uniq_f0), "F0_truncated_frac": float(np.mean(np.array(ntok) > maxlen - 2)),
                      "F3_parts_cut_frac": cut["F3_parts_cut"] / max(1, cut["F3_parts"])}
        uniq_f3 = sorted({r["F3"] for r in rows})
        stats[key]["F3_truncated_frac"] = float(np.mean(np.array(enc.n_tokens(key, uniq_f3)) > maxlen - 2))
        enc.embed(key, "Q", [q["question"] for q in qs])
        enc.embed(key, "F0", uniq_f0)
        enc.embed(key, "F3", uniq_f3)
        stats[key]["seconds"] = round(time.time() - t0, 1)
        print(key, stats[key], flush=True)
    os.makedirs(RES, exist_ok=True)
    json.dump(stats, open(os.path.join(RES, "embed_base_%s.json" % SPLIT), "w"), indent=1)


def embed_llm():
    p = load_prep()
    qs, idx = split_nodes(p)
    llm = llm_faces()
    stats = {}
    for key in enc.MODELS:
        rows, _, maxlen = face_strings(p, idx, key, llm)
        miss = sum(1 for i in idx if p["nodes"][i]["h"] not in llm)
        uniq_b2 = sorted({r["B2"] for r in rows})
        stats[key] = {"nodes": len(rows), "llm_missing": miss,
                      "llm_fail_frac": float(np.mean([r["llm_fail"] for r in rows])),
                      "B2_truncated_frac": float(np.mean(np.array(enc.n_tokens(key, uniq_b2)) > maxlen - 2))}
        enc.embed(key, "F1", sorted({r["F1"] for r in rows}))
        enc.embed(key, "F2", sorted({r["F2"] for r in rows}))
        enc.embed(key, "B2", uniq_b2)
        print(key, stats[key], flush=True)
    json.dump(stats, open(os.path.join(RES, "embed_llm_%s.json" % SPLIT), "w"), indent=1)


# ------------------------------------------------------------------ run

def run(stage):
    import baselines as bl
    import stats as st
    p = load_prep()
    qs, idx = split_nodes(p)
    nodes, texts = p["nodes"], p["texts"]
    full = stage == "full"
    llm = llm_faces() if full else None
    faces_order = ["F0", "F1", "F2", "F3"] if full else ["F0", "F3"]
    rng = np.random.default_rng(0)
    perq, summary, extra = [], {}, {}
    # BM25 once (model independent)
    bm25 = {}
    for q in qs:
        a, b = q["node_range"]
        assert all(nodes[i]["qid"] == q["qid"] for i in range(a, b))          # no cross-question leakage
        bm25[q["qid"]] = bl.bm25([texts[nodes[i]["h"]] for i in range(a, b)], q["question"])
    for key in enc.MODELS:
        rows, _, _ = face_strings(p, idx, key, llm)
        row_of = {i: r for i, r in zip(idx, rows)}
        look = {f: enc.embed(key, f, []) for f in faces_order + (["B2"] if full else [])}
        qv = enc.embed(key, "Q", [])
        sample = rng.choice(len(idx), min(20000, len(idx)), replace=False)
        fit = np.vstack([np.stack([look[f](row_of[idx[j]][f]) for j in sample]) for f in faces_order])
        variants = [("main", 128, True, "pca"), ("main", 384, True, "pca")]
        if full:
            variants += [("nocenter", 128, False, "pca"), ("nocenter", 384, False, "pca"), ("gauss", 128, True, "gauss")]
        for vname, d, center, proj in variants:
            prep = enc.Prep(fit, d=d, center=center, proj=proj)
            tag = "%s_d%d%s" % (key, d, "" if vname == "main" else "_" + vname)
            degen_n, ortho_max, anti, lat = 0, 0.0, [], []
            for q in qs:
                if q["abs"]:
                    continue
                a, b = q["node_range"]
                ids = list(range(a, b))
                F = np.stack([prep(np.stack([look[f](row_of[i][f]) for i in ids])) for f in faces_order], 1)
                qq = prep(qv(q["question"])[None])[0]
                rel = np.array([nodes[i]["rel"] for i in ids])
                sess = np.array([nodes[i]["si"] for i in ids])
                srel = np.zeros(sess.max() + 1, bool)
                hay = p["questions"][q["qid"]]
                sid_of = {nodes[i]["si"]: nodes[i]["sid"] for i in ids}
                for s_i, sid in sid_of.items():
                    srel[s_i] = sid in hay["answer_session_ids"]
                sc = {}
                if vname == "main":
                    sc["B0"] = bm25[q["qid"]]
                    sc["B1"] = bl.flat(F[:, 0], qq)
                if full:
                    if vname == "main":
                        B2 = prep(np.stack([look["B2"](row_of[i]["B2"]) for i in ids]))
                        sc["B2"] = bl.flat(B2, qq)
                        sc["B3"] = bl.mean_face(F, qq)
                        sc["B4"] = bl.max_sim(F, qq)
                        sc["B5"] = bl.sum_sq(F, qq)
                    t0 = time.perf_counter()
                    Y, degen, err = enc.subspaces(F, ids)
                    g1, _ = enc.g1_scores(Y, qq.astype(np.float32))
                    lat.append((time.perf_counter() - t0) * 1e3)
                    sc["G1"] = g1
                    degen_n += int(degen.sum())
                    ortho_max = max(ortho_max, err)
                    top = np.argsort(-g1)[:10]
                    anti.append(float(((F[top] @ qq).sum(axis=1) < 0).mean()))
                    if vname == "main":
                        sc["H_B4"] = bl.rrf(sc["B4"], sc["B0"])          # H = the one whose base wins on dev
                        sc["H_G1"] = bl.rrf(sc["G1"], sc["B0"])
                        for sub, cols in (("G1_F0F1", [0, 1]), ("G1_F0F1F2", [0, 1, 2])):
                            Ys, _, _ = enc.subspaces(F[:, cols], ids)
                            sc[sub] = enc.g1_scores(Ys, qq.astype(np.float32))[0]
                for m, s in sc.items():
                    met = st.metrics(s, rel)
                    smet = st.metrics(st.session_scores(s, sess, len(srel)), srel)
                    r = dict(qid=q["qid"], type=q["type"], model=key, d=d, variant=vname, method=m)
                    r.update({k: v for k, v in met.items()})
                    r.update({"sess_" + k: v for k, v in smet.items()})
                    perq.append(r)
            if full:
                extra[tag] = {"degenerate_nodes": degen_n, "ortho_err_max": ortho_max,
                              "g1_top10_antidirection_frac": float(np.mean(anti)),
                              "g1_latency_ms_per_question_p50": float(np.median(lat))}
            print(tag, "done", flush=True)
    import pandas as pd
    df = pd.DataFrame(perq)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, "perq_%s_%s.csv" % (SPLIT, stage)), index=False)
    # hybrid H: RRF(best of B4/G1 on this split, BM25) - added to perq separately in report step
    cols = ["recall_any@1", "recall_any@5", "recall_any@10", "recall_all@5", "recall_all@10", "ndcg@10", "mrr",
            "sess_recall_any@5", "sess_recall_any@10"]
    tab = df.groupby(["model", "d", "variant", "method"])[cols].mean().mul(100).round(2)
    tab.to_csv(os.path.join(RES, "table_%s_%s.csv" % (SPLIT, stage)))
    print(tab.to_string())
    if full:
        tests = {}
        for (key, d), g in df[df.variant == "main"].groupby(["model", "d"]):
            means = g.groupby("method")["recall_any@5"].mean()
            best = means[["B2", "B3", "B4", "B5"]].idxmax()
            a = g[g.method == "G1"].set_index("qid")["recall_any@5"]
            b = g[g.method == best].set_index("qid")["recall_any@5"].loc[a.index]
            diff, lo, hi = st.paired_bootstrap(a.values, b.values)
            n01, n10, pv = st.mcnemar_exact(a.values > 0, b.values > 0)
            tests["%s_d%d" % (key, d)] = {"best_baseline": best, "G1": float(a.mean()) * 100, "best": float(b.mean()) * 100,
                                          "diff_pp": diff * 100, "ci95_pp": [lo * 100, hi * 100],
                                          "mcnemar_best_only_vs_g1_only": [n01, n10], "mcnemar_p": pv, "n_questions": len(a)}
        json.dump({"tests": tests, "extra": extra}, open(os.path.join(RES, "stats_%s.json" % SPLIT), "w"), indent=1)
        print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "embed-base":
        embed_base()
    elif cmd == "embed-llm":
        embed_llm()
    elif cmd == "run":
        run(sys.argv[2] if len(sys.argv) > 2 else "full")
    else:
        raise SystemExit("unknown command " + cmd)
