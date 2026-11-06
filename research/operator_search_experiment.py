#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/operator_search_experiment.py

Does the NGP 4.6 reconsolidation operator change what the agent retrieves?

In agent_memory_traffic.py the operator W_phase is updated in Memory.write and
read nowhere on the search path: Glyphs.search ranks by vecs @ q times a slot
weight. So the memory arm and the noengine arm should return byte-identical
results, and the difference between them in paid runs is noise. This stand
checks that offline, then tries to put the operator on the search path and
measures whether that helps against stale values.

Offline only: no model API, embedder paraphrase-multilingual-MiniLM-L12-v2,
background corpus = this repository's own .md files.

Scenario (build_scenario): 20 entities x 2-3 attributes, each attribute
rewritten 3-5 times, writes shuffled with per-slot order kept; after a write,
with probability 0.5, a paraphrased question about the current value of a slot
written so far. Writes and searches go straight into Memory.write/search.

Variants. Hypotheses written before the code, in the form they are tested:

  a) query rotation, score = Re<W q, x>. The engine trains on input x_new with
     target d = (x_new - x_old)/|.|; Oja moves W x_new towards d, so for q near
     x_new, <Wq, x_new> - <Wq, x_old> grows by about eta <q, x_new> |x_new-x_old|
     > 0. Right direction, but eta ~0.02 with clip 0.05 per update and crosstalk
     between 50+ slots in one operator: expected effect is small. Diagnostic:
     shift = margin(Wq) - margin(q), margin = <., x_new> - mean <., x_old>.

  b) rotate agent records at write time, corpus untouched. Each record is
     rotated by a different W (the one at its write); the newest record sees a
     W already trained on its own transition, so it is pushed towards d, which
     points away from x_old - but the query is not rotated. Expected weak.

  c) staleness detector |W x - x| instead of / together with DEPRECATED_W.
     Oja's input is x_new, never x_old: |W x_old - x_old| ~ |<x_new, x_old>|
     while x_new itself moves most. Expected: the detector flags the CURRENT
     value more than the stale ones, i.e. it is backwards. Diagnostic: AUC of
     P(dev_stale > dev_current). Rule fixed in advance: flag if dev > median dev
     over the agent's records.

  g) added after the first run: (a) with the slot rule off, compared against
     nosupersede - can the operator alone remove stale values?

Baselines: noengine (current Memory, engine=None) and nosupersede (slots, no
deprecation weight, DEPRECATED_W = 1.0). "memory" is the current code with the
engine on; the null test says it equals noengine.

Acceptance rule, fixed before the run: a variant counts if its mean paired
improvement over noengine in stale@5 (lower) or fresh@5 (higher) exceeds the
noengine standard deviation across seeds, has the same sign on >= 8 of 10
seeds, and the other metric does not get worse on average.

Usage: python3 research/operator_search_experiment.py [--seeds 10]
Licensed under the Business Source License 1.1 (SPDX: BUSL-1.1); see research/LICENSE.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent_memory_traffic as amt  # noqa: E402

ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"
CORPUS_LIMIT = 500
QUERY_P = 0.5

ENTITIES = ["drylab", "cooling loop", "north warehouse", "pump station",
            "client orion", "server rack", "greenhouse", "boiler",
            "lab freezer", "solar array", "delivery van", "conference hall",
            "water tank", "kiln", "fermenter", "wind turbine", "aquarium",
            "ice rink", "bakery oven", "paint booth"]

NAMES = ["Alice", "Bogdan", "Chen", "Dana", "Emil", "Farah", "Goran", "Hana",
         "Ivo", "Jana", "Kai", "Lena"]
ROOMS = ["room 101", "room 214", "basement", "roof", "hall B", "dock 3",
         "annex", "yard", "floor 4", "bay 7"]
STATES = ["running", "paused", "under repair", "idle", "calibrating",
          "offline", "warming up", "draining"]

ATTRS = {
    "temperature": lambda r: "%d C" % r.integers(5, 95),
    "pressure": lambda r: "%.1f bar" % (r.integers(5, 90) / 10),
    "ph": lambda r: "%.1f" % (r.integers(40, 95) / 10),
    "flow rate": lambda r: "%d L/min" % r.integers(2, 120),
    "voltage": lambda r: "%d V" % r.integers(12, 400),
    "humidity": lambda r: "%d %%" % r.integers(10, 95),
    "owner": lambda r: str(r.choice(NAMES)),
    "status": lambda r: str(r.choice(STATES)),
    "location": lambda r: str(r.choice(ROOMS)),
    "firmware": lambda r: "v%d.%d" % (r.integers(1, 9), r.integers(0, 20)),
}

# --synonyms: the agent does not repeat its own names, as measured in the paid
# runs (loop / cooling_loop / synthesis_loop, temp / temperature). Each write
# picks one spelling at random; questions always use the canonical one.
ENTITY_ALIASES = {
    "drylab": ["dry lab", "the drylab"], "cooling loop": ["loop", "cooling_loop"],
    "north warehouse": ["warehouse north", "n. warehouse"],
    "pump station": ["pump stn", "pumping station"],
    "client orion": ["orion", "orion account"], "server rack": ["rack", "server_rack"],
    "greenhouse": ["green house", "glasshouse"], "boiler": ["main boiler", "boiler unit"],
    "lab freezer": ["freezer", "lab_freezer"], "solar array": ["solar", "pv array"],
    "delivery van": ["van", "delivery_van"], "conference hall": ["conf hall", "hall"],
    "water tank": ["tank", "water_tank"], "kiln": ["the kiln", "kiln 1"],
    "fermenter": ["fermentor", "fermentation tank"], "wind turbine": ["turbine", "wind_turbine"],
    "aquarium": ["fish tank", "the aquarium"], "ice rink": ["rink", "ice_rink"],
    "bakery oven": ["oven", "bakery_oven"], "paint booth": ["booth", "paint_booth"],
}
ATTR_ALIASES = {
    "temperature": ["temp", "temperature reading"], "pressure": ["pressure level", "press"],
    "ph": ["acidity", "ph level"], "flow rate": ["flow", "flowrate"],
    "voltage": ["supply voltage", "volts"], "humidity": ["relative humidity", "rh"],
    "owner": ["responsible person", "owned by"], "status": ["state", "current status"],
    "location": ["placement", "located in"], "firmware": ["firmware version", "fw"],
}

QUERIES = ["What is the current %(attr)s of the %(ent)s?",
           "Right now, which %(attr)s does the %(ent)s have?",
           "Tell me the latest %(attr)s for the %(ent)s.",
           "%(ent)s: what %(attr)s is in effect at the moment?"]


# --- scenario -------------------------------------------------------------

def build_scenario(seed, synonyms=False):
    """Deterministic in seed. Returns slots [(ent, attr, [values])] and events:
    ("write", sid, ver, ent_as_written, attr_as_written) /
    ("query", sid, ver_current, text)."""
    rng = np.random.default_rng(seed)
    srng = np.random.default_rng(10_000 + seed)   # keeps the base scenario equal
    slots = []
    for ent in ENTITIES:
        k = int(rng.integers(2, 4))
        for attr in rng.choice(sorted(ATTRS), size=k, replace=False):
            n = int(rng.integers(3, 6))
            vals, tries = [], 0
            while len(vals) < n and tries < 200:
                v = ATTRS[str(attr)](rng)
                tries += 1
                if v not in vals:
                    vals.append(v)
            slots.append((ent, str(attr), vals))
    order = [sid for sid, s in enumerate(slots) for _ in s[2]]
    rng.shuffle(order)
    seen = [0] * len(slots)
    events, written = [], []
    for sid in order:
        ver = seen[sid]
        seen[sid] += 1
        ent, attr, _ = slots[sid]
        if synonyms:
            ent = str(srng.choice([ent] + ENTITY_ALIASES[ent]))
            attr = str(srng.choice([attr] + ATTR_ALIASES[attr]))
        events.append(("write", sid, ver, ent, attr))
        if ver == 0:
            written.append(sid)
        if rng.random() < QUERY_P:
            q = written[int(rng.integers(len(written)))]
            ent, attr, _ = slots[q]
            tpl = QUERIES[int(rng.integers(len(QUERIES)))]
            events.append(("query", q, seen[q] - 1,
                           tpl % {"ent": ent, "attr": attr}))
    return {"seed": seed, "slots": slots, "events": events,
            "synonyms": synonyms}


# --- environment ----------------------------------------------------------

class CachedEncoder:
    """Same call shape as SentenceTransformer.encode inside Memory._vec, one
    text at a time so a cached vector is bit-identical to a fresh one."""

    def __init__(self, model):
        self.m, self.cache = model, {}

    def encode(self, texts, normalize_embeddings=True, **kw):
        out = []
        for t in texts:
            v = self.cache.get(t)
            if v is None:
                v = self.m.encode([t], normalize_embeddings=True,
                                  show_progress_bar=False)[0].astype(np.float32)
                self.cache[t] = v
            out.append(v)
        return np.stack(out)


def setup():
    from sentence_transformers import SentenceTransformer
    os.makedirs(amt.CACHE, exist_ok=True)
    model = SentenceTransformer(ENCODER, device="cpu")
    chunks = amt.collect_chunks([os.path.dirname(HERE)], CORPUS_LIMIT)
    key = hashlib.sha1(json.dumps([c["text"] for c in chunks]).encode()
                       ).hexdigest()[:12]
    path = os.path.join(amt.CACHE, "opsearch_%s.npy" % key)
    if os.path.exists(path):
        vecs = np.load(path)
    else:
        vecs = model.encode([c["text"] for c in chunks], batch_size=64,
                            show_progress_bar=False,
                            normalize_embeddings=True).astype(np.float32)
        np.save(path, vecs)
    return chunks, np.ascontiguousarray(vecs), CachedEncoder(model)


# --- arms -----------------------------------------------------------------

def _rotate(W, v):
    r = np.real(W @ v)
    return (r / (np.linalg.norm(r) + 1e-12)).astype(np.float32)


class NoSupersede(amt.Memory):
    DEPRECATED_W = 1.0


class QueryRotation(amt.Memory):
    """(a) score = Re<W q, x>: only the query is rotated."""
    _in_search = False

    def search(self, query_text):
        self._in_search = True
        try:
            return super().search(query_text)
        finally:
            self._in_search = False

    def _vec(self, text):
        v = super()._vec(text)
        if self._in_search and self.engine is not None:
            t0 = time.perf_counter()
            v = _rotate(self.engine.W_phase, v)
            self.read_time += time.perf_counter() - t0
        return v


class WriteRotation(amt.Memory):
    """(b) each agent record is stored as Re(W x) with W at write time. The
    operator itself keeps learning from unrotated vectors, so its training
    signal is the same as in the memory arm."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._raw, self._rot = {}, {}

    def write(self, note, entity=None, attribute=None, value=None):
        for i, v in self._raw.items():
            self.g.vecs[i] = v
        out = super().write(note, entity, attribute, value)
        idx = len(self.g.chunks) - 1
        self._raw[idx] = self.g.vecs[idx].copy()
        if self.engine is not None:
            r = _rotate(self.engine.W_phase, self._raw[idx])
            self._rot[idx] = r
            self.g.packed[idx] = np.frombuffer(
                bytes(self.g.encode(r, raw=True)), dtype=np.uint8)
        for i, v in self._rot.items():
            self.g.vecs[i] = v
        return out


class Detector(amt.Memory):
    """(c) a record is flagged stale if |W x - x| exceeds the median over the
    agent's records. REPLACE=True drops the slot rule (DEPRECATED_W) and keeps
    only the detector; False multiplies both."""
    REPLACE = True
    _flag = frozenset()

    def search(self, query_text):
        t0 = time.perf_counter()
        self._flag = set()
        if self.engine is not None and self.meta:
            idx = np.fromiter(self.meta, int)
            X = self.g.vecs[idx].T.astype(np.complex128)
            dev = np.linalg.norm(self.engine.W_phase @ X - X, axis=0)
            self._flag = set(idx[dev > np.median(dev)].tolist())
        self.read_time += time.perf_counter() - t0
        return super().search(query_text)

    def _weight(self, idx):
        m = self.meta.get(idx)
        if m is None:
            return 1.0
        w = max(math.exp(-(self.latest - m["turn"]) / self.tau),
                self.DECAY_FLOOR)
        if m["deprecated"] and not self.REPLACE:
            w *= self.DEPRECATED_W
        if idx in self._flag:
            w *= self.DEPRECATED_W
        return w


class DetectorTogether(Detector):
    REPLACE = False


class QueryRotationNoSlots(QueryRotation):
    """(g) query rotation with the slot rule switched off: can the operator
    alone push stale values out, where nothing else does? Compared against
    nosupersede, not noengine. Added after the first run showed (a) moves the
    margin in the right direction (+2.6e-03, 90 % of queries) but too little
    to matter next to DEPRECATED_W."""
    DEPRECATED_W = 1.0


ARMS = [  # name, class, engine on
    ("noengine", amt.Memory, False),
    ("nosupersede", NoSupersede, False),
    ("memory", amt.Memory, True),
    ("a_query_rot", QueryRotation, True),
    ("b_write_rot", WriteRotation, True),
    ("c_detector", Detector, True),
    ("c_det+slots", DetectorTogether, True),
    ("g_rot_noslots", QueryRotationNoSlots, True),
]


# --- run ------------------------------------------------------------------

def run_arm(cls, use_engine, sc, chunks, vecs, enc, diagnostics=False):
    eng = amt.load_engine(vecs.shape[1]) if use_engine else None
    mem = cls(amt.Glyphs(vecs.copy(), list(chunks)), enc, eng)
    note_of = {}                     # note id -> (sid, ver)
    rec = {}                         # (sid, ver) -> index in glyphs
    owner = {}                       # index in glyphs -> sid
    results, stale, fresh = [], [], []
    shifts = []
    slot_miss = false_dep = lost_to_false_dep = 0
    for turn, ev in enumerate(sc["events"]):
        mem.turn = turn
        if ev[0] == "write":
            sid, ver, ent, attr = ev[1], ev[2], ev[3], ev[4]
            vals = sc["slots"][sid][2]
            before = {i for i, m in mem.meta.items() if m["deprecated"]}
            mem.write("", entity=ent, attribute=attr, value=vals[ver])
            note_of["note_%d" % len(mem.notes)] = (sid, ver)
            rec[(sid, ver)] = len(mem.g.chunks) - 1
            # Slot accounting: did this rewrite retire its own previous value,
            # and did it retire anything it should not have?
            newly = {i for i, m in mem.meta.items() if m["deprecated"]} - before
            own = rec.get((sid, ver - 1))
            if ver > 0 and not mem.meta[own]["deprecated"]:
                slot_miss += 1          # previous value still live
            false_dep += sum(1 for i in newly if owner[i] != sid)
            owner[rec[(sid, ver)]] = sid
            continue
        _, sid, cur, text = ev
        rows = mem.search(text)
        results.append(rows)
        hits = [note_of.get(r["id"]) for r in rows]
        fresh.append(any(h == (sid, cur) for h in hits))
        if not fresh[-1] and mem.meta[rec[(sid, cur)]]["deprecated"]:
            lost_to_false_dep += 1
        if cur > 0:
            stale.append(any(h is not None and h[0] == sid and h[1] < cur
                             for h in hits))
            if diagnostics and eng is not None:
                byver = {v: i for i, v in note_of.items()}
                ids = {c["id"]: i for i, c in enumerate(mem.g.chunks)
                       if c["id"].startswith("note_")}
                xn = mem.g.vecs[ids[byver[(sid, cur)]]]
                xo = [mem.g.vecs[ids[byver[(sid, v)]]] for v in range(cur)]
                q = enc.encode([text])[0]
                wq = np.real(eng.W_phase @ q)
                m0 = q @ xn - np.mean([q @ x for x in xo])
                m1 = wq @ xn - np.mean([wq @ x for x in xo])
                shifts.append(float(m1 - m0))
    out = {"stale": float(np.mean(stale)), "fresh": float(np.mean(fresh)),
           "lat_us": 1e6 * mem.read_time / max(mem.reads, 1),
           "queries": len(fresh), "stale_queries": len(stale),
           "superseded": mem.superseded,
           "expected_superseded": sum(1 for e in sc["events"]
                                      if e[0] == "write" and e[2] > 0),
           "slot_miss": slot_miss, "false_dep": false_dep,
           "lost_to_false_dep": lost_to_false_dep,
           "results": results}
    if eng is not None:
        I = np.eye(eng.dim, dtype=np.complex128)
        out["w_shift"] = float(np.linalg.norm(eng.W_phase - I, "fro"))
    if diagnostics and eng is not None:
        out["shift_mean"] = float(np.mean(shifts))
        out["shift_pos"] = float(np.mean(np.array(shifts) > 0))
        # detector AUC: P(dev of a stale record > dev of a current record)
        cur_ver = {}
        for v_id, (s, v) in note_of.items():
            cur_ver[s] = max(cur_ver.get(s, -1), v)
        ids = {c["id"]: i for i, c in enumerate(mem.g.chunks)
               if c["id"].startswith("note_")}
        dev = {}
        for v_id in note_of:
            x = mem.g.vecs[ids[v_id]].astype(np.complex128)
            dev[v_id] = float(np.linalg.norm(eng.W_phase @ x - x))
        ds = [dev[k] for k, (s, v) in note_of.items() if v < cur_ver[s]]
        dc = [dev[k] for k, (s, v) in note_of.items() if v == cur_ver[s]]
        out["det_auc"] = float(np.mean([a > b for a in ds for b in dc]))
        out["dev_stale"], out["dev_current"] = float(np.mean(ds)), float(np.mean(dc))
    return out


def verdict(base, var, other_ok_tol=0.0):
    """Paired against noengine, per the rule in the module docstring."""
    res = {}
    for metric, better in (("stale", -1), ("fresh", +1)):
        d = better * (np.array(var[metric]) - np.array(base[metric]))
        sd = float(np.std(base[metric], ddof=1)) if len(d) > 1 else 0.0
        res[metric] = {"mean_gain": float(d.mean()), "sd_noengine": sd,
                       "seeds_better": int((d > 0).sum()),
                       "seeds_same": int((d == 0).sum())}
    n = len(base["stale"])
    win = False
    for m, o in (("stale", "fresh"), ("fresh", "stale")):
        r = res[m]
        if (r["mean_gain"] > r["sd_noengine"] and r["seeds_better"] >= math.ceil(0.8 * n)
                and res[o]["mean_gain"] >= -other_ok_tol):
            win = True
    res["counts"] = win
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(amt.CACHE,
                                                  "operator_search.json"))
    ap.add_argument("--synonyms", action="store_true",
                    help="the agent writes names in varying spellings")
    args = ap.parse_args()
    chunks, vecs, enc = setup()
    print("корпус: %d фрагментов, dim %d" % (len(chunks), vecs.shape[1]))
    per = {name: {"stale": [], "fresh": [], "lat_us": []} for name, _, _ in ARMS}
    diag = []
    slots_tot = {"rewrites": 0, "slot_miss": 0, "false_dep": 0,
                 "lost_to_false_dep": 0, "queries": 0}
    for seed in range(args.seeds):
        sc = build_scenario(seed, synonyms=args.synonyms)
        base_res = None
        for name, cls, on in ARMS:
            r = run_arm(cls, on, sc, chunks, vecs, enc,
                        diagnostics=(name == "memory"))
            for k in ("stale", "fresh", "lat_us"):
                per[name][k].append(r[k])
            if name == "noengine":
                base_res = r["results"]
                n_q, n_sq = r["queries"], r["stale_queries"]
                sup = (r["superseded"], r["expected_superseded"])
                slots_tot["rewrites"] += r["expected_superseded"]
                for k in ("slot_miss", "false_dep", "lost_to_false_dep",
                          "queries"):
                    slots_tot[k] += r[k]
            if name == "memory":
                same = json.dumps(r["results"]) == json.dumps(base_res)
                diag.append({k: r[k] for k in ("shift_mean", "shift_pos",
                                               "det_auc", "dev_stale",
                                               "dev_current", "w_shift")})
                diag[-1]["identical_to_noengine"] = same
        print("seed %d: запросов %d (со старыми версиями %d), вытеснено %d из "
              "%d ожидаемых, memory==noengine: %s"
              % (seed, n_q, n_sq, sup[0], sup[1], diag[-1]["identical_to_noengine"]))

    print("\nслоты (noengine, все зёрна, имена %s): перезаписей %d, промахов "
          "слота %d (%.1f %%), ложных вытеснений %d, актуальное потеряно из-за "
          "них в %d из %d запросов (%.1f %%)"
          % ("с синонимами" if args.synonyms else "точные",
             slots_tot["rewrites"], slots_tot["slot_miss"],
             100.0 * slots_tot["slot_miss"] / slots_tot["rewrites"],
             slots_tot["false_dep"], slots_tot["lost_to_false_dep"],
             slots_tot["queries"],
             100.0 * slots_tot["lost_to_false_dep"] / slots_tot["queries"]))
    print("\n%-13s %16s %16s %12s" % ("ветка", "stale@5", "fresh@5", "поиск, мкс"))
    print("-" * 62)
    for name, _, _ in ARMS:
        p = per[name]
        print("%-13s %8.3f ± %.3f %8.3f ± %.3f %12.0f"
              % (name, np.mean(p["stale"]), np.std(p["stale"], ddof=1),
                 np.mean(p["fresh"]), np.std(p["fresh"], ddof=1),
                 np.mean(p["lat_us"])))
    print("\nвердикт против noengine (правило в docstring):")
    verdicts = {}
    for name, _, on in ARMS:
        if name == "noengine":
            continue
        v = verdict(per["noengine"], per[name])
        verdicts[name] = v
        print("  %-13s stale: выигрыш %+.3f (sd %.3f, лучше на %d, равно на %d)  "
              "fresh: %+.3f (sd %.3f, лучше на %d, равно на %d)  -> %s"
              % (name, v["stale"]["mean_gain"], v["stale"]["sd_noengine"],
                 v["stale"]["seeds_better"], v["stale"]["seeds_same"],
                 v["fresh"]["mean_gain"], v["fresh"]["sd_noengine"],
                 v["fresh"]["seeds_better"], v["fresh"]["seeds_same"],
                 "ЗАСЧИТАН" if v["counts"] else "нет"))
    g = verdict(per["nosupersede"], per["g_rot_noslots"])
    verdicts["g_vs_nosupersede"] = g
    print("  g против nosupersede: stale %+.3f (sd %.3f, лучше на %d)  "
          "fresh %+.3f (sd %.3f, лучше на %d)  -> %s"
          % (g["stale"]["mean_gain"], g["stale"]["sd_noengine"],
             g["stale"]["seeds_better"], g["fresh"]["mean_gain"],
             g["fresh"]["sd_noengine"], g["fresh"]["seeds_better"],
             "сдвиг есть" if g["counts"] else "нет"))
    print("\nдиагностика оператора (ветка memory, среднее по зёрнам):")
    print("  ||W - I|| в конце              : %.3e" % np.mean([d["w_shift"] for d in diag]))
    print("  (а) сдвиг отрыва нового от старого Re<Wq,.> - <q,.>: %+.2e, "
          "положителен в %.0f %% запросов"
          % (np.mean([d["shift_mean"] for d in diag]),
             100 * np.mean([d["shift_pos"] for d in diag])))
    print("  (в) AUC P(dev устар. > dev актуал.): %.3f  "
          "(dev устар. %.2e, актуал. %.2e)"
          % (np.mean([d["det_auc"] for d in diag]),
             np.mean([d["dev_stale"] for d in diag]),
             np.mean([d["dev_current"] for d in diag])))
    with open(args.out, "w") as fh:
        json.dump({"synonyms": args.synonyms, "slots": slots_tot,
                   "per_seed": per, "verdicts": verdicts, "diagnostics": diag},
                  fh, indent=1)
    print("\nJSON: %s" % args.out)


if __name__ == "__main__":
    main()
