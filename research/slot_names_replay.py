#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/slot_names_replay.py

Replays the (entity, attribute) names a model actually wrote during a chained
run through two versions of Memory._find_slot, and scores both against ground
truth. Offline, no model API.

Input: the JSONL written by cost_comparison.py --writes-log. Each row is one
write: arm, round, turn, chain, raw entity/attribute/value/note.

Ground truth: within one chain, writes about the same quantity belong to one
slot and different quantities to different slots. The quantity of a write is
assigned by quantity() from the chain's own vocabulary; rows it cannot place
are reported, not guessed. Writes made on filler turns are corpus facts; each
(normalised entity, attribute) of those is its own quantity.

  slot miss     a write whose quantity already had a live record, and that
                record was not retired
  false merge   a write retired a record of a different quantity
  slots         distinct live slots at the end, against distinct quantities

Versions are loaded straight from git, so the replay does not depend on the
working tree: --old 151d2ae --new df984fa by default.

Usage: python3 research/slot_names_replay.py writes.jsonl [--old REV] [--new REV]
Licensed under the Business Source License 1.1 (SPDX: BUSL-1.1); see research/LICENSE.
"""
import argparse
import collections
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"

# chain -> [(quantity, regex)], first match wins. The regex runs over
# "e=<entity> a=<attribute> v=<value> n=<note>", lower case. Written from the
# names DeepSeek produced on 2026-11-06 (writes-ds3.jsonl, 216 writes), and
# deciding only what is one quantity, never which slot it should land in.
# Judgement calls, stated so they can be disputed:
#   simulated_cell_baseline and simulated_cell are one cell (the chain says so)
#   SHM speed is one quantity whichever entity carries it (SHM, source_x, ...)
#   the trust of source X and of source Y are two quantities
#   agent_alpha.key, .key_status and ed25519_k1.key_status are one: K1's state
#   rights of alpha and of beta are two; visit count/location are not the log
VOCAB = {
    "DryLab: перезапись рецептуры": [
        ("sensor", r"e=loop_sensor"),
        ("coolant_volume", r"a=\S*volume"),
        ("flow_index", r"a=\S*flow"),
        ("temperature", r"a=\S*temp"),
    ],
    "TFLN MZI: каскадный пересчёт": [
        ("phi2_relation", r"a=coupling_relation"),
        ("phi2", r"e=tile_2"),
        ("phi1", r"e=(tile_1|phi_1)"),
    ],
    "TokenShield: отзыв ключа": [
        ("tx_attempt", r"transaction|signing_attempt|latest_signing_use"),
        ("anomaly", r"e=sybil"),
        ("beta_key", r"e=agent_beta a=(key|rights_holder_with_key)"),
        ("beta_rights", r"e=agent_beta a=rights"),
        ("alpha_rights", r"e=agent_alpha a=rights"),
        ("alpha_status", r"e=agent_alpha a=(status|slash_status|slashing_status)"),
        ("alpha_deposit", r"a=deposit"),
        ("k1_state", r"e=agent_alpha a=key|e=(ed25519_k1|key_ed25519_k1) a=key_status"),
    ],
    "Zero-PII: политика отменяет данные": [
        ("policy", r"e=compliance"),
        ("identifier", r"a=(replacement_identifier|identifier_status|raw_identifier_status)"),
        ("visit_count", r"a=(visit_count|gate_entry_count)"),
        ("visit_location", r"a=visit_location"),
        ("visit_log", r"a=(raw_visit_log|visit_log|visit_log_status)"),
    ],
    "Уоддингтон: длинный горизонт": [
        ("h3k9ac", r"a=h3k9ac"),
        ("line1", r"a=line"),
        ("sirt6", r"a=sirt6"),
        ("exposure", r"a=exposure"),
    ],
    "Разрешение противоречий по доверию": [
        ("speed_source", r"a=transit_speed_source"),
        ("trust_x", r"e=source_x a=trust"),
        ("trust_y", r"e=source_y a=trust"),
        ("shm_speed", r"speed|transit"),
    ],
}


def quantity(row):
    """Quantity label of one write, or None when it cannot be placed."""
    if row["kind"] == "filler" or not row.get("chain"):
        return "filler:%s.%s" % (str(row.get("entity")).lower(),
                                 str(row.get("attribute")).lower())
    text = "e=%s a=%s v=%s n=%s" % tuple(str(row.get(k) or "") for k in
                                         ("entity", "attribute", "value", "note"))
    text = text.lower()
    for q, pat in VOCAB.get(row["chain"], []):
        if re.search(pat, text):
            return "%s:%s" % (row["chain"][:12], q)
    return None


def load_version(rev):
    """A git revision, or "worktree" for the file as it is on disk."""
    if rev == "worktree":
        with open(os.path.join(HERE, "agent_memory_traffic.py"), "rb") as fh:
            src = fh.read()
    else:
        src = subprocess.run(["git", "show", "%s:research/agent_memory_traffic.py" % rev],
                             cwd=HERE, check=True, capture_output=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "amt_%s.py" % rev)
    with open(path, "wb") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("amt_%s" % rev, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, HERE)
    spec.loader.exec_module(mod)
    mod.HERE = HERE                          # load_core looks for ../src from here
    mod.SRC = os.path.join(os.path.dirname(HERE), "src")
    return mod


def replay(mod, rows, enc, vecs):
    """One run's writes, in order, through this version's Memory."""
    chunks = [{"id": "bg_%d" % i, "facet": "research", "src": "bg", "text": ""}
              for i in range(len(vecs))]
    mem = mod.Memory(mod.Glyphs(vecs.copy(), chunks), enc, None)
    owner, names = {}, {}
    miss = false_merge = 0
    live = {}                                # quantity -> record index
    merges = []
    for r in rows:
        q = r["_q"]
        mem.turn = r["turn"]
        before = {i for i, m in mem.meta.items() if m["deprecated"]}
        mem.write(r.get("note") or "", r.get("entity"), r.get("attribute"),
                  r.get("value"))
        idx = len(mem.g.chunks) - 1
        newly = {i for i, m in mem.meta.items() if m["deprecated"]} - before
        if q in live and live[q] not in newly and not mem.meta[live[q]]["deprecated"]:
            miss += 1
            merges.append("MISS %s: %s.%s did not retire %s.%s"
                          % (q, r["entity"], r["attribute"],
                             names[live[q]][0], names[live[q]][1]))
        names[idx] = (r["entity"], r["attribute"])
        for i in newly:
            if owner[i] != q:
                false_merge += 1
                merges.append("%s  retired  %s" % (q, owner[i]))
        owner[idx] = q
        live[q] = idx
    slots = len({k for k, i in mem.slots.items() if not mem.meta[i]["deprecated"]})
    return dict(writes=len(rows), miss=miss, false_merge=false_merge,
                slots=slots, quantities=len(live), merges=merges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("writes")
    ap.add_argument("--old", default="151d2ae")
    ap.add_argument("--new", default="df984fa")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.writes, encoding="utf-8")]
    rows = [r for r in rows if r.get("entity") and r.get("attribute")]
    unplaced = [r for r in rows if quantity(r) is None]
    for r in rows:
        r["_q"] = quantity(r) or "unplaced:%s.%s" % (r["entity"], r["attribute"])

    from sentence_transformers import SentenceTransformer

    class Enc:
        def __init__(self):
            self.m, self.cache = SentenceTransformer(ENCODER, device="cpu"), {}

        def encode(self, texts, normalize_embeddings=True, **kw):
            out = []
            for t in texts:
                if t not in self.cache:
                    self.cache[t] = self.m.encode(
                        [t], normalize_embeddings=True,
                        show_progress_bar=False)[0].astype(np.float32)
                out.append(self.cache[t])
            return np.stack(out)

    enc = Enc()
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((200, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    os.makedirs("/tmp/genesis_eval", exist_ok=True)

    runs = collections.OrderedDict()
    for r in rows:
        runs.setdefault((r["arm"], r["round"]), []).append(r)

    print("writes %d in %d runs; unplaced by quantity(): %d"
          % (len(rows), len(runs), len(unplaced)))
    for r in unplaced:
        print("  UNPLACED %s | %s | %s = %s" % (r["chain"], r["entity"],
                                               r["attribute"], r["value"]))
    for rev in (args.old, args.new):
        mod = load_version(rev)
        tot = collections.Counter()
        merges = collections.Counter()
        # Every arm's names are real names; the notes arm just never acted on
        # them. Both are replayed through this version's own Memory.
        for (arm, rnd), rr in runs.items():
            res = replay(mod, rr, enc, vecs)
            for k in ("writes", "miss", "false_merge", "slots", "quantities"):
                tot[k] += res[k]
            merges.update(res["merges"])
        print("\n%s: writes %d, slot misses %d, false merges %d, live slots %d "
              "against %d quantities"
              % (rev, tot["writes"], tot["miss"], tot["false_merge"],
                 tot["slots"], tot["quantities"]))
        for m, n in merges.most_common(40):
            print("  %2d  %s" % (n, m))

    pairs = collections.Counter((str(r["entity"]), str(r["attribute"]))
                                for r in rows)
    print("\nunique (entity, attribute) pairs written: %d" % len(pairs))
    for (e, a), n in sorted(pairs.items()):
        print("  %3d  %s | %s" % (n, e, a))


if __name__ == "__main__":
    main()
