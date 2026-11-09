#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Business Source License 1.1 - см. research/LICENSE
"""
research/swarm/swarm_p2p_simulation.py

Координатор симулятора роя. Строит мир и расписание событий из зерна,
запускает узлы отдельными процессами (multiprocessing, spawn), собирает их
журналы и считает метрики. Сам координатор в обмене не участвует: никакого
центрального сервера у узлов нет, координатор только читает итоги.

  python swarm_p2p_simulation.py --branch B --duration 60 --seed 1
  python swarm_p2p_simulation.py --grid all --workers 2 --out results.jsonl
  python swarm_p2p_simulation.py --report results.jsonl

Всё время в узлах симулированное; масштаб к реальному времени не задаётся,
прогон идёт с максимальной скоростью, измеренное сжатие пишется в итог.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import statistics
import sys
import threading
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import swarm_net as net                                     # noqa: E402
import swarm_node as sn                                     # noqa: E402

TICK_MS = 100
CONFLICT_RATE_PER_S = 5.0       # смен истины в секунду на рой -> ~20 % событий

DEFAULTS = dict(n=5, seed=1, branch="B", loss=0.05, topology="full",
                partition="none", attack="none", attacker=4, trust_init=0.5,
                duration_s=600, events_until_s=480, tick_ms=TICK_MS, K=10,
                revoke_at_ms=240_000, base_port=21000, slot_digest=True,
                batch_ms=500, slash=True,          # v1.1; v1.0 = batch_ms 0, slash False
                rate_mode="kseq",      # v1.2 kseq; v1.2 = "seq", v1.1 = "hlc"
                sensor_noise=0.0,      # v1.3: доля неверных измерений датчика
                slash_window=3,        # v1.3: порог оконного штрафа; 1 = v1.2
                refute_quorum=2,       # v1.3: опровержение действует от N узлов
                confirm_quorum=1,      # v1.3: подтверждение действует от N узлов
                quorum_scope="all", confirm_cap=None, refute_cap=None,
                rebut=False, admit_rule="trust", liar_k=0,
                vote_order="arrival", sensor_filter=True,   # v1.4
                self_trust="full", pull_priority="log", pull_quiet_cap=0,
                self_admit="always", arbiter="scalar", kinematic_gate=0,   # v2.0
                apsa_rule=None, tomb_gap=20, liar_vote=False)

PARTITIONS = {"none": "none",
              "isolate": "isolate:2@180+120",
              "split": "split:0,1|2,3,4@120+300"}


# --------------------------------------------------------------------- мир

def build_world(seed, n, events_until_ms, tick_ms=TICK_MS):
    """Истина по общим ключам и расписание событий каждого узла.

    Конфликт честных узлов - это смена истины в момент t: один узел пишет
    старое значение за 1-25 мс до t, другой новое через 1-25 мс после. Оба
    правы на своё время. Остальные события - 10 в секунду на узел."""
    rng = random.Random("world:%s" % seed)
    shared = sn.shared_keys()
    truth = {}
    for k in shared:
        if k.endswith(".status"):
            truth[k] = ([0], [rng.choice(sn.STATUSES)])
        else:
            truth[k] = ([0], [[rng.randint(0, 200_000), rng.randint(0, 200_000)]])
    changes = []
    t = 0
    while True:
        t += int(rng.expovariate(CONFLICT_RATE_PER_S / 1000.0)) + 1
        if t >= events_until_ms - 1000:
            break
        k = rng.choice(shared)
        old = truth[k][1][-1]
        if k.endswith(".status"):
            new = rng.choice([s for s in sn.STATUSES if s != old])
        else:
            new = [max(0, old[0] + rng.choice((-1, 1)) * rng.randint(500, 1500)),
                   max(0, old[1] + rng.choice((-1, 1)) * rng.randint(500, 1500))]
        truth[k][0].append(t)
        truth[k][1].append(new)
        changes.append((t, k, old, new))

    def noisy(v):
        if isinstance(v, list):
            return [v[0] + rng.randint(-50, 50), v[1] + rng.randint(-50, 50)]
        return v

    sched = [dict() for _ in range(n)]
    pairs = 0
    for pid, (tc, k, old, new) in enumerate(changes):
        a, b = rng.sample(range(n), 2)
        ta, tb = tc - rng.randint(1, 25), tc + rng.randint(1, 25)
        if ta // tick_ms in sched[a] or tb // tick_ms in sched[b] \
                or tb >= events_until_ms:
            continue
        sched[a][ta // tick_ms] = (ta, k, noisy(old), ("C", pid, "old"))
        sched[b][tb // tick_ms] = (tb, k, noisy(new), ("C", pid, "new"))
        pairs += 1
    for i in range(n):
        pos = [rng.randint(0, 200_000), rng.randint(0, 200_000), rng.randint(500, 5000)]
        battery, payload = 10_000, rng.randint(0, 5000)
        for tick in range(events_until_ms // tick_ms):
            if tick in sched[i]:
                continue
            tm = tick * tick_ms + rng.randint(0, tick_ms - 1)
            r = rng.random()
            if r < 0.30:
                pos = [pos[0] + rng.randint(-200, 200), pos[1] + rng.randint(-200, 200),
                       max(0, pos[2] + rng.randint(-50, 50))]
                ev = (tm, "drone_%d.pos" % i, list(pos), None)
            elif r < 0.40:
                battery = max(0, battery - rng.randint(0, 3))
                ev = (tm, "drone_%d.battery" % i, battery, None)
            elif r < 0.45:
                payload = rng.randint(0, 5000)
                ev = (tm, "drone_%d.payload" % i, payload, None)
            else:
                k = rng.choice(shared[:sn.N_SECTORS] if r < 0.80
                               else shared[sn.N_SECTORS:])
                ev = (tm, k, noisy(sn.truth_at(truth, k, tm)), None)
            sched[i][tick] = ev
    schedules = [sorted(s.values(), key=lambda e: (e[0], e[1])) for s in sched]
    total = sum(len(s) for s in schedules)
    return {"truth": truth, "schedules": schedules, "pairs": pairs,
            "conflict_share": 2 * pairs / max(total, 1), "events": total}


# ------------------------------------------------------------------- прогон

def run_once(cfg):
    cfg = dict(DEFAULTS, **cfg)
    n = cfg["n"]
    world = build_world(cfg["seed"], n, cfg["events_until_s"] * 1000,
                        cfg["tick_ms"])
    base = dict(n=n, seed=cfg["seed"], branch=cfg["branch"], loss=cfg["loss"],
                topology=cfg["topology"],
                partition=PARTITIONS.get(cfg["partition"], cfg["partition"]),
                attack=cfg["attack"], attacker=cfg["attacker"],
                trust_init=cfg["trust_init"], tick_ms=cfg["tick_ms"],
                K=cfg["K"], duration_ms=cfg["duration_s"] * 1000,
                events_until_ms=cfg["events_until_s"] * 1000,
                revoke_at_ms=cfg["revoke_at_ms"], base_port=cfg["base_port"],
                truth=world["truth"], drifts=cfg.get("drifts"),
                slot_digest=cfg["slot_digest"], batch_ms=cfg["batch_ms"],
                slash=cfg["slash"], rate_mode=cfg["rate_mode"],
                sensor_noise=cfg["sensor_noise"],
                slash_window=cfg["slash_window"],
                refute_quorum=cfg["refute_quorum"],
                confirm_quorum=cfg["confirm_quorum"],
                quorum_scope=cfg["quorum_scope"], confirm_cap=cfg["confirm_cap"],
                refute_cap=cfg["refute_cap"], rebut=cfg["rebut"],
                admit_rule=cfg["admit_rule"], liar_k=cfg["liar_k"],
                vote_order=cfg["vote_order"], sensor_filter=cfg["sensor_filter"],
                self_trust=cfg["self_trust"], pull_priority=cfg["pull_priority"],
                pull_quiet_cap=cfg["pull_quiet_cap"], self_admit=cfg["self_admit"],
                arbiter=cfg["arbiter"], kinematic_gate=cfg["kinematic_gate"],
                apsa_rule=cfg["apsa_rule"], tomb_gap=cfg["tomb_gap"],
                liar_vote=cfg["liar_vote"])
    ctx = mp.get_context("spawn")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    barrier = ctx.Barrier(n)
    counts = ctx.Array("i", 2 * n * n, lock=False)
    q = ctx.Queue()
    procs = []
    wall0 = time.perf_counter()
    for i in range(n):
        ncfg = dict(base, schedule=world["schedules"][i])
        p = ctx.Process(target=sn.run_node, args=(ncfg, i, barrier, counts, q),
                        daemon=True)
        p.start()
        procs.append(p)
    results = [None] * n
    try:
        for _ in range(n):
            r = q.get(timeout=1800)
            results[r["idx"]] = r
    except queue.Empty:
        for p in procs:
            p.terminate()
        raise RuntimeError("run did not finish: %r" % {k: cfg[k] for k in (
            "branch", "loss", "topology", "partition", "attack", "seed")})
    for p in procs:
        p.join(timeout=30)
    wall = time.perf_counter() - wall0
    m = compute_metrics(cfg, world, results)
    m["wall_s"] = round(wall, 2)
    m["compression"] = round(cfg["duration_s"] / wall, 1)
    rec = {k: cfg[k] for k in ("n", "seed", "branch", "loss", "topology",
                               "partition", "attack", "attacker", "trust_init",
                               "duration_s", "events_until_s", "tick_ms", "K")}
    if not cfg["slot_digest"]:
        rec["slot_digest"] = False
    rec["batch_ms"], rec["slash"] = cfg["batch_ms"], cfg["slash"]
    rec["rate_mode"] = cfg["rate_mode"]
    if cfg["sensor_noise"]:
        rec["sensor_noise"] = cfg["sensor_noise"]
    rec["slash_window"] = cfg["slash_window"]
    if cfg["refute_quorum"] > 1:
        rec["refute_quorum"] = cfg["refute_quorum"]
    if cfg["confirm_quorum"] > 1:
        rec["confirm_quorum"] = cfg["confirm_quorum"]
    if cfg["quorum_scope"] != "all":
        rec["quorum_scope"] = cfg["quorum_scope"]
    if cfg["confirm_cap"]:
        rec["confirm_cap"] = cfg["confirm_cap"]
    if cfg["refute_cap"]:
        rec["refute_cap"] = cfg["refute_cap"]
    if cfg["rebut"]:
        rec["rebut"] = True
    if cfg["admit_rule"] != "trust":
        rec["admit_rule"] = cfg["admit_rule"]
    if cfg["liar_k"]:
        rec["liar_k"] = cfg["liar_k"]
    if cfg["vote_order"] != "arrival":
        rec["vote_order"] = cfg["vote_order"]
    if not cfg["sensor_filter"]:
        rec["sensor_filter"] = False
    if cfg["self_trust"] != "full":
        rec["self_trust"] = cfg["self_trust"]
    if cfg["pull_priority"] != "log":
        rec["pull_priority"] = cfg["pull_priority"]
    if cfg["pull_quiet_cap"]:
        rec["pull_quiet_cap"] = cfg["pull_quiet_cap"]
    if cfg["self_admit"] != "always":
        rec["self_admit"] = cfg["self_admit"]
    if cfg["arbiter"] != "scalar":
        rec["arbiter"] = cfg["arbiter"]
    if cfg["kinematic_gate"]:
        rec["kinematic_gate"] = cfg["kinematic_gate"]
    if cfg["apsa_rule"]:
        rec["apsa_rule"] = cfg["apsa_rule"]
    if cfg["liar_vote"]:
        rec["liar_vote"] = True
    if cfg["arbiter"] != "scalar":
        rec["tomb_gap"] = cfg["tomb_gap"]
    rec["metrics"] = m
    return rec, results


# ----------------------------------------------------------------- метрики

def _states(results, nodes, n_keys, sample_ticks):
    """Картина каждого узла (кортеж eid по ключам) на заданных тактах."""
    out = {}
    for i in nodes:
        ch = results[i]["changes"]
        cur = [""] * n_keys
        p = 0
        snaps = []
        for st in sample_ticks:
            while p < len(ch) and ch[p][0] <= st:
                cur[ch[p][1]] = ch[p][2]
                p += 1
            snaps.append(tuple(cur))
        out[i] = snaps
    return out


def _div(states, nodes, s, kset):
    if len(nodes) < 2:
        return 0.0
    bad = 0
    for k in kset:
        v = states[nodes[0]][s][k]
        if any(states[j][s][k] != v for j in nodes[1:]):
            bad += 1
    return bad / len(kset)


def _r(x, nd=4):
    return None if x is None else round(x, nd)


def compute_metrics(cfg, world, results):
    n, tick = cfg["n"], cfg["tick_ms"]
    keys = sn.keys_universe(n)
    nk = len(keys)
    kidx = {k: i for i, k in enumerate(keys)}
    attacked = cfg["attack"] != "none"
    honest = [i for i in range(n) if not (attacked and i == cfg["attacker"])]
    D = cfg["duration_s"]
    E = cfg["events_until_s"]
    m = {"events_generated": world["events"],
         "conflict_share": _r(world["conflict_share"])}

    labels = {}
    for r in results:
        for (eid, key, vh, l, c, gen, label, tag) in r["own"]:
            labels[eid] = (key, vh, l, c, r["idx"], gen, label, tag)
    # Эталон - все события с меткой «честное», включая правдивые события
    # атакующего по общим ключам (в T5 и T7 он о мире не врёт). Ключи
    # drone_<атакующий>.* из сравнения исключены: правило владельца принимает
    # всё, что атакующий пишет о себе, и это по замыслу не ловится.
    att = cfg["attacker"] if attacked else None
    kset = [k for k, key in enumerate(keys) if att is None or sn.owner_of(key) != att]
    ref = {}
    for eid, (key, vh, l, c, node, gen, label, tag) in labels.items():
        if label != "H":
            continue
        o = (l, c, node, eid)
        if key not in ref or o > ref[key][0]:
            ref[key] = (o, eid)
    ref = {k: v[1] for k, v in ref.items()}

    # --- сходимость
    samples = [s * 10 + 9 for s in range(D)]
    st = _states(results, honest, nk, samples)
    div = [_div(st, honest, s, kset) for s in range(D)]
    final = {i: st[i][-1] for i in honest}
    agree = sum(1 for k in kset
                if all(final[j][k] == final[honest[0]][k] for j in honest)) / len(kset)
    conv_s = None
    if div[-1] == 0:
        s0 = D - 1
        while s0 - 1 >= E and div[s0 - 1] == 0:
            s0 -= 1
        conv_s = s0 + 1 - E
    m["converged"] = div[-1] == 0
    m["converge_s_after_last_event"] = conv_s
    m["final_agree_frac"] = _r(agree)
    m["world_hashes_equal"] = len({results[i]["world_hash"] for i in honest}) == 1
    m["keys_compared"] = len(kset)
    causes = {"lost": 0, "quarantined": 0, "order_or_other": 0}
    ok = 0
    for i in honest:
        arr = results[i]["arrivals"]
        quar = set(results[i]["quarantined"])
        for k in kset:
            r_eid = ref.get(keys[k], "")
            if final[i][k] == r_eid:
                ok += 1
            elif r_eid not in arr:
                causes["lost"] += 1
            elif r_eid in quar:
                causes["quarantined"] += 1
            else:
                causes["order_or_other"] += 1
    tot = len(kset) * len(honest)
    m["ref_match_frac"] = _r(ok / tot)
    m["ref_diff"] = {k: _r(v / tot) for k, v in causes.items()}
    # v2.0: эталон только по слотам, где эталонное событие писал честный узел
    # (правдивые события атакующего после надгробия намеренно не принимаются)
    hk = [k for k in kset if ref.get(keys[k]) in labels
          and labels[ref[keys[k]]][4] in honest]
    okh = sum(1 for i in honest for k in hk if final[i][k] == ref.get(keys[k], ""))
    m["ref_match_honest_frac"] = _r(okh / max(len(hk) * len(honest), 1))
    m["div_per_s"] = [round(x, 3) for x in div]
    _, win = net.parse_partition(PARTITIONS.get(cfg["partition"], cfg["partition"]))
    if win:
        lo, hi = win[0] // 1000, win[1] // 1000
        b0 = max(0, lo - 60)
        baseline = statistics.mean(div[b0:lo]) if lo > b0 else 0.0
        m["div_baseline"] = _r(baseline)
        m["div_during_partition"] = _r(statistics.mean(div[lo:hi]))
        catch = None
        for s in range(hi, D):
            if div[s] <= baseline + 0.02:
                catch = s - hi
                break
        m["catchup_s"] = catch
        m["catchup_ticks"] = None if catch is None else catch * 1000 // tick
    m["div_steady_mean"] = _r(statistics.mean(div[min(60, E // 4):E]))

    # --- трафик (честные узлы, активное окно 0..E)
    tr = {c: [] for c in net.CLASSES}
    pk = {c: [] for c in net.CLASSES}
    envs = []
    totals = []
    for i in honest:
        tx = results[i]["tx_active"]
        for c in net.CLASSES:
            tr[c].append(tx[c][0] * 8 / 1000 / E)
            pk[c].append(tx[c][1] / E)
        envs.append(sum(tx[c][2] for c in net.CLASSES))
        totals.append(sum(tx[c][0] for c in net.CLASSES))
    m["kbit_s"] = {c: _r(statistics.mean(v), 2) for c, v in tr.items()}
    m["kbit_s_total_mean"] = _r(sum(statistics.mean(v) for v in tr.values()), 2)
    m["kbit_s_total_max"] = _r(max(
        sum(tr[c][j] for c in net.CLASSES) for j in range(len(honest))), 2)
    m["pkt_s"] = {c: _r(statistics.mean(v), 2) for c, v in pk.items()}
    sigs = sum(results[i]["tx_active"][c][3] for i in honest for c in net.CLASSES)
    m["sig_share"] = _r((64 if cfg["batch_ms"] else 96) * sigs / max(sum(totals), 1))
    m["sigs_per_s"] = _r(sigs / len(honest) / E, 2)
    m["envelopes_per_s"] = _r(statistics.mean(envs) / E, 2)
    vn = sum(results[i]["verify_n"] for i in range(n))
    vs = sum(results[i]["verify_s"] for i in range(n))
    m["verify_us"] = _r(1e6 * vs / max(vn, 1), 1)
    m["verifies_per_node_s"] = _r(vn / n / D, 1)
    m["cpu_ms_per_sim_s"] = _r(1000 * statistics.mean(
        results[i]["busy_s"] for i in range(n)) / D, 2)
    m["transport_timeouts"] = sum(r["transport_timeouts"] for r in results)
    m["lost_dgrams"] = sum(r["lost"] for r in results)
    m["rejects"] = {}
    for i in honest:
        for k, v in results[i]["rejects"].items():
            m["rejects"][k] = m["rejects"].get(k, 0) + v
    # v1.2: отказы честным авторам у честных узлов и запас лимита по seq
    hon = set(honest)
    m["honest_author_rejects"] = {}
    for i in honest:
        for k, v in results[i].get("rejects_by", {}).items():
            why, a = k.rsplit("@", 1)
            if int(a) in hon:
                m["honest_author_rejects"][why] = m["honest_author_rejects"].get(why, 0) + v
    leads = [v for i in honest for a, v in results[i].get("seq_lead", {}).items()
             if int(a) in hon]
    m["honest_seq_lead_max"] = max(leads) if leads else None
    m["deprecated_audit_mean"] = _r(statistics.mean(
        results[i]["deprecated"] for i in honest), 1)
    m["self_val_ignored"] = sum(results[i]["self_val_ignored"] for i in honest)
    m["validations_sent"] = sum(results[i]["validations_sent"] for i in honest)

    # --- конфликты честных: выбрала ли B новое значение
    pairs = {}
    for eid, (key, vh, l, c, node, gen, label, tag) in labels.items():
        if tag and label == "H":
            pairs.setdefault(tag[1], {})[tag[2]] = (l, c, node, eid)
    full = [p for p in pairs.values() if len(p) == 2]
    m["conflict_pairs"] = len(full)
    m["conflict_new_wins"] = _r(sum(1 for p in full if p["new"] > p["old"])
                                / max(len(full), 1))

    # --- ложные срабатывания карантина и доверие честных
    fp_tot = fp_bad = 0
    per_min = {}
    for i in honest:
        for (eid, author, arr, wait) in results[i]["fp_records"]:
            if author not in honest or labels.get(eid, (0,) * 7)[6] != "H":
                continue
            bad = wait < 0 or wait > sn.FP_WAIT_MS
            fp_tot += 1
            fp_bad += bad
            mm = per_min.setdefault(arr // 60_000, [0, 0])
            mm[0] += 1
            mm[1] += bad
    m["fp_shared"] = _r(fp_bad / max(fp_tot, 1))
    m["fp_shared_n"] = fp_tot
    m["fp_per_min"] = {str(k): _r(v[1] / max(v[0], 1)) for k, v in sorted(per_min.items())}
    hon_glyphs = sum(1 for v in labels.values() if v[6] == "H")
    m["fp_all"] = _r(fp_bad / max(hon_glyphs * (len(honest) - 1), 1))
    tmin, tmean = [], []
    for s in range(D):
        vals = [results[i]["trust_series"][s][a] for i in honest for a in honest
                if a != i and s < len(results[i]["trust_series"])]
        if vals:
            tmin.append(min(vals))
            tmean.append(statistics.mean(vals))
    m["trust_honest_min_per_min"] = [_r(min(tmin[j:j + 60]), 3)
                                     for j in range(0, len(tmin), 60)]
    m["trust_honest_mean_per_min"] = [_r(statistics.mean(tmean[j:j + 60]), 3)
                                      for j in range(0, len(tmean), 60)]
    if attacked:
        att = cfg["attacker"]
        m["trust_in_attacker_per_min"] = [
            _r(statistics.mean(results[i]["trust_series"][s][att] for i in honest), 3)
            for s in range(0, D, 60) if s < len(results[honest[0]]["trust_series"])]

    # --- v1.1: штраф обвинителю и задержка доставки
    sl = [(i, int(v), ms) for i in honest for v, ms in results[i]["slashed"].items()]
    m["slashed_honest"] = sum(1 for (_, v, _) in sl if v in honest)
    m["slashed_attacker_by"] = sum(1 for (_, v, _) in sl if v not in honest)
    m["sensor_flips"] = sum(results[i].get("sensor_flips", 0) for i in honest)
    m["thymus_blocked"] = sum(results[i].get("thymus_blocked", 0) for i in honest)
    # v1.4 шаг 0: совпадают ли множества голосов и решения о приёме у честных
    vs = [results[i].get("vstate") for i in honest]
    if all(v is not None for v in vs):
        common = set(vs[0])
        for v in vs[1:]:
            common |= set(v)
        same = diff = dis_same = dis_diff = 0
        for e in common:
            # сравниваем узлы, кроме автора (автор принимает своё всегда);
            # событие считаем, только если оно есть у всех остальных честных
            au = next(v[e][2] for v in vs if e in v)
            have = [v for i, v in zip(honest, vs) if i != au]
            if not all(e in v for v in have):
                continue
            digs = {v[e][0] for v in have}
            adm = {v[e][1] for v in have}
            if len(digs) == 1:
                same += 1
                dis_same += len(adm) > 1
            else:
                diff += 1
                dis_diff += len(adm) > 1
        m["vset_equal_frac"] = _r(same / max(same + diff, 1))
        m["admit_disagree_same_set"] = dis_same
        m["admit_disagree_diff_set"] = dis_diff
    m["replays"] = sum(results[i].get("replays", 0) for i in honest)
    # v1.4: недоставленные записи журнала честных авторов к концу прогона
    if all("held_n" in results[i] for i in honest):
        gaps = [results[j]["own_seq"] - results[i]["held_n"][j]
                for i in honest for j in honest if i != j]
        m["journal_gap_sum"] = sum(gaps)
        m["journal_gap_max"] = max(gaps) if gaps else 0
    m["strikes_on_honest"] = sum(k for i in honest for v, k in results[i].get("strikes", {}).items()
                                 if int(v) in honest)
    if attacked:
        ts = [ms - sn.ATTACK_START_MS for (_, v, ms) in sl if v == cfg["attacker"]]
        m["slash_after_attack_s"] = (None if not ts else
                                     _r(statistics.median(ts) / 1000, 2))
        m["slash_after_attack_max_s"] = None if not ts else _r(max(ts) / 1000, 2)
    gen = {eid: (g, node) for eid, (key, vh, l, c, node, g, label, tag)
           in labels.items() if label == "H" and node in honest}
    dl = []
    for i in honest:
        arr = results[i]["arrivals"]
        for eid, (g, node) in gen.items():
            if node != i and eid in arr:
                dl.append(max(0, arr[eid] * tick - g))
    dl.sort()
    m["delay_ms_p50"] = dl[len(dl) // 2] if dl else None
    m["delay_ms_p95"] = dl[int(len(dl) * 0.95)] if dl else None

    if attacked:
        m["attacks"] = attack_metrics(cfg, results, honest, keys, kidx, tick)
    return m


def attack_metrics(cfg, results, honest, keys, kidx, tick):
    log = results[cfg["attacker"]]["attack_log"]
    by = {}
    for (typ, eid, key, vh, t) in log:
        by.setdefault(typ, []).append((eid, key, vh, t))
    ever_eid, ever_kv = set(), set()
    win_ticks = {}
    for i in honest:
        for (tk, ki, eid, vh) in results[i]["changes"]:
            ever_eid.add(eid)
            ever_kv.add((ki, vh))
            win_ticks.setdefault((i, eid), []).append(tk)
    out = {}
    start = sn.ATTACK_START_MS
    for typ, items in sorted(by.items()):
        r = {"sent": len(items)}
        if typ in ("T1", "T2"):
            passed = sum(1 for (eid, key, vh, t) in items
                         if (kidx[key], vh) in ever_kv
                         or (typ == "T1" and eid in ever_eid))
        elif typ == "T3":
            passed = 0
            for (eid, key, vh, t) in items:
                rt = t // tick
                for i in honest:
                    arr = results[i]["arrivals"].get(eid)
                    if arr is not None and arr < rt and any(
                            tk >= rt for tk in win_ticks.get((i, eid), ())):
                        passed += 1
                        break
            r["note"] = ("passed = winner change to a replayed id after the "
                         "replay at a node that already had it")
        elif typ == "T7":
            passed = None
        else:
            ids = {eid for (eid, _, _, _) in items}
            passed = sum(1 for e in ids if e in ever_eid)
        r["passed"] = passed
        r["blocked_frac"] = None if passed is None else _r(1 - passed / max(len(items), 1))
        if typ == "T4":
            ids = {eid for (eid, _, _, _) in items}
            # v1.3: время вытеснения (TtE) - сколько секунд каждая ложь была
            # в картине большинства честных узлов
            delta = {}
            for i in honest:
                cur = {}
                for (tk, ki, eid, vh) in results[i]["changes"]:
                    old = cur.get(ki)
                    if old in ids:
                        delta.setdefault(old, {})
                        delta[old][tk] = delta[old].get(tk, 0) - 1
                    if eid in ids:
                        delta.setdefault(eid, {})
                        delta[eid][tk] = delta[eid].get(tk, 0) + 1
                    cur[ki] = eid
            end_tk = cfg["duration_s"] * 1000 // tick
            half = len(honest) / 2
            ttes = []
            for eid, d in delta.items():
                cnt, prev, maj = 0, None, 0
                for tk in sorted(d):
                    if prev is not None and cnt > half:
                        maj += tk - prev
                    cnt += d[tk]
                    prev = tk
                if prev is not None and cnt > half:
                    maj += end_tk - prev
                if maj:
                    ttes.append(maj * tick / 1000)
            r["lies_in_majority"] = len(ttes)
            r["tte_max_s"] = _r(max(ttes), 2) if ttes else 0.0
            r["tte_over_1_5s"] = sum(1 for x in ttes if x > 1.5)
            trans = []
            for i in honest:
                cur, cnt = {}, 0
                for (tk, ki, eid, vh) in results[i]["changes"]:
                    was = cur.get(ki) in ids
                    now = eid in ids
                    cur[ki] = eid
                    if was != now:
                        before = cnt > 0
                        cnt += 1 if now else -1
                        if before != (cnt > 0):
                            trans.append((tk, 1 if cnt > 0 else -1))
            trans.sort()
            live = peak = 0
            last = None
            for tk, d in trans:
                live += d
                peak = max(peak, live)
                if d < 0:
                    last = tk
            r["max_honest_nodes_with_lie"] = peak
            r["majority_reached"] = peak > len(honest) / 2
            r["still_in_picture_at_end"] = live > 0
            r["last_lie_exit_s_after_start"] = (None if last is None
                                                else _r((last * tick - start) / 1000, 1))
        if typ == "T5":
            ids = [eid for (eid, _, _, _) in items]
            r["stored_frac_at_honest"] = _r(statistics.mean(
                sum(1 for e in ids if e in results[i]["arrivals"]) / max(len(ids), 1)
                for i in honest))
            fr = [results[i]["first_rate_reject"].get(cfg["attacker"])
                  for i in honest]
            fr = [x for x in fr if x is not None]
            r["time_to_block_s"] = (None if not fr
                                    else _r((statistics.median(fr) - start) / 1000, 2))
        if typ == "T6":
            rv = [results[i]["revoked_recv_ms"] for i in honest]
            r["revocation_known_all_s"] = (None if None in rv else
                                           _r((max(rv) - cfg["revoke_at_ms"]) / 1000, 2))
            r["key_holder_in_sim"] = "node 0 (in reality: the operator)"
        if typ == "T7":
            r["refutations_sent"] = len(items)
        out[typ] = r
    return out


# -------------------------------------------------------------------- сетка

def grid(which):
    cfgs = []
    seeds = range(1, 11)
    if which in ("all", "conv"):
        # Сокращено, чтобы сетка уложилась примерно в час (замер: ~6.5 с на
        # прогон при 2 параллельных): ветка A идёт только на полной сетке и
        # только при потерях 0/5/20 %. Вырезано 90 прогонов A, см. отчёт.
        for branch in ("A", "B"):
            for topo in (("full", "chain") if branch == "B" else ("full",)):
                for loss in ((0.0, 0.05, 0.2, 0.4) if branch == "B"
                             else (0.0, 0.05, 0.2)):
                    for part in ("none", "isolate", "split"):
                        for s in seeds:
                            cfgs.append(dict(branch=branch, topology=topo,
                                             loss=loss, partition=part, seed=s))
    if which == "probe14":
        # v1.4: probe13 + худшие по сходимости конфигурации v1.3 (шум 3 %)
        for att in ("none", "T4", "T7"):
            for s in range(1, 6):
                cfgs.append(dict(branch="B", loss=0.05, attack=att, seed=s))
        for topo, loss, part in (("chain", 0.4, "isolate"), ("full", 0.2, "isolate"),
                                 ("chain", 0.2, "none"), ("full", 0.4, "none")):
            for s in range(1, 6):
                cfgs.append(dict(branch="B", topology=topo, loss=loss,
                                 partition=part, seed=s))
    if which == "probe13":
        # разведка v1.3: без атаки, T4, T7 x 5 зёрен + цепочка 40 % isolate x 5
        for att in ("none", "T4", "T7"):
            for s in range(1, 6):
                cfgs.append(dict(branch="B", loss=0.05, attack=att, seed=s))
        for s in range(1, 6):
            cfgs.append(dict(branch="B", topology="chain", loss=0.4,
                             partition="isolate", seed=s))
    if which == "check12":
        # короткая проверка раздельных номеров: цепочка 40 % и трафик без атак
        for part in ("none", "isolate", "split"):
            for s in seeds:
                cfgs.append(dict(branch="B", topology="chain", loss=0.4,
                                 partition=part, seed=s))
        for s in seeds:
            cfgs.append(dict(branch="B", loss=0.05, seed=s))
    if which in ("all", "attacks", "check12"):
        for att in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "all"):
            for s in seeds:
                cfgs.append(dict(branch="B", loss=0.05, attack=att, seed=s))
    if which in ("all", "sens"):
        for tinit in (1.0, 0.05):
            for att in ("none", "T4", "T7", "all"):
                for s in seeds:
                    cfgs.append(dict(branch="B", loss=0.05, attack=att, seed=s,
                                     trust_init=tinit))
    return cfgs


def run_grid(cfgs, out, workers, extra):
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add(_cfg_key(r))
    todo = [dict(c, **extra) for c in cfgs]
    todo = [c for c in todo if _cfg_key(dict(DEFAULTS, **c)) not in done]
    print("grid: %d runs to do, %d already in %s" % (len(todo), len(done), out),
          flush=True)
    ports = queue.Queue()
    for w in range(workers):
        ports.put(21000 + 100 * w)
    lock = threading.Lock()
    t0 = time.perf_counter()
    counter = [0]

    def one(c):
        port = ports.get()
        try:
            rec, _ = run_once(dict(c, base_port=port))
        except Exception as e:             # прогон упал - пишем как есть
            rec = {k: dict(DEFAULTS, **c)[k] for k in (
                "n", "seed", "branch", "loss", "topology", "partition", "attack",
                "attacker", "trust_init", "duration_s", "events_until_s",
                "tick_ms", "K", "batch_ms", "slash", "rate_mode")}
            if dict(DEFAULTS, **c)["sensor_noise"]:
                rec["sensor_noise"] = dict(DEFAULTS, **c)["sensor_noise"]
            rec["slash_window"] = dict(DEFAULTS, **c)["slash_window"]
            for q in ("refute_quorum", "confirm_quorum"):
                if dict(DEFAULTS, **c)[q] > 1:
                    rec[q] = dict(DEFAULTS, **c)[q]
            if dict(DEFAULTS, **c)["quorum_scope"] != "all":
                rec["quorum_scope"] = dict(DEFAULTS, **c)["quorum_scope"]
            if dict(DEFAULTS, **c)["admit_rule"] != "trust":
                rec["admit_rule"] = dict(DEFAULTS, **c)["admit_rule"]
            if dict(DEFAULTS, **c)["liar_k"]:
                rec["liar_k"] = dict(DEFAULTS, **c)["liar_k"]
            if dict(DEFAULTS, **c)["vote_order"] != "arrival":
                rec["vote_order"] = dict(DEFAULTS, **c)["vote_order"]
            if not dict(DEFAULTS, **c)["sensor_filter"]:
                rec["sensor_filter"] = False
            if dict(DEFAULTS, **c)["self_trust"] != "full":
                rec["self_trust"] = dict(DEFAULTS, **c)["self_trust"]
            if dict(DEFAULTS, **c)["pull_priority"] != "log":
                rec["pull_priority"] = dict(DEFAULTS, **c)["pull_priority"]
            if dict(DEFAULTS, **c)["pull_quiet_cap"]:
                rec["pull_quiet_cap"] = dict(DEFAULTS, **c)["pull_quiet_cap"]
            if dict(DEFAULTS, **c)["self_admit"] != "always":
                rec["self_admit"] = dict(DEFAULTS, **c)["self_admit"]
            if dict(DEFAULTS, **c)["arbiter"] != "scalar":
                rec["arbiter"] = dict(DEFAULTS, **c)["arbiter"]
            if dict(DEFAULTS, **c)["kinematic_gate"]:
                rec["kinematic_gate"] = dict(DEFAULTS, **c)["kinematic_gate"]
            if dict(DEFAULTS, **c)["apsa_rule"]:
                rec["apsa_rule"] = dict(DEFAULTS, **c)["apsa_rule"]
            if dict(DEFAULTS, **c)["liar_vote"]:
                rec["liar_vote"] = True
            if dict(DEFAULTS, **c)["arbiter"] != "scalar":
                rec["tomb_gap"] = dict(DEFAULTS, **c)["tomb_gap"]
            for q in ("confirm_cap", "refute_cap", "rebut"):
                if dict(DEFAULTS, **c)[q]:
                    rec[q] = dict(DEFAULTS, **c)[q]
            rec["error"] = repr(e)
        finally:
            ports.put(port)
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter[0] += 1
            el = time.perf_counter() - t0
            print("[%d/%d] %.0fs %s %s loss=%s %s %s seed=%s trust=%s %s" % (
                counter[0], len(todo), el, rec["branch"], rec["topology"],
                rec["loss"], rec["partition"], rec["attack"], rec["seed"],
                rec["trust_init"], rec.get("error", "")), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))


def _cfg_key(r):
    return tuple(r[k] for k in ("seed", "branch", "loss", "topology",
                                "partition", "attack", "trust_init",
                                "duration_s")) + (r.get("batch_ms", 0),
                                                  r.get("slash", False),
                                                  r.get("rate_mode", "hlc"),
                                                  r.get("sensor_noise", 0.0),
                                                  r.get("slash_window", 1),
                                                  r.get("refute_quorum", 1),
                                                  r.get("confirm_quorum", 1),
                                                  r.get("quorum_scope", "all"),
                                                  r.get("confirm_cap"),
                                                  r.get("refute_cap"),
                                                  r.get("rebut", False),
                                                  r.get("admit_rule", "trust"),
                                                  r.get("liar_k", 0),
                                                  r.get("vote_order", "arrival"),
                                                  r.get("sensor_filter", True),
                                                  r.get("self_trust", "full"),
                                                  r.get("pull_priority", "log"),
                                                  r.get("pull_quiet_cap", 0),
                                                  r.get("self_admit", "always"),
                                                  r.get("arbiter", "scalar"),
                                                  r.get("kinematic_gate", 0),
                                                  r.get("apsa_rule"),
                                                  r.get("tomb_gap", 20),
                                                  r.get("liar_vote", False))


# ------------------------------------------------------------------- отчёт

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _med(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else statistics.median(xs)


def _fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        return ("%." + str(nd) + "f") % x
    return str(x)


def report(path):
    rows = load(path)
    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print("# Отчёт по %s: %d прогонов, %d упали\n" % (path, len(rows) + len(errs), len(errs)))
    for e in errs:
        print("- упал:", {k: e[k] for k in ("branch", "topology", "loss",
                                             "partition", "attack", "seed")},
              e["error"])
    groups = {}
    for r in rows:
        k = (r["branch"], r["topology"], r["loss"], r["partition"], r["attack"],
             r["trust_init"], r["duration_s"])
        groups.setdefault(k, []).append(r)

    print("\n## Сходимость (без атак, доверие 0.5)\n")
    print("| ветка | топология | потери | разрыв | сошлось | время, с (медиана) "
          "| доля совпадающих слотов (мин) | расхождение во время разрыва "
          "| догон, с (медиана / макс) | = эталону | потеряно в сети "
          "| застряло в карантине | порядок / прочее |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(groups):
        b, topo, loss, part, att, ti, dur = k
        if att != "none" or ti != 0.5:
            continue
        g = [r["metrics"] for r in groups[k]]
        conv = sum(1 for x in g if x["converged"])
        cs = [x.get("catchup_s") for x in g]
        print("| %s | %s | %d %% | %s | %d/%d | %s | %s | %s | %s / %s | %s | %s | %s | %s |" % (
            b, topo, round(loss * 100), part, conv, len(g),
            _fmt(_med([x["converge_s_after_last_event"] for x in g]), 0),
            _fmt(min(x["final_agree_frac"] for x in g), 3),
            _fmt(_med([x.get("div_during_partition") for x in g]), 3),
            _fmt(_med(cs), 0),
            "не догнал" if None in cs and part != "none" else _fmt(max([c for c in cs if c is not None], default=None), 0),
            _fmt(statistics.mean(x["ref_match_frac"] for x in g), 3),
            _fmt(statistics.mean(x["ref_diff"]["lost"] for x in g), 4),
            _fmt(statistics.mean(x["ref_diff"]["quarantined"] for x in g), 4),
            _fmt(statistics.mean(x["ref_diff"]["order_or_other"] for x in g), 4)))

    print("\n## Трафик на честный узел, активное окно (без атак, без разрыва)\n")
    print("| ветка | топология | потери | PUSH кбит/с | PULL+digest кбит/с "
          "| служебное кбит/с | всего (ср / макс узел) | пакетов/с всего "
          "| конвертов/с | доля подписи | проверка подписи, мкс | CPU мс на сим-с "
          "| сжатие |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(groups):
        b, topo, loss, part, att, ti, dur = k
        if att != "none" or part != "none" or ti != 0.5:
            continue
        g = [r["metrics"] for r in groups[k]]
        mean = lambda f: statistics.mean(f(x) for x in g)  # noqa: E731
        print("| %s | %s | %d %% | %.1f | %.1f | %.1f | %.1f / %.1f | %.0f | %.1f | %.2f | %.0f | %.0f | %.0f× |" % (
            b, topo, round(loss * 100),
            mean(lambda x: x["kbit_s"]["push"]), mean(lambda x: x["kbit_s"]["pull"]),
            mean(lambda x: x["kbit_s"]["svc"]), mean(lambda x: x["kbit_s_total_mean"]),
            max(x["kbit_s_total_max"] for x in g),
            mean(lambda x: sum(x["pkt_s"].values())), mean(lambda x: x["envelopes_per_s"]),
            mean(lambda x: x["sig_share"]), mean(lambda x: x["verify_us"]),
            mean(lambda x: x["cpu_ms_per_sim_s"]), mean(lambda x: x["compression"])))

    print("\n## Атаки (ветка B, full mesh, потери 5 %, без разрыва)\n")
    print("| доверие | атака | тип | отправлено (ср) | заблокировано (мин по зёрнам) "
          "| прошло (сумма) | в картине большинства (зёрен) | время до блокировки, с "
          "| FP общих ключей | доверие к честным, мин |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(groups, key=lambda k: (-k[5], k[4])):
        b, topo, loss, part, att, ti, dur = k
        if att == "none" and ti == 0.5:
            continue
        g = [r["metrics"] for r in groups[k]]
        fp = _fmt(statistics.mean(x["fp_shared"] for x in g), 4)
        tmin = _fmt(min(min(x["trust_honest_min_per_min"]) for x in g), 3)
        if att == "none":
            print("| %s | none | - | - | - | - | - | - | %s | %s |" % (ti, fp, tmin))
            continue
        types = sorted({t for x in g for t in x.get("attacks", {})})
        for t in types:
            xs = [x["attacks"][t] for x in g if t in x.get("attacks", {})]
            blocked = [y["blocked_frac"] for y in xs if y["blocked_frac"] is not None]
            maj = sum(1 for y in xs if y.get("majority_reached"))
            ttb = None
            if t == "T4":
                ttb = _med([y.get("last_lie_exit_s_after_start") for y in xs])
            elif t == "T5":
                ttb = _med([y.get("time_to_block_s") for y in xs])
            elif t == "T6":
                ttb = _med([y.get("revocation_known_all_s") for y in xs])
            print("| %s | %s | %s | %.0f | %s | %s | %s | %s | %s | %s |" % (
                ti, att, t, statistics.mean(y["sent"] for y in xs),
                _fmt(min(blocked), 4) if blocked else "-",
                _fmt(sum(y["passed"] for y in xs if y["passed"] is not None)) if blocked else "-",
                ("%d/%d" % (maj, len(xs))) if t == "T4" else "-",
                _fmt(ttb, 1), fp, tmin))

    print("\n## Ложные срабатывания и доверие честных по минутам (B, full, 5 %, без разрыва)\n")
    for k in sorted(groups):
        b, topo, loss, part, att, ti, dur = k
        if not (b == "B" and topo == "full" and loss == 0.05 and part == "none"
                and att in ("none", "T4", "T7", "all")):
            continue
        g = [r["metrics"] for r in groups[k]]
        mins = sorted({int(mm) for x in g for mm in x["fp_per_min"]})
        fp = [statistics.mean(x["fp_per_min"].get(str(mm), 0) for x in g) for mm in mins]
        tm = [statistics.mean(x["trust_honest_mean_per_min"][j] for x in g)
              for j in range(len(g[0]["trust_honest_mean_per_min"]))]
        tn = [min(x["trust_honest_min_per_min"][j] for x in g)
              for j in range(len(g[0]["trust_honest_min_per_min"]))]
        print("- атака %s, доверие %s: FP по минутам %s" % (
            att, ti, " ".join("%d:%.3f" % (mm, v) for mm, v in zip(mins, fp))))
        print("  доверие честных ср %s; мин %s" % (
            " ".join("%.2f" % v for v in tm), " ".join("%.2f" % v for v in tn)))
        if att != "none":
            ta = [statistics.mean(x["trust_in_attacker_per_min"][j] for x in g)
                  for j in range(len(g[0]["trust_in_attacker_per_min"]))]
            print("  доверие честных к атакующему, срез в начале минуты %s" % (
                " ".join("%.2f" % v for v in ta)))

    print("\n## Честные конфликты: доля, где B выбрала новое значение\n")
    bs = [r["metrics"] for r in rows if r["branch"] == "B" and r["attack"] == "none"]
    if bs:
        print("пар на прогон %.0f, доля конфликтных событий %.3f, новое побеждает %.3f (мин %.3f)" % (
            statistics.mean(x["conflict_pairs"] for x in bs),
            statistics.mean(x["conflict_share"] for x in bs),
            statistics.mean(x["conflict_new_wins"] for x in bs),
            min(x["conflict_new_wins"] for x in bs)))


# --------------------------------------------------------------------- CLI

def _rule_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if "error" not in r]

    def sel(**kw):
        d = dict(branch="B", topology="full", loss=0.05, partition="none",
                 attack="none", trust_init=0.5)
        d.update(kw)
        return [r["metrics"] for r in rows if all(r[k] == v for k, v in d.items())]

    out = {}
    conv = [r["metrics"]["converged"] for r in rows
            if r["branch"] == "B" and r["attack"] == "none"
            and r["trust_init"] == 0.5 and r["loss"] <= 0.2]
    out["1. B, потери <= 20 %, оба разрыва: сошлось"] = "%d/%d" % (sum(conv), len(conv))
    for typ in ("T1", "T2", "T3", "T6"):
        g = sel(attack=typ)
        ga = sel(attack="all")
        out["2. %s заблокировано (мин; одна / all)" % typ] = "%s / %s" % (
            _fmt(min(x["attacks"][typ]["blocked_frac"] for x in g), 4),
            _fmt(min(x["attacks"][typ]["blocked_frac"] for x in ga), 4))
    for att in ("T4", "all"):
        g = sel(attack=att)
        out["3. T4 в картине большинства (%s), зёрен" % att] = "%d/%d" % (
            sum(x["attacks"]["T4"]["majority_reached"] for x in g), len(g))
    for att in ("none", "T7", "all"):
        g = sel(attack=att)
        out["4. FP общих ключей (%s)" % att] = _fmt(
            statistics.mean(x["fp_shared"] for x in g), 4)
    g = sel()
    out["5. трафик B full 5 %, кбит/с на узел"] = _fmt(
        statistics.mean(x["kbit_s_total_mean"] for x in g), 1)
    sh = [r["metrics"].get("slashed_honest") for r in rows]
    out["6. честных оштрафовано (все прогоны)"] = (
        "-" if all(x is None for x in sh) else str(sum(x or 0 for x in sh)))
    t7 = sel(attack="T7")
    out["T7: сошлось / = эталону"] = "%d/%d / %s" % (
        sum(x["converged"] for x in t7), len(t7),
        _fmt(statistics.mean(x["ref_match_frac"] for x in t7), 4))
    out["T7: доверие к честным, мин после 1-й минуты"] = _fmt(min(
        min(x["trust_honest_min_per_min"][1:]) for x in t7), 2)
    # v1.2: лимит частоты
    for topo in ("chain", "full"):
        for part in ("none", "isolate", "split"):
            g = sel(topology=topo, loss=0.4, partition=part)
            out["v1.2: %s, потери 40 %%, %s: сошлось" % (topo, part)] = "%d/%d" % (
                sum(x["converged"] for x in g), len(g))
    nh = [r for r in rows if r["attack"] == "none"]
    out["v1.2: отказы по лимиту у честных, без атак (сумма)"] = str(sum(
        r["metrics"]["rejects"].get("rate", 0) for r in nh))
    ha = [r["metrics"].get("honest_author_rejects") for r in rows]
    out["v1.2: отказы честным авторам, все прогоны (сумма)"] = (
        "-" if all(x is None for x in ha) else
        str(sum(sum((x or {}).values()) for x in ha)))
    t5 = sel(attack="T5")
    out["v1.2: T5 заблокировано / хранится у честных / до блокировки, с"] = \
        "%s / %s / %s" % (
            _fmt(min(x["attacks"]["T5"]["blocked_frac"] for x in t5), 4),
            _fmt(statistics.mean(x["attacks"]["T5"]["stored_frac_at_honest"]
                                 for x in t5), 4),
            _fmt(_med([x["attacks"]["T5"]["time_to_block_s"] for x in t5]), 2))
    t4 = sel(attack="T4")
    out["v1.3: T4 прошло лжи (сумма по зёрнам)"] = str(sum(
        x["attacks"]["T4"]["passed"] or 0 for x in t4))
    out["v1.2: T5 = эталону"] = _fmt(statistics.mean(
        x["ref_match_frac"] for x in t5), 4)
    dl = [x for x in sel() if x.get("delay_ms_p50") is not None]
    out["v1.2: задержка p50 / p95, мс (B full 5 %)"] = (
        "%s / %s" % (_fmt(statistics.mean(x["delay_ms_p50"] for x in dl), 0),
                     _fmt(statistics.mean(x["delay_ms_p95"] for x in dl), 0))
        if dl else "-")
    return len(rows), out


def compare(old, new):
    n_old, a = _rule_rows(old)
    n_new, b = _rule_rows(new)
    print("| пункт | %s (%d) | %s (%d) |" % (os.path.basename(old), n_old,
                                           os.path.basename(new), n_new))
    print("|---|---|---|")
    for k in a:
        print("| %s | %s | %s |" % (k, a[k], b.get(k, "-")))


def byte_breakdown():
    """Разбор байтов типичного glyph и validation (настоящие ключи и подпись).
    Поле = len(канон "имя":значение) плюс запятая; скобки и kind/gep - прочее."""
    k = sn.Keys(1, 0)
    g = sn.make_env(k, "glyph", {"key": "target_3.pos", "value": [41250, 18730],
                                 "hlc": [123456, 0], "node": 0, "seq": 4321})
    v = sn.make_env(k, "validation", {
        "glyph_hash": g["id"], "verdict": "confirmed",
        "evidence": {"applied_in": "sensor:target_3.pos", "note": None},
        "hlc": [123460, 1], "node": 0, "seq": 4322})
    for name, env in (("glyph", g), ("validation", v)):
        total = len(sn.canon(env))
        parts = {f: len(sn.canon({f: env[f]})) - 2 + 1
                 for f in ("sig", "author", "id", "body")}
        parts["прочее"] = total - sum(parts.values())
        print("%s: %d B  " % (name, total) + "  ".join(
            "%s %d" % kv for kv in parts.items()))
    # v1.1: бинарный пакет из 5 событий (типичный квант 500 мс при 10 ev/s)
    import swarm_wire as wire
    codec = wire.Codec(sn.keys_universe(5), [sn.Keys(1, i).author for i in range(5)])
    bodies = [("glyph", {"key": "target_3.pos", "value": [41250, 18730],
                         "hlc": [123456, 0], "node": 0, "seq": 4321}),
              ("glyph", {"key": "sector_7.status", "value": "hazard",
                         "hlc": [123561, 0], "node": 0, "seq": 4322}),
              ("glyph", {"key": "drone_0.pos", "value": [101200, 55310, 2400],
                         "hlc": [123660, 0], "node": 0, "seq": 4323}),
              ("glyph", {"key": "drone_0.battery", "value": 87,
                         "hlc": [123770, 0], "node": 0, "seq": 4324}),
              ("validation", {"glyph_hash": g["id"], "verdict": "confirmed",
                              "evidence": {"applied_in": "sensor:target_3.pos",
                                           "note": None},
                              "hlc": [123880, 1], "node": 0, "seq": 4325})]
    rec = codec.enc_group(0, 5, [123456, 0], b"\x00" * 64,
                          [(i, kd, b, None) for i, (kd, b) in enumerate(bodies)], True)
    ev_sizes = []
    for kd, b in bodies:
        buf = bytearray()
        codec.enc_event(buf, kd, b, 0, 123456)
        ev_sizes.append(len(buf))
    print("batch v1.1 (5 событий): %d B  заголовок+подпись %d  события %s" % (
        len(rec), len(rec) - sum(ev_sizes), " ".join(map(str, ev_sizes))))
    one = codec.enc_group(0, 5, [123456, 0], b"\x00" * 64,
                          [(0, "glyph", bodies[0][1], [b"\x00" * 32] * 3)], False)
    print("одно событие с путём Меркла (n=5): %d B" % len(one))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", default="B", choices=("A", "B", "C"))
    ap.add_argument("--loss", type=float, default=0.05)
    ap.add_argument("--topology", default="full", choices=("full", "chain"))
    ap.add_argument("--partition", default="none",
                    help="none | isolate | split | или спецификация swarm_net")
    ap.add_argument("--attack", default="none",
                    choices=("none", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "all"))
    ap.add_argument("--attacker", type=int, default=4)
    ap.add_argument("--trust-init", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=int, default=600, help="сим-секунд")
    ap.add_argument("--events-until", type=int, default=480, help="сим-секунд")
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--tick-ms", type=int, default=TICK_MS)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--grid", choices=("all", "conv", "attacks", "sens", "check12", "probe13",
                                       "probe14"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--report")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"),
                    help="правило успеха для двух файлов результатов рядом")
    ap.add_argument("--bytes", action="store_true",
                    help="разбор байтов конверта glyph и validation")
    ap.add_argument("--batch-ms", type=int, default=500,
                    help="квант пакета, мс; 0 = v1.0 (подпись на событие, JSON)")
    ap.add_argument("--slash", default="on", choices=("on", "off"),
                    help="штраф обвинителю (v1.1); off = политика v1.0")
    ap.add_argument("--rate", default="kseq", choices=("kseq", "seq", "hlc"),
                    help="лимит частоты: kseq - раздельные номера glyph/validation "
                         "(v1.2, по умолчанию) | seq - общий номер (v1.2, первая "
                         "сетка) | hlc - по секунде HLC (v1.0, v1.1)")
    ap.add_argument("--sensor-noise", type=float, default=0.0,
                    help="v1.3: доля измерений датчика с неверным вердиктом (0.01 = 1 %%)")
    ap.add_argument("--slash-window", type=int, default=3,
                    help="v1.3: штраф после K ложных опровержений (и >=25 %% проверенно-верных "
                         "из окна 200); 1 = мгновенный штраф v1.2")
    ap.add_argument("--refute-quorum", type=int, default=2,
                    help="v1.3: опровержение действует, когда событие опровергли N разных узлов")
    ap.add_argument("--confirm-quorum", type=int, default=1,
                    help="v1.3: подтверждение действует, когда событие подтвердили N разных узлов")
    ap.add_argument("--quorum-scope", default="all", choices=("all", "trust"),
                    help="v1.3: кворум держит вес события и доверие (all) или только доверие (trust)")
    ap.add_argument("--confirm-cap", type=float, default=None,
                    help="v1.3: потолок веса одного подтверждения (ниже порога приёма 1.0)")
    ap.add_argument("--refute-cap", type=float, default=None,
                    help="v1.3: потолок веса одного опровержения")
    ap.add_argument("--admit-rule", default="trust",
                    choices=("trust", "net-balance", "log-odds", "sensor-veto"),
                    help="v1.3: правило приёма - доверие Q2 или функция от множеств оценщиков")
    ap.add_argument("--liar-k", type=int, default=0,
                    help="v1.3: автор с K событиями, опровергнутыми >=2 узлами, теряет вес (0 = выкл)")
    ap.add_argument("--vote-order", default="arrival", choices=("arrival", "canonical"),
                    help="v1.4: применять голоса в порядке прихода или в каноничном (HLC, id)")
    ap.add_argument("--sensor-filter", default="on", choices=("on", "off"),
                    help="v1.4: off - собственный датчик судьи не фильтрует опровержения при приёме")
    ap.add_argument("--self-trust", default="full", choices=("full", "symmetric"),
                    help="v1.4: доверие к себе 1.0 (full) или считается как к другим (symmetric)")
    ap.add_argument("--pull-priority", default="log", choices=("log", "validations"),
                    help="v1.4: PULL по порядку журнала (log) или сначала недостающие голоса")
    ap.add_argument("--pull-quiet-cap", type=int, default=0,
                    help="v1.4: записей в ответе PULL, когда своих событий уже нет (0 = как всегда, 24)")
    ap.add_argument("--self-admit", default="always", choices=("always", "rule"),
                    help="v1.4: своё событие по общему ключу - принимать всегда или по общему правилу")
    ap.add_argument("--arbiter", default="scalar", choices=("scalar", "anis", "anis-literal"),
                    help="v2.0: scalar - доверие W (v1.4); anis - APSA + надгробия с поправками; "
                         "anis-literal - буквально по ТЗ ANIS v2.0")
    ap.add_argument("--kinematic-gate", type=int, default=0,
                    help="v2.0: скачок координаты цели больше N за одно изменение - аномалия автора (0 = выкл)")
    ap.add_argument("--apsa-rule", default=None, choices=("share", "tz"),
                    help="v2.0: правило INTENTIONAL - доля >= 25 %% (share, по умолчанию для anis) "
                         "или S >= 0.70 по ТЗ (tz, по умолчанию для anis-literal)")
    ap.add_argument("--tomb-gap", type=int, default=20,
                    help="v2.0: на сколько событий окно должно обновиться до второй фазы карантина")
    ap.add_argument("--liar-vote", action="store_true",
                    help="v2.0: хоронить автора, у которого >= 25 %% из 200 его общих событий опровергнуты кворумом 2")
    ap.add_argument("--rebut", action="store_true",
                    help="v1.3: отвечать подтверждением на опровержение того, что мой сенсор видит верным")
    ap.add_argument("--no-slot-digest", action="store_true",
                    help="PULL v1: digest без карты слотов (только журнал)")
    a = ap.parse_args()
    if a.bytes:
        byte_breakdown()
        return
    if a.compare:
        compare(*a.compare)
        return
    if a.report:
        report(a.report)
        return
    extra = dict(duration_s=a.duration, events_until_s=a.events_until)
    if a.no_slot_digest:
        extra["slot_digest"] = False
    extra["batch_ms"] = a.batch_ms
    extra["slash"] = a.slash == "on"
    extra["rate_mode"] = a.rate
    extra["sensor_noise"] = a.sensor_noise
    extra["slash_window"] = a.slash_window
    extra["refute_quorum"] = a.refute_quorum
    extra["confirm_quorum"] = a.confirm_quorum
    extra["quorum_scope"] = a.quorum_scope
    extra["confirm_cap"] = a.confirm_cap
    extra["refute_cap"] = a.refute_cap
    extra["rebut"] = a.rebut
    extra["admit_rule"] = a.admit_rule
    extra["liar_k"] = a.liar_k
    extra["vote_order"] = a.vote_order
    extra["sensor_filter"] = a.sensor_filter == "on"
    extra["self_trust"] = a.self_trust
    extra["pull_priority"] = a.pull_priority
    extra["pull_quiet_cap"] = a.pull_quiet_cap
    extra["self_admit"] = a.self_admit
    extra["arbiter"] = a.arbiter
    extra["kinematic_gate"] = a.kinematic_gate
    extra["apsa_rule"] = a.apsa_rule
    extra["tomb_gap"] = a.tomb_gap
    extra["liar_vote"] = a.liar_vote
    if a.grid:
        run_grid(grid(a.grid), a.out, a.workers, extra)
        return
    cfg = dict(branch=a.branch, loss=a.loss, topology=a.topology,
               partition=a.partition, attack=a.attack, attacker=a.attacker,
               trust_init=a.trust_init, seed=a.seed, n=a.nodes,
               tick_ms=a.tick_ms, K=a.K, **extra)
    rec, _ = run_once(cfg)
    m = dict(rec["metrics"])
    m.pop("div_per_s", None)
    print(json.dumps(dict(rec, metrics=m), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
