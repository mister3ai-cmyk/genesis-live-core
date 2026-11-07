#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Business Source License 1.1 - см. research/LICENSE
"""
research/swarm/swarm_node.py

Узел роя: подписанный журнал событий GEP v0.1, слоты, доверие у читателя,
карантин, отзыв ключа, gossip PUSH + digest PULL.

Что взято и откуда
  подпись      схема p2p_edge_signing.py (ngp-core): Ed25519 из
               cryptography, raw-ключи, json.dumps(sort_keys) для подписи.
               Модуль не импортируется: на уровне модуля он тянет argon2
               (нет в Windows Python) и SQLite. Здесь минимальный аналог:
               без SQLite, без PoW, без A2A-карточки; ключ узла выводится из
               зерна, чтобы прогон был воспроизводим.
  канонизация  json.dumps(sort_keys=True, separators=(",", ":"),
               ensure_ascii=False), без float в подписываемых полях. Это НЕ
               полный RFC 8785: нет правил JCS для чисел и экранирования, они
               здесь не нужны, потому что все числа целые.
  доверие      константы agent_reputation.py (ngp-core): старт
               0.5, +0.10 за подтверждение, -0.20 за противоречие, клип [0, 1].
               Вес события - политика RECOMMENDED из GEP Q2.
  слоты        перенос research/agent_memory_traffic.py Memory._find_slot /
               write на точные ключи: одно актуальное значение на слот,
               прежнее помечается отменённым (аудит), без эмбеддингов.
"""
import base64
import binascii
import hashlib
import heapq
import json
import random
import socket
import time
from bisect import bisect_left, bisect_right, insort

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import swarm_net as net
import swarm_wire as wire

GEP = "0.1"
STATUSES = ("clear", "obstacle", "hazard", "unknown")
N_SECTORS = 16
N_TARGETS = 4
ATTACK_START_MS = 60_000
T5_WINDOW_MS = 60_000           # флуд x50 идёт 60 с: дольше сетка не влезает в час
MAX_FUTURE_MS = 10_000          # HLC: событие из будущего дальше - отказ
FP_WAIT_MS = 5_000              # карантин дольше - ложное срабатывание
PULL_CAP = 24                   # событий в одном ответе на digest
BITMAP = 128                    # окно «что у меня есть» за непрерывным seq


# ---------------------------------------------------------------- ключи мира

def keys_universe(n):
    ks = []
    for i in range(n):
        ks += ["drone_%d.pos" % i, "drone_%d.battery" % i,
               "drone_%d.payload" % i]
    ks += ["sector_%d.status" % s for s in range(N_SECTORS)]
    ks += ["target_%d.pos" % k for k in range(N_TARGETS)]
    return ks


def shared_keys():
    return (["sector_%d.status" % s for s in range(N_SECTORS)]
            + ["target_%d.pos" % k for k in range(N_TARGETS)])


def is_shared(key):
    return key.startswith("sector_") or key.startswith("target_")


def owner_of(key):
    if key.startswith("drone_"):
        return int(key[6:key.index(".")])
    return None


# ------------------------------------------------------------ канонизация

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def assert_no_float(obj):
    if isinstance(obj, float):
        raise TypeError("float in signed field: %r" % obj)
    if isinstance(obj, dict):
        for v in obj.values():
            assert_no_float(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            assert_no_float(v)


def sha(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def vhash(value):
    return hashlib.sha256(canon(value)).hexdigest()[:8]


def b64(b):
    return base64.b64encode(b).decode("ascii")


# ------------------------------------------------------------------ подписи

def revoke_statement(author):
    return {"action": "revoke", "key": author, "reason": "compromised"}


class Keys:
    """Ed25519-пара узла. Ключ выводится из (зерно, номер): иначе прогон
    нельзя повторить побайтно. Сертификат отзыва подписывается сразу при
    генерации (GEP Q3 п.1) и хранится отдельно от узла."""

    def __init__(self, seed, idx):
        raw = hashlib.sha256(("swarm-key:%s:%d" % (seed, idx)).encode()).digest()
        self._sk = ed25519.Ed25519PrivateKey.from_private_bytes(raw)
        pub = self._sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.pub = ed25519.Ed25519PublicKey.from_public_bytes(pub)
        self.author = "ed25519:" + pub.hex()
        self.revoke_cert = b64(self.sign(canon(revoke_statement(self.author))))

    def sign(self, msg):
        return self._sk.sign(msg)


VERIFY = {"n": 0, "s": 0.0}     # на процесс: число проверок и их время


def verify_sig(pub, sig_b64, msg):
    t0 = time.perf_counter()
    try:
        pub.verify(base64.b64decode(sig_b64, validate=True), msg)
        ok = True
    except (InvalidSignature, ValueError, binascii.Error, TypeError):
        ok = False
    VERIFY["n"] += 1
    VERIFY["s"] += time.perf_counter() - t0
    return ok


def env_core(author, kind, body):
    return {"gep": GEP, "kind": kind, "author": author, "body": body}


def make_env(keys, kind, body, author=None):
    """Конверт GEP. id - sha256 от канона без id и sig, sig - Ed25519 по тому
    же канону. author можно подменить (атака T1): подпись всё равно ставит
    keys, и проверка её отвергнет."""
    assert_no_float(body)
    core = env_core(author or keys.author, kind, body)
    c = canon(core)
    return dict(core, id=sha(c), sig=b64(keys.sign(c)))


def check_env(env, pubs):
    """None, если конверт цел и подписан своим автором, иначе причина."""
    try:
        core = env_core(env["author"], env["kind"], env["body"])
        if env["gep"] != GEP:
            return "schema"
        eid, sig = env["id"], env["sig"]
    except (KeyError, TypeError):
        return "schema"
    c = canon(core)
    if sha(c) != eid:
        return "hash"
    pub = pubs.get(core["author"])
    if pub is None:
        return "unknown_key"
    return None if verify_sig(pub, sig, c) else "sig"


def verify_raw(pub, sig, msg):
    """Проверка сырой 64-байтной подписи (v1.1: пакеты и digest)."""
    t0 = time.perf_counter()
    try:
        pub.verify(sig, msg)
        ok = True
    except (InvalidSignature, ValueError, TypeError):
        ok = False
    VERIFY["n"] += 1
    VERIFY["s"] += time.perf_counter() - t0
    return ok


def unsigned_env(author, kind, body):
    """v1.1: событие без собственной подписи, его подписывает корень пакета."""
    assert_no_float(body)
    core = env_core(author, kind, body)
    return dict(core, id=sha(canon(core)))


# ---------------------------------------------------------------------- HLC

class HLC:
    """Гибридные логические часы (Kulkarni et al. 2014). Физическая часть -
    симулированное время плюс дрейф узла; реальные часы не используются."""

    def __init__(self, drift_ms):
        self.drift = drift_ms
        self.l = 0
        self.c = 0

    def pt(self, sim_ms):
        return sim_ms + self.drift

    def now(self, sim_ms):
        pt = self.pt(sim_ms)
        if pt > self.l:
            self.l, self.c = pt, 0
        else:
            self.c += 1
        return [self.l, self.c]

    def recv(self, sim_ms, h):
        pt = self.pt(sim_ms)
        lm, cm = h
        l = max(self.l, lm, pt)
        if l == self.l and l == lm:
            c = max(self.c, cm) + 1
        elif l == self.l:
            c = self.c + 1
        elif l == lm:
            c = cm + 1
        else:
            c = 0
        self.l, self.c = l, c


# -------------------------------------------------------------------- слоты

class SlotTable:
    """Слоты Memory на точных ключах «сущность.атрибут».

    В Memory запись в слот гасит прежнее значение (deprecated), а соседние
    атрибуты той же сущности не трогает. Здесь то же самое, но без сведения
    имён по эмбеддингу: в рое ключи структурированные. Отличие веток - только
    в том, что считать «последним»:
      A  последнее пришедшее на этот узел (как Memory.write)
      B  максимум по (HLC, node_id); пришедшее позже, но более старое,
         ложится сразу отменённым
      C  без слотов: хранится всё, картина - последнее по локальному приёму.
         По построению совпадает с A; аудита отменённых нет.
    admit - предикат карантина: недопущенные события лежат, но не побеждают.
    """

    def __init__(self, branch):
        if branch not in ("A", "B", "C"):
            raise ValueError(branch)
        self.branch = branch
        self.slots = {}
        self.n = 0

    def add(self, key, eid, value, hlc=None, node=None):
        self.n += 1
        if self.branch == "B":
            order = (hlc[0], hlc[1], node, eid)
        else:
            order = (self.n,)
        lst = self.slots.setdefault(key, [])
        if self.branch == "B":
            insort(lst, (order, eid, value))
        else:
            lst.append((order, eid, value))

    def winner(self, key, admit=None):
        for _, eid, value in reversed(self.slots.get(key, ())):
            if admit is None or admit(eid):
                return eid, value
        return None

    def picture(self, admit=None):
        out = {}
        for k in sorted(self.slots):
            w = self.winner(k, admit)
            if w is not None:
                out[k] = w
        return out

    def values(self, admit=None):
        return {k: v for k, (_, v) in self.picture(admit).items()}

    def deprecated(self, admit=None):
        if self.branch == "C":
            return set()
        win = {eid for eid, _ in self.picture(admit).values()}
        return {eid for lst in self.slots.values() for _, eid, _ in lst
                if eid not in win}

    def world_hash(self, admit=None):
        pic = {k: eid for k, (eid, _) in self.picture(admit).items()}
        return hashlib.sha256(canon(pic)).hexdigest()


# ------------------------------------------------------------------- сенсор

class Sensor:
    """Собственное наблюдение узла: истина мира с покрытием ~60 % ключей на
    эпоху 30 с. Значение подтверждается, если было истинным где-то в окне
    [hlc - 5 с, min(hlc + 5 с, сейчас)]: окно шире дрейфа часов (2 с) и
    задержки, поэтому оба значения честного конфликта при смене истины
    проходят."""

    COVER = 0.6
    EPOCH_MS = 30_000
    WINDOW_MS = 5_000
    POS_TOL = 300

    def __init__(self, truth, seed, idx):
        self.truth, self.seed, self.idx = truth, seed, idx

    def covers(self, key, now_ms):
        h = hashlib.sha256(("cov:%s:%d:%s:%d" % (
            self.seed, self.idx, key, now_ms // self.EPOCH_MS)).encode()).digest()
        return int.from_bytes(h[:4], "big") < self.COVER * 2 ** 32

    def values_in(self, key, lo, hi):
        times, vals = self.truth[key]
        i = max(bisect_right(times, lo) - 1, 0)
        j = bisect_right(times, max(hi, lo))
        return vals[i:max(j, i + 1)]

    def check(self, key, value, hlc_l, now_ms):
        cand = self.values_in(key, hlc_l - self.WINDOW_MS,
                              min(hlc_l + self.WINDOW_MS, now_ms))
        if key.endswith(".status"):
            return value in cand
        try:
            return any(abs(value[0] - v[0]) <= self.POS_TOL
                       and abs(value[1] - v[1]) <= self.POS_TOL for v in cand)
        except (TypeError, IndexError):
            return False


def truth_at(truth, key, t_ms):
    times, vals = truth[key]
    return vals[max(bisect_right(times, t_ms) - 1, 0)]


# --------------------------------------------------------------------- узел

class Ev:
    __slots__ = ("eid", "kind", "author", "key", "value", "hlc", "seq",
                 "arrival", "env", "vh", "raw", "bref")

    def __init__(self, eid, kind, author, body, arrival, env):
        self.eid, self.kind, self.author = eid, kind, author
        self.key = body.get("key")
        self.value = body.get("value")
        self.hlc = body.get("hlc")
        self.seq = body.get("seq")
        self.arrival = arrival
        self.env = env
        self.vh = vhash(self.value) if kind == "glyph" else ""
        self.raw = None
        self.bref = None            # v1.1: ((автор, корень), индекс листа)

    def bytes(self):
        if self.raw is None:
            self.raw = canon(self.env)
        return self.raw


class Node:
    CONFIRM = 0.10              # agent_reputation.CONFIRM_DELTA
    REFUTE = -0.20              # agent_reputation.CONTRADICT_DELTA
    THRESH = 1.0                # GEP Q2: hypothesis >= 1 эфф. веса
    EVIDENCE = 2.0              # GEP Q2: evidence_bonus при непустом applied_in
    RATE = {"glyph": 20, "validation": 40}   # на ключ автора в секунду HLC
    # v1.2: лимит по seq автора - seq <= SEQ_RATE*(сек HLC + 1) + SEQ_BURST.
    # Решение записано в подписанном событии и не зависит от порядка прихода
    # и от того, сколько событий легло в одну секунду HLC при отставших часах.
    SEQ_RATE = 20                            # 2x честного темпа (10 событий/с)
    SEQ_BURST = 200

    def __init__(self, cfg, idx):
        self.cfg, self.me, self.n = cfg, idx, cfg["n"]
        seed = cfg["seed"]
        self.keys = [Keys(seed, i) for i in range(self.n)]
        self.k = self.keys[idx]
        self.pubs = {k.author: k.pub for k in self.keys}
        self.idx_of = {k.author: i for i, k in enumerate(self.keys)}
        self.rng = random.Random("node:%s:%d" % (seed, idx))
        drift = random.Random("drift:%s:%d" % (seed, idx)).randint(-2000, 2000)
        if cfg.get("drifts") is not None:
            drift = cfg["drifts"][idx]
        self.hlc = HLC(drift)
        self.tick_ms = cfg["tick_ms"]
        self.table = SlotTable(cfg["branch"])
        self.keys_list = keys_universe(self.n)
        self.events = {}
        self.held = [[] for _ in range(self.n)]     # отсортированные seq
        self.seen_seq = [set() for _ in range(self.n)]
        self.contig = [0] * self.n
        self.by_seq = {}
        self.seq = 0
        self.trust = [float(cfg["trust_init"])] * self.n
        self.trust[idx] = 1.0
        self.vsum = {}
        self.vseen = set()
        self.vpending = {}
        self.revoked = {}               # автор -> (effective_from, recv_ms)
        self.key_events = []
        self.rate = {}
        self.rate_mode = cfg.get("rate_mode", "hlc")    # "seq" = v1.2, "kseq"
        self.kseq = {"glyph": 0, "validation": 0}   # свои номера по видам
        self.by_kseq = set()                        # (автор, вид, kseq)
        self.seq_lead = {}      # автор -> max(seq - SEQ_RATE*(сек HLC + 1))
        self.sensor = Sensor(cfg["truth"], seed, idx)
        self.truth = cfg["truth"]
        self.nbrs = net.neighbors(cfg["topology"], self.n, idx)
        attack = cfg.get("attack") or "none"
        self.attacks = (set("T%d" % i for i in range(1, 8)) if attack == "all"
                        else (set() if attack == "none" else {attack}))
        self.attacker = cfg.get("attacker", 4)
        self.is_attacker = bool(self.attacks) and idx == self.attacker
        self.revoke_at = cfg.get("revoke_at_ms", 240_000)
        self.events_until = cfg["events_until_ms"]
        # выход
        self.fresh = {"push": [], "svc": [], "all": []}
        self.changes = []
        self.last_win = {}
        self.win_full = {}
        self.trust_series = []
        self.fp_pending = {}
        self.fp_records = []
        self.own = []
        self.attack_log = []
        self.rejects = {}
        self.rejects_by = {}            # "причина@автор" -> число
        self.dups = 0
        self.self_val_ignored = 0
        self.first_rate_reject = {}
        self.revoked_at_recv = None
        self.validations_sent = 0
        self.replay_pool = []
        self.t2_budget = 0
        self.lie_rng = random.Random("lie:%s" % seed)
        self.marker = 0
        # v1.1: пакеты (batch_ms = 0 - поведение v1.0) и штраф обвинителю
        self.batch_ms = int(cfg.get("batch_ms", 0) or 0)
        self.bq = max(1, self.batch_ms // self.tick_ms) if self.batch_ms else 0
        self.slash = bool(cfg.get("slash", False))
        self.codec = (wire.Codec(self.keys_list, [k.author for k in self.keys])
                      if self.bq else None)
        self.outbox = []            # свои события до закрытия пакета
        self.batches = {}           # (автор, корень) -> заголовок, листья, пути
        self.verified = set()       # проверенные корни: подпись - раз на пакет
        self.raw_out = []           # готовые записи атакующего
        self.slashed = {}           # обвинитель -> момент штрафа
        self.vlog = []              # применённые validation, для пересчёта
        self.clock = 0
        self.batches_made = 0

    # ----------------------------------------------------------- доверие

    def admitted(self, eid):
        ev = self.events[eid]
        a = ev.author
        r = self.revoked.get(a)
        if r is not None and ev.hlc[0] >= r[0]:
            return False
        if owner_of(ev.key) is not None or a == self.me:
            return True                 # правило владельца: хватает подписи
        return self.trust[a] + self.vsum.get(eid, 0.0) >= self.THRESH - 1e-9

    def apply_validation(self, vev):
        body = vev.env["body"]
        target = self.events.get(body.get("glyph_hash"))
        if target is None:
            self.vpending.setdefault(body.get("glyph_hash"), []).append(vev)
            return
        if target.kind != "glyph":
            return
        v, subj = vev.author, target.author
        if v == subj:               # автор не подтверждает сам себя
            self.self_val_ignored += 1
            return
        if (v, target.eid) in self.vseen:
            return
        self.vseen.add((v, target.eid))
        verdict = body.get("verdict")
        if verdict not in ("confirmed", "refuted"):
            return
        if self.slash and v in self.slashed:
            return
        if (self.slash and verdict == "refuted" and v != self.me
                and not self.is_attacker
                and self.sensor.covers(target.key, self.clock)
                and self.sensor.check(target.key, target.value, target.hlc[0],
                                      self.clock)):
            # Accuser slashing (v1.1): мой собственный сенсор видит, что
            # опровергнутое значение было правдой. Обвинитель лжёт - доверие
            # к нему 0 навсегда, его прошлые оценки отменяются пересчётом.
            self.slashed[v] = self.clock
            self.recompute_trust()
            return
        ev_in = (body.get("evidence") or {}).get("applied_in")
        bonus = self.EVIDENCE if ev_in else 1.0
        self.vlog.append((v, target.eid, subj, verdict, bonus))
        self._apply(v, target.eid, subj, verdict, bonus)

    def _apply(self, v, eid, subj, verdict, bonus):
        w = self.trust[v] * bonus
        if verdict == "confirmed":
            self.vsum[eid] = self.vsum.get(eid, 0.0) + w
            d = self.CONFIRM * w
        else:
            self.vsum[eid] = self.vsum.get(eid, 0.0) - w
            d = self.REFUTE * w
        if subj != self.me and subj not in self.slashed:
            self.trust[subj] = min(1.0, max(0.0, self.trust[subj] + d))

    def recompute_trust(self):
        """Доверие и веса так, будто оштрафованные обвинители не говорили
        ничего: тот же порядок validation, те же правила, без их голосов."""
        self.trust = [float(self.cfg["trust_init"])] * self.n
        self.trust[self.me] = 1.0
        for s in self.slashed:
            self.trust[s] = 0.0
        self.vsum = {}
        keep = []
        for rec in self.vlog:
            if rec[0] in self.slashed:
                continue
            keep.append(rec)
            self._apply(*rec)
        self.vlog = keep

    # ------------------------------------------------------------ журнал

    def next_seq(self):
        self.seq += 1
        return self.seq

    def seqs(self, kind):
        """Номера для тела: журнальный seq и, в режиме kseq, номер по виду."""
        d = {"seq": self.next_seq()}
        if self.rate_mode == "kseq":
            self.kseq[kind] += 1
            d["kseq"] = self.kseq[kind]
        return d

    def mark_seq(self, a, s):
        if not isinstance(s, int) or s <= 0 or s in self.seen_seq[a] \
                or s <= self.contig[a]:
            return
        self.seen_seq[a].add(s)
        while self.contig[a] + 1 in self.seen_seq[a]:
            self.contig[a] += 1
            self.seen_seq[a].discard(self.contig[a])

    def store(self, env, a, now_ms):
        ev = Ev(env["id"], env["kind"], a, env["body"], now_ms, env)
        self.events[ev.eid] = ev
        if ev.seq is not None:
            self.by_seq[(a, ev.seq)] = ev.eid
            insort(self.held[a], ev.seq)
            self.mark_seq(a, ev.seq)
        if ev.kind == "glyph":
            self.table.add(ev.key, ev.eid, ev.value, ev.hlc, a)
            for vev in self.vpending.pop(ev.eid, ()):
                self.apply_validation(vev)
        elif ev.kind == "validation":
            self.apply_validation(ev)
        return ev

    def reject(self, reason, a=None, seq=None, now_ms=0):
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        if a is not None:
            k = "%s@%d" % (reason, a)
            self.rejects_by[k] = self.rejects_by.get(k, 0) + 1
        if a is not None and seq is not None:
            self.mark_seq(a, seq)       # видели и отвергли: не дозапрашивать
        if reason == "rate" and a is not None:
            self.first_rate_reject.setdefault(a, now_ms)

    def accept(self, env, now_ms, bref=None):
        """Входящий конверт. Возвращает Ev, если он новый и принят.
        bref задан (v1.1) - конверт пришёл в пакете, чья подпись над корнем
        Меркла уже проверена в on_group, а id пересчитан из тела."""
        self.clock = now_ms
        eid = env.get("id") if isinstance(env, dict) else None
        if eid is None:
            self.reject("schema")
            return None
        if eid in self.events:
            self.dups += 1
            return None
        why = None if bref is not None else check_env(env, self.pubs)
        if why:
            self.reject(why)
            return None
        a = self.idx_of[env["author"]]
        body, kind = env["body"], env["kind"]
        if kind == "key_event":
            return self.accept_key_event(env, a, now_ms)
        if kind not in ("glyph", "validation") or not isinstance(body, dict):
            self.reject("schema")
            return None
        hlc, seq = body.get("hlc"), body.get("seq")
        if (not isinstance(hlc, list) or len(hlc) != 2
                or body.get("node") != a):
            self.reject("schema")
            return None
        r = self.revoked.get(a)
        if r is not None:
            # после отзыва: новое по дате - refuted, «задним числом» без
            # локального якоря - тоже не принимаем (GEP Q3 п.4)
            self.reject("revoked" if hlc[0] >= r[0] else "unanchored",
                        a, seq, now_ms)
            return None
        if hlc[0] > self.hlc.pt(now_ms) + MAX_FUTURE_MS:
            self.reject("future", a, seq, now_ms)
            return None
        if kind == "glyph":
            own = owner_of(body.get("key", ""))
            if own is not None and own != a:
                self.reject("owner", a, seq, now_ms)
                return None
        if self.rate_mode == "kseq":
            # v1.2 + раздельные номера: лимит по номеру события своего вида,
            # бюджеты v1.1 (20 glyph, 40 validation в секунду HLC)
            ks = body.get("kseq")
            if (not isinstance(seq, int) or not isinstance(ks, int)
                    or not isinstance(hlc[0], int) or not 1 <= ks <= seq):
                self.reject("schema")
                return None
            if (a, seq) in self.by_seq or (a, kind, ks) in self.by_kseq:
                self.reject("equivocation", a, seq, now_ms)
                return None
            lead = ks - self.RATE[kind] * (hlc[0] // 1000 + 1)
            if lead > self.SEQ_BURST:
                self.reject("rate", a, seq, now_ms)
                return None
            self.by_kseq.add((a, kind, ks))
            if lead > self.seq_lead.get(a, -10**12):
                self.seq_lead[a] = lead
        elif self.rate_mode == "seq":
            if not isinstance(seq, int) or not isinstance(hlc[0], int):
                self.reject("schema")
                return None
            if (a, seq) in self.by_seq:     # второе событие под тем же seq
                self.reject("equivocation", a, seq, now_ms)
                return None
            lead = seq - self.SEQ_RATE * (hlc[0] // 1000 + 1)
            if lead > self.SEQ_BURST:
                # навсегда: seq и hlc подписаны, повторный запрос даст то же
                self.reject("rate", a, seq, now_ms)
                return None
            if lead > self.seq_lead.get(a, -10**12):
                self.seq_lead[a] = lead
        else:
            bucket = (a, kind, hlc[0] // 1000)
            cnt = self.rate.get(bucket, 0)
            if cnt >= self.RATE[kind]:
                self.reject("rate", a, seq, now_ms)
                return None
            self.rate[bucket] = cnt + 1
        self.hlc.recv(now_ms, hlc)
        ev = self.store(env, a, now_ms)
        ev.bref = bref
        if kind == "glyph":
            self.fresh["push"].append(ev)
            if is_shared(ev.key) and a != self.me:
                self.on_shared_glyph(ev, now_ms)
        else:
            self.fresh["svc"].append(ev)
        return ev

    def accept_key_event(self, env, a, now_ms):
        body = env["body"]
        try:
            target = self.idx_of[body["key"]]
            stmt = canon(revoke_statement(body["key"]))
            ok = (body["action"] == "revoke"
                  and verify_sig(self.keys[target].pub, body["proof"], stmt))
            eff = int(body["effective_from_ms"])
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            self.reject("key_event")
            return None
        ev = Ev(env["id"], "key_event", a, body, now_ms, env)
        self.events[ev.eid] = ev
        if target not in self.revoked:
            self.revoked[target] = (eff, now_ms)
            if self.revoked_at_recv is None:
                self.revoked_at_recv = now_ms
        self.key_events.append(ev)
        self.fresh["all"].append(ev)
        return ev

    # -------------------------------------------------------- валидации

    def on_shared_glyph(self, ev, now_ms):
        if self.is_attacker:
            if "T7" in self.attacks and now_ms >= ATTACK_START_MS:
                self.emit_validation(ev, "refuted", now_ms, label="T7")
            if "T2" in self.attacks:
                self.maybe_tamper(ev, now_ms)
            if "T3" in self.attacks:
                self.replay_pool.append(ev)
            return
        # Подтверждаем только то, что в карантине (иначе трафик удвоится),
        # опровергаем всегда: ложь, уже вошедшую в картину, надо выбить.
        if self.sensor.covers(ev.key, now_ms):
            ok = self.sensor.check(ev.key, ev.value, ev.hlc[0], now_ms)
            if ok and not self.admitted(ev.eid):
                self.emit_validation(ev, "confirmed", now_ms)
            elif not ok:
                self.emit_validation(ev, "refuted", now_ms)
        if self.admitted(ev.eid):
            self.fp_records.append((ev.eid[7:23], ev.author, now_ms, 0))
        else:
            self.fp_pending[ev.eid] = now_ms

    def emit_validation(self, ev, verdict, now_ms, label=None):
        body = {"glyph_hash": ev.eid, "verdict": verdict,
                "evidence": {"applied_in": "sensor:" + ev.key, "note": None},
                "hlc": self.hlc.now(now_ms), "node": self.me,
                **self.seqs("validation")}
        self.clock = now_ms
        env = self.new_env("validation", body)
        vev = self.store(env, self.me, now_ms)
        self.emit(vev, "svc")
        self.validations_sent += 1
        if label:
            self.attack_log.append((label, vev.eid[7:23], ev.key, "", now_ms))

    def check_pending(self, now_ms):
        done = []
        for eid, arr in self.fp_pending.items():
            if self.admitted(eid):
                self.fp_records.append((eid[7:23], self.events[eid].author,
                                        arr, now_ms - arr))
                done.append(eid)
            elif now_ms - arr > FP_WAIT_MS:
                self.fp_records.append((eid[7:23], self.events[eid].author,
                                        arr, -1))
                done.append(eid)
        for eid in done:
            del self.fp_pending[eid]

    # ------------------------------------------------------- генерация

    def gen_own(self, t_ms, key, value, tag, label="H"):
        body = {"key": key, "value": value, "hlc": self.hlc.now(t_ms),
                "node": self.me, **self.seqs("glyph")}
        self.clock = t_ms
        env = self.new_env("glyph", body)
        ev = self.store(env, self.me, t_ms)
        self.emit(ev, "push")
        self.own.append((ev.eid[7:23], key, ev.vh, ev.hlc[0], ev.hlc[1],
                         t_ms, label, tag))
        return ev

    def lie(self, key, t_ms):
        if key.endswith(".status"):
            seen = set(Sensor(self.truth, 0, 0).values_in(
                key, t_ms - 10_000, t_ms + 10_000))
            opts = [s for s in STATUSES if s not in seen]
            if opts:
                return self.lie_rng.choice(opts)
            key = "target_%d.pos" % self.lie_rng.randrange(N_TARGETS)
        tv = truth_at(self.truth, key, t_ms)
        dx = self.lie_rng.choice((-1, 1)) * self.lie_rng.randint(3000, 5000)
        dy = self.lie_rng.choice((-1, 1)) * self.lie_rng.randint(3000, 5000)
        return key, [tv[0] + dx, tv[1] + dy]

    def generate(self, t_ms, key, value, tag):
        a = self.attacks if self.is_attacker else ()
        if "T6" in a and t_ms >= self.revoke_at:
            ev = self.gen_own(t_ms, key, value, tag, label="T6")
            self.attack_log.append(("T6", ev.eid[7:23], key, ev.vh, t_ms))
            return
        if "T4" in a and t_ms >= ATTACK_START_MS and is_shared(key):
            r = self.lie(key, t_ms)
            if isinstance(r, tuple):
                key, value = r
            else:
                value = r
            ev = self.gen_own(t_ms, key, value, None, label="T4")
            self.attack_log.append(("T4", ev.eid[7:23], key, ev.vh, t_ms))
            return
        self.gen_own(t_ms, key, value, tag)

    def next_marker(self):
        self.marker += 1
        return [9_000_000 + self.marker, 9_000_000 + self.me]

    def attack_tick(self, tick, t_ms):
        a = self.attacks
        if t_ms < ATTACK_START_MS:
            return
        if "T5" in a and t_ms < ATTACK_START_MS + T5_WINDOW_MS:
            for j in range(49):
                key = "drone_%d.%s" % (self.me, ("pos", "battery", "payload")[j % 3])
                val = ([self.rng.randint(0, 100_000), self.rng.randint(0, 100_000),
                        self.rng.randint(0, 5000)] if key.endswith("pos")
                       else self.rng.randint(0, 100))
                ev = self.gen_own(t_ms + j % self.tick_ms, key, val, None,
                                  label="T5")
                self.attack_log.append(("T5", ev.eid[7:23], key, ev.vh, t_ms))
        if "T6" in a and t_ms >= self.revoke_at and t_ms < self.events_until \
                and tick % 5 == 0:
            r = self.revoked_eff()
            key = "target_%d.pos" % self.rng.randrange(N_TARGETS)
            body = {"key": key, "value": self.next_marker(),
                    "hlc": [r - self.rng.randint(1000, 60_000), 0],
                    "node": self.me, **self.seqs("glyph")}
            env = self.new_env("glyph", body)
            ev = self.store(env, self.me, t_ms)
            self.emit(ev, "push")
            self.attack_log.append(("T6", ev.eid[7:23], key, ev.vh, t_ms))
        if t_ms >= self.events_until or tick % 2:
            return
        honest = [i for i in range(self.n) if i != self.me]
        if "T1" in a:
            j = self.rng.choice(honest)
            key = "target_%d.pos" % self.rng.randrange(N_TARGETS)
            val = self.next_marker()
            body = {"key": key, "value": val, "hlc": self.hlc.now(t_ms),
                    "node": j, "seq": 10_000_000 + self.marker}
            if self.rate_mode == "kseq":
                body["kseq"] = body["seq"]
            if self.bq:
                env = self.forge_group(j, body)
            else:
                env = make_env(self.k, "glyph", body, author=self.keys[j].author)
                self.fresh["push"].append(_Raw(env))
            self.attack_log.append(("T1", env["id"][7:23], key, vhash(val), t_ms))
        if "T3" in a:
            old = [e for e in self.replay_pool[:2000]
                   if t_ms - e.arrival >= 30_000]
            if old:
                e = self.rng.choice(old)
                if self.bq:
                    self.raw_out.extend(self.encode_events([e]))
                else:
                    self.fresh["push"].append(_Raw(e.env))
                self.attack_log.append(("T3", e.eid[7:23], e.key, e.vh, t_ms))
        self.t2_budget = 1

    def maybe_tamper(self, ev, t_ms):
        if (t_ms < ATTACK_START_MS or t_ms >= self.events_until
                or not ev.key.startswith("target_") or self.t2_budget <= 0
                or self.rng.random() >= 0.5):
            return
        self.t2_budget -= 1
        val = self.next_marker()
        env = dict(ev.env)
        env["body"] = dict(ev.env["body"], value=val)
        if self.bq:
            # подмена значения листа при старом корне и старой подписи
            if ev.bref is None:
                return
            bkey, i = ev.bref
            info = self.batches[bkey]
            proof = self.proof_for(info, i)
            if proof is None:
                return
            rec = self.codec.enc_group(info["aidx"], info["n"], info["hlc"],
                                       info["sig"],
                                       [(i, "glyph", env["body"], proof)], False)
            self.raw_out.append((rec, 1))
        else:
            self.fresh["push"].append(_Raw(env))
        self.attack_log.append(("T2", ev.eid[7:23], ev.key, vhash(val), t_ms))

    def revoked_eff(self):
        return self.revoke_at

    def issue_revocation(self, t_ms):
        """Узел 0 публикует заранее подписанный сертификат отзыва ключа
        атакующего. В жизни сертификат держит оператор, а не соседний дрон."""
        victim = self.keys[self.attacker]
        body = {"action": "revoke", "key": victim.author,
                "effective_from_ms": self.revoke_at, "reason": "compromised",
                "proof": victim.revoke_cert, "hlc": self.hlc.now(t_ms),
                "node": self.me}
        env = make_env(self.k, "key_event", body)
        self.accept_key_event(env, self.me, t_ms)

    # ---------------------------------------------------- пакеты (v1.1)

    BATCH_ROOM = net.MAX_DGRAM - net.HDR.size
    BATCH_BODY = 900                # байт событий в своём пакете

    def new_env(self, kind, body):
        if self.bq:
            return unsigned_env(self.k.author, kind, body)
        return make_env(self.k, kind, body)

    def emit(self, ev, cls):
        """Своё событие: v1.0 - сразу в рассылку, v1.1 - в открытый пакет."""
        if self.bq:
            self.outbox.append(ev)
        else:
            self.fresh[cls].append(ev)

    def sign_batch(self, aidx, leaves, hlc):
        root = wire.merkle_root(leaves)
        msg = canon(wire.batch_header(self.keys[aidx].author, root, hlc,
                                      len(leaves)))
        return root, self.k.sign(msg)

    def close_batch(self):
        """Закрыть открытый пакет: свои события за квант -> один или
        несколько пакетов (по размеру датаграммы), одна подпись на корень."""
        if not self.outbox:
            return
        chunks, cur, size = [], [], 0
        for ev in self.outbox:
            b = bytearray()
            self.codec.enc_event(b, ev.kind, ev.env["body"], self.me,
                                 ev.hlc[0] if not cur else cur[0].hlc[0])
            if cur and (size + len(b) > self.BATCH_BODY or len(cur) >= 255):
                chunks.append(cur)
                cur, size = [], 0
                b = bytearray()
                self.codec.enc_event(b, ev.kind, ev.env["body"], self.me,
                                     ev.hlc[0])
            cur.append(ev)
            size += len(b)
        chunks.append(cur)
        for evs in chunks:
            hlc = [evs[0].hlc[0], evs[0].hlc[1]]
            leaves = [bytes.fromhex(e.eid[7:]) for e in evs]
            root, sig = self.sign_batch(self.me, leaves, hlc)
            bkey = (self.me, root)
            self.batches[bkey] = {"aidx": self.me, "n": len(evs), "hlc": hlc,
                                  "sig": sig, "leaves": leaves, "proofs": {},
                                  "eids": {}}
            self.verified.add(bkey)
            for i, e in enumerate(evs):
                e.bref = (bkey, i)
            self.fresh["push"].extend(evs)
            self.batches_made += 1
        self.outbox = []

    def forge_group(self, j, body):
        """T1 в v1.1: пакет от имени узла j, подписанный ключом атакующего."""
        env = unsigned_env(self.keys[j].author, "glyph", body)
        leaves = [bytes.fromhex(env["id"][7:])]
        hlc = list(body["hlc"])
        _, sig = self.sign_batch(j, leaves, hlc)
        rec = self.codec.enc_group(j, 1, hlc, sig,
                                   [(0, "glyph", body, None)], True)
        self.raw_out.append((rec, 1))
        return env

    @staticmethod
    def proof_for(info, i):
        if info["leaves"] is not None:
            return wire.merkle_proof(info["leaves"], i)
        return info["proofs"].get(i)

    @staticmethod
    def _eid(info, i):
        if info["leaves"] is not None:
            return "sha256:" + info["leaves"][i].hex()
        return info["eids"].get(i)

    def encode_events(self, evs):
        """События -> записи провода [(байты, событий)]. Пакет, все события
        которого у меня есть, идёт целиком без путей, если так не длиннее;
        иначе - выбранные события с путями Меркла, при нужде несколькими
        записями (каждая со своим заголовком и подписью пакета)."""
        out, groups, seen = [], {}, set()
        for e in evs:
            if e.eid in seen:
                continue
            seen.add(e.eid)
            if e.kind == "key_event":
                out.append((wire.enc_json(canon(e.env)), 1))
            elif e.bref is not None:
                groups.setdefault(e.bref[0], []).append(e.bref[1])
        for bkey, idxs in groups.items():
            info = self.batches[bkey]
            n, aidx, hlc, sig = info["n"], info["aidx"], info["hlc"], info["sig"]
            full = None
            if info["leaves"] is not None:
                every = [self.events.get(self._eid(info, i)) for i in range(n)]
                if all(x is not None for x in every):
                    full = self.codec.enc_group(
                        aidx, n, hlc, sig,
                        [(i, x.kind, x.env["body"], None)
                         for i, x in enumerate(every)], True)
            items = []
            for i in sorted(set(idxs)):
                proof = self.proof_for(info, i)
                x = self.events.get(self._eid(info, i))
                if proof is not None and x is not None:
                    items.append((i, x.kind, x.env["body"], proof))
            recs, cur = [], []
            for it in items:
                trial = self.codec.enc_group(aidx, n, hlc, sig, cur + [it], False)
                if cur and len(trial) > self.BATCH_ROOM:
                    recs.append((self.codec.enc_group(aidx, n, hlc, sig, cur,
                                                      False), len(cur)))
                    cur = [it]
                else:
                    cur.append(it)
            if cur:
                recs.append((self.codec.enc_group(aidx, n, hlc, sig, cur, False),
                             len(cur)))
            psize = sum(len(r) for r, _ in recs)
            if full is not None and len(full) <= self.BATCH_ROOM and (
                    not recs or len(full) <= psize):
                out.append((full, n))
            else:
                out.extend(recs)
        return out

    def on_group(self, g, now_ms):
        """Пакет с провода: пересчитать id и корень, проверить подпись над
        корнем (раз на пакет), затем принять события как обычно."""
        self.clock = now_ms
        aidx, n = g["aidx"], g["n"]
        author = self.keys[aidx].author
        items = []
        for i, kind, body, proof in g["items"]:
            env = unsigned_env(author, kind, body)
            items.append((i, env, bytes.fromhex(env["id"][7:]), proof))
        try:
            if g["complete"]:
                leaves = [x[2] for x in items]
                roots = {wire.merkle_root(leaves)}
            else:
                leaves = None
                roots = {wire.root_from_proof(x[2], x[0], n, x[3]) for x in items}
        except (ValueError, IndexError):
            roots = set()
        if len(roots) != 1:
            for _ in items:
                self.reject("sig")
            return
        root = roots.pop()
        bkey = (aidx, root)
        if bkey not in self.verified:
            msg = canon(wire.batch_header(author, root, g["hlc"], n))
            if not verify_raw(self.keys[aidx].pub, g["sig"], msg):
                for _ in items:
                    self.reject("sig")
                return
            self.verified.add(bkey)
        info = self.batches.get(bkey)
        if info is None:
            info = self.batches[bkey] = {"aidx": aidx, "n": n, "hlc": g["hlc"],
                                         "sig": g["sig"], "leaves": None,
                                         "proofs": {}, "eids": {}}
        if leaves is not None and info["leaves"] is None:
            info["leaves"] = leaves
        for i, env, leaf, proof in items:
            if info["leaves"] is None:
                info["proofs"][i] = proof
                info["eids"][i] = env["id"]
            self.accept(env, now_ms, bref=(bkey, i))

    def digest_bin(self, t_ms):
        vv = []
        for a in range(self.n):
            c = self.contig[a]
            bits = 0
            for s in self.seen_seq[a]:
                if c < s <= c + BITMAP:
                    bits |= 1 << (s - c - 1)
            vv.append((c, bits))
        sl = None
        if self.cfg.get("slot_digest", True):
            sl = [bytes.fromhex(self.win_full[ki][7:13]) if ki in self.win_full
                  else b"\0\0\0" for ki in range(len(self.keys_list))]
        return self.codec.enc_digest(self.me, self.hlc.now(t_ms), vv,
                                     sorted(self.revoked), sl, self.k.sign)

    def on_digest(self, g):
        a = g["aidx"]
        if a >= self.n or not verify_raw(self.keys[a].pub, g["sig"], g["signed"]):
            self.reject("digest")
            return []
        return self.answer_body(g["body"])

    # ------------------------------------------------------------ PULL

    def digest(self, t_ms):
        vv = {}
        for a in range(self.n):
            c = self.contig[a]
            bits = 0
            for s in self.seen_seq[a]:
                if c < s <= c + BITMAP:
                    bits |= 1 << (s - c - 1)
            vv[str(a)] = [c, "%x" % bits]
        # Карта по слотам: 6 hex победителя на ключ в порядке keys_universe.
        # Нужна, потому что журнал по диапазонам досылается от старого к
        # новому: после 5-минутного разделения свежие пропуски ждали за
        # хвостом ~9000 событий, и узлы не сходились (прогон v1, см. отчёт).
        body = {"vv": vv, "rev": sorted(self.revoked), "node": self.me,
                "hlc": self.hlc.now(t_ms)}
        if self.cfg.get("slot_digest", True):     # --no-slot-digest = v1
            body["sl"] = "".join(self.win_full.get(ki, "")[7:13] or "------"
                                 for ki in range(len(self.keys_list)))
        return make_env(self.k, "digest", body)

    def answer_digest(self, env):
        if check_env(env, self.pubs):
            self.reject("digest")
            return []
        return self.answer_body(env["body"])

    def answer_body(self, body):
        out = []
        for ke in self.key_events:
            if self.idx_of[ke.env["body"]["key"]] not in body.get("rev", []):
                out.append(ke)
        sl = body.get("sl", "")
        if isinstance(sl, str) and len(sl) == 6 * len(self.keys_list):
            for ki in range(len(self.keys_list)):
                mine = self.win_full.get(ki)
                if mine and mine[7:13] != sl[6 * ki:6 * ki + 6]:
                    out.append(self.events[mine])
        vv = body.get("vv", {})
        for a in range(self.n):
            c, bm = vv.get(str(a), [0, "0"])
            try:
                bits = int(bm, 16)
            except (TypeError, ValueError):
                bits = 0
            held = self.held[a]
            i = bisect_right(held, c)
            while i < len(held) and len(out) < PULL_CAP:
                s = held[i]
                i += 1
                if s - c <= BITMAP and bits >> (s - c - 1) & 1:
                    continue
                out.append(self.events[self.by_seq[(a, s)]])
            if len(out) >= PULL_CAP:
                break
        return out

    # ---------------------------------------------------------- картина

    def world_hash(self):
        return self.table.world_hash(self.admitted)

    def record_picture(self, tick):
        for ki, key in enumerate(self.keys_list):
            w = self.table.winner(key, self.admitted)
            cur = (w[0][7:23], self.events[w[0]].vh) if w else ("", "")
            if w:
                self.win_full[ki] = w[0]
            else:
                self.win_full.pop(ki, None)
            if self.last_win.get(ki) != cur:
                self.last_win[ki] = cur
                self.changes.append((tick, ki, cur[0], cur[1]))


class _Raw:
    """Конверт, который атакующий шлёт как есть, не принимая его сам."""
    __slots__ = ("env", "raw")

    def __init__(self, env):
        self.env = env
        self.raw = None

    def bytes(self):
        if self.raw is None:
            self.raw = canon(self.env)
        return self.raw


# ---------------------------------------------------------- цикл процесса

def send_v11(node, link, pull_replies, t, now, K, idx):
    """Отправка v1.1: всё бинарными записями, датаграмма <= MAX_DGRAM."""
    room = net.MAX_DGRAM - net.HDR.size

    def out(dst, cls, recs):
        for payload, n_ev, n_sig in wire.pack_records(recs, room):
            link.send(dst, cls, payload, t, now, n_ev, n_sig)

    for src, evs in pull_replies:
        if evs:
            out(src, "pull", node.encode_events(evs))
    if t % K == idx % K and node.nbrs and not node.is_attacker:
        out(node.rng.choice(node.nbrs), "pull", [(node.digest_bin(now), 1)])
    evs = node.fresh["push"] + node.fresh["svc"]
    recs = node.encode_events(evs) + node.raw_out
    if recs:
        dsts = (node.nbrs if len(node.nbrs) <= 2
                else node.rng.sample(node.nbrs, 2))
        for dst in dsts:
            out(dst, "push", recs)
    node.fresh["push"], node.fresh["svc"], node.raw_out = [], [], []
    if node.fresh["all"]:
        recs = [(wire.enc_json(canon(e.env)), 1) for e in node.fresh["all"]]
        for dst in node.nbrs:
            out(dst, "svc", recs)
        node.fresh["all"] = []


def run_node(cfg, idx, barrier, counts, q):
    """Процесс узла. Lockstep: такт = tick_ms сим-времени; после барьера
    узел дочитывает из сокета все датаграммы прошлого такта (их число
    объявлено отправителем в counts), поэтому порядок доставки не зависит от
    планировщика ОС."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cpu0, wall0 = time.process_time(), time.perf_counter()
    node = Node(cfg, idx)
    n, tick_ms = cfg["n"], cfg["tick_ms"]
    cut, _ = net.parse_partition(cfg["partition"])
    link = net.Link(idx, n, cfg["base_port"], cfg["loss"], cfg["seed"], cut,
                    active_until_ms=cfg["events_until_ms"])
    link.sock.settimeout(2.0)
    sched = cfg["schedule"]
    si = 0
    ticks = cfg["duration_ms"] // tick_ms
    K = cfg["K"]
    heap = []
    got = {}                    # (src, send_tick) -> датаграмм получено
    transport_timeouts = 0
    wait_s = 0.0                # ожидание барьера и сокета: не CPU узла
    try:
        barrier.wait(120)
        busy0 = time.perf_counter()
        for t in range(ticks):
            t0 = t * tick_ms
            link.new_tick()
            base = (t % 2) * n * n
            for j in range(n):
                counts[base + idx * n + j] = 0
            # 1. дочитать датаграммы такта t-1
            if t > 0:
                pb = ((t - 1) % 2) * n * n
                need = {src: counts[pb + src * n + idx] for src in range(n)
                        if src != idx}
                while any(got.get((s, t - 1), 0) < c for s, c in need.items()):
                    tw = time.perf_counter()
                    try:
                        data, _ = link.sock.recvfrom(2048)
                        wait_s += time.perf_counter() - tw
                    except socket.timeout:
                        wait_s += time.perf_counter() - tw
                        transport_timeouts += 1
                        break
                    except ConnectionResetError:
                        continue
                    src, st, dms, dseq, cls = net.HDR.unpack_from(data)
                    got[(src, st)] = got.get((src, st), 0) + 1
                    heapq.heappush(heap, (dms, src, dseq, st, cls, data))
                for s in need:
                    got.pop((s, t - 1), None)
            # 2. собрать доставку этого такта и свою генерацию по времени
            items = []
            defer = []
            limit = t0 + tick_ms
            while heap and heap[0][0] < limit:
                it = heapq.heappop(heap)
                if it[3] >= t:
                    defer.append(it)
                else:
                    items.append((max(it[0], t0), 0, it[1], it[2], it))
            for it in defer:
                heapq.heappush(heap, it)
            while si < len(sched) and sched[si][0] < limit:
                items.append((sched[si][0], 1, si, 0, sched[si]))
                si += 1
            items.sort(key=lambda x: x[:4])
            pull_replies = []
            for now_ms, kind, _, _, it in items:
                if kind == 1:
                    node.generate(now_ms, it[1], it[2], it[3])
                    continue
                src, cls, data = it[1], it[4], it[5]
                if node.bq:
                    try:
                        recs = node.codec.decode(data[net.HDR.size:])
                    except (ValueError, IndexError, KeyError):
                        node.reject("schema")
                        continue
                    for typ, g in recs:
                        if typ == "group":
                            node.on_group(g, now_ms)
                        elif typ == "digest":
                            if not node.is_attacker:
                                pull_replies.append((src, node.on_digest(g)))
                        else:
                            try:
                                env = json.loads(g)
                            except ValueError:
                                node.reject("schema")
                                continue
                            node.accept(env, now_ms)
                    continue
                try:
                    msgs = json.loads(data[net.HDR.size:])
                except ValueError:
                    node.reject("schema")
                    continue
                for env in msgs:
                    if isinstance(env, dict) and env.get("kind") == "digest":
                        if not node.is_attacker:
                            pull_replies.append((src, node.answer_digest(env)))
                        continue
                    node.accept(env, now_ms)
            # 3. атаки и отзыв
            if node.is_attacker:
                node.attack_tick(t, t0)
            if ("T6" in node.attacks and idx == 0
                    and t0 <= node.revoke_at < t0 + tick_ms):
                node.issue_revocation(node.revoke_at)
            # 4. отправка
            now = t0 + tick_ms - 1
            if node.bq:
                if t % node.bq == node.bq - 1:
                    node.close_batch()
                send_v11(node, link, pull_replies, t, now, K, idx)
            else:
                for src, evs in pull_replies:
                    if evs:
                        for p in net.pack([e.bytes() for e in evs]):
                            link.send(src, "pull", p, t, now, p.count(b'"gep"'))
                if t % K == idx % K and node.nbrs and not node.is_attacker:
                    dst = node.rng.choice(node.nbrs)
                    d = canon(node.digest(now))
                    link.send(dst, "pull", b"[" + d + b"]", t, now, 1)
                for cls in ("push", "svc"):
                    evs = node.fresh[cls]
                    if not evs:
                        continue
                    dsts = (node.nbrs if len(node.nbrs) <= 2
                            else node.rng.sample(node.nbrs, 2))
                    packets = net.pack([e.bytes() for e in evs])
                    for dst in dsts:
                        for p in packets:
                            link.send(dst, cls, p, t, now, p.count(b'"gep"'))
                    node.fresh[cls] = []
                if node.fresh["all"]:
                    packets = net.pack([e.bytes() for e in node.fresh["all"]])
                    for dst in node.nbrs:
                        for p in packets:
                            link.send(dst, "svc", p, t, now, p.count(b'"gep"'))
                    node.fresh["all"] = []
            # 5. картина, карантин, доверие
            node.record_picture(t)
            if t % 5 == 4:
                node.check_pending(now)
            if t % 10 == 9:
                node.trust_series.append([round(x, 3) for x in node.trust])
            for j in range(n):
                counts[base + idx * n + j] = link.sent_to[j]
            tw = time.perf_counter()
            barrier.wait(120)
            wait_s += time.perf_counter() - tw
        busy_s = time.perf_counter() - busy0 - wait_s
        node.check_pending(ticks * tick_ms)
        for eid, arr in node.fp_pending.items():
            node.fp_records.append((eid[7:23], node.events[eid].author, arr, -1))
        barrier.wait(120)
    finally:
        link.close()
    glyphs = [e for e in node.events.values() if e.kind == "glyph"]
    res = {
        "idx": idx,
        "own": node.own,
        "attack_log": node.attack_log,
        "changes": node.changes,
        "trust_series": node.trust_series,
        "fp_records": node.fp_records,
        "arrivals": {e.eid[7:23]: e.arrival // tick_ms for e in glyphs},
        "quarantined": [e.eid[7:23] for e in glyphs
                        if not node.admitted(e.eid)],
        "world_hash": node.world_hash(),
        "deprecated": len(node.table.deprecated(node.admitted)),
        "stored": len(node.events),
        "rejects": node.rejects,
        "rejects_by": node.rejects_by,
        "dups": node.dups,
        "self_val_ignored": node.self_val_ignored,
        "validations_sent": node.validations_sent,
        "first_rate_reject": node.first_rate_reject,
        "seq_lead": node.seq_lead,
        "revoked_recv_ms": node.revoked_at_recv,
        "tx": link.tx, "tx_active": link.tx_active,
        "lost": link.lost, "cut_drops": link.cut_drops, "dups_sent": link.dups,
        "verify_n": VERIFY["n"], "verify_s": VERIFY["s"],
        "cpu_s": time.process_time() - cpu0,
        "busy_s": busy_s,
        "wall_s": time.perf_counter() - wall0,
        "transport_timeouts": transport_timeouts,
        "drift": node.hlc.drift,
        "slashed": {str(v): ms for v, ms in node.slashed.items()},
        "batches_made": node.batches_made,
    }
    q.put(res)
