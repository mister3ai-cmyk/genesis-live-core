# Business Source License 1.1 - см. research/LICENSE
"""Тесты симулятора роя (маркер concept: это модель, а не движок)."""
import hashlib
import os
import random
import sys

import pytest

pytest.importorskip("cryptography")
np = pytest.importorskip("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM = os.path.dirname(HERE)
RESEARCH = os.path.dirname(SWARM)
for p in (SWARM, RESEARCH):
    if p not in sys.path:
        sys.path.insert(0, p)

import swarm_net as net                                     # noqa: E402
import swarm_node as sn                                     # noqa: E402
import swarm_p2p_simulation as sim                          # noqa: E402

pytestmark = pytest.mark.concept


def mk_node(idx, n=3, trust=1.0, branch="B", drifts=None, truth=None, attack="none",
            **extra):
    if truth is None:
        truth = sim.build_world(1, n, 20_000)["truth"]
    cfg = dict(n=n, seed=1, tick_ms=100, branch=branch, trust_init=trust,
               truth=truth, topology="full", attack=attack, attacker=n - 1,
               events_until_ms=20_000, drifts=drifts or [0] * n, **extra)
    return sn.Node(cfg, idx)


def mk_reader(idx=2, **kw):
    """Читатель без своего сенсора: проверяется правило слота, а не доверие."""
    r = mk_node(idx, **kw)
    r.sensor.covers = lambda key, now: False
    return r


def some_events(n_per=30, seed=3):
    """Подписанные события узлов 0 и 1 по пересекающимся ключам."""
    a, b = mk_node(0), mk_node(1)
    rng = random.Random(seed)
    envs = []
    keys = ["sector_1.status", "sector_2.status", "target_0.pos"]
    for i in range(n_per):
        for node in (a, b):
            k = rng.choice(keys)
            v = (rng.choice(sn.STATUSES) if k.endswith("status")
                 else [rng.randint(0, 1000), rng.randint(0, 1000)])
            ev = node.gen_own(1000 + i * 200 + rng.randint(0, 150), k, v, None)
            envs.append(ev.env)
    return envs


# 1 ------------------------------------------------------------------------

def test_forgery_and_tamper_rejected():
    k0, k1 = sn.Keys(1, 0), sn.Keys(1, 1)
    pubs = {k0.author: k0.pub, k1.author: k1.pub}
    body = {"key": "target_0.pos", "value": [1, 2], "hlc": [5, 0], "node": 0, "seq": 1}
    good = sn.make_env(k0, "glyph", body)
    assert sn.check_env(good, pubs) is None
    forged = sn.make_env(k1, "glyph", body, author=k0.author)      # T1
    assert sn.check_env(forged, pubs) == "sig"
    tampered = dict(good, body=dict(body, value=[9, 9]))            # T2
    assert sn.check_env(tampered, pubs) == "hash"
    reader = mk_node(2)
    assert reader.accept(forged, 100) is None
    assert reader.accept(tampered, 100) is None
    assert reader.rejects == {"sig": 1, "hash": 1}
    assert reader.accept(good, 100) is not None


def test_no_float_in_signed_fields():
    with pytest.raises(TypeError):
        sn.make_env(sn.Keys(1, 0), "glyph", {"value": [1.5, 2]})


def test_oversized_datagram_is_an_error():
    with pytest.raises(ValueError):
        net.pack([b"x" * 1300])
    for p in net.pack([b"y" * 400] * 7):
        assert net.HDR.size + len(p) <= net.MAX_DGRAM


# 2 ------------------------------------------------------------------------

def test_branch_b_order_independent():
    envs = some_events()
    hashes = set()
    for s in range(20):
        perm = envs[:]
        random.Random(s).shuffle(perm)
        r = mk_reader()
        for e in perm:
            assert r.accept(e, 5000) is not None
        hashes.add(r.world_hash())
    assert len(hashes) == 1


def test_branch_a_depends_on_order():
    """Контроль: на ветке A перестановка картину меняет - иначе тест выше
    ничего не доказывает."""
    envs = some_events()
    hashes = set()
    for s in range(20):
        perm = envs[:]
        random.Random(s).shuffle(perm)
        r = mk_reader(branch="A")
        for e in perm:
            r.accept(e, 5000)
        hashes.add(r.world_hash())
    assert len(hashes) > 1


def test_trust_policy_makes_admission_order_dependent():
    """Найденное свойство, не цель: когда читатель опровергает события
    (своим сенсором), доверие с клипом [0, 1] и подтверждением «только того,
    что в карантине» зависит от порядка прихода. Сходимость B гарантирована
    правилом слота, но не допуском из карантина."""
    envs = some_events()
    hashes = set()
    for s in range(10):
        perm = envs[:]
        random.Random(s).shuffle(perm)
        r = mk_node(2)                       # сенсор включён, значения случайны
        for e in perm:
            r.accept(e, 5000)
        hashes.add(r.world_hash())
    assert len(hashes) > 1


# 3 ------------------------------------------------------------------------

def test_replay_does_not_change_picture():
    envs = some_events()
    r = mk_reader()
    for e in envs:
        r.accept(e, 5000)
    h = r.world_hash()
    for e in random.Random(1).sample(envs, 15):
        assert r.accept(e, 9000) is None
    assert r.world_hash() == h
    assert r.dups == 15


# 4 ------------------------------------------------------------------------

def test_revoked_key_rejected_after_key_event():
    issuer, victim, reader = mk_node(0), mk_node(1), mk_node(2)
    before = victim.gen_own(1000, "sector_1.status", "clear", None)
    assert reader.accept(before.env, 1100) is not None
    # узел 0 публикует заранее подписанный сертификат отзыва ключа 1
    issuer.attacker = 1
    issuer.revoke_at = 5000
    issuer.issue_revocation(5000)
    ke = issuer.key_events[0].env
    assert reader.accept(ke, 5100) is not None
    after = victim.gen_own(6000, "sector_2.status", "hazard", None)
    assert reader.accept(after.env, 6100) is None
    assert reader.rejects.get("revoked") == 1
    backdated = sn.make_env(victim.k, "glyph", {
        "key": "sector_3.status", "value": "obstacle", "hlc": [2000, 0],
        "node": 1, "seq": 99})
    assert reader.accept(backdated.copy(), 6200) is None
    assert reader.rejects.get("unanchored") == 1
    # принятое до отзыва с меткой раньше effective_from остаётся в картине
    assert reader.admitted(before.eid)
    # поддельный сертификат (подписан не тем ключом) не отзывает
    fake = sn.make_env(issuer.k, "key_event", dict(ke["body"], key=issuer.k.author,
                                                  proof=victim.k.revoke_cert))
    r2 = mk_node(2)
    assert r2.accept(fake, 100) is None


# 5 ------------------------------------------------------------------------

def _fingerprint(results):
    return [(r["world_hash"], len(r["changes"]), r["stored"],
             sorted(r["rejects"].items()), r["tx"]["push"], r["tx"]["pull"],
             r["dups"]) for r in results]


def test_small_run_deterministic():
    cfg = dict(n=3, duration_s=30, events_until_s=20, seed=7, loss=0.2,
               base_port=23000, batch_ms=0, slash=False)          # v1.0
    rec1, res1 = sim.run_once(cfg)
    rec2, res2 = sim.run_once(dict(cfg, base_port=23100))
    assert _fingerprint(res1) == _fingerprint(res2)
    assert rec1["metrics"]["transport_timeouts"] == 0


# 6 ------------------------------------------------------------------------

class _StubEncoder:
    """Детерминированный энкодер: случайный единичный вектор от sha256 текста.
    Косинус двух разных имён ~0, ниже порогов Memory (0.5 и 0.6), поэтому
    смыслового сведения нет и остаётся только точное совпадение ключа."""

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(384)
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


class _StubGlyphs:
    def __init__(self):
        self.vecs = np.zeros((0, 384), dtype=np.float32)
        self.chunks = []

    def add(self, vec, chunk):
        self.vecs = np.vstack([self.vecs, np.asarray(vec, dtype=np.float32)[None]])
        self.chunks.append(chunk)
        return len(self.chunks) - 1


def test_slot_table_matches_memory_on_exact_keys():
    amt = pytest.importorskip("agent_memory_traffic")
    mem = amt.Memory(_StubGlyphs(), _StubEncoder(), None)
    tab = sn.SlotTable("A")
    keys = sn.keys_universe(5)
    rng = random.Random(11)
    for i in range(300):
        key = rng.choice(keys)
        ent, attr = key.split(".")
        val = "v%d" % rng.randint(0, 40)
        mem.turn = i
        mem.write("", ent, attr, val)
        tab.add(key, "e%d" % i, val)
    assert mem.merged_attrs == 0
    mem_now = {}
    for (ent, attr), idx in mem.slots.items():
        assert not mem.meta[idx]["deprecated"]
        mem_now["%s.%s" % (ent, attr)] = mem.notes[idx].split(" = ", 1)[1]
    assert mem_now == tab.values()
    assert mem.superseded == len(tab.deprecated())


# 7 ------------------------------------------------------------------------

def test_branch_c_equals_branch_a():
    a, c = sn.SlotTable("A"), sn.SlotTable("C")
    rng = random.Random(5)
    for i in range(200):
        k = "sector_%d.status" % rng.randrange(4)
        v = rng.choice(sn.STATUSES)
        a.add(k, "e%d" % i, v)
        c.add(k, "e%d" % i, v)
    assert a.picture() == c.picture()
    assert c.deprecated() == set() and len(a.deprecated()) == 200 - len(a.picture())


# 8 ------------------------------------------------------------------------

def _conflict_truth():
    truth = sim.build_world(1, 3, 20_000)["truth"]
    truth = dict(truth)
    truth["sector_0.status"] = ([0, 10_000], ["clear", "hazard"])
    return truth


def test_honest_conflict_both_confirmed_and_b_picks_new():
    truth = _conflict_truth()
    sensor = sn.Sensor(truth, 1, 2)
    # дрейф до 2 с в любую сторону: оба значения правы на своё время
    for drift in (-2000, 0, 2000):
        assert sensor.check("sector_0.status", "clear", 9_990 + drift, 12_000)
        assert sensor.check("sector_0.status", "hazard", 10_010 + drift, 12_000)
    old_w, new_w, reader = (mk_node(0, truth=truth), mk_node(1, truth=truth),
                            mk_node(2, truth=truth))
    e_old = old_w.gen_own(9_990, "sector_0.status", "clear", None)
    e_new = new_w.gen_own(10_010, "sector_0.status", "hazard", None)
    for e in (e_new, e_old):
        reader.accept(e.env, 10_200)
    assert reader.table.values(reader.admitted)["sector_0.status"] == "hazard"


def test_clock_drift_can_make_b_pick_old():
    """Свойство HLC, а не баг: если часы писавшего старое значение спешат,
    B выберет старое. В отчёте - доля таких конфликтов."""
    truth = _conflict_truth()
    nodes = [mk_node(i, truth=truth, drifts=[1500, 0, 0]) for i in range(3)]
    e_old = nodes[0].gen_own(9_990, "sector_0.status", "clear", None)
    e_new = nodes[1].gen_own(10_010, "sector_0.status", "hazard", None)
    for e in (e_new, e_old):
        nodes[2].accept(e.env, 10_200)
    assert nodes[2].table.values(nodes[2].admitted)["sector_0.status"] == "clear"


# 9 ------------------------------------------------------------------------

def test_self_and_relayed_validations_do_not_count():
    author, other, reader = mk_node(0, trust=0.5), mk_node(1, trust=0.5), mk_node(2, trust=0.5)
    reader.sensor.covers = lambda key, now: False     # читатель сам не видит
    e = author.gen_own(1000, "target_1.pos", [5, 5], None)
    reader.accept(e.env, 1100)
    assert not reader.admitted(e.eid)
    # автор подтверждает сам себя - не считается
    author.emit_validation(e, "confirmed", 1200)
    selfv = author.fresh["svc"][-1].env
    reader.accept(selfv, 1300)
    assert reader.self_val_ignored == 1 and not reader.admitted(e.eid)
    # узел 1 выдаёт чужое наблюдение за своё (body.node = 2) - отказ
    fake = sn.make_env(other.k, "validation", {
        "glyph_hash": e.eid, "verdict": "confirmed",
        "evidence": {"applied_in": "sensor:target_1.pos", "note": None},
        "hlc": [1300, 0], "node": 2, "seq": 1})
    assert reader.accept(fake, 1350) is None
    # собственное наблюдение узла 1 - считается, и только один раз
    other.events[e.eid] = e
    other.emit_validation(e, "confirmed", 1400)
    v = other.fresh["svc"][-1].env
    reader.accept(v, 1500)
    assert reader.admitted(e.eid)
    t = reader.trust[0]
    reader.accept(v, 1600)                 # пересылка той же валидации
    assert reader.trust[0] == t and reader.dups == 1


# ===================================================================== v1.1

def _batch_node(idx, **kw):
    return mk_node(idx, batch_ms=500, **kw)


def _own_batch(node, k=5, t0=1000):
    truth = node.truth
    for i in range(k):
        key = "sector_%d.status" % i
        node.gen_own(t0 + 30 * i, key, sn.truth_at(truth, key, t0), None)
    node.close_batch()
    evs, node.fresh["push"] = node.fresh["push"], []
    return evs


def _deliver(reader, recs, now=2000):
    for rec, _ in recs:
        for typ, g in reader.codec.decode(rec):
            assert typ == "group"
            reader.on_group(g, now)


def test_v11_merkle_batch_verifies_and_leaf_tamper_rejected():
    a, r = _batch_node(0), _batch_node(2)
    evs = _own_batch(a)
    recs = a.encode_events(evs)
    assert len(recs) == 1 and recs[0][1] == 5       # один полный пакет
    _deliver(r, recs)
    assert all(e.eid in r.events for e in evs) and not r.rejects
    # подмена значения одного листа при старой подписи (T2) - весь пакет мимо
    r2 = _batch_node(2)
    g = r2.codec.decode(recs[0][0])[0][1]
    i, kind, body, _ = g["items"][2]
    g["items"][2] = (i, kind, dict(body, value="hazard" if body["value"] != "hazard"
                                   else "clear"), None)
    r2.on_group(g, 2000)
    assert r2.rejects == {"sig": 5} and not r2.events


def test_v11_single_event_with_merkle_path():
    a = _batch_node(0)
    evs = _own_batch(a)
    info = a.batches[evs[0].bref[0]]
    for i in (0, 3, 4):
        r = _batch_node(2)
        rec = a.codec.enc_group(0, info["n"], info["hlc"], info["sig"],
                                [(i, "glyph", evs[i].env["body"],
                                  a.proof_for(info, i))], False)
        _deliver(r, [(rec, 1)])
        assert list(r.events) == [evs[i].eid]
    r = _batch_node(2)                       # путь от чужого листа
    rec = a.codec.enc_group(0, info["n"], info["hlc"], info["sig"],
                            [(3, "glyph", evs[3].env["body"],
                              a.proof_for(info, 1))], False)
    _deliver(r, [(rec, 1)])
    assert not r.events and r.rejects == {"sig": 1}


def test_v11_binary_wire_roundtrip_keeps_canon_and_id():
    a = _batch_node(0)
    a.gen_own(1000, "drone_0.pos", [10, -20, 3000], None)
    a.gen_own(1010, "drone_0.battery", 97, None)
    g = a.gen_own(1020, "target_1.pos", [123456, 7], None)
    a.emit_validation(g, "refuted", 1030)
    a.close_batch()
    evs = a.fresh["push"]
    b = _batch_node(1)
    _deliver(b, a.encode_events(evs))
    for e in evs:
        got = b.events[e.eid]
        assert sn.canon(got.env) == sn.canon(e.env)


def test_v11_forged_author_batch_rejected():
    att = _batch_node(2, attack="T1")
    r = _batch_node(1)
    body = {"key": "target_0.pos", "value": [1, 2], "hlc": [1000, 0],
            "node": 0, "seq": 10_000_001}
    att.forge_group(0, body)
    _deliver(r, att.raw_out)
    assert r.rejects == {"sig": 1} and not r.events


def _slash_trio(covers):
    """автор 0 (честный), обвинитель 1, судья 2; доверие стартует с 0.5."""
    truth = sim.build_world(1, 3, 20_000)["truth"]
    au, ac, ju = (mk_node(i, trust=0.5, truth=truth, slash=True) for i in range(3))
    ju.sensor.covers = covers
    return truth, au, ac, ju


def test_v11_false_refutation_slashes_accuser_and_restores_victim():
    truth, au, ac, ju = _slash_trio(lambda k, now: k == "sector_2.status")
    ctrl = mk_node(2, trust=0.5, truth=truth, slash=True)
    ctrl.sensor.covers = ju.sensor.covers
    e1 = au.gen_own(1000, "sector_1.status",
                    sn.truth_at(truth, "sector_1.status", 1000), None)
    e2 = au.gen_own(1100, "sector_2.status",
                    sn.truth_at(truth, "sector_2.status", 1100), None)
    for node in (ju, ctrl, ac):
        node.accept(e1.env, 1200)
    ac.emit_validation(e1, "refuted", 1300)          # судья этот ключ не видит
    ju.accept(ac.fresh["svc"][-1].env, 1400)
    assert ju.trust[0] < 0.5 and not ju.slashed
    for node in (ju, ctrl, ac):
        node.accept(e2.env, 1500)
    ac.emit_validation(e2, "refuted", 1600)          # а этот видит: правда
    ju.accept(ac.fresh["svc"][-1].env, 1700)
    assert 1 in ju.slashed and ju.trust[1] == 0.0
    # ущерб откатан: как у судьи, который клеветы не слышал
    assert ju.trust[0] == ctrl.trust[0]
    assert {k: v for k, v in ju.vsum.items() if v} ==         {k: v for k, v in ctrl.vsum.items() if v}


def test_v11_correct_refutation_of_a_lie_is_not_slashed():
    truth, au, ac, ju = _slash_trio(lambda k, now: True)
    tv = sn.truth_at(truth, "target_0.pos", 1000)
    lie = au.gen_own(1000, "target_0.pos", [tv[0] + 4000, tv[1] - 4000], None)
    ju.accept(lie.env, 1100)
    ac.accept(lie.env, 1100)
    ac.emit_validation(lie, "refuted", 1200)
    ju.accept(ac.fresh["svc"][-1].env, 1300)
    assert not ju.slashed and ju.trust[0] < 0.5


def test_v11_attacker_confirmations_cannot_frame_honest_refuter():
    for covers in (lambda k, now: True, lambda k, now: False):
        truth = sim.build_world(1, 4, 20_000)["truth"]
        liar, refuter, judge, confirmer = (
            mk_node(i, n=4, trust=0.5, truth=truth, slash=True) for i in range(4))
        judge.sensor.covers = covers
        tv = sn.truth_at(truth, "target_0.pos", 1000)
        lie = liar.gen_own(1000, "target_0.pos", [tv[0] + 4000, tv[1]], None)
        for node in (judge, refuter, confirmer):
            node.accept(lie.env, 1100)
        confirmer.emit_validation(lie, "confirmed", 1150)   # сговор с лжецом
        refuter.emit_validation(lie, "refuted", 1200)
        judge.accept(confirmer.fresh["svc"][-1].env, 1250)
        judge.accept(refuter.fresh["svc"][-1].env, 1300)
        assert not judge.slashed


def test_v11_small_run_deterministic():
    cfg = dict(n=3, duration_s=30, events_until_s=20, seed=7, loss=0.2,
               base_port=23200, batch_ms=500, slash=True)
    rec1, res1 = sim.run_once(cfg)
    rec2, res2 = sim.run_once(dict(cfg, base_port=23300))
    assert _fingerprint(res1) == _fingerprint(res2)
    assert rec1["metrics"]["transport_timeouts"] == 0
    assert rec1["metrics"]["converged"]


# v1.2 ---------------------------------------------------------------------

def _burst(author, n, hlc_ms=5000, seq0=1, key="drone_0.battery"):
    """n подписанных событий автора с одной секундой HLC (часы замёрзли)."""
    return [sn.make_env(author.k, "glyph",
                        {"key": key, "value": i, "hlc": [hlc_ms, i],
                         "node": author.me, "seq": seq0 + i})
            for i in range(n)]


def test_v12_frozen_hlc_burst_lost_by_hlc_limit_kept_by_seq_limit():
    """Дыра v1.1-Д1: 60 честных событий в одной секунде HLC."""
    a = mk_node(0)
    envs = _burst(a, 60)
    old, new = mk_reader(rate_mode="hlc"), mk_reader(rate_mode="seq")
    got_old = sum(1 for e in envs if old.accept(e, 5000))
    got_new = sum(1 for e in envs if new.accept(e, 5000))
    assert got_old == sn.Node.RATE["glyph"] and old.rejects["rate"] == 40
    assert 41 in old.seen_seq[0] or old.contig[0] >= 41   # и не дозапросится
    assert got_new == 60 and not new.rejects and new.contig[0] == 60


def test_v12_seq_beyond_allowance_rejected_for_good():
    a = mk_node(0)
    r = mk_reader(rate_mode="seq")
    cap = sn.Node.SEQ_RATE * (5000 // 1000 + 1) + sn.Node.SEQ_BURST
    ok, bad = _burst(a, 1, seq0=cap)[0], _burst(a, 1, seq0=cap + 1)[0]
    assert r.accept(ok, 5000) is not None
    assert r.accept(bad, 5000) is None and r.rejects["rate"] == 1
    assert cap + 1 in r.seen_seq[0]          # решение навсегда: не дозапрашивать


def test_v12_second_event_under_same_seq_rejected():
    """Без этой проверки флудер шлёт сколько угодно событий под одним seq."""
    a = mk_node(0)
    r = mk_reader(rate_mode="seq")
    e1 = _burst(a, 1, seq0=7)[0]
    e2 = sn.make_env(a.k, "glyph", {"key": "drone_0.battery", "value": 99,
                                    "hlc": [5000, 0], "node": 0, "seq": 7})
    assert r.accept(e1, 5000) is not None
    assert r.accept(e2, 5000) is None and r.rejects["equivocation"] == 1


def _glyph_k(author, seq, kseq, value=1, hlc_ms=5000):
    return sn.make_env(author.k, "glyph",
                       {"key": "drone_0.battery", "value": value,
                        "hlc": [hlc_ms, 0], "node": author.me,
                        "seq": seq, "kseq": kseq})


def test_v12_kseq_validation_stream_does_not_gag_glyphs():
    """Сценарий T7: журнальный seq убежал из-за опровержений, а свой номер
    glyph мал. Общий номер (seq) такой glyph отвергает, раздельный - нет."""
    a = mk_node(0)
    seq = sn.Node.SEQ_RATE * (5000 // 1000 + 1) + sn.Node.SEQ_BURST + 500
    env = _glyph_k(a, seq, 50)
    old, new = mk_reader(rate_mode="seq"), mk_reader(rate_mode="kseq")
    assert old.accept(env, 5000) is None and old.rejects["rate"] == 1
    assert new.accept(env, 5000) is not None and not new.rejects


def test_v12_kseq_reuse_rejected_and_kseq_above_seq_is_schema():
    a = mk_node(0)
    r = mk_reader(rate_mode="kseq")
    assert r.accept(_glyph_k(a, 10, 3), 5000) is not None
    assert r.accept(_glyph_k(a, 11, 3, value=2), 5000) is None
    assert r.rejects["equivocation"] == 1
    assert r.accept(_glyph_k(a, 12, 13), 5000) is None and r.rejects["schema"] == 1


def test_v12_kseq_wire_roundtrip_glyph_and_validation():
    a = _batch_node(0, rate_mode="kseq")
    a.gen_own(1000, "drone_0.battery", 97, None)
    g = a.gen_own(1010, "target_1.pos", [123456, 7], None)
    a.emit_validation(g, "refuted", 1030)
    a.close_batch()
    evs = a.fresh["push"] + a.fresh["svc"]
    assert {e.kind for e in evs} == {"glyph", "validation"}
    assert all("kseq" in e.env["body"] for e in evs)
    b = _batch_node(1, rate_mode="kseq")
    _deliver(b, a.encode_events(evs))
    for e in evs:
        assert sn.canon(b.events[e.eid].env) == sn.canon(e.env)
    assert not b.rejects


# v1.3 ---------------------------------------------------------------------

def test_v13_sensor_noise_is_deterministic_and_close_to_its_rate():
    truth = sim.build_world(1, 3, 60_000)["truth"]
    keys = sorted(k for k in truth if k.endswith(".status"))
    clean = sn.Sensor(truth, 1, 0)
    a, b = sn.Sensor(truth, 1, 0, noise=0.05), sn.Sensor(truth, 1, 0, noise=0.05)
    probes = [(k, sn.truth_at(truth, k, t), t) for k in keys for t in range(1000, 60_000, 500)]
    ra = [a.check(k, v, t, t + 100) for k, v, t in probes]
    assert ra == [b.check(k, v, t, t + 100) for k, v, t in probes]   # одно измерение на событие
    wrong = sum(x != clean.check(k, v, t, t + 100) for x, (k, v, t) in zip(ra, probes))
    assert 0.02 < wrong / len(probes) < 0.08 and wrong == a.flips


def _judge(k=3):
    truth = sim.build_world(1, 3, 20_000)["truth"]
    ju = mk_node(2, trust=0.5, truth=truth, slash=True, slash_window=k)
    return ju


def test_v13_window_ignores_sporadic_false_refutations():
    """Честный с шумом: 3 ложных опровержения на 200 проверенно-верных - не штраф;
    клеветник, опровергающий четверть и больше, - штраф. Окно 1 = v1.2."""
    ju = _judge(k=3)
    ju.true_seq = 1000
    assert not any(ju.strike(0) for _ in range(3))
    hits = []
    for i in range(60):
        ju.true_seq += 1
        hits.append(ju.strike(1))
    assert hits.index(True) == int(sn.Node.SLASH_SHARE * sn.Node.SLASH_WIN) - 1
    ju.true_seq += 10_000                       # старые удары вышли из окна
    assert not ju.strike(1)


def test_v13_slash_window_1_is_instant_and_k3_waits():
    for k, want in ((1, True), (3, False)):
        truth, au, ac, ju = _slash_trio(lambda key, now: True)
        ju.slash_k = k
        e = au.gen_own(1000, "sector_1.status",
                       sn.truth_at(truth, "sector_1.status", 1000), None)
        ju.accept(e.env, 1100)
        ac.accept(e.env, 1100)
        before = ju.trust[0]
        ac.emit_validation(e, "refuted", 1200)
        ju.accept(ac.fresh["svc"][-1].env, 1300)
        assert (1 in ju.slashed) is want
        assert ju.trust[0] == before            # ложное опровержение не учтено


def test_v13_refutation_needs_quorum_of_two():
    truth = sim.build_world(1, 4, 20_000)["truth"]
    au, r1, r2, ju = (mk_node(i, n=4, trust=1.0, truth=truth, refute_quorum=2)
                      for i in range(4))
    ju.sensor.covers = lambda k, now: False
    e = au.gen_own(1000, "sector_1.status", "hazard", None)
    for node in (ju, r1, r2):
        node.accept(e.env, 1100)
    r1.emit_validation(e, "refuted", 1200)
    ju.accept(r1.fresh["svc"][-1].env, 1300)
    assert ju.trust[0] == 1.0 and ju.admitted(e.eid)      # один голос - шум
    r2.emit_validation(e, "refuted", 1400)
    ju.accept(r2.fresh["svc"][-1].env, 1500)
    assert ju.trust[0] < 1.0 and not ju.admitted(e.eid)   # второй - действует


def test_v13_log_odds_admission_does_not_depend_on_vote_order():
    """Правило Линзы: решение - функция от множеств, порядок голосов не важен."""
    truth = sim.build_world(1, 5, 20_000)["truth"]
    au, v1, v2, v3 = (mk_node(i, n=5, trust=0.5, truth=truth) for i in range(4))
    e = au.gen_own(1000, "sector_1.status", "hazard", None)
    votes = []
    for node, verdict, t in ((v1, "refuted", 1200), (v2, "confirmed", 1300),
                             (v3, "refuted", 1400)):
        node.accept(e.env, 1100)
        node.emit_validation(e, verdict, t)
        votes.append(node.fresh["svc"][-1].env)
    seen = set()
    for order in ((0, 1, 2), (2, 1, 0), (1, 0, 2)):
        ju = mk_node(4, n=5, trust=0.5, truth=truth, admit_rule="log-odds")
        ju.sensor.covers = lambda k, now: False
        ju.accept(e.env, 1500)
        for i in order:
            ju.accept(votes[i], 1600)
        seen.add(ju.admitted(e.eid))
    assert seen == {False}                     # 1 подтверждение < 2 опровержений


def test_v13_liar_needs_a_share_not_a_count():
    """Автор с 3 событиями, опровергнутыми кворумом: из 1000 общих - шум, его новое
    событие принимается; из 20 - лжец, без подтверждения не принимается."""
    truth = sim.build_world(1, 3, 20_000)["truth"]
    au = mk_node(0, truth=truth)
    for shared_n, want in ((1000, True), (20, False)):
        ju = mk_node(2, truth=truth, admit_rule="log-odds", liar_k=3)
        ju.sensor.covers = lambda k, now: False
        e = au.gen_own(1000 + shared_n, "sector_1.status", "hazard", None)
        ju.accept(e.env, 2000)
        ju.refuted_q[0] = {"a", "b", "c"}
        ju.shared_n[0] = shared_n
        assert ju.admitted(e.eid) is want


# v1.4 ---------------------------------------------------------------------

def _votes_on(e, truth, plan):
    """plan: [(idx, verdict, t)] -> подписанные оценки разных узлов."""
    out = []
    for idx, verdict, t in plan:
        node = mk_node(idx, n=5, trust=0.5, truth=truth)
        node.accept(e.env, t - 50)
        node.emit_validation(e, verdict, t)
        out.append(node.fresh["svc"][-1].env)
    return out


def test_v14_canonical_vote_order_gives_one_answer_for_one_set():
    truth = sim.build_world(1, 5, 20_000)["truth"]
    au = mk_node(0, n=5, trust=0.5, truth=truth)
    e = au.gen_own(1000, "sector_1.status", "hazard", None)
    votes = _votes_on(e, truth, [(1, "refuted", 1200), (2, "confirmed", 1300),
                                 (3, "refuted", 1400)])
    got = set()
    for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
        ju = mk_node(4, n=5, trust=0.5, truth=truth, vote_order="canonical",
                     refute_quorum=2)
        ju.sensor.covers = lambda k, now: False
        ju.accept(e.env, 1500)
        for i in order:
            ju.accept(votes[i], 1600)
        ju.flush_votes()
        got.add((tuple(ju.trust), round(ju.vsum.get(e.eid, 0.0), 9), ju.admitted(e.eid)))
    assert len(got) == 1


def test_v14_sensor_filter_off_counts_refutation_my_sensor_disputes():
    truth = sim.build_world(1, 5, 20_000)["truth"]
    au = mk_node(0, n=5, trust=1.0, truth=truth)
    e = au.gen_own(1000, "sector_1.status",
                   sn.truth_at(truth, "sector_1.status", 1000), None)
    votes = _votes_on(e, truth, [(1, "refuted", 1200), (2, "refuted", 1300)])
    for flt, want in ((True, True), (False, False)):
        ju = mk_node(4, n=5, trust=1.0, truth=truth, slash=True, slash_window=3,
                     refute_quorum=2, sensor_filter=flt)
        ju.sensor.covers = lambda k, now: True      # мой датчик: значение верное
        ju.accept(e.env, 1500)
        for v in votes:
            ju.accept(v, 1600)
        assert ju.admitted(e.eid) is want           # off: кворум опровержений действует
        assert not ju.slashed


# v2.0-anis ----------------------------------------------------------------

import apsa_engine as apsa                                  # noqa: E402
import tombstone_registry as tomb                           # noqa: E402


def test_v20_tz_formula_buries_one_error_without_warmup_but_share_rule_does_not():
    """Канон ТЗ: A = 1, F = 0 -> S = 1.5 >= 0.70 -> INTENTIONAL (дыра формулы).
    Правило доли с прогревом не хоронит ни одну ошибку и ни 5 % шума."""
    assert apsa.classify_behavior(0, 1) == apsa.INTENTIONAL
    a = apsa.APSASanitaryAudit(rule="share")
    for _ in range(200):
        a.saw_true(1000)
    for _ in range(10):                          # 5 % окна - шум честного
        a.false_refutation(1, 1000)
    assert a.verdict(1, 9000)[0] != apsa.INTENTIONAL
    for _ in range(50):                          # 30 % окна - клеветник
        a.false_refutation(2, 1000)
    assert a.verdict(2, 9000)[0] == apsa.INTENTIONAL


def test_v20_tz_formula_threshold_falls_with_window_size():
    """При S = 1.5*A^2/(W-A) >= 0.7 доля, при которой хоронят, падает с окном."""
    def first_a(w):
        return next(a for a in range(1, w) if apsa.s_index(a, w - a) >= 0.70)
    assert first_a(200) / 200 > first_a(500) / 500 > first_a(2000) / 2000


def test_v20_two_phase_quarantine_then_tombstone():
    r = tomb.TombstoneRegistry(5, 0, two_phase=True)
    assert not r.intentional("acc", 3, pos=100, gap=20, now_ms=1)
    assert r.is_suspect(3) and not r.is_dead(3)
    assert not r.intentional("acc", 3, pos=110, gap=20, now_ms=2)   # окно почти то же
    assert r.intentional("acc", 3, pos=120, gap=20, now_ms=3) and r.is_dead(3)


def test_v20_foreign_tombstone_needs_quorum_of_two_live_issuers():
    r = tomb.TombstoneRegistry(5, 0, quorum=2)
    assert not r.foreign(1, 2, 10) and not r.is_dead(2)              # одиночное - подстава
    assert r.foreign(4, 2, 20) and r.is_dead(2)
    r2 = tomb.TombstoneRegistry(5, 0, quorum=2, thymus="literal", air=1.0)
    r2.dead[4] = 0                                                   # похороненный не голосует
    r2.foreign(4, 2, 10)
    assert not r2.foreign(1, 2, 20) and not r2.is_dead(2)


def test_v20_thymus_bft_allows_one_of_five_literal_allows_first_only():
    r = tomb.TombstoneRegistry(5, 0, two_phase=False, thymus="bft")
    assert r.intentional("acc", 1, 0, 0, 1)
    assert not r.intentional("acc", 2, 0, 0, 2) and r.blocked == 1
    lit = tomb.TombstoneRegistry(5, 0, two_phase=False, thymus="literal", air=0.01)
    assert lit.intentional("acc", 1, 0, 0, 1)        # AIR до добавления 0/5 <= 0.01
    assert not lit.intentional("acc", 4, 0, 0, 2)    # дальше никого, даже клеветника


def test_v20_buried_author_rejected_at_input():
    truth = sim.build_world(1, 3, 20_000)["truth"]
    au = mk_node(0, truth=truth)
    ju = mk_node(2, truth=truth, arbiter="anis")
    e1 = au.gen_own(1000, "sector_1.status", "hazard", None)
    assert ju.accept(e1.env, 1100) is not None
    ju.tombs.dead[0] = 1200
    ju.on_buried(0)
    e2 = au.gen_own(1300, "sector_2.status", "hazard", None)
    assert ju.accept(e2.env, 1400) is None and ju.rejects["tombstoned"] == 1
    assert not ju.admitted(e1.eid)


def test_v20_thymus_blocks_second_tombstone_in_node_contour():
    """Негативный тест (n = 5): Тимус встроен в контур узла. Второе надгробие -
    своим решением APSA или чужим кворумом - отказ, в реестр не попадает, наружу
    не рассылается. Лимит BFT: (n - 1) // 3 = 1."""
    truth = sim.build_world(1, 5, 20_000)["truth"]
    ju = mk_node(0, n=5, truth=truth, arbiter="anis")
    ju.clock = 1000
    log = lambda m: print("[thymus] " + m)

    # 1. Первое надгробие, двухфазно: SUSPICIOUS, затем TOMBSTONE после окна.
    ju.bury("acc", 3, 0.40, 20, 100)
    assert ju.tombs.is_suspect(3) and not ju.tombs.is_dead(3)
    ju.bury("acc", 3, 0.40, 20, 120)
    assert ju.tombs.is_dead(3) and 3 in ju.slashed
    sent = len(ju.tomb_events)
    log("1-е надгробие узлу 3: принято, AIR = %d/5 = %.2f, разослано %d"
        % (len(ju.tombs.dead), len(ju.tombs.dead) / 5, sent))

    # 2. Второе надгробие своим решением: обе фазы пройдены - Тимус отказывает.
    ju.bury("acc", 4, 0.40, 20, 200)
    ju.bury("acc", 4, 0.40, 20, 220)
    assert not ju.tombs.is_dead(4) and 4 not in ju.slashed
    assert ju.tombs.blocked == 1 and len(ju.tomb_events) == sent
    log("2-е надгробие узлу 4 (своё): ОТКАЗ, AIR был бы 2/5 = 0.40 > 0.20, "
        "блокировок %d, в реестре %s" % (ju.tombs.blocked, sorted(ju.tombs.dead)))

    # 3. Второе надгробие чужим кворумом (2 живых узла) - тоже отказ.
    assert not ju.tombs.foreign(1, 2, 300)
    assert not ju.tombs.foreign(4, 2, 310)          # кворум набран, Тимус - нет
    assert not ju.tombs.is_dead(2) and ju.tombs.blocked == 2
    log("2-е надгробие узлу 2 (кворум чужих 1+4): ОТКАЗ, блокировок %d, "
        "в реестре %s" % (ju.tombs.blocked, sorted(ju.tombs.dead)))

    # 4. Отказ не ломает приём: события непохороненных узлов 2 и 4 принимаются.
    for a in (2, 4):
        e = mk_node(a, n=5, truth=truth).gen_own(1500, "sector_1.status", "hazard", None)
        assert ju.accept(e.env, 1600) is not None and ju.admitted(e.eid)
    log("события узлов 2 и 4 после отказа принимаются, узел 3 - нет")


def test_v21_event_window_ignores_late_burst_of_old_refutations():
    """Дыра v2.0: в тишине окно судьи стоит, PULL досылает пачку старых шумовых
    опровержений - по времени прихода доля 30 %, честный хоронится. По событиям
    окна старые события не считаются."""
    def judge(kind):
        a = apsa.APSASanitaryAudit(rule="share", win_kind=kind)
        for i in range(60):
            a.saw_true(1000, "old%d" % i)        # проверены давно
        for i in range(200):
            a.saw_true(2000, "new%d" % i)        # окно - последние 200
        for i in range(60):                      # пачка поздних опровержений старого
            a.false_refutation(7, 575_000, "old%d" % i)
        return a.verdict(7, 575_000)[0]
    assert judge("arrival") == apsa.INTENTIONAL
    assert judge("event") != apsa.INTENTIONAL
    a = apsa.APSASanitaryAudit(rule="share", win_kind="event")
    for i in range(200):
        a.saw_true(1000, "e%d" % i)
    for i in range(60):                          # клеветник: 30 % событий окна
        a.false_refutation(4, 2000, "e%d" % i)
    assert a.verdict(4, 9000)[0] == apsa.INTENTIONAL


def test_v22_fast_refute_quota_and_single_event_batch_verifies():
    """v2.2: опровержение уходит сразу пакетом из одного события, не больше
    квоты за 500 мс; остальное - в обычный пакет. Получатель проверяет подпись
    пакета из одного листа так же, как обычного."""
    a = _batch_node(0, fast_refute=1)
    g1 = a.gen_own(1000, "target_1.pos", [100, 7], None)
    g2 = a.gen_own(1010, "target_2.pos", [200, 7], None)
    g3 = a.gen_own(1020, "target_1.pos", [300, 7], None)
    a.emit_validation(g1, "refuted", 1100)       # быстрая полоса
    a.emit_validation(g2, "refuted", 1200)       # квота 1 за 500 мс исчерпана
    a.emit_validation(g3, "refuted", 1700)       # окно сдвинулось - снова быстро
    assert a.fast_sent == 2
    fast = [e for e in a.fresh["push"] if e.kind == "validation"]
    assert len(fast) == 2 and all(a.batches[e.bref[0]]["n"] == 1 for e in fast)
    assert any(e.kind == "validation" for e in a.outbox)   # второе ждёт пакета
    b = _batch_node(1)
    _deliver(b, a.encode_events(fast))
    assert all(e.eid in b.events for e in fast) and not b.rejects
    off = _batch_node(0)
    off.emit_validation(off.gen_own(1000, "target_1.pos", [1, 7], None), "refuted", 1100)
    assert off.fast_sent == 0 and not any(e.kind == "validation" for e in off.fresh["push"])


# ------------------------------------------------------------ v3: Марзулло

import marzullo as mz                                        # noqa: E402


def test_v3_marzullo_judge_matches_brute_force_on_random_1d():
    """judge = критерий Марзулло: истина в точке, покрытой >= m - f
    интервалами. Сверка с перебором целых точек на случайных отрезках."""
    rng = random.Random(7)
    for _ in range(3000):
        s = rng.randint(1, 4)
        iv = []
        for _ in range(s + 1):
            c, h = rng.randint(0, 40), rng.randint(0, 6)
            iv.append([(c - h, c + h)])
        e, src = iv[0], iv[1:]
        pts = range(-10, 51)
        cover = lambda p, b: b[0][0] <= p <= b[0][1]
        m = s + 1
        if s < 2:
            want = mz.UNDECIDED
        elif any(cover(p, e) and 1 + sum(cover(p, b) for b in src) >= m - 1 for p in pts):
            want = mz.CONSISTENT
        elif any(sum(cover(p, b) for b in src) >= m - 1 for p in pts):
            want = mz.REFUTED
        else:
            want = mz.UNDECIDED
        assert mz.judge(e, src, 1) == want, (e, src)


def test_v3_marzullo_box_tolerates_one_faulty_source():
    e_true, e_lie = mz.box([1000, 1000], 150), mz.box([5000, 5000], 150)
    honest = [mz.box([1100, 900], 1750), mz.box([2200, 1300], 1750)]
    gross = mz.box([4600, -2500], 1750)
    assert mz.judge(e_lie, honest, 1) == mz.REFUTED
    assert mz.judge(e_true, honest, 1) == mz.CONSISTENT
    assert mz.judge(e_true, honest + [gross], 1) == mz.CONSISTENT   # один сбойный
    assert mz.judge(e_lie, honest[:1], 1) == mz.UNDECIDED           # m < 3


def test_v3_vote_with_measurement_wire_roundtrip():
    a = _batch_node(0, sensor_model="interval")
    g = a.gen_own(1000, "target_1.pos", [123456, 7], None)
    s = a.gen_own(1010, "sector_1.status", "hazard", None)
    a.emit_validation(g, "refuted", 1030)
    a.emit_validation(s, "refuted", 1040)
    a.close_batch()
    evs = a.fresh["push"]
    notes = [e.env["body"]["evidence"]["note"] for e in evs if e.kind == "validation"]
    assert isinstance(notes[0], list) and len(notes[0]) == 2 and notes[1] is None
    b = _batch_node(1)
    _deliver(b, a.encode_events(evs))
    for e in evs:
        assert sn.canon(b.events[e.eid].env) == sn.canon(e.env)


def test_v3_interval_sensor_is_deterministic_and_gross_error_shifts_reading():
    truth = sim.build_world(1, 5, 20_000)["truth"]
    s = sn.Sensor(truth, 1, 2, noise=0.5, model="interval")
    tv = sn.truth_at(truth, "target_0.pos", 5000)
    seen = {"ok": 0, "gross": 0}
    for v in range(200):
        val = [tv[0] + v % 7, tv[1]]
        r1, r2 = s.reading("target_0.pos", val, 5000), s.reading("target_0.pos", val, 5000)
        assert r1 == r2
        d = max(abs(r1[0] - tv[0]), abs(r1[1] - tv[1]))
        if s.gross("target_0.pos", val, 5000):
            assert 2950 <= d <= 5050
            seen["gross"] += 1
        else:
            assert d <= 50
            seen["ok"] += 1
    assert seen["ok"] > 50 and seen["gross"] > 50


def _mz_world():
    truth = sim.build_world(1, 5, 20_000)["truth"]
    k = "target_1.pos"
    t = next(t for t in range(2000, 20_000, 100)
             if sn.truth_at(truth, k, t - 1200) == sn.truth_at(truth, k, t + 1200))
    return truth, k, t, sn.truth_at(truth, k, t)


def test_v3_marzullo_admission_blocks_pos_lie_on_arrival_and_ignores_own_sensor():
    """Ложь о координате при двух свежих заявках других авторов не входит в
    картину сразу, без голосов. Решение одинаково у узлов с разными датчиками
    (один - со 100 % сбоев) и не зависит от порядка прихода заявок."""
    truth, k, t, tv = _mz_world()
    authors = [mk_node(i, n=5, truth=truth) for i in range(5)]
    h1 = authors[1].gen_own(t - 300, k, [tv[0] + 40, tv[1] - 30], None)
    h2 = authors[2].gen_own(t - 200, k, [tv[0] - 20, tv[1] + 10], None)
    lie = authors[4].gen_own(t, k, [tv[0] + 3500, tv[1] - 4200], None)
    hon = authors[3].gen_own(t + 50, k, [tv[0] + 10, tv[1] + 10], None)
    out = []
    for idx, noise, order in ((0, 0.0, (h1, h2, lie, hon)), (0, 1.0, (hon, lie, h2, h1))):
        ju = mk_node(idx, n=5, truth=truth, arbiter="anis", marzullo=1,
                     sensor_model="interval", sensor_noise=noise)
        for e in order:
            assert ju.accept(e.env, t + 500) is not None
        out.append((ju.admitted(lie.eid), ju.admitted(hon.eid)))
    assert out == [(False, True), (False, True)]


def test_v3_marzullo_off_keeps_v22_admission():
    truth, k, t, tv = _mz_world()
    authors = [mk_node(i, n=5, truth=truth) for i in range(5)]
    evs = [authors[1].gen_own(t - 300, k, list(tv), None),
           authors[2].gen_own(t - 200, k, list(tv), None),
           authors[4].gen_own(t, k, [tv[0] + 3500, tv[1] - 4200], None)]
    ju = mk_node(0, n=5, truth=truth, arbiter="anis")
    for e in evs:
        ju.accept(e.env, t + 500)
    assert ju.admitted(evs[2].eid)          # без голосов v2.2 пускает


def test_v3_status_prior_defers_change_until_corroborated():
    """Вариант B: новый статус против согласия двух других авторов ждёт;
    вторая заявка того же значения другим автором снимает отсрочку."""
    truth = sim.build_world(1, 5, 20_000)["truth"]
    k = "sector_3.status"
    au = [mk_node(i, n=5, truth=truth) for i in range(5)]
    ju = mk_node(0, n=5, truth=truth, arbiter="anis", status_prior=True)
    ju.sensor.covers = lambda key, now: False       # судья сам не голосует
    for e in (au[1].gen_own(4000, k, "clear", None), au[2].gen_own(4100, k, "clear", None)):
        ju.accept(e.env, 4300)
    new = au[3].gen_own(4400, k, "hazard", None)
    ju.accept(new.env, 4500)
    assert not ju.admitted(new.eid)
    ju.accept(au[2].gen_own(4700, k, "hazard", None).env, 4800)
    assert ju.admitted(new.eid)


def test_v3_validator_suppresses_own_gross_error():
    """Мой датчик со 100 % сбоев против двух согласных заявок - опровержение
    не отправляется, счётчик подавленных растёт."""
    truth, k, t, tv = _mz_world()
    au = [mk_node(i, n=5, truth=truth) for i in range(5)]
    ju = mk_node(0, n=5, truth=truth, arbiter="anis", marzullo=1,
                 sensor_model="interval", sensor_noise=1.0)
    ju.sensor.covers = lambda key, now: True
    for i, dt in ((1, -300), (2, -200)):
        ju.accept(au[i].gen_own(t + dt, k, [tv[0] + 5 * i, tv[1]], None).env, t)
    e = au[3].gen_own(t, k, [tv[0], tv[1] + 20], None)
    sent = ju.validations_sent
    ju.accept(e.env, t + 100)                       # приём сам зовёт проверку
    assert ju.mz_suppressed == 1 and ju.validations_sent == sent


# v3.0 release -------------------------------------------------------------

# Отпечаток прогона ниже, снятый на main f620f41 (v2.2) и совпавший с v3 при flip/0.
V22_ANIS_T4_FP = "d3870e0858f7b78b7675e4fe16efb53cf9cb24ddacdaec8a9537443b31a84a24"


def test_v3_flip_and_marzullo_off_reproduce_v22_byte_for_byte():
    cfg = dict(n=5, duration_s=30, events_until_s=20, seed=2, loss=0.05,
               attack="T4", base_port=23400, arbiter="anis", rebut=True,
               liar_vote=True, pull_priority="validations", pull_quiet_cap=96,
               sensor_noise=0.05, sensor_model="flip", marzullo=0)
    rec, res = sim.run_once(cfg)
    fp = hashlib.sha256(repr(_fingerprint(res)).encode()).hexdigest()
    assert fp == V22_ANIS_T4_FP
    assert rec["metrics"]["converged"]


def test_v3_cli_defaults_turn_on_marzullo_only_for_anis():
    assert sim._v3_defaults("anis")[:2] == ("interval", 1)
    assert sim._v3_defaults("anis-literal")[:2] == ("interval", 1)
    assert sim._v3_defaults("scalar")[:2] == ("flip", 0)
    assert sim._v3_defaults("anis", "flip", 0)[:2] == ("flip", 0)    # явное не трогаем


def test_v31_cli_slot_votes_default_only_for_anis():
    assert sim._v3_defaults("anis")[2] is True
    assert sim._v3_defaults("anis-literal")[2] is True
    assert sim._v3_defaults("scalar")[2] is False
    assert sim._v3_defaults("anis", slot_votes=False)[2] is False     # --no-slot-votes
    assert sim._v3_defaults("scalar", slot_votes=True)[2] is True     # --slot-votes
    assert sim.DEFAULTS["slot_votes"] is False   # run_once без флага = v3.0 (старые конфиги)


# v3.1 ---------------------------------------------------------------------

# Отпечаток того же прогона на v3.0 (7a135fd, интервалы + Марзулло, без досылки).
V30_ANIS_T4_FP = "8a0ec37411256d1511bb11d7f05b62fcc4ad522067c42721a741600325a22d2d"


def test_v31_without_slot_votes_reproduces_v30_byte_for_byte():
    cfg = dict(n=5, duration_s=30, events_until_s=20, seed=2, loss=0.05,
               attack="T4", base_port=23500, arbiter="anis", rebut=True,
               liar_vote=True, pull_priority="validations", pull_quiet_cap=96,
               sensor_noise=0.05, sensor_model="interval", marzullo=1, slot_votes=False)
    rec, res = sim.run_once(cfg)
    assert hashlib.sha256(repr(_fingerprint(res)).encode()).hexdigest() == V30_ANIS_T4_FP


def _slot_dispute(slot_votes):
    truth = sim.build_world(1, 5, 20_000)["truth"]
    k = "sector_1.status"
    e1 = mk_node(1, n=5, truth=truth).gen_own(1000, k, sn.STATUSES[0], None)
    e2 = mk_node(2, n=5, truth=truth).gen_own(1500, k, sn.STATUSES[1], None)
    v1 = _votes_on(e1, truth, [(4, "confirmed", 1600)])
    v2 = _votes_on(e2, truth, [(3, "refuted", 1600)])
    r = mk_node(0, n=5, truth=truth, arbiter="anis", slot_votes=slot_votes)
    for env in [e1.env, e2.env] + v1 + v2:
        r.accept(env, 1700)
    r.record_picture(17)
    ki = r.keys_list.index(k)
    assert r.win_full[ki] == e2.eid                 # у меня победитель e2, у соседа e1
    full = {str(a): [r.contig[a], "%x" % sum(1 << (s - r.contig[a] - 1)
                                             for s in r.held[a] if s > r.contig[a])]
            for a in range(5)}                      # у соседа тот же журнал
    no_votes = dict(full, **{"3": [0, "0"], "4": [0, "0"]})   # нет голосов узлов 3 и 4
    return r, ki, e1, e2, {env["id"] for env in v1 + v2}, full, no_votes


def test_v31_slot_dispute_sends_only_missing_votes_on_both_disputed_events():
    r, ki, e1, e2, votes, full, no_votes = _slot_dispute(True)
    sent = {v.eid for v in r._slot_dispute_votes(ki, e2.eid, e1.eid[7:13], no_votes)}
    assert sent == votes                            # голоса по обоим спорным
    assert r._slot_dispute_votes(ki, e2.eid, e1.eid[7:13], full) == []   # дублей нет
    sl = "".join(e1.eid[7:13] if i == ki else (r.win_full[i][7:13] if i in r.win_full else "------")
                 for i in range(len(r.keys_list)))
    r0 = _slot_dispute(False)[0]                    # v3.0: только победитель
    out = {ev.eid for ev in r0.answer_body({"sl": sl, "vv": full})}
    assert e2.eid in out and not (votes & out)


# v4.0 ---------------------------------------------------------------------

def test_v40_default_world_is_the_old_steps_world():
    a = sim.build_world(3, 5, 60_000)
    b = sim.build_world(3, 5, 60_000, world="steps")
    assert a == b


def test_v40_kinematic_world_moves_targets_smoothly():
    w = sim.build_world(1, 5, 120_000, world="kinematic")
    for k in sn.shared_keys():
        times, vals = w["truth"][k]
        if k.endswith(".pos"):
            assert times[0] == 0 and all(b - a == 100 for a, b in zip(times, times[1:]))
            steps = [max(abs(q[0] - p[0]), abs(q[1] - p[1])) for p, q in zip(vals, vals[1:])]
            assert max(steps) <= 71 and min(steps) > 0      # 150-700 ед/с, шаг 100 мс
        else:
            assert len(times) > 1                           # статусы по-прежнему меняются


def test_v40_t8_lie_is_inside_marzullo_and_outside_sensor_tolerance():
    truth = sim.build_world(1, 5, 120_000, world="kinematic")["truth"]
    liar = mk_node(4, n=5, truth=truth, attack="T8")
    t = sn.ATTACK_START_MS + 1000
    liar.generate(t, "target_2.pos", [0, 0], None)
    ev = liar.events[max(liar.events, key=lambda e: liar.events[e].hlc[0])]
    tv = sn.truth_at(truth, "target_2.pos", t)
    off = [abs(ev.value[i] - tv[i]) for i in (0, 1)]
    assert all(600 <= o <= 1500 for o in off)
    assert all(o > sn.Sensor.POS_TOL for o in off)
    assert all(o < sn.MZ_H + sn.MZ_D for o in off)
    assert liar.attack_log[-1][0] == "T8"


def test_v40_predictor_fits_motion_and_survives_one_liar_author():
    import kinematics as kin
    line = lambda t: (10_000 + 0.4 * t, 20_000 - 0.25 * t)       # 400 и -250 ед/с
    cl = [(t, a, line(t)) for a in range(4) for t in range(1000 + 230 * a, 4000, 900)]
    px, py = kin.predict(cl, 4000)
    assert abs(px - line(4000)[0]) < 1 and abs(py - line(4000)[1]) < 1
    bad = [(t, 0, (x + 1200, y - 900)) if a == 0 else (t, a, (x, y)) for t, a, (x, y) in cl]
    px, py = kin.predict(bad, 4000)                              # автор 0 лжёт
    assert abs(px - line(4000)[0]) < 1 and abs(py - line(4000)[1]) < 1
    assert kin.predict([c for c in cl if c[1] == 1], 4000) is None   # один автор - нет прогноза
    assert kin.predict(cl[:3], 4000) is None                         # мало точек


def _gate_node(**kw):
    truth = sim.build_world(1, 5, 120_000, world="kinematic")["truth"]
    k, t = "target_1.pos", 70_000
    r = mk_node(0, n=5, truth=truth, arbiter="anis", sensor_model="interval", marzullo=1,
                traj_gate=True, drifts=[0] * 5, **kw)
    for a in (1, 2, 3):
        au = mk_node(a, n=5, truth=truth, drifts=[0] * 5)
        for tm in range(t - 3600 + 300 * a, t, 1000):
            r.accept(au.gen_own(tm, k, list(sn.truth_at(truth, k, tm)), None).env, tm + 50)
    liar = mk_node(4, n=5, truth=truth, drifts=[0] * 5)
    tv = sn.truth_at(truth, k, t)
    return r, liar, k, t, tv


def test_v40_traj_gate_refutes_t8_keeps_honest_and_fails_safe():
    r, liar, k, t, tv = _gate_node()
    lie = liar.gen_own(t, k, [tv[0] + 1100, tv[1] - 900], None)
    hon = liar.gen_own(t + 10, k, [tv[0] + 30, tv[1] - 20], None)
    for e in (lie, hon):
        r.accept(e.env, t + 60)
    assert r._traj_refutes(r.events[lie.eid]) is True
    assert r._traj_refutes(r.events[hon.eid]) is False
    # свой грубый сбой: датчик далёк и от заявки, и от прогноза - решения нет (v3.1)
    r2, liar2, k, t, tv = _gate_node()
    r2.sensor.reading = lambda key, value, h: [tv[0] + 4000, tv[1] + 3500]
    e = liar2.gen_own(t, k, [tv[0] + 30, tv[1] - 20], None)
    r2.accept(e.env, t + 60)
    assert r2._traj_refutes(r2.events[e.eid]) is False


def test_v40_traj_gate_needs_the_kinematic_world():
    """Мир steps: дрейф часов +-2 с, шлюз не включается (иначе опровергает
    честные заявки - 143 980 на сетке 490)."""
    rec, res = sim.run_once(dict(n=3, duration_s=8, events_until_s=6, seed=4, base_port=23700,
                                 arbiter="anis", sensor_model="interval", marzullo=1,
                                 traj_gate=True))
    assert rec["metrics"]["traj_checks"] == 0


def test_v40_tombstones_in_pull_back_off_per_peer():
    r = mk_node(0, n=5, tomb_backoff=True)
    r.tomb_events = [sn.Ev("t" * 64, "tombstone", 0, {}, 0, {})]
    sent = []
    for ms in range(0, 40_000, 500):
        r.clock = ms
        if r._tombs_for(1):
            sent.append(ms)
    assert sent == [0, 1000, 3000, 7000, 15_000, 31_000]      # отступ 1, 2, 4, 8, 16 с
    r.clock = 31_500
    assert r._tombs_for(2)                                       # свой счёт у каждого соседа
    plain = mk_node(0, n=5)
    plain.tomb_events = r.tomb_events
    assert plain._tombs_for(1) == plain._tombs_for(1) == r.tomb_events   # v3.1: в каждом ответе


def test_v40_claim_truth_is_judged_at_the_claim_moment():
    truth = sim.build_world(1, 5, 120_000, world="kinematic")["truth"]
    r = mk_node(0, n=5, truth=truth, world="kinematic")
    au = mk_node(1, n=5, truth=truth)
    k, t = "target_0.pos", 70_000
    tv = sn.truth_at(truth, k, t)
    ok = au.gen_own(t, k, [tv[0] + 40, tv[1] - 40], None)
    lie = au.gen_own(t + 5, k, [tv[0] + 900, tv[1] + 900], None)
    assert r._claim_true(ok) and not r._claim_true(lie)
    assert r.point_truth and not mk_node(0, n=5).point_truth
