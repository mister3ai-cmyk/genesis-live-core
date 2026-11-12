# Business Source License 1.1 - см. research/LICENSE
"""GEP v0.1 Lite: кадр, порча байтов, повтор, метка из будущего, карантин
источника и автора, отступ надгробий. Связка со шлюзом - если собран
librt_core.so (WSL / Linux)."""
import os
import sys

import pytest

pytest.importorskip("cryptography")

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
if RT not in sys.path:
    sys.path.insert(0, RT)

import gep_lite as gl                                        # noqa: E402

pytestmark = pytest.mark.concept

KEYS = {i: gl.keypair(i) for i in range(6)}
PUB = {i: pk for i, (sk, pk) in KEYS.items()}


def _claim(a=1, seq=1, t=1_000_000, xyz=(1.0, 2.0, 3.0), target=0):
    return gl.pack(KEYS[a][0], gl.CLAIM, a, target, seq, t, *xyz)


def test_roundtrip_all_types_and_sizes():
    sk = KEYS[2][0]
    cases = [(gl.CLAIM, (1.5, -2.25, 1e5), 110), (gl.REFUTE, (3, gl.REASON_TRAJ_VETO, 0, 77), 94),
             (gl.TOMBSTONE, (4, gl.REASON_TRAJ_VETO, 5, 123), 98), (gl.SYNC_PING, (10, -20), 102)]
    for ftype, payload, size in cases:
        fr = gl.pack(sk, ftype, 2, 7, 42, -5, *payload)
        assert len(fr) == size
        f = gl.unpack(fr, PUB)
        assert (f["type"], f["author"], f["target"], f["seq"], f["t_us"]) == (ftype, 2, 7, 42, -5)
        assert f["payload"] == payload


def test_every_single_byte_flip_is_rejected():
    fr = _claim()
    for i in range(len(fr)):
        for bit in (0x01, 0x80):
            bad = bytearray(fr)
            bad[i] ^= bit
            with pytest.raises(gl.FrameError):
                gl.unpack(bytes(bad), PUB)


def test_format_errors_have_specific_reasons():
    fr = _claim()
    cases = {"short": fr[:50], "magic": b"XX" + fr[2:], "version": fr[:2] + b"\x02" + fr[3:],
             "type": fr[:3] + b"\x09" + fr[4:], "length": fr + b"\x00",
             "author": gl.pack(gl.keypair(99)[0], gl.CLAIM, 99, 0, 1, 0, 0.0, 0.0, 0.0)}
    for reason, bad in cases.items():
        with pytest.raises(gl.FrameError) as e:
            gl.unpack(bad, PUB)
        assert e.value.reason == reason
    forged = gl.pack(KEYS[3][0], gl.CLAIM, 1, 0, 1, 0, 0.0, 0.0, 0.0)   # подписал 3 от имени 1
    with pytest.raises(gl.FrameError) as e:
        gl.unpack(forged, PUB)
    assert e.value.reason == "signature"


def test_forgery_quarantines_source_for_a_series_and_a_ttl_not_the_named_author():
    n = gl.Node(0, KEYS[0][0], PUB, src_viol_n=3, src_viol_window_us=10_000_000, src_ttl_us=30_000_000)
    bad = "10.0.0.66:9000"
    forged = lambda s: gl.pack(KEYS[3][0], gl.CLAIM, 1, 0, s, 1000, 0.0, 0.0, 0.0)
    assert n.receive(forged(1), bad, 1_000_000)[0] == "reject_signature"
    assert bad not in n.q_sources                         # один битый кадр - не повод
    assert n.receive(_claim(1, 1, 1500), "10.0.0.1:9000", 1_000_000)[0] == "pass"
    n.receive(forged(2), bad, 2_000_000)
    n.receive(forged(3), bad, 3_000_000)                  # третий за 10 с - карантин
    assert bad in n.q_sources and 1 not in n.q_authors    # автор 1 цел
    assert n.receive(_claim(1, 2, 3_500_000), bad, 3_500_000)[0] == "drop_source"
    assert n.receive(_claim(1, 3, 33_000_000), bad, 33_000_001)[0] == "pass"   # срок вышел
    assert bad not in n.q_sources
    m = gl.Node(0, KEYS[0][0], PUB)                       # нарушения реже окна - карантина нет
    for k in range(5):
        m.receive(forged(10 + k), bad, k * 11_000_000)
    assert bad not in m.q_sources


def test_replay_and_future_timestamp():
    n = gl.Node(0, KEYS[0][0], PUB, max_skew_us=50_000)
    assert n.receive(_claim(1, 5, 1_000_000), "s1", 1_000_100)[0] == "pass"
    assert n.receive(_claim(1, 5, 1_000_000), "s1", 1_000_200)[0] == "replay"   # точный дубль
    assert n.receive(_claim(1, 4, 1_000_000), "s1", 1_000_200)[0] == "pass"     # обогнан в сети - принять
    assert n.receive(_claim(1, 4, 1_000_000), "s1", 1_000_200)[0] == "replay"   # но только один раз
    assert n.receive(_claim(1, 6, 900_000), "s1", 1_000_300)[0] == "pass"       # задержка доставки - норма
    assert n.receive(_claim(1, 200, 900_000), "s1", 1_000_300)[0] == "pass"
    assert n.receive(_claim(1, 100, 900_000), "s1", 1_000_300)[0] == "replay"   # старше окна 64
    # метка из будущего: кадр отброшен, карантин - только после льготного окна
    assert n.receive(_claim(2, 1, 1_100_000), "s2", 1_000_000)[0] == "future_drop"
    assert n.q_authors == {}
    assert n.receive(_claim(2, 2, 6_100_000), "s2", 6_000_000)[0] == "quarantine_future"
    assert n.q_authors == {2: gl.REASON_FUTURE}
    assert n.receive(_claim(2, 3, 6_000_000), "s2", 6_000_000)[0] == "drop_quarantined"
    tomb = gl.unpack(n.tombs[0][0], PUB)
    assert tomb["type"] == gl.TOMBSTONE and tomb["payload"][0] == 2


def test_future_strike_resets_when_author_recovers_and_needs_my_lock():
    n = gl.Node(0, KEYS[0][0], PUB, max_skew_us=50_000)
    assert n.receive(_claim(2, 1, 1_100_000), "s2", 1_000_000)[0] == "future_drop"
    assert n.receive(_claim(2, 2, 3_000_000), "s2", 3_000_000)[0] == "pass"    # выправился
    assert n.receive(_claim(2, 3, 7_100_000), "s2", 7_000_000)[0] == "future_drop"  # счёт заново
    cs = gl.ClockSync(0, range(6))                            # мои часы не синхронизированы
    m = gl.Node(0, KEYS[0][0], PUB, clock=cs)
    assert m.receive(_claim(2, 1, 1_100_000), "s2", 1_000_000)[0] == "future_drop"
    assert m.receive(_claim(2, 2, 9_100_000), "s2", 9_000_000)[0] == "future_drop"
    assert m.q_authors == {}                                  # «будущее» может быть мной


# --- SYNC_PING / PONG -------------------------------------------------------

def test_ntp_four_timestamps():
    off, d_go, d_back, proc = 37_000, 4_000, 26_000, 300      # сосед впереди на 37 мс
    t1 = 1_000_000
    t2 = t1 + d_go + off
    t3 = t2 + proc
    t4 = t3 - off + d_back
    delta, theta = gl.ntp(t1, t2, t3, t4)
    assert delta == d_go + d_back
    assert theta == off + (d_go - d_back) / 2                 # ошибка - половина асимметрии
    assert gl.ntp(t1, t1 + 5_000 + off, t1 + 5_000 + off, t1 + 10_000)[1] == off


def _simulate(my_off_us, liars=(), lie_us=1_000_000, dur_us=3_000_000, seed=0):
    """Я (узел 0) и 6 соседей; истинное время T, мои сырые часы T + my_off.
    Задержка в каждую сторону 5-30 мс. Лжецы подставляют t2/t3 со сдвигом."""
    import heapq
    import random
    rnd = random.Random(seed)
    cs = gl.ClockSync(0, range(7))
    ev, k, T = [], 0, 0
    while T < dur_us:
        for p in cs.pings_due(cs.now(T + my_off_us)):
            t1 = cs.out[p]
            d1, d2 = rnd.uniform(5e3, 30e3), rnd.uniform(5e3, 30e3)
            sh = lie_us if p in liars else 0
            t2 = int(T + d1) + sh
            heapq.heappush(ev, (T + d1 + d2, k, p, t1, t2, t2 + 100))
            k += 1
        while ev and ev[0][0] <= T:
            ta, _, p, t1, t2, t3 = heapq.heappop(ev)
            cs.on_pong(p, t1, t2, t3, cs.now(ta + my_off_us))
        T += 1_000
    return cs


@pytest.mark.parametrize("my_off", [-200_000, 200_000])
def test_clock_sync_aligns_within_two_seconds(my_off):
    cs = _simulate(my_off, dur_us=2_000_000)
    assert cs.locked
    assert abs(cs.offset_us + my_off) < 10_000                # остаток < 10 мс
    assert cs.corrections and cs.corrections[0][1] * my_off < 0


@pytest.mark.parametrize("liars", [(1,), (1, 2)])
def test_clock_sync_ignores_byzantine_minority(liars):
    clean = _simulate(0, dur_us=3_000_000)
    for lie in (1_000_000, -1_000_000):
        cs = _simulate(0, liars=liars, lie_us=lie, dur_us=3_000_000)
        assert abs(cs.offset_us - clean.offset_us) < 10_000 and abs(cs.offset_us) < 10_000
        cs = _simulate(-200_000, liars=liars, lie_us=lie, dur_us=3_000_000)
        assert abs(cs.offset_us - 200_000) < 10_000


def test_pong_rate_limit_and_echo_check():
    cs = gl.ClockSync(0, range(4), pong_per_s=2)
    assert [cs.allow_pong(1, t) for t in (0, 100, 200, 1_000_150)] == [True, True, False, True]
    assert cs.allow_pong(2, 200)                              # у каждого соседа свой лимит
    cs.pings_due(5_000)                                       # PING всем соседям, t1 = 5000
    assert not cs.on_pong(1, 4_999, 6_000, 6_000, 7_000)      # чужое эхо t1
    assert cs.on_pong(1, 5_000, 6_000, 6_000, 7_000)
    assert not cs.on_pong(1, 5_000, 6_000, 6_000, 7_000)      # повтор того же ответа


def test_sync_frames_bypass_future_check_and_reply():
    ca, cb = gl.ClockSync(0, range(3)), gl.ClockSync(1, range(3), pong_per_s=2)
    a = gl.Node(0, KEYS[0][0], PUB, clock=ca)
    b = gl.Node(1, KEYS[1][0], PUB, clock=cb)
    pings = a.pings(1_000_000)
    assert len(pings) == 2 and all(len(p) == 102 for p in pings)
    to_b = [p for p in pings if gl.unpack(p, PUB)["target"] == 1][0]
    # часы b впереди на 0.4 с - PING для b «из прошлого», ответ для a «из будущего»
    r, pong = b.receive(to_b, "a", 1_410_000)
    assert r == "ping_in" and pong is not None
    f = gl.unpack(pong, PUB)
    assert f["flags"] & gl.FLAG_PONG and f["payload"] == (1_000_000, 1_410_000)
    assert a.receive(pong, "b", 1_020_000)[0] == "pong_in"    # не quarantine_future
    assert a.q_authors == {} and ca.samples[1] == [(20_000, 400_000.0)]
    c = gl.Node(2, KEYS[2][0], PUB, clock=gl.ClockSync(2, range(3)))
    assert c.receive(pong, "b", 1_020_000)[0] == "sync_other"  # адресован не мне
    ping = lambda s, t: gl.pack(KEYS[0][0], gl.SYNC_PING, 0, 1, s, t, t, 0)
    res = [b.receive(ping(100 + i, t), "a", t)[0] for i, t in enumerate((1_420_000, 1_430_000))]
    assert res == ["ping_in", "ping_flood"]                   # третий PING за секунду - флуд
    assert b.receive(ping(200, 2_420_000), "a", 2_420_000)[0] == "ping_in"   # окно сдвинулось
    # групповой PING: один кадр, отвечают все; PONG с target = BCAST не принимается
    g = a.pings(2_000_000, group=True)
    assert len(g) == 1 and gl.unpack(g[0], PUB)["target"] == gl.BCAST
    assert a.pings(2_100_000, group=True) == []               # срок ещё не подошёл
    assert c.receive(g[0], "a", 2_010_000)[0] == "ping_in"
    fake = gl.pack(KEYS[1][0], gl.SYNC_PING, 1, gl.BCAST, 300, 2_020_000, 2_000_000, 2_010_000,
                   flags=gl.FLAG_PONG)
    assert a.receive(fake, "b", 2_030_000)[0] == "sync_other"


def test_unsynced_node_does_not_judge():
    rt_core, lib = _lib()
    cs = gl.ClockSync(0, range(6))
    n = gl.Node(0, KEYS[0][0], PUB, kin=rt_core.Kin(lib), clock=cs)
    far = lambda tg, t: (1e3, 1e3, 1e3)                      # «мой датчик» с другого конца света
    for s in range(1, 20):
        assert n.receive(_claim(1, s, s * 40_000), "s1", s * 40_000 + 5_000, mine=far)[0] == "unsynced"
    assert n.q_authors == {}


def test_tomb_backoff_per_peer():
    b = gl.TombBackoff()
    sent = [t for t in range(0, 40_000_000, 500_000) if b.due("p1", "x", t)]
    assert sent == [0, 1_000_000, 3_000_000, 7_000_000, 15_000_000, 31_000_000]
    assert b.due("p2", "x", 31_500_000)                       # у каждого соседа свой счёт
    assert b.due("p1", "y", 31_500_000)                       # и у каждого надгробия
    n = gl.Node(0, KEYS[0][0], PUB)
    n.quarantine_author(4, gl.REASON_TRAJ_VETO, 0)
    assert len(n.tombs_for("p1", 0)) == 1 and n.tombs_for("p1", 500_000) == []


# --- связка с C99-шлюзом ---------------------------------------------------

def _lib():
    try:
        import rt_core
        return rt_core, rt_core.load()
    except OSError:
        pytest.skip("librt_core.so не собран или не грузится на этой платформе")


def test_kinematic_veto_quarantines_liar_and_forgets_it():
    rt_core, lib = _lib()
    kin = rt_core.Kin(lib)
    n = gl.Node(0, KEYS[0][0], PUB, kin=kin, veto_k=3)
    line = lambda t: (10_000 + 400 * t / 1e6, 20_000 - 250 * t / 1e6, 500.0)
    t, seqs, res = 200_000, {a: 0 for a in range(1, 6)}, []
    while t < 6_000_000:
        for a in (1, 2, 3, 4, 5):
            ta = t + 23_000 * a
            seqs[a] += 1
            x = line(ta)
            if a == 5 and ta > 4_000_000:
                x = (x[0] + 1100, x[1] - 900, x[2])          # T8 после 4 с
            r, reply = n.receive(gl.pack(KEYS[a][0], gl.CLAIM, a, 0, seqs[a], ta, *x), "s%d" % a,
                                 ta + 5_000, mine=lambda tg, tc: line(tc))
            res.append((a, r))
        t += 250_000
    hon = [r for a, r in res if a != 5]
    assert all(r in ("pass", "failsafe") for r in hon)        # 0 опровержений честных
    assert n.q_authors == {5: gl.REASON_TRAJ_VETO}
    liar = [r for a, r in res if a == 5]
    assert liar.count("refute") == 2 and "quarantine_veto" in liar
    assert liar[-1] == "drop_quarantined"
    # лжец забыт кольцом: прогноз «со всеми» равен прогнозу «без автора 5»
    p_all, p_x5 = kin.predict(0, 6_100_000), kin.predict(0, 6_100_000, exclude=5)
    assert p_all is not None and max(abs(a - b) for a, b in zip(p_all, p_x5)) < 1e-9
