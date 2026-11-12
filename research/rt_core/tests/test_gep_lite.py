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


def test_forgery_quarantines_source_not_the_named_author():
    n = gl.Node(0, KEYS[0][0], PUB)
    forged = gl.pack(KEYS[3][0], gl.CLAIM, 1, 0, 1, 1000, 0.0, 0.0, 0.0)
    assert n.receive(forged, "10.0.0.66:9000", 2000)[0] == "reject_signature"
    assert "10.0.0.66:9000" in n.q_sources and 1 not in n.q_authors
    assert n.receive(_claim(1, 1, 1500), "10.0.0.1:9000", 2000)[0] == "pass"    # автор 1 цел
    assert n.receive(_claim(1, 2, 1600), "10.0.0.66:9000", 2000)[0] == "drop_source"


def test_replay_and_future_timestamp():
    n = gl.Node(0, KEYS[0][0], PUB, max_skew_us=50_000)
    assert n.receive(_claim(1, 5, 1_000_000), "s1", 1_000_100)[0] == "pass"
    assert n.receive(_claim(1, 5, 1_000_000), "s1", 1_000_200)[0] == "replay"
    assert n.receive(_claim(1, 4, 1_000_000), "s1", 1_000_200)[0] == "replay"
    assert n.receive(_claim(1, 6, 900_000), "s1", 1_000_300)[0] == "pass"       # задержка доставки - норма
    assert n.receive(_claim(2, 1, 1_100_000), "s2", 1_000_000)[0] == "quarantine_future"
    assert n.q_authors == {2: 3}
    assert n.receive(_claim(2, 2, 1_000_000), "s2", 1_000_000)[0] == "drop_quarantined"
    tomb = gl.unpack(n.tombs[0][0], PUB)
    assert tomb["type"] == gl.TOMBSTONE and tomb["payload"][0] == 2


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
