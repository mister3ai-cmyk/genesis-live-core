# Business Source License 1.1 - см. research/LICENSE
"""Живая шина S4/S5.1: N независимых процессов ОС, реальное время, кадры
GEP v0.1 Lite (Ed25519), у каждого узла свой C99-шлюз (librt_core.so через
ctypes) и своя синхронизация часов (gep_lite.ClockSync по SYNC_PING/PONG).

Мир: траектория дрона из EuRoC (state_groundtruth_estimate0) или
синтетическая. Каждый узел - борт-наблюдатель: rate Гц шлёт CLAIM о дроне
(истина + шум sigma; у атакующего с attack_s - ложь T8 20-50 sigma по оси).
Свой датчик для шлюза - истина + шум, в 5 % грубый сбой; читается в момент,
который узел считает меткой заявки (с ошибкой своих часов).

Транспорт (S5.1): групповая рассылка UDP multicast 239.255.42.99 на
loopback - один sendto на кадр. Режим unicast - отдельный sendto каждому
соседу (как в S4), для сравнения. Потери и задержку эмулирует ПОЛУЧАТЕЛЬ:
каждый процесс сам решает, потерять ли кадр (loss), и задерживает его на
равномерные jitter_ms - у разных получателей разные потери.

Часы узла = общий монотонный таймер + сдвиг clock_off_ms + поправка
ClockSync. Атакующий может лгать в SYNC: отвечать на PING со сдвигом
sync_liar_ms.

  python3 live_bus.py data/euroc/V1_02_medium      # все сценарии
"""
import heapq
import json
import multiprocessing as mp
import os
import random
import select
import socket
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SIGMA = 0.02
BASE_PORT = 47100
GROUP, GROUP_PORT = "239.255.42.99", 47500


def load_truth(seq_dir):
    """(t, pos) истины; seq_dir None - синтетическая восьмёрка."""
    if seq_dir:
        gt = np.loadtxt(os.path.join(seq_dir, "state_groundtruth_estimate0.csv"), delimiter=",", comments="#")
        return (gt[:, 0] - gt[0, 0]) * 1e-9, gt[:, 1:4]
    t = np.arange(0, 120, 0.005)
    return t, np.stack([2 * np.sin(0.6 * t), np.sin(1.2 * t), 1 + 0.2 * np.sin(0.3 * t)], 1)


def open_sockets(cfg, idx):
    """(tx, rx, адресаты). tx привязан к своему порту - адрес источника у
    каждого узла свой и в групповом режиме (нужно для карантина источника)."""
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.bind(("127.0.0.1", cfg["base_port"] + idx))
    if cfg["mode"] == "unicast":
        tx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        tx.setblocking(False)
        return tx, tx, [("127.0.0.1", cfg["base_port"] + j) for j in range(cfg["n"]) if j != idx]
    tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton("127.0.0.1"))
    tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 0)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    rx.bind(("", cfg["group_port"]))
    try:
        rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                      struct.pack("4s4s", socket.inet_aton(GROUP), socket.inet_aton("127.0.0.1")))
    except OSError as e:
        raise RuntimeError("не удалось войти в группу %s на loopback (%s): запустите с --unicast" % (GROUP, e))
    rx.setblocking(False)
    return tx, rx, [(GROUP, cfg["group_port"])]


def node_main(idx, cfg, out_q):
    import gep_lite as gl
    import rt_core
    rng = random.Random(1000 + idx)
    nrng = np.random.default_rng(2000 + idx)
    tt, pp = load_truth(cfg["seq"])

    def truth(t_s):
        t = cfg["t_offset_s"] + t_s
        return [float(np.interp(t, tt, pp[:, k])) for k in range(3)]

    n, mcast = cfg["n"], cfg["mode"] != "unicast"
    keys = {i: gl.keypair(i) for i in range(n)}
    pub = {i: pk for i, (sk, pk) in keys.items()}
    lib = rt_core.load()
    kin = rt_core.Kin(lib, horizon_us=500_000, max_skew_us=50_000, drop_tol=14 * SIGMA,
                      thr=20 * SIGMA, pos_tol=10 * SIGMA, min_pts=4)
    clock = gl.ClockSync(idx, range(n))
    node = gl.Node(idx, keys[idx][0], pub, kin=kin, clock=clock)
    attacker = idx == cfg["attacker"]
    lie_us = int(cfg["sync_liar_ms"] * 1000) if attacker and cfg["sync_liar_ms"] is not None else None
    off_us = cfg["clock_off_ms"].get(str(idx), 0) * 1000
    tx, rx, dests = open_sockets(cfg, idx)
    start_ns, dur_us = cfg["start_ns"], int(cfg["duration_s"] * 1e6)
    real_us = lambda: (time.monotonic_ns() - start_ns) // 1000           # общее (истинное) время
    now_local = lambda: clock.now(real_us() + off_us)                    # мои часы с поправкой
    rxq, k = [], 0
    st = dict(sent_frames=0, sent_bytes=0, tomb_frames=0, tomb_bytes=0, sync_frames=0, sync_bytes=0,
              dropped=0, recv_frames=0, recv_bytes=0, sync_skipped=0,
              proc_us=[], e2e_ms=[], verdicts={}, quarantine=[], bytes_per_s={}, corrections=[], locked_s=None)

    def send(frame, to=None):
        for d in (to or dests):
            tx.sendto(frame, d)
            st["sent_frames"] += 1
            st["sent_bytes"] += len(frame)
            if frame[3] == gl.TOMBSTONE:
                st["tomb_frames"] += 1
                st["tomb_bytes"] += len(frame)
            elif frame[3] == gl.SYNC_PING:
                st["sync_frames"] += 1
                st["sync_bytes"] += len(frame)
            sec = str(real_us() // 1_000_000)
            st["bytes_per_s"][sec] = st["bytes_per_s"].get(sec, 0) + len(frame)

    def mine(tg, t_claim_us):
        t_s = (t_claim_us - off_us - clock.offset_us) / 1e6   # мой датчик - в моём времени
        x = truth(t_s)
        if nrng.random() < 0.05:
            return [x[k] + nrng.choice((-1, 1)) * nrng.uniform(60 * SIGMA, 100 * SIGMA) for k in range(3)]
        return [x[k] + nrng.normal(0, SIGMA) for k in range(3)]

    def for_me(data):
        """SYNC-кадры, адресованные не мне, отбрасываются до проверки подписи
        (ничего не принимается - только экономится Ed25519)."""
        if len(data) < 8 or data[3] != gl.SYNC_PING:
            return True
        tg = int.from_bytes(data[6:8], "big")
        return tg == idx or (tg == gl.BCAST and not data[5] & gl.FLAG_PONG)

    while time.monotonic_ns() < start_ns:
        time.sleep(0.0005)
    period_us = int(1e6 / cfg["rate"])
    next_claim = int(idx * period_us / n)
    next_tomb = next_ping = int(idx * 50_000 / n)
    n_corr = 0
    cpu0 = time.process_time()
    while True:
        r = real_us()
        if r >= dur_us:
            break
        local = clock.now(r + off_us)
        if r >= next_claim:
            x = truth(r / 1e6)
            if attacker and r >= cfg["attack_s"] * 1e6:
                x = [x[k] + nrng.choice((-1, 1)) * nrng.uniform(20 * SIGMA, 50 * SIGMA) for k in range(3)]
            else:
                x = [x[k] + nrng.normal(0, SIGMA) for k in range(3)]
            send(node.frame(gl.CLAIM, 0, local, *x))
            next_claim += period_us
        if r >= next_ping:
            if not attacker:
                for fr in node.pings(local, group=mcast):
                    tg = int.from_bytes(fr[6:8], "big")
                    send(fr, None if mcast else [("127.0.0.1", cfg["base_port"] + tg)])
            next_ping += 50_000
        if r >= next_tomb:
            if mcast:
                for fr in node.tombs_for("group", local):
                    send(fr)
            else:
                for j, p in enumerate(dests):
                    for fr in node.tombs_for(p, local):
                        send(fr, [p])
            next_tomb += 100_000
        while rxq and rxq[0][0] <= r:
            _, _, data, src = heapq.heappop(rxq)
            if not for_me(data):
                st["sync_skipped"] += 1
                continue
            c0 = time.perf_counter_ns()
            t_now = now_local()
            if attacker:
                verdict, reply = "attacker", None
                if lie_us is not None and data[3] == gl.SYNC_PING and not data[5] & gl.FLAG_PONG:
                    f = gl.unpack(data, pub)                  # ложь в SYNC: t2, t3 со сдвигом
                    reply = node.frame(gl.SYNC_PING, f["author"], t_now + lie_us, f["t_us"], t_now + lie_us,
                                       flags=gl.FLAG_PONG | gl.FLAG_LOCKED)
            else:
                verdict, reply = node.receive(data, src, t_now, mine=mine)
                st["proc_us"].append((time.perf_counter_ns() - c0) / 1000)
                if len(data) == 110 and verdict in ("pass", "failsafe", "refute", "quarantine_veto"):
                    t_claim = int.from_bytes(data[12:20], "big", signed=True)
                    st["e2e_ms"].append((t_now - t_claim) / 1000)
            a = data[4] if len(data) > 4 else -1
            key = "%s@%d" % (verdict, a)
            st["verdicts"][key] = st["verdicts"].get(key, 0) + 1
            if verdict.startswith("quarantine"):
                st["quarantine"].append((a, verdict, real_us() / 1e6))
            if reply:
                if mcast:
                    send(reply)
                else:
                    send(reply, [("127.0.0.1", cfg["base_port"] + int.from_bytes(reply[6:8], "big"))]
                         if reply[3] == gl.SYNC_PING else None)
        if len(clock.corrections) > n_corr:
            for _, c in clock.corrections[n_corr:]:
                st["corrections"].append((real_us() / 1e6, c))
            n_corr = len(clock.corrections)
        if st["locked_s"] is None and clock.locked:
            st["locked_s"] = real_us() / 1e6
        wait = min(next_claim, next_tomb, next_ping, rxq[0][0] if rxq else next_claim) - real_us()
        rl, _, _ = select.select([rx], [], [], max(0.0, min(wait, 2000)) / 1e6)
        if not rl:
            continue
        r = real_us()
        while True:
            try:
                data, src = rx.recvfrom(2048)
            except BlockingIOError:
                break
            if len(data) > 4 and data[4] == idx:              # своя петля multicast
                continue
            st["recv_frames"] += 1
            st["recv_bytes"] += len(data)
            if rng.random() < cfg["loss"]:
                st["dropped"] += 1
                continue
            k += 1
            heapq.heappush(rxq, (r + rng.uniform(*cfg["jitter_ms"]) * 1000, k, data, src))
    st["cpu_s"] = time.process_time() - cpu0
    st["clock_offset_us"] = clock.offset_us
    st["clock_residual_us"] = off_us + clock.offset_us              # ошибка моих часов против истины
    st["q_authors"] = {str(k): v for k, v in node.q_authors.items()}
    st["q_sources"] = len(node.q_sources)
    st["proc_us"] = [float(np.percentile(st["proc_us"], 50)), float(np.percentile(st["proc_us"], 99))] \
        if st["proc_us"] else [0, 0]
    st["e2e_ms"] = [float(np.percentile(st["e2e_ms"], 50)), float(np.percentile(st["e2e_ms"], 99))] \
        if st["e2e_ms"] else [0, 0]
    out_q.put((idx, st))
    tx.close()
    if rx is not tx:
        rx.close()


def run(seq, n=7, attacker=6, rate=25.0, loss=0.0, jitter_ms=(5, 30), duration_s=30.0, attack_s=5.0,
        clock_off_ms=None, t_offset_s=10.0, base_port=BASE_PORT, mode="mcast", group_port=GROUP_PORT,
        sync_liar_ms=None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    cfg = dict(seq=seq, n=n, attacker=attacker, rate=rate, loss=loss, jitter_ms=list(jitter_ms),
               duration_s=duration_s, attack_s=attack_s, clock_off_ms={str(k): v for k, v in (clock_off_ms or {}).items()},
               t_offset_s=t_offset_s, base_port=base_port, mode=mode, group_port=group_port,
               sync_liar_ms=sync_liar_ms, start_ns=time.monotonic_ns() + int(3e9))
    procs = [ctx.Process(target=node_main, args=(i, cfg, q)) for i in range(n)]
    for p in procs:
        p.start()
    res, deadline = {}, time.monotonic() + duration_s + 60
    while len(res) < n:
        try:
            i, st = q.get(timeout=1.0)
            res[i] = st
        except Exception:
            dead = [p.pid for p in procs if not p.is_alive() and p.exitcode]
            if dead or time.monotonic() > deadline:
                for p in procs:
                    p.kill()
                raise RuntimeError("узлы шины упали или не ответили: PID %s" % dead)
    for p in procs:
        p.join(timeout=10)
    return cfg, res, [p.pid for p in procs]


def aligned_at(st, tol_us=15_000, off_us=0):
    """Момент (с), после которого ошибка часов узла не выходила за tol:
    по журналу поправок; 0.0 - если поправки не понадобились. Допуск 15 мс:
    при джиттере 5-30 мс в каждую сторону ошибка одного замера NTP - до
    половины асимметрии, 12.5 мс."""
    err, t_ok = off_us, 0.0 if abs(off_us) < tol_us else None
    for t, c in st["corrections"]:
        err += c
        if abs(err) < tol_us:
            t_ok = t if t_ok is None else t_ok
        else:
            t_ok = None
    return t_ok


def summarize(cfg, res):
    n, att = cfg["n"], cfg["attacker"]
    honest = [i for i in range(n) if i != att]
    offs = {int(k): v for k, v in cfg["clock_off_ms"].items() if v}
    desync = set(offs)
    ok_obs = [i for i in honest if i not in desync]          # наблюдатели с верными часами
    iso = [next((t for a, v, t in res[i]["quarantine"] if a == att), None) for i in honest]

    def refutes(obs):
        return sum(c for i in obs for k, c in res[i]["verdicts"].items()
                   if k.startswith("refute@") and int(k.split("@")[1]) != att)
    honest_claims = sum(c for i in honest for k, c in res[i]["verdicts"].items()
                        if k.split("@")[0] in ("pass", "failsafe", "refute") and int(k.split("@")[1]) != att)
    false_q = [(i, a, v) for i in honest for a, v in res[i]["q_authors"].items() if int(a) != att]
    dur = cfg["duration_s"]
    kbit = [res[i]["sent_bytes"] * 8 / 1000 / dur for i in range(n)]
    rx_kbit = [res[i]["recv_bytes"] * 8 / 1000 / dur for i in range(n)]
    peak = [max(res[i]["bytes_per_s"].values()) * 8 / 1000 for i in range(n)]
    cpu = [100 * res[i]["cpu_s"] / dur for i in range(n)]
    sync = {i: dict(aligned_s=aligned_at(res[i], off_us=offs.get(i, 0) * 1000),
                    locked_s=res[i]["locked_s"], n_corr=len(res[i]["corrections"]),
                    residual_ms=round(res[i]["clock_residual_us"] / 1000, 2)) for i in honest}
    return dict(
        mode=cfg["mode"], isolated=sum(t is not None for t in iso), observers=len(honest),
        iso_after_attack_s=[round(t - cfg["attack_s"], 2) for t in iso if t is not None],
        false_alarm=refutes(honest), false_alarm_desync_nodes=refutes(desync), honest_claims=honest_claims,
        false_q=false_q,
        unsynced_claims=sum(c for i in honest for k, c in res[i]["verdicts"].items() if k.startswith("unsynced@")),
        future_drops=sum(c for i in honest for k, c in res[i]["verdicts"].items() if k.startswith("future_drop@")),
        desync=sorted(desync), desync_sync={i: sync[i] for i in desync},
        residual_ms_max=max(abs(s["residual_ms"]) for s in sync.values()),
        locked_s_max=max((s["locked_s"] or 99) for s in sync.values()),
        proc_us_p50=float(np.median([res[i]["proc_us"][0] for i in ok_obs])),
        proc_us_p99=float(np.max([res[i]["proc_us"][1] for i in ok_obs])),
        e2e_ms_p50=float(np.median([res[i]["e2e_ms"][0] for i in ok_obs])),
        e2e_ms_p99=float(np.max([res[i]["e2e_ms"][1] for i in ok_obs])),
        kbit_mean=float(np.mean(kbit)), kbit_peak=float(np.max(peak)), rx_kbit_mean=float(np.mean(rx_kbit)),
        sync_kbit_per_node=float(np.mean([res[i]["sync_bytes"] * 8 / 1000 / dur for i in honest])),
        cpu_mean=float(np.mean(cpu)), cpu_max=float(np.max(cpu)),
        dropped=sum(res[i]["dropped"] for i in range(n)), sent=sum(res[i]["sent_frames"] for i in range(n)),
        tomb_frames_per_node=float(np.mean([res[i]["tomb_frames"] for i in ok_obs])),
        tomb_kbit_per_node=float(np.mean([res[i]["tomb_bytes"] * 8 / 1000 / dur for i in ok_obs])))


SCENARIOS = [
    ("потери 0 %", dict(loss=0.0)),
    ("потери 5 %", dict(loss=0.05)),
    ("потери 20 %", dict(loss=0.20)),
    ("потери 5 %, часы узла 2 на +200 мс", dict(loss=0.05, clock_off_ms={2: 200})),
    ("потери 5 %, часы узла 2 на -200 мс", dict(loss=0.05, clock_off_ms={2: -200})),
    ("потери 5 %, лжец в SYNC (+1 с)", dict(loss=0.05, sync_liar_ms=1000)),
    ("потери 5 %, часы узла 2 на -200 мс + лжец в SYNC", dict(loss=0.05, clock_off_ms={2: -200}, sync_liar_ms=1000)),
    ("потери 5 %, unicast (как S4)", dict(loss=0.05, mode="unicast")),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seq = args[0] if args else None
    force_uni = "--unicast" in sys.argv
    print("Живая шина: 7 процессов ОС, UDP 127.0.0.1 (%s), GEP v0.1 Lite + SYNC, 25 Гц на узел, "
          "джиттер 5-30 мс у получателя, 30 с, атакующий - узел 6 (T8 с 5-й с); траектория: %s"
          % ("unicast" if force_uni else "multicast " + GROUP, os.path.basename(seq) if seq else "синтетика"))
    for j, (name, kw) in enumerate(SCENARIOS):
        kw = dict(kw, group_port=GROUP_PORT + j)
        if force_uni:
            kw["mode"] = "unicast"
        cfg, res, pids = run(seq, **kw)
        s = summarize(cfg, res)
        print("\n== %s (PID %s)" % (name, ",".join(map(str, pids))))
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
