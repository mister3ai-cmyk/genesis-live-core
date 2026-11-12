# Business Source License 1.1 - см. research/LICENSE
"""Живая шина S4: N независимых процессов ОС (свой PID, свой UDP-сокет на
127.0.0.1), реальное время, кадры GEP v0.1 Lite (Ed25519), у каждого узла
свой C99-шлюз (librt_core.so через ctypes).

Мир: траектория дрона из EuRoC (state_groundtruth_estimate0) или
синтетическая. Каждый узел - борт-наблюдатель: rate Гц шлёт CLAIM о дроне
(истина + шум sigma; у атакующего с attack_s - ложь T8 20-50 sigma по оси).
Свой датчик для шлюза - истина + шум, в 5 % грубый сбой.

Транспорт эмулирует отправитель: потеря каждого кадра с вероятностью loss,
задержка доставки - равномерно jitter_ms. Часы узла = общий монотонный
таймер + сдвиг clock_off_ms (сценарий рассинхрона).

  python3 live_bus.py data/euroc/V1_02_medium      # все сценарии
"""
import heapq
import json
import multiprocessing as mp
import os
import random
import select
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SIGMA = 0.02
BASE_PORT = 47100


def load_truth(seq_dir):
    """(t, pos) истины; seq_dir None - синтетическая восьмёрка."""
    if seq_dir:
        gt = np.loadtxt(os.path.join(seq_dir, "state_groundtruth_estimate0.csv"), delimiter=",", comments="#")
        return (gt[:, 0] - gt[0, 0]) * 1e-9, gt[:, 1:4]
    t = np.arange(0, 120, 0.005)
    return t, np.stack([2 * np.sin(0.6 * t), np.sin(1.2 * t), 1 + 0.2 * np.sin(0.3 * t)], 1)


def node_main(idx, cfg, out_q):
    import gep_lite as gl
    import rt_core
    rng = random.Random(1000 + idx)
    nrng = np.random.default_rng(2000 + idx)
    tt, pp = load_truth(cfg["seq"])

    def truth(t_s):
        t = cfg["t_offset_s"] + t_s
        return [float(np.interp(t, tt, pp[:, k])) for k in range(3)]

    n = cfg["n"]
    keys = {i: gl.keypair(i) for i in range(n)}
    pub = {i: pk for i, (sk, pk) in keys.items()}
    lib = rt_core.load()
    kin = rt_core.Kin(lib, horizon_us=500_000, max_skew_us=50_000, drop_tol=14 * SIGMA,
                      thr=20 * SIGMA, pos_tol=10 * SIGMA, min_pts=4)
    node = gl.Node(idx, keys[idx][0], pub, kin=kin)
    attacker = idx == cfg["attacker"]
    off_us = cfg["clock_off_ms"].get(str(idx), 0) * 1000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    sock.bind(("127.0.0.1", cfg["base_port"] + idx))
    sock.setblocking(False)
    peers = [("127.0.0.1", cfg["base_port"] + j) for j in range(n) if j != idx]
    start_ns, dur_us = cfg["start_ns"], int(cfg["duration_s"] * 1e6)
    real_us = lambda: (time.monotonic_ns() - start_ns) // 1000           # общее время
    heap, seqno = [], 0
    st = dict(sent_frames=0, sent_bytes=0, tomb_frames=0, tomb_bytes=0, dropped=0, recv_frames=0, recv_bytes=0,
              proc_us=[], e2e_ms=[], verdicts={}, quarantine=[], bytes_per_s={})

    def send(frame, now_r):
        nonlocal seqno
        for p in peers:
            if rng.random() < cfg["loss"]:
                st["dropped"] += 1
                continue
            d = rng.uniform(cfg["jitter_ms"][0], cfg["jitter_ms"][1]) * 1000
            seqno += 1
            heapq.heappush(heap, (now_r + d, seqno, frame, p))

    def mine(tg, t_claim_us):
        t_s = (t_claim_us - off_us) / 1e6                  # мой датчик - в моём времени
        x = truth(t_s)
        if nrng.random() < 0.05:
            return [x[k] + nrng.choice((-1, 1)) * nrng.uniform(60 * SIGMA, 100 * SIGMA) for k in range(3)]
        return [x[k] + nrng.normal(0, SIGMA) for k in range(3)]

    while time.monotonic_ns() < start_ns:
        time.sleep(0.0005)
    period_us = int(1e6 / cfg["rate"])
    next_claim = int(idx * period_us / n)
    next_tomb = 0
    cpu0 = time.process_time()
    while True:
        r = real_us()
        if r >= dur_us:
            break
        local = r + off_us
        if r >= next_claim:
            x = truth(r / 1e6)
            if attacker and r >= cfg["attack_s"] * 1e6:
                x = [x[k] + nrng.choice((-1, 1)) * nrng.uniform(20 * SIGMA, 50 * SIGMA) for k in range(3)]
            else:
                x = [x[k] + nrng.normal(0, SIGMA) for k in range(3)]
            send(node.frame(gl.CLAIM, 0, local, *x), r)
            next_claim += period_us
        if r >= next_tomb:
            for p in peers:
                for fr in node.tombs_for(p, local):
                    if rng.random() >= cfg["loss"]:
                        seqno += 1
                        heapq.heappush(heap, (r + rng.uniform(*cfg["jitter_ms"]) * 1000, seqno, fr, p))
            next_tomb += 100_000
        while heap and heap[0][0] <= r:
            _, _, fr, p = heapq.heappop(heap)
            sock.sendto(fr, p)
            st["sent_frames"] += 1
            st["sent_bytes"] += len(fr)
            if fr[3] == 3:                                 # TOMBSTONE
                st["tomb_frames"] += 1
                st["tomb_bytes"] += len(fr)
            sec = str(r // 1_000_000)
            st["bytes_per_s"][sec] = st["bytes_per_s"].get(sec, 0) + len(fr)
        wait = min(next_claim, next_tomb, heap[0][0] if heap else next_claim) - real_us()
        rl, _, _ = select.select([sock], [], [], max(0.0, min(wait, 2000)) / 1e6)
        if not rl:
            continue
        while True:
            try:
                data, src = sock.recvfrom(2048)
            except BlockingIOError:
                break
            st["recv_frames"] += 1
            st["recv_bytes"] += len(data)
            if attacker:
                continue
            c0 = time.perf_counter_ns()
            now_local = real_us() + off_us
            verdict, reply = node.receive(data, src, now_local, mine=mine)
            st["proc_us"].append((time.perf_counter_ns() - c0) / 1000)
            if len(data) == 110 and verdict in ("pass", "failsafe", "refute", "quarantine_veto"):
                t_claim = int.from_bytes(data[12:20], "big", signed=True)
                st["e2e_ms"].append((now_local - t_claim) / 1000)
            a = data[4] if len(data) > 4 else -1
            key = "%s@%d" % (verdict, a)
            st["verdicts"][key] = st["verdicts"].get(key, 0) + 1
            if verdict.startswith("quarantine"):
                st["quarantine"].append((a, verdict, real_us() / 1e6))
            if reply:
                send(reply, real_us())
    st["cpu_s"] = time.process_time() - cpu0
    st["q_authors"] = {str(k): v for k, v in node.q_authors.items()}
    st["q_sources"] = len(node.q_sources)
    st["proc_us"] = [float(np.percentile(st["proc_us"], 50)), float(np.percentile(st["proc_us"], 99))] \
        if st["proc_us"] else [0, 0]
    st["e2e_ms"] = [float(np.percentile(st["e2e_ms"], 50)), float(np.percentile(st["e2e_ms"], 99))] \
        if st["e2e_ms"] else [0, 0]
    out_q.put((idx, st))
    sock.close()


def run(seq, n=7, attacker=6, rate=25.0, loss=0.0, jitter_ms=(5, 30), duration_s=30.0, attack_s=5.0,
        clock_off_ms=None, t_offset_s=10.0, base_port=BASE_PORT):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    cfg = dict(seq=seq, n=n, attacker=attacker, rate=rate, loss=loss, jitter_ms=list(jitter_ms),
               duration_s=duration_s, attack_s=attack_s, clock_off_ms={str(k): v for k, v in (clock_off_ms or {}).items()},
               t_offset_s=t_offset_s, base_port=base_port, start_ns=time.monotonic_ns() + int(3e9))
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


def summarize(cfg, res):
    n, att = cfg["n"], cfg["attacker"]
    honest = [i for i in range(n) if i != att]
    desync = {int(k) for k, v in cfg["clock_off_ms"].items() if v}
    ok_obs = [i for i in honest if i not in desync]          # исправные наблюдатели
    iso = [next((t for a, v, t in res[i]["quarantine"] if a == att), None) for i in ok_obs]
    false_alarm = sum(c for i in ok_obs for k, c in res[i]["verdicts"].items()
                      if k.startswith("refute@") and int(k.split("@")[1]) not in (att, *desync))
    honest_claims = sum(c for i in ok_obs for k, c in res[i]["verdicts"].items()
                        if k.split("@")[0] in ("pass", "failsafe", "refute") and int(k.split("@")[1]) not in (att, *desync))
    false_q = [(i, a) for i in ok_obs for a in res[i]["q_authors"] if int(a) != att and int(a) not in desync]
    desync_q = [(i, a, v) for i in ok_obs for a, v in res[i]["q_authors"].items() if int(a) in desync]
    desync_iso_s = [round(t, 2) for i in ok_obs for a, v, t in res[i]["quarantine"] if a in desync]
    desync_view = {i: sorted(int(a) for a in res[i]["q_authors"]) for i in desync}   # кого отрезал сам
    dur = cfg["duration_s"]
    kbit = [res[i]["sent_bytes"] * 8 / 1000 / dur for i in range(n)]
    peak = [max(res[i]["bytes_per_s"].values()) * 8 / 1000 for i in range(n)]
    cpu = [100 * res[i]["cpu_s"] / dur for i in range(n)]
    return dict(
        isolated=sum(t is not None for t in iso), observers=len(ok_obs),
        iso_after_attack_s=[round(t - cfg["attack_s"], 2) for t in iso if t is not None],
        false_alarm=false_alarm, honest_claims=honest_claims, false_q=false_q, desync_q=desync_q,
        desync_isolated_at_s=desync_iso_s, desync_node_cut=desync_view,
        proc_us_p50=float(np.median([res[i]["proc_us"][0] for i in ok_obs])),
        proc_us_p99=float(np.max([res[i]["proc_us"][1] for i in ok_obs])),
        e2e_ms_p50=float(np.median([res[i]["e2e_ms"][0] for i in ok_obs])),
        e2e_ms_p99=float(np.max([res[i]["e2e_ms"][1] for i in ok_obs])),
        kbit_mean=float(np.mean(kbit)), kbit_peak=float(np.max(peak)),
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
]


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else None
    print("Живая шина: 7 процессов ОС, UDP 127.0.0.1, GEP v0.1 Lite, 25 Гц на узел, джиттер 5-30 мс, 30 с, "
          "атакующий - узел 6 (T8 с 5-й с); траектория: %s" % (os.path.basename(seq) if seq else "синтетика"))
    for name, kw in SCENARIOS:
        cfg, res, pids = run(seq, **kw)
        s = summarize(cfg, res)
        print("\n== %s (PID %s)" % (name, ",".join(map(str, pids))))
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
