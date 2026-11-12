# Business Source License 1.1 - см. research/LICENSE
"""Цена слоёв GEP v0.1 Lite на один кадр CLAIM (110 B), p50 и p99 по
повторам, time.perf_counter_ns. Разблюдовка для шага 3: UDP против
Ed25519 против C99-шлюза.

  python3 gep_bench.py
"""
import os
import socket
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gep_lite as gl                                        # noqa: E402

N = 20_000


def measure(fn, n=N):
    t = np.empty(n)
    for i in range(n):
        c = time.perf_counter_ns()
        fn(i)
        t[i] = time.perf_counter_ns() - c
    return np.percentile(t, 50) / 1e3, np.percentile(t, 99) / 1e3


def main():
    keys = {i: gl.keypair(i) for i in range(8)}
    pub = {i: pk for i, (sk, pk) in keys.items()}
    sk1 = keys[1][0]
    frames = [gl.pack(sk1, gl.CLAIM, 1, 0, i + 1, 1_000_000 + i, 1.0, 2.0, 3.0) for i in range(N)]
    msg = frames[0][:-64]
    rows = []
    rows.append(("struct: заголовок + нагрузка (без подписи)",
                 measure(lambda i: gl.HDR.pack(gl.MAGIC, 1, 1, 1, 0, 0, i, i, 24) + gl.PAYLOAD[gl.CLAIM].pack(1.0, 2.0, 3.0))))
    rows.append(("Ed25519: подпись 46 B", measure(lambda i: sk1.sign(msg))))
    rows.append(("pack: кадр целиком (struct + подпись)",
                 measure(lambda i: gl.pack(sk1, gl.CLAIM, 1, 0, i + 1, i, 1.0, 2.0, 3.0))))
    rows.append(("unpack: проверки формата + Ed25519 verify", measure(lambda i: gl.unpack(frames[i], pub))))
    try:
        import rt_core
        lib = rt_core.load()
        kin = rt_core.Kin(lib)
        for j in range(8):
            for a in range(4):
                t = int((0.05 + 0.5 * j + 0.11 * a) * 1e6)
                kin.push(0, a, t, (10_000 + 400 * t / 1e6, 20_000, 500.0), t)
        mine = (11_600.0, 20_000.0, 500.0)
        claim_far = (mine[0] + 1100, mine[1] - 900, mine[2])
        rows.append(("C99-шлюз через ctypes, быстрый путь",
                     measure(lambda i: kin.gate(0, 5, 4_000_000, 4_000_500, mine, mine))))
        rows.append(("C99-шлюз через ctypes, с прогнозом",
                     measure(lambda i: kin.gate(0, 5, 4_000_000, 4_000_500, claim_far, mine))))
        node_kin = rt_core.Kin(lib)
        node = gl.Node(0, keys[0][0], pub, kin=node_kin)
        seq = {"n": 0}

        def recv(i):
            seq["n"] += 1
            a = 1 + i % 4
            t = 1_000_000 + 5_000 * i
            fr = gl.pack(keys[a][0], gl.CLAIM, a, 0, seq["n"], t, 10_000 + 0.4 * i, 20_000.0, 500.0)
            c = time.perf_counter_ns()
            node.receive(fr, "s%d" % a, t + 1000, mine=lambda tg, tc: (10_000 + 0.4 * i, 20_000.0, 500.0))
            return time.perf_counter_ns() - c
        ts = np.array([recv(i) for i in range(N)])
        rows.append(("Node.receive: формат + Ed25519 + карантин + шлюз + push",
                     (np.percentile(ts, 50) / 1e3, np.percentile(ts, 99) / 1e3)))
    except OSError:
        rows.append(("C99-шлюз", (float("nan"), float("nan"))))

    a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    b.bind(("127.0.0.1", 0))
    dst = b.getsockname()
    rows.append(("UDP loopback: sendto + recvfrom 110 B",
                 measure(lambda i: (a.sendto(frames[i], dst), b.recvfrom(2048)), n=5000)))
    print("Цена слоёв GEP v0.1 Lite, кадр CLAIM %d B, Python %s, мкс" % (len(frames[0]), sys.version.split()[0]))
    print("  %-56s %8s %8s" % ("слой", "p50", "p99"))
    for name, (p50, p99) in rows:
        print("  %-56s %8.2f %8.2f" % (name, p50, p99))


if __name__ == "__main__":
    main()
