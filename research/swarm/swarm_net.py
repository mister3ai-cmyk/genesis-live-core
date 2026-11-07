#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Business Source License 1.1 - см. research/LICENSE
"""
research/swarm/swarm_net.py

Транспорт роя: настоящий UDP на 127.0.0.1, по порту на узел. Потери,
задержка, дубли и разрывы связи ЭМУЛИРУЮТСЯ с зерном; UDP здесь только
транспорт, а не модель радиоканала.

Где что эмулируется:
  отправитель  потеря с вероятностью p, дубль 2 %, задержка 5-200 мс сим,
               разрыв связи (пакет в отрезанную сторону не уходит)
  получатель   перестановка: пакеты лежат в куче по моменту доставки и
               отдаются узлу по (deliver_ms, src, dseq), а не по приходу

Заголовок датаграммы - 15 байт struct: src, send_tick, deliver_ms, dseq, cls.
Полезная нагрузка - JSON-массив конвертов GEP. Датаграмма больше 1200 байт -
ValueError: тихой обрезки нет.
"""
import random
import socket
import struct

HDR = struct.Struct("!HIIIB")
MAX_DGRAM = 1200
CLASSES = ("push", "pull", "svc")
CLS_ID = {c: i for i, c in enumerate(CLASSES)}


def neighbors(topology, n, i):
    if topology == "full":
        return [j for j in range(n) if j != i]
    if topology == "chain":
        return [j for j in (i - 1, i + 1) if 0 <= j < n]
    raise ValueError("unknown topology %r" % topology)


def parse_partition(spec):
    """none | isolate:<node>@<start_s>+<dur_s> | split:<a,b>|<c,d,e>@<start_s>+<dur_s>

    Возвращает (cut(a, b, t_ms) -> bool, (start_ms, end_ms) или None)."""
    if not spec or spec == "none":
        return (lambda a, b, t: False), None
    kind, rest = spec.split(":", 1)
    groups, when = rest.split("@", 1)
    start_s, dur_s = when.split("+", 1)
    lo, hi = int(start_s) * 1000, (int(start_s) + int(dur_s)) * 1000
    if kind == "isolate":
        victim = int(groups)

        def cut(a, b, t):
            return lo <= t < hi and (a == victim) != (b == victim)
        return cut, (lo, hi)
    if kind == "split":
        side = {}
        for gi, g in enumerate(groups.split("|")):
            for x in g.split(","):
                side[int(x)] = gi

        def cut(a, b, t):
            return lo <= t < hi and side.get(a, -1) != side.get(b, -1)
        return cut, (lo, hi)
    raise ValueError("unknown partition %r" % spec)


def pack(env_bytes_list):
    """Пакует сериализованные конверты в датаграммы <= MAX_DGRAM вместе с
    заголовком. Один конверт больше лимита - ошибка, а не обрезка."""
    room = MAX_DGRAM - HDR.size - 2
    out, cur, size = [], [], 0
    for b in env_bytes_list:
        if len(b) > room:
            raise ValueError("envelope of %d bytes does not fit a %d-byte "
                             "datagram" % (len(b), MAX_DGRAM))
        add = len(b) + (1 if cur else 0)
        if cur and size + add > room:
            out.append(b"[" + b",".join(cur) + b"]")
            cur, size = [], 0
            add = len(b)
        cur.append(b)
        size += add
    if cur:
        out.append(b"[" + b",".join(cur) + b"]")
    return out


class Link:
    def __init__(self, idx, n, base_port, loss, seed, cut, dup=0.02,
                 dmin=5, dmax=200, active_until_ms=None):
        self.me, self.n, self.base = idx, n, base_port
        self.loss, self.dup, self.dmin, self.dmax = loss, dup, dmin, dmax
        self.cut = cut
        self.rng = random.Random("net:%s:%d" % (seed, idx))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
        self.sock.bind(("127.0.0.1", base_port + idx))
        self.dseq = 0
        self.sent_to = [0] * n          # датаграмм каждому в этом такте
        self.active_until_ms = active_until_ms
        # счётчики: [класс] -> [байт, пакетов, конвертов, подписей] за весь
        # прогон и за активное окно (пока идёт поток событий)
        self.tx = {c: [0, 0, 0, 0] for c in CLASSES}
        self.tx_active = {c: [0, 0, 0, 0] for c in CLASSES}
        self.lost = self.cut_drops = self.dups = 0

    def new_tick(self):
        self.sent_to = [0] * self.n

    def send(self, dst, cls, payload, tick, now_ms, n_env, n_sig=None):
        size = HDR.size + len(payload)
        if size > MAX_DGRAM:
            raise ValueError("datagram of %d bytes > %d" % (size, MAX_DGRAM))
        for acc in ((self.tx,) if self.active_until_ms is not None
                    and now_ms >= self.active_until_ms
                    else (self.tx, self.tx_active)):
            a = acc[cls]
            a[0] += size
            a[1] += 1
            a[2] += n_env
            a[3] += n_env if n_sig is None else n_sig
        if self.cut(self.me, dst, now_ms):
            self.cut_drops += 1
            return
        if self.rng.random() < self.loss:
            self.lost += 1
            return
        copies = 2 if self.rng.random() < self.dup else 1
        self.dups += copies - 1
        for _ in range(copies):
            delay = self.rng.randint(self.dmin, self.dmax)
            hdr = HDR.pack(self.me, tick, now_ms + delay, self.dseq,
                           CLS_ID[cls])
            self.dseq += 1
            self.sock.sendto(hdr + payload, ("127.0.0.1", self.base + dst))
            self.sent_to[dst] += 1

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
