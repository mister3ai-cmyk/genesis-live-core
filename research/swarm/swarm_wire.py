#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Business Source License 1.1 - см. research/LICENSE
"""
research/swarm/swarm_wire.py  (v1.1)

Пакеты событий с одной подписью на корень Меркла и бинарный провод.

Семантика GEP не меняется: лист дерева - это прежний event_id, то есть
sha256 от канонического JSON конверта без подписи. Подписывается канон
заголовка пакета {gep, kind:"batch", author, body:{root, hlc, n}}. Бинарный
формат - только упаковка: получатель восстанавливает из него канонический
JSON, сам считает event_id и корень, и только потом проверяет подпись.
Всё, что в бинарный формат не ложится (чужая схема тела), - ValueError на
отправке и отказ на приёме.

Типы записей в датаграмме (первый байт записи):
  1  пакет событий: автор, n, HLC пакета, подпись 64 B, режим
       0 = полный: все n листов подряд, корень считается из них
       1 = частичный: m событий, у каждого индекс и путь Меркла
  2  JSON-конверт как есть (key_event: редкий, подписан отдельно)
  3  digest: version vector, отзывы, карта слотов по 3 B, подпись 64 B
"""
import hashlib

GEP = "0.1"
STATUSES = ("clear", "obstacle", "hazard", "unknown")
SIG = 64
BITMAP_BYTES = 16               # = swarm_node.BITMAP / 8


# ------------------------------------------------------------------ Меркл

def _h2(a, b):
    return hashlib.sha256(b"\x01" + a + b).digest()


def merkle_root(leaves):
    lv = list(leaves)
    if not lv:
        raise ValueError("empty batch")
    while len(lv) > 1:
        nxt = [_h2(lv[i], lv[i + 1]) for i in range(0, len(lv) - 1, 2)]
        if len(lv) % 2:
            nxt.append(lv[-1])          # нечётный лист поднимается без пары
        lv = nxt
    return lv[0]


def merkle_proof(leaves, i):
    lv, path = list(leaves), []
    while len(lv) > 1:
        if not (i == len(lv) - 1 and len(lv) % 2):
            path.append(lv[i ^ 1])
        nxt = [_h2(lv[j], lv[j + 1]) for j in range(0, len(lv) - 1, 2)]
        if len(lv) % 2:
            nxt.append(lv[-1])
        lv, i = nxt, i // 2
    return path


def root_from_proof(leaf, i, n, path):
    cur, m, p = leaf, n, 0
    if not 0 <= i < n:
        raise ValueError("leaf index")
    while m > 1:
        if i == m - 1 and m % 2:
            pass
        else:
            sib = path[p]
            p += 1
            cur = _h2(cur, sib) if i % 2 == 0 else _h2(sib, cur)
        i, m = i // 2, (m + 1) // 2
    if p != len(path):
        raise ValueError("proof length")
    return cur


def batch_header(author, root, hlc, n):
    """То, что подписывается: канон заголовка пакета."""
    return {"gep": GEP, "kind": "batch", "author": author,
            "body": {"root": root.hex(), "hlc": [hlc[0], hlc[1]], "n": n}}


# --------------------------------------------------------------- varint

def _uv(out, x):
    if x < 0:
        raise ValueError("negative uvarint")
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def _sv(out, x):
    _uv(out, (x << 1) if x >= 0 else ((-x << 1) - 1))


def _ruv(d, p):
    x = s = 0
    while True:
        b = d[p]
        p += 1
        x |= (b & 0x7F) << s
        if not b & 0x80:
            return x, p
        s += 7
        if s > 70:
            raise ValueError("varint too long")


def _rsv(d, p):
    u, p = _ruv(d, p)
    return ((u >> 1) if not u & 1 else -((u + 1) >> 1)), p


# ------------------------------------------------------------- события

class Codec:
    """Таблицы, общие для всего роя: индекс ключа и индекс автора."""

    def __init__(self, keys_list, authors):
        self.keys = list(keys_list)
        self.kidx = {k: i for i, k in enumerate(self.keys)}
        self.authors = list(authors)

    # тело события -> байты (автор и node в заголовке пакета)
    def enc_event(self, out, kind, body, node, bl):
        if body.get("node") != node:
            raise ValueError("node != batch author")
        hl, hc = body["hlc"]
        ks = 2 if "kseq" in body else 0     # v1.2: типы 3/4 несут kseq
        if kind == "glyph":
            if set(body) - {"kseq"} != {"key", "value", "hlc", "node", "seq"}:
                raise ValueError("glyph schema")
            out.append(1 + ks)
            out.append(self.kidx[body["key"]])
            v = body["value"]
            if isinstance(v, bool) or v is None:
                raise ValueError("value type")
            if isinstance(v, int):
                out.append(0)
                _sv(out, v)
            elif isinstance(v, str):
                out.append(2)
                out.append(STATUSES.index(v))
            elif isinstance(v, list) and all(
                    isinstance(x, int) and not isinstance(x, bool) for x in v):
                out.append(1)
                out.append(len(v))
                for x in v:
                    _sv(out, x)
            else:
                raise ValueError("value type")
        elif kind == "validation":
            if set(body) - {"kseq"} != {"glyph_hash", "verdict", "evidence",
                                        "hlc", "node", "seq"}:
                raise ValueError("validation schema")
            gh = body["glyph_hash"]
            if not gh.startswith("sha256:") or len(gh) != 71:
                raise ValueError("glyph_hash")
            out.append(2 + ks)
            out += bytes.fromhex(gh[7:])
            evd = body["evidence"]
            ai = evd.get("applied_in")
            note = evd.get("note")
            if set(evd) != {"applied_in", "note"} or not ai.startswith("sensor:") \
                    or not (note is None or (isinstance(note, list) and len(note) == 2
                            and all(isinstance(x, int) and not isinstance(x, bool)
                                    for x in note))):
                raise ValueError("evidence")
            # v3: бит 2 - в голосе измерение [x, y]
            out.append(("confirmed", "refuted").index(body["verdict"])
                       | (2 if note is not None else 0))
            out.append(self.kidx[ai[7:]])
            if note is not None:
                _sv(out, note[0])
                _sv(out, note[1])
        else:
            raise ValueError("kind %r" % kind)
        _sv(out, hl - bl)
        _uv(out, hc)
        _uv(out, body["seq"])
        if ks:
            if not 0 <= body["seq"] - body["kseq"]:
                raise ValueError("kseq > seq")
            _uv(out, body["seq"] - body["kseq"])

    def dec_event(self, d, p, node, bl):
        t = d[p]
        p += 1
        ks = t in (3, 4)
        if ks:
            t -= 2
        if t == 1:
            key = self.keys[d[p]]
            p += 1
            tag = d[p]
            p += 1
            if tag == 0:
                v, p = _rsv(d, p)
            elif tag == 2:
                v = STATUSES[d[p]]
                p += 1
            elif tag == 1:
                ln = d[p]
                p += 1
                v = []
                for _ in range(ln):
                    x, p = _rsv(d, p)
                    v.append(x)
            else:
                raise ValueError("value tag")
            kind, body = "glyph", {"key": key, "value": v}
        elif t == 2:
            gh = "sha256:" + d[p:p + 32].hex()
            if len(gh) != 71:
                raise ValueError("short hash")
            p += 32
            if d[p] > 3:
                raise ValueError("verdict")
            verdict = ("confirmed", "refuted")[d[p] & 1]
            has = d[p] & 2
            key = self.keys[d[p + 1]]
            p += 2
            note = None
            if has:
                x, p = _rsv(d, p)
                y, p = _rsv(d, p)
                note = [x, y]
            kind, body = "validation", {
                "glyph_hash": gh, "verdict": verdict,
                "evidence": {"applied_in": "sensor:" + key, "note": note}}
        else:
            raise ValueError("event type")
        dl, p = _rsv(d, p)
        hc, p = _ruv(d, p)
        seq, p = _ruv(d, p)
        body["hlc"] = [bl + dl, hc]
        body["node"] = node
        body["seq"] = seq
        if ks:
            dk, p = _ruv(d, p)
            body["kseq"] = seq - dk
        return kind, body, p

    # ------------------------------------------------------------ пакет
    def enc_group(self, aidx, n, hlc, sig, items, complete):
        """items: [(i, kind, body, proof)]; при complete - все n по порядку."""
        out = bytearray([1, aidx])
        _uv(out, n)
        _uv(out, hlc[0])
        _uv(out, hlc[1])
        if len(sig) != SIG:
            raise ValueError("sig length")
        out += sig
        if complete:
            out.append(0)
            for _, kind, body, _ in items:
                self.enc_event(out, kind, body, aidx, hlc[0])
        else:
            out.append(1)
            _uv(out, len(items))
            for i, kind, body, proof in items:
                _uv(out, i)
                self.enc_event(out, kind, body, aidx, hlc[0])
                out.append(len(proof))
                for h in proof:
                    out += h
        return bytes(out)

    def dec_group(self, d, p):
        aidx = d[p]
        p += 1
        if aidx >= len(self.authors):
            raise ValueError("author index")
        n, p = _ruv(d, p)
        hl, p = _ruv(d, p)
        hc, p = _ruv(d, p)
        sig = bytes(d[p:p + SIG])
        if len(sig) != SIG:
            raise ValueError("short sig")
        p += SIG
        mode = d[p]
        p += 1
        items = []
        if mode == 0:
            if not 0 < n <= 255:
                raise ValueError("batch size")
            for i in range(n):
                kind, body, p = self.dec_event(d, p, aidx, hl)
                items.append((i, kind, body, None))
        elif mode == 1:
            m, p = _ruv(d, p)
            for _ in range(m):
                i, p = _ruv(d, p)
                kind, body, p = self.dec_event(d, p, aidx, hl)
                k = d[p]
                p += 1
                proof = [bytes(d[p + 32 * j:p + 32 * j + 32]) for j in range(k)]
                if any(len(h) != 32 for h in proof):
                    raise ValueError("short proof")
                p += 32 * k
                items.append((i, kind, body, proof))
        else:
            raise ValueError("mode")
        return {"aidx": aidx, "n": n, "hlc": [hl, hc], "sig": sig,
                "items": items, "complete": mode == 0}, p

    # ----------------------------------------------------------- digest
    def enc_digest(self, aidx, hlc, vv, rev, sl, sign):
        out = bytearray([3, aidx])
        _uv(out, hlc[0])
        _uv(out, hlc[1])
        for a in range(len(self.authors)):
            c, bits = vv[a]
            _uv(out, c)
            out += bits.to_bytes(BITMAP_BYTES, "big")
        out.append(len(rev))
        out += bytes(rev)
        if sl is None:
            out.append(0)
        else:
            out.append(1)
            for w in sl:
                out += w
        out += sign(bytes(out))
        return bytes(out)

    def dec_digest(self, d, p):
        p0 = p - 1
        aidx = d[p]
        p += 1
        hl, p = _ruv(d, p)
        hc, p = _ruv(d, p)
        vv = {}
        for a in range(len(self.authors)):
            c, p = _ruv(d, p)
            bits = int.from_bytes(d[p:p + BITMAP_BYTES], "big")
            p += BITMAP_BYTES
            vv[str(a)] = [c, "%x" % bits]
        k = d[p]
        p += 1
        rev = list(d[p:p + k])
        p += k
        has = d[p]
        p += 1
        sl = ""
        if has:
            raw = d[p:p + 3 * len(self.keys)]
            if len(raw) != 3 * len(self.keys):
                raise ValueError("short slot map")
            p += 3 * len(self.keys)
            sl = "".join("------" if raw[3 * i:3 * i + 3] == b"\0\0\0"
                         else raw[3 * i:3 * i + 3].hex()
                         for i in range(len(self.keys)))
        signed = bytes(d[p0:p])
        sig = bytes(d[p:p + SIG])
        if len(sig) != SIG:
            raise ValueError("short sig")
        p += SIG
        body = {"vv": vv, "rev": rev, "node": aidx, "hlc": [hl, hc]}
        if has:
            body["sl"] = sl
        return {"aidx": aidx, "body": body, "signed": signed, "sig": sig}, p

    # ------------------------------------------------------ датаграмма
    def decode(self, d):
        out, p = [], 0
        while p < len(d):
            t = d[p]
            p += 1
            if t == 1:
                g, p = self.dec_group(d, p)
                out.append(("group", g))
            elif t == 2:
                ln, p = _ruv(d, p)
                out.append(("json", bytes(d[p:p + ln])))
                p += ln
            elif t == 3:
                g, p = self.dec_digest(d, p)
                out.append(("digest", g))
            else:
                raise ValueError("record type %d" % t)
        return out


def enc_json(raw):
    out = bytearray([2])
    _uv(out, len(raw))
    out += raw
    return bytes(out)


def pack_records(records, room):
    """Склеивает записи в полезные нагрузки датаграмм <= room байт.
    Возвращает [(нагрузка, событий, подписей)]: каждая запись несёт ровно
    одну подпись (пакет, digest или key_event). Запись больше room - ошибка:
    дробить пакеты обязан кодировщик."""
    out, cur, ne, ns = [], bytearray(), 0, 0
    for rec, n_ev in records:
        if len(rec) > room:
            raise ValueError("record of %d bytes > %d" % (len(rec), room))
        if cur and len(cur) + len(rec) > room:
            out.append((bytes(cur), ne, ns))
            cur, ne, ns = bytearray(), 0, 0
        cur += rec
        ne += n_ev
        ns += 1
    if cur:
        out.append((bytes(cur), ne, ns))
    return out
