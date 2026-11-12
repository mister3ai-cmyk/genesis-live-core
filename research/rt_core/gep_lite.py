# Business Source License 1.1 - см. research/LICENSE
"""GEP v0.1 Lite: компактный подписанный бинарный кадр для телеметрии роя
(100-200 Гц) и защитный слой узла перед C99-шлюзом.

Кадр (сетевой порядок байт, big-endian), заголовок 22 байта:

  смещ. размер поле
    0     2    magic  b"GL"
    2     1    version = 1
    3     1    type: 1 CLAIM, 2 REFUTE, 3 TOMBSTONE, 4 SYNC_PING
    4     1    author - номер узла (ключ Ed25519 из реестра)
    5     1    flags (0, зарезервировано)
    6     2    target - номер цели
    8     4    seq    - номер кадра автора, строго растёт
   12     8    t_us   - метка общего времени, мкс (int64)
   20     2    plen   - длина полезной нагрузки
   22  plen    payload (длина фиксирована для типа)
  22+plen 64   подпись Ed25519 автора над байтами [0, 22 + plen)

Полезная нагрузка:
  CLAIM      24 B: x, y, z (float64)                            кадр 110 B
  REFUTE      8 B: суть-автор u8, причина u8, резерв u16, seq заявки u32   94 B
  TOMBSTONE  12 B: похоронен u8, причина u8, улик u16, с какого t_us i64   98 B
  SYNC_PING  16 B: t_send_us i64, t_echo_us i64                  102 B

Проверка при приёме строго по порядку: длина -> magic -> версия -> тип ->
plen согласован с типом и длиной -> автор в реестре -> подпись Ed25519 ->
повтор (seq) -> метка из будущего. До подписи данные не используются.

Карантин (суверенный, локальный - GEP Q6.4):
  * битая подпись или испорченный кадр - автор НЕ установлен (поле author
    может подставить кто угодно), поэтому в карантин идёт транспортный
    источник (адрес), а не автор: иначе мусором «от имени» честного узла
    его можно было бы изолировать. Карантин источника - за серию
    (src_viol_n нарушений за src_viol_window) и на срок src_ttl, потом
    реабилитация: единичная порча в канале не отрезает борт навсегда;
  * подписанная метка из будущего (> max_skew) - автор (подпись доказывает
    авторство);
  * подтверждённое кинематическое вето - автор: veto_k опровержений его
    заявок шлюзом за veto_window. Его отсчёты удаляются из колец
    (rt_kin_forget_author), заявки дальше не принимаются, рассылается
    TOMBSTONE.
Отступ надгробий (из v4.0-B): одно надгробие одному соседу - через 0, 1, 3,
7, 15, 31 с, далее раз в 30 с.
"""
import hashlib
import struct

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

MAGIC, VERSION = b"GL", 1
CLAIM, REFUTE, TOMBSTONE, SYNC_PING = 1, 2, 3, 4
HDR = struct.Struct(">2sBBBBHIqH")
SIG_LEN = 64
PAYLOAD = {CLAIM: struct.Struct(">ddd"), REFUTE: struct.Struct(">BBHI"),
           TOMBSTONE: struct.Struct(">BBHq"), SYNC_PING: struct.Struct(">qq")}
REASON_TRAJ_VETO, REASON_SENSOR = 1, 2
assert HDR.size == 22
_PK_CACHE = {}          # байты ключа -> разобранный ключ (разбор дорогой)
REPLAY_WIN = 64         # окно повторов: столько номеров назад кадр ещё принимается


class FrameError(ValueError):
    """Кадр отвергнут; reason - машинный код причины."""

    def __init__(self, reason, signed=False):
        super().__init__(reason)
        self.reason = reason
        self.signed = signed      # True - подпись проверена, автор достоверен


def keypair(seed):
    """Детерминированный ключ узла из зерна (для стенда)."""
    sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"gep-lite:%d" % seed).digest())
    pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sk, pk


def pack(sk, ftype, author, target, seq, t_us, *payload):
    body = PAYLOAD[ftype].pack(*payload)
    msg = HDR.pack(MAGIC, VERSION, ftype, author, 0, target, seq, t_us, len(body)) + body
    return msg + sk.sign(msg)


def unpack(frame, pubkeys):
    """-> dict(type, author, target, seq, t_us, payload). FrameError при
    любой ошибке; до проверки подписи содержимое не возвращается."""
    if len(frame) < HDR.size + SIG_LEN:
        raise FrameError("short")
    magic, ver, ftype, author, _flags, target, seq, t_us, plen = HDR.unpack_from(frame, 0)
    if magic != MAGIC:
        raise FrameError("magic")
    if ver != VERSION:
        raise FrameError("version")
    st = PAYLOAD.get(ftype)
    if st is None:
        raise FrameError("type")
    if plen != st.size or len(frame) != HDR.size + plen + SIG_LEN:
        raise FrameError("length")
    pk = pubkeys.get(author)
    if pk is None:
        raise FrameError("author")
    msg, sig = frame[:HDR.size + plen], frame[HDR.size + plen:]
    key = _PK_CACHE.get(pk)
    if key is None:
        key = _PK_CACHE[pk] = Ed25519PublicKey.from_public_bytes(pk)
    try:
        key.verify(sig, msg)
    except InvalidSignature:
        raise FrameError("signature") from None
    return dict(type=ftype, author=author, target=target, seq=seq, t_us=t_us,
                payload=st.unpack_from(frame, HDR.size))


class TombBackoff:
    """Когда слать надгробие соседу: 0, 1, 3, 7, 15, 31 с, далее раз в 30 с."""

    def __init__(self, first_gap_us=1_000_000, max_gap_us=30_000_000):
        self.first, self.max, self.st = first_gap_us, max_gap_us, {}

    def due(self, peer, tomb_id, now_us):
        nxt, gap = self.st.get((peer, tomb_id), (0, self.first))
        if now_us < nxt:
            return False
        self.st[(peer, tomb_id)] = (now_us + gap, min(2 * gap, self.max))
        return True


class Node:
    """Узел шины: подписывает свои кадры, проверяет чужие, держит карантин и
    кормит C99-шлюз (kin - rt_core.Kin или None для транспорта без шлюза)."""

    def __init__(self, me, sk, pubkeys, kin=None, max_skew_us=50_000, veto_k=3, veto_window_us=10_000_000,
                 src_viol_n=3, src_viol_window_us=10_000_000, src_ttl_us=30_000_000):
        self.me, self.sk, self.pub, self.kin = me, sk, dict(pubkeys), kin
        self.max_skew, self.veto_k, self.veto_window = max_skew_us, veto_k, veto_window_us
        self.src_viol_n, self.src_viol_window, self.src_ttl = src_viol_n, src_viol_window_us, src_ttl_us
        self.seq = 0
        self.last_seq = {}            # автор -> (старший seq, битовая карта окна)
        self.q_authors = {}           # автор -> причина (бессрочно: атака подписана)
        self.q_sources = {}           # транспортный адрес -> до какого момента, мкс
        self.src_viol = {}            # адрес -> метки нарушений формата/подписи
        self.vetoes = {}              # автор -> метки времени опровержений
        self.targets_seen = set()
        self.tombs = []               # надгробия к рассылке: (кадр, id)
        self.backoff = TombBackoff()
        self.stats = {}

    def _count(self, k):
        self.stats[k] = self.stats.get(k, 0) + 1

    def frame(self, ftype, target, t_us, *payload):
        self.seq += 1
        return pack(self.sk, ftype, self.me, target, self.seq, t_us, *payload)

    def quarantine_author(self, a, reason, now_us):
        if a in self.q_authors or a == self.me:
            return
        self.q_authors[a] = reason
        if self.kin is not None:
            for tg in self.targets_seen:
                self.kin.lib.rt_kin_forget_author(self.kin.buf, tg, a)
        ev = len(self.vetoes.get(a, ()))
        tomb = self.frame(TOMBSTONE, 0, now_us, a, reason, ev, now_us)
        self.tombs.append((tomb, (a, reason)))
        self._count("quarantine_author")

    def _fresh(self, a, seq):
        """Окно повторов как в IPsec: старший seq и битовая карта
        REPLAY_WIN предыдущих. Кадры, обогнавшие друг друга в сети, не
        теряются; точный дубль и слишком старый номер - повтор."""
        top, bits = self.last_seq.get(a, (0, 0))
        if seq > top:
            shift = seq - top
            bits = ((bits << shift) | (1 << (shift - 1))) & ((1 << REPLAY_WIN) - 1) if shift <= REPLAY_WIN else 0
            self.last_seq[a] = (seq, bits)
            return True
        back = top - seq
        if back == 0 or back > REPLAY_WIN or bits >> (back - 1) & 1:
            return False
        self.last_seq[a] = (top, bits | (1 << (back - 1)))
        return True

    def tombs_for(self, peer, now_us):
        return [f for f, tid in self.tombs if self.backoff.due(peer, tid, now_us)]

    def receive(self, frame, src, now_us, mine=None):
        """Принять кадр. mine(target, t_us) -> моё измерение или None.
        Возвращает (что сделано, кадр-ответ или None)."""
        until = self.q_sources.get(src)
        if until is not None:
            if now_us < until:
                self._count("drop_source")
                return "drop_source", None
            del self.q_sources[src]                       # срок вышел - реабилитация
            self.src_viol.pop(src, None)
            self._count("source_released")
        try:
            f = unpack(frame, self.pub)
        except FrameError as e:
            self._count("reject_" + e.reason)
            if e.reason in ("signature", "length", "magic", "version", "type", "short"):
                # автор не установлен - виноват источник; но один битый кадр
                # бывает и от порчи в канале: карантин только за серию, и на срок
                vs = [t for t in self.src_viol.get(src, []) if t > now_us - self.src_viol_window] + [now_us]
                self.src_viol[src] = vs
                if len(vs) >= self.src_viol_n:
                    self.q_sources[src] = now_us + self.src_ttl
                    self._count("quarantine_source")
            return "reject_" + e.reason, None
        a = f["author"]
        if a in self.q_authors:
            self._count("drop_quarantined")
            return "drop_quarantined", None
        if not self._fresh(a, f["seq"]):
            self._count("replay")
            return "replay", None
        if f["t_us"] > now_us + self.max_skew:
            self.quarantine_author(a, 3, now_us)          # подписанная метка из будущего
            return "quarantine_future", None
        if f["type"] == TOMBSTONE:
            self._count("tombstone_in")
            return "tombstone_in", None
        if f["type"] != CLAIM:
            self._count("other_in")
            return "other_in", None
        tg, claim = f["target"], f["payload"]
        self.targets_seen.add(tg)
        verdict, reply = "pass", None
        if self.kin is not None:
            m = mine(tg, f["t_us"]) if mine else None
            if m is not None:
                v, _ = self.kin.gate(tg, a, f["t_us"], now_us, claim, m)
                if v == 1:                                # RT_REFUTE
                    verdict = "refute"
                    reply = self.frame(REFUTE, tg, now_us, a, REASON_TRAJ_VETO, 0, f["seq"] & 0xFFFFFFFF)
                    vs = [t for t in self.vetoes.get(a, []) if t > now_us - self.veto_window] + [now_us]
                    self.vetoes[a] = vs
                    if len(vs) >= self.veto_k:
                        self.quarantine_author(a, REASON_TRAJ_VETO, now_us)
                        return "quarantine_veto", reply
                elif v == 2:
                    verdict = "failsafe"
            if verdict != "refute":
                self.kin.push(tg, a, f["t_us"], claim, now_us)
        self._count(verdict)
        return verdict, reply
