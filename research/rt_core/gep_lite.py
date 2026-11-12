# Business Source License 1.1 - см. research/LICENSE
"""GEP v0.1 Lite: компактный подписанный бинарный кадр для телеметрии роя
(100-200 Гц) и защитный слой узла перед C99-шлюзом.

Кадр (сетевой порядок байт, big-endian), заголовок 22 байта:

  смещ. размер поле
    0     2    magic  b"GL"
    2     1    version = 1
    3     1    type: 1 CLAIM, 2 REFUTE, 3 TOMBSTONE, 4 SYNC_PING
    4     1    author - номер узла (ключ Ed25519 из реестра)
    5     1    flags: у SYNC_PING бит 0 - PONG, бит 1 - часы автора
                 синхронизированы; у остальных типов 0
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
             PING (flags 0): target - сосед или BCAST, t_us = t1, нагрузка (t1, 0);
             PONG (flags 1): target - инициатор, t_us = t3, нагрузка (t1, t2)

Проверка при приёме строго по порядку: длина -> magic -> версия -> тип ->
plen согласован с типом и длиной -> автор в реестре -> подпись Ed25519 ->
повтор (seq) -> метка из будущего (кроме SYNC_PING: в нём метка - сама
измеряемая величина). До подписи данные не используются.

Карантин (суверенный, локальный - GEP Q6.4):
  * битая подпись или испорченный кадр - автор НЕ установлен (поле author
    может подставить кто угодно), поэтому в карантин идёт транспортный
    источник (адрес), а не автор: иначе мусором «от имени» честного узла
    его можно было бы изолировать. Карантин источника - за серию
    (src_viol_n нарушений за src_viol_window) и на срок src_ttl, потом
    реабилитация: единичная порча в канале не отрезает борт навсегда;
  * подписанная метка из будущего (> max_skew) - кадр отбрасывается;
    автор уходит в карантин, только если шлёт «будущее» дольше
    future_grace и мои часы синхронизированы (ClockSync.locked): иначе
    «будущее» может означать, что отстаю я;
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
REASON_TRAJ_VETO, REASON_SENSOR, REASON_FUTURE = 1, 2, 3
FLAG_PONG, FLAG_LOCKED = 0x01, 0x02          # поле flags кадра SYNC_PING
BCAST = 0xFFFF                                # target группового PING
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


def pack(sk, ftype, author, target, seq, t_us, *payload, flags=0):
    body = PAYLOAD[ftype].pack(*payload)
    msg = HDR.pack(MAGIC, VERSION, ftype, author, flags, target, seq, t_us, len(body)) + body
    return msg + sk.sign(msg)


def unpack(frame, pubkeys):
    """-> dict(type, author, target, seq, t_us, payload). FrameError при
    любой ошибке; до проверки подписи содержимое не возвращается."""
    if len(frame) < HDR.size + SIG_LEN:
        raise FrameError("short")
    magic, ver, ftype, author, flags, target, seq, t_us, plen = HDR.unpack_from(frame, 0)
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
    return dict(type=ftype, author=author, flags=flags, target=target, seq=seq, t_us=t_us,
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


def ntp(t1, t2, t3, t4):
    """Задержка и смещение соседа относительно меня по 4 меткам (NTP):
    delta = (t4 - t1) - (t3 - t2), theta = ((t2 - t1) + (t3 - t4)) / 2.
    theta > 0 - часы соседа впереди моих. Точность - половина асимметрии
    задержек туда и обратно."""
    return (t4 - t1) - (t3 - t2), ((t2 - t1) + (t3 - t4)) / 2.0


class ClockSync:
    """Поправка к своим часам по медиане смещений соседей.

    - PING каждому соседу раз в ping_period_us; на чужие PING отвечаю не
      чаще pong_per_s раз в секунду на соседа (защита от флуда; 3 при
      периоде 0.5 с - запас на джиттер доставки);
    - по соседу хранятся keep последних замеров, берётся theta с минимальной
      delta (меньше всего асимметрии джиттера), нужно >= min_samples;
    - медиана по >= min_peers соседям (мои часы не участвуют): один-два лжеца
      из шести её не сдвигают;
    - |медиана| > step_us - шаг поправки, иначе при > slew_min_us - половина
      медианы; после поправки замеры сбрасываются (они в старой шкале);
    - locked: медиана посчитана и |медиана| <= step_us."""

    def __init__(self, me, peers, ping_period_us=500_000, pong_per_s=3, keep=8, min_samples=2,
                 min_peers=3, step_us=20_000, slew_min_us=2_000):
        self.me, self.peers = me, [p for p in peers if p != me]
        self.period, self.pong_per_s, self.keep = ping_period_us, pong_per_s, keep
        self.min_samples, self.min_peers, self.step, self.slew_min = min_samples, min_peers, step_us, slew_min_us
        self.offset_us = 0.0
        self.samples = {p: [] for p in self.peers}
        self.out = {}                     # сосед -> t1 последнего PING
        self.next_ping = {p: 0 for p in self.peers}
        self.pongs = {}                   # сосед -> метки моих ответов (для лимита)
        self.locked = False
        self.corrections = []             # (момент, поправка, мкс)

    def now(self, raw_us):
        return int(raw_us + self.offset_us)

    def pings_due(self, now_us):
        due = []
        for i, p in enumerate(self.peers):
            if now_us >= self.next_ping[p]:
                self.out[p] = now_us
                self.next_ping[p] = now_us + self.period
                due.append(p)
        return due

    def allow_pong(self, peer, now_us):
        ts = [t for t in self.pongs.get(peer, []) if t > now_us - 1_000_000]
        if len(ts) >= self.pong_per_s:
            self.pongs[peer] = ts
            return False
        self.pongs[peer] = ts + [now_us]
        return True

    def on_pong(self, peer, t1, t2, t3, t4):
        if peer not in self.samples or self.out.get(peer) != t1:
            return False                  # не мой запрос или устаревший
        del self.out[peer]
        delta, theta = ntp(t1, t2, t3, t4)
        if delta < 0:
            return False
        sm = self.samples[peer]
        sm.append((delta, theta))
        del sm[:-self.keep]
        self._adjust(t4)
        return True

    def median(self):
        best = [min(sm)[1] for sm in self.samples.values() if len(sm) >= self.min_samples]
        if len(best) < self.min_peers:
            return None, len(best)
        best.sort()
        k = len(best)
        return (best[k // 2] if k % 2 else 0.5 * (best[k // 2 - 1] + best[k // 2])), k

    def _adjust(self, now_us):
        med, k = self.median()
        if med is None:
            return
        if abs(med) > self.step:
            corr = med
        elif abs(med) > self.slew_min:
            corr = 0.5 * med
        else:
            self.locked = True
            return
        self.offset_us += corr
        self.corrections.append((now_us, corr))
        for p in self.samples:
            self.samples[p] = []
        self.out.clear()                  # PING в полёте несут t1 в старой шкале
        self.locked = abs(med) <= self.step


class Node:
    """Узел шины: подписывает свои кадры, проверяет чужие, держит карантин и
    кормит C99-шлюз (kin - rt_core.Kin или None для транспорта без шлюза)."""

    def __init__(self, me, sk, pubkeys, kin=None, max_skew_us=50_000, veto_k=3, veto_window_us=10_000_000,
                 src_viol_n=3, src_viol_window_us=10_000_000, src_ttl_us=30_000_000,
                 clock=None, future_grace_us=5_000_000):
        self.me, self.sk, self.pub, self.kin = me, sk, dict(pubkeys), kin
        self.clock = clock            # ClockSync или None (часы считаются верными)
        self.future_grace = future_grace_us
        self.future_first = {}        # автор -> момент первой метки из будущего
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

    def frame(self, ftype, target, t_us, *payload, flags=0):
        self.seq += 1
        return pack(self.sk, ftype, self.me, target, self.seq, t_us, *payload, flags=flags)

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

    def _sync(self, f, now_us):
        """SYNC_PING / PONG, адресованные мне; чужие - игнор."""
        a, pong = f["author"], f["flags"] & FLAG_PONG
        if self.clock is None or not (f["target"] == self.me or (f["target"] == BCAST and not pong)):
            self._count("sync_other")
            return "sync_other", None
        t1, t2 = f["payload"]
        if pong:
            ok = self.clock.on_pong(a, t1, t2, f["t_us"], now_us)
            self._count("pong_in" if ok else "pong_bad")
            return ("pong_in" if ok else "pong_bad"), None
        if not self.clock.allow_pong(a, now_us):
            self._count("ping_flood")
            return "ping_flood", None
        reply = self.frame(SYNC_PING, a, now_us, f["t_us"], now_us,
                           flags=FLAG_PONG | (FLAG_LOCKED if self.clock.locked else 0))
        self._count("ping_in")
        return "ping_in", reply

    def pings(self, now_us, group=False):
        """Кадры SYNC_PING, которым подошёл срок: по одному на соседа или
        (group) один на всех с target = BCAST."""
        if self.clock is None:
            return []
        due = self.clock.pings_due(now_us)
        fl = FLAG_LOCKED if self.clock.locked else 0
        if group:
            return [self.frame(SYNC_PING, BCAST, now_us, now_us, 0, flags=fl)] if due else []
        return [self.frame(SYNC_PING, peer, now_us, now_us, 0, flags=fl) for peer in due]

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
        if f["type"] == SYNC_PING:                        # метка в нём - измеряемая величина
            return self._sync(f, now_us)
        if f["t_us"] > now_us + self.max_skew:
            # Метка из будущего: кадр не принимается. Постоянный карантин -
            # только если автор не выправился за льготное окно и мои
            # собственные часы синхронизированы (иначе «будущее» - это я).
            first = self.future_first.setdefault(a, now_us)
            locked = self.clock is None or self.clock.locked
            if locked and now_us - first >= self.future_grace:
                self.quarantine_author(a, REASON_FUTURE, now_us)
                return "quarantine_future", None
            self._count("future_drop")
            return "future_drop", None
        self.future_first.pop(a, None)
        if f["type"] == TOMBSTONE:
            self._count("tombstone_in")
            return "tombstone_in", None
        if f["type"] != CLAIM:
            self._count("other_in")
            return "other_in", None
        if self.clock is not None and not self.clock.locked:
            # свои часы не выровнены: мой датчик читается не в тот момент,
            # шлюз не судит и кольцо не кормится
            self._count("unsynced")
            return "unsynced", None
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
