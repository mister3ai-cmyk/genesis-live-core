# Business Source License 1.1 - см. research/LICENSE
"""APSA L3.5 - санитарный аудит поведения узла (swarm-sim v2.0-anis).

Строк: ~100 · Дата: 09.11.2026 · Модель, не движок.

Индекс саботажа по ТЗ ANIS v2.0:  S = (A / max(1, F)) * (A * 1.5)
  A - аномальные действия наблюдаемого узла в окне,
  F - события «естественного трения» в том же окне.

Два источника аномалий, оба судятся только СВОИМ датчиком судьи:
  - accuser (T7): узел опроверг событие, которое мой датчик признал верным.
    F - проверенно-верные события окна, которые он не опроверг: молчание -
    согласие. Окно - последние `win` проверенно-верных событий судьи.
  - author (T4): кинематический скачок координаты цели против текущего
    победителя слота. F - его же события по координатам без скачка.

Правило INTENTIONAL (rule):
  "tz"    - S >= 0.70, буквально по ТЗ. Порог по доле зависит от окна:
            из 1.5*A^2/(W-A) >= 0.7 следует A ~ sqrt(0.47*W) - чем больше
            окно, тем ниже доля, при которой честный узел с шумом хоронится
            (при W = 200 это ~5 %). Замер v2.0: честные хоронятся при шуме 3 %.
  "share" - доля A/(A+F) >= 0.25 и A >= 5 (окно v1.3: 0 штрафов честным
            в 1050 прогонах при шуме 0-5 %). Не зависит от размера окна.

Окно обвинителя (win_kind), режим "count":
  "arrival" (v2.0) - удары, ПРИШЕДШИЕ за время последних win проверенно-верных
            событий. Дыра: в тишине окно стоит, а PULL досылает пачку старых
            шумовых опровержений - доля раздувается, честный хоронится
            (chain 20 %, isolate, шум 5 %, зерно 8: 25 % на 575.6 с).
  "event"   (v2.1) - сколько ИМЕННО событий из окна последних win проверенно-
            верных опроверг узел. Опровержение старого события вне окна на
            долю не влияет; пришедшее раньше события - засчитывается, когда
            событие дойдёт до проверки.

Режимы окна:
  "count" (по умолчанию) - окно из `win` событий; прогрев: F + A >= warmup
           и с первого наблюдения прошло >= warm_ms;
  "time"  - буквально по ТЗ: окно dt_ms по времени, без прогрева.
"""
from collections import deque

NOMINAL, ACCIDENTAL, SUSPICIOUS, INTENTIONAL = (
    "NOMINAL", "ACCIDENTAL", "SUSPICIOUS", "INTENTIONAL")


def s_index(a, f, gamma=1.5):
    return (a / max(1, f)) * (a * gamma)


def classify_behavior(f_events, a_forced, thresh=0.70, gamma=1.5):
    """Канон ТЗ: NOMINAL / ACCIDENTAL / SUSPICIOUS / INTENTIONAL."""
    if f_events == 0 and a_forced == 0:
        return NOMINAL
    if s_index(a_forced, f_events, gamma) >= thresh:
        return INTENTIONAL
    return SUSPICIOUS if a_forced > 0 else ACCIDENTAL


class APSASanitaryAudit:
    def __init__(self, win=200, mode="count", thresh=0.70, gamma=1.5,
                 warmup=10, warm_ms=5000, dt_ms=2000, rule="tz",
                 share=0.25, min_a=5, win_kind="arrival"):
        self.win, self.mode, self.thresh, self.gamma = win, mode, thresh, gamma
        self.rule, self.share, self.min_a = rule, share, min_a
        self.warmup, self.warm_ms, self.dt_ms = warmup, warm_ms, dt_ms
        self.true_seq = 0                # проверенно-верных событий у судьи
        self.true_times = deque()        # их время (режим "time")
        self.acc = {}                    # обвинитель -> deque отметок ударов
        self.aut = {}                    # автор -> deque (отметка, скачок?)
        self.first = {}                  # (источник, узел) -> первое наблюдение
        self.win_kind = win_kind
        self.ev_win = deque()            # "event": eid последних win проверенно-верных
        self.ev_ref = {}                 # eid в окне -> кто его опроверг
        self.ev_cnt = {}                 # обвинитель -> сколько событий окна опроверг
        self.ev_pend = {}                # eid ещё не проверен -> кто уже опроверг

    # -------------------------------------------------- наблюдения судьи
    def saw_true(self, now_ms, eid=None):
        self.true_seq += 1
        if self.mode == "time":
            self.true_times.append(now_ms)
        if self.win_kind == "event" and eid is not None and eid not in self.ev_ref:
            refs = self.ev_pend.pop(eid, set())
            self.ev_win.append(eid)
            self.ev_ref[eid] = refs
            for v in refs:
                self.ev_cnt[v] = self.ev_cnt.get(v, 0) + 1
            while len(self.ev_win) > self.win:
                for v in self.ev_ref.pop(self.ev_win.popleft(), ()):
                    self.ev_cnt[v] -= 1

    def false_refutation(self, v, now_ms, eid=None):
        self.first.setdefault(("acc", v), now_ms)
        if self.win_kind == "event" and self.mode == "count":
            refs = self.ev_ref.get(eid)
            if refs is None:
                self.ev_pend.setdefault(eid, set()).add(v)
            elif v not in refs:
                refs.add(v)
                self.ev_cnt[v] = self.ev_cnt.get(v, 0) + 1
            return
        self.acc.setdefault(v, deque()).append(
            now_ms if self.mode == "time" else self.true_seq)

    def kinematic(self, a, jump, now_ms):
        self.first.setdefault(("aut", a), now_ms)
        q = self.aut.setdefault(a, deque())
        q.append((now_ms, bool(jump)))
        if self.mode == "count" and len(q) > self.win:
            q.popleft()

    # -------------------------------------------------------- счёт окна
    def counts(self, v, now_ms, src="acc"):
        """(A, F) по окну."""
        if src == "acc":
            if self.win_kind == "event" and self.mode == "count":
                a = self.ev_cnt.get(v, 0)
                return a, max(len(self.ev_win) - a, 0)
            q = self.acc.get(v, ())
            if self.mode == "time":
                lo = now_ms - self.dt_ms
                while q and q[0] <= lo:
                    q.popleft()
                while self.true_times and self.true_times[0] <= lo:
                    self.true_times.popleft()
                base = len(self.true_times)
            else:
                while q and q[0] <= self.true_seq - self.win:
                    q.popleft()
                base = min(self.true_seq, self.win)
            a = len(q)
            return a, max(base - a, 0)
        q = self.aut.get(v, ())
        if self.mode == "time":
            while q and q[0][0] <= now_ms - self.dt_ms:
                q.popleft()
        a = sum(1 for _, j in q if j)
        return a, len(q) - a

    def verdict(self, v, now_ms, src="acc"):
        """(статус, S, A, F); в режиме "count" до прогрева - не выше ACCIDENTAL.
        Прогрев (решение пользователя): F + A >= warmup ИЛИ наблюдение >= warm_ms."""
        a, f = self.counts(v, now_ms, src)
        t0 = self.first.get((src, v))
        warm = (f + a >= self.warmup
                or (t0 is not None and now_ms - t0 >= self.warm_ms))
        if self.mode == "count" and not warm:
            return (NOMINAL if a == 0 else ACCIDENTAL), 0.0, a, f
        sv = s_index(a, f, self.gamma)
        if self.rule == "share":
            frac = a / max(a + f, 1)
            st = (INTENTIONAL if frac >= self.share and a >= self.min_a
                  else SUSPICIOUS if a > 0 else
                  (NOMINAL if f == 0 else ACCIDENTAL))
            return st, frac, a, f
        return classify_behavior(f, a, self.thresh, self.gamma), sv, a, f
