# Business Source License 1.1 - см. research/LICENSE
"""Tombstone Registry + Тимус (L4-L5) для swarm-sim v2.0-anis.

Строк: ~80 · Дата: 09.11.2026 · Модель, не движок.

Своё решение:
  - двухфазное ("two_phase"): первое INTENTIONAL -> SUSPICIOUS (голоса не
    учитываются, ничего не рассылается); надгробие - при повторном
    INTENTIONAL, когда наблюдение продвинулось не меньше чем на `gap`
    (окно обновилось, повтор не того же самого);
  - буквальное (ТЗ): первое INTENTIONAL -> надгробие сразу.
Чужое надгробие действует, когда его выставили `quorum` разных узлов, сами
не похороненные. Своё - сразу.
Тимус: перед новым надгробием
  - "bft":     похоронено + 1 <= (n - 1) // 3;
  - "literal": похоронено / n <= air (ТЗ: 0.01) - проверка ДО добавления.
Воскрешения нет.
"""


class TombstoneRegistry:
    def __init__(self, n, me, two_phase=True, quorum=2, thymus="bft",
                 air=0.01):
        self.n, self.me = n, me
        self.two_phase, self.quorum = two_phase, quorum
        self.thymus, self.air = thymus, air
        self.suspect = {}           # (источник, узел) -> позиция первого срабатывания
        self.dead = {}              # узел -> время, когда надгробие вступило в силу
        self.issued = set()         # кого похоронил я сам (рассылаю)
        self.votes = {}             # узел -> множество выставивших надгробие
        self.blocked = 0            # сколько надгробий заблокировал Тимус

    def is_dead(self, v):
        return v in self.dead

    def is_suspect(self, v):
        return any(k[1] == v for k in self.suspect)

    def _thymus_ok(self):
        k = len(self.dead)
        if self.thymus == "literal":
            return k / self.n <= self.air
        return k + 1 <= (self.n - 1) // 3

    def _bury(self, v, now_ms):
        if v in self.dead:
            return False
        if not self._thymus_ok():
            self.blocked += 1
            return False
        self.dead[v] = now_ms
        return True

    def intentional(self, src, v, pos, gap, now_ms):
        """Моё APSA сказало INTENTIONAL. True - надгробие выставлено мной."""
        if v in self.dead or v == self.me:
            return False
        key = (src, v)
        if self.two_phase:
            first = self.suspect.get(key)
            if first is None:
                self.suspect[key] = pos
                return False
            if pos - first < gap:
                return False
        if self._bury(v, now_ms):
            self.issued.add(v)
            self.votes.setdefault(v, set()).add(self.me)
            return True
        return False

    def foreign(self, issuer, v, now_ms):
        """Пришло чужое надгробие. True - вступило в силу у меня."""
        if issuer in self.dead or v == issuer:
            return False
        s = self.votes.setdefault(v, set())
        s.add(issuer)
        live = {x for x in s if x not in self.dead}
        if len(live) >= self.quorum:
            return self._bury(v, now_ms)
        return False
