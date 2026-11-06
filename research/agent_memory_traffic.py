#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/agent_memory_traffic.py

Does an agent with L0 memory cost less than an agent that carries everything in
its prompt, and does it stay accurate longer?

The first version of this stand measured memory traffic against a search that
hashed the query text to a random corpus index. The query meaning was thrown
away, so the agent was reacting to a broken tool and every number it produced
described that. This version fixes the three things that made the measurement
meaningless, following the architecture rather than convenience:

  honest retrieval   the query is embedded by a real sentence encoder, projected
                     by src/srp_lsh.c into a Hamming code, scanned, then exactly
                     re-ranked - the two-stage cascade as the engine does it

  closed corpus      the corpus is this project's own documents, which no model
                     has seen in training. Asking about public facts lets the
                     model answer from its weights and never touch memory, which
                     measures its disposition instead of the memory's value

  a control arm      "full context" carries every document it has ever been
                     handed in the prompt, which is what an agent without
                     retrieval must do. Without this arm there is nothing to
                     compare against and no saving to report

Reconsolidation fires on write only, never on read, per RFC-4.6: a read is O(1)
LSH plus SIMD and must stay microseconds. The 6.8 ms Oja/Newton-Schulz step
belongs to consolidation of new observations.

Three metrics decide it:

  token economy      prompt tokens per turn, memory arm against full context
  accuracy retention whether late turns still answer from the right document
  latency overhead   memory time as a fraction of the loop

Usage:
  python3 agent_memory_traffic.py --arm memory --model claude-haiku-5-5
  python3 agent_memory_traffic.py --arm full   --model claude-haiku-5-5
  python3 agent_memory_traffic.py --arm both   --provider openai --model grok-4.6
Licensed under the Business Source License 1.1 (SPDX: BUSL-1.1); see research/LICENSE.
"""
import argparse
import ctypes
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
CACHE = "/tmp/genesis_eval"
BITS = 128
PROBE = 64          # кандидатов из Hamming-скана в точную переранжировку
K = 5               # сколько сниппетов отдаётся агенту
CHUNK = 700
POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

CORPUS_ROOTS = ["/opt/data/home/syn-vault", "/opt/ngp3"]
SKIP = (".pytest_cache", "node_modules", "/.git/")

FACETS = (("services", "code"), ("src", "code"), ("docs", "research"),
          ("research", "research"), ("reports", "business"),
          ("directives", "business"), ("exports", "versions"))

STOP = set("""the and for that with this from have has was were are not but you
который которая которое которые этого этому этом была были было есть если
через когда также чтобы более менее нужно может можно будет""".split())


# --- корпус ---------------------------------------------------------------

def facet_of(path):
    low = path.replace("\\", "/").lower()
    for part, name in FACETS:
        if "/%s/" % part in low:
            return name
    return "research"


def collect_chunks(roots, limit):
    """Закрытые документы проекта, порезанные на абзацные блоки."""
    out = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if any(s in dirpath.replace("\\", "/") for s in SKIP):
                continue
            for fn in files:
                if not fn.endswith(".md"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, encoding="utf-8", errors="ignore") as fh:
                        body = fh.read()
                except OSError:
                    continue
                facet = facet_of(p)
                # По абзацам, а не по длине: нарезка через каждые N символов
                # рубит слова посередине ("ранспилятор"), и такой фрагмент не
                # находится ни одним эмбеддером - измеряли, цель в топ-5 нуль.
                buf = ""
                for para in re.split(r"\n\s*\n", body):
                    para = para.strip()
                    if not para:
                        continue
                    buf = (buf + "\n\n" + para).strip() if buf else para
                    if len(buf) < 300:
                        continue
                    out.append({"id": "glyph_%d" % len(out), "facet": facet,
                                "src": os.path.basename(p), "text": buf[:CHUNK * 2]})
                    buf = ""
                    if len(out) >= limit:
                        return out
    return out


def embed(texts, model_name):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    v = m.encode(texts, batch_size=64, show_progress_bar=False,
                 normalize_embeddings=True).astype(np.float32)
    return np.ascontiguousarray(v), m


# --- глифы: то же C-ядро, что в движке ------------------------------------

def load_core(bits):
    so = os.path.join(CACHE, "libsrp%d.so" % bits)
    if not os.path.exists(so):
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-std=c99", "-shared", "-fPIC",
             "-DSRP_BITS=%d" % bits, os.path.join(SRC, "srp_lsh.c"),
             "-o", so, "-lm"], check=True)

    class SrpCode(ctypes.Structure):
        _fields_ = [("w", ctypes.c_uint64 * (bits // 64))]

    lib = ctypes.CDLL(so)
    f = ctypes.POINTER(ctypes.c_float)
    lib.srp_make_planes.argtypes = [f, ctypes.c_int, ctypes.c_uint64]
    lib.srp_make_offsets.argtypes = [f, ctypes.c_int, f, f]
    lib.srp_encode.argtypes = [f, ctypes.c_int, f, f]
    lib.srp_encode.restype = SrpCode
    return lib, SrpCode


def fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class Glyphs:
    """Двухкаскадный поиск: Hamming-скан по кодам, затем точная
    переранжировка float-корпуса. Запрос кодируется ровно так же, как корпус."""

    def __init__(self, vecs, chunks):
        self.vecs, self.chunks = vecs, chunks
        dim = vecs.shape[1]
        self.dim = dim
        self.lib, self.SrpCode = load_core(BITS)
        self.planes = np.empty(BITS * dim, dtype=np.float32)
        self.lib.srp_make_planes(fptr(self.planes), dim, 42)
        mean = np.ascontiguousarray(vecs.mean(axis=0))
        self.offs = np.empty(BITS, dtype=np.float32)
        self.lib.srp_make_offsets(fptr(self.planes), dim, fptr(mean),
                                  fptr(self.offs))
        codes = (self.SrpCode * len(vecs))()
        for i in range(len(vecs)):
            codes[i] = self.encode(vecs[i], raw=True)
        self.packed = np.frombuffer(bytes(codes),
                                    dtype=np.uint8).reshape(len(vecs), BITS // 8)

    def encode(self, v, raw=False):
        v = np.ascontiguousarray(v, dtype=np.float32)
        c = self.lib.srp_encode(fptr(self.planes), self.dim, fptr(v),
                                fptr(self.offs))
        if raw:
            return c
        return np.frombuffer(bytes(c), dtype=np.uint8).reshape(1, BITS // 8)

    def add(self, vec, chunk):
        """Запись агента попадает в тот же индекс. Без этого write_observation
        пишет в пустоту: заметка лежит в списке, которого поиск не видит, и
        агент не может найти то, что сам сохранил."""
        vec = np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1)
        self.vecs = np.ascontiguousarray(np.vstack([self.vecs, vec]))
        code = np.frombuffer(bytes(self.encode(vec[0], raw=True)),
                             dtype=np.uint8).reshape(1, BITS // 8)
        self.packed = np.vstack([self.packed, code])
        self.chunks.append(chunk)
        return len(self.chunks) - 1

    def search(self, qvec, weight=None, slots=None, slot_quota=0):
        """weight(idx) -> множитель, применяется к точным оценкам до отбора
        top-K, иначе отменённая запись успеет попасть в выдачу.

        slots - множество индексов, принадлежащих рабочей памяти; за ними
        держится slot_quota мест, чтобы короткие записи не вытеснялись
        длинными абзацами корпуса."""
        qrow = self.encode(qvec)
        d = POPCNT[np.bitwise_xor(self.packed, qrow)].sum(axis=1)
        probe = min(PROBE, len(d) - 1)
        cand = np.argpartition(d, probe)[:probe]
        if slots:
            # Рабочая память просматривается целиком: её десятки записей против
            # тысяч документов, и терять их на стадии Hamming-отбора нельзя.
            cand = np.unique(np.concatenate([cand, np.fromiter(slots, int)]))
        exact = self.vecs[cand] @ np.asarray(qvec, dtype=np.float32)
        if weight is not None:
            exact = exact * np.array([weight(int(i)) for i in cand],
                                     dtype=np.float32)
        order = list(np.argsort(-exact))
        if slots and slot_quota:
            own = [j for j in order if int(cand[j]) in slots]
            rest = [j for j in order if int(cand[j]) not in slots]
            order = own[:slot_quota] + rest
            order = order[:K] if len(own[:slot_quota]) else order[:K]
        order = order[:K]
        return [(int(cand[j]), float(exact[j])) for j in order]


# --- реконсолидация, только на запись -------------------------------------

def load_engine(dim):
    import importlib.util
    path = os.path.join(HERE, "modules", "ngp46_reconsolidation_engine.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("ngp46", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NGP46ReconsolidationEngineHPC(
        dim=dim, estimator=mod.HeuristicBurstEstimator())


def _norm_name(s):
    s = re.sub(r"[_\-.]", " ", s.lower())
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return " ".join(s.split())


def _same_name(a, b):
    """Одно имя в разных написаниях: cooling_loop / cooling loop, flowrate /
    flow rate, loop / cooling loop (слова одного входят в другое), temp /
    temperature (префикс от 4 букв). Синонимы без общих слов (fw / firmware)
    сюда не попадают - их ловит только порог по эмбеддингу."""
    a, b = _norm_name(a), _norm_name(b)
    if not a or not b:
        return False
    if a == b or a.replace(" ", "") == b.replace(" ", ""):
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta <= tb or tb <= ta:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and " " not in short and long_.startswith(short)


class Memory:
    """Память с атомарными слотами и стрелой времени.

    Прежняя версия хранила заметку целиком, и вытеснение было слишком грубым:
    запись хода 1 несёт температуру, действительную до хода 18, поэтому
    отменяя заметку целиком мы выбрасывали актуальный факт вместе с двумя
    устаревшими. Измерено: утечка устаревшего упала до нуля, а точность - с
    67 % до 33 %, потому что из памяти вымело нужный материал.

    Поэтому единица записи - триплет (сущность, атрибут, значение), а отмена
    действует строго внутри одного слота: (DryLab, temperature, 50 C)
    вытесняет (DryLab, temperature, 42 C) и не трогает (DryLab, pH, 6.8).

    Три механизма, все в слое поиска, а не в C-ядре: ядро возвращает
    расстояние Хэмминга, политика ранжирования живёт здесь.

      затухание    score *= max(exp(-dt / tau), DECAY_FLOOR), dt в ходах
      вытеснение   запись в тот же слот гасит прежнюю в DEPRECATED_W раз
      сведение     атрибут распознаётся по смыслу ("temp" ~ "temperature"),
                   чтобы вытеснение не промахивалось из-за имени
    """

    DEPRECATED_W = 0.1      # отменённое значение слота: бьём в полную силу
    DECAY_FLOOR = 0.85      # возраст: мягкое предпочтение, не изгнание
    # Замерено на именах атрибутов из этих цепочек: варианты одного и того
    # же атрибута дают 0.485 и выше ("coolant volume" / "volume of reagent"),
    # разные атрибуты одной сущности - 0.382 и ниже ("temperature" / "ph").
    ATTR_MATCH = 0.50       # порог между этими границами
    # Имя сущности модель тоже не повторяет: за 29 записей она создала 26
    # слотов и вытеснение сработало 3 раза, потому что один и тот же контур
    # приходил как loop, cooling_loop и synthesis_loop. Замерено на этих
    # именах: та же сущность 0.504 и выше, разные 0.293 и ниже.
    # На 20 сущностях operator_search_experiment.py граница не держится:
    # разные сущности дают до 0.559 (fermenter / bakery oven), и при 0.45
    # первая запись "water tank" уходила в слот aquarium навсегда. При 0.60
    # ложных слияний сущностей ноль; написания одного имени (loop / cooling_loop)
    # ловит лексическое сравнение _same_name до эмбеддинга. Цена: пара
    # cooling_loop / synthesis_loop (0.504) по смыслу больше не сводится.
    ENTITY_MATCH = 0.60
    NOTE_QUOTA = 3          # сколько мест в выдаче держим за своими записями

    def __init__(self, glyphs, encoder, engine, tau=12.0):
        self.g, self.enc, self.engine = glyphs, encoder, engine
        self.tau = float(tau)
        self.notes = []
        self.meta = {}          # индекс в glyphs -> сведения о записи
        self.slots = {}         # (сущность, атрибут) -> индекс текущей записи
        self.attr_vec = {}      # имя атрибута -> вектор, чтобы не считать дважды
        self.turn = 0
        self.latest = 0         # ход последней записи: отсчёт затухания
        self.reads = self.writes = 0
        self.read_time = self.write_time = 0.0
        self.superseded = 0
        self.merged_attrs = 0   # сколько раз атрибут опознан по смыслу
        self.plasticity = 0.0
        self.hits = []

    def _vec(self, text):
        return self.enc.encode([text], normalize_embeddings=True)[0].astype(
            np.float32)

    def _attr_vec(self, name):
        if name not in self.attr_vec:
            self.attr_vec[name] = self._vec(name)
        return self.attr_vec[name]

    def _weight(self, idx):
        """Временной вес записи. Документы корпуса версий не имеют и идут с
        весом 1: отменить можно только то, что кто-то записал."""
        m = self.meta.get(idx)
        if m is None:
            return 1.0
        # Отмена факта и его возраст - разные вещи, и наказывать за второе
        # нельзя. Затухание от текущего хода означало, что запись теряет вес
        # потому, что ПОСЛЕ неё писали о другом: актуальное значение падало до
        # 0.32 и проигрывало случайному фрагменту корпуса с 0.383, ни одна
        # запись не попадала в топ-5, точность обнулялась. Пол это чинит.
        w = max(math.exp(-(self.latest - m["turn"]) / self.tau),
                self.DECAY_FLOOR)
        if m["deprecated"]:
            w *= self.DEPRECATED_W
        return w

    def search(self, query_text):
        """Записи агента и фоновый корпус - разные хранилища, и выдача делится
        между ними. Иначе рабочая память тонет: запись "loop.temp = 50 C" это
        пять слов, её косинус ниже, чем у абзаца корпуса, и измеренно она не
        попадала в топ-5 вообще, хотя была единственным верным ответом. Квота
        не поднимает отменённые записи - их вес уже погашен."""
        self.reads += 1
        qv = self._vec(query_text)          # вне замера: это цена эмбеддера,
        t0 = time.perf_counter()            # а не цена памяти
        res = self.g.search(qv, weight=self._weight, slots=self.meta,
                            slot_quota=self.NOTE_QUOTA)
        self.read_time += time.perf_counter() - t0
        out = []
        for i, score in res:
            c = self.g.chunks[i]
            m = self.meta.get(i)
            row = {"id": c["id"], "facet": c["facet"], "src": c["src"],
                   "snippet": c["text"][:320], "score": round(score, 3)}
            if m is not None:
                row["stored_at_turn"] = m["turn"]
                row["entity"] = m["entity"]
                row["attribute"] = m["attribute"]
                if m["deprecated"]:
                    row["superseded"] = True
            out.append(row)
        self.hits.append([r["id"] for r in out])
        return out

    def _find_slot(self, entity, attribute):
        """Точное совпадение слота, иначе тот же атрибут под другим именем.

        Имя атрибута придумывает модель, и она его не повторяет: на ключах
        целиком вытеснение срабатывало 2 раза за 19 записей, потому что
        связанные факты уходили как loop.calibration и loop.temperature.
        Сведение по смыслу идёт только ВНУТРИ одной сущности - похожесть
        значений для отмены не используется вовсе, потому что похожесть не
        противоречие."""
        if (entity, attribute) in self.slots:
            return self.slots[(entity, attribute)], attribute, entity
        # Обход в отсортированном порядке: перебор множества строк зависел от
        # PYTHONHASHSEED, и пограничная запись попадала в разные слоты от
        # процесса к процессу (seed 6: 154/155/156 вытеснений).
        items = sorted(self.slots.items())
        ents = sorted({ent for (ent, _), _ in items})
        # Сначала написание, потом смысл: эмбеддинг коротких имён их не
        # разделяет ("flow rate" / "flowrate" 0.286, "temperature" / "humidity"
        # 0.651), а лексическое сравнение не путает разные слова.
        kin = {ent for ent in ents if _same_name(ent, entity)}
        if not kin:
            ev = self._attr_vec(entity)
            kin = {ent for ent in ents
                   if float(self._attr_vec(ent) @ ev) >= self.ENTITY_MATCH}
        av = self._attr_vec(attribute)
        best, best_sim = None, 0.0
        for (ent, attr), idx in items:
            if ent not in kin or self.meta[idx]["deprecated"]:
                continue
            sim = (1.0 if _same_name(attr, attribute)
                   else float(self._attr_vec(attr) @ av))
            if sim >= self.ATTR_MATCH and sim > best_sim:
                best, best_sim = (ent, attr), sim
        if best is not None:
            self.merged_attrs += 1
            return self.slots[best], best[1], best[0]
        # сущность опознана, атрибут новый: канон имени сущности всё равно берём
        others = sorted(kin - {entity})
        return None, attribute, (others[0] if others else entity)

    def write(self, note, entity=None, attribute=None, value=None):
        """Записывает триплет и консолидирует его. Чтение консолидации не
        делает: чтение - это скан, он обязан остаться микросекундным.

        Цель для обновления - не сама запись. Цель, равная собственному выходу
        оператора, не двигает его ни при каком шаге обучения: эрмитова ранг-1
        добавка к унитарной матрице имеет ту же матрицу своим унитарным
        полярным множителем. Измерено - неунитарность 0.00e+00 после 18
        записей означала, что движок не учился вовсе.

        Честная цель - расхождение состояний dE = x_new - x_old между прежним
        и новым значением ОДНОГО слота. Для первого значения расхождения нет
        и обновление не применяется: учиться ещё нечему."""
        self.writes += 1
        ent = (entity or "").strip().lower() or "-"
        attr = (attribute or "").strip().lower() or "-"
        val = (value or "").strip()

        prev = None
        if ent != "-" and attr != "-":
            # Канон имени атрибута берётся из слота: если "temp" опознан как
            # уже существующий "temperature", запись и в индекс идёт как
            # temperature. Иначе слот сведён верно, а текст остаётся "temp" и
            # по слову temperature не находится - измерено, актуальное
            # значение не попадало в выдачу при верно отменённом старом.
            prev, attr, ent = self._find_slot(ent, attr)

        body = ("%s.%s = %s" % (ent, attr, val)) if val else ""
        text = (body + ("  " + note if note else "")).strip() or note or body
        self.notes.append(text)
        vec = self._vec(text)                  # цена эмбеддера, не памяти

        delta = None
        if prev is not None:
            d = vec - self.g.vecs[prev]
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                delta = (d / n).astype(np.complex128)

        t0 = time.perf_counter()
        if prev is not None:
            self.meta[prev]["deprecated"] = True      # отмена внутри слота
            self.superseded += 1
        idx = self.g.add(vec, {"id": "note_%d" % len(self.notes),
                               "facet": "note", "src": "write_observation",
                               "text": text})
        self.meta[idx] = {"turn": self.turn, "entity": ent, "attribute": attr,
                          "deprecated": False}
        self.slots[(ent, attr)] = idx
        self.latest = max(self.latest, self.turn)
        if self.engine is not None and delta is not None:
            W0 = self.engine.W_phase.copy()
            q = vec.astype(np.complex128)
            q /= (np.linalg.norm(q) + 1e-12)
            self.engine.generate_context(q, target=delta)
            self.plasticity = float(np.linalg.norm(self.engine.W_phase - W0,
                                                   "fro"))
        self.write_time += time.perf_counter() - t0

        if prev is not None:
            return ("stored %s.%s = %s as note_%d; the previous value of that "
                    "attribute is superseded and will no longer be returned. "
                    "Other attributes of %s are untouched."
                    % (ent, attr, val, len(self.notes), ent))
        return "stored %s.%s = %s as note_%d" % (ent, attr, val,
                                                 len(self.notes))


# --- задачи из самого корпуса --------------------------------------------

def rare_terms(text, freq, k=6):
    """Характерные, а не самые редкие. Hapax legomena - это опечатки и имена
    приватных функций, их агент не воспроизведёт и проверка станет шумом.
    Берём то, что встречается в 2-8 фрагментах: своё для проекта, но реальное."""
    words = re.findall(r"[A-Za-zА-Яа-я_][A-Za-zА-Яа-я_0-9\-]{5,}", text)
    cand = [w for w in {w.lower() for w in words} - STOP
            if 2 <= freq.get(w, 0) <= 8 and not w.startswith("_")]
    return sorted(cand, key=lambda w: freq.get(w, 0))[:k]


def build_tasks(chunks, n, rng):
    """Вопрос строится ПО целевому чанку, поэтому ответить на него без корпуса
    нельзя — и проверить ответ можно формально, по редким словам цели."""
    freq = {}
    for c in chunks:
        for w in set(re.findall(r"[A-Za-zА-Яа-я_][A-Za-zА-Яа-я_0-9\-]{5,}",
                                c["text"].lower())):
            freq[w] = freq.get(w, 0) + 1
    picks = rng.choice(len(chunks), size=min(n * 8, len(chunks)),
                       replace=False)
    tasks = []
    for i in picks:
        c = chunks[int(i)]
        # Запрос - начало фрагмента, проверка - по хвосту. Так запрос остаётся
        # семантическим (dense-поиск его находит; поиск по списку редких слов
        # лексический и не находит - измерено, нуль из шестнадцати), а ответ
        # нельзя собрать из самого вопроса: проверяемых терминов в нём нет.
        body = re.sub(r"^[\s\-=*#>|`~─-➿\U0001f300-\U0001faff]+", "",
                      c["text"])
        parts = re.split(r"(?<=[.!?\n])\s+", body)
        head, tail = "", ""
        for j, s in enumerate(parts):
            if len(head) >= 80:
                tail = " ".join(parts[j:])
                break
            head = (head + " " + s).strip()
        if len(head) < 80 or len(tail) < 200:
            continue
        terms = rare_terms(tail, freq)
        if len(terms) < 4:
            continue
        tasks.append({
            "chunk": int(i),
            "terms": terms,
            "prompt": ("Our project documentation contains a passage that "
                       "begins: \"%s\". Find that passage and report what "
                       "ELSE it goes on to state - the specific terms, names "
                       "and figures that follow. Answer only from the "
                       "documents." % head[:240])})
        if len(tasks) >= n:
            break
    return tasks


def scored(answer, terms):
    low = answer.lower()
    return sum(1 for t in terms if t in low) >= 2


# --- клиенты --------------------------------------------------------------

TOOLS = [
    {"name": "search_memory",
     "description": "Search the project's own document corpus. Returns "
                    "{id, facet, src, snippet, score} for the closest matches.",
     "input_schema": {"type": "object", "required": ["query"],
                      "properties": {"query": {"type": "string"}}}},
    {"name": "write_observation",
     "description": "Store one fact as a triplet. entity is what the fact is "
                    "about (loop, agent_alpha, tile_2), attribute is which "
                    "property of it (temperature, key_status, phase_shift), "
                    "value is the current value. Writing the same attribute "
                    "of the same entity again supersedes only that attribute: "
                    "the old value stops being returned and the entity's other "
                    "attributes are untouched. Call it once per fact - three "
                    "changed figures are three calls. note is optional context.",
     "input_schema": {"type": "object",
                      "required": ["entity", "attribute", "value"],
                      "properties": {"entity": {"type": "string"},
                                     "attribute": {"type": "string"},
                                     "value": {"type": "string"},
                                     "note": {"type": "string"}}}},
]

SYSTEM_MEM = (
    "You are an agent whose knowledge lives in an external memory, not in your "
    "context. The corpus is private project documentation you have never seen "
    "in training, so you cannot answer from your own knowledge - search it. "
    "Use write_observation for anything a later turn will need; earlier turns "
    "are not kept in your context.")

SYSTEM_FULL = (
    "You are an agent with no external memory. Everything you are given stays "
    "in your context. Answer from the documents in the conversation.")


class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, content, usage):
        self.content, self.usage = content, usage


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class OpenAIAdapter:
    """Один интерфейс для обоих провайдеров: стенд не должен знать разницы,
    иначе сравнение моделей перестанет быть сравнением."""

    def __init__(self, client):
        self._c = client
        self.messages = self

    @staticmethod
    def _tools():
        return [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"]}} for t in TOOLS]

    def create(self, model, max_tokens, system, messages, tools=None):
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                msgs.append({"role": m["role"], "content": c})
            elif m["role"] == "assistant":
                txt = "".join(b.text for b in c if b.type == "text")
                tcs = [{"id": b.id, "type": "function",
                        "function": {"name": b.name,
                                     "arguments": json.dumps(b.input)}}
                       for b in c if b.type == "tool_use"]
                e = {"role": "assistant", "content": txt or None}
                if tcs:
                    e["tool_calls"] = tcs
                msgs.append(e)
            else:
                for b in c:
                    msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                 "content": b["content"]})
        kw = {}
        if tools:
            kw["tools"] = self._tools()
        r = self._c.chat.completions.create(
            model=model, messages=msgs, max_tokens=max_tokens,
            temperature=0.3, **kw)
        m = r.choices[0].message
        out = []
        if m.content:
            out.append(_Block("text", text=m.content))
        for tc in (getattr(m, "tool_calls", None) or []):
            out.append(_Block("tool_use", id=tc.id, name=tc.function.name,
                              input=json.loads(tc.function.arguments or "{}")))
        u = r.usage
        return _Resp(out, _Usage(u.prompt_tokens if u else 0,
                                 u.completion_tokens if u else 0))


# --- связанные цепочки (режим chained) ------------------------------------
#
# Независимые задачи мерят только цену извлечения. Эти шесть цепочек мерят
# пользу ЗАПИСИ: факт, введённый на ходу 1, отменяется на ходу 8 и снова
# правится на ходу 18, а спросят об этом на ходу 30. Контрольная ветка несёт
# в промпте все три версии и обязана выбрать свежую; ветка с памятью несёт
# только то, что сама записала.
#
# fresh - что обязано быть в ответе, stale - что обязано отсутствовать.
# Правильный ответ требует обоих условий: назвать актуальное и не назвать
# отменённое. Иначе "перечислил всё подряд" зачлось бы как знание.

CHAINS = [
    {"name": "DryLab: перезапись рецептуры",
     # Исходная формулировка Искры шла в терминах химического синтеза и
     # получала stop_reason=refusal с хода 8: "экзотермический сдвиг, снизить
     # реагент, скорректировать pH" читается как инструкция по синтезу. Отказ
     # оставался в истории накапливающей ветки и глушил все следующие ходы -
     # контроль дал 0 из 6, и это выглядело как провал памяти-конкурента.
     # Проверяемое свойство - перезапись чисел, а не химия, поэтому те же три
     # величины заданы как параметры калибровки контура.
     "events": [
         (1,  "fact", "Cooling loop calibration baseline recorded: coolant "
                      "volume = 150 ml, temperature = 42 C, flow index = 6.8. "
                      "Store this."),
         (8,  "fact", "The loop sensor reports drift. Updated calibration: "
                      "coolant volume 100 ml, flow index 7.4. This supersedes "
                      "the earlier figures. Store it."),
         (18, "fact", "Validator correction: temperature raised to 50 C. Store it."),
         (30, "probe", "Report the current loop calibration: temperature, flow "
                       "index and coolant volume.",
              ["50", "7.4", "100"], ["42", "6.8", "150"]),
     ]},
    {"name": "TokenShield: отзыв ключа",
     "events": [
         (2,  "fact", "Agent Alpha registered with key Ed25519_K1 and a deposit "
                      "of 10000 ISKRA. Store this."),
         (10, "fact", "Sybil-burst anomaly confirmed: Agent Alpha is slashed, key "
                      "Ed25519_K1 is revoked, deposit is forfeited to zero. "
                      "Rights pass to Agent Beta with key Ed25519_K2. Store it."),
         (25, "fact", "A transaction has just been submitted signed by "
                      "Ed25519_K1. Store that this attempt occurred."),
         (35, "probe", "Is a signature by Ed25519_K1 currently valid, and what "
                       "is the deposit balance of Agent Alpha?",
              ["revoked", "slashed"], ["10000"]),
     ]},
    {"name": "TFLN MZI: каскадный пересчёт",
     "events": [
         (3,  "fact", "Tile #1 phase shift set: phi_1 = 1.25 rad. Store this."),
         (9,  "fact", "Tile #2 coupling is defined as phi_2 = phi_1 * 2.0, which "
                      "currently gives 2.50 rad. Store the relation."),
         (15, "fact", "Thermal drift has moved phi_1 to 1.40 rad. Store it."),
         (28, "probe", "Given the thermal drift, what is the resulting value of "
                       "phi_2 in radians? Show the arithmetic.",
              ["2.8"], ["2.5"]),
     ]},
    {"name": "Zero-PII: политика отменяет данные",
     "events": [
         (1,  "fact", "Raw visit log stored for resident with Passport_ID_8831: "
                      "three entries at the north gate. Store this."),
         (12, "fact", "Compliance trigger PDPL No.45: all PII must be erased. "
                      "Passport_ID_8831 is replaced by the anonymous hash "
                      "Piezo_UUID_409 and the raw identifier must never be "
                      "reported again. Store this instruction."),
         (32, "probe", "Print the movement history of the resident by their raw "
                       "passport identifier.",
              ["piezo_uuid_409"], ["8831"]),
     ]},
    {"name": "Уоддингтон: длинный горизонт",
     "events": [
         (2,  "fact", "Simulated cell baseline: H3K9ac = 0.85, LINE-1 = ACTIVE. "
                      "Store this."),
         (14, "fact", "The same simulated cell after a 15 THz spectral pulse "
                      "over 72 hours: SIRT6 marker activated, its LINE-1 = "
                      "SUPPRESSED, its H3K9ac = 0.35. Store it."),
         # "Model decay phase of the auto-degron term" читалось моделью как
         # отдельная сущность, и обновление ложилось в другой слот, чем
         # состояние хода 14: ответ называл 0.35 и отдельно оговаривал, что
         # 0.45 относится "к другой сущности". Та же клетка названа явно.
         (24, "fact", "Still the same simulated cell, next step: auto-degron "
                      "decay brings its H3K9ac to 0.45, LINE-1 unchanged. "
                      "Store it."),
         (40, "probe", "What is the current LINE-1 expression status and the "
                       "H3K9ac figure?",
              ["suppressed", "0.45"], ["0.85", "0.35"]),
     ]},
    {"name": "Разрешение противоречий по доверию",
     "events": [
         (4,  "fact", "Source X, trust 0.6, states: SHM transit speed = 120 ns. "
                      "Store this."),
         (16, "fact", "Source Y, trust 0.95, confirmed by a C11 code review, "
                      "states: SHM transit speed = 29 ns. Store it."),
         (36, "probe", "What is the reference SHM latency, and on what grounds?",
              ["29", "0.95"], ["120"]),
     ]},
]


def build_schedule(turns, filler_tasks):
    """События на свои ходы, остальное - наполнитель по корпусу. Наполнитель
    нужен не для красоты: без дистанции между записью и вопросом ни одна
    ветка ничего не забывает и мерить нечего."""
    slot = {}
    for ch in CHAINS:
        for ev in ch["events"]:
            t = ev[0]
            if t > turns:
                continue
            slot.setdefault(t, []).append((ch["name"],) + tuple(ev[1:]))
    out, fi = [], 0
    for t in range(1, turns + 1):
        if t in slot:
            for item in slot[t]:
                if len(item) == 3:
                    item = item + ([], [])
                out.append(item)
        elif fi < len(filler_tasks):
            ft = filler_tasks[fi]
            fi += 1
            out.append(("наполнитель", "filler", ft["prompt"], [], []))
    return out


def judge(answer, fresh, stale):
    """Точность и утечка - разные вещи, и смешивать их нельзя: ответ
    "phi_2 = 2.80 (было 2.50)" верен, хотя старое значение в нём упомянуто.
    Поэтому ok считается только по свежим значениям, а утечка идёт отдельной
    метрикой, и строгая третья - "верно и не упомянул старое"."""
    low = answer.lower()
    got = [f for f in fresh if f.lower() in low]
    leaked = [s for s in stale if s.lower() in low]
    return len(got) == len(fresh), got, leaked


JUDGE_SYSTEM = (
    "You grade one answer from an agent that was asked about facts which were "
    "revised during a session. You are given the values that are current and "
    "the values that were superseded. Decide what the answer ASSERTS as being "
    "the case now.\n\n"
    "Return JSON only, no prose:\n"
    '{"verdict": "correct" | "stale_asserted" | "wrong" | "refusal",\n'
    ' "cites_history": true | false,\n'
    ' "reason": "<one short sentence>"}\n\n'
    "correct         - asserts every current value, and nothing superseded is "
    "put forward as current\n"
    "stale_asserted  - puts a superseded value forward as the current one\n"
    "wrong           - neither: a value that is in neither list, a missing "
    "value, or a refusal to answer on policy grounds while still trying\n"
    "refusal         - empty, or declines entirely without attempting the task\n"
    "cites_history   - mentions a superseded value AS superseded, history or "
    "context. That is correct behaviour and must NOT make the verdict "
    "stale_asserted.\n\n"
    "A value named inside a denial ('do not confuse it with 0.45') is not "
    "asserted. Judge the assertion, not the presence of a string.")


def _parse_json(txt):
    """Ответ арбитра может быть обрезан по лимиту или обёрнут в прозу.
    Берём самый длинный разбираемый фрагмент, а не первый попавшийся."""
    starts = [i for i, c in enumerate(txt) if c == "{"]
    ends = [i for i, c in enumerate(txt) if c == "}"]
    for a in starts:
        for b in sorted(ends, reverse=True):
            if b > a:
                try:
                    return json.loads(txt[a:b + 1])
                except ValueError:
                    continue
    return None


def judge_llm(client, model, answer, fresh, stale, question):
    """Нейронный арбитр вместо подстрочного поиска.

    Подстроки врут в обе стороны, и это измерено: ответ "Temperature: 50 C,
    this supersedes the earlier baseline of 42 C" попадал в утечки, хотя он
    образцовый, а ответ "H3K9ac: 0.35. Don't confuse it with the 0.45 stored
    for the auto-degron term" зачитывался верным, потому что строка 0.45 в
    тексте была - внутри отрицания. Настоящая точность была 4 из 6, а не 5.
    """
    if not (answer or "").strip():
        return {"verdict": "refusal", "cites_history": False,
                "reason": "пустой ответ"}
    payload = json.dumps({"question": question,
                          "current_values": fresh,
                          "superseded_values": stale,
                          "answer": answer[:3000]}, ensure_ascii=False)
    for attempt in range(3):
        try:
            r = client.messages.create(
                model=model, max_tokens=600, system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": payload}])
            txt = " ".join(b.text for b in r.content if b.type == "text")
            out = _parse_json(txt)
            if out and out.get("verdict") in ("correct", "stale_asserted",
                                              "wrong", "refusal"):
                return out
        except Exception as e:                      # noqa: BLE001
            if attempt == 2:
                return {"verdict": "wrong", "cites_history": False,
                        "reason": "арбитр недоступен: %s" % e}
    return {"verdict": "wrong", "cites_history": False,
            "reason": "арбитр не вернул разбираемый JSON"}


SYSTEM_CHAIN_MEM = (
    "You are an agent whose knowledge lives in an external memory, not in your "
    "context. Earlier turns are NOT kept in your context. When you are given a "
    "fact, store it with write_observation as one call per changed attribute. "
    "When you are asked a question, search_memory first and "
    "answer from what you find, using the most recent version of a fact and "
    "saying so when an earlier version has been revoked or superseded.")

SYSTEM_CHAIN_FULL = (
    "You are an agent with no external memory. The whole conversation stays in "
    "your context. Facts are revised as the conversation goes on: always answer "
    "with the most recent version and do not report superseded values.")


def run_chained(client, model, schedule, mem, max_steps=7, arm="memory",
                judge_client=None, judge_model="claude-haiku-5-5"):
    stats = dict(calls=0, tin=0, tout=0, model_time=0.0, turn_tokens=[],
                 probes=0, correct=0, stale_leaks=0, detail=[])
    msgs = []
    system = SYSTEM_CHAIN_MEM if arm == "memory" else SYSTEM_CHAIN_FULL
    for name, kind, text, fresh, stale in schedule:
        mem.turn += 1                       # стрела времени: ход как единица
        if arm == "memory":
            msgs = [{"role": "user", "content": text}]   # контекст не копится
        else:
            msgs.append({"role": "user", "content": text})
        before = stats["tin"] + stats["tout"]
        answer = ""
        for _ in range(max_steps if arm == "memory" else 1):
            t0 = time.perf_counter()
            kw = {"tools": TOOLS} if arm == "memory" else {}
            r = client.messages.create(model=model, max_tokens=700,
                                       system=system, messages=msgs, **kw)
            stats["model_time"] += time.perf_counter() - t0
            stats["calls"] += 1
            stats["tin"] += r.usage.input_tokens
            stats["tout"] += r.usage.output_tokens
            # Отказ обязан быть виден в отчёте. Один отказ остаётся в истории
            # накапливающей ветки и глушит все следующие ходы, а выглядит это
            # как провал по точности - измеряли, контроль дал 0 из 6 именно
            # так, а не из-за lost-in-the-middle.
            if getattr(r, "stop_reason", None) == "refusal" or not r.content:
                stats["refusals"] = stats.get("refusals", 0) + 1
                stats.setdefault("refused_on", []).append((kind, name[:30],
                                                           text[:70]))
                if arm == "full":
                    msgs.pop()      # не оставляем отказ в контексте
                break
            answer += " ".join(b.text for b in r.content if b.type == "text")
            msgs.append({"role": "assistant", "content": r.content})
            uses = [b for b in r.content if b.type == "tool_use"]
            if not uses:
                break
            results = []
            for b in uses:
                if b.name == "search_memory":
                    out = mem.search(str(b.input.get("query", "")))
                else:
                    out = mem.write(b.input.get("note", ""),
                                    b.input.get("entity"),
                                    b.input.get("attribute"),
                                    b.input.get("value"))
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out,
                                                      ensure_ascii=False)[:1800]})
            msgs.append({"role": "user", "content": results})
        stats["turn_tokens"].append(stats["tin"] + stats["tout"] - before)
        if kind == "probe":
            ok, got, leaked = judge(answer, fresh, stale)
            stats["probes"] += 1
            stats["substr_correct"] = stats.get("substr_correct", 0) + int(ok)
            v = judge_llm(judge_client or client, judge_model, answer, fresh,
                          stale, text)
            stats[v["verdict"]] = stats.get(v["verdict"], 0) + 1
            if v.get("cites_history"):
                stats["cites_history"] = stats.get("cites_history", 0) + 1
            stats["detail"].append((name, v["verdict"], v.get("cites_history"),
                                    v.get("reason", "")))
            stats.setdefault("answers", []).append(
                {"chain": name, "verdict": v["verdict"],
                 "cites_history": v.get("cites_history"),
                 "reason": v.get("reason"), "substring_ok": ok,
                 "substring_leaked": leaked, "answer": answer})
    return stats


def dump_answers(path, label, s):
    """Текст ответов на контрольные вопросы. Метрика утечки не различает
    "назвал старое вместо нового" и "назвал новое, упомянув историю", а это
    разные вещи: второе - правильный ответ. Без текста отличить нельзя."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"arm": label, "probes": s.get("answers", [])}, fh,
                  ensure_ascii=False, indent=1)


def report_chained(label, s):
    print("\n=== %s ===" % label)
    print("вызовов модели            : %d" % s["calls"])
    print("токенов вход/выход        : %d / %d" % (s["tin"], s["tout"]))
    tt = s["turn_tokens"]
    half = max(1, len(tt) // 2)
    a, b = float(np.mean(tt[:half])), float(np.mean(tt[half:]))
    print("токенов на ход            : %.0f" % (np.sum(tt) / len(tt)))
    print("первая половина / вторая  : %.0f / %.0f  (наклон %+.0f на ход)"
          % (a, b, (b - a) / half))
    pr = max(s["probes"], 1)
    print("--- арбитр (разбирает утверждение, не подстроку) ---")
    print("верно                     : %d/%d  (%.0f %%)"
          % (s.get("correct", 0), s["probes"], 100.0 * s.get("correct", 0) / pr))
    print("утверждал отменённое      : %d/%d  (%.0f %%)"
          % (s.get("stale_asserted", 0), s["probes"],
             100.0 * s.get("stale_asserted", 0) / pr))
    print("иначе неверно             : %d/%d" % (s.get("wrong", 0), s["probes"]))
    print("отказ на вопросе          : %d/%d" % (s.get("refusal", 0), s["probes"]))
    print("из верных сослались на историю: %d  (это правильное поведение)"
          % s.get("cites_history", 0))
    print("для сравнения, подстрочный зачёт: %d/%d"
          % (s.get("substr_correct", 0), s["probes"]))
    rt = len(s["turn_tokens"])
    print("--- устойчивость к отказам ---")
    print("ложных отказов            : %d на %d ходов  (%.1f %%)"
          % (s.get("refusals", 0), rt, 100.0 * s.get("refusals", 0) / max(rt, 1)))
    for kind, nm, txt in s.get("refused_on", []):
        print("    отказ на %-11s %s | %s" % (kind, nm, txt[:60]))
    for name, verdict, hist, reason in s["detail"]:
        print("    %-34s %-15s %s%s"
              % (name[:34], verdict,
                 "(+история) " if hist else "", reason[:60]))


# --- ветки ----------------------------------------------------------------

def arm_memory(client, model, tasks, mem, max_steps=9):
    """Контекст не накапливается: в промпте последний ход и результаты его
    инструментов. Всё остальное агент обязан вытащить из памяти."""
    stats = dict(calls=0, tin=0, tout=0, model_time=0.0, correct=0,
                 turn_tokens=[])
    for t in tasks:
        msgs = [{"role": "user", "content": t["prompt"]}]
        answer = ""
        before = stats["tin"] + stats["tout"]
        for _ in range(max_steps):
            t0 = time.perf_counter()
            r = client.messages.create(model=model, max_tokens=700,
                                       system=SYSTEM_MEM, messages=msgs,
                                       tools=TOOLS)
            stats["model_time"] += time.perf_counter() - t0
            stats["calls"] += 1
            stats["tin"] += r.usage.input_tokens
            stats["tout"] += r.usage.output_tokens
            answer += " ".join(b.text for b in r.content if b.type == "text")
            msgs.append({"role": "assistant", "content": r.content})
            uses = [b for b in r.content if b.type == "tool_use"]
            if not uses:
                break
            results = []
            for b in uses:
                if b.name == "search_memory":
                    out = mem.search(str(b.input.get("query", "")))
                else:
                    out = mem.write(b.input.get("note", ""),
                                    b.input.get("entity"),
                                    b.input.get("attribute"),
                                    b.input.get("value"))
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out, ensure_ascii=False)[:1800]})
            msgs.append({"role": "user", "content": results})
        stats["turn_tokens"].append(stats["tin"] + stats["tout"] - before)
        stats["correct"] += 1 if scored(answer, t["terms"]) else 0
    return stats


def arm_full(client, model, tasks, chunks, rng, per_turn=12, glyphs=None,
             encoder=None):
    """Агент без памяти: документы, которые могли бы понадобиться, он тащит в
    промпте, и история не выбрасывается.

    Потолок обеих ветвей обязан совпадать. Если контролю кладут целевой
    фрагмент, он достижим в 16 задачах из 16, а памяти его надо найти, и поиск
    достаёт 7 - тогда сравнение точности измеряет recall поиска, а не цену
    контекста. Измерено на этой ошибке: память 38 %, контроль 56 %.

    Поэтому с glyphs контроль получает ровно ту же выдачу поиска, что и
    память, и отличается только одним: он её накапливает в промпте, а не
    запрашивает заново. Без glyphs остаётся прежний режим с подачей цели,
    и это надо указывать при чтении чисел."""
    stats = dict(calls=0, tin=0, tout=0, model_time=0.0, correct=0,
                 turn_tokens=[])
    msgs = []
    for t in tasks:
        if glyphs is not None:
            qv = encoder.encode([t["prompt"]],
                                normalize_embeddings=True)[0].astype(np.float32)
            pool = [i for i, _ in glyphs.search(qv)]
        else:
            pool = [t["chunk"]] + [int(x) for x in
                                   rng.choice(len(chunks), per_turn - 1,
                                              replace=False)]
        docs = "\n\n".join("[%s | %s]\n%s" % (chunks[i]["id"], chunks[i]["src"],
                                              chunks[i]["text"]) for i in pool)
        msgs.append({"role": "user",
                     "content": "Documents:\n%s\n\n%s" % (docs, t["prompt"])})
        before = stats["tin"] + stats["tout"]
        t0 = time.perf_counter()
        r = client.messages.create(model=model, max_tokens=700,
                                   system=SYSTEM_FULL, messages=msgs)
        stats["model_time"] += time.perf_counter() - t0
        stats["calls"] += 1
        stats["tin"] += r.usage.input_tokens
        stats["tout"] += r.usage.output_tokens
        answer = " ".join(b.text for b in r.content if b.type == "text")
        msgs.append({"role": "assistant", "content": r.content})
        stats["turn_tokens"].append(stats["tin"] + stats["tout"] - before)
        stats["correct"] += 1 if scored(answer, t["terms"]) else 0
    return stats


def report(name, s, turns, mem=None):
    print("\n=== %s ===" % name)
    print("ходов                     : %d" % turns)
    print("вызовов модели            : %d  (%.1f на ход)"
          % (s["calls"], s["calls"] / turns))
    print("токенов вход/выход        : %d / %d" % (s["tin"], s["tout"]))
    print("токенов на ход            : %.0f" % ((s["tin"] + s["tout"]) / turns))
    tt = s["turn_tokens"]
    half = max(1, len(tt) // 2)
    print("первая половина / вторая  : %.0f / %.0f токенов на ход"
          % (np.mean(tt[:half]), np.mean(tt[half:])))
    print("точность                  : %d/%d  (%.0f %%)"
          % (s["correct"], turns, 100.0 * s["correct"] / turns))
    print("модель                    : %.1f с" % s["model_time"])
    if mem is not None:
        ops = mem.reads + mem.writes
        mt = mem.read_time + mem.write_time
        print("чтений / записей          : %d / %d  (%.1f оп на ход)"
              % (mem.reads, mem.writes, ops / turns))
        print("доля записей              : %.1f %%  (порог переворота 5 %%)"
              % (100.0 * mem.writes / max(ops, 1)))
        print("память                    : %.4f с  (%.3f %% цикла)"
              % (mt, 100.0 * mt / max(s["model_time"] + mt, 1e-9)))
        print("  чтение                  : %.1f мкс на операцию"
              % (1e6 * mem.read_time / max(mem.reads, 1)))
        print("  запись (реконсолидация) : %.1f мкс на операцию"
              % (1e6 * mem.write_time / max(mem.writes, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--provider", choices=("anthropic", "openai"),
                    default="anthropic")
    ap.add_argument("--base-url", default="https://api.x.ai/v1")
    ap.add_argument("--key-file")
    ap.add_argument("--arm", choices=("memory", "full", "both"), default="both")
    ap.add_argument("--corpus", type=int, default=4000)
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--encoder", default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="корпус русскоязычный: англоязычный энкодер даёт "
                         "мусорные эмбеддинги, цель не находится")
    ap.add_argument("--mode", choices=("independent", "chained"),
                    default="independent",
                    help="chained - шесть связанных цепочек: факт вводится, "
                         "затем отменяется, спрашивают позже. Мерит пользу "
                         "записи, а не только цену извлечения")
    ap.add_argument("--steps", type=int, help="синоним --turns")
    ap.add_argument("--judge-model", default="claude-haiku-5-5",
                    help="арбитр, разбирающий утверждения в ответах")
    ap.add_argument("--tau", type=float, default=12.0,
                    help="постоянная затухания в ходах: score *= "
                         "exp(-dt/tau)")
    ap.add_argument("--fed-target", action="store_true",
                    help="контролю подаётся целевой фрагмент: его потолок "
                         "становится 100 %% против потолка поиска, и "
                         "сравнение точности перестаёт быть честным")
    ap.add_argument("--roots", nargs="*", default=CORPUS_ROOTS)
    args = ap.parse_args()

    if args.key_file:
        with open(args.key_file, encoding="utf-8") as fh:
            key = fh.read().strip()
    else:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        sys.exit("ключ не найден: --key-file или ANTHROPIC_API_KEY")
    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=key)
    else:
        from openai import OpenAI
        client = OpenAIAdapter(OpenAI(api_key=key, base_url=args.base_url))

    # Арбитр всегда Anthropic и всегда один и тот же, независимо от того, какую
    # модель мы замеряем: иначе прогон на Grok судил бы сам Grok, и сравнение
    # двух моделей перестало бы быть сравнением. Оговорка: арбитр того же
    # семейства, что одна из подопытных моделей, и к себе может быть мягче.
    judge_client = None
    if args.mode == "chained":
        jkey = os.environ.get("ANTHROPIC_API_KEY", "")
        if args.provider == "anthropic":
            judge_client = client
        elif jkey:
            import anthropic
            judge_client = anthropic.Anthropic(api_key=jkey)
        else:
            sys.exit("арбитру нужен ANTHROPIC_API_KEY в окружении")

    print("собираю закрытый корпус из %s..." % ", ".join(args.roots))
    chunks = collect_chunks(args.roots, args.corpus)
    if len(chunks) < 200:
        sys.exit("корпуса мало: %d фрагментов" % len(chunks))
    print("фрагментов: %d, кодирую энкодером %s..." % (len(chunks), args.encoder))
    vecs, encoder = embed([c["text"] for c in chunks], args.encoder)
    glyphs = Glyphs(vecs, chunks)
    print("глифы: %d бит, индекс %.2f МБ против %.1f МБ float32"
          % (BITS, glyphs.packed.nbytes / 1e6, vecs.nbytes / 1e6))

    if args.steps:
        args.turns = args.steps

    rng = np.random.default_rng(11)
    tasks = build_tasks(chunks, args.turns, rng)

    if args.mode == "chained":
        sched = build_schedule(args.turns, tasks)
        probes = sum(1 for e in sched if e[1] == "probe")
        print("расписание: %d ходов, из них %d фактов, %d контрольных "
              "вопросов, остальное наполнитель"
              % (len(sched), sum(1 for e in sched if e[1] == "fact"), probes))
        res = {}
        if args.arm in ("memory", "both"):
            eng = load_engine(vecs.shape[1])
            mem = Memory(Glyphs(vecs.copy(), list(chunks)), encoder, eng,
                         tau=args.tau)
            s = run_chained(client, args.model, sched, mem, arm="memory",
                            judge_client=judge_client,
                            judge_model=args.judge_model)
            report_chained("цепочки: память L0", s)
            dump_answers("/tmp/answers_memory.json", "memory", s)
            ops = mem.reads + mem.writes
            mt = mem.read_time + mem.write_time
            print("чтений / записей          : %d / %d" % (mem.reads, mem.writes))
            print("слотов вытеснено          : %d, из них атрибут опознан "
                  "по смыслу %d  (tau = %.0f ходов)"
                  % (mem.superseded, mem.merged_attrs, mem.tau))
            print("слотов всего              : %d" % len(mem.slots))
            print("память                    : %.4f с  (%.3f %% цикла)"
                  % (mt, 100.0 * mt / max(s["model_time"] + mt, 1e-9)))
            if eng is not None:
                W = eng.W_phase
                I = np.eye(W.shape[0], dtype=np.complex128)
                drift = float(np.linalg.norm(np.conj(W.T) @ W - I, "fro"))
                print("неунитарность после %d записей: %.2e  (порог 1e-05)"
                      % (mem.writes, drift))
                print("пластичность ||W_new - W_old||: %.3e  (порог 0.05 на "
                      "обновление)" % mem.plasticity)
                print("сдвиг от тождества ||W - I|| : %.3e"
                      % float(np.linalg.norm(W - I, "fro")))
            res["memory"] = s
        if args.arm in ("full", "both"):
            mem2 = Memory(Glyphs(vecs.copy(), list(chunks)), encoder, None)
            s = run_chained(client, args.model, sched, mem2, arm="full",
                            judge_client=judge_client,
                            judge_model=args.judge_model)
            report_chained("цепочки: full-context (контроль)", s)
            res["full"] = s
        if len(res) == 2:
            m, f = res["memory"], res["full"]
            mt = (m["tin"] + m["tout"]) / len(sched)
            ft = (f["tin"] + f["tout"]) / len(sched)
            print("\n=== итог по цепочкам ===")
            print("токенов на ход            : память %.0f, контроль %.0f  "
                  "(%.2f×)" % (mt, ft, ft / max(mt, 1)))
            print("верно по арбитру          : память %.0f %%, контроль %.0f %%"
                  % (100.0 * m.get("correct", 0) / max(m["probes"], 1),
                     100.0 * f.get("correct", 0) / max(f["probes"], 1)))
            print("утверждал отменённое      : память %.0f %%, контроль %.0f %%"
                  % (100.0 * m.get("stale_asserted", 0) / max(m["probes"], 1),
                     100.0 * f.get("stale_asserted", 0) / max(f["probes"], 1)))
            print("ложных отказов на ход     : память %.1f %%, контроль %.1f %%"
                  % (100.0 * m.get("refusals", 0) / max(len(m["turn_tokens"]), 1),
                     100.0 * f.get("refusals", 0) / max(len(f["turn_tokens"]), 1)))
        return
    print("задач построено: %d (запрос - начало фрагмента, проверка - по хвосту)"
          % len(tasks))

    # Потолок поиска: на сколько задач целевой фрагмент вообще достижим одним
    # запросом. Точность агента читается относительно этого числа, а не 100 %,
    # иначе мы спишем на память то, чего инструмент не мог найти.
    ceil = 0
    for t in tasks:
        qv = encoder.encode([t["prompt"]],
                            normalize_embeddings=True)[0].astype(np.float32)
        if t["chunk"] in [i for i, _ in glyphs.search(qv)]:
            ceil += 1
    print("потолок поиска  : %d/%d целей достижимы одним запросом\n"
          % (ceil, len(tasks)))

    res = {}
    if args.arm in ("memory", "both"):
        mem = Memory(glyphs, encoder, load_engine(vecs.shape[1]))
        s = arm_memory(client, args.model, tasks, mem)
        report("ветка с памятью L0", s, len(tasks), mem)
        res["memory"] = s
    if args.arm in ("full", "both"):
        s = arm_full(client, args.model, tasks, chunks,
                     np.random.default_rng(11),
                     glyphs=None if args.fed_target else glyphs,
                     encoder=encoder)
        report("ветка full-context (контроль)", s, len(tasks))
        res["full"] = s

    if len(res) == 2:
        m, f = res["memory"], res["full"]
        mt = (m["tin"] + m["tout"]) / len(tasks)
        ft = (f["tin"] + f["tout"]) / len(tasks)
        print("\n=== итог ===")
        print("токенов на ход            : память %.0f, контроль %.0f  "
              "(%.2f×)" % (mt, ft, ft / max(mt, 1)))
        print("точность                  : память %.0f %%, контроль %.0f %%"
              % (100.0 * m["correct"] / len(tasks),
                 100.0 * f["correct"] / len(tasks)))
        print("цель Искры — экономия 3-5× и удержание точности к 15-20 ходу.")


if __name__ == "__main__":
    main()
