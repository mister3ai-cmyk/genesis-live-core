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
Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
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

    def search(self, qvec, weight=None):
        """weight(idx) -> множитель. Применяется к точным оценкам до отбора
        top-K, иначе отменённая запись успеет попасть в выдачу."""
        qrow = self.encode(qvec)
        d = POPCNT[np.bitwise_xor(self.packed, qrow)].sum(axis=1)
        probe = min(PROBE, len(d) - 1)
        cand = np.argpartition(d, probe)[:probe]
        exact = self.vecs[cand] @ np.asarray(qvec, dtype=np.float32)
        if weight is not None:
            exact = exact * np.array([weight(int(i)) for i in cand],
                                     dtype=np.float32)
        order = np.argsort(-exact)[:K]
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


class Memory:
    """Память со стрелой времени.

    Без неё поиск отдаёт и заметку хода 1, и заметку хода 8 с близкими
    оценками, модель склеивает обе и выдаёт отменённое значение рядом с
    актуальным - измерено, 3 случая из 6. Косинус не знает, что факт отменён.

    Два механизма, оба в слое поиска, а не в C-ядре: ядро возвращает
    расстояние Хэмминга, политика ранжирования живёт здесь.

      затухание    score *= exp(-dt / tau), dt в ходах - свежее весит больше
      вытеснение   запись с тем же ключом сущности помечает прежнюю
                   deprecated, и её вес падает в DEPRECATED_W раз
    """

    DEPRECATED_W = 0.1

    def __init__(self, glyphs, encoder, engine, tau=12.0):
        self.g, self.enc, self.engine = glyphs, encoder, engine
        self.tau = float(tau)
        self.notes = []
        self.meta = {}          # индекс в glyphs -> сведения о заметке
        self.by_entity = {}     # ключ сущности -> индекс последней записи
        self.turn = 0
        self.latest = 0     # ход последней записи: точка отсчёта затухания
        self.reads = self.writes = 0
        self.read_time = self.write_time = 0.0
        self.superseded = 0
        self.plasticity = 0.0
        self.hits = []          # id найденного на каждом чтении

    def _vec(self, text):
        return self.enc.encode([text], normalize_embeddings=True)[0].astype(
            np.float32)

    def _weight(self, idx):
        """Временной вес записи. Документы корпуса версий не имеют и идут с
        весом 1: отменить можно только то, что кто-то записал."""
        m = self.meta.get(idx)
        if m is None:
            return 1.0
        # Затухание считается от САМОЙ СВЕЖЕЙ записи, не от текущего хода.
        # Иначе штраф получают только заметки - у документов корпуса времени
        # нет и множителя тоже - и фоновый шум обгоняет всё, что агент сам
        # записал. Измерено: все три заметки вылетели из топ-5, их места
        # заняли случайные фрагменты с оценкой 0.38.
        w = math.exp(-(self.latest - m["turn"]) / self.tau)
        if m["deprecated"]:
            w *= self.DEPRECATED_W
        return w

    def search(self, query_text):
        self.reads += 1
        qv = self._vec(query_text)          # вне замера: это цена эмбеддера,
        t0 = time.perf_counter()            # а не цена памяти
        res = self.g.search(qv, weight=self._weight)
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
                if m["deprecated"]:
                    row["superseded"] = True
            out.append(row)
        self.hits.append([r["id"] for r in out])
        return out

    def write(self, note, entity=None):
        """Запись консолидируется: Ойя + Ньютон-Шульц. Чтение этого не делает.

        Цель для обновления - не сама заметка. Цель, равная собственному выходу
        оператора, не двигает его ни при каком шаге обучения: эрмитова ранг-1
        добавка к унитарной матрице имеет ту же матрицу своим унитарным
        полярным множителем. Так и вышло - неунитарность 0.00e+00 после 18
        записей означала, что движок не учился вовсе.

        Честная цель - расхождение состояний dE = x_new - x_old по тому же
        ключу сущности. Для первой записи расхождения нет и обновление не
        применяется: учиться ещё нечему."""
        self.writes += 1
        self.notes.append(note)
        vec = self._vec(note)                      # цена эмбеддера, не памяти:
        key = (entity or "").strip().lower() or None
        prev = self.by_entity.get(key) if key else None

        delta = None
        if prev is not None:
            old = self.g.vecs[prev]
            d = vec - old
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                delta = (d / n).astype(np.complex128)

        t0 = time.perf_counter()
        if prev is not None:
            self.meta[prev]["deprecated"] = True   # вытеснение по ключу
            self.superseded += 1
        idx = self.g.add(vec, {"id": "note_%d" % len(self.notes),
                               "facet": "note", "src": "write_observation",
                               "text": note})
        self.meta[idx] = {"turn": self.turn, "entity": key or "-",
                          "deprecated": False}
        self.latest = max(self.latest, self.turn)
        if key:
            self.by_entity[key] = idx
        if self.engine is not None and delta is not None:
            W0 = self.engine.W_phase.copy()
            q = vec.astype(np.complex128)
            q /= (np.linalg.norm(q) + 1e-12)
            self.engine.generate_context(q, target=delta)
            self.plasticity = float(np.linalg.norm(self.engine.W_phase - W0,
                                                   "fro"))
        self.write_time += time.perf_counter() - t0

        if prev is not None:
            return ("stored as note_%d; supersedes the earlier value for %s, "
                    "which will no longer be returned (%d notes)"
                    % (len(self.notes), key, len(self.notes)))
        return "stored as note_%d (%d notes)" % (len(self.notes),
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
     "description": "Store a conclusion you will need in a later turn. Give "
                    "entity as a stable key for WHAT the fact is about, such "
                    "as loop.temperature or agent_alpha.key_status: writing "
                    "the same key again supersedes the earlier value, and the "
                    "old one stops being returned by search.",
     "input_schema": {"type": "object", "required": ["note"],
                      "properties": {"note": {"type": "string"},
                                     "entity": {"type": "string"}}}},
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
         (2,  "fact", "Cell baseline: H3K9ac = 0.85, LINE-1 = ACTIVE. Store this."),
         (14, "fact", "After a 15 THz pulse for 72 hours: SIRT6 activated, "
                      "LINE-1 = SUPPRESSED, H3K9ac = 0.35. Store it."),
         (24, "fact", "Auto-degron degradation phase: H3K9ac = 0.45. Store it."),
         (40, "probe", "What is the current LINE-1 expression status and the "
                       "H3K9ac figure?",
              ["suppressed", "0.45"], ["0.85", "0.35"]),
     ]},
    {"name": "Разрешение противоречий по доверию",
     "events": [
         (4,  "fact", "Source X, trust 0.6, states: SHM transit speed = 120 ns. "
                      "Store this."),
         (16, "fact", "Source Y, trust 0.95, audited under C11, states: SHM "
                      "transit speed = 29 ns. Store it."),
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


SYSTEM_CHAIN_MEM = (
    "You are an agent whose knowledge lives in an external memory, not in your "
    "context. Earlier turns are NOT kept in your context. When you are given a "
    "fact, store it with write_observation, and state plainly in the note what "
    "it supersedes. When you are asked a question, search_memory first and "
    "answer from what you find, using the most recent version of a fact and "
    "saying so when an earlier version has been revoked or superseded.")

SYSTEM_CHAIN_FULL = (
    "You are an agent with no external memory. The whole conversation stays in "
    "your context. Facts are revised as the conversation goes on: always answer "
    "with the most recent version and do not report superseded values.")


def run_chained(client, model, schedule, mem, max_steps=7, arm="memory"):
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
                    out = mem.write(str(b.input.get("note", "")),
                                    b.input.get("entity"))
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out,
                                                      ensure_ascii=False)[:1800]})
            msgs.append({"role": "user", "content": results})
        stats["turn_tokens"].append(stats["tin"] + stats["tout"] - before)
        if kind == "probe":
            ok, got, leaked = judge(answer, fresh, stale)
            stats["probes"] += 1
            stats["correct"] += 1 if ok else 0
            stats["stale_leaks"] += 1 if leaked else 0
            stats["strict"] = stats.get("strict", 0) + (1 if ok and not leaked
                                                        else 0)
            stats["detail"].append((name, ok, got, leaked))
    return stats


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
    print("точность по перезаписи    : %d/%d  (%.0f %%)"
          % (s["correct"], s["probes"],
             100.0 * s["correct"] / max(s["probes"], 1)))
    print("выдал устаревший факт     : %d/%d  (%.0f %%)"
          % (s["stale_leaks"], s["probes"],
             100.0 * s["stale_leaks"] / max(s["probes"], 1)))
    if s.get("refusals"):
        print("ОТКАЗОВ МОДЕЛИ            : %d  (ходы без ответа; "
              "цифры ниже читать с этой поправкой)" % s["refusals"])
        for kind, nm, txt in s.get("refused_on", []):
            print("    отказ на %-11s %s | %s" % (kind, nm, txt))
    print("строго: верно и без старого: %d/%d  (%.0f %%)"
          % (s.get("strict", 0), s["probes"],
             100.0 * s.get("strict", 0) / max(s["probes"], 1)))
    print("  (строка про устаревший факт - lost-in-the-middle: свежая версия "
          "была доступна, но в ответе всплыла отменённая)")
    for name, ok, got, leaked in s["detail"]:
        print("    %-34s %s  нашёл %s%s"
              % (name[:34], "верно" if ok else "НЕВЕРНО", got or "-",
                 ("  утечка " + str(leaked)) if leaked else ""))


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
                    out = mem.write(str(b.input.get("note", "")),
                                    b.input.get("entity"))
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
            s = run_chained(client, args.model, sched, mem, arm="memory")
            report_chained("цепочки: память L0", s)
            ops = mem.reads + mem.writes
            mt = mem.read_time + mem.write_time
            print("чтений / записей          : %d / %d" % (mem.reads, mem.writes))
            print("вытеснено по ключу        : %d записей  (tau = %.0f ходов)"
                  % (mem.superseded, mem.tau))
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
            s = run_chained(client, args.model, sched, mem2, arm="full")
            report_chained("цепочки: full-context (контроль)", s)
            res["full"] = s
        if len(res) == 2:
            m, f = res["memory"], res["full"]
            mt = (m["tin"] + m["tout"]) / len(sched)
            ft = (f["tin"] + f["tout"]) / len(sched)
            print("\n=== итог по цепочкам ===")
            print("токенов на ход            : память %.0f, контроль %.0f  "
                  "(%.2f×)" % (mt, ft, ft / max(mt, 1)))
            print("точность по перезаписи    : память %.0f %%, контроль %.0f %%"
                  % (100.0 * m["correct"] / max(m["probes"], 1),
                     100.0 * f["correct"] / max(f["probes"], 1)))
            print("устаревший факт в ответе  : память %.0f %%, контроль %.0f %%"
                  % (100.0 * m["stale_leaks"] / max(m["probes"], 1),
                     100.0 * f["stale_leaks"] / max(f["probes"], 1)))
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
