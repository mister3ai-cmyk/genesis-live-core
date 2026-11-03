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

    def search(self, qvec):
        qrow = self.encode(qvec)
        d = POPCNT[np.bitwise_xor(self.packed, qrow)].sum(axis=1)
        cand = np.argpartition(d, PROBE)[:PROBE]
        exact = self.vecs[cand] @ np.asarray(qvec, dtype=np.float32)
        order = cand[np.argsort(-exact)][:K]
        return [(int(i), float(self.vecs[i] @ qvec)) for i in order]


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
    def __init__(self, glyphs, encoder, engine):
        self.g, self.enc, self.engine = glyphs, encoder, engine
        self.notes = []
        self.reads = self.writes = 0
        self.read_time = self.write_time = 0.0
        self.hits = []          # id найденного на каждом чтении

    def _vec(self, text):
        return self.enc.encode([text], normalize_embeddings=True)[0].astype(
            np.float32)

    def search(self, query_text):
        self.reads += 1
        qv = self._vec(query_text)          # вне замера: это цена эмбеддера,
        t0 = time.perf_counter()            # а не цена памяти
        res = self.g.search(qv)
        self.read_time += time.perf_counter() - t0
        out = []
        for i, score in res:
            c = self.g.chunks[i]
            out.append({"id": c["id"], "facet": c["facet"], "src": c["src"],
                        "snippet": c["text"][:320], "score": round(score, 3)})
        self.hits.append([r["id"] for r in out])
        return out

    def write(self, note):
        """Запись консолидируется: Ойя + Ньютон-Шульц. Чтение этого не делает."""
        self.writes += 1
        self.notes.append(note)
        v = self._vec(note).astype(np.complex128)   # цена эмбеддера, не памяти:
        q = v / (np.linalg.norm(v) + 1e-12)         # вне замера, как при чтении
        t0 = time.perf_counter()
        if self.engine is not None:
            self.engine.generate_context(q, target=q)
        self.write_time += time.perf_counter() - t0
        return "stored (%d notes)" % len(self.notes)


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
     "description": "Store a conclusion you will need in a later turn.",
     "input_schema": {"type": "object", "required": ["note"],
                      "properties": {"note": {"type": "string"}}}},
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
                    out = mem.write(str(b.input.get("note", "")))
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

    rng = np.random.default_rng(11)
    tasks = build_tasks(chunks, args.turns, rng)
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
