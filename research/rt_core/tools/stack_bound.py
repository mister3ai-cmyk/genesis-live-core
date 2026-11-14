# Business Source License 1.1 - см. research/LICENSE
"""Статическая граница стека rt_core по графу вызовов GCC (-fcallgraph-info=su).

Читает .ci-файлы (формат VCG), проверяет:
  * нет рекурсии (циклов в графе вызовов);
  * нет стека переменного размера (VLA, alloca) - только «static»;
и печатает для каждой экспортируемой функции худшую цепочку стека: свой
кадр + самая глубокая цепочка вызываемых. Служебные функции libgcc
(__aeabi_*, программная double на M4) в .ci без размера - их кадры не
входят в сумму; итог сверяется с водяным знаком на голом железе.

  python3 tools/stack_bound.py build/ci-m4/*.ci
Код выхода 1 - найдена рекурсия или стек переменного размера.
"""
import re
import sys

NODE = re.compile(r'node: \{ title: "([^"]+)" label: "([^"]*)"')
EDGE = re.compile(r'edge: \{ sourcename: "([^"]+)" targetname: "([^"]+)"')
SIZE = re.compile(r"\\n(\d+) bytes \((\w+)\)")


def load(paths):
    size, kind, calls = {}, {}, {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            m = NODE.search(line)
            if m:
                s = SIZE.search(m.group(2))
                if s:
                    size[m.group(1)], kind[m.group(1)] = int(s.group(1)), s.group(2)
                calls.setdefault(m.group(1), set())
                continue
            m = EDGE.search(line)
            if m:
                calls.setdefault(m.group(1), set()).add(m.group(2))
    return size, kind, calls


def worst(f, size, calls, memo, path):
    """(байт, цепочка); RecursionError-подобный случай - через path."""
    if f in path:
        raise ValueError("рекурсия: " + " -> ".join(path + [f]))
    if f in memo:
        return memo[f]
    best = (0, [])
    for g in calls.get(f, ()):
        if g in size:                                  # своя функция; libgcc - без размера
            best = max(best, worst(g, size, calls, memo, path + [f]), key=lambda x: x[0])
    memo[f] = (size.get(f, 0) + best[0], [f] + best[1])
    return memo[f]


def main(paths):
    size, kind, calls = load(paths)
    bad = [f for f, k in kind.items() if k != "static"]
    if bad:
        print("FAIL: стек переменного размера:", ", ".join(bad))
        return 1
    memo = {}
    try:
        rows = [(f, *worst(f, size, calls, memo, [])) for f in sorted(size) if ":" not in f]
    except ValueError as e:
        print("FAIL:", e)
        return 1
    ext = sorted({g for gs in calls.values() for g in gs if g not in size})
    print("рекурсии нет, стек везде фиксированный (static); функций: %d" % len(size))
    print("%-24s %8s  %s" % ("функция", "байт", "худшая цепочка"))
    for f, b, chain in sorted(rows, key=lambda r: -r[1]):
        print("%-24s %8d  %s" % (f, b, " -> ".join(c.split(":")[-1] for c in chain)))
    print("без размера (libgcc, в сумму не входят): %s" % (" ".join(ext) or "нет"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
