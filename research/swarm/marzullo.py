"""Шаг 3 (v3): согласование интервалов по Марзулло с допуском f сбойных.

Интервал - отрезок по каждой оси (для координаты - прямоугольник). Критерий
Марзулло (как в алгоритме пересечения NTP): из m интервалов не больше f
сбойных, значит истина лежит в области, которую покрывают не меньше m - f
интервалов. Для малого m (в рое m <= n) область ищется перебором подмножеств -
это точно и для прямоугольников, где классический проход по одной оси не годится.

judge(e, S, f) - про заявку e при остальных источниках S:
  -1  опровергнута: S согласны между собой с допуском f, и ни одно такое
      согласие не задевает e;
  +1  согласуется: e лежит в области, покрытой >= m - f интервалами вместе с e;
   0  решить нельзя: источников меньше 2f + 1 или у S нет согласия.
"""
from itertools import combinations

REFUTED, UNDECIDED, CONSISTENT = -1, 0, 1


def box(center, half):
    return [(c - half, c + half) for c in center]


def meet(a, b):
    r = [(max(x0, y0), min(x1, y1)) for (x0, x1), (y0, y1) in zip(a, b)]
    return r if all(lo <= hi for lo, hi in r) else None


def common(boxes):
    it = iter(boxes)
    acc = next(it)
    for b in it:
        acc = meet(acc, b)
        if acc is None:
            return None
    return acc


def judge(e, sources, f=1):
    s = len(sources)
    if f < 1 or s < 2 * f:              # m = s + 1 < 2f + 1
        return UNDECIDED
    for t in combinations(sources, s - f):
        if common((e,) + t) is not None:
            return CONSISTENT
    if any(common(t) is not None for t in combinations(sources, s + 1 - f)):
        return REFUTED
    return UNDECIDED
