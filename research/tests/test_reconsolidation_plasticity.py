#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the reconsolidation operator.

Eighty tests in this repository pass today, and none of them noticed that the
update rule in modules/ngp46_reconsolidation_engine.py is an identity: after two
thousand updates the operator sits at ||W - I|| = 5.4e-17, numerically unmoved.
The rule reinforces "W maps q to Wq", which is a tautology, and the Newton-Schulz
projection removes whatever Hermitian residue is left.

These tests encode the two properties the operator must have, so that the defect
cannot pass review again - and so that a C port cannot pass review either
without demonstrating the same two numbers.

  plasticity   updates with an external target must move the operator
  stability    they must not take it off the unitary manifold

Thresholds come from measurement rather than taste, and both are stated with
their cost: in the engine, with its norm clipping and corrective step, the
corrected rule reaches ||W - I|| = 0.362 after 500 updates while holding
non-unitarity at 4.44e-06, against 1.76e-06 for the inert rule. Plasticity rises
by five orders of magnitude and stability costs a factor of 2.5 - the existing
guardrails absorb it, which is what they were for.

Run: python3 -m pytest research/tests/test_reconsolidation_plasticity.py -q
Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
"""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(HERE), "modules",
                      "ngp46_reconsolidation_engine.py")

DIM = 64
STEPS = 500

# Измерено: исправленное правило даёт 0.362 за 500 шагов, прежнее 8.8e-07.
# Порог посередине на логарифмической шкале — ловит тождество, не придирается
# к скорости обучения.
MIN_PLASTICITY = 0.05
# Движок с клиппингом ||dW||_F <= 0.05 и корректирующим шагом держит 4.44e-06
# против 1.76e-06 у инертного правила. Порог примерно в 20 раз выше измеренного:
# ловит реальный уход с многообразия, не придирается к шуму.
MAX_NON_UNITARITY = 1.0e-04


def load_engine():
    if not os.path.exists(ENGINE):
        pytest.skip("engine module not present")
    spec = importlib.util.spec_from_file_location("ngp46", ENGINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def drive(engine, steps, external):
    """Прогон с внешней целью или без неё. Без цели — текущее поведение."""
    rng = np.random.default_rng(7)
    for _ in range(steps):
        q = unit(rng.standard_normal(DIM) + 1j * rng.standard_normal(DIM))
        if external:
            t = unit(rng.standard_normal(DIM) + 1j * rng.standard_normal(DIM))
            engine.generate_context(q, target=t)
        else:
            engine.generate_context(q)
    return engine.W_phase


def fresh():
    mod = load_engine()
    return mod, mod.NGP46ReconsolidationEngineHPC(
        dim=DIM, estimator=mod.HeuristicBurstEstimator())


def test_external_target_accepted():
    """Без входа для цели обучать нечем: цель, равная собственному выходу,
    даёт тождество. Это первое, что должен уметь исправленный движок."""
    mod, engine = fresh()
    q = unit(np.random.default_rng(1).standard_normal(DIM) + 0j)
    try:
        engine.generate_context(q, target=q)
    except TypeError as e:
        pytest.fail(
            "generate_context не принимает target. Пока цель вычисляется из "
            "собственного выхода, правило остаётся тождеством: измерено "
            "||W-I|| = 5.4e-17 после 2000 обновлений. Подробности: %s" % e)


def test_plasticity_under_external_target():
    """Оператор обязан двигаться, когда ему дают настоящую цель."""
    mod, engine = fresh()
    I = np.eye(DIM, dtype=np.complex128)
    W = drive(engine, STEPS, external=True)
    moved = float(np.linalg.norm(W - I, "fro"))
    assert moved > MIN_PLASTICITY, (
        "после %d обновлений с внешней целью ||W-I|| = %.3e, порог %.3e. "
        "Оператор не учится: либо цель всё ещё берётся из собственного выхода, "
        "либо проекция Ньютона-Шульца съедает добавку целиком."
        % (STEPS, moved, MIN_PLASTICITY))


def test_stability_under_external_target():
    """И обязан остаться на многообразии, пока движется."""
    mod, engine = fresh()
    I = np.eye(DIM, dtype=np.complex128)
    W = drive(engine, STEPS, external=True)
    drift = float(np.linalg.norm(np.conj(W.T) @ W - I, "fro"))
    assert drift < MAX_NON_UNITARITY, (
        "неунитарность %.3e превышает порог %.3e: правило Ойи уводит оператор "
        "с Грассманиана быстрее, чем шаг Ньютона-Шульца возвращает. Нужен "
        "второй корректирующий шаг или меньший eta." % (drift, MAX_NON_UNITARITY))


def test_self_target_is_inert_by_construction():
    """Характеризующий тест, а не требование: фиксирует ПОЧЕМУ нужна внешняя
    цель. Цель, равная собственному выходу, не двигает оператор ни при каком
    шаге обучения — эрмитова ранг-1 добавка к унитарной матрице имеет ту же
    матрицу своим унитарным полярным множителем."""
    mod, engine = fresh()
    I = np.eye(DIM, dtype=np.complex128)
    W = drive(engine, 200, external=False)
    moved = float(np.linalg.norm(W - I, "fro"))
    assert moved < 1.0e-02, (
        "ожидалась инертность при цели из собственного выхода, получено %.3e. "
        "Если это изменилось — обновите разбор в разделе 7 сводки." % moved)


if __name__ == "__main__":
    # ТЗ ревьюера называет этот файл как исполняемую проверку, а pytest-модуль
    # при прямом запуске молча завершается с кодом 0 - то есть выглядит как
    # пройденная проверка, ничего не проверив. Поэтому здесь явный прогон с
    # печатью самих величин: аудитору нужны числа, а не слово "passed".
    mod, engine = fresh()
    I = np.eye(DIM, dtype=np.complex128)
    W = drive(engine, STEPS, external=True)
    moved = float(np.linalg.norm(W - I, "fro"))
    drift = float(np.linalg.norm(np.conj(W.T) @ W - I, "fro"))

    mod2, inert = fresh()
    W2 = drive(inert, 200, external=False)
    inert_moved = float(np.linalg.norm(W2 - I, "fro"))

    print("dim %d, %d updates with an external target" % (DIM, STEPS))
    print("plasticity   ||W - I||      = %.3e  (threshold > %.2e)"
          % (moved, MIN_PLASTICITY))
    print("stability    ||W^H W - I||  = %.3e  (threshold < %.2e)"
          % (drift, MAX_NON_UNITARITY))
    print("control: target taken from the operator's own output, 200 updates")
    print("             ||W - I||      = %.3e  (inert by construction)"
          % inert_moved)
    ok = moved > MIN_PLASTICITY and drift < MAX_NON_UNITARITY
    print("RESULT: %s" % ("invariants hold" if ok else "INVARIANTS BROKEN"))
    raise SystemExit(0 if ok else 1)
