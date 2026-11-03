#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
research/retention_experiment.py

The claim, in the form its architect states it: reconsolidation is not a
retrieval optimiser. It exists to hold a changing memory together - to keep old
knowledge while new knowledge arrives, instead of overwriting it. The thing to
observe is the stability of the trajectory over many turns, not the recall of a
single query.

Seven earlier experiments measured recall and found nothing, which is what they
should have found: recall belongs to the cascade, and the cascade already
reaches 0.906. This measures the property reconsolidation is actually for.

Standard continual-learning shape:

  phase 1   teach a set of old associations q -> x
  phase 2   teach a disjoint set of new ones, never revisiting the old
  phase 3   ask whether the old ones survived, and whether the new ones took

  retention    cos(W q_old, x_old) after phase 2
  acquisition  cos(W q_new, x_new) after phase 2
  forgetting   retention measured right after phase 1, minus retention after 2

Three arms, all given the same external targets:

  naive          accumulate the rank-one updates, no projection at all
  reconsolidate  Oja's rule plus the Newton-Schulz projection, as the engine
                 now implements it
  frozen         no learning, the control that cannot forget because it never
                 learned

Capacity note, so the numbers are read correctly: a unitary operator on C^d has
d^2 real parameters, so it can hold on the order of d associations, not an
unlimited number. Retention falling as the new set grows is expected; the
question is whether reconsolidation falls more slowly than naive accumulation.

Usage: python3 retention_experiment.py [--dim 64] [--old 20] [--new 60]
Licensed under the Apache License, Version 2.0 (SPDX: Apache-2.0); see LICENSE.
"""
import argparse
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "modules", "ngp46_reconsolidation_engine.py")


def load_engine():
    spec = importlib.util.spec_from_file_location("ngp46", ENGINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def pairs(n, dim, rng):
    return [(unit(rng.standard_normal(dim) + 1j * rng.standard_normal(dim)),
             unit(rng.standard_normal(dim) + 1j * rng.standard_normal(dim)))
            for _ in range(n)]


def alignment(W, ps):
    """Насколько оператор отображает q в x. 1.0 — точно, 0 — никак."""
    out = []
    for q, x in ps:
        y = W @ q
        y = y / (np.linalg.norm(y) + 1e-12)
        out.append(abs(complex(np.vdot(x, y))))
    return float(np.mean(out))


def teach_naive(W, ps, eta, rounds):
    for _ in range(rounds):
        for q, x in ps:
            y = W @ q
            W = W + eta * (np.outer(x, np.conj(y)) - np.outer(y, np.conj(y)) @ W)
    return W


def teach_engine(engine, ps, rounds):
    for _ in range(rounds):
        for q, x in ps:
            engine.generate_context(q, target=x)
    return engine.W_phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--old", type=int, default=20)
    ap.add_argument("--new", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--eta", type=float, default=0.015)
    args = ap.parse_args()

    rng = np.random.default_rng(5)
    dim = args.dim
    I = np.eye(dim, dtype=np.complex128)
    old = pairs(args.old, dim, rng)
    new = pairs(args.new, dim, rng)

    mod = load_engine()
    print("dim %d, старых ассоциаций %d, новых %d, проходов %d\n"
          % (dim, args.old, args.new, args.rounds))
    print("%-16s %11s %11s %12s %12s %11s" %
          ("режим", "удерж. п1", "удерж. п2", "забывание", "усвоение", "неунит."))
    print("-" * 80)

    # --- frozen -----------------------------------------------------------
    print("%-16s %11.3f %11.3f %12s %12.3f %11.1e" %
          ("заморожен", alignment(I, old), alignment(I, old), "—",
           alignment(I, new), 0.0))

    # --- naive ------------------------------------------------------------
    W = teach_naive(I.copy(), old, args.eta, args.rounds)
    r1 = alignment(W, old)
    W = teach_naive(W, new, args.eta, args.rounds)
    r2, acq = alignment(W, old), alignment(W, new)
    drift = float(np.linalg.norm(np.conj(W.T) @ W - I, "fro"))
    print("%-16s %11.3f %11.3f %12.3f %12.3f %11.1e" %
          ("наивное", r1, r2, r1 - r2, acq, drift))

    # --- reconsolidation --------------------------------------------------
    eng = mod.NGP46ReconsolidationEngineHPC(
        dim=dim, estimator=mod.HeuristicBurstEstimator())
    W = teach_engine(eng, old, args.rounds)
    r1 = alignment(W, old)
    W = teach_engine(eng, new, args.rounds)
    r2, acq = alignment(W, old), alignment(W, new)
    drift = float(np.linalg.norm(np.conj(W.T) @ W - I, "fro"))
    print("%-16s %11.3f %11.3f %12.3f %12.3f %11.1e" %
          ("реконсолидация", r1, r2, r1 - r2, acq, drift))

    print("-" * 80)
    print("удерж. п1 — соответствие старым парам сразу после обучения на них")
    print("удерж. п2 — оно же после обучения на новых, старые не повторялись")
    print("забывание — разница между ними; меньше значит лучше")
    print("Тезис подтверждается, если у реконсолидации забывание заметно ниже")
    print("наивного при сопоставимом усвоении нового.")


if __name__ == "__main__":
    main()
