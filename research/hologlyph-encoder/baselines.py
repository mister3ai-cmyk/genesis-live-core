# Business Source License 1.1 - see research/LICENSE
"""baselines.py - scorers of TZ v2 §4 over one question's haystack.
2026-11-21 · 9 functions. Inputs are preprocessed (unit) vectors; every scorer
returns one score per node of the haystack.

  B0 bm25(F0)            B1 <f0, q>             B2 <f_concat, q>
  B3 <mean_f / |.|, q>   B4 max_f <f, q>        B5 sum_f <f, q>^2 (no QR)
  G1 |Y q|^2 (QR of the faces, encoder_v1.subspaces)
  H  RRF(best of B4 / G1, BM25)
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[a-z0-9]+")


def tokens(text):
    return _TOK.findall(text.lower())


def bm25(doc_texts, query):
    return np.asarray(BM25Okapi([tokens(t) or ["_"] for t in doc_texts]).get_scores(tokens(query)))


def flat(F0, q):
    return F0 @ q


def mean_face(F, q):                      # F: (n, k, d)
    m = F.mean(axis=1)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return m @ q


def max_sim(F, q):
    return (F @ q).max(axis=1)


def sum_sq(F, q):
    p = F @ q
    return (p * p).sum(axis=1)


def rrf(*score_lists, k=60):
    out = np.zeros(len(score_lists[0]))
    for s in score_lists:
        rank = np.empty(len(s))
        rank[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + rank)
    return out
