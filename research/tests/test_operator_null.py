#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Null test: the reconsolidation operator does not change what Memory.search
returns.

Memory.write updates W_phase (research/agent_memory_traffic.py, write), and
Memory.search -> Glyphs.search ranks by vecs @ q times a slot weight; W_phase is
not on that path. So the same scenario run with engine=load_engine(dim) and
with engine=None must give byte-identical search results. If this test fails,
something on the search path started reading the operator - find out what
before trusting any memory-vs-noengine comparison.

Each test also requires that the operator actually moved (||W - I|| > 0):
identical output from an operator that never learned would prove nothing.

Two layers:
  stub encoder   deterministic hash vectors, needs only gcc and numpy, so it
                 runs in CI on every push
  real encoder   paraphrase-multilingual-MiniLM-L12-v2 on this repository's
                 documents; skipped where sentence-transformers is absent

Run: python3 -m pytest research/tests/test_operator_null.py -q
Licensed under the Business Source License 1.1 (SPDX: BUSL-1.1); see research/LICENSE.
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.concept

if shutil.which("gcc") is None:
    pytest.skip("gcc not available: Glyphs needs to compile src/srp_lsh.c",
                allow_module_level=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent_memory_traffic as amt  # noqa: E402
import operator_search_experiment as ox  # noqa: E402

STUB_DIM = 64


class StubEncoder:
    """Same text -> same unit vector, in any process. No meaning at all, which
    is enough: the test asks whether two arms agree, not whether they are
    right."""

    def encode(self, texts, normalize_embeddings=True, **kw):
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(STUB_DIM)
            out.append((v / np.linalg.norm(v)).astype(np.float32))
        return np.stack(out)


@pytest.fixture(scope="module")
def stub_env():
    os.makedirs(amt.CACHE, exist_ok=True)
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((300, STUB_DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [{"id": "glyph_%d" % i, "facet": "research", "src": "stub",
               "text": "background %d" % i} for i in range(len(vecs))]
    return chunks, np.ascontiguousarray(vecs), StubEncoder()


def _assert_null(sc, env):
    with_engine = ox.run_arm(amt.Memory, True, sc, *env)
    without = ox.run_arm(amt.Memory, False, sc, *env)
    assert with_engine["w_shift"] > 1e-3, "operator did not learn"
    assert with_engine["queries"] > 0
    assert json.dumps(with_engine["results"]) == json.dumps(without["results"])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("synonyms", [False, True])
def test_engine_does_not_change_search_stub(stub_env, seed, synonyms):
    _assert_null(ox.build_scenario(seed, synonyms=synonyms), stub_env)


@pytest.fixture(scope="module")
def real_env():
    pytest.importorskip("sentence_transformers")
    return ox.setup()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engine_does_not_change_search_real_encoder(real_env, seed):
    _assert_null(ox.build_scenario(seed), real_env)
