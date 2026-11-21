# Business Source License 1.1 - see research/LICENSE
"""encoder_v1.py - faces F0..F3, embeddings, preprocessing and Gr(4, R^d) subspaces.
2026-11-21 · TZ v2 §2. Embeddings are cached per (model, face) in data/emb/.

Faces of a turn node:
  F0 text of the turn (embedder truncates at its max_seq_length; share reported)
  F1 LLM one-sentence summary, F2 LLM entities/numbers/predicates
     (LLM failure -> F1 = F2 = F0 text, identically for every method)
  F3 local context: previous and next turn of the same session, each cut to
     128 tokens of the embedder's tokenizer
Preprocessing: centre (mean of the dev corpus) -> PCA 384 -> 128 fit on dev
(or Gaussian projection, ablation; none for d = 384) -> L2 normalise.
Subspace: QR in float64 of the 4 face vectors, check |Y^T Y - I|_F <= 1e-5 in
fp32; sigma_min(raw) < 1e-3 -> face replaced by a deterministic random direction
orthogonal to the others (seed = node id).
"""
import hashlib
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "data", "emb")
MODELS = {"bge": "BAAI/bge-small-en-v1.5", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
CTX_TOKENS = 128
_loaded = {}


def model(key):
    if key not in _loaded:
        import torch
        from sentence_transformers import SentenceTransformer
        _loaded[key] = SentenceTransformer(MODELS[key], device="cuda" if torch.cuda.is_available() else "cpu")
    return _loaded[key]


def model_revision(key):
    from huggingface_hub import model_info
    return model_info(MODELS[key]).sha


def n_tokens(key, texts):
    tok = model(key).tokenizer
    return [len(x) for x in tok(texts, add_special_tokens=False, truncation=False)["input_ids"]]


def cut_tokens(key, text, k=CTX_TOKENS):
    tok = model(key).tokenizer
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    return (tok.decode(ids[:k]), len(ids) > k) if len(ids) > k else (text, False)


def f3_text(key, prev_text, next_text):
    """Context face: neighbours, each cut to 128 tokens. Returns (text, n_parts_cut)."""
    parts, cut = [], 0
    for t in (prev_text, next_text):
        if t:
            s, c = cut_tokens(key, t)
            parts.append(s)
            cut += c
    return ("\n".join(parts) if parts else ""), cut


def embed(key, face, strings, batch=64):
    """Embed unique strings; cache to data/emb/{key}_{face}.npy + index json (string sha -> row)."""
    os.makedirs(EMB, exist_ok=True)
    base = os.path.join(EMB, "%s_%s" % (key, face))
    idx_path, npy_path = base + ".json", base + ".npy"
    index, mat = {}, np.zeros((0, 384), np.float32)
    if os.path.exists(idx_path):
        index = json.load(open(idx_path))
        mat = np.load(npy_path)
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    todo = []
    seen = set(index)
    for s in strings:
        h = sha(s)
        if h not in seen:
            seen.add(h)
            todo.append((h, s))
    if todo:
        t0 = time.time()
        m = model(key)
        new = m.encode([s for _, s in todo], batch_size=batch, normalize_embeddings=False,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        for i, (h, _) in enumerate(todo):
            index[h] = len(mat) + i
        mat = np.vstack([mat, new])
        np.save(npy_path, mat)
        json.dump(index, open(idx_path, "w"))
        print("embedded %d %s/%s in %.0f s" % (len(todo), key, face, time.time() - t0), flush=True)
    return lambda s: mat[index[sha(s)]]


# ------------------------------------------------------------ preprocessing

class Prep:
    """centre -> projection -> L2. fit on dev corpus vectors (rows)."""

    def __init__(self, X_fit, d=128, center=True, proj="pca", seed=0):
        X = X_fit.astype(np.float64)
        self.mu = X.mean(0) if center else np.zeros(X.shape[1])
        Xc = X - self.mu
        if d >= X.shape[1]:
            self.P = None
        elif proj == "pca":
            _, _, Vt = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
            self.P = Vt[:d].T
        else:                                   # Gaussian random projection (ablation)
            self.P = np.random.default_rng(seed).standard_normal((X.shape[1], d)) / np.sqrt(d)

    def __call__(self, X):
        Z = np.asarray(X, np.float64) - self.mu
        if self.P is not None:
            Z = Z @ self.P
        n = np.linalg.norm(Z, axis=-1, keepdims=True)
        return Z / np.where(n > 0, n, 1.0)


def subspaces(F, node_ids):
    """F: (n, 4, d) preprocessed faces (float64). Returns (Y fp32 (n, 4, d) orthonormal rows,
    degenerate mask, max |Y^T Y - I|_F)."""
    n, k, d = F.shape
    Q, R = np.linalg.qr(np.transpose(F, (0, 2, 1)))            # (n, d, 4)
    sv = np.linalg.svd(F, compute_uv=False)                     # (n, 4)
    degen = sv[:, -1] < 1e-3
    for i in np.flatnonzero(degen):
        A = F[i].T.copy()                                       # d x 4
        rng = np.random.default_rng(int(node_ids[i]))
        basis = []
        for j in range(k):
            v = A[:, j] if np.linalg.norm(A[:, j]) > 1e-3 else rng.standard_normal(d)
            for b in basis:
                v = v - (v @ b) * b
            if np.linalg.norm(v) < 1e-3:                        # dependent on previous faces
                v = rng.standard_normal(d)
                for b in basis:
                    v = v - (v @ b) * b
            basis.append(v / np.linalg.norm(v))
        Q[i] = np.stack(basis, 1)
    Y = np.transpose(Q, (0, 2, 1)).astype(np.float32)          # rows = faces
    G = np.einsum("nad,nbd->nab", Y, Y) - np.eye(k, dtype=np.float32)
    err = float(np.sqrt((G ** 2).sum(axis=(1, 2))).max())
    return Y, degen, err


def g1_scores(Y, q):
    """||Y q||^2 for each node; q unit (d,)."""
    p = Y @ q
    return (p * p).sum(axis=1), p.sum(axis=1)
