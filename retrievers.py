"""
retrievers.py
=============
Four complementary retrieval channels used to pick GRAPH ENTRY NODES, fused
with Reciprocal Rank Fusion (RRF). Each channel covers the others' blind spots:

  * BM25            — exact terms, IDs, proper nouns, acronyms.
  * Dense           — paraphrase / synonyms (BM25 misses these).
  * VSA             — causal DIRECTION (both others are direction-blind).
  * PathSignature   — sequential order and trajectory shape of the narrative.

PathSignature channel (Rough Path Theory):
  A document passage is treated not as a static point in embedding space but
  as a continuous parametric curve X_t ∈ R^d.  The truncated path signature
  S(X)_{s,t} = (1, ∫dX, ∫∫dX⊗dX, ∫∫∫dX⊗dX⊗dX, ...)
  is a complete, fixed-size characterisation of the path up to level M.

  Non-commutativity of the iterated integrals (∫dX_i dX_j ≠ ∫dX_j dX_i)
  mathematically guarantees that the *order* in which events appear is encoded
  — chunking-invariant sequential structure that static embeddings cannot
  represent.

  Implementation notes:
    • Sentence embeddings (d=384) are projected to d=proj_dim (default 16)
      via a fixed random Gaussian matrix before computing signatures.
      This controls the d^M explosion: at M=3, d=16 gives 4368 dims.
    • Signatures are computed via Chen's recursive formula applied to
      the discrete sequence of projected sentence embeddings.
    • Queries are augmented with their top BM25 context sentences so that
      both the query path and document paths are multi-point trajectories.

Reference: Lyons (1998), Differential equations driven by rough signals.
"""

from __future__ import annotations
import hashlib
import math
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


_TOK = re.compile(r"\w+", re.UNICODE)   # unicode-aware: keeps accented words whole


def tokenize(s: str) -> List[str]:
    return _TOK.findall(s.lower())


def _stable_hash(s: str) -> int:
    """Process-independent hash. Python's builtin hash() is randomized per
    process (PYTHONHASHSEED), which would make hashed embeddings differ between
    runs. blake2b keeps the index reproducible across restarts."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


# --------------------------------------------------------------------------- #
#  BM25 over node "documents" (the sentences each node participates in)
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: Dict[str, List[str]] = {}     # node -> tokens
        self.df: Counter = Counter()
        self.avgdl = 0.0
        self._N = 0

    def index(self, node_docs: Dict[str, str]) -> None:
        self.docs = {n: tokenize(t) for n, t in node_docs.items()}
        self._N = len(self.docs)
        self.df = Counter()
        total = 0
        for toks in self.docs.values():
            total += len(toks)
            for t in set(toks):
                self.df[t] += 1
        self.avgdl = (total / self._N) if self._N else 0.0

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self._N - n + 0.5) / (n + 0.5))

    def score(self, query: str) -> List[Tuple[float, str]]:
        q = tokenize(query)
        out = []
        for node, toks in self.docs.items():
            if not toks:
                continue
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self._idf(term)
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * num / den
            if s > 0:
                out.append((s, node))
        out.sort(key=lambda x: -x[0])
        return out


# --------------------------------------------------------------------------- #
#  Dense channel — hashed trigram bag, cosine. Stand-in for a real encoder.
# --------------------------------------------------------------------------- #
class HashingDense:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.vecs: Dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        for tok in toks:
            t = f"#{tok}#"
            for i in range(len(t) - 2):
                g = t[i:i + 3]
                v[_stable_hash(g) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def index(self, node_docs: Dict[str, str]) -> None:
        self.vecs = {n: self._embed(t) for n, t in node_docs.items()}

    def score(self, query: str) -> List[Tuple[float, str]]:
        q = self._embed(query)
        out = [(float(np.dot(q, v)), n) for n, v in self.vecs.items()]
        out = [(s, n) for s, n in out if s > 0]
        out.sort(key=lambda x: -x[0])
        return out


# --------------------------------------------------------------------------- #
#  Dense channel — real SentenceTransformers encoder (preferred).
#  Falls back to HashingDense automatically if the package isn't installed.
# --------------------------------------------------------------------------- #
# Process-wide model cache: loading the ~90 MB encoder once and sharing it across
# every retriever/GraphRAG instance avoids redundant loads (a real efficiency
# win) and the native crash from loading it dozens of times in one process.
_ST_MODELS: dict = {}


def shared_st_model(model_name: str = "all-MiniLM-L6-v2"):
    """Return a single shared SentenceTransformer per model name per process."""
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
    if model_name not in _ST_MODELS:
        _ST_MODELS[model_name] = SentenceTransformer(model_name)
    return _ST_MODELS[model_name]


class SentenceTransformerDense:
    """Cosine similarity over sentence-transformer embeddings.
    Uses all-MiniLM-L6-v2 by default: 384-dim, ~80 MB, no API key required.
    Accepts an optional pre-loaded model to share across retrievers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model=None):
        self._model = model if model is not None else shared_st_model(model_name)
        self.vecs: Dict[str, np.ndarray] = {}
        self._nodes: List[str] = []
        self._matrix: np.ndarray | None = None

    def index(self, node_docs: Dict[str, str]) -> None:
        self._nodes = list(node_docs.keys())
        if not self._nodes:                       # zero-edge document -> empty index
            self.vecs = {}
            self._matrix = None
            return
        texts = [node_docs[n] for n in self._nodes]
        embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.vecs = {n: embs[i] for i, n in enumerate(self._nodes)}
        self._matrix = embs  # shape (N, D)

    def score(self, query: str) -> List[Tuple[float, str]]:
        if not self._nodes:
            return []
        q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = (self._matrix @ q).tolist()
        out = [(float(s), n) for s, n in zip(sims, self._nodes) if s > 0]
        out.sort(key=lambda x: -x[0])
        return out


def make_dense() -> "HashingDense | SentenceTransformerDense":
    """Return a SentenceTransformerDense if available, else HashingDense."""
    try:
        return SentenceTransformerDense()
    except Exception:
        return HashingDense()


# --------------------------------------------------------------------------- #
#  Path Signature channel (Rough Path Theory)
# --------------------------------------------------------------------------- #
class PathSignatureRetriever:
    """
    Retrieval via truncated path signatures of sentence-embedding trajectories.

    Each graph node's associated text is embedded sentence-by-sentence to form
    a discrete path X_0, X_1, ..., X_N in projected R^proj_dim space.  The
    level-M truncated signature captures the sequential shape of the passage:

        S^1 = Σ ΔX_i                         (displacement, d dims)
        S^2 = Σ_{i<j} ΔX_i ⊗ ΔX_j           (area / ordering, d² dims)
        S^3 = Σ_{i<j<k} ΔX_i ⊗ ΔX_j ⊗ ΔX_k  (triple ordering, d³ dims)

    For d=16, M=3 this gives 16+256+4096 = 4 368 dimensions — compact enough
    for cosine inner-product search, yet encoding genuine sequential structure.
    """

    def __init__(self, embed_dim: int = 384, proj_dim: int = 16, level: int = 3,
                 model=None):
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.level = level
        self._proj: np.ndarray | None = None      # (embed_dim, proj_dim)
        self._model = model   # injected model or None (lazy-loaded on first use)
        self._nodes: List[str] = []
        self._sig_matrix: np.ndarray | None = None  # (N_nodes, sig_dim)
        self._node_sents: Dict[str, List[str]] = {}  # for BM25 context injection

    # -- random projection --------------------------------------------------- #
    def _projection(self) -> np.ndarray:
        if self._proj is None:
            rng = np.random.default_rng(0xDEADBEEF)
            P = rng.standard_normal((self.embed_dim, self.proj_dim)).astype(np.float32)
            # Johnson-Lindenstrauss normalisation
            P /= np.sqrt(self.proj_dim)
            self._proj = P
        return self._proj

    # -- sentence encoder (used only for query path in score()) -------------- #
    def _encode(self, sentences: List[str]) -> np.ndarray:
        """Return projected embeddings, shape (N, proj_dim). Used for queries."""
        if not sentences:
            return np.zeros((0, self.proj_dim), dtype=np.float32)
        if self._model is None:
            try:
                self._model = shared_st_model("all-MiniLM-L6-v2")
            except Exception:
                self._model = "random"
        if self._model == "random":
            rng = np.random.default_rng(_stable_hash(sentences[0]) % (2**31))
            return rng.standard_normal((len(sentences), self.proj_dim)).astype(np.float32)
        embs = self._model.encode(sentences, convert_to_numpy=True,
                                  normalize_embeddings=True)
        return (embs @ self._projection()).astype(np.float32)

    # -- path signature ------------------------------------------------------ #
    @staticmethod
    def _signature(path: np.ndarray, level: int) -> np.ndarray:
        """
        Truncated path signature via Chen's recursive formula.

        path : (N, d)  — N waypoints in R^d
        level: 1, 2 or 3

        For each increment ΔX the running totals are updated *high-to-low*
        to avoid using already-updated lower-level values in the same step:

            S^3 += S^2_prev ⊗ ΔX
            S^2 += S^1_prev ⊗ ΔX
            S^1 += ΔX
        """
        N, d = path.shape
        sig_dim = sum(d ** k for k in range(1, level + 1))
        if N < 2:
            return np.zeros(sig_dim, dtype=np.float32)

        dX = np.diff(path, axis=0).astype(np.float32)  # (N-1, d)

        S1 = np.zeros(d, dtype=np.float32)
        S2 = np.zeros((d, d), dtype=np.float32) if level >= 2 else None
        S3 = np.zeros((d, d, d), dtype=np.float32) if level >= 3 else None

        for delta in dX:
            # Update high-to-low (Chen's formula — order matters)
            if level >= 3 and S3 is not None and S2 is not None:
                S3 += np.einsum("jk,m->jkm", S2, delta, optimize=True)
            if level >= 2 and S2 is not None:
                S2 += np.outer(S1, delta)
            S1 += delta

        parts: List[np.ndarray] = [S1]
        if level >= 2 and S2 is not None:
            parts.append(S2.ravel())
        if level >= 3 and S3 is not None:
            parts.append(S3.ravel())
        return np.concatenate(parts)

    # -- split doc text into ordered sentences ------------------------------- #
    @staticmethod
    def _split(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]

    # -- public API ---------------------------------------------------------- #
    def index(self, node_docs: Dict[str, str]) -> None:
        self._nodes = list(node_docs.keys())
        if not self._nodes:                       # zero-edge document -> empty index
            self._node_sents = {}
            self._sig_matrix = None
            return
        self._node_sents = {n: self._split(node_docs[n]) or [node_docs[n]]
                            for n in self._nodes}

        # Batch-encode all sentences from all nodes in ONE model.encode() call
        # instead of N separate calls. Dramatically faster for large corpora.
        all_sents: List[str] = []
        offsets: List[int] = [0]
        for node in self._nodes:
            all_sents.extend(self._node_sents[node])
            offsets.append(len(all_sents))

        if all_sents:
            # Encode the full batch at once, then project
            if self._model is None:
                try:
                    self._model = shared_st_model("all-MiniLM-L6-v2")
                except Exception:
                    self._model = "random"
            if self._model == "random":
                rng = np.random.default_rng(_stable_hash(all_sents[0]) % (2**31))
                all_embs_raw = rng.standard_normal(
                    (len(all_sents), self.embed_dim)).astype(np.float32)
            else:
                all_embs_raw = self._model.encode(
                    all_sents, convert_to_numpy=True,
                    normalize_embeddings=True).astype(np.float32)
            proj = self._projection()
            all_embs = all_embs_raw @ proj   # (total_sents, proj_dim)
        else:
            all_embs = np.zeros((0, self.proj_dim), dtype=np.float32)

        sigs = []
        for i, node in enumerate(self._nodes):
            node_embs = all_embs[offsets[i]:offsets[i + 1]]
            sigs.append(self._signature(node_embs, self.level))
        self._sig_matrix = np.stack(sigs).astype(np.float32)  # (N, sig_dim)

    def score(self, query: str,
              bm25_context: List[str] | None = None) -> List[Tuple[float, str]]:
        """
        Score nodes by inner product between their signature and the query
        signature.

        The query becomes a multi-point path by prepending BM25 context
        sentences (in their original document order) before the query itself.
        This lets the level-2 and level-3 components carry real sequential
        information rather than being zero (as they would for a single point).
        """
        if self._sig_matrix is None or not self._nodes:
            return []

        path_sents = (bm25_context or []) + [query]
        path = self._encode(path_sents)
        q_sig = self._signature(path, self.level).astype(np.float32)

        # Cosine-normalised inner product
        q_norm = np.linalg.norm(q_sig) + 1e-9
        d_norms = np.linalg.norm(self._sig_matrix, axis=1, keepdims=True) + 1e-9
        scores = (self._sig_matrix / d_norms) @ (q_sig / q_norm)

        out = [(float(s), n) for s, n in zip(scores.tolist(), self._nodes) if s > 0]
        out.sort(key=lambda x: -x[0])
        return out


# --------------------------------------------------------------------------- #
#  Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def rrf_fuse(ranked_lists: List[List[Tuple[float, str]]], k: int = 60,
             weights: Optional[List[float]] = None) -> List[Tuple[float, str]]:
    """Combine multiple ranked node lists by rank position only (scale-agnostic)."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused: Dict[str, float] = defaultdict(float)
    for lst, w in zip(ranked_lists, weights):
        for rank, (_, node) in enumerate(lst):
            fused[node] += w * 1.0 / (k + rank + 1)
    out = sorted(fused.items(), key=lambda x: -x[1])
    return [(score, node) for node, score in out]
