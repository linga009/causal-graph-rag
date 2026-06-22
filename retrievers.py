"""
retrievers.py
=============
Three complementary retrieval channels used to pick GRAPH ENTRY NODES, fused
with Reciprocal Rank Fusion (RRF). Each channel covers the others' blind spots:

  * BM25      — exact terms, IDs, proper nouns, acronyms (dense smooths these).
  * Dense     — paraphrase / synonyms (BM25 misses these).
  * VSA       — causal DIRECTION (both others are direction-blind).

Dependency-light: BM25 is implemented from scratch; the "dense" channel is a
hashed-bag-of-trigrams embedding (a stand-in for a real sentence encoder —
swap in SentenceTransformers / OpenAI / Voyage in production). RRF is
parameter-free (k=60) and scale-agnostic, so mixing these heterogeneous
scorers is safe.

These return ranked NODES (events). The pipeline then traverses the causal
graph from those nodes to assemble whole chains.
"""

from __future__ import annotations
import math
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


_TOK = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> List[str]:
    return _TOK.findall(s.lower())


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
                v[hash(g) % self.dim] += 1.0
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
class SentenceTransformerDense:
    """Cosine similarity over sentence-transformer embeddings.
    Uses all-MiniLM-L6-v2 by default: 384-dim, ~80 MB, no API key required.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
        self._model = SentenceTransformer(model_name)
        self.vecs: Dict[str, np.ndarray] = {}
        self._nodes: List[str] = []
        self._matrix: np.ndarray | None = None

    def index(self, node_docs: Dict[str, str]) -> None:
        self._nodes = list(node_docs.keys())
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
#  Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def rrf_fuse(ranked_lists: List[List[Tuple[float, str]]], k: int = 60,
             weights: List[float] = None) -> List[Tuple[float, str]]:
    """Combine multiple ranked node lists by rank position only (scale-agnostic)."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused: Dict[str, float] = defaultdict(float)
    for lst, w in zip(ranked_lists, weights):
        for rank, (_, node) in enumerate(lst):
            fused[node] += w * 1.0 / (k + rank + 1)
    out = sorted(fused.items(), key=lambda x: -x[1])
    return [(score, node) for node, score in out]
