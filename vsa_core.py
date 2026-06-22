"""
vsa_core.py
===========
Vector-Symbolic Architecture (VSA) primitives for a relation-aware RAG engine.

Design decisions (and why they differ from the naive write-up):

1.  GRANULARITY = TRIPLE, NOT DOCUMENT.
    Collapsing a whole document into a single bundled hypervector destroys
    recall: bundling N bound pairs injects O(sqrt(N)) crosstalk noise, so a
    single query triple cannot reliably resonate against a 200-triple bundle.
    Instead every document is decomposed into (subject, relation, object)
    triples; each triple is its own clean role-filler vector stored in an
    item-memory. Retrieval scores a query triple against every stored triple.

2.  ROLE-FILLER BINDING, NOT POSITIONAL PERMUTATION.
    Binding (elementwise multiply for bipolar vectors) ties a filler to its
    grammatical role. This makes phrasing-invariant matching possible while
    still distinguishing AGENT<->PATIENT swaps (the inflation/unemployment case).

3.  HYBRID FILLER VECTORS (the refinement).
    A pure-symbolic filler vector makes synonyms orthogonal ("joblessness" vs
    "unemployment" -> ~0 similarity). Each filler is therefore the bundle of
    a stable random "identity" component and a quantized semantic component
    derived from the token's hashed character n-grams (a cheap, dependency-free
    stand-in for an embedding; swap in real embeddings in production). This
    yields graceful degradation: exact match scores highest, near-synonyms
    still rank above unrelated triples.

All vectors are bipolar {-1, +1}, dimension D (default 10000).
"""

from __future__ import annotations
import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# --------------------------------------------------------------------------- #
#  Low-level bipolar hypervector algebra
# --------------------------------------------------------------------------- #
def _seed_from_string(s: str) -> int:
    """Deterministic 64-bit seed from a string (stable across processes)."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def random_hv(dim: int, key: str) -> np.ndarray:
    """A deterministic random bipolar hypervector keyed by `key`."""
    rng = np.random.default_rng(_seed_from_string(key))
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=dim)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding = elementwise multiply. Result is orthogonal to both inputs.
    Self-inverse for bipolar vectors: bind(bind(a,b), b) == a."""
    return (a * b).astype(np.int8)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Bundling = elementwise sum then sign-threshold back to bipolar.
    Ties (sum == 0) are broken deterministically toward +1."""
    if not vectors:
        raise ValueError("bundle() needs at least one vector")
    acc = np.sum(np.stack(vectors).astype(np.int32), axis=0)
    out = np.where(acc >= 0, 1, -1).astype(np.int8)
    return out


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized agreement in [-1, 1]. +1 identical, 0 orthogonal,
    -1 opposite. Equivalent to cosine for bipolar vectors but computed
    with a cheap dot product (which is popcount-equivalent in hardware)."""
    return float(np.dot(a.astype(np.int32), b.astype(np.int32)) / a.shape[0])


# --------------------------------------------------------------------------- #
#  Lexicon: roles + hybrid (identity + semantic) filler vectors
# --------------------------------------------------------------------------- #
# Fixed abstract grammatical roles.
ROLE_NAMES = ("AGENT", "ACTION", "PATIENT")


class Lexicon:
    """Owns role vectors and lazily-built, cached filler vectors."""

    def __init__(self, dim: int = 10000, semantic_weight: int = 1):
        self.dim = dim
        # semantic_weight = how many copies of the semantic component get
        # bundled in relative to the identity component. >1 favours synonym
        # robustness; 0 makes fillers purely symbolic (strict matching).
        self.semantic_weight = semantic_weight
        self.roles: Dict[str, np.ndarray] = {
            r: random_hv(dim, f"ROLE::{r}") for r in ROLE_NAMES
        }
        self._filler_cache: Dict[str, np.ndarray] = {}

    # -- semantic component -------------------------------------------------- #
    def _semantic_hv(self, token: str) -> np.ndarray:
        """Cheap embedding stand-in: bundle hypervectors of the token's
        character trigrams. Tokens sharing trigrams (run/running, employ/
        unemployment) get correlated vectors -> nonzero similarity.
        Replace this method with a real embedding -> quantization in prod."""
        t = f"#{token.lower()}#"
        grams = [t[i:i + 3] for i in range(len(t) - 2)] or [t]
        return bundle([random_hv(self.dim, f"GRAM::{g}") for g in grams])

    # -- public filler accessor --------------------------------------------- #
    def filler(self, token: str) -> np.ndarray:
        """Returns the bundled (identity+semantic) filler. Kept for direct use
        and similarity probes. NOTE: encoding a triple does NOT use this raw
        bundle (see encode_triple); it role-binds identity and semantic parts
        separately to prevent cross-role semantic leakage on AGENT<->PATIENT
        swaps."""
        token = token.lower().strip()
        if token in self._filler_cache:
            return self._filler_cache[token]
        idn, sem = self.filler_parts(token)
        if self.semantic_weight <= 0 or sem is None:
            hv = idn
        else:
            hv = bundle([idn] + [sem] * self.semantic_weight)
        self._filler_cache[token] = hv
        return hv

    def filler_parts(self, token: str):
        """Return (identity_hv, semantic_hv_or_None) without bundling them."""
        token = token.lower().strip()
        identity = random_hv(self.dim, f"FILL::{token}")
        if self.semantic_weight <= 0:
            return identity, None
        return identity, self._semantic_hv(token)

    def role(self, name: str) -> np.ndarray:
        return self.roles[name]


# --------------------------------------------------------------------------- #
#  Triple encoding
# --------------------------------------------------------------------------- #
@dataclass
class Triple:
    agent: str
    action: str
    patient: str

    def text(self) -> str:
        return f"{self.agent} --{self.action}--> {self.patient}"


def encode_triple(t: Triple, lex: Lexicon) -> np.ndarray:
    """E(t) = bundle over slots of  ROLE ⊗ (identity [+ semantic copies]).

    Crucially, the semantic component is bound to the SAME role as its
    identity, so a token contributes to similarity ONLY when it occupies the
    same role in both triples. This makes an AGENT<->PATIENT swap (opposite
    meaning) collapse to near-orthogonal even though both fillers reappear,
    while still letting synonyms in the *same* role pull the score up."""
    parts: List[np.ndarray] = []
    for role_name, token in (("AGENT", t.agent),
                             ("ACTION", t.action),
                             ("PATIENT", t.patient)):
        role = lex.role(role_name)
        idn, sem = lex.filler_parts(token)
        parts.append(bind(role, idn))
        if sem is not None:
            for _ in range(lex.semantic_weight):
                parts.append(bind(role, sem))
    return bundle(parts)


# --------------------------------------------------------------------------- #
#  Item memory (the retrieval index)
# --------------------------------------------------------------------------- #
@dataclass
class MemoryRecord:
    hv: np.ndarray
    triple: Triple
    doc_id: str
    chunk_text: str


class VSAMemory:
    """Holds encoded relation hypervectors and supports nearest-match query."""

    def __init__(self, lex: Lexicon):
        self.lex = lex
        self.records: List[MemoryRecord] = []
        self._matrix: Optional[np.ndarray] = None  # cached stack for speed

    def add(self, triple: Triple, doc_id: str, chunk_text: str) -> None:
        hv = encode_triple(triple, self.lex)
        self.records.append(MemoryRecord(hv, triple, doc_id, chunk_text))
        self._matrix = None  # invalidate cache

    def _build_matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.stack([r.hv for r in self.records]).astype(np.int32)
        return self._matrix

    def query(self, q_triple: Triple, top_k: int = 5
              ) -> List[Tuple[float, MemoryRecord]]:
        if not self.records:
            return []
        q = encode_triple(q_triple, self.lex).astype(np.int32)
        M = self._build_matrix()
        sims = (M @ q) / self.lex.dim          # vectorized Hamming similarity
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.records[i]) for i in order]
