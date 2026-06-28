"""
graph_backend.py
================
Abstract base class (protocol) that every causal-graph backend must satisfy.

Both CausalGraph (in-memory) and Neo4jCausalGraph implement this interface,
making them interchangeable in GraphRAG without any isinstance checks.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple


class GraphBackend(ABC):
    """Formal contract for causal-graph storage and traversal backends."""

    # -- mutation ------------------------------------------------------------ #

    @abstractmethod
    def add_edge(self, e) -> None:
        """Persist one causal edge (CausalEdge) to the backend."""

    # -- read --------------------------------------------------------------- #

    @property
    @abstractmethod
    def edges(self) -> List:
        """All GraphEdge objects currently stored."""

    @abstractmethod
    def nodes(self) -> Set[str]:
        """All node names (cause and effect) currently stored."""

    @abstractmethod
    def word_index(self) -> Dict[str, List[str]]:
        """Inverted index: content word → list of node names containing it.
        Used for O(1) entry-node lookup during retrieval."""

    # -- VSA scoring -------------------------------------------------------- #

    @abstractmethod
    def score_edges_by_triple(self, q, top_k: int = 5) -> List[Tuple[float, object]]:
        """Score stored edges by VSA similarity to the query Triple q."""

    # -- traversal ---------------------------------------------------------- #

    @abstractmethod
    def forward_chain(self, start: str, max_depth: int = 6) -> List[List]:
        """All causal chains flowing OUT of *start* (what it ultimately causes)."""

    @abstractmethod
    def backward_chain(self, start: str, max_depth: int = 6) -> List[List]:
        """All chains flowing INTO *start* (root causes), in causal order."""

    @abstractmethod
    def path_between(self, src: str, dst: str, max_depth: int = 6) -> Optional[List]:
        """Shortest causal path src → dst (BFS). None if unconnected."""

    # -- static helpers (shared across backends) ----------------------------- #

    @staticmethod
    def chain_polarity(path: List) -> int:
        p = 1
        for e in path:
            p *= e.polarity
        return p

    @staticmethod
    def chain_text(path: List) -> str:
        if not path:
            return ""
        parts = [path[0].cause]
        for e in path:
            arrow = "->" if e.polarity > 0 else "-/->"
            parts.append(f"{arrow}({e.relation}) {e.effect}")
        return " ".join(parts)
