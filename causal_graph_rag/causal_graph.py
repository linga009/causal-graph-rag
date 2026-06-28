"""
causal_graph.py
===============
In-memory causal graph with VSA-encoded edges and BFS traversal.

Improvements over the original DFS implementation:
  - Implements GraphBackend ABC (interchangeable with Neo4jCausalGraph).
  - Word inverted index for O(1) entry-node lookup (token -> [node_names]).
  - BFS traversal: shorter paths first, stack-safe, MAX_PATHS reduced to 500.
  - Per-node visit cap prevents dense sub-graphs from dominating results.
  - GraphEdge stores confidence from the originating CausalEdge.
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .graph_backend import GraphBackend
from .vsa_core import Lexicon, Triple, encode_triple
from .causal_extractor import CausalEdge

import re as _re
_TOK = _re.compile(r"\w+", _re.UNICODE)   # unicode-aware: keeps accented words whole


def _tokenize(s: str) -> List[str]:
    return _TOK.findall(s.lower())


@dataclass
class GraphEdge:
    cause: str
    relation: str
    effect: str
    polarity: int
    source_sent: str
    hv: np.ndarray            # VSA encoding of (cause, relation, effect)
    edge_id: int
    confidence: float = 0.85  # from the originating CausalEdge


class CausalGraph(GraphBackend):
    # Hard cap on enumerated paths per traversal. BFS naturally returns shorter
    # (higher-quality) paths first, so the cap mostly cuts long tail paths.
    MAX_PATHS = 500
    # Max times a single node may appear as the START of a path in results.
    # Prevents a densely connected hub node from flooding the result set.
    _NODE_CAP = 5

    def __init__(self, lex: Lexicon):
        self.lex = lex
        self._edges: List[GraphEdge] = []
        self.out_adj: Dict[str, List[int]] = defaultdict(list)  # node -> edge ids
        self.in_adj: Dict[str, List[int]] = defaultdict(list)
        self._edge_matrix: Optional[np.ndarray] = None
        self._word_idx: Optional[Dict[str, List[str]]] = None  # token -> [nodes]

    # -- GraphBackend: edges property ---------------------------------------- #
    @property
    def edges(self) -> List[GraphEdge]:
        return self._edges

    # -- build --------------------------------------------------------------- #
    def add_edge(self, e: CausalEdge) -> None:
        hv = encode_triple(Triple(e.cause, e.relation, e.effect), self.lex)
        confidence = getattr(e, "confidence", 0.85)
        ge = GraphEdge(e.cause, e.relation, e.effect, e.polarity,
                       e.source_sent, hv, len(self._edges), confidence)
        self.out_adj[e.cause].append(ge.edge_id)
        self.in_adj[e.effect].append(ge.edge_id)
        self._edges.append(ge)
        self._edge_matrix = None
        self._word_idx = None  # invalidate

    def nodes(self) -> Set[str]:
        return set(self.out_adj) | set(self.in_adj)

    # -- word inverted index ------------------------------------------------- #
    def _build_word_index(self) -> Dict[str, List[str]]:
        idx: Dict[str, List[str]] = defaultdict(list)
        seen: Set[str] = set()
        for node in self.nodes():
            if node in seen:
                continue
            seen.add(node)
            for tok in _tokenize(node):
                idx[tok].append(node)
        return dict(idx)

    def word_index(self) -> Dict[str, List[str]]:
        if self._word_idx is None:
            self._word_idx = self._build_word_index()
        return self._word_idx

    # -- VSA entry-point scoring -------------------------------------------- #
    def _matrix(self) -> np.ndarray:
        if self._edge_matrix is None:
            if not self._edges:
                return np.zeros((0, self.lex.dim), dtype=np.int32)
            self._edge_matrix = np.stack([e.hv for e in self._edges]).astype(np.int32)
        return self._edge_matrix

    def score_edges_by_triple(self, q: Triple, top_k: int = 5
                              ) -> List[Tuple[float, GraphEdge]]:
        if not self._edges:
            return []
        qv = encode_triple(q, self.lex).astype(np.int32)
        mat = self._matrix()
        if mat.shape[0] == 0:
            return []
        sims = (mat @ qv) / self.lex.dim
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self._edges[i]) for i in order]

    # -- traversal: BFS (shorter paths first, stack-safe) ------------------- #
    def forward_chain(self, start: str, max_depth: int = 6) -> List[List[GraphEdge]]:
        """All causal chains flowing OUT of `start`. BFS returns shorter paths
        first; per-node visit cap prevents hub-node flooding."""
        return self._bfs(start, self.out_adj, forward=True, max_depth=max_depth)

    def backward_chain(self, start: str, max_depth: int = 6) -> List[List[GraphEdge]]:
        """All chains flowing INTO `start` (root causes), in causal order."""
        raw = self._bfs(start, self.in_adj, forward=False, max_depth=max_depth)
        return [list(reversed(p)) for p in raw]

    def _bfs(self, start: str, adj: Dict[str, List[int]],
             forward: bool, max_depth: int) -> List[List[GraphEdge]]:
        """BFS traversal. Each queue item is (current_node, path, visited).
        Collects complete paths (dead ends or max_depth reached)."""
        paths: List[List[GraphEdge]] = []
        node_count: Dict[str, int] = defaultdict(int)

        # Queue: (node, path_edges, visited_nodes)
        queue: deque = deque()
        queue.append((start, [], frozenset({start})))

        while queue and len(paths) < self.MAX_PATHS:
            node, path, visited = queue.popleft()

            edge_ids = adj.get(node, [])

            if not edge_ids or len(path) >= max_depth:
                if path:
                    paths.append(list(path))
                continue

            extended = False
            for eid in edge_ids:
                if len(paths) >= self.MAX_PATHS:
                    break
                e = self._edges[eid]
                nxt = e.effect if forward else e.cause

                if nxt in visited:
                    continue

                # Per-node cap: don't start too many paths from the same hub
                if node_count[nxt] >= self._NODE_CAP:
                    continue

                extended = True
                node_count[nxt] += 1
                queue.append((nxt, path + [e], visited | {nxt}))

            if not extended and path:
                paths.append(list(path))

        return paths

    def path_between(self, src: str, dst: str, max_depth: int = 6
                     ) -> Optional[List[GraphEdge]]:
        """Shortest causal path src -> dst (BFS). None if unconnected."""
        all_nodes = self.nodes()
        if src not in all_nodes or dst not in all_nodes:
            return None
        queue: deque = deque([(src, [])])
        visited = {src}
        while queue:
            node, path = queue.popleft()
            if node == dst and path:
                return path
            if len(path) >= max_depth:
                continue
            for eid in self.out_adj.get(node, []):
                e = self._edges[eid]
                if e.effect not in visited:
                    visited.add(e.effect)
                    queue.append((e.effect, path + [e]))
        return None

    # -- chain helpers ------------------------------------------------------- #
    @staticmethod
    def chain_polarity(path: List[GraphEdge]) -> int:
        p = 1
        for e in path:
            p *= e.polarity
        return p

    @staticmethod
    def chain_text(path: List[GraphEdge]) -> str:
        if not path:
            return ""
        parts = [path[0].cause]
        for e in path:
            arrow = "->" if e.polarity > 0 else "-/->"
            parts.append(f"{arrow}({e.relation}) {e.effect}")
        return " ".join(parts)
