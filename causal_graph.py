"""
causal_graph.py
===============
A directed causal graph whose edges are VSA-encoded so direction survives,
and which can be TRAVERSED to return whole consequential chains rather than
isolated chunks.

This is the component that fixes the structure-loss problem: instead of
storing chunks and hoping similarity search reconstructs the logic, we store
the cause->effect topology explicitly and walk it.

Retrieval entry points come from three fused channels (VSA + BM25 + dense),
implemented in retrievers.py; this module owns the graph + traversal.
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from vsa_core import Lexicon, Triple, encode_triple, hamming_similarity
from causal_extractor import CausalEdge


@dataclass
class GraphEdge:
    cause: str
    relation: str
    effect: str
    polarity: int
    source_sent: str
    hv: np.ndarray            # VSA encoding of (cause, relation, effect)
    edge_id: int


class CausalGraph:
    def __init__(self, lex: Lexicon):
        self.lex = lex
        self.edges: List[GraphEdge] = []
        self.out_adj: Dict[str, List[int]] = defaultdict(list)  # node -> edge ids
        self.in_adj: Dict[str, List[int]] = defaultdict(list)
        self._edge_matrix: Optional[np.ndarray] = None

    # -- build --------------------------------------------------------------- #
    def add_edge(self, e: CausalEdge) -> None:
        # encode the relation as a role-filler triple so direction is preserved
        hv = encode_triple(Triple(e.cause, e.relation, e.effect), self.lex)
        ge = GraphEdge(e.cause, e.relation, e.effect, e.polarity,
                       e.source_sent, hv, len(self.edges))
        self.out_adj[e.cause].append(ge.edge_id)
        self.in_adj[e.effect].append(ge.edge_id)
        self.edges.append(ge)
        self._edge_matrix = None

    def nodes(self) -> Set[str]:
        s = set(self.out_adj) | set(self.in_adj)
        return s

    # -- VSA entry-point scoring -------------------------------------------- #
    def _matrix(self) -> np.ndarray:
        if self._edge_matrix is None:
            self._edge_matrix = np.stack([e.hv for e in self.edges]).astype(np.int32)
        return self._edge_matrix

    def score_edges_by_triple(self, q: Triple, top_k: int = 5
                              ) -> List[Tuple[float, GraphEdge]]:
        if not self.edges:
            return []
        qv = encode_triple(q, self.lex).astype(np.int32)
        sims = (self._matrix() @ qv) / self.lex.dim
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.edges[i]) for i in order]

    # -- traversal: the whole point ----------------------------------------- #
    def forward_chain(self, start: str, max_depth: int = 6) -> List[List[GraphEdge]]:
        """All causal chains flowing OUT of `start` (what it ultimately causes).
        Returns a list of paths, each a list of GraphEdges. Cycle-safe."""
        return self._walk(start, self.out_adj, lambda e: e.effect, max_depth)

    def backward_chain(self, start: str, max_depth: int = 6) -> List[List[GraphEdge]]:
        """All chains flowing INTO `start` (root causes of it). Paths are
        returned in TRUE CAUSAL ORDER (root cause first ... -> start), so they
        render and read correctly downstream."""
        raw = self._walk(start, self.in_adj, lambda e: e.cause, max_depth)
        # _walk collected edges target-first while walking upstream; reverse
        # each path so it reads root-cause -> ... -> start.
        return [list(reversed(p)) for p in raw]

    def _walk(self, start, adj, next_node, max_depth):
        paths: List[List[GraphEdge]] = []

        def dfs(node, path, visited, depth):
            edge_ids = adj.get(node, [])
            if not edge_ids or depth >= max_depth:
                if path:
                    paths.append(list(path))
                return
            extended = False
            for eid in edge_ids:
                e = self.edges[eid]
                nxt = next_node(e)
                if nxt in visited:           # cycle guard
                    continue
                extended = True
                path.append(e)
                visited.add(nxt)
                dfs(nxt, path, visited, depth + 1)
                path.pop()
                visited.discard(nxt)
            if not extended and path:
                paths.append(list(path))

        dfs(start, [], {start}, 0)
        return paths

    def path_between(self, src: str, dst: str, max_depth: int = 6
                     ) -> Optional[List[GraphEdge]]:
        """Shortest causal path src -> dst (BFS). The 'how does X connect to Y'
        query. Returns None if unconnected."""
        if src not in self.nodes() or dst not in self.nodes():
            return None
        queue = deque([(src, [])])
        visited = {src}
        while queue:
            node, path = queue.popleft()
            if node == dst and path:
                return path
            if len(path) >= max_depth:
                continue
            for eid in self.out_adj.get(node, []):
                e = self.edges[eid]
                if e.effect not in visited:
                    visited.add(e.effect)
                    queue.append((e.effect, path + [e]))
        return None

    # -- chain polarity: does the chain net-promote or net-suppress? -------- #
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
