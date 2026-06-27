"""
neo4j_graph.py
==============
Neo4j-backed persistent causal graph for >1M node graphs.

Uses Neo4j as the backend instead of in-memory networkx.DiGraph.
Implements the same interface as CausalGraph for drop-in replacement.

Usage
-----
    from neo4j_graph import Neo4jCausalGraph
    from vsa_core import Lexicon

    lex = Lexicon(dim=10000)
    graph = Neo4jCausalGraph(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="password",
        lex=lex
    )

    # Use exactly like CausalGraph
    graph.add_edge(causal_edge)
    chains = graph.forward_chain("overheating", max_depth=5)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import re as _re
import numpy as np

from graph_backend import GraphBackend
from causal_extractor import CausalEdge
from causal_graph import GraphEdge
from vsa_core import Lexicon, Triple, encode_triple

_TOK = _re.compile(r"\w+", _re.UNICODE)   # unicode-aware: keeps accented words whole


def _tokenize(s: str) -> List[str]:
    return _TOK.findall(s.lower())


class Neo4jCausalGraph(GraphBackend):
    """
    Persistent causal graph backed by Neo4j.

    Handles graphs with millions of nodes/edges efficiently.
    Implements the same interface as in-memory CausalGraph for compatibility.
    """

    def __init__(
        self,
        uri: str = "neo4j://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        lex: Optional[Lexicon] = None,
        clear_on_init: bool = False,
    ):
        """
        Initialize Neo4j-backed causal graph.

        Parameters
        ----------
        uri : str
            Neo4j connection URI (e.g., "neo4j://localhost:7687")
        user : str
            Neo4j username
        password : str
            Neo4j password
        database : str
            Database name to use
        lex : Lexicon
            VSA lexicon for edge encoding
        clear_on_init : bool
            If True, clear all nodes/edges on initialization (for testing)
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j is required. Install with: pip install neo4j>=5.0"
            )

        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.lex = lex or Lexicon(dim=10000)

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Test connection
        try:
            with self.driver.session(database=database) as session:
                session.run("RETURN 1")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j at {uri}: {e}")

        # Initialize schema
        self._init_schema()

        if clear_on_init:
            self._clear_all()

        # Cache for VSA matrix and adjacency (rebuilt when edges change)
        self._edge_matrix_cache: Optional[np.ndarray] = None
        self._edge_cache: Optional[List[GraphEdge]] = None
        self._out_adj_cache: Optional[Dict[str, List[int]]] = None
        self._in_adj_cache: Optional[Dict[str, List[int]]] = None
        self._word_idx_cache: Optional[Dict[str, List[str]]] = None

        # Monotonic edge-id counter. Seeded once from the DB so ids stay unique
        # across restarts; incremented in-memory thereafter. This avoids a
        # MAX() scan on every insert (which is O(n) per edge -> O(n^2) bulk load)
        # and the read-modify-write race two concurrent inserts would hit.
        self._next_edge_id: int = self._max_edge_id() + 1

    def _max_edge_id(self) -> int:
        """Highest edge_id currently stored on :CAUSES relationships (-1 if none)."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH ()-[e:CAUSES]->() RETURN max(e.edge_id) AS max_id"
            )
            row = result.single()
            return (row["max_id"] if row and row["max_id"] is not None else -1)

    def _init_schema(self) -> None:
        """Create indexes and constraints for efficient queries.

        Edges are :CAUSES *relationships*, not :Edge nodes, so the uniqueness
        constraint and lookup indexes must target the relationship type.
        """
        with self.driver.session(database=self.database) as session:
            # Node-name lookup (used by nodes() / traversal entry points)
            session.run(
                "CREATE INDEX node_name IF NOT EXISTS FOR (n:Node) ON (n.name)"
            )
            # Relationship-property indexes (Neo4j 5.x supports REL indexes)
            for stmt in (
                "CREATE INDEX rel_cause IF NOT EXISTS FOR ()-[e:CAUSES]-() ON (e.cause)",
                "CREATE INDEX rel_effect IF NOT EXISTS FOR ()-[e:CAUSES]-() ON (e.effect)",
                "CREATE CONSTRAINT rel_edge_id IF NOT EXISTS "
                "FOR ()-[e:CAUSES]-() REQUIRE e.edge_id IS UNIQUE",
            ):
                try:
                    session.run(stmt)
                except Exception:
                    # Relationship indexes/constraints require Neo4j 5.x+.
                    # Degrade gracefully on older servers rather than fail init.
                    pass

    def _clear_all(self) -> None:
        """Delete all nodes and edges (for testing)."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate all caches after any mutation."""
        self._edge_matrix_cache = None
        self._edge_cache = None
        self._out_adj_cache = None
        self._in_adj_cache = None
        self._word_idx_cache = None

    # -- Build --------------------------------------------------------------- #

    def add_edge(self, e: CausalEdge) -> None:
        """Add a causal edge to the graph.

        Nodes are MERGEd (deduplicated by name) and the :CAUSES relationship is
        created with a unique, monotonically increasing edge_id. The whole
        operation is a single round-trip.
        """
        hv = encode_triple(Triple(e.cause, e.relation, e.effect), self.lex)
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        hv_str = self._serialize_vector(hv)

        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (cause:Node {name: $cause})
                MERGE (effect:Node {name: $effect})
                CREATE (cause)-[:CAUSES {
                    edge_id: $edge_id,
                    cause: $cause,
                    relation: $relation,
                    effect: $effect,
                    polarity: $polarity,
                    source_sent: $source_sent,
                    hv: $hv
                }]->(effect)
                """,
                cause=e.cause,
                effect=e.effect,
                edge_id=edge_id,
                relation=e.relation,
                polarity=e.polarity,
                source_sent=e.source_sent,
                hv=hv_str,
            )

        self._invalidate_cache()

    def add_edges(self, edges: List[CausalEdge]) -> None:
        """Bulk-insert edges in a single transaction (UNWIND).

        Far faster than repeated add_edge for large corpora — one round-trip
        for the whole batch instead of one per edge.
        """
        if not edges:
            return
        rows = []
        for e in edges:
            hv = encode_triple(Triple(e.cause, e.relation, e.effect), self.lex)
            rows.append({
                "edge_id": self._next_edge_id,
                "cause": e.cause,
                "relation": e.relation,
                "effect": e.effect,
                "polarity": e.polarity,
                "source_sent": e.source_sent,
                "hv": self._serialize_vector(hv),
            })
            self._next_edge_id += 1

        with self.driver.session(database=self.database) as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (cause:Node {name: row.cause})
                MERGE (effect:Node {name: row.effect})
                CREATE (cause)-[:CAUSES {
                    edge_id: row.edge_id,
                    cause: row.cause,
                    relation: row.relation,
                    effect: row.effect,
                    polarity: row.polarity,
                    source_sent: row.source_sent,
                    hv: row.hv
                }]->(effect)
                """,
                rows=rows,
            )
        self._invalidate_cache()

    def nodes(self) -> Set[str]:
        """Get all node names."""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (n:Node) RETURN n.name AS name")
            return {record["name"] for record in result}

    # -- GraphBackend: edges property ---------------------------------------- #

    @property
    def edges(self) -> List[GraphEdge]:
        """All edges currently stored (via cached fetch)."""
        return self._get_edges()

    # -- Word inverted index ------------------------------------------------- #

    def word_index(self) -> Dict[str, List[str]]:
        """Inverted index token -> [node_names]. Built from the edge cache."""
        if self._word_idx_cache is not None:
            return self._word_idx_cache
        idx: Dict[str, List[str]] = defaultdict(list)
        seen: set = set()
        for e in self._get_edges():
            for node in (e.cause, e.effect):
                if node in seen:
                    continue
                seen.add(node)
                for tok in _tokenize(node):
                    idx[tok].append(node)
        self._word_idx_cache = dict(idx)
        return self._word_idx_cache

    # -- Cached adjacency ---------------------------------------------------- #

    def _get_out_adj(self) -> Dict[str, List[int]]:
        if self._out_adj_cache is None:
            self._out_adj_cache = self._build_adjacency("out", self._get_edges())
        return self._out_adj_cache

    def _get_in_adj(self) -> Dict[str, List[int]]:
        if self._in_adj_cache is None:
            self._in_adj_cache = self._build_adjacency("in", self._get_edges())
        return self._in_adj_cache

    # -- VSA scoring --------------------------------------------------------- #

    def _serialize_vector(self, v: np.ndarray) -> str:
        """Serialize numpy array to hex string for storage."""
        return v.astype(np.int8).tobytes().hex()

    def _deserialize_vector(self, hex_str: str) -> np.ndarray:
        """Deserialize hex string back to numpy array."""
        return np.frombuffer(bytes.fromhex(hex_str), dtype=np.int8).astype(np.int32)

    def _get_edges(self) -> List[GraphEdge]:
        """Fetch all edges from database (cached)."""
        if self._edge_cache is not None:
            return self._edge_cache

        edges = []
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH ()-[e:CAUSES]->()
                RETURN e.edge_id, e.cause, e.relation, e.effect,
                       e.polarity, e.source_sent, e.hv
                ORDER BY e.edge_id
                """
            )
            for record in result:
                edge_id, cause, relation, effect, polarity, source_sent, hv_str = record.values()
                hv = self._deserialize_vector(hv_str)
                edges.append(
                    GraphEdge(
                        cause=cause,
                        relation=relation,
                        effect=effect,
                        polarity=polarity,
                        source_sent=source_sent,
                        hv=hv,
                        edge_id=edge_id,
                    )
                )

        self._edge_cache = edges
        return edges

    def _matrix(self) -> np.ndarray:
        """Get VSA matrix of all edges (cached)."""
        if self._edge_matrix_cache is not None:
            return self._edge_matrix_cache

        edges = self._get_edges()
        if not edges:
            return np.array([], dtype=np.int32).reshape(0, self.lex.dim)

        self._edge_matrix_cache = np.stack([e.hv for e in edges]).astype(np.int32)
        return self._edge_matrix_cache

    def score_edges_by_triple(
        self, q: Triple, top_k: int = 5
    ) -> List[Tuple[float, GraphEdge]]:
        """Score edges by VSA similarity to query triple."""
        edges = self._get_edges()
        if not edges:
            return []

        matrix = self._matrix()
        qv = encode_triple(q, self.lex).astype(np.int32)
        sims = (matrix @ qv) / self.lex.dim
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), edges[i]) for i in order]

    # -- Traversal ------------------------------------------------------------ #

    def forward_chain(
        self, start: str, max_depth: int = 6
    ) -> List[List[GraphEdge]]:
        """All chains flowing OUT of start. Uses cached adjacency — no DB fetch per call."""
        edges = self._get_edges()
        if not edges:
            return []
        edge_map = {e.edge_id: e for e in edges}
        out_adj = self._get_out_adj()
        return self._bfs(start, out_adj, edge_map, forward=True, max_depth=max_depth)

    def backward_chain(
        self, start: str, max_depth: int = 6
    ) -> List[List[GraphEdge]]:
        """All chains flowing INTO start (root causes), in causal order."""
        edges = self._get_edges()
        if not edges:
            return []
        edge_map = {e.edge_id: e for e in edges}
        in_adj = self._get_in_adj()
        raw = self._bfs(start, in_adj, edge_map, forward=False, max_depth=max_depth)
        return [list(reversed(p)) for p in raw]

    def _bfs(self, start: str, adj: Dict[str, List[int]],
             edge_map: Dict[int, GraphEdge], forward: bool,
             max_depth: int) -> List[List[GraphEdge]]:
        """BFS traversal shared by forward/backward chain. MAX_PATHS=500."""
        MAX_PATHS = 500
        NODE_CAP = 3
        paths: List[List[GraphEdge]] = []
        node_count: Dict[str, int] = defaultdict(int)

        from collections import deque
        queue = deque([(start, [], frozenset({start}))])

        while queue and len(paths) < MAX_PATHS:
            node, path, visited = queue.popleft()
            edge_ids = adj.get(node, [])

            if not edge_ids or len(path) >= max_depth:
                if path:
                    paths.append(list(path))
                continue

            extended = False
            for eid in edge_ids:
                if len(paths) >= MAX_PATHS:
                    break
                e = edge_map[eid]
                nxt = e.effect if forward else e.cause
                if nxt in visited:
                    continue
                if node_count[nxt] >= NODE_CAP:
                    continue
                extended = True
                node_count[nxt] += 1
                queue.append((nxt, path + [e], visited | {nxt}))

            if not extended and path:
                paths.append(list(path))

        return paths

    def path_between(
        self, src: str, dst: str, max_depth: int = 6
    ) -> Optional[List[GraphEdge]]:
        """Shortest path from src to dst using BFS."""
        edges = self._get_edges()
        if not edges:
            return None
        if src not in self.nodes() or dst not in self.nodes():
            return None
        edge_map = {e.edge_id: e for e in edges}
        out_adj = self._get_out_adj()

        from collections import deque
        queue = deque([(src, [])])
        visited = {src}

        while queue:
            node, path = queue.popleft()
            if node == dst and path:
                return path
            if len(path) >= max_depth:
                continue
            for eid in out_adj.get(node, []):
                e = edge_map[eid]
                if e.effect not in visited:
                    visited.add(e.effect)
                    queue.append((e.effect, path + [e]))
        return None

    # -- Helpers -------------------------------------------------------------- #

    def _build_adjacency(
        self, direction: str, edges: List[GraphEdge]
    ) -> Dict[str, List[int]]:
        """Build adjacency dict: node -> edge_ids."""
        adj = {}
        for e in edges:
            if direction == "out":
                key, val = e.cause, e.effect
            else:  # "in"
                key, val = e.effect, e.cause

            if key not in adj:
                adj[key] = []
            adj[key].append(e.edge_id)

        return adj

    @staticmethod
    def chain_polarity(path: List[GraphEdge]) -> int:
        """Net polarity of a chain."""
        p = 1
        for e in path:
            p *= e.polarity
        return p

    @staticmethod
    def chain_text(path: List[GraphEdge]) -> str:
        """Text representation of a chain."""
        if not path:
            return ""
        parts = [path[0].cause]
        for e in path:
            arrow = "->" if e.polarity > 0 else "-/->"
            parts.append(f"{arrow}({e.relation}) {e.effect}")
        return " ".join(parts)

    def close(self) -> None:
        """Close the database connection (idempotent)."""
        driver = getattr(self, "driver", None)
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass
            self.driver = None

    def __del__(self):
        # __del__ can run during interpreter shutdown when modules are already
        # gone; guard everything so GC never raises.
        try:
            self.close()
        except Exception:
            pass
