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

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from causal_extractor import CausalEdge
from causal_graph import GraphEdge
from vsa_core import Lexicon, Triple, encode_triple


class Neo4jCausalGraph:
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

        # Cache for VSA matrix (rebuilt when edges change)
        self._edge_matrix_cache: Optional[np.ndarray] = None
        self._edge_cache: Optional[List[GraphEdge]] = None

    def _init_schema(self) -> None:
        """Create indexes and constraints for efficient queries."""
        with self.driver.session(database=self.database) as session:
            # Node indexes
            session.run(
                "CREATE INDEX node_name IF NOT EXISTS FOR (n:Node) ON (n.name)"
            )
            # Edge indexes
            session.run(
                "CREATE INDEX edge_cause IF NOT EXISTS FOR (e:Edge) ON (e.cause)"
            )
            session.run(
                "CREATE INDEX edge_effect IF NOT EXISTS FOR (e:Edge) ON (e.effect)"
            )
            # Constraints
            session.run(
                "CREATE CONSTRAINT edge_id IF NOT EXISTS FOR (e:Edge) REQUIRE e.edge_id IS UNIQUE"
            )

    def _clear_all(self) -> None:
        """Delete all nodes and edges (for testing)."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate cached edge matrix (call after any mutation)."""
        self._edge_matrix_cache = None
        self._edge_cache = None

    # -- Build --------------------------------------------------------------- #

    def add_edge(self, e: CausalEdge) -> None:
        """Add a causal edge to the graph."""
        # Encode the edge using VSA
        hv = encode_triple(Triple(e.cause, e.relation, e.effect), self.lex)

        with self.driver.session(database=self.database) as session:
            # Get next edge ID
            result = session.run(
                "MATCH (e:Edge) RETURN max(e.edge_id) AS max_id"
            )
            max_id = result.single()[0] or -1
            edge_id = max_id + 1

            # Create/merge nodes
            session.run(
                "MERGE (cause:Node {name: $cause}) "
                "MERGE (effect:Node {name: $effect})",
                cause=e.cause,
                effect=e.effect,
            )

            # Create edge with VSA encoding (stored as serialized numpy array)
            hv_str = self._serialize_vector(hv)
            session.run(
                """
                MATCH (cause:Node {name: $cause})
                MATCH (effect:Node {name: $effect})
                CREATE (cause)-[edge:CAUSES {
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
                relation=e.relation,
                effect=e.effect,
                edge_id=edge_id,
                polarity=e.polarity,
                source_sent=e.source_sent,
                hv=hv_str,
            )

        self._invalidate_cache()

    def nodes(self) -> Set[str]:
        """Get all node names."""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (n:Node) RETURN n.name AS name")
            return {record["name"] for record in result}

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
        """All chains flowing OUT of start (what it ultimately causes)."""
        edges = self._get_edges()
        if not edges:
            return []

        edge_map = {e.edge_id: e for e in edges}
        out_adj = self._build_adjacency("out", edges)
        paths = []

        def dfs(node, path, visited, depth):
            edge_ids = out_adj.get(node, [])
            if not edge_ids or depth >= max_depth:
                if path:
                    paths.append([edge_map[eid] for eid in path])
                return

            extended = False
            for eid in edge_ids:
                e = edge_map[eid]
                if e.effect in visited:
                    continue
                extended = True
                path.append(eid)
                visited.add(e.effect)
                dfs(e.effect, path, visited, depth + 1)
                path.pop()
                visited.discard(e.effect)

            if not extended and path:
                paths.append([edge_map[eid] for eid in path])

        dfs(start, [], {start}, 0)
        return paths

    def backward_chain(
        self, start: str, max_depth: int = 6
    ) -> List[List[GraphEdge]]:
        """All chains flowing INTO start (root causes), in causal order."""
        edges = self._get_edges()
        if not edges:
            return []

        edge_map = {e.edge_id: e for e in edges}
        in_adj = self._build_adjacency("in", edges)
        paths = []

        def dfs(node, path, visited, depth):
            edge_ids = in_adj.get(node, [])
            if not edge_ids or depth >= max_depth:
                if path:
                    paths.append([edge_map[eid] for eid in reversed(path)])
                return

            extended = False
            for eid in edge_ids:
                e = edge_map[eid]
                if e.cause in visited:
                    continue
                extended = True
                path.append(eid)
                visited.add(e.cause)
                dfs(e.cause, path, visited, depth + 1)
                path.pop()
                visited.discard(e.cause)

            if not extended and path:
                paths.append([edge_map[eid] for eid in reversed(path)])

        dfs(start, [], {start}, 0)
        return paths

    def path_between(
        self, src: str, dst: str, max_depth: int = 6
    ) -> Optional[List[GraphEdge]]:
        """Shortest path from src to dst using BFS."""
        edges = self._get_edges()
        if not edges:
            return None

        nodes = self.nodes()
        if src not in nodes or dst not in nodes:
            return None

        edge_map = {e.edge_id: e for e in edges}
        out_adj = self._build_adjacency("out", edges)

        from collections import deque

        queue = deque([(src, [])])
        visited = {src}

        while queue:
            node, path = queue.popleft()
            if node == dst and path:
                return [edge_map[eid] for eid in path]
            if len(path) >= max_depth:
                continue

            for eid in out_adj.get(node, []):
                e = edge_map[eid]
                if e.effect not in visited:
                    visited.add(e.effect)
                    queue.append((e.effect, path + [eid]))

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
        """Close the database connection."""
        if self.driver:
            self.driver.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
