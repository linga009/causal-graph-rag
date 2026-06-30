"""
mongo_graph.py
==============
MongoDB / MongoDB Atlas-backed causal graph — a drop-in `GraphBackend` so the
whole Causal Graph RAG stack runs natively on MongoDB.

Why this fits MongoDB:
  * Causal edges are stored as ordinary documents in one collection — the
    document model maps cleanly onto `(cause, relation, effect, polarity,
    source_sent, hv)`.
  * Native graph traversal uses MongoDB's `$graphLookup` aggregation stage
    (`reachable()` — forward "impact set" / backward "root-cause set"), so
    reachability runs in the database, not the client.
  * The dense coverage channel pairs naturally with **Atlas Vector Search**
    (store the sentence embeddings in MongoDB and `$vectorSearch` them) — see
    the README for wiring that side.

Chain *enumeration* (forward_chain / backward_chain / path_between) returns
distinct paths, which `$graphLookup` (a set/reachability operator) does not
provide, so those use a cached-edge BFS — identical semantics to the in-memory
and Neo4j backends.

Usage
-----
    from causal_graph_rag import GraphRAG
    rag = GraphRAG(mongo_uri="mongodb+srv://user:pass@cluster.mongodb.net")
    rag.ingest(text)
    answer, chains = rag.answer("What caused the outage?")
    rag.close()
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple
import re as _re

import numpy as np

from .graph_backend import GraphBackend
from .causal_extractor import CausalEdge
from .causal_graph import GraphEdge
from .vsa_core import Lexicon, Triple, encode_triple

_TOK = _re.compile(r"\w+", _re.UNICODE)


def _tokenize(s: str) -> List[str]:
    return _TOK.findall(s.lower())


class MongoCausalGraph(GraphBackend):
    """Causal graph persisted in MongoDB; native `$graphLookup` reachability."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "causal_rag",
        collection: str = "causal_edges",
        lex: Optional[Lexicon] = None,
        clear_on_init: bool = False,
        client=None,
    ):
        """
        Parameters
        ----------
        uri : MongoDB / Atlas connection string (e.g. mongodb+srv://...).
        db_name, collection : where edges are stored.
        lex : VSA lexicon for edge encoding.
        clear_on_init : drop existing edges first (tests / fresh ingest).
        client : an existing MongoClient (or mongomock client) to inject —
                 lets unit tests run without a server.
        """
        if client is None:
            try:
                from pymongo import MongoClient
            except ImportError:
                raise ImportError(
                    "pymongo is required for the MongoDB backend. "
                    "Install with: pip install 'causal-graph-rag[mongo]'"
                )
            client = MongoClient(uri)
        self.client = client
        self.db = client[db_name]
        self.col = self.db[collection]
        self.lex = lex or Lexicon(dim=10000)

        # Indexes: fast cause/effect lookup for $graphLookup + unique edge ids.
        try:
            self.col.create_index("cause")
            self.col.create_index("effect")
            self.col.create_index("edge_id", unique=True)
        except Exception:
            pass

        if clear_on_init:
            self.col.delete_many({})

        self._edge_cache: Optional[List[GraphEdge]] = None
        self._matrix_cache: Optional[np.ndarray] = None
        self._out_adj_cache: Optional[Dict[str, List[int]]] = None
        self._in_adj_cache: Optional[Dict[str, List[int]]] = None
        self._word_idx_cache: Optional[Dict[str, List[str]]] = None
        self._next_edge_id: int = self._max_edge_id() + 1

    # -- serialization ------------------------------------------------------- #
    @staticmethod
    def _ser(v: np.ndarray) -> str:
        return v.astype(np.int8).tobytes().hex()

    @staticmethod
    def _deser(hex_str: str) -> np.ndarray:
        return np.frombuffer(bytes.fromhex(hex_str), dtype=np.int8).astype(np.int32)

    def _max_edge_id(self) -> int:
        doc = self.col.find_one(sort=[("edge_id", -1)], projection={"edge_id": 1})
        return doc["edge_id"] if doc else -1

    def _invalidate(self) -> None:
        self._edge_cache = self._matrix_cache = None
        self._out_adj_cache = self._in_adj_cache = self._word_idx_cache = None

    # -- build --------------------------------------------------------------- #
    def _edge_doc(self, e: CausalEdge) -> dict:
        hv = encode_triple(Triple(e.cause, e.relation, e.effect), self.lex)
        doc = {
            "edge_id": self._next_edge_id, "cause": e.cause, "relation": e.relation,
            "effect": e.effect, "polarity": e.polarity, "source_sent": e.source_sent,
            "hv": self._ser(hv),
        }
        self._next_edge_id += 1
        return doc

    def add_edge(self, e: CausalEdge) -> None:
        self.col.insert_one(self._edge_doc(e))
        self._invalidate()

    def add_edges(self, edges: List[CausalEdge]) -> None:
        if not edges:
            return
        self.col.insert_many([self._edge_doc(e) for e in edges])
        self._invalidate()

    def nodes(self) -> Set[str]:
        return set(self.col.distinct("cause")) | set(self.col.distinct("effect"))

    # -- GraphBackend: edges ------------------------------------------------- #
    @property
    def edges(self) -> List[GraphEdge]:
        return self._get_edges()

    def _get_edges(self) -> List[GraphEdge]:
        if self._edge_cache is not None:
            return self._edge_cache
        out = []
        for d in self.col.find().sort("edge_id", 1):
            out.append(GraphEdge(
                cause=d["cause"], relation=d["relation"], effect=d["effect"],
                polarity=d["polarity"], source_sent=d.get("source_sent", ""),
                hv=self._deser(d["hv"]), edge_id=d["edge_id"],
                confidence=d.get("confidence", 0.85)))
        self._edge_cache = out
        return out

    # -- word index ---------------------------------------------------------- #
    def word_index(self) -> Dict[str, List[str]]:
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

    # -- VSA scoring --------------------------------------------------------- #
    def _matrix(self) -> np.ndarray:
        if self._matrix_cache is None:
            edges = self._get_edges()
            self._matrix_cache = (np.stack([e.hv for e in edges]).astype(np.int32)
                                  if edges else np.zeros((0, self.lex.dim), np.int32))
        return self._matrix_cache

    def score_edges_by_triple(self, q: Triple, top_k: int = 5
                              ) -> List[Tuple[float, GraphEdge]]:
        edges = self._get_edges()
        if not edges:
            return []
        qv = encode_triple(q, self.lex).astype(np.int32)
        sims = (self._matrix() @ qv) / self.lex.dim
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), edges[i]) for i in order]

    # -- adjacency + BFS (path enumeration) ---------------------------------- #
    def _adj(self, direction: str) -> Dict[str, List[int]]:
        cache = self._out_adj_cache if direction == "out" else self._in_adj_cache
        if cache is None:
            cache = defaultdict(list)
            for e in self._get_edges():
                key = e.cause if direction == "out" else e.effect
                cache[key].append(e.edge_id)
            cache = dict(cache)
            if direction == "out":
                self._out_adj_cache = cache
            else:
                self._in_adj_cache = cache
        return cache

    def forward_chain(self, start: str, max_depth: int = 6) -> List[List[GraphEdge]]:
        edges = self._get_edges()
        if not edges:
            return []
        emap = {e.edge_id: e for e in edges}
        return self._bfs(start, self._adj("out"), emap, True, max_depth)

    def backward_chain(self, start: str, max_depth: int = 6) -> List[List[GraphEdge]]:
        edges = self._get_edges()
        if not edges:
            return []
        emap = {e.edge_id: e for e in edges}
        raw = self._bfs(start, self._adj("in"), emap, False, max_depth)
        return [list(reversed(p)) for p in raw]

    def _bfs(self, start, adj, emap, forward, max_depth):
        MAX_PATHS, NODE_CAP = 500, 3
        paths: List[List[GraphEdge]] = []
        node_count: Dict[str, int] = defaultdict(int)
        queue = deque([(start, [], frozenset({start}))])
        while queue and len(paths) < MAX_PATHS:
            node, path, visited = queue.popleft()
            eids = adj.get(node, [])
            if not eids or len(path) >= max_depth:
                if path:
                    paths.append(list(path))
                continue
            extended = False
            for eid in eids:
                if len(paths) >= MAX_PATHS:
                    break
                e = emap[eid]
                nxt = e.effect if forward else e.cause
                if nxt in visited or node_count[nxt] >= NODE_CAP:
                    continue
                extended = True
                node_count[nxt] += 1
                queue.append((nxt, path + [e], visited | {nxt}))
            if not extended and path:
                paths.append(list(path))
        return paths

    def path_between(self, src: str, dst: str, max_depth: int = 6
                     ) -> Optional[List[GraphEdge]]:
        edges = self._get_edges()
        if not edges:
            return None
        nodes = self.nodes()
        if src not in nodes or dst not in nodes:
            return None
        # Fast native pre-check: is dst even reachable from src? (skips BFS when not)
        reach = self.reachable(src, "forward", max_depth)
        if reach is not None and dst not in reach:
            return None
        emap = {e.edge_id: e for e in edges}
        out_adj = self._adj("out")
        queue = deque([(src, [])])
        visited = {src}
        while queue:
            node, path = queue.popleft()
            if node == dst and path:
                return path
            if len(path) >= max_depth:
                continue
            for eid in out_adj.get(node, []):
                e = emap[eid]
                if e.effect not in visited:
                    visited.add(e.effect)
                    queue.append((e.effect, path + [e]))
        return None

    # -- native MongoDB traversal: $graphLookup ------------------------------ #
    def reachable(self, start: str, direction: str = "forward",
                  max_depth: int = 6) -> Optional[Dict[str, int]]:
        """All nodes reachable from `start` (forward = downstream impact set;
        backward = upstream root-cause set) with their minimum hop depth, computed
        IN the database with MongoDB's `$graphLookup`.

        Returns {node: depth}, or None if the server doesn't support `$graphLookup`
        (e.g. mongomock) — callers fall back to BFS.
        """
        if direction == "backward":
            seed_match, start_field, conn_from, conn_to = "effect", "$cause", "cause", "effect"
            out_field = "cause"
        else:
            seed_match, start_field, conn_from, conn_to = "cause", "$effect", "effect", "cause"
            out_field = "effect"
        pipeline = [
            {"$match": {seed_match: start}},
            {"$graphLookup": {
                "from": self.col.name, "startWith": start_field,
                "connectFromField": conn_from, "connectToField": conn_to,
                "as": "reached", "maxDepth": max(0, max_depth - 1),
                "depthField": "depth"}},
        ]
        try:
            out: Dict[str, int] = {}
            for doc in self.col.aggregate(pipeline):
                # the seed edge's immediate effect/cause is depth 1
                first = doc.get(out_field)
                if first is not None:
                    out[first] = min(out.get(first, 1), 1)
                for r in doc.get("reached", []):
                    n = r.get(out_field)
                    if n is not None:
                        d = int(r.get("depth", 0)) + 2
                        out[n] = min(out.get(n, d), d)
            return out
        except Exception:
            return None   # $graphLookup unsupported -> signal BFS fallback

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

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            self.client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
