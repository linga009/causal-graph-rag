"""
test_core.py — core engine smoke + regression tests.

Run: pytest tests/ -q
These tests need no API key and no Neo4j server (the Neo4j test uses a fake
driver to exercise the edge-id logic in isolation).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from vsa_core import Lexicon, Triple, encode_triple, hamming_similarity
from causal_graph import CausalGraph
from causal_extractor import CausalEdge
from retrievers import _stable_hash, HashingDense, rrf_fuse
from graph_rag import GraphRAG


CORPUS = (
    "The reactor overheated. The coolant valve failed. "
    "This triggered a shutdown. The shutdown caused an outage. "
    "The outage disrupted hospital operations."
)


# --------------------------------------------------------------------------- #
#  VSA core
# --------------------------------------------------------------------------- #
def test_vsa_triple_self_similarity():
    lex = Lexicon(dim=10000, semantic_weight=0)
    t = Triple("reactor", "overheat", "valve")
    hv = encode_triple(t, lex)
    assert hamming_similarity(hv, hv) == pytest.approx(1.0)


def test_vsa_agent_patient_swap_is_distinct():
    """AGENT<->PATIENT swap must NOT look identical (direction matters)."""
    lex = Lexicon(dim=10000, semantic_weight=0)
    a = encode_triple(Triple("inflation", "drives", "unemployment"), lex)
    b = encode_triple(Triple("unemployment", "drives", "inflation"), lex)
    assert hamming_similarity(a, b) < 0.5


# --------------------------------------------------------------------------- #
#  Determinism (regression: builtin hash() was randomized per process)
# --------------------------------------------------------------------------- #
def test_stable_hash_is_deterministic():
    assert _stable_hash("reactor") == _stable_hash("reactor")


def test_hashing_dense_embeddings_reproducible():
    d1 = HashingDense(dim=256)
    d2 = HashingDense(dim=256)
    d1.index({"a": "reactor overheated badly"})
    d2.index({"a": "reactor overheated badly"})
    assert np.allclose(d1.vecs["a"], d2.vecs["a"])


# --------------------------------------------------------------------------- #
#  Graph traversal
# --------------------------------------------------------------------------- #
def _edge(cause, rel, effect, src):
    return CausalEdge(cause, rel, effect, 1, src)


def test_forward_and_backward_chain():
    g = CausalGraph(Lexicon(dim=2000, semantic_weight=0))
    for c, r, e in [("a", "cause", "b"), ("b", "cause", "c"), ("c", "cause", "d")]:
        g.add_edge(_edge(c, r, e, f"{c} {r} {e}"))
    fwd = g.forward_chain("a", max_depth=6)
    assert any(p[-1].effect == "d" for p in fwd)
    bwd = g.backward_chain("d", max_depth=6)
    # backward chains are returned root-cause-first
    assert any(p[0].cause == "a" for p in bwd)


def test_cycle_safe_traversal():
    g = CausalGraph(Lexicon(dim=2000, semantic_weight=0))
    g.add_edge(_edge("a", "cause", "b", "s1"))
    g.add_edge(_edge("b", "cause", "a", "s2"))  # cycle
    # must terminate, not hang
    paths = g.forward_chain("a", max_depth=10)
    assert len(paths) >= 1


def test_distinct_edge_ids():
    g = CausalGraph(Lexicon(dim=2000, semantic_weight=0))
    g.add_edge(_edge("a", "cause", "b", "s1"))
    g.add_edge(_edge("b", "cause", "c", "s2"))
    ids = [e.edge_id for e in g.edges]
    assert ids == [0, 1]


# --------------------------------------------------------------------------- #
#  End-to-end (MockLLM, no API key)
# --------------------------------------------------------------------------- #
def test_graphrag_end_to_end():
    rag = GraphRAG()
    n = rag.ingest(CORPUS)
    assert n >= 3
    chains = rag.retrieve("What did the overheating ultimately disrupt?", top_k=3)
    assert len(chains) >= 1
    # the disruption of hospital operations should be reachable
    joined = " ".join(c.text() for c in chains)
    assert "operations" in joined.lower()
    answer, used = rag.answer("What caused the outage?")
    assert isinstance(answer, str) and answer


def test_close_is_idempotent_for_inmemory():
    rag = GraphRAG()
    rag.close()
    rag.close()  # must not raise


# --------------------------------------------------------------------------- #
#  Neo4j edge-id logic (regression: every edge previously got edge_id=0)
#  Uses a fake driver so no server is required.
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        q = " ".join(query.split())
        if "max(e.edge_id)" in q:
            ids = [r["edge_id"] for r in self._store["rels"]]
            return _FakeResult({"max_id": max(ids) if ids else None})
        if "CREATE (cause)-[:CAUSES" in q and "UNWIND" not in q:
            self._store["rels"].append({"edge_id": params["edge_id"]})
        return _FakeResult(None)


class _FakeDriver:
    def __init__(self):
        self.store = {"rels": []}

    def session(self, database=None):
        return _FakeSession(self.store)

    def close(self):
        pass


def test_neo4j_edge_ids_are_unique(monkeypatch):
    """Regression: the old MATCH (e:Edge) query matched nothing, so every edge
    got edge_id 0 and traversal collapsed all edges into one.

    Injects a fake `neo4j` module so the test runs without the driver or a
    live server.
    """
    import types

    fake = _FakeDriver()
    fake_neo4j = types.ModuleType("neo4j")
    fake_neo4j.GraphDatabase = type(
        "GraphDatabase", (), {"driver": staticmethod(lambda uri, auth=None: fake)}
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    import importlib
    import neo4j_graph
    importlib.reload(neo4j_graph)

    g = neo4j_graph.Neo4jCausalGraph(
        uri="neo4j://fake", lex=Lexicon(dim=2000, semantic_weight=0)
    )
    g.add_edge(_edge("a", "cause", "b", "s1"))
    g.add_edge(_edge("b", "cause", "c", "s2"))
    g.add_edge(_edge("c", "cause", "d", "s3"))

    ids = [r["edge_id"] for r in fake.store["rels"]]
    assert ids == [0, 1, 2], f"edge_ids must be distinct, got {ids}"
