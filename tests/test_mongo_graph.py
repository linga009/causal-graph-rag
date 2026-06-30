"""
Tests for the MongoDB backend (MongoCausalGraph) using mongomock — no server.

mongomock doesn't implement $graphLookup, so reachable() returns None there and
the code falls back to BFS; these tests exercise storage, the GraphBackend
contract, and end-to-end GraphRAG-on-Mongo via an injected client.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mongomock = pytest.importorskip("mongomock")

from causal_graph_rag.mongo_graph import MongoCausalGraph
from causal_graph_rag.causal_extractor import CausalEdge
from causal_graph_rag.vsa_core import Lexicon


def _edge(c, r, e, pol=1, src=""):
    return CausalEdge(c, r, e, pol, src or f"{c} {r} {e}")


def _graph(clear=True):
    client = mongomock.MongoClient()
    return MongoCausalGraph(lex=Lexicon(dim=1000), client=client, clear_on_init=clear)


CHAIN = [
    _edge("pump", "cause", "overheat"),
    _edge("overheat", "trigger", "scram"),
    _edge("scram", "lead_to", "outage"),
    _edge("outage", "disrupt", "hospital"),
]


def test_add_and_nodes():
    g = _graph()
    g.add_edges(CHAIN)
    assert g.nodes() == {"pump", "overheat", "scram", "outage", "hospital"}
    assert len(list(g.edges)) == 4


def test_unique_monotonic_edge_ids():
    g = _graph()
    g.add_edges(CHAIN)
    ids = [e.edge_id for e in g.edges]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_word_index():
    g = _graph()
    g.add_edges(CHAIN)
    widx = g.word_index()
    assert "pump" in widx and "hospital" in widx
    assert "overheat" in widx["overheat"]


def test_forward_chain_enumerates_full_path():
    g = _graph()
    g.add_edges(CHAIN)
    chains = g.forward_chain("pump", max_depth=6)
    assert chains, "expected at least one forward chain"
    longest = max(chains, key=len)
    assert [e.cause for e in longest][0] == "pump"
    assert longest[-1].effect == "hospital"


def test_backward_chain_root_cause_order():
    g = _graph()
    g.add_edges(CHAIN)
    chains = g.backward_chain("hospital", max_depth=6)
    longest = max(chains, key=len)
    # backward chain returned in causal order: starts at the root cause
    assert longest[0].cause == "pump"
    assert longest[-1].effect == "hospital"


def test_path_between():
    g = _graph()
    g.add_edges(CHAIN)
    path = g.path_between("pump", "hospital", max_depth=6)
    assert path is not None and path[0].cause == "pump" and path[-1].effect == "hospital"
    assert g.path_between("hospital", "pump") is None        # wrong direction
    assert g.path_between("pump", "nonexistent") is None


def test_score_edges_by_triple_smoke():
    from causal_graph_rag.vsa_core import Triple
    g = _graph()
    g.add_edges(CHAIN)
    scored = g.score_edges_by_triple(Triple("pump", "cause", "overheat"), top_k=2)
    assert scored and isinstance(scored[0][0], float)


def test_reachable_graphlookup_native():
    """Native MongoDB $graphLookup reachability with correct hop depths."""
    g = _graph()
    g.add_edges(CHAIN)
    fwd = g.reachable("pump", "forward")          # downstream impact set
    assert fwd == {"overheat": 1, "scram": 2, "outage": 3, "hospital": 4}
    bwd = g.reachable("hospital", "backward")     # upstream root-cause set
    assert bwd is not None and "pump" in bwd and bwd["outage"] == 1


def test_cache_invalidated_on_add():
    g = _graph()
    g.add_edges(CHAIN[:2])
    assert len(list(g.edges)) == 2
    g.add_edge(CHAIN[2])
    assert len(list(g.edges)) == 3          # cache rebuilt after mutation


def test_satisfies_graphbackend_contract():
    from causal_graph_rag.graph_backend import GraphBackend
    g = _graph()
    assert isinstance(g, GraphBackend)
    for m in ("add_edge", "nodes", "edges", "word_index", "score_edges_by_triple",
              "forward_chain", "backward_chain", "path_between"):
        assert hasattr(g, m)
