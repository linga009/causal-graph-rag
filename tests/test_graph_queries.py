"""Direct causal graph queries: root_causes (backward), impact (forward),
connect (path). Tested on a hand-built, cleanly-connected graph so the query
logic is verified independent of extraction quality."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag import GraphRAG
from causal_extractor import CausalEdge


def _chain_graph():
    """fire -> smoke -> alarm -> evacuation (a clean linear cascade)."""
    rag = GraphRAG(dim=10000)
    for c, r, e in [("fire", "produce", "smoke"),
                    ("smoke", "trigger", "alarm"),
                    ("alarm", "cause", "evacuation")]:
        rag.graph.add_edge(CausalEdge(c, r, e, 1, f"{c} {r} {e}"))
    return rag


def test_root_causes_backward():
    rag = _chain_graph()
    node, chains = rag.root_causes("evacuation")
    assert node == "evacuation"
    text = " ".join(c.text() for c in chains)
    assert "fire" in text and "smoke" in text and "alarm" in text  # full chain back


def test_impact_forward():
    rag = _chain_graph()
    node, chains = rag.impact("fire")
    assert node == "fire"
    text = " ".join(c.text() for c in chains)
    assert "evacuation" in text  # reaches the distal effect


def test_connect_path():
    rag = _chain_graph()
    s, d, chain = rag.connect("fire", "evacuation")
    assert s == "fire" and d == "evacuation" and chain is not None
    assert "smoke" in chain.text() and "alarm" in chain.text()


def test_fuzzy_node_resolution():
    rag = _chain_graph()
    # partial term still resolves via token overlap
    node, chains = rag.impact("the fire")
    assert node == "fire" and chains


def test_unknown_event_returns_empty():
    rag = _chain_graph()
    node, chains = rag.root_causes("earthquake")
    assert node is None and chains == []


def test_no_path_returns_none():
    rag = _chain_graph()
    # evacuation has no forward edge to fire
    s, d, chain = rag.connect("evacuation", "fire")
    assert chain is None
