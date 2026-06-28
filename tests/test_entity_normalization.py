"""Entity normalization: variant surface forms collapse to one canonical node
so causal chains connect instead of fragmenting."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph_rag.graph_rag import GraphRAG, _canon_entity, _entity_tokens
from causal_graph_rag.causal_extractor import CausalEdge


def test_canon_strips_article_and_case():
    assert _canon_entity("The Cooling Pump") == "cooling pump"
    assert _canon_entity("  a  Reactor's ") == "reactor"
    assert _canon_entity("Overheating.") == "overheating"


def test_entity_tokens_stem():
    # 'overheating' stems to 'overheat'; subset relation holds with 'reactor overheat'
    assert _entity_tokens("overheating") < _entity_tokens("reactor overheating")


def test_subset_merge_into_most_specific():
    rag = GraphRAG(dim=10000, normalize_entities=True)
    edges = [
        CausalEdge("the cooling pump", "degrade", "reactor core", 1, "s1"),
        CausalEdge("pump", "cause", "overheating", 1, "s2"),          # pump -> cooling pump
        CausalEdge("reactor overheating", "trigger", "shutdown", 1, "s3"),  # overheating -> reactor overheating
    ]
    out = rag._normalize_edges(edges)
    causes = {e.cause for e in out}
    effects = {e.effect for e in out}
    nodes = causes | effects
    assert "cooling pump" in nodes and "pump" not in nodes      # merged
    assert "reactor overheating" in nodes and "overheating" not in nodes


def test_does_not_merge_distinct_entities():
    rag = GraphRAG(dim=10000, normalize_entities=True)
    edges = [
        CausalEdge("power output", "drop", "grid", 1, "s1"),
        CausalEdge("power loss", "cause", "blackout", 1, "s2"),
    ]
    out = rag._normalize_edges(edges)
    nodes = {n for e in out for n in (e.cause, e.effect)}
    # 'power output' and 'power loss' share 'power' but neither is a subset
    assert "power output" in nodes and "power loss" in nodes


def test_normalization_can_be_disabled():
    rag = GraphRAG(dim=10000, normalize_entities=False)
    edges = [CausalEdge("the pump", "x", "reactor", 1, "s")]
    out = rag._normalize_edges(edges)
    # lexical canon still applies (article stripped) but no subset merging
    assert out[0].cause == "pump"


def test_self_loops_dropped_after_merge():
    rag = GraphRAG(dim=10000, normalize_entities=True)
    edges = [CausalEdge("overheating", "cause", "reactor overheating", 1, "s")]
    out = rag._normalize_edges(edges)
    # both collapse to 'reactor overheating' -> self-loop -> dropped
    assert out == []
