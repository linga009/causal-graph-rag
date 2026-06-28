"""Robustness: degenerate inputs must not crash (real docs sometimes have no
extractable causality, and CI runs without spaCy = rule-based extraction)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph_rag.graph_rag import GraphRAG


def test_zero_edge_document_does_not_crash():
    # A document with no extractable causal relation -> empty graph. Retrieval
    # must degrade gracefully (empty indices), not raise on np.stack([]).
    rag = GraphRAG(dim=10000)
    rag.ingest("Blue. Green. Round things and tall things.")
    chains = rag.retrieve("what caused it?", top_k=3)
    assert chains == []
    answer, chains2 = rag.answer("what caused it?")
    assert isinstance(answer, str) and chains2 == []


def test_empty_string_ingest_does_not_crash():
    rag = GraphRAG(dim=10000)
    n = rag.ingest("")
    assert n == 0
    assert rag.retrieve("anything", top_k=3) == []


def test_query_before_any_ingest():
    rag = GraphRAG(dim=10000)
    assert rag.retrieve("what caused the outage?", top_k=3) == []
