"""Persistence (save/load) and lazy indexing."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph_rag.graph_rag import GraphRAG

DOC = """# Outage Report

## Timeline
The reactor overheated. The overheating caused the coolant valve to fail.
The valve failure triggered an emergency shutdown.

## Impact
The shutdown reduced power output. Lower power output disrupted hospital operations.
"""


def test_lazy_indexing_marks_dirty_then_builds_on_retrieve():
    rag = GraphRAG(dim=10000)
    rag.ingest(DOC, schema="general")
    assert rag._dirty is True, "ingest should defer indexing"
    chains = rag.retrieve("What did the overheating cause?", top_k=3)
    assert rag._dirty is False, "retrieve should build indices"
    assert chains


def test_repeated_ingest_single_index_build():
    rag = GraphRAG(dim=10000)
    rag.ingest("The fire caused a power loss.", schema="general")
    rag.ingest("The power loss disrupted the hospital.", schema="general")
    assert rag._dirty is True            # still deferred after 2 ingests
    rag.retrieve("What did the fire cause?")
    assert rag._dirty is False


def test_save_load_roundtrip_preserves_answers():
    rag = GraphRAG(dim=10000)
    rag.ingest(DOC, schema="general")
    before = [c.text() for c in rag.retrieve("What did the overheating cause?", top_k=3)]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.pkl")
        rag.save(path)
        assert os.path.getsize(path) > 0

        loaded = GraphRAG.load(path)
        # indices rebuild lazily on first retrieve
        assert loaded._dirty is True
        after = [c.text() for c in loaded.retrieve("What did the overheating cause?", top_k=3)]

    assert before == after, "loaded graph must answer identically"


def test_load_preserves_graph_structure():
    rag = GraphRAG(dim=10000)
    rag.ingest(DOC, schema="general")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.pkl")
        rag.save(path)
        loaded = GraphRAG.load(path)
    assert loaded.graph.nodes() == rag.graph.nodes()
    assert len(loaded.graph.edges) == len(rag.graph.edges)
    # edge ids stay distinct after reconstruction
    ids = [e.edge_id for e in loaded.graph.edges]
    assert len(set(ids)) == len(ids)


def test_locate_exact_match_fast_path():
    rag = GraphRAG(dim=10000)
    rag.ingest(DOC, schema="general")
    # a verbatim ingested sentence hits the O(1) exact map
    meta = rag._locate("The shutdown reduced power output.")
    assert meta is not None
    assert meta["heading_path"] == ["Outage Report", "Impact"]
