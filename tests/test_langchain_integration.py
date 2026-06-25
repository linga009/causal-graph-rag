"""
Tests for langchain_integration.py.

Uses a tiny in-memory GraphRAG so no API keys are needed.
langchain-core must be installed (it's in the dev extras).
"""
import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from graph_rag import GraphRAG
from langchain_integration import (
    VSAGraphRetriever,
    LangChainLLMAdapter,
    _find_node,
    build_graph_tools,
)

_TEXT = (
    "The pump failed. This caused the reactor to overheat. "
    "The overheating triggered an emergency scram. "
    "The scram led to a 12-hour outage. "
    "The outage disrupted hospital operations."
)


@pytest.fixture(scope="module")
def rag():
    r = GraphRAG()
    r.ingest(_TEXT)
    return r


# --- _find_node --------------------------------------------------------------

def test_find_node_exact_token(rag):
    node = _find_node(rag, "pump")
    assert node is not None
    assert "pump" in node.lower()


def test_find_node_multi_token(rag):
    node = _find_node(rag, "reactor overheat")
    assert node is not None


def test_find_node_no_match(rag):
    node = _find_node(rag, "xyzzy quux")
    assert node is None


# --- VSAGraphRetriever — chains mode (default) -------------------------------

def test_retriever_chains_returns_documents(rag):
    r = VSAGraphRetriever(graph_rag=rag, top_k=2)
    docs = r.invoke("What caused the scram?")
    assert isinstance(docs, list)
    for doc in docs:
        assert doc.page_content.startswith("Chain:")
        assert doc.metadata["type"] == "chain"


def test_retriever_chain_metadata_fields(rag):
    r = VSAGraphRetriever(graph_rag=rag, top_k=2)
    docs = r.invoke("What caused the outage?")
    if docs:
        m = docs[0].metadata
        assert "hop_count" in m
        assert "chain_confidence" in m
        assert "chain_polarity" in m
        assert m["chain_polarity"] in (1, -1)
        assert 0.0 <= m["chain_confidence"] <= 1.0
        assert m["hop_count"] >= 1


# --- VSAGraphRetriever — coverage mode ---------------------------------------

def test_retriever_coverage_mode(rag):
    r = VSAGraphRetriever(graph_rag=rag, top_k=3, mode="coverage")
    docs = r.invoke("What disrupted hospital operations?")
    for doc in docs:
        assert doc.metadata["type"] == "coverage"
        assert "Chain:" not in doc.page_content


# --- VSAGraphRetriever — hybrid mode -----------------------------------------

def test_retriever_hybrid_mode_has_both_types(rag):
    r = VSAGraphRetriever(graph_rag=rag, top_k=3, mode="hybrid")
    docs = r.invoke("What caused the scram?")
    types = {doc.metadata["type"] for doc in docs}
    assert "chain" in types
    assert "coverage" in types


def test_retriever_hybrid_mode_chain_docs_first(rag):
    r = VSAGraphRetriever(graph_rag=rag, top_k=2, mode="hybrid")
    docs = r.invoke("What caused the scram?")
    chain_docs = [d for d in docs if d.metadata["type"] == "chain"]
    coverage_docs = [d for d in docs if d.metadata["type"] == "coverage"]
    assert len(chain_docs) <= 2
    assert len(coverage_docs) > 0


# --- build_graph_tools -------------------------------------------------------

def test_graph_tools_returns_three(rag):
    tools = build_graph_tools(rag)
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"causal_rootcause", "causal_impact", "causal_path"}


def test_rootcause_tool(rag):
    tools = {t.name: t for t in build_graph_tools(rag)}
    result = tools["causal_rootcause"].invoke({"effect": "scram"})
    assert isinstance(result, str)
    assert len(result) > 10


def test_impact_tool(rag):
    tools = {t.name: t for t in build_graph_tools(rag)}
    result = tools["causal_impact"].invoke({"cause": "pump"})
    assert isinstance(result, str)
    assert len(result) > 10


def test_path_tool_finds_connection(rag):
    tools = {t.name: t for t in build_graph_tools(rag)}
    result = tools["causal_path"].invoke({"source": "pump", "target": "outage"})
    assert isinstance(result, str)
    # Should either find a path or explain it wasn't found
    assert "pump" in result.lower() or "no graph node" in result.lower()


def test_path_tool_unknown_node(rag):
    tools = {t.name: t for t in build_graph_tools(rag)}
    result = tools["causal_path"].invoke({"source": "xyzzy", "target": "quux"})
    assert "No graph node found" in result


def test_rootcause_tool_unknown_node(rag):
    tools = {t.name: t for t in build_graph_tools(rag)}
    result = tools["causal_rootcause"].invoke({"effect": "xyzzy quux blorb"})
    assert "No graph node found" in result
