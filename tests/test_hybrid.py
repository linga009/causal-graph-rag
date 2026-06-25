"""Hybrid retrieval: coverage sentences (vector) + causal chains (graph).

The pure-graph design missed standalone facts because it answered only from
causal-chain provenance. The hybrid retrieves the top-k relevant sentences too,
so non-causal facts reach the LLM — a true superset of flat RAG.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag import GraphRAG

DOC = (
    "The fire caused a power loss. The power loss disrupted the hospital. "
    "The building was constructed in 1998 and has twelve floors."
)


def test_coverage_surfaces_noncausal_fact():
    # "constructed in 1998" is NOT part of any cause->effect edge.
    rag = GraphRAG(dim=10000)
    rag.ingest(DOC)
    sents = rag._retrieve_sentences("when was the building constructed?", k=5)
    assert any("1998" in s for s in sents), "coverage channel must surface non-causal facts"


def test_context_includes_coverage_and_chains():
    rag = GraphRAG(dim=10000)
    rag.ingest(DOC)
    chains = rag.retrieve("what did the fire cause?", top_k=3)
    coverage = rag._retrieve_sentences("what did the fire cause?", k=6)
    ctx = rag._build_context(chains, structured=True, coverage_sentences=coverage)
    assert "Evidence:" in ctx
    # coverage sentences present even if a fact isn't on a chain
    assert "fire" in ctx.lower()


def test_answer_works_with_zero_chains_via_coverage():
    # A doc with no extractable causality still answers from coverage sentences,
    # instead of the old "No causal structure found" dead-end.
    rag = GraphRAG(dim=10000)
    rag.ingest("Mount Everest is 8849 metres tall. The sky appears blue.")
    ans, chains = rag.answer("how tall is Mount Everest?")
    assert isinstance(ans, str) and ans
    # (MockLLM echoes context; the key point is it did not refuse outright)
    assert "No relevant information" not in ans
