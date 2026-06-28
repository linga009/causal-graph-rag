"""Phase 1: causal-structure-to-LLM context assembly."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph_rag.graph_rag import GraphRAG

DOC = (
    "The reactor overheated. The overheating caused the coolant valve to fail. "
    "The valve failure triggered an emergency shutdown. "
    "The shutdown reduced power output. "
    "Lower power output disrupted hospital operations."
)


def _ingested():
    rag = GraphRAG(dim=10000)          # MockLLM, no API key needed
    rag.ingest(DOC)
    return rag


def test_structured_context_includes_chain_arrows():
    rag = _ingested()
    chains = rag.retrieve("What did the overheating ultimately disrupt?", top_k=3)
    assert chains, "expected at least one chain"
    ctx = rag._build_context(chains, structured=True)
    assert "Causal chains" in ctx
    assert "Evidence:" in ctx
    assert "->" in ctx, "structured context must contain causal arrows"


def test_unstructured_context_is_sentences_only():
    rag = _ingested()
    chains = rag.retrieve("What did the overheating ultimately disrupt?", top_k=3)
    ctx = rag._build_context(chains, structured=False)
    assert "Causal chains" not in ctx
    assert "->" not in ctx
    # still carries the evidence sentences
    assert "reactor" in ctx.lower() or "shutdown" in ctx.lower()


def test_structured_default_in_answer():
    rag = _ingested()
    # answer() defaults to structured=True and must return chains + a string
    ans, chains = rag.answer("What did the overheating ultimately disrupt?")
    assert isinstance(ans, str) and ans
    assert chains


def test_provenance_dedup():
    rag = _ingested()
    chains = rag.retrieve("Why did hospital operations get disrupted?", top_k=3)
    prov = GraphRAG._dedup_provenance(chains)
    assert len(prov) == len(set(prov)), "provenance should be de-duplicated"
