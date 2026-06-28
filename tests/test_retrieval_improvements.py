"""Retrieval-side structure: contextual indexing, MMR diversity, prose chains."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph_rag.graph_rag import GraphRAG, ChainResult
from causal_graph_rag.causal_graph import GraphEdge
import numpy as np

DOC = """# Subprime Crisis

## Housing Market
Lax lending standards fueled risky mortgages. The risky mortgages inflated a
housing bubble. The housing bubble eventually burst.

## Financial Markets
Securitization spread the mortgage risk to investors. The spread risk triggered
bank losses. The bank losses froze interbank lending.
"""


def test_contextual_indexing_populates_node_context():
    rag = GraphRAG(dim=10000)
    rag.ingest(DOC, schema="general")
    # at least some nodes carry their section heading as context
    ctx_words = " ".join(" ".join(v) for v in rag._node_context.values()).lower()
    assert "housing market" in ctx_words or "financial markets" in ctx_words


def test_contextual_index_matches_section_topic():
    rag = GraphRAG(dim=10000)
    rag.ingest(DOC, schema="general")
    rag._ensure_indexed()   # indices build lazily; flush before poking BM25
    # a query using a section topic word should surface that section's nodes,
    # even though the sentence text may not repeat the heading verbatim
    scored = rag.bm25.score("financial markets")
    assert scored, "contextual BM25 should match the section heading words"


def _edge(cause, effect, eid):
    return GraphEdge(cause, "cause", effect, 1, f"{cause} {effect}",
                     np.zeros(4, dtype=np.int8), eid)


def _chain(nodes, score):
    edges = [_edge(nodes[i], nodes[i + 1], i) for i in range(len(nodes) - 1)]
    c = ChainResult(edges, nodes[0], 0.0, score, "forward")
    return c


def test_mmr_prefers_diverse_chains():
    rag = GraphRAG(dim=10000)
    # c1 is the top chain. c2 (near-duplicate of c1) and c3 (disjoint) have
    # EQUAL relevance, so the only thing separating them is redundancy — MMR
    # must break the tie toward the diverse, disjoint chain.
    c1 = _chain(["a", "b", "c"], 1.0)
    c2 = _chain(["a", "b", "d"], 0.9)   # shares a,b with c1
    c3 = _chain(["x", "y", "z"], 0.9)   # disjoint from c1
    picked = rag._mmr_select([c1, c2, c3], top_k=2, lam=0.5)
    names = {id(c) for c in picked}
    assert id(c1) in names and id(c3) in names, "MMR should add the disjoint chain"


def test_mmr_noop_when_few_candidates():
    rag = GraphRAG(dim=10000)
    c1 = _chain(["a", "b"], 1.0)
    assert rag._mmr_select([c1], top_k=3) == [c1]


def test_prose_chain_has_no_arrows():
    c = _chain(["overheating", "valve_failure", "shutdown"], 1.0)
    prose = GraphRAG._chain_prose(c)
    assert "->" not in prose and "-/->" not in prose
    assert "overheating" in prose and "shutdown" in prose
    assert "which" in prose  # multi-hop connector
