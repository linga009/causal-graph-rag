"""Causal Graph RAG — retrieval that traverses cause→effect chains.

Public API:
    from causal_graph_rag import GraphRAG, AgenticCausalRAG
"""
from .graph_rag import GraphRAG, ChainResult
from .agentic_rag import AgenticCausalRAG

__version__ = "0.3.0"
__all__ = ["GraphRAG", "ChainResult", "AgenticCausalRAG", "__version__"]
