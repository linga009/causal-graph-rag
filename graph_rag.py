"""
graph_rag.py
============
Consequential-Graph VSA-RAG engine.

Pipeline:
  INGEST    extract causal edges -> build VSA-encoded graph + index BM25/dense
  RETRIEVE  three channels pick entry NODES -> RRF fuse
  TRAVERSE  walk causal chains from fused entry nodes (the structure recovery)
  RERANK    score whole chains for query relevance (lexical-overlap stand-in
            for a cross-encoder; swap in Cohere/Voyage rerank in production)
  GENERATE  feed ordered causal chains as context to the LLM

This directly targets the failure mode where chunking + embedding lose the
consequential structure between chunks: chains are first-class retrieval units.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np

from vsa_core import Lexicon, Triple
from causal_extractor import extract_edges
from causal_graph import CausalGraph, GraphEdge
from retrievers import BM25, HashingDense, make_dense, rrf_fuse, tokenize
from parser import parse_triples
from pipeline import MockLLM


@dataclass
class ChainResult:
    chain: List[GraphEdge]
    entry_node: str
    rrf_score: float
    rerank_score: float
    direction: str            # "forward" | "backward" | "path"

    def text(self) -> str:
        return CausalGraph.chain_text(self.chain)

    def provenance(self) -> List[str]:
        seen, out = set(), []
        for e in self.chain:
            if e.source_sent not in seen:
                seen.add(e.source_sent)
                out.append(e.source_sent)
        return out


# intent detection for query direction
_FORWARD_Q = ["lead to", "result in", "ultimately", "consequence",
              "effect of", "happens when", "what does", "downstream",
              "what will", "what happens"]
_BACKWARD_Q = ["why", "root cause", "caused by", "because", "reason for",
               "what led to", "due to", "upstream", "origin of",
               "what caused", "what causes", "what triggered", "what triggers",
               "how did", "cause of"]


class GraphRAG:
    def __init__(self, dim: int = 10000, semantic_weight: int = 0,
                 llm: Optional[object] = None, max_depth: int = 6):
        self.lex = Lexicon(dim=dim, semantic_weight=semantic_weight)
        self.graph = CausalGraph(self.lex)
        self.bm25 = BM25()
        self.dense = make_dense()
        self.llm = llm or MockLLM()
        self.max_depth = max_depth
        self._node_docs: Dict[str, str] = {}

    # -- ingest -------------------------------------------------------------- #
    def ingest(self, text: str, doc_id: str = "doc") -> int:
        edges = extract_edges(text)
        for e in edges:
            self.graph.add_edge(e)
            # accumulate the sentences each node appears in (for BM25/dense)
            for node in (e.cause, e.effect):
                self._node_docs.setdefault(node, "")
                if e.source_sent not in self._node_docs[node]:
                    self._node_docs[node] += " " + e.source_sent
        self.bm25.index(self._node_docs)
        self.dense.index(self._node_docs)
        return len(edges)

    # -- entry-node selection (three-channel fusion) ------------------------- #
    def _entry_nodes(self, question: str, top_n: int = 4) -> List[Tuple[float, str]]:
        from retrievers import tokenize
        q_terms = set(tokenize(question))

        # Channel 0: direct node-name match. The query usually names the event
        # it is about ("what did budget CUTS lead to" -> node 'cuts'). This is
        # the strongest, most literal signal and anchors traversal correctly.
        direct: List[Tuple[float, str]] = []
        for node in self.graph.nodes():
            nt = set(tokenize(node))
            if nt & q_terms:
                direct.append((1.0, node))

        # VSA channel: structural/directional entry points.
        vsa_ranked: List[Tuple[float, str]] = []
        for qt in parse_triples(question):
            for score, edge in self.graph.score_edges_by_triple(qt, top_k=top_n):
                vsa_ranked.append((score, edge.cause))
                vsa_ranked.append((score * 0.9, edge.effect))
        vsa_ranked.sort(key=lambda x: -x[0])

        bm25_ranked = self.bm25.score(question)
        dense_ranked = self.dense.score(question)

        fused = rrf_fuse([direct, vsa_ranked, bm25_ranked, dense_ranked],
                         weights=[2.0, 1.2, 1.0, 1.0])   # direct match dominates
        return fused[:top_n]

    # -- query direction ----------------------------------------------------- #
    def _direction(self, question: str) -> str:
        low = question.lower()
        if any(p in low for p in _BACKWARD_Q):
            return "backward"
        return "forward"

    # -- rerank chains ------------------------------------------------------- #
    def _rerank(self, question: str, chains: List[ChainResult]) -> None:
        q_terms = set(tokenize(question))

        # Semantic similarity: encode query once and compare against the
        # average of the per-node embeddings already stored in the dense index.
        # Falls back to zero when HashingDense is active (no _model attribute).
        q_emb = None
        if hasattr(self.dense, '_model') and getattr(self.dense, 'vecs', None):
            q_emb = self.dense._model.encode(
                [question], convert_to_numpy=True, normalize_embeddings=True)[0]

        for c in chains:
            chain_terms = set()
            for e in c.chain:
                chain_terms |= set(tokenize(f"{e.cause} {e.relation} {e.effect}"))
            overlap = len(q_terms & chain_terms)
            score = overlap + 0.25 * len(c.chain)
            # direction-aware bonus: a backward (why/root-cause) query wants the
            # chain to TERMINATE at the queried event; a forward query wants it
            # to ORIGINATE there. Reward the matching endpoint.
            if c.chain:
                head = set(tokenize(c.chain[0].cause))
                tail = set(tokenize(c.chain[-1].effect))
                if c.direction == "backward" and (tail & q_terms):
                    score += 2.0
                if c.direction == "forward" and (head & q_terms):
                    score += 2.0
            # Semantic similarity via dense encoder (3× weight so it is
            # competitive with the lexical overlap score on short chains)
            if q_emb is not None:
                node_embs = [
                    self.dense.vecs[n]
                    for e in c.chain
                    for n in (e.cause, e.effect)
                    if n in self.dense.vecs
                ]
                if node_embs:
                    chain_emb = np.mean(node_embs, axis=0)
                    norm = np.linalg.norm(chain_emb)
                    if norm:
                        chain_emb /= norm
                    score += float(q_emb @ chain_emb) * 3.0
            c.rerank_score = score

    # -- retrieve ------------------------------------------------------------ #
    def retrieve(self, question: str, top_k: int = 3) -> List[ChainResult]:
        direction = self._direction(question)
        entries = self._entry_nodes(question, top_n=4)
        results: List[ChainResult] = []
        for rrf_score, node in entries:
            if node not in self.graph.nodes():
                continue
            if direction == "backward":
                paths = self.graph.backward_chain(node, self.max_depth)
            else:
                paths = self.graph.forward_chain(node, self.max_depth)
            for path in paths:
                if path:
                    results.append(ChainResult(path, node, rrf_score, 0.0, direction))
        # dedupe identical chains
        seen, uniq = set(), []
        for r in results:
            key = tuple((e.cause, e.relation, e.effect) for e in r.chain)
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        self._rerank(question, uniq)
        uniq.sort(key=lambda r: (-r.rerank_score, -r.rrf_score))
        return uniq[:top_k]

    # -- generate ------------------------------------------------------------ #
    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[ChainResult]]:
        chains = self.retrieve(question, top_k=top_k)
        if not chains:
            return ("No causal structure matching the query was found.", [])
        ctx_lines = []
        for i, c in enumerate(chains, 1):
            ctx_lines.append(f"Causal chain {i}: {c.text()}")
            for s in c.provenance():
                ctx_lines.append(f"   source: {s}")
        context = "\n".join(ctx_lines)
        prompt = (
            "Using the following RETRIEVED CAUSAL CHAINS as context, answer the "
            "question. Each chain shows cause->effect direction explicitly; "
            "respect that direction.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n"
        )
        return self.llm.generate(prompt), chains
