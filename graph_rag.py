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
from causal_extractor import extract_edges, extract_edges_hybrid
from causal_graph import CausalGraph, GraphEdge
from retrievers import BM25, HashingDense, make_dense, PathSignatureRetriever, rrf_fuse, tokenize
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
    def __init__(
        self,
        dim: int = 10000,
        semantic_weight: int = 0,
        llm: Optional[object] = None,
        max_depth: int = 6,
        neo4j_uri: Optional[str] = None,
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
    ):
        """
        Initialize GraphRAG with in-memory or Neo4j backend.

        Parameters
        ----------
        dim : int
            VSA hypervector dimension
        semantic_weight : int
            Weight for semantic similarity in VSA
        llm : optional object
            LLM for generation (defaults to MockLLM)
        max_depth : int
            Max chain depth for traversal
        neo4j_uri : optional str
            If provided, use Neo4j backend (e.g., "neo4j://localhost:7687")
        neo4j_user : str
            Neo4j username
        neo4j_password : str
            Neo4j password
        neo4j_database : str
            Neo4j database name
        """
        self.lex = Lexicon(dim=dim, semantic_weight=semantic_weight)

        # Choose backend: Neo4j or in-memory
        if neo4j_uri:
            try:
                from neo4j_graph import Neo4jCausalGraph

                self.graph = Neo4jCausalGraph(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                    database=neo4j_database,
                    lex=self.lex,
                )
                self.using_neo4j = True
            except ImportError:
                raise ImportError(
                    "neo4j package required for Neo4j backend. "
                    "Install with: pip install neo4j>=5.0"
                )
        else:
            self.graph = CausalGraph(self.lex)
            self.using_neo4j = False

        self.bm25 = BM25()
        self.dense = make_dense()
        self.sig = PathSignatureRetriever()
        self.llm = llm or MockLLM()
        self.max_depth = max_depth
        self._node_docs: Dict[str, str] = {}
        # Document-structure index (Phase 2): each ingested sentence's location
        # in the document — heading path, reading position, synthesis score and
        # (optional) discourse role. Used to annotate evidence with WHERE it
        # came from, the domain-agnostic "contextual retrieval" signal.
        self._struct_index: List[Tuple[set, dict, str]] = []

    # -- ingest -------------------------------------------------------------- #
    def ingest(self, text: str, doc_id: str = "doc",
               llm_extractor: Optional[object] = None,
               llm_mode: str = "augment", schema: str = "general") -> int:
        """
        Extract causal edges and build the retrieval indices.

        Parameters
        ----------
        text          : Document text to ingest.
        doc_id        : Identifier (unused internally, kept for API symmetry).
        llm_extractor : Optional LLM (any .generate(str)->str object) to run
                        alongside spaCy.  When supplied, uses hybrid extraction
                        (borrowed from CausalRAG) to catch implicit causality
                        that dependency parsing misses.
        llm_mode      : "augment" — LLM only fills sentences spaCy missed.
                        "full"    — LLM processes every sentence (highest recall,
                        same cost profile as CausalRAG).
        schema        : Document-structure preset for the ingested text — the
                        user's choice. "general" (default) captures hierarchy +
                        position with no domain assumptions; "research"/
                        "clinical"/"incident"/"auto" additionally tag discourse
                        roles. Structure is captured either way.
        """
        from doc_structure import parse
        ds = parse(text, schema=schema)
        # Extract causality from the clean BODY sentences only, so markdown
        # heading lines ("## Impact") never leak into edges or provenance.
        clean_text = " ".join(s.text for s in ds.sentences()) or text

        if llm_extractor is not None:
            edges = extract_edges_hybrid(clean_text, llm_extractor, mode=llm_mode)
        else:
            edges = extract_edges(clean_text)

        for e in edges:
            self.graph.add_edge(e)
            for node in (e.cause, e.effect):
                self._node_docs.setdefault(node, "")
                if e.source_sent not in self._node_docs[node]:
                    self._node_docs[node] += " " + e.source_sent
        self.bm25.index(self._node_docs)
        self.dense.index(self._node_docs)
        self.sig.index(self._node_docs)
        self._index_document_structure(ds)
        return len(edges)

    # -- document structure index (Phase 2) ---------------------------------- #
    def _embed_fn(self):
        """Return a str->unit-vector encoder if a real dense model is loaded,
        else None (synthesis score falls back to token overlap)."""
        model = getattr(self.dense, "_model", None)
        if model is None:
            return None
        return lambda t: model.encode([t], convert_to_numpy=True,
                                      normalize_embeddings=True)[0]

    def _index_document_structure(self, ds) -> None:
        from doc_structure import _content_words
        syn = ds.synthesis_scores(embed=self._embed_fn())
        for s in ds.sentences():
            sec = ds.section_of(s.block_id)
            meta = {
                "heading_path": ds.heading_path(s.block_id),
                "position": round(ds.position_of(s.block_id), 3),
                "role": ds.role_of(s.block_id),
                "synthesis": syn.get(sec.block_id, 0.0) if sec else 0.0,
            }
            self._struct_index.append((set(_content_words(s.text)), meta, s.text))

    def _locate(self, sentence: str) -> Optional[dict]:
        """Best-matching structural location for a (possibly coref-rewritten)
        sentence, by content-word Jaccard. None if nothing matches well."""
        if not self._struct_index:
            return None
        from doc_structure import _content_words
        q = set(_content_words(sentence))
        if not q:
            return None
        best, best_j = None, 0.0
        for toks, meta, _ in self._struct_index:
            if not toks:
                continue
            j = len(q & toks) / len(q | toks)
            if j > best_j:
                best, best_j = meta, j
        return best if best_j >= 0.4 else None

    def _annotate(self, sentence: str) -> str:
        """Prefix a sentence with its heading path, e.g. '[Results › Ablations] ...'."""
        meta = self._locate(sentence)
        if not meta:
            return sentence
        hp = " > ".join(meta["heading_path"])
        return f"[{hp}] {sentence}" if hp else sentence

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

        # Path signature channel: augment the query with top BM25 context
        # sentences (in document order) so the query forms a multi-point path
        # and levels 2 & 3 of the signature carry genuine sequential structure.
        bm25_context = [self._node_docs.get(node, "")
                        for _, node in bm25_ranked[:3] if node in self._node_docs]
        sig_ranked = self.sig.score(question, bm25_context=bm25_context)

        fused = rrf_fuse([direct, vsa_ranked, bm25_ranked, dense_ranked, sig_ranked],
                         weights=[2.0, 1.2, 1.0, 1.0, 0.8])
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

    # -- causal path summarization (borrowed from CausalRAG) ---------------- #
    def _causal_summary(self, question: str, chains: List[ChainResult]) -> str:
        """
        Compress retrieved causal chains into a single coherent narrative before
        passing to the final generator.

        Inspired by CausalRAG's "causal summary" stage: rather than feeding raw
        chain text + provenance sentences verbatim, a dedicated LLM pass traces
        the key path and compresses it.  This reduces context window usage and
        removes noise from duplicate or peripheral edges.

        Returns a compact causal paragraph (3-5 sentences).
        """
        chain_block = "\n".join(
            f"Chain {i} [{c.direction}]: {c.text()}\n"
            + "\n".join(f"  src: {s}" for s in c.provenance())
            for i, c in enumerate(chains, 1)
        )
        prompt = (
            "You are a causal reasoning assistant.\n"
            "Given the causal chains below, write a single concise paragraph "
            "(3-5 sentences) that summarises the KEY cause-effect relationships "
            "relevant to the question. Preserve causal direction (use words like "
            "'caused', 'led to', 'resulted in'). Do NOT add any information not "
            "present in the chains.\n\n"
            f"Causal chains:\n{chain_block}\n\n"
            f"Question: {question}\n\n"
            "Causal summary:"
        )
        return self.llm.generate(prompt)

    # -- context assembly ---------------------------------------------------- #
    @staticmethod
    def _dedup_provenance(chains: List[ChainResult]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for c in chains:
            for s in c.provenance():
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out

    def _build_context(self, chains: List[ChainResult], structured: bool,
                       contextual: bool = True) -> str:
        """Assemble the LLM context from retrieved chains.

        Two independent structure signals, each separately toggleable so their
        contributions can be measured in isolation:

        structured : prepend the causal CHAIN PATHS with direction + polarity
                     arrows ('->' promotes, '-/->' inhibits). Carries the graph
                     topology flat sentences cannot.
        contextual : annotate each evidence sentence with its document location,
                     e.g. '[Results > Ablations] ...' — the "where did this come
                     from" signal (contextual retrieval). No-op when no document
                     structure was indexed.

        Both False  -> flat evidence sentences (legacy baseline).
        """
        prov = self._dedup_provenance(chains)
        evidence = "\n".join((self._annotate(s) if contextual else s) for s in prov)
        if not structured:
            return evidence
        chain_block = "\n".join(f"  {c.text()}" for c in chains)
        return f"Causal chains:\n{chain_block}\n\nEvidence:\n{evidence}"

    # -- generate ------------------------------------------------------------ #
    def answer(self, question: str, top_k: int = 3,
               summarize: bool = False, structured: bool = True,
               contextual: bool = True) -> Tuple[str, List[ChainResult]]:
        """
        Retrieve causal chains and generate an answer.

        Parameters
        ----------
        question   : Natural-language query.
        top_k      : Number of chains to retrieve.
        summarize  : When True, runs a dedicated causal-summary compression step
                     before the final generation (borrowed from CausalRAG).
                     Costs one extra LLM call but produces tighter, more coherent
                     answers on multi-hop queries.
        structured : When True (default), the prompt includes the causal chain
                     paths with direction + polarity arrows, not just the flat
                     provenance sentences — so the model can reason over the
                     graph structure (ordering, promote vs inhibit). Set False
                     for the legacy sentence-only context (used for A/B tests).
        """
        chains = self.retrieve(question, top_k=top_k)
        if not chains:
            return ("No causal structure matching the query was found.", [])

        if summarize:
            # Two-step generation: compress first, then answer
            causal_ctx = self._causal_summary(question, chains)
            prompt = (
                "You are a causal reasoning assistant. "
                "Answer the question using ONLY the causal evidence provided. "
                "Be direct and concise. Do not reference chain numbers or labels.\n\n"
                f"Causal evidence:\n{causal_ctx}\n\nQuestion: {question}\n\nAnswer:"
            )
        else:
            causal_ctx = self._build_context(chains, structured=structured,
                                             contextual=contextual)
            legend = (
                "In the causal chains, '->' means the cause promotes/produces "
                "the effect and '-/->' means it inhibits/reduces the effect; "
                "respect this direction and polarity.\n"
                if structured else ""
            )
            prompt = (
                "You are a causal reasoning assistant. "
                "Answer the question using ONLY the evidence provided. "
                f"{legend}"
                "Be direct and concise. Do not reference chain numbers or labels.\n\n"
                f"{causal_ctx}\n\nQuestion: {question}\n\nAnswer:"
            )

        return self.llm.generate(prompt), chains

    def close(self) -> None:
        """Close database connections (for Neo4j backend). Idempotent."""
        if getattr(self, "using_neo4j", False):
            graph = getattr(self, "graph", None)
            if graph is not None and hasattr(graph, "close"):
                graph.close()

    def __del__(self):
        # Guard against partial __init__ and interpreter-shutdown teardown:
        # GC must never raise.
        try:
            self.close()
        except Exception:
            pass
