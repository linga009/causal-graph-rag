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
import logging
import pickle
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

log = logging.getLogger("causal_rag")

# On-disk format version for save()/load(). Bump on any incompatible change.
_SAVE_VERSION = 1


def _norm_sent(s: str) -> str:
    """Normalized key for fast exact sentence lookup (whitespace/case-insensitive)."""
    return " ".join(s.lower().split())


# --- Entity normalization (canonical node names so chains don't fragment) ---- #
_ENT_ARTICLES = ("the ", "a ", "an ", "its ", "their ", "this ", "that ")
_ENT_STOP = frozenset("the a an of to in on at and or for by with from".split())


def _canon_entity(name: str) -> str:
    """Lexical canonical form: lowercase, collapse whitespace, strip a leading
    article and a trailing possessive. Merges 'The Cooling Pump' -> 'cooling pump'."""
    n = " ".join(name.lower().split()).strip(" .,:;\"'`()")
    for art in _ENT_ARTICLES:
        if n.startswith(art):
            n = n[len(art):]
            break
    if n.endswith("'s"):
        n = n[:-2]
    return n.strip()


def _stem_tok(t: str) -> str:
    """Light, consistent stem: drop a trailing 'ing' or plural 's' (not 'ss')."""
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def _entity_tokens(name: str) -> frozenset:
    return frozenset(_stem_tok(t) for t in re.findall(r"[a-z0-9]+", name.lower())
                     if t not in _ENT_STOP)


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
        normalize_entities: bool = True,
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
        self.normalize_entities = normalize_entities
        self._node_docs: Dict[str, str] = {}
        # Per-node document context (heading paths the node's sentences sit
        # under). Folded into the BM25/dense index so retrieval can match on
        # section-level topic words — Anthropic "Contextual Retrieval".
        self._node_context: Dict[str, set] = {}
        # Document-structure index (Phase 2): each ingested sentence's location
        # in the document — heading path, reading position, synthesis score and
        # (optional) discourse role. Used to annotate evidence with WHERE it
        # came from, the domain-agnostic "contextual retrieval" signal.
        self._struct_index: List[Tuple[set, dict, str]] = []
        # Fast exact-match lookup (normalized sentence -> meta) so _locate is
        # O(1) on the common path; the Jaccard scan is only a fallback for
        # coref-rewritten sentences.
        self._sent_meta: Dict[str, dict] = {}
        # Separated index/query pipelines: ingest() only accumulates and marks
        # the indices dirty; they are (re)built lazily on the next retrieve().
        # Keeps repeated ingest() O(total) instead of O(docs x nodes).
        self._dirty = False
        # Sentence-level coverage index (HYBRID retrieval). The graph alone only
        # surfaces causally-connected content; standalone facts fall through.
        # We ALSO retrieve the top-k most relevant raw sentences (like vector RAG)
        # and feed them alongside the causal chains — structure is additive, not a
        # replacement for coverage (MS GraphRAG local search; NVIDIA/BlackRock
        # HybridRAG: graph+vector beats either alone).
        self._sentences: List[str] = []
        self._sent_vecs = None   # np.ndarray (N, D) once indexed

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

        # Canonicalize node names so the same event ("pump"/"cooling pump") is one
        # node — longer connected chains instead of fragments.
        edges = self._normalize_edges(edges)

        # Build the structure index FIRST so node docs can be enriched with the
        # document context (heading path) each sentence came from.
        self._index_document_structure(ds)
        # Accumulate ALL sentences (not just causal ones) for coverage retrieval.
        self._sentences.extend(s.text for s in ds.sentences())

        for e in edges:
            self.graph.add_edge(e)
            for node in (e.cause, e.effect):
                self._node_docs.setdefault(node, "")
                if e.source_sent not in self._node_docs[node]:
                    self._node_docs[node] += " " + e.source_sent
                    meta = self._locate(e.source_sent)
                    if meta and meta["heading_path"]:
                        self._node_context.setdefault(node, set()).add(
                            " ".join(meta["heading_path"]))

        # Defer index building to the next retrieve() so repeated ingest() calls
        # don't each pay a full re-index (separated index/query pipelines).
        self._dirty = True
        log.info("ingested %d edges (%d nodes total); indices marked dirty",
                 len(edges), len(self._node_docs))
        return len(edges)

    def _normalize_edges(self, edges):
        """Canonicalize cause/effect node names so variant surface forms of the
        same event become one node. Two conservative, dependency-free passes:
          1. lexical: lowercase, strip articles/possessives.
          2. subset-merge: a name whose stemmed content tokens are a STRICT
             subset of another's is the less-specific variant -> merge it into
             the most-specific (longest) superset. 'pump' -> 'cooling pump',
             'overheat' -> 'reactor overheat'. Cannot merge distinct entities
             like 'power output' / 'power loss' (neither is a subset).
        Self-loops created by a merge are dropped.
        """
        for e in edges:
            e.cause = _canon_entity(e.cause)
            e.effect = _canon_entity(e.effect)
        if self.normalize_entities:
            names = sorted({n for e in edges for n in (e.cause, e.effect)}
                           | set(self.graph.nodes()))
            toks = {n: _entity_tokens(n) for n in names}
            direct = {}
            for a in names:
                if not toks[a]:
                    continue
                supers = [b for b in names if b != a and toks[a] < toks[b]]
                if supers:
                    direct[a] = max(supers, key=len)   # most specific superset

            def resolve(x):
                seen = set()
                while x in direct and x not in seen:
                    seen.add(x)
                    x = direct[x]
                return x

            cmap = {n: resolve(n) for n in names}
            for e in edges:
                e.cause = cmap.get(e.cause, e.cause)
                e.effect = cmap.get(e.effect, e.effect)
        return [e for e in edges if e.cause != e.effect]   # drop self-loops

    def _ensure_indexed(self) -> None:
        """(Re)build BM25/dense/path-signature indices if ingest marked them
        dirty. Called by retrieve(); idempotent and cheap when clean."""
        if not self._dirty:
            return
        # Contextual Retrieval: index context-enriched node docs (heading path
        # folded in) for BM25 + dense; path signatures keep raw sentence order.
        enriched = {
            n: (" ".join(self._node_context.get(n, ())) + " " + d).strip()
            for n, d in self._node_docs.items()
        }
        self.bm25.index(enriched)
        self.dense.index(enriched)
        self.sig.index(self._node_docs)
        # Sentence-level coverage index, reusing the dense model (no 2nd load).
        model = getattr(self.dense, "_model", None)
        if model is not None and self._sentences:
            self._sent_vecs = model.encode(self._sentences, convert_to_numpy=True,
                                           normalize_embeddings=True)
        else:
            self._sent_vecs = None
        self._dirty = False
        log.info("rebuilt retrieval indices over %d nodes, %d sentences",
                 len(self._node_docs), len(self._sentences))

    def _retrieve_sentences(self, question: str, k: int = 6) -> List[str]:
        """Top-k most relevant raw sentences by dense cosine — the coverage
        channel a pure graph misses. Empty if no dense model (hashing fallback)."""
        self._ensure_indexed()
        if self._sent_vecs is None or not self._sentences:
            return []
        model = self.dense._model
        q = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self._sent_vecs @ q
        idx = np.argsort(-sims)[:k]
        return [self._sentences[i] for i in idx]

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
            self._sent_meta[_norm_sent(s.text)] = meta   # O(1) exact-match path

    def _locate(self, sentence: str) -> Optional[dict]:
        """Structural location for a sentence. O(1) exact match on the common
        path; falls back to a content-word Jaccard scan only for coref-rewritten
        sentences that don't match a stored sentence verbatim."""
        if not self._sent_meta:
            return None
        exact = self._sent_meta.get(_norm_sent(sentence))
        if exact is not None:
            return exact
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
        """Prefix a sentence with its heading path, e.g. '[Results > Ablations] ...'."""
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

    # -- diversity selection (MMR) ------------------------------------------ #
    @staticmethod
    def _chain_nodes(c: ChainResult) -> set:
        s: set = set()
        for e in c.chain:
            s.add(e.cause)
            s.add(e.effect)
        return s

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _mmr_select(self, chains: List[ChainResult], top_k: int,
                    lam: float = 0.6) -> List[ChainResult]:
        """Maximal Marginal Relevance: pick top_k chains trading off relevance
        against redundancy (shared nodes), so the selection COVERS more of the
        document instead of returning near-duplicate local chains. This is the
        lever for global / multi-hop questions (Vendi-RAG, MMR)."""
        if len(chains) <= top_k:
            return chains
        scores = [c.rerank_score for c in chains]
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        rel = {id(c): (c.rerank_score - lo) / rng for c in chains}
        nodesets = {id(c): self._chain_nodes(c) for c in chains}

        selected: List[ChainResult] = []
        pool = list(chains)
        while pool and len(selected) < top_k:
            if not selected:
                best = max(pool, key=lambda c: rel[id(c)])
            else:
                def mmr(c: ChainResult) -> float:
                    red = max(self._jaccard(nodesets[id(c)], nodesets[id(s)])
                              for s in selected)
                    return lam * rel[id(c)] - (1 - lam) * red
                best = max(pool, key=mmr)
            selected.append(best)
            pool.remove(best)
        return selected

    # -- retrieve ------------------------------------------------------------ #
    def retrieve(self, question: str, top_k: int = 3,
                 diversify: bool = True) -> List[ChainResult]:
        self._ensure_indexed()
        direction = self._direction(question)
        # Pull more candidate entry nodes than top_k so MMR has room to cover
        # different parts of the document.
        entries = self._entry_nodes(question, top_n=max(4, top_k * 2))
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
        if diversify:
            return self._mmr_select(uniq, top_k)
        return uniq[:top_k]

    # -- direct graph queries (what flat RAG cannot do) ---------------------- #
    def _resolve_node(self, term: str) -> Optional[str]:
        """Map a free-text event to the best-matching graph node (exact, then
        content-word Jaccard). None if nothing matches."""
        nodes = list(self.graph.nodes())
        if not nodes:
            return None
        low = {n.lower(): n for n in nodes}
        if term.lower().strip() in low:
            return low[term.lower().strip()]
        q = set(tokenize(term))
        if not q:
            return None
        best, best_j = None, 0.0
        for n in nodes:
            nt = set(tokenize(n))
            if not nt:
                continue
            j = len(q & nt) / len(q | nt)
            if j > best_j:
                best, best_j = n, j
        return best if best_j > 0 else None

    def root_causes(self, event: str, max_depth: int = 6):
        """Backward causal chains INTO the event — its root causes.
        Returns (resolved_node, [ChainResult,...]). Pure graph query, no LLM."""
        node = self._resolve_node(event)
        if node is None:
            return None, []
        chains = [ChainResult(p, node, 0.0, 0.0, "backward")
                  for p in self.graph.backward_chain(node, max_depth) if p]
        return node, chains

    def impact(self, event: str, max_depth: int = 6):
        """Forward causal chains OUT of the event — its downstream impact /
        blast radius. Returns (resolved_node, [ChainResult,...])."""
        node = self._resolve_node(event)
        if node is None:
            return None, []
        chains = [ChainResult(p, node, 0.0, 0.0, "forward")
                  for p in self.graph.forward_chain(node, max_depth) if p]
        return node, chains

    def connect(self, src: str, dst: str, max_depth: int = 6):
        """Shortest causal path from src to dst.
        Returns (resolved_src, resolved_dst, ChainResult|None)."""
        s, d = self._resolve_node(src), self._resolve_node(dst)
        if s is None or d is None:
            return s, d, None
        path = self.graph.path_between(s, d, max_depth)
        return s, d, (ChainResult(path, s, 0.0, 0.0, "path") if path else None)

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

    @staticmethod
    def _chain_prose(c: ChainResult) -> str:
        """Render a chain as a natural-language sentence instead of arrow
        notation: 'A reduced B, which disrupted C'. Polarity is carried by the
        relation verb. Avoids the symbolic-notation parsing tax that weaker
        models pay on '-/->' arrows."""
        if not c.chain:
            return ""
        parts = [f"{c.chain[0].cause} {c.chain[0].relation.replace('_', ' ')} {c.chain[0].effect}"]
        for e in c.chain[1:]:
            parts.append(f"which {e.relation.replace('_', ' ')} {e.effect}")
        return ", ".join(parts)

    def _build_context(self, chains: List[ChainResult], structured: bool,
                       contextual: bool = True, prose_chains: bool = False,
                       coverage_sentences: Optional[List[str]] = None) -> str:
        """Assemble the LLM context — HYBRID: coverage sentences + causal chains.

        Evidence = the top-k dense-retrieved sentences (coverage, like vector
        RAG) UNION the chains' provenance, deduped, in that order. Standalone
        facts that aren't part of any chain are therefore included — the fix for
        the pure-graph coverage gap.

        structured : also prepend the causal CHAIN PATHS (direction + polarity
                     arrows) as a cause-effect scaffold for reasoning questions.
        contextual : annotate evidence with its document location ([Section > ...]).
        """
        evidence_sents = list(coverage_sentences or [])
        for s in self._dedup_provenance(chains):          # add chain provenance not already covered
            if s not in evidence_sents:
                evidence_sents.append(s)
        evidence = "\n".join((self._annotate(s) if contextual else s)
                             for s in evidence_sents)
        if not structured or not chains:
            return f"Evidence:\n{evidence}"
        render = self._chain_prose if prose_chains else (lambda c: c.text())
        chain_block = "\n".join(f"  {render(c)}" for c in chains)
        return (f"Causal chains (cause->effect structure):\n{chain_block}\n\n"
                f"Evidence:\n{evidence}")

    # -- generate ------------------------------------------------------------ #
    def answer(self, question: str, top_k: int = 3,
               summarize: bool = False, structured: bool = True,
               contextual: bool = True, prose_chains: bool = False
               ) -> Tuple[str, List[ChainResult]]:
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
        # Hybrid: also pull the top-k most relevant sentences for coverage.
        coverage = self._retrieve_sentences(question, k=max(6, top_k * 2))
        return self.generate(question, chains, summarize=summarize,
                             structured=structured, contextual=contextual,
                             prose_chains=prose_chains,
                             coverage_sentences=coverage), chains

    def generate(self, question: str, chains: List[ChainResult],
                 summarize: bool = False, structured: bool = True,
                 contextual: bool = True, prose_chains: bool = False,
                 coverage_sentences: Optional[List[str]] = None) -> str:
        """Generate an answer from retrieved chains + coverage sentences (LLM step
        only). Split out from answer() so a server can retrieve under a lock and
        run the stateless LLM call without holding it (concurrent queries)."""
        if not chains and not coverage_sentences:
            return "No relevant information found for the query."

        if summarize and chains:
            # Two-step generation: compress chains first, then answer
            causal_ctx = self._causal_summary(question, chains)
            cov = "\n".join(coverage_sentences or [])
            prompt = (
                "You are a careful assistant. Answer the question using the "
                "evidence below. Be direct and concise.\n\n"
                f"Causal summary:\n{causal_ctx}\n\nEvidence:\n{cov}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        else:
            ctx = self._build_context(chains, structured=structured,
                                      contextual=contextual,
                                      prose_chains=prose_chains,
                                      coverage_sentences=coverage_sentences)
            legend = (
                "Use the evidence sentences to answer factual details; use the "
                "causal chains to understand cause-effect structure ('->' = "
                "promotes/produces, '-/->' = inhibits/reduces).\n"
                if (structured and chains) else ""
            )
            prompt = (
                "You are a careful assistant. Answer the question using ONLY the "
                "evidence provided. "
                f"{legend}"
                "Be direct and concise. Do not reference chain numbers or labels.\n\n"
                f"{ctx}\n\nQuestion: {question}\n\nAnswer:"
            )

        return self.llm.generate(prompt)

    # -- persistence (warm startup without a database) ----------------------- #
    def save(self, path: str) -> None:
        """Persist an in-memory graph + structure to disk so it can be reloaded
        without re-ingesting (warm startup). The sentence-transformer model is
        NOT pickled — retrieval indices are rebuilt lazily on first query after
        load(). Not for the Neo4j backend (the database is the persistence)."""
        if getattr(self, "using_neo4j", False):
            raise RuntimeError("Neo4j-backed graphs persist in the database; save() is for in-memory only.")
        state = {
            "version": _SAVE_VERSION,
            "dim": self.lex.dim,
            "semantic_weight": self.lex.semantic_weight,
            "max_depth": self.max_depth,
            "edges": [(e.cause, e.relation, e.effect, e.polarity,
                       e.source_sent, e.hv, e.edge_id) for e in self.graph.edges],
            "node_docs": self._node_docs,
            "node_context": self._node_context,
            "struct_index": self._struct_index,
            "sent_meta": self._sent_meta,
            "sentences": self._sentences,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("saved graph (%d edges, %d nodes) to %s",
                 len(self.graph.edges), len(self._node_docs), path)

    @classmethod
    def load(cls, path: str, llm: Optional[object] = None) -> "GraphRAG":
        """Load a graph saved with save(). Retrieval indices rebuild on first query."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("version") != _SAVE_VERSION:
            raise ValueError(
                f"incompatible save version {state.get('version')} "
                f"(expected {_SAVE_VERSION}); re-ingest with this build.")
        rag = cls(dim=state["dim"], semantic_weight=state["semantic_weight"],
                  llm=llm, max_depth=state["max_depth"])
        for cause, rel, eff, pol, src, hv, eid in state["edges"]:
            ge = GraphEdge(cause, rel, eff, pol, src, hv, eid)
            rag.graph.out_adj[cause].append(eid)
            rag.graph.in_adj[eff].append(eid)
            rag.graph.edges.append(ge)
        rag.graph._edge_matrix = None
        rag._node_docs = state["node_docs"]
        rag._node_context = state["node_context"]
        rag._struct_index = state["struct_index"]
        rag._sent_meta = state["sent_meta"]
        rag._sentences = state.get("sentences", [])
        rag._dirty = True   # rebuild BM25/dense/sig + sentence index on first query
        log.info("loaded graph (%d edges, %d nodes) from %s",
                 len(rag.graph.edges), len(rag._node_docs), path)
        return rag

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
