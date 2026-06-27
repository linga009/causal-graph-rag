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

        # Experimental component flags (screened in eval_corpus/screen_components).
        # Default OFF -> the shipped pipeline is unchanged unless explicitly enabled.
        # Validated retrieval components (free screen + Haiku/Sonnet benchmark):
        self.flag_proposition = True        # proposition-aware rerank (edge source sentences)
        self.flag_calibrated_fusion = True  # min-max calibrated channel fusion (vs rank-only RRF)
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
        # Tracks (cause, relation, effect) tuples already in the graph to
        # prevent duplicate edges across multiple ingest() calls.
        self._edge_set: set = set()
        # Sentence position index (populated in _ensure_indexed)
        self._sent_idx: Dict[str, int] = {}
        self._dirty = False
        # Sentence-level coverage index (HYBRID retrieval). The graph alone only
        # surfaces causally-connected content; standalone facts fall through.
        # We ALSO retrieve the top-k most relevant raw sentences (like vector RAG)
        # and feed them alongside the causal chains — structure is additive, not a
        # replacement for coverage (MS GraphRAG local search; NVIDIA/BlackRock
        # HybridRAG: graph+vector beats either alone).
        self._sentences: List[str] = []
        self._sent_vecs = None        # np.ndarray (N, D) once indexed
        self._sent_bm25: Optional[BM25] = None  # sentence-level keyword index
        self._edge_sent_vec: dict = {}  # edge source_sent -> embedding (proposition rerank)
        self._q_emb_cache = None        # (question, emb) memo: one query encoded once

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

        try:
            if llm_extractor is not None:
                edges = extract_edges_hybrid(clean_text, llm_extractor, mode=llm_mode)
            else:
                edges = extract_edges(clean_text)
        except Exception as exc:
            log.warning("edge extraction failed, ingesting 0 edges: %s", exc)
            edges = []

        # Canonicalize node names so the same event ("pump"/"cooling pump") is one
        # node — longer connected chains instead of fragments.
        edges = self._normalize_edges(edges)

        # Build the structure index FIRST so node docs can be enriched with the
        # document context (heading path) each sentence came from.
        self._index_document_structure(ds)
        # Accumulate ALL sentences (not just causal ones) for coverage retrieval.
        self._sentences.extend(s.text for s in ds.sentences())

        for e in edges:
            key = (e.cause, e.relation, e.effect)
            if key in self._edge_set:
                # Duplicate edge: merge source sentence into existing node docs
                for node in (e.cause, e.effect):
                    if node in self._node_docs and e.source_sent not in self._node_docs[node]:
                        self._node_docs[node] += " " + e.source_sent
                continue
            self._edge_set.add(key)
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

        # Pass 3: semantic similarity clustering (only when dense model available).
        # Embeds all entity names, merges pairs with cosine > 0.88 that don't have
        # conflicting token sets. Catches synonyms the subset rule misses:
        # "emergency shutdown" / "scram", "reactor cooling" / "cooling system".
        edges = self._semantic_entity_merge(edges)

        return [e for e in edges if e.cause != e.effect]   # drop self-loops

    def _semantic_entity_merge(self, edges):
        """Merge semantically near-identical entity names using dense embeddings.
        Only runs when a real sentence-transformer model is available."""
        model = getattr(self.dense, "_model", None)
        if model is None or not edges:
            return edges

        names = sorted({n for e in edges for n in (e.cause, e.effect)}
                       | set(self.graph.nodes()))
        if len(names) < 2:
            return edges

        try:
            embs = model.encode(names, convert_to_numpy=True, normalize_embeddings=True)
        except Exception:
            return edges

        # Build similarity matrix and find pairs cosine > 0.88
        sim = embs @ embs.T
        merge_map: dict = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if sim[i, j] < 0.88:
                    continue
                a, b = names[i], names[j]
                # Don't merge if they share no content tokens (fully distinct concepts)
                ta, tb = _entity_tokens(a), _entity_tokens(b)
                if ta and tb and not (ta & tb):
                    continue
                # Representative = the longer (more specific) name
                rep = b if len(b) >= len(a) else a
                sub = a if rep == b else b
                # Chain through any existing mapping
                while merge_map.get(rep, rep) != rep:
                    rep = merge_map[rep]
                merge_map[sub] = rep

        if not merge_map:
            return edges

        def resolve_sem(x):
            seen: set = set()
            while x in merge_map and x not in seen:
                seen.add(x)
                x = merge_map[x]
            return x

        for e in edges:
            e.cause = resolve_sem(e.cause)
            e.effect = resolve_sem(e.effect)
        log.debug("semantic entity merge: %d merges applied", len(merge_map))
        return edges

    def _ensure_indexed(self) -> None:
        """(Re)build BM25/dense/path-signature indices if ingest marked them
        dirty. Called by retrieve(); idempotent and cheap when clean."""
        if not self._dirty:
            return
        import time
        t0 = time.monotonic()
        # Contextual Retrieval: index context-enriched node docs (heading path
        # folded in) for BM25 + dense; path signatures keep raw sentence order.
        enriched = {
            n: (" ".join(self._node_context.get(n, ())) + " " + d).strip()
            for n, d in self._node_docs.items()
        }
        self.bm25.index(enriched)
        self.dense.index(enriched)
        self.sig.index(self._node_docs)
        # Sentence-level coverage index: dense vectors + BM25 for hybrid retrieval.
        model = getattr(self.dense, "_model", None)
        if model is not None and self._sentences:
            self._sent_vecs = model.encode(self._sentences, convert_to_numpy=True,
                                           normalize_embeddings=True)
        else:
            self._sent_vecs = None
        # Sentence-level BM25 — keyed by string index so score() returns (score,"i")
        if self._sentences:
            sent_docs = {str(i): s for i, s in enumerate(self._sentences)}
            self._sent_bm25 = BM25()
            self._sent_bm25.index(sent_docs)
        else:
            self._sent_bm25 = None
        # Precompute edge source-sentence embeddings ONCE at ingest, so the
        # proposition-aware rerank is a cached dot product at query time
        # (not a model.encode() call per query). Keeps query latency low.
        if model is not None:
            uniq_src = list({e.source_sent for e in self.graph.edges if e.source_sent})
            if uniq_src:
                ev = model.encode(uniq_src, convert_to_numpy=True,
                                  normalize_embeddings=True)
                self._edge_sent_vec = {s: ev[i] for i, s in enumerate(uniq_src)}
            else:
                self._edge_sent_vec = {}
        else:
            self._edge_sent_vec = {}
        # O(1) sentence-position lookup for provenance expansion
        self._sent_idx: Dict[str, int] = {s: i for i, s in enumerate(self._sentences)}
        self._dirty = False
        elapsed = time.monotonic() - t0
        log.info("rebuilt retrieval indices: %d nodes, %d sentences in %.2fs",
                 len(self._node_docs), len(self._sentences), elapsed)

    def _encode_query(self, question: str):
        """Encode the query once and memoize it for the duration of one query.
        answer() drives several encoders on the SAME question (_rerank up to
        twice via the bridge pass, _retrieve_sentences, optional beam) — without
        this memo each re-encodes on CPU (~tens of ms each). Returns None if no
        dense model is loaded."""
        model = getattr(self.dense, "_model", None)
        if model is None:
            return None
        cache = self._q_emb_cache
        if cache is not None and cache[0] == question:
            return cache[1]
        emb = model.encode([question], convert_to_numpy=True,
                           normalize_embeddings=True)[0]
        self._q_emb_cache = (question, emb)
        return emb

    def _retrieve_sentences(self, question: str, k: int = 6,
                            chain_nodes: Optional[set] = None) -> List[str]:
        """Top-k most relevant raw sentences via hybrid BM25+dense RRF with MMR diversity.

        Hybrid retrieval catches what each channel alone misses:
          - Dense: semantic equivalence, paraphrases, conceptual similarity
          - BM25:  exact keyword match, rare technical terms, named entities
        RRF (k=60) fuses both rank lists before the MMR diversity pass.
        chain_nodes bonus: sentences mentioning chain nodes score +0.15."""
        self._ensure_indexed()
        if self._sent_vecs is None or not self._sentences:
            return []
        q = self._encode_query(question)
        dense_sims = self._sent_vecs @ q   # shape (N,)
        n = len(self._sentences)

        # --- Hybrid RRF: fuse dense and BM25 rank lists --------------------- #
        _RRF_K = 60  # standard constant; smooths rank differences
        dense_rank = np.empty(n, dtype=np.float32)
        dense_rank[np.argsort(-dense_sims)] = np.arange(n)

        if self._sent_bm25 is not None:
            bm25_raw = np.zeros(n, dtype=np.float32)
            for score, key in self._sent_bm25.score(question):
                bm25_raw[int(key)] = score
            bm25_rank = np.empty(n, dtype=np.float32)
            bm25_rank[np.argsort(-bm25_raw)] = np.arange(n)
            fused = (1.0 / (_RRF_K + dense_rank) + 1.0 / (_RRF_K + bm25_rank))
        else:
            fused = 1.0 / (_RRF_K + dense_rank)

        # Normalise fused to [0,1] so chain-node bonus has consistent scale
        fused_max = fused.max()
        if fused_max > 0:
            fused = fused / fused_max

        # Build chain-node term set for coverage bonus
        chain_terms: set = set()
        if chain_nodes:
            for node in chain_nodes:
                chain_terms.update(tokenize(node))

        # Retrieve a larger candidate pool, then apply MMR
        pool_size = min(k * 4, n)
        pool_idx = np.argsort(-fused)[:pool_size]
        pool_scores = fused[pool_idx].copy()

        # Apply chain-node relevance bonus before MMR
        if chain_terms:
            for i, idx in enumerate(pool_idx):
                sent_words = set(tokenize(self._sentences[idx]))
                overlap = len(chain_terms & sent_words) / max(1, len(chain_terms))
                pool_scores[i] += 0.15 * overlap

        pool_vecs = self._sent_vecs[pool_idx]

        # MMR: lam=0.7 (relevance-weighted), iteratively select diverse sentences.
        # Relevance = hybrid RRF score; redundancy = dense cosine (best for dedup).
        lam = 0.7
        selected_idx: List[int] = []
        selected_vecs: List[np.ndarray] = []

        while len(selected_idx) < k and len(selected_idx) < len(pool_idx):
            best_i, best_score = -1, float("-inf")
            for i in range(len(pool_idx)):
                if i in selected_idx:
                    continue
                rel = float(pool_scores[i])
                if selected_vecs:
                    red = max(float(pool_vecs[i] @ sv) for sv in selected_vecs)
                else:
                    red = 0.0
                score = lam * rel - (1 - lam) * red
                if score > best_score:
                    best_score, best_i = score, i
            if best_i < 0:
                break
            selected_idx.append(best_i)
            selected_vecs.append(pool_vecs[best_i])

        return [self._sentences[pool_idx[i]] for i in selected_idx]

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
    def _entry_nodes(self, question: str, top_n: int = 4,
                     direction: str = "forward") -> List[Tuple[float, str]]:
        from retrievers import tokenize
        q_terms = set(tokenize(question))

        # Channel 0: direct node-name match via inverted index — O(|q_terms|).
        # Direction-aware scoring: rootcause queries prefer effect-heavy nodes
        # (high in-degree); forward/impact queries prefer cause-heavy nodes.
        degree_bonus: dict = {}
        for e in self.graph.edges:
            if direction == "backward":
                degree_bonus[e.effect] = degree_bonus.get(e.effect, 0) + 1
            else:
                degree_bonus[e.cause] = degree_bonus.get(e.cause, 0) + 1

        direct: List[Tuple[float, str]] = []
        widx = self.graph.word_index()
        seen_direct: set = set()
        for term in q_terms:
            for node in widx.get(term, []):
                if node not in seen_direct:
                    seen_direct.add(node)
                    d = degree_bonus.get(node, 0)
                    score = 1.0 + min(0.5, d * 0.1)   # up to +0.5 for high-degree nodes
                    direct.append((score, node))

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

        _channels = [direct, vsa_ranked, bm25_ranked, dense_ranked, sig_ranked]
        _weights = [1.5, 2.0, 1.0, 1.0, 1.2]
        if self.flag_calibrated_fusion:
            fused = self._calibrated_fuse(_channels, _weights)
        else:
            fused = rrf_fuse(_channels, weights=_weights)
        result = fused[:top_n]
        log.debug("entry nodes for %r: %s", question[:50],
                  [(n, round(s, 3)) for s, n in result])
        return result

    # -- query direction ----------------------------------------------------- #
    def _direction(self, question: str) -> str:
        low = question.lower()
        if any(p in low for p in _BACKWARD_Q):
            return "backward"
        return "forward"

    # -- query-adaptive traversal depth ------------------------------------- #
    _DEEP_Q  = ("trace", "chain of", "step by step", "series of", "sequence of",
                "propagat", "ultimately", "eventually")
    _SHALLOW_Q = ("what is", "how many", "how much", "when was", "when did",
                  "who ", "which year", "what year", "what date", "how old")

    def _adaptive_depth(self, question: str) -> int:
        """Return BFS max_depth tuned to question complexity.
        'Trace how X ultimately led to Y' needs longer chains than 'What is X?'."""
        low = question.lower()
        if any(w in low for w in self._DEEP_Q):
            return min(self.max_depth + 2, 10)   # deep chains for trace questions
        if any(w in low for w in self._SHALLOW_Q):
            return max(self.max_depth - 2, 3)    # shallow for fact lookups
        return self.max_depth

    # ===================================================================== #
    #  Channel fusion (calibrated)
    # ===================================================================== #
    @staticmethod
    def _calibrated_fuse(lists, weights):
        """Min-max-calibrated fusion: scale each channel's scores to [0,1], then
        weighted-sum over the union of nodes. Recovers the score-magnitude signal
        rank-only RRF discards. Unlike z-norm, a single-element or zero-variance
        channel maps its node to 1.0 — so the strong exact-name-match (`direct`)
        channel keeps full weight, and weak presence is a small positive, never a
        penalty."""
        agg: dict = {}
        for lst, w in zip(lists, weights):
            if not lst:
                continue
            scores = np.array([sc for sc, _ in lst], dtype=float)
            lo, hi = scores.min(), scores.max()
            rng = hi - lo
            for sc, node in lst:
                norm = (sc - lo) / rng if rng > 0 else 1.0
                agg[node] = agg.get(node, 0.0) + w * norm
        return sorted(((sc, n) for n, sc in agg.items()), key=lambda x: -x[0])

    # -- rerank chains ------------------------------------------------------- #
    def _rerank(self, question: str, chains: List[ChainResult]) -> None:
        q_terms = set(tokenize(question))

        # Encode query once (memoized); build a node->embedding lookup from the index.
        q_emb = None
        node_embs: dict = {}
        if getattr(self.dense, 'vecs', None):
            q_emb = self._encode_query(question)
            # Pre-collect all unique nodes across ALL chains in one pass (batch)
            all_nodes = {n for c in chains for e in c.chain
                         for n in (e.cause, e.effect)}
            node_embs = {n: self.dense.vecs[n] for n in all_nodes
                         if n in self.dense.vecs}

        # Proposition-aware rerank: use the edge source-sentence embeddings
        # precomputed at ingest (no query-time model.encode -> microsecond cost).
        prop_vec = self._edge_sent_vec if (self.flag_proposition and q_emb is not None) else {}

        for c in chains:
            chain_terms = set()
            for e in c.chain:
                chain_terms |= set(tokenize(f"{e.cause} {e.relation} {e.effect}"))
            overlap = len(q_terms & chain_terms)
            score = overlap + 0.25 * len(c.chain)

            # Direction-aware endpoint bonus (strong signal: chain anchors on query)
            if c.chain:
                head = set(tokenize(c.chain[0].cause))
                tail = set(tokenize(c.chain[-1].effect))
                if c.direction == "backward" and (tail & q_terms):
                    score += 5.0
                if c.direction == "forward" and (head & q_terms):
                    score += 5.0

            # Semantic similarity: mean of pre-fetched node embeddings
            if q_emb is not None and node_embs:
                vecs = [node_embs[n] for e in c.chain
                        for n in (e.cause, e.effect) if n in node_embs]
                if vecs:
                    chain_emb = np.mean(vecs, axis=0)
                    norm = np.linalg.norm(chain_emb)
                    if norm:
                        chain_emb /= norm
                    score += float(q_emb @ chain_emb) * 3.0

            # Proposition-aware: query similarity of the chain's source sentences
            # (the full proposition text, not just node names). Default ON.
            if prop_vec and c.chain:
                sims = [float(q_emb @ prop_vec[e.source_sent])
                        for e in c.chain if e.source_sent in prop_vec]
                if sims:
                    score += 3.0 * (sum(sims) / len(sims))

            # Confidence weighting: geometric mean of edge confidences.
            # Scaled [0.75, 1.0] so high-confidence chains score meaningfully higher.
            if c.chain:
                confs = [getattr(e, "confidence", 0.85) for e in c.chain]
                chain_conf = float(np.prod(confs) ** (1.0 / len(confs)))
                score *= (0.75 + 0.25 * chain_conf)  # scales in [0.75, 1.0]

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

        # Track (relation, effect) pairs used by selected chains to penalise
        # chains that repeat the same causal mechanism.
        used_mechanisms: set = set()

        selected: List[ChainResult] = []
        pool = list(chains)
        while pool and len(selected) < top_k:
            if not selected:
                best = max(pool, key=lambda c: rel[id(c)])
            else:
                def mmr(c: ChainResult) -> float:
                    node_red = max(self._jaccard(nodesets[id(c)], nodesets[id(s)])
                                   for s in selected)
                    # Extra penalty if this chain reuses an already-selected mechanism
                    chain_mechs = {(e.relation, e.effect) for e in c.chain}
                    mech_overlap = len(chain_mechs & used_mechanisms) / max(1, len(chain_mechs))
                    red = node_red + 0.3 * mech_overlap
                    return lam * rel[id(c)] - (1 - lam) * red
                best = max(pool, key=mmr)
            selected.append(best)
            used_mechanisms.update((e.relation, e.effect) for e in best.chain)
            pool.remove(best)
        return selected

    # -- retrieve ------------------------------------------------------------ #
    def retrieve(self, question: str, top_k: int = 3,
                 diversify: bool = True) -> List[ChainResult]:
        self._ensure_indexed()
        direction = self._direction(question)
        depth = self._adaptive_depth(question)
        entries = self._entry_nodes(question, top_n=max(4, top_k * 2),
                                    direction=direction)
        graph_nodes = self.graph.nodes()

        def _bfs_from(node: str, rrf: float) -> None:
            if node not in graph_nodes:
                return
            paths = (self.graph.backward_chain(node, depth)
                     if direction == "backward"
                     else self.graph.forward_chain(node, depth))
            for path in paths:
                if path:
                    results.append(ChainResult(path, node, rrf, 0.0, direction))

        results: List[ChainResult] = []
        for rrf_score, node in entries:
            _bfs_from(node, rrf_score)

        # Dedup
        seen: set = set()
        uniq: List[ChainResult] = []
        for r in results:
            key = tuple((e.cause, e.relation, e.effect) for e in r.chain)
            if key not in seen:
                seen.add(key)
                uniq.append(r)

        self._rerank(question, uniq)
        uniq.sort(key=lambda r: (-r.rerank_score, -r.rrf_score))

        # Bridge-evidence 2nd pass: if the tails of the top chains don't overlap
        # with query terms, the traversal stopped short. Reseed from those tails
        # to extend chains one more hop toward the answer.
        q_terms = set(tokenize(question))
        top_chains = uniq[:3]
        tail_nodes = {c.chain[-1].effect for c in top_chains if c.chain}
        tail_terms = {t for n in tail_nodes for t in tokenize(n)}
        if tail_nodes and not (q_terms & tail_terms):
            n_results_before = len(results)   # true boundary of pre-2nd-pass results
            n_uniq_before = len(uniq)
            for node in tail_nodes:
                _bfs_from(node, 0.3)   # lower rrf score for 2nd-pass seeds
            for r in results[n_results_before:]:   # only chains appended in the 2nd pass
                key = tuple((e.cause, e.relation, e.effect) for e in r.chain)
                if key not in seen:
                    seen.add(key)
                    uniq.append(r)
            if len(uniq) > n_uniq_before:     # re-rank only if new chains were actually added
                self._rerank(question, uniq)
                uniq.sort(key=lambda r: (-r.rerank_score, -r.rrf_score))
                log.debug("bridge 2nd pass added %d chains", len(uniq) - n_uniq_before)

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

    def _expand_provenance(self, provenance: List[str], cap: int = 4) -> List[str]:
        """Return up to `cap` document-adjacent sentences that follow each
        provenance sentence in document order. Adds narrative context around
        each chain hop without extra encoding calls — helps multi-hop questions
        where the source sentence alone doesn't explain the mechanism."""
        if not self._sentences or not provenance or not self._sent_idx:
            return []
        extras: List[str] = []
        added: set = set(provenance)
        for sent in provenance:
            if len(extras) >= cap:
                break
            idx = self._sent_idx.get(sent)
            if idx is None:
                continue
            # Take the immediately following sentence (forward context is most useful)
            nxt = idx + 1
            if nxt < len(self._sentences):
                neighbor = self._sentences[nxt]
                if neighbor not in added and neighbor.strip():
                    extras.append(neighbor)
                    added.add(neighbor)
        return extras

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
        chain_prov = self._dedup_provenance(chains)
        for s in chain_prov:                              # add chain provenance not already covered
            if s not in evidence_sents:
                evidence_sents.append(s)
        # Add one document-adjacent sentence per provenance hop (richer per-hop context)
        for s in self._expand_provenance(chain_prov):
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
        # Score gate: if no chain is meaningfully relevant, skip chain context
        # and fall back to coverage sentences only. Prevents irrelevant graph
        # traversal from polluting purely factual or poorly-covered queries.
        _CHAIN_GATE = 2.0
        if chains and max(c.rerank_score for c in chains) < _CHAIN_GATE:
            log.debug("chain gate: best score %.2f < %.2f; coverage-only mode",
                      max(c.rerank_score for c in chains), _CHAIN_GATE)
            chains = []
        # Hybrid: pull the top-k most relevant sentences, boosting those that
        # mention nodes in the selected chains (chain-aware coverage).
        chain_nodes = {n for c in chains for e in c.chain
                       for n in (e.cause, e.effect)}
        coverage = self._retrieve_sentences(question, k=max(6, top_k * 2),
                                            chain_nodes=chain_nodes or None)
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
                       e.source_sent, e.hv, e.edge_id,
                       getattr(e, "confidence", 0.85)) for e in self.graph.edges],
            "node_docs": self._node_docs,
            "node_context": self._node_context,
            "struct_index": self._struct_index,
            "sent_meta": self._sent_meta,
            "sentences": self._sentences,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("saved graph (%d edges, %d nodes) to %s",
                 len(list(self.graph.edges)), len(self._node_docs), path)

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
        for row in state["edges"]:
            if len(row) == 8:
                cause, rel, eff, pol, src, hv, eid, conf = row
            else:
                cause, rel, eff, pol, src, hv, eid = row
                conf = 0.85
            ge = GraphEdge(cause, rel, eff, pol, src, hv, eid, conf)
            rag.graph.out_adj[cause].append(eid)
            rag.graph.in_adj[eff].append(eid)
            rag.graph._edges.append(ge)
        rag.graph._edge_matrix = None
        rag._node_docs = state["node_docs"]
        rag._node_context = state["node_context"]
        rag._struct_index = state["struct_index"]
        rag._sent_meta = state["sent_meta"]
        rag._sentences = state.get("sentences", [])
        # Restore the dedup set so ingest() AFTER load() doesn't re-add duplicate
        # edges (the set is keyed by (cause, relation, effect)).
        rag._edge_set = {(e.cause, e.relation, e.effect) for e in rag.graph.edges}
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
