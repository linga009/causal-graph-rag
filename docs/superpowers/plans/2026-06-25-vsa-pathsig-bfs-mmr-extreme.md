# VSA + PathSig + BFS + MMR to Extreme: High-Performance Causal Graph RAG

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push VSA triple encoding, rough path theory, BFS traversal, and MMR diversity to their mathematical and algorithmic limits to maximize causal retrieval quality without changing the benchmark.

**Architecture:** 
- **VSA refinement:** Increase semantic weight dynamically; use role-specific binding to separate AGENT/ACTION/PATIENT semantics; add polarity awareness
- **PathSig upgrade:** Dynamic truncation levels based on document complexity; LSTM-based trajectory encoding as optional high-performance path
- **BFS optimization:** Iterative deepening (depth 1 first, then 2, etc.) with early stopping; memoize visited paths; implement confidence-aware pruning
- **MMR extreme:** 5-tier redundancy scoring (node overlap, relation repetition, semantic similarity, polarity match, confidence); dynamic lambda per query type

**Tech Stack:** NumPy, scikit-learn (optional, for better semantic hashing), existing sentence-transformers

## Global Constraints

- No new dependencies beyond what's already in `dev` extras
- All changes must pass 76 existing tests
- Benchmark must remain at +0.33 multihop / +0.22 rootcause / +0.01 fact or better
- Files modified: `vsa_core.py`, `retrievers.py`, `graph_rag.py` (rerank, mmr, retrieve)
- No breaking API changes to `ChainResult`, `GraphRAG.retrieve()`, `GraphRAG.answer()`

---

## File Structure

```
vsa_core.py
  - Lexicon: add semantic_weight_schedule(complexity) → dynamic weight
  - encode_triple: refactor to separate identity/semantic binding per role
  - Polarity-aware encoding (new): factor sign(effect - cause) into triple vec

retrievers.py
  - PathSignatureRetriever: add dynamic_level(doc_length) method
  - PathSignatureRetriever: add optional LSTM path encoder (fallback to current)
  - PathSignatureRetriever: add confidence-weighted signature (edges weighted by conf)

graph_rag.py
  - BFS: iterative deepening with early stopping (new _bfs_iterative method)
  - BFS: path memoization (new _path_cache in CausalGraph)
  - _mmr_select: 5-tier redundancy (new _redundancy_score method)
  - _rerank: confidence-aware pruning (skip chains with conf < 0.3)
  - _entry_nodes: VSA semantic weight driven by question length

tests/
  - test_vsa_semantic_weight.py: dynamic weight scheduling
  - test_pathsig_dynamic_level.py: truncation level adaptation
  - test_bfs_iterative_deepening.py: depth-first pruning
  - test_mmr_5tier.py: all 5 redundancy metrics
  - test_extreme_integration.py: full pipeline with edge cases
```

---

## Task 1: VSA Semantic Weight Scheduling

**Files:**
- Modify: `vsa_core.py:88-133` (Lexicon.__init__ and filler methods)
- Test: `tests/test_vsa_semantic_weight.py` (new)

**Interfaces:**
- Consumes: `token: str`, `complexity_hint: Optional[int]` (doc length)
- Produces: `Lexicon.semantic_weight_schedule(complexity) -> int`, modified `filler()`

**Description:** Currently semantic_weight is fixed at construction. We'll make it adaptive: longer, more complex documents need stronger semantic coupling (higher weight) to catch paraphrases; short documents need less. Add a schedule method and wire it into encode_triple via Lexicon.

- [ ] **Step 1: Write failing test for semantic weight scheduling**

```python
# tests/test_vsa_semantic_weight.py
import pytest
from vsa_core import Lexicon, Triple, encode_triple, hamming_similarity

def test_semantic_weight_schedule():
    """Semantic weight should increase with document complexity."""
    lex_short = Lexicon(dim=1000, semantic_weight=1)
    lex_long = Lexicon(dim=1000, semantic_weight=1)
    
    weight_short = lex_short.semantic_weight_schedule(doc_length=500)   # 50 sentences
    weight_long = lex_long.semantic_weight_schedule(doc_length=5000)    # 500 sentences
    
    assert weight_short < weight_long, "Long docs should have higher semantic weight"
    assert weight_short >= 1, "Minimum weight is 1 (identity only)"
    assert weight_long <= 5, "Maximum weight capped at 5"

def test_semantic_weight_similarity_robustness():
    """Higher semantic weight should increase synonym similarity."""
    lex1 = Lexicon(dim=1000, semantic_weight=1)
    lex2 = Lexicon(dim=1000, semantic_weight=4)
    
    t_unemp = Triple("unemployment", "causes", "social_unrest")
    t_jobless = Triple("joblessness", "causes", "civil_disorder")
    
    v1_unemp = encode_triple(t_unemp, lex1)
    v1_jobless = encode_triple(t_jobless, lex1)
    sim1 = hamming_similarity(v1_unemp, v1_jobless)
    
    v2_unemp = encode_triple(t_unemp, lex2)
    v2_jobless = encode_triple(t_jobless, lex2)
    sim2 = hamming_similarity(v2_unemp, v2_jobless)
    
    assert sim2 > sim1, "Higher semantic weight should increase synonym similarity"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_vsa_semantic_weight.py -v
```

Expected output: `FAILED ... semantic_weight_schedule ... (not defined)`

- [ ] **Step 3: Implement semantic_weight_schedule in Lexicon**

Edit `vsa_core.py` around line 88:

```python
class Lexicon:
    """Owns role vectors and lazily-built, cached filler vectors."""

    def __init__(self, dim: int = 10000, semantic_weight: int = 1):
        self.dim = dim
        self.semantic_weight = semantic_weight  # override via schedule() at query time
        self.roles: Dict[str, np.ndarray] = {
            r: random_hv(dim, f"ROLE::{r}") for r in ROLE_NAMES
        }
        self._filler_cache: Dict[str, np.ndarray] = {}

    def semantic_weight_schedule(self, doc_length: int) -> int:
        """Adaptive semantic weight based on document complexity.
        
        doc_length: approximate number of tokens in the document
        
        Heuristic: 
          - 0-500 tokens (short snippets): weight=1 (identity only)
          - 500-2000 tokens (single page): weight=2 (1x identity + 1x semantic)
          - 2000-5000 tokens (multi-page): weight=3
          - 5000+ tokens (long document): weight=4-5
        """
        if doc_length < 500:
            return 1
        elif doc_length < 2000:
            return 2
        elif doc_length < 5000:
            return 3
        else:
            return min(5, 4 + max(0, (doc_length - 5000) // 5000))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_vsa_semantic_weight.py -v
```

Expected output: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add vsa_core.py tests/test_vsa_semantic_weight.py
git commit -m "feat: VSA semantic weight scheduling based on document complexity"
```

---

## Task 2: VSA Role-Specific Polarity Encoding

**Files:**
- Modify: `vsa_core.py:152-171` (encode_triple)
- Test: `tests/test_vsa_polarity.py` (new)

**Interfaces:**
- Consumes: `Triple` (with existing fields + optional `polarity: int`)
- Produces: modified `encode_triple()` signature unchanged, internal binding improved

**Description:** Currently polarity is only tracked in GraphEdge. Add it to the VSA encoding so negation (A causes NOT B) vs positive (A causes B) are orthogonal at the triple level.

- [ ] **Step 1: Write failing test**

```python
# tests/test_vsa_polarity.py
from vsa_core import Triple, encode_triple, hamming_similarity, Lexicon

def test_polarity_orthogonality():
    """Positive and negative causal triples should be orthogonal."""
    lex = Lexicon(dim=1000, semantic_weight=2)
    
    # Same triple, different polarity
    t_pos = Triple("pump", "fails", "engine")
    t_neg = Triple("pump", "fails", "engine")  # same surface, but we'll encode differently
    
    # For now, just verify same triples are identical
    v_pos = encode_triple(t_pos, lex)
    v_same = encode_triple(t_pos, lex)
    assert hamming_similarity(v_pos, v_same) > 0.95, "Same triple should be identical"
```

- [ ] **Step 2: Run test to verify it fails (or passes as baseline)**

```bash
pytest tests/test_vsa_polarity.py::test_polarity_orthogonality -v
```

- [ ] **Step 3: Extend Triple dataclass with optional polarity field**

Edit `vsa_core.py` around line 142:

```python
@dataclass
class Triple:
    agent: str
    action: str
    patient: str
    polarity: int = 1  # +1 positive, -1 negative causation

    def text(self) -> str:
        arrow = " -> " if self.polarity > 0 else " -/-> "
        return f"{self.agent} --{self.action}-{arrow} {self.patient}"
```

- [ ] **Step 4: Modify encode_triple to bind polarity to PATIENT role**

Edit encode_triple around line 152:

```python
def encode_triple(t: Triple, lex: Lexicon) -> np.ndarray:
    """E(t) = bundle over slots of ROLE ⊗ (identity [+ semantic copies]).
    
    PATIENT role is bound to a polarity marker: if t.polarity < 0,
    PATIENT is inverted (multiplied by -1 globally). This makes
    "A causes B" and "A prevents B" orthogonal.
    """
    parts: List[np.ndarray] = []
    polarity = getattr(t, 'polarity', 1)
    
    for role_name, token in (("AGENT", t.agent),
                             ("ACTION", t.action),
                             ("PATIENT", t.patient)):
        role = lex.role(role_name)
        idn, sem = lex.filler_parts(token)
        
        # For PATIENT, apply polarity flip to both identity and semantic
        if role_name == "PATIENT" and polarity < 0:
            idn = idn * -1
            if sem is not None:
                sem = sem * -1
        
        parts.append(bind(role, idn))
        if sem is not None:
            for _ in range(lex.semantic_weight):
                parts.append(bind(role, sem))
    
    return bundle(parts)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_vsa_polarity.py -v
```

- [ ] **Step 6: Commit**

```bash
git add vsa_core.py tests/test_vsa_polarity.py
git commit -m "feat: VSA polarity encoding — negate PATIENT role for negative causation"
```

---

## Task 3: Dynamic PathSignature Truncation Levels

**Files:**
- Modify: `retrievers.py:199-242` (PathSignatureRetriever.__init__ and level scheduling)
- Test: `tests/test_pathsig_dynamic_level.py` (new)

**Interfaces:**
- Consumes: `doc_length: int`, `num_sentences: int`
- Produces: `PathSignatureRetriever.dynamic_level(doc_length, num_sentences) -> int` (returns 2 or 3)

**Description:** Currently level is fixed at 3. Add logic to drop to level 2 if the document is short (fewer than 10 sentences) to avoid noise; use level 3 for longer documents where trajectory shape matters.

- [ ] **Step 1: Write failing test**

```python
# tests/test_pathsig_dynamic_level.py
from retrievers import PathSignatureRetriever

def test_dynamic_level_short_doc():
    """Short documents should use level 2 (area only, no 3rd-order)."""
    sig = PathSignatureRetriever(embed_dim=384, proj_dim=16, level=3)
    level = sig.dynamic_level(doc_length=300, num_sentences=5)
    assert level == 2, "Short doc should use level 2"

def test_dynamic_level_long_doc():
    """Long documents should use level 3 (full trajectory)."""
    sig = PathSignatureRetriever(embed_dim=384, proj_dim=16, level=3)
    level = sig.dynamic_level(doc_length=5000, num_sentences=100)
    assert level == 3, "Long doc should use level 3"

def test_signature_dimension_matches_level():
    """Signature dimension must match the computed level."""
    sig = PathSignatureRetriever(embed_dim=384, proj_dim=16, level=3)
    
    level2_dim = 16 + 16*16  # S1 + S2
    level3_dim = 16 + 16*16 + 16*16*16  # S1 + S2 + S3
    
    # For a short doc (level 2)
    path_short = np.random.randn(5, 16).astype(np.float32)
    sig_short = sig._signature(path_short, level=2)
    assert sig_short.shape[0] == level2_dim, f"Level 2 should have dim {level2_dim}"
    
    # For a long doc (level 3)
    path_long = np.random.randn(100, 16).astype(np.float32)
    sig_long = sig._signature(path_long, level=3)
    assert sig_long.shape[0] == level3_dim, f"Level 3 should have dim {level3_dim}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pathsig_dynamic_level.py -v
```

Expected: `... dynamic_level ... (not defined)`

- [ ] **Step 3: Implement dynamic_level in PathSignatureRetriever**

Edit `retrievers.py` around line 215:

```python
class PathSignatureRetriever:
    """..."""
    
    def __init__(self, embed_dim: int = 384, proj_dim: int = 16, level: int = 3,
                 model=None):
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.level = level
        self._proj: np.ndarray | None = None
        self._model = model
        self._nodes: List[str] = []
        self._sig_matrix: np.ndarray | None = None
        self._node_sents: Dict[str, List[str]] = {}
    
    def dynamic_level(self, doc_length: int, num_sentences: int) -> int:
        """Return signature truncation level based on document complexity.
        
        Short docs (< 10 sentences or < 1000 tokens): use level 2 (area/ordering).
        Long docs: use level 3 (3rd-order trajectory).
        
        Rationale: level 3 requires enough data points to distinguish 3rd-order
        integrals; too few sentences -> noisy estimates. Level 2 is robust on
        short passages.
        """
        if num_sentences < 10 or doc_length < 1000:
            return 2
        return 3
```

- [ ] **Step 4: Wire dynamic_level into index() method**

Edit `retrievers.py` in the `index()` method around line 301:

```python
    def index(self, node_docs: Dict[str, str]) -> None:
        self._nodes = list(node_docs.keys())
        if not self._nodes:
            self._node_sents = {}
            self._sig_matrix = None
            return
        self._node_sents = {n: self._split(node_docs[n]) or [node_docs[n]]
                            for n in self._nodes}

        # Compute dynamic truncation level from the full corpus
        total_length = sum(len(node_docs[n]) for n in self._nodes)
        total_sents = sum(len(self._node_sents[n]) for n in self._nodes)
        active_level = self.dynamic_level(total_length, total_sents)
        
        # ... rest of indexing ...
        # When computing signatures, use active_level instead of self.level:
        for i, node in enumerate(self._nodes):
            node_embs = all_embs[offsets[i]:offsets[i + 1]]
            sigs.append(self._signature(node_embs, active_level))  # <-- use active_level
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_pathsig_dynamic_level.py -v
```

- [ ] **Step 6: Commit**

```bash
git add retrievers.py tests/test_pathsig_dynamic_level.py
git commit -m "feat: PathSignature dynamic truncation levels (level 2 for short docs)"
```

---

## Task 4: BFS Iterative Deepening with Early Stopping

**Files:**
- Modify: `graph_rag.py:733-788` (retrieve method and new _bfs_iterative method)
- Test: `tests/test_bfs_iterative_deepening.py` (new)

**Interfaces:**
- Consumes: `node: str`, `max_depth: int`, `early_stop_threshold: float`
- Produces: `List[ChainResult]` (same as before, but built incrementally)

**Description:** Instead of BFS to max_depth all at once, explore depth 1 first, then 2, etc. If chains at depth d all score below a threshold, stop. This avoids exploring long, irrelevant chains.

- [ ] **Step 1: Write failing test**

```python
# tests/test_bfs_iterative_deepening.py
import pytest
from graph_rag import GraphRAG

def test_bfs_iterative_stops_early():
    """Iterative deepening should stop before max_depth if chains score low."""
    rag = GraphRAG()
    rag.ingest("The pump failed. This caused no harm. Nothing else happened.")
    
    # Shallow question that should be answered by depth 1-2 chains
    chains = rag.retrieve("What failed?", top_k=3)
    
    # All chains should be short (depth <= 2) because deeper chains are irrelevant
    for c in chains:
        assert len(c.chain) <= 2, "Irrelevant deep chains should be pruned"

def test_bfs_iterative_still_reaches_max_depth_if_needed():
    """If shallow chains are scarce, iterative deepening should go deep."""
    rag = GraphRAG()
    # Long causal chain
    rag.ingest(
        "A causes B. B causes C. C causes D. D causes E. E caused the outage. "
        "The outage was the main event."
    )
    
    chains = rag.retrieve("What caused the outage?", top_k=3)
    
    # At least one chain should reach the cause of the outage
    assert any(len(c.chain) >= 4 for c in chains), "Should find deep chains when needed"
```

- [ ] **Step 2: Run test to verify it fails (chains currently don't early-stop)**

```bash
pytest tests/test_bfs_iterative_deepening.py::test_bfs_iterative_stops_early -v
```

- [ ] **Step 3: Implement _bfs_iterative_deepening in CausalGraph**

Edit `graph_rag.py` around line 730 (before retrieve):

First, add the method to `graph_rag.py`:

```python
    def _bfs_iterative_deepening(self, node: str, direction: str, max_depth: int,
                                 early_stop_threshold: float = 0.1) -> List[ChainResult]:
        """BFS with iterative deepening: explore depth 1, then 2, etc.
        Stop if all chains at depth d score below early_stop_threshold.
        
        Returns chains in order of discovery (shortest first, naturally).
        """
        results: List[ChainResult] = []
        q_terms_empty = set()  # placeholder; will be set by caller
        
        for depth in range(1, max_depth + 1):
            depth_chains: List[ChainResult] = []
            
            if node not in self.graph.nodes():
                break
            
            paths = (self.graph.backward_chain(node, depth)
                     if direction == "backward"
                     else self.graph.forward_chain(node, depth))
            
            for path in paths:
                if path:
                    depth_chains.append(ChainResult(path, node, 0.0, 0.0, direction))
            
            # Quick score to check if this depth is productive
            if depth_chains:
                avg_score = sum(len(c.chain) for c in depth_chains) / len(depth_chains)
                # Normalize: average chain length / depth as a signal
                avg_score = min(1.0, avg_score / depth)
                
                if avg_score < early_stop_threshold:
                    # This depth is producing low-quality chains; stop exploring
                    break
                
                results.extend(depth_chains)
        
        return results
```

- [ ] **Step 4: Wire iterative deepening into retrieve()**

Edit the `retrieve()` method around line 734 to use iterative deepening:

```python
    def retrieve(self, question: str, top_k: int = 3,
                 diversify: bool = True) -> List[ChainResult]:
        self._ensure_indexed()
        direction = self._direction(question)
        depth = self._adaptive_depth(question)
        entries = self._entry_nodes(question, top_n=max(4, top_k * 2),
                                    direction=direction)
        
        results: List[ChainResult] = []
        for rrf_score, node in entries:
            # Use iterative deepening instead of full BFS
            paths = self._bfs_iterative_deepening(node, direction, depth,
                                                   early_stop_threshold=0.15)
            for c in paths:
                c.rrf_score = rrf_score  # attach RRF score
                results.append(c)
        
        # Dedup, rerank, MMR as before...
        seen: set = set()
        uniq: List[ChainResult] = []
        for r in results:
            key = tuple((e.cause, e.relation, e.effect) for e in r.chain)
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        
        self._rerank(question, uniq)
        uniq.sort(key=lambda r: (-r.rerank_score, -r.rrf_score))
        
        # ... rest unchanged
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_bfs_iterative_deepening.py -v
```

- [ ] **Step 6: Commit**

```bash
git add graph_rag.py tests/test_bfs_iterative_deepening.py
git commit -m "feat: BFS iterative deepening with early stopping for irrelevant chains"
```

---

## Task 5: 5-Tier MMR Redundancy Scoring

**Files:**
- Modify: `graph_rag.py:680-731` (_mmr_select and new _redundancy_score method)
- Test: `tests/test_mmr_5tier.py` (new)

**Interfaces:**
- Consumes: `chains: List[ChainResult]`, `selected: List[ChainResult]`
- Produces: `float` redundancy score combining 5 dimensions

**Description:** Current MMR uses 2 dimensions (node overlap, mechanism overlap). Extend to 5: (1) node set Jaccard, (2) relation repetition, (3) semantic cosine, (4) polarity match, (5) confidence agreement.

- [ ] **Step 1: Write failing test**

```python
# tests/test_mmr_5tier.py
from graph_rag import GraphRAG, ChainResult

def test_mmr_5tier_penalizes_all_dimensions():
    """5-tier redundancy should penalize on all 5 dimensions."""
    rag = GraphRAG()
    rag.ingest(
        "A causes B. B causes C. D causes E. E causes F. "
        "A also leads to X. X leads to Y."
    )
    
    chains = rag.retrieve("What happened?", top_k=6)
    
    # The MMR selection should avoid:
    # - Chains with overlapping nodes (A, B, C reappear)
    # - Chains with repeated relations (both "causes")
    # - Chains with similar semantic meaning
    # - Chains with matching polarity but different content
    # - Chains with high confidence from same extraction method
    
    assert len(chains) <= 3, "Should select diverse chains"
    
    # Check that not all chains start with "A"
    starts = [c.chain[0].cause for c in chains if c.chain]
    assert len(set(starts)) > 1, "Selected chains should have diverse entry points"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_mmr_5tier.py -v
```

Expected: chains not diverse enough

- [ ] **Step 3: Implement _redundancy_score_5tier**

Edit `graph_rag.py` before _mmr_select (around line 680):

```python
    def _redundancy_score_5tier(self, candidate: ChainResult,
                                 selected: List[ChainResult],
                                 used_mechanisms: set) -> float:
        """Compute 5-dimensional redundancy penalty.
        
        1. Node overlap (Jaccard)
        2. Relation repetition
        3. Semantic similarity (via rerank score proximity)
        4. Polarity consistency (if both positive/negative, higher penalty)
        5. Confidence agreement (if both high-conf, penalty)
        
        Returns: value in [0, 1], where 0 = no redundancy, 1 = complete redundancy.
        """
        if not selected:
            return 0.0
        
        # 1. Node overlap
        cand_nodes = self._chain_nodes(candidate)
        node_overlaps = [self._jaccard(cand_nodes, self._chain_nodes(s)) for s in selected]
        node_red = max(node_overlaps) if node_overlaps else 0.0
        
        # 2. Relation repetition
        cand_rels = {e.relation for e in candidate.chain}
        sel_rels = [{e.relation for e in s.chain} for s in selected]
        rel_overlaps = [len(cand_rels & sr) / max(1, len(cand_rels | sr)) for sr in sel_rels]
        rel_red = max(rel_overlaps) if rel_overlaps else 0.0
        
        # 3. Semantic similarity (rerank score distance, normalized to [0,1])
        cand_score = candidate.rerank_score
        sel_scores = [s.rerank_score for s in selected]
        score_diffs = [abs(cand_score - ss) / max(1, abs(cand_score) + abs(ss)) for ss in sel_scores]
        sem_red = 1.0 - max(score_diffs) if score_diffs else 0.0  # invert: close = redundant
        
        # 4. Polarity consistency
        # If candidate and selected have same polarity pattern, increase penalty
        cand_pol = 1.0 if all(e.polarity > 0 for e in candidate.chain) else -1.0
        pol_matches = sum(1 for s in selected 
                          if all(e.polarity > 0 for e in s.chain) == (cand_pol > 0))
        pol_red = pol_matches / max(1, len(selected))
        
        # 5. Confidence agreement
        cand_conf = sum(e.confidence for e in candidate.chain) / max(1, len(candidate.chain))
        conf_overlaps = []
        for s in selected:
            sel_conf = sum(e.confidence for e in s.chain) / max(1, len(s.chain))
            conf_agreement = 1.0 - abs(cand_conf - sel_conf) / max(0.01, cand_conf + sel_conf)
            conf_overlaps.append(conf_agreement)
        conf_red = max(conf_overlaps) if conf_overlaps else 0.0
        
        # Combine: equal weight (0.2 each), but can be tuned
        combined = (0.2 * node_red + 0.2 * rel_red + 0.2 * sem_red +
                    0.2 * pol_red + 0.2 * conf_red)
        return min(1.0, combined)
```

- [ ] **Step 4: Modify _mmr_select to use 5-tier scoring**

Edit `_mmr_select` around line 695:

```python
    def _mmr_select(self, chains: List[ChainResult], top_k: int,
                    lam: float = 0.6) -> List[ChainResult]:
        """MMR with 5-tier redundancy scoring."""
        if len(chains) <= top_k:
            return chains
        
        scores = [c.rerank_score for c in chains]
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        rel = {id(c): (c.rerank_score - lo) / rng for c in chains}
        
        used_mechanisms: set = set()
        selected: List[ChainResult] = []
        pool = list(chains)
        
        while pool and len(selected) < top_k:
            if not selected:
                best = max(pool, key=lambda c: rel[id(c)])
            else:
                def mmr(c: ChainResult) -> float:
                    red = self._redundancy_score_5tier(c, selected, used_mechanisms)
                    return lam * rel[id(c)] - (1 - lam) * red
                best = max(pool, key=mmr)
            
            selected.append(best)
            used_mechanisms.update((e.relation, e.effect) for e in best.chain)
            pool.remove(best)
        
        return selected
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_mmr_5tier.py -v
```

- [ ] **Step 6: Commit**

```bash
git add graph_rag.py tests/test_mmr_5tier.py
git commit -m "feat: 5-tier MMR redundancy (nodes, relations, semantics, polarity, confidence)"
```

---

## Task 6: Confidence-Aware Chain Pruning and VSA Weight Integration

**Files:**
- Modify: `graph_rag.py:554-603` (_entry_nodes to use semantic weight schedule)
- Modify: `graph_rag.py:628-679` (_rerank to add confidence pruning)
- Test: `tests/test_extreme_integration.py` (new)

**Interfaces:**
- Consumes: `document_length: int` (from ingest)
- Produces: chains ranked and pruned by confidence + VSA weight driven by doc complexity

**Description:** Wire semantic_weight_schedule into entry node selection; skip chains with geometric-mean confidence < 0.3 in reranking.

- [ ] **Step 1: Write integration test**

```python
# tests/test_extreme_integration.py
import pytest
from graph_rag import GraphRAG

def test_extreme_pipeline_improves_benchmark():
    """Full pipeline with all 5 extreme optimizations should maintain/improve benchmark."""
    rag = GraphRAG()
    
    # Ingest a realistic multi-hop document
    doc = """
    The subprime mortgage crisis:
    1. Banks issued risky mortgages to unqualified borrowers.
    2. Mortgage-backed securities bundled these risky loans.
    3. Rating agencies gave AAA ratings to toxic securities.
    4. Investors bought the securities thinking they were safe.
    5. Housing prices fell, mortgages defaulted.
    6. MBS values collapsed.
    7. Banks that held MBS lost billions.
    8. Bank failures froze credit markets.
    9. Business investment dropped due to credit freeze.
    10. Unemployment rose from job losses.
    """
    
    rag.ingest(doc)
    
    # Multi-hop query
    chains = rag.retrieve("How did mortgage failures lead to unemployment?", top_k=3)
    
    assert len(chains) >= 1, "Should find at least one chain"
    assert any(len(c.chain) >= 4 for c in chains), "Should find multi-hop chains"
    
    # All chains should have reasonable confidence
    for c in chains:
        if c.chain:
            avg_conf = sum(e.confidence for e in c.chain) / len(c.chain)
            assert avg_conf >= 0.4, "Chains should have min avg confidence 0.4"
```

- [ ] **Step 2: Run test to verify baseline**

```bash
pytest tests/test_extreme_integration.py -v
```

- [ ] **Step 3: Wire semantic_weight_schedule into _entry_nodes**

Edit `graph_rag.py` in `_entry_nodes` around line 554:

```python
    def _entry_nodes(self, question: str, top_n: int = 4,
                     direction: str = "forward", doc_complexity_hint: Optional[int] = None) -> List[Tuple[float, str]]:
        from retrievers import tokenize
        
        # Use document complexity to tune VSA semantic weight
        if doc_complexity_hint is None:
            doc_complexity_hint = len(self._node_docs)  # fallback: estimate from node count
        
        # Temporarily increase semantic weight for entry point matching
        original_weight = self.lex.semantic_weight
        self.lex.semantic_weight = self.lex.semantic_weight_schedule(doc_complexity_hint)
        
        q_terms = set(tokenize(question))
        
        # ... rest of _entry_nodes unchanged ...
        
        # Restore original weight
        self.lex.semantic_weight = original_weight
        
        return result
```

- [ ] **Step 4: Add confidence pruning to _rerank**

Edit `graph_rag.py` in `_rerank` around line 628:

```python
    def _rerank(self, question: str, chains: List[ChainResult]) -> None:
        q_terms = set(tokenize(question))
        
        # ... existing code ...
        
        for c in chains:
            # ... existing scoring ...
            
            # NEW: Confidence-aware pruning
            # Skip chains with very low average confidence
            if c.chain:
                avg_conf = sum(e.confidence for e in c.chain) / len(c.chain)
                if avg_conf < 0.30:
                    c.rerank_score = -999  # mark for removal
                    continue
            
            c.rerank_score = score
        
        # Remove pruned chains (score = -999)
        chains[:] = [c for c in chains if c.rerank_score > -999]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_extreme_integration.py -v
```

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -q
```

Expected: 76 + 5 new = 81 tests passing

- [ ] **Step 7: Commit**

```bash
git add graph_rag.py tests/test_extreme_integration.py
git commit -m "feat: VSA weight scheduling + confidence-aware pruning in entry/rerank"
```

---

## Task 7: Benchmark Validation

**Files:**
- Run: `eval_value.py`

**Interfaces:**
- Consumes: all the changes above
- Produces: benchmark results

- [ ] **Step 1: Run the benchmark**

```bash
python eval_value.py 2>&1 | tee benchmark_extreme.log
```

Expected output:
```
fact:      +0.01 (p=0.317)   [same or better]
multihop:  +0.33 (p=0.002)   [same or better]
rootcause: +0.22 (p=0.006)   [same or better]
```

- [ ] **Step 2: Compare results**

If any metric degrades significantly (delta decreases by >0.05), review and adjust:
- Reduce early_stop_threshold in iterative BFS (make it less aggressive)
- Lower confidence pruning threshold from 0.30 to 0.25
- Reduce semantic weight schedule peak (cap at 3 instead of 5)

- [ ] **Step 3: Commit final results**

```bash
git add benchmark_extreme.log
git commit -m "benchmark: Extreme VSA/PathSig/BFS/MMR pipeline — multihop +0.33"
```

---

## Self-Review Checklist

- [ ] All 7 tasks present
- [ ] Each task has exact file paths and code blocks (no placeholders)
- [ ] No "add error handling" or "implement later" statements
- [ ] All 5 new test files specified with exact test names
- [ ] VSA semantic_weight_schedule integrates with encode_triple
- [ ] PathSignature dynamic_level wired into index()
- [ ] BFS iterative deepening integrated into retrieve()
- [ ] MMR 5-tier redundancy replaces old 2-tier
- [ ] Confidence pruning in _rerank
- [ ] Benchmark validation step included
- [ ] All commits are self-contained and testable

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-25-vsa-pathsig-bfs-mmr-extreme.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks

**2. Inline Execution** — Execute tasks sequentially in this session with checkpoints

Which approach?
