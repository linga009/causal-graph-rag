# Research Notes — Components to Improve Causal Graph RAG

Date: 2026-06-27. Constraint lens for every idea below: **no query-time LLM
calls**, small-model focus, in-memory/CPU-friendly, algorithmic retrieval.
Sources are arXiv IDs + links at the bottom.

## The one finding that recurs everywhere

**Triple extraction is lossy.** HippoRAG2, LightRAG, and the GraphRAG survey all
note that compressing context-dependent relations into atomic `(s, r, o)` triples
discards nuance. This confirms our own diagnosis: **extraction quality is the
bottleneck**, and we under-use the full source sentence (the "proposition")
that each edge already carries. Several cheap wins below exploit that.

---

## TIER 1 — highest value, perfect constraint fit (pure graph algorithms)

### 1. Personalized PageRank retrieval (HippoRAG) ⭐ top pick
- **What:** build seed set from query (our entry nodes + query-term node matches);
  run **Personalized PageRank / random-walk-with-restart** with teleport mass on
  the seeds; rank ALL nodes by stationary PPR mass. Nodes reachable through *many*
  paths from the seeds score high — this is associative **multi-hop** relevance,
  computed with zero LLM. HippoRAG reports up to **+20% on multi-hop QA**.
- **Why it fits us perfectly:** PPR is a power-iteration on a sparse matrix —
  cheap, CPU, deterministic, no query-time LLM. It is the canonical LLM-free
  multi-hop retrieval method.
- **Our novel twist (beyond HippoRAG):** weight the transition matrix by our
  **edge confidence** and make it **direction-aware** (forward edges for impact
  queries, reverse for root-cause). HippoRAG uses an undirected entity graph; we
  have a *directed, confidence-scored causal* graph — PPR on it is strictly more
  informative.
- **How it plugs in:** new `flag_ppr`. After entry-node selection, run PPR
  (~20 iterations) seeded on entries; add a chain score term = mean PPR mass of
  the chain's nodes; optionally weight coverage sentences by PPR mass of the
  chain nodes they mention. Screen with the free concept-coverage proxy first.
- **Effort:** ~40 lines (sparse matrix from `out_adj`/`in_adj`, power iteration).
  Refs: HippoRAG 2405.14831; Context-Aware Traversal 2602.01965; SeedER 2605.23753.

### 2. Calibrated channel fusion (replace/augment RRF) ⭐
- **What:** "Calibrated Fusion for Heterogeneous Graph-Vector Retrieval"
  (2603.28886) shows graph scores (PPR) and dense scores have different
  distributions and are **not directly comparable**. We currently use RRF
  (rank-only), which sidesteps scale but **throws away score magnitude**.
- **Improvement:** z-score or quantile-normalize each channel's raw scores, then
  weighted-sum — recovers the magnitude signal RRF discards while staying
  scale-robust. Cheap, algorithmic.
- **How it plugs in:** `flag_calibrated_fusion` in `_entry_nodes` — normalize the
  5 channel score lists, weighted-sum, compare against current RRF. ~25 lines.

### 3. Seeded propagation / SPRIG (CPU-only, linear)
- "Democratizing GraphRAG" (2602.23372): label propagation from seeds, linear,
  CPU-only, no LLM-heavy inference. A cheaper cousin of PPR — **PPR subsumes it**,
  so implement PPR (#1) and treat SPRIG as the budget fallback if PPR is too slow
  on >1M-node graphs. Ref: 2602.23372.

---

## TIER 2 — strong, small/no new dependency

### 4. Proposition-aware chain rerank (no new dep) ⭐ cheap win
- **What:** PropRAG (2504.18070) retrieves over **propositions** (richer atomic
  statements) instead of lossy triples, with beam search over proposition paths
  (validates our `flag_beam`). We already store each edge's `source_sent` but only
  rerank on node *names*.
- **Improvement:** add a rerank term = dense similarity between the query and the
  edge **source sentences** along the chain (we already embed all sentences, so
  this is near-free). Directly counters the "triples are lossy" problem by
  scoring on the full proposition text.
- **How it plugs in:** `flag_proposition_rerank` in `_rerank`. ~15 lines using
  the existing `_sent_vecs`. High value-per-line. Ref: PropRAG 2504.18070.

### 5. ColBERT-style late interaction for coverage sentences
- **What:** token-level **late interaction (MaxSim)** beats single-vector cosine
  on nuanced matches, at tens-of-ms latency, **no LLM**. Improves the coverage
  (flat) side of our hybrid.
- **Cost:** needs a ColBERT/ModernBERT model (~110 MB) — a new optional
  dependency. Keep off by default, opt-in extra. Ref: ModernBERT+ColBERT
  2510.04757; learnable late interaction 2406.17968.
- **Priority:** medium — high ceiling, but only worth it if Tier-1 plateaus.

---

## TIER 3 — structural / adaptive (medium effort)

### 6. Adaptive breadth + depth traversal
- "Autonomous KG Exploration with Adaptive Breadth-Depth" (2601.13969): tune both
  branching factor and depth per query. We already adapt *depth*; **adaptive
  breadth** (beam width / `_NODE_CAP` as a function of query type) is the missing
  half. Pairs naturally with `flag_beam`. Ref: 2601.13969.

### 7. LLM-free community structure (dual-level, à la LightRAG)
- LightRAG (2410.05779) / GraphRAG add a high-level "theme" layer. The summary
  step needs an LLM (violates our constraint), but **Louvain/Leiden community
  detection** over the causal graph is LLM-free and could add a coarse topic
  signal for global questions. Lower priority — summaries are where the value is,
  and those need an LLM. Refs: 2410.05779; GraphRAG survey 2408.08921.

---

## Recommended next experiments (in order)

1. **PPR retrieval** (`flag_ppr`, confidence-weighted + direction-aware) — screen
   free, then benchmark. Highest expected multi-hop/root-cause gain.
2. **Proposition-aware rerank** (`flag_proposition_rerank`) — cheapest win,
   attacks the lossy-triple bottleneck directly.
3. **Calibrated fusion** (`flag_calibrated_fusion`) — improves entry selection
   that everything downstream depends on.
4. Revisit ColBERT coverage + adaptive breadth only if 1–3 plateau.

All four are screenable for free with `eval_corpus/screen_components.py` before
spending a cent on LLMs — same discipline as the current component screen.

---

## Sources
- HippoRAG — https://arxiv.org/abs/2405.14831
- Context-Aware Traversal (Breaking the Static Graph) — https://arxiv.org/abs/2602.01965
- Calibrated Fusion for Heterogeneous Graph-Vector Retrieval — https://arxiv.org/abs/2603.28886
- Democratizing GraphRAG (SPRIG, CPU-only) — https://arxiv.org/abs/2602.23372
- SeedER (Seed-and-Expand) — https://arxiv.org/abs/2605.23753
- Autonomous KG Exploration (Adaptive Breadth-Depth) — https://arxiv.org/abs/2601.13969
- HopRAG (logic-aware multi-hop) — https://arxiv.org/abs/2502.12442
- PropRAG (beam over proposition paths) — https://arxiv.org/abs/2504.18070
- LightRAG — https://arxiv.org/abs/2410.05779
- Graph-Based Re-ranking survey — https://arxiv.org/abs/2503.14802
- Graph RAG survey — https://arxiv.org/abs/2408.08921
- CausalRAG (ACL 2025) — https://arxiv.org/abs/2503.19878
- ModernBERT + ColBERT biomedical reranking — https://arxiv.org/abs/2510.04757
- Learnable late interactions — https://arxiv.org/abs/2406.17968
