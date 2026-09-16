# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org).

## [0.3.1] — 2026-09-17

Extraction-quality bug fixes, plus new causal-edge fields identified by
auditing real extracted edges against `eval_corpus` documents. No breaking
changes; new `CausalEdge`/`GraphEdge` fields are additive with defaults.

### Fixed
- `_event_head()` (used by implicit/forward/backward-connective chaining)
  produced sentence fragments as graph nodes (e.g. `"via mbs)"`, `"calm
  according"`) because its fallback had no notion of grammatical structure.
  Now prefers a spaCy noun-chunk lookup; the old word-scan is a last resort.
- `which`/`who`/`whom`/`whose` were missing from the pronoun blocklists,
  letting relative pronouns become graph nodes.
- `_org_edges()`'s regexes capped capture groups by **character** count
  (`[\w\s]{1,30}`), truncating mid-word (`"future value"` → `"future
  valu"`). Switched to word-count-bounded groups, which can't truncate
  mid-word by construction.
- `parser.py`'s no-spaCy fallback triple extractor (the *only* path CI
  exercises — `spacy` is an optional extra, not a core dependency)
  misidentified plural nouns ending in "-s"/"-es" as verbs (e.g.
  "eyewitnesses"), producing wrong triples. Restricted the suffix-guess
  fallback to "-ed", a reliable verb-only inflection.
- Added a single well-formedness chokepoint (`_is_well_formed_entity`,
  called from `_validate_edge`) rejecting bare numeric/percent/currency
  spans and unmatched-parenthesis fragments regardless of which of the
  four extraction methods produced them.

### Added
- **Schema-aware routing**: `extract_edges()`/`extract_edges_hybrid()` now
  accept `schema` (already threaded from `GraphRAG.ingest()`, previously
  unused past document-structure parsing) and skip the narrative-tuned
  implicit-adjacency and org-governance heuristics for `research`/
  `clinical` schemas, which misfire on method/definition prose.
- **Epistemic status + attribution**: `CausalEdge.epistemic_status`
  (`"stated"`/`"hypothesis"`/`"disputed"`) and `.attributed_to`, detected
  from hedge-phrase cues and two precision-guarded attribution patterns.
  Edges sharing an effect with disagreeing causes, where at least one is
  hedged, are marked `"disputed"` (e.g. two published competing
  explanations for the same historical event).
- **Magnitude capture**: `CausalEdge.magnitude` captures nearby percent/
  currency spans instead of discarding them or leaving bare numbers as
  cause/effect nodes.
- 38 new tests (`tests/test_extraction_quality.py`, `tests/test_parser.py`),
  each keyed to a real malformed edge found by running extraction against
  `eval_corpus` documents, not a synthetic worst case.

### Note on impact
A small paired comparison (12 multihop/rootcause questions across 3
`eval_corpus` documents, pre- vs post-fix, Haiku generation) showed **no
statistically significant downstream answer-quality change** (mean delta
+0.012, paired Wilcoxon p=0.73, n=12 — too small to detect anything but a
large effect). These are correctness fixes to extraction output, not a
demonstrated end-to-end RAG quality improvement; the full multi-field
benchmark numbers above were not rerun.

## [0.3.0] — 2026-06-28

First PyPI release. RAG that traverses cause→effect chains instead of returning
similarity-matched chunks — strong multi-hop and root-cause retrieval with **no
query-time LLM**.

### Added
- **MongoDB / MongoDB Atlas backend** (`MongoCausalGraph`, `pip install
  "causal-graph-rag[mongo]"`): causal edges stored as documents, native graph
  traversal via MongoDB's `$graphLookup` (`graph.reachable()` for impact /
  root-cause sets). Drop-in via `GraphRAG(mongo_uri=...)`; pairs with Atlas Vector
  Search for the dense channel.
- **Agentic mode** (opt-in): `AgenticCausalRAG`, a ReAct controller whose action
  space is the LLM-free graph tools (`rootcause` / `impact` / `path` / `retrieve`).
  CLI `causal-rag agent`; the default `answer()` path is unchanged.
- **Proper `causal_graph_rag` package** (was flat top-level modules), so installing
  no longer pollutes the global import namespace.
- **Multi-field benchmark harness** (`eval_corpus/`): 23 documents across 5 fields,
  138 typed questions, Haiku + Sonnet generation, free no-LLM component screen.
- LangChain `VSAGraphRetriever` hybrid mode + `build_graph_tools`; REST endpoints
  `/retrieve` `/rootcause` `/impact` `/path`; demo GIF + headless renderer.

### Changed
- **Relicensed under the PolyForm Noncommercial License 1.0.0** — free for personal,
  academic, and noncommercial use; commercial use requires a separate license
  (contact lingamraju26@gmail.com).
- **Two retrieval components promoted to default-on**, validated by free screen +
  benchmark: proposition-aware rerank (scores chains by their source sentences) and
  min-max calibrated channel fusion.
- Coverage-sentence retrieval is now hybrid BM25 + dense (RRF).

### Fixed
- **`/query` API** returned degraded chain-only answers (missing the score gate and
  hybrid coverage sentences); now uses the full `answer()` path.
- `load()` restores the edge-dedup set (no duplicate edges on load-then-ingest).
- Unicode-aware tokenization (`\w+`) so accented/non-ASCII words stay whole.
- Polarity inference strips trailing punctuation; bridge-pass boundary corrected.
- Packaging: include all shipped modules; precompute edge embeddings at ingest so
  retrieval stays ~300 ms (no per-query embedding model calls).

### Removed
- Six experimental components that were built, screened for free, and **dropped as
  empirically inert/negative** (real-embedding VSA, log-signature, VSA holography,
  beam search, DPP selection, Personalized PageRank).

### Benchmark (paired Wilcoxon vs a strong dense-RAG baseline)
| Question type | Haiku Δ | Sonnet Δ |
|---|---|---|
| Fact lookups | +0.12 | +0.17 |
| Multi-hop | +0.29 | +0.30 |
| Root-cause | +0.30 | +0.28 |

Wins every category on both models; positive in all five fields; advantage holds as
the model scales (helps cheap/local models most).

[0.3.0]: https://github.com/linga009/causal-graph-rag/releases/tag/v0.3.0
