# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org).

## [0.3.0] — 2026-06-28

First PyPI release. RAG that traverses cause→effect chains instead of returning
similarity-matched chunks — strong multi-hop and root-cause retrieval with **no
query-time LLM**.

### Added
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
