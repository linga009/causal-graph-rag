# Bug Audit — 5 passes (2026-06-27)

Five critical-audit passes over the codebase for correctness, performance, and
efficiency, run while the multi-model benchmark executed. Severity: **HIGH**
(ships wrong results / data loss), **MED** (degrades quality silently),
**LOW** (inefficiency / fragile-but-works / cosmetic). Status: ✅ fixed,
📝 documented for cleanup.

Summary: 10 findings — 1 HIGH (fixed), 1 MED, 8 LOW. Plus 3 efficiency bugs
fixed earlier this session (proposition query-time encode, redundant query
encoding, dpp dead no-op).

---

## Pass 1 — Retrieval orchestration (`graph_rag.retrieve`)

**B1 [LOW, 📝] Bridge 2nd-pass uses the wrong slice boundary.**
`for r in results[len(uniq):]` assumes the new 2nd-pass chains begin at index
`len(uniq)` of the raw `results` list — but `results` (pre-dedup, with dupes) is
longer than `uniq` (deduped), so `len(uniq)` is *not* the pre-2nd-pass length of
`results`. It re-scans already-deduped entries; only correct because the `seen`
set filters them. Also `extra: List[ChainResult] = []` is declared and never used.
*Fix:* capture `n_before = len(results)` immediately before the 2nd-pass loop and
iterate `results[n_before:]`; delete `extra`.

**B2 [LOW, 📝] Bridge "added new chains?" guard doesn't test that.**
`if uniq[len(top_chains):]:` (with `top_chains = uniq[:3]`) is truthy whenever the
*first* pass already produced >3 chains, regardless of whether the 2nd pass added
anything — so the re-rank fires unnecessarily. The comment claims "re-rank only if
new chains were added." *Fix:* compare `len(uniq)` before vs after the 2nd-pass
append.

---

## Pass 2 — Causal extraction (`causal_extractor.py`)

**B4 [MED, 📝] `_infer_polarity` misses punctuation-attached suppression words.**
```python
words = set(effect_sentence.lower().split())     # NOT punctuation-stripped
return -1 if words & _SUPPRESS_WORDS else +1
```
"...were reduced." → token `"reduced."` ∉ `_SUPPRESS_WORDS`, so a suppressive edge
is mislabeled polarity **+1**. `_implicit_edges` strips punctuation
(`w.rstrip(".,;:!?")`) but `_infer_polarity` doesn't — inconsistent. Polarity is a
core signal (promote vs inhibit) used in VSA encoding and chain rendering, so this
silently corrupts some negative-causation edges. *Fix:* strip punctuation before
membership test (reuse the `_implicit_edges` tokenization).

**B3 [LOW, 📝] Duplicate keys in `CAUSAL_VERBS`.**
`improve / improves / improved` are defined twice (≈ lines 51 and 69) with
identical values — the later silently overwrites the earlier. Harmless but dead.
*Fix:* delete the duplicate block.

---

## Pass 3 — Graph & VSA (`causal_graph.py`, `vsa_core.py`)

**B5 [LOW, 📝] `_ppr_scores` builds a dense N×N matrix per query.**
O(N²) memory/time — fine for the per-document graphs here (hundreds of nodes),
but unusable for large/Neo4j-backed graphs. The component was screened and
**dropped** (flag off), so not live; if ever revived, use `scipy.sparse` + sparse
mat-vec. Same applies to the other dropped components (`logsig`, `holo`, `beam`,
`dpp`) — all dead code to remove in cleanup.

*Confirmed-not-a-bug:* `CausalGraph.path_between` marks visited on enqueue (correct
BFS shortest path); `_bfs` per-node cap correctly limits hub flooding;
`score_edges_by_triple` int32 matmul is sound.

---

## Pass 4 — Retrievers & indexing (`retrievers.py`, `_ensure_indexed`, `_retrieve_sentences`)

**B7 [LOW, 📝] `_calibrated_fuse` zeroes single-element / zero-variance channels.**
Z-normalizing a channel with one node gives `z=(s-mu)/sd = 0`, so a strong **exact
name-match** (the `direct` channel, often a single node) contributes **nothing**.
It still net-helped in screening, but it discards a high-precision signal.
*Fix:* use min-max (absence=0, weak=small-positive, exact-match preserved) or give
single-element channels a fixed positive contribution. **Re-screen next cycle.**

**B6 [LOW, 📝] `_ensure_indexed` re-encodes everything on every ingest.**
When `_dirty`, it re-encodes all node docs, all sentences, and all edge source
sentences. For repeated incremental `ingest()` calls this is O(total) repeated each
time. Fine for one-shot ingest; for streaming ingestion, cache embeddings by id.

**B8 [LOW, 📝] Retriever-internal query encodes bypass the memo.**
`SentenceTransformerDense.score()` and `PathSignatureRetriever.score()` each encode
the query inside `retrievers.py`, outside `_encode_query`'s memo → 2 redundant
CPU encodes per query (~40–60 ms). *Fix:* thread an optional precomputed query
vector into both `.score()` methods.

---

## Pass 5 — API / CLI / integration

**B9 [HIGH, ✅ FIXED] `/query` shipped degraded chain-only answers.**
The production endpoint did:
```python
chains = _rag.retrieve(q, top_k)
answer = _rag.generate(q, chains)          # no score gate, no coverage_sentences
```
This is the **same defect** that scored **−0.42 on fact questions** in the buggy
benchmark — bypassing the score gate and the hybrid coverage sentences. The shipped
API was returning answers materially worse than the system's real `answer()`.
*Fix applied:* replicate `answer()`'s prep under the lock (score gate + hybrid
coverage retrieval), then call `generate(..., coverage_sentences=coverage)` outside
the lock (preserving the concurrency split). Verified by the corrected benchmark
(fact +0.15/+0.17) which uses the same path.

*Confirmed-not-a-bug:* `cli.py` query/ask use `rag.answer()` (correct).
`langchain_integration.build_rag_tool` uses `rag.answer()` (correct).

---

## Already fixed earlier this session (efficiency)

| id | issue | fix |
|---|---|---|
| E1 | Proposition rerank ran the embedder **per query** (+1.2 s) | precompute edge-sentence embeddings at ingest; query-time = cached dot product |
| E2 | Query embedded 4–5× per `answer()` | `_encode_query` memo (one encode per query) |
| E3 | Dead `for t in range(it): pass` no-op in `_dpp_select` | removed |

---

## Recommended cleanup commit (after promotion)

1. Apply B1, B2 (bridge-pass boundary + guard).
2. Apply B4 (polarity punctuation strip) — quality-affecting, worth a quick re-bench.
3. Apply B3 (dup verb keys).
4. Delete dropped components + flags: `logsig`, `chain_holo`, `beam`, `dpp`, `ppr`
   and their screen configs (B5). Keep only `proposition` + `calibrated_fusion`.
5. B7 (calibrated min-max) → re-screen free before shipping.
6. B8 (thread q_emb into retrievers) — optional perf.

---

# Deep audit — round 2 (passes 6–10)

Five further passes on areas not covered in round 1: persistence, ingest dedup,
Neo4j parity, VSA internals, concurrency. 5 findings (2 MED, 3 LOW); 1 fixed.

## Pass 6 — Persistence (`save`/`load`)

**B11 [MED, ✅ FIXED] `load()` didn't restore the dedup set → duplicate edges.**
`save()` does not persist `_edge_set`, and `load()` appends edges straight into
`graph._edges`/`out_adj`/`in_adj` without repopulating it. So `_edge_set` was empty
after load, and any `ingest()` *after* `load()` re-added edges already present
(the `key in self._edge_set` guard always missed). Pure query-after-load was fine;
load-then-ingest silently duplicated edges. *Fix applied:* rebuild
`rag._edge_set = {(e.cause, e.relation, e.effect) for e in rag.graph.edges}` in
`load()`.

## Pass 7 — Ingest (`GraphRAG.ingest`)

**B14 [LOW, 📝] `_node_docs` built by repeated string `+=`; substring dup-check.**
`self._node_docs[node] += " " + e.source_sent` for a hub node touched by many edges
is O(total²) in string length, and the guard `if e.source_sent not in self._node_docs[node]`
is a *substring* test — a short source sentence that is a substring of an existing
longer one is silently dropped. *Fix:* accumulate per-node sentence **lists** with a
`set` membership check, join once at index time.

## Pass 8 — Neo4j parity (`neo4j_graph.py`)

**B17 [LOW/note, 📝] New graph components silently no-op on the Neo4j backend.**
`_ppr_scores` and `_beam_chains` guard on `hasattr(g, "out_adj")`; the in-memory
`CausalGraph` exposes `out_adj`/`in_adj` attributes, but `Neo4jCausalGraph` exposes
`_get_out_adj()`/`_get_in_adj()` methods instead — so those components disable
themselves on Neo4j. Harmless today (both are dropped components), but if PPR is ever
revived it must call the backend-agnostic accessors. The **shipping** components are
fine on Neo4j: `proposition` uses `graph.edges` (implemented) and `calibrated_fusion`
is backend-agnostic.

## Pass 9 — VSA encoding (`causal_graph.add_edge`)

**B15 [WITHDRAWN — invalid finding].**
Original claim: "VSA encoding ignores edge polarity; the polarity-aware branch in
`encode_triple` is dead." **This was wrong.** `vsa_core.Triple` has exactly three
fields `(agent, action, patient)` and `encode_triple` has **no** polarity logic —
the polarity-aware encoding was a *proposed* feature in an abandoned plan, never
shipped. I conflated the plan with the code. Attempting the "fix"
(`Triple(..., e.polarity)`) raised `TypeError: Triple takes 4 positional arguments
but 5 were given` — caught immediately by tests, reverted. No code change.
Lesson: verify the claimed-buggy line exists in the actual code before writing it up.
(Adding polarity to the VSA encoding remains a *possible enhancement*, not a bug.)

## Pass 10 — Concurrency / edge cases

**B16 [LOW, 📝] `_q_emb_cache` single-entry memo is not thread-safe.**
The shared `_rag` in `api.py` is used across threadpool workers. The memo
(`self._q_emb_cache`) is read-modified without a lock; two concurrent queries with
different questions could interleave and one could read the other's embedding.
**Safe as shipped** — every `_encode_query` call happens under `_rag_lock`
(retrieve and `_retrieve_sentences` are inside the lock; only the LLM `generate` runs
outside). But it's a latent hazard for any direct multithreaded use of `GraphRAG`
without external locking. *Fix:* key the memo per-call or drop it under concurrency.

*Confirmed-clean:* empty-graph and single-sentence ingest degrade gracefully
(`answer()` → "No relevant information found"); `clear_graph()` global swap under lock
is safe; `save()` correctly omits the (rebuildable) model + indices.

---

## Consolidated fix queue (cleanup commit, then re-screen + re-bench)

Quality-affecting (need re-screen): **B4** (done), **B7** (done), **B15** (VSA polarity).
Correctness: **B9** (done), **B11** (done), **B1/B2** (done).
Hygiene/perf: **B3** (done), **B14**, **B16**, **B17**, **B8**, **B6**; delete dropped
components (B5).

---

# Dynamic debug — round 3 (passes 11–20, code EXERCISED not just read)

Ran 10 edge-case categories against live code. **9/10 robust**; 1 real bug found.

## Robustness confirmed (no crashes, graceful degradation)
- **P11** empty / whitespace / single-word / no-causality docs → `answer()` returns
  "No relevant information found", 0 chains. Clean.
- **P12** `parse_triples` on empty/`"???"`/500-char/question/em-dash inputs → no crash.
- **P14** entity normalization merges "pump"/"cooling pump"/"pump failure" sensibly.
- **P15** coverage with `k > num_sentences`, retrieve on a no-match query → safe.
- **P17** ✅ **B11 fix verified**: save → load → re-ingest the same text produces
  **no duplicate edges** (3 → 3). The dedup-set restore works.
- **P18** LLM extractor on non-JSON garbage → 0 edges, no crash.
- **P19** coverage with empty / populated `chain_nodes` → safe.
- **P20** agentic controller on an EMPTY graph → terminates with an answer, no crash.

## Bug found

**B18 [MED, 📝] Unicode tokenization mangles non-ASCII text.**
`tokenize` (and its three siblings) use `re.compile(r"[a-z0-9]+")` on lowercased
text, so accented characters act as delimiters:
`"défaillance"` → `["d", "faillance"]`, `"café"` → `["caf"]`, `"naïve"` → `["na","ve"]`.
The mangling is *consistent* across query and document sides, so matching partially
survives, but precision drops (content words split; junk single-char tokens like
`"d"` appear) for any non-English / accented corpus. Found in **4 files**:
`retrievers.py:46`, `causal_graph.py:26`, `langchain_integration.py:59`,
`neo4j_graph.py:39`. *Fix:* `re.compile(r"\w+", re.UNICODE)` (keeps accented words
whole) or accent-fold via `unicodedata.normalize`. Changes tokenization → re-screen.
Current 23-doc corpus is English, so this does not affect the reported numbers.

*Note:* P13's failure was a probe-harness error (wrong import name), not a code bug;
`doc_structure` parses empty/heading-less/malformed markdown fine (schema=auto passed).
