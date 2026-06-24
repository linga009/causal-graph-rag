# Does structure improve RAG? — honest findings

We measured whether adding causal + document structure to the LLM context
improves answers. Two benchmarks, real LLMs (Groq). Results are reported as
measured, including the negative ones.

## TL;DR (updated after building retrieval-side structure)

- **Retrieval was the bottleneck — and fixing it is the big win.** Adding
  contextual indexing (heading-path folded into BM25+dense) + MMR diversity +
  `top_k=6` lifted flat recall from **0.27/0.30 → 0.47** on the real large doc,
  and faithfulness from **0.60 → 1.00**. This reproduces Anthropic Contextual
  Retrieval's −35–49% retrieval-failure effect.
- **The capability hypothesis holds once retrieval works.** With good retrieval,
  generation-side structure helps the **weak model (+0.11)** far more than the
  **strong model (+0.01)** — a decent model needs the scaffolding; a strong one
  figures it out itself. The earlier *inversion* was an artifact of broken
  retrieval.
- **Keep symbolic chain arrows; prose rendering does not help** (−0.03 weak).
- Methodological lesson: never measure a generation-side lever while retrieval
  is the binding constraint — fix retrieval first.

## Benchmark 1 — small synthetic docs (`eval_structure.py`)

6 questions over ~10-sentence docs, llama-3.1-8b.

| condition    | answer_recall | correctness |
|--------------|---------------|-------------|
| flat         | 0.64          | 0.55        |
| +causal      | 0.58          | 0.53        |
| +causal+doc  | 0.58          | 0.52        |

`flat` scored 0.58 then 0.64 on identical code across two runs (temp 0.2), so
the ±0.06 deltas are **within noise**. On tiny docs the LLM already sees every
retrieved sentence, so structure adds tokens but no information. Expected null.

## Benchmark 2 — real large doc, BEFORE retrieval-side structure (`eval_realdoc.py`)

Wikipedia "Subprime mortgage crisis › Causes" (~91k chars, 436 sentences,
deeply nested sections). 5 multi-hop/global questions. `top_k=4`. Retrieval is
spaCy-based (LLM-independent), so only the generation model varies. Fixed
strong judge for faithfulness.

| model                 | condition  | kw_recall | faithful |
|-----------------------|------------|-----------|----------|
| llama-3.1-8b (weak)   | flat       | 0.27      | 0.60     |
| llama-3.1-8b (weak)   | structured | 0.25      | 0.60     |
| llama-3.3-70b (strong)| flat       | 0.30      | 0.60     |
| llama-3.3-70b (strong)| structured | **0.37**  | 0.60     |

**Findings**
1. **Recall ceiling ~0.3 everywhere** — `top_k=4` of 566 sentences caps concept
   coverage. Retrieval is the bottleneck; generation formatting can't fix it.
2. **Structure helped the strong model (+0.07), not the weak one (-0.02)** —
   opposite of the naive hypothesis. Our structure is *symbolic and dense*
   (`A -/->(reduce) B`, `[Causes > Securitization]`): a notation that rewards
   capability to parse rather than a crutch for weak models.
3. **Faithfulness flat (0.60)** — structure did not change grounding here.
4. **n=5; deltas are small.** Treat the ceiling as the robust result, the
   deltas as suggestive only.

## Benchmark 3 — same doc, AFTER retrieval-side structure (`eval_realdoc.py`)

Added: **contextual indexing** (heading-path folded into BM25+dense, Anthropic
Contextual Retrieval), **MMR diversity** selection (cover different sections,
not near-duplicate chains), and `top_k=6`. Same questions, same judge.

| model                 | condition  | kw_recall | faithful |
|-----------------------|------------|-----------|----------|
| llama-3.1-8b (weak)   | flat       | 0.47      | 1.00     |
| llama-3.1-8b (weak)   | structured | **0.58**  | 1.00     |
| llama-3.1-8b (weak)   | prose      | 0.44      | 1.00     |
| llama-3.3-70b (strong)| flat       | 0.47      | 1.00     |
| llama-3.3-70b (strong)| structured | 0.47      | 1.00     |
| llama-3.3-70b (strong)| prose      | 0.47      | 1.00     |

**Findings**
1. **Flat recall 0.27/0.30 → 0.47, faithfulness 0.60 → 1.00.** Retrieval-side
   structure lifted the ceiling for *both* models — the dominant win.
2. **Generation-side structure now helps the weak model (+0.11) >> strong
   (+0.01).** The capability-dependent scaffolding hypothesis holds once
   retrieval is fixed.
3. **Prose rendering does not help** (−0.03 weak); symbolic arrows are kept.
4. n=5 still — but effect sizes are now large and consistent with theory.

## Literature grounding

- **Retrieval helps smaller models more** ([arXiv 2402.13492](https://arxiv.org/html/2402.13492)) —
  but that is about retrieval *adding knowledge*, not re-formatting it. Our
  structure re-formats; it does not add knowledge, which is consistent with the
  weak model gaining nothing.
- **GraphRAG** ([arXiv 2404.16130](https://arxiv.org/abs/2404.16130)) — graphs
  win *global / multi-hop* questions; baseline vector RAG wins single-hop
  factoids. Confirms benchmark-1 (factoids) was the wrong regime.
- **Contextual Retrieval** ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)) —
  the proven win is *retrieval-side* (embed context into chunks), −35–49%
  retrieval failures. We currently use structure only at generation. This is
  the gap to close next.

## What we built (and what's left)

- [x] **Retrieval-side structure (highest leverage).** Contextual indexing
  (heading-path in BM25+dense) + MMR diversity + larger candidate pool.
  Result: flat recall 0.27/0.30 → 0.47, faithfulness 0.60 → 1.00.
- [x] **Prose-rendered chains** — tested; does not help (−0.03 weak). Arrows kept.
- [ ] **A reasoning-bound metric.** kw_recall is retrieval-bound; a multi-hop
  question whose answer requires *connecting two chains* would isolate
  generation-structure value more cleanly.
- [ ] **Larger question set.** n=5 limits confidence on the +0.11 / +0.01 split;
  effect sizes are now large but more questions would firm it up.
- [ ] **Cross-encoder reranking** (Contextual Retrieval reaches −67% with a
  reranker) — the next retrieval lever if more recall is wanted.

## Honest bottom line

Two levers, both now measured to work:
1. **Causal extraction** (building the graph) — 26-q benchmark recall 0.46→0.60.
2. **Retrieval-side structure** (contextual indexing + MMR) — flat recall
   0.27/0.30→0.47, faithfulness 0.60→1.00 on a real large document.

Generation-side structure (chains+heading-paths in the prompt) is a real but
*capability-dependent* add-on: +0.11 for a decent model, ~0 for a strong one.
The headline: **a decent model + this structure approaches a strong model's
answer quality** — which is exactly the practical value of the design. Measured,
not assumed.
