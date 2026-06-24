# Does structure improve RAG? — honest findings

We measured whether adding causal + document structure to the LLM context
improves answers. Two benchmarks, real LLMs (Groq). Results are reported as
measured, including the negative ones.

## TL;DR

- **Generation-side structure (showing chains + heading-paths to the LLM) is a
  second-order lever.** On small docs it was noise; on a large real doc it gave
  a small recall bump to a *strong* model and none to a *weak* one — the
  opposite of "scaffolding helps weak models."
- **The binding constraint is retrieval, not generation.** With `top_k=4` over
  566 sentences, answer recall is capped ~0.3 regardless of how the context is
  formatted. The highest-leverage work is *retrieval-side* structure
  (Anthropic Contextual Retrieval reports −35–49% retrieval failures from
  embedding context into the index — a lever we have **not** built yet).

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

## Benchmark 2 — real large doc, weak vs strong model (`eval_realdoc.py`)

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

## What to build next (prioritized)

1. **Retrieval-side structure (highest leverage).** Embed heading-path + a
   one-line situating context into each indexed unit; retrieve more/better
   chains. Attack the 0.3 recall ceiling directly. *This is where the
   literature says the gains are.*
2. **Prose-rendered chains.** Render `A -/->(reduce) B` as "A reduced B" in the
   prompt; re-test the weak model — does removing the parsing tax flip -0.02?
3. **A reasoning-bound metric.** A multi-hop question whose answer requires
   *connecting two retrieved chains*, to isolate generation-side structure value
   from retrieval coverage (kw_recall is retrieval-bound and a weak proxy).

## Honest bottom line

Causal **extraction** (building the graph) showed a real retrieval win earlier
(26-q benchmark: recall 0.46→0.60 with LLM-augment). Causal/document structure
**in the generation context** is, so far, a small and capability-dependent
effect — not the free win the idea suggests. The next real gains are on the
**retrieval** side, which we have not yet built. Measure, don't assume.
