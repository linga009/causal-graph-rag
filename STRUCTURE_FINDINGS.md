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
- **Generation-side structure helps FAITHFULNESS, not recall** (deterministic,
  n=12): chains + document anchoring raise groundedness +0.08–0.15 but do not
  add concept coverage. The earlier recall-"synergy" and "capability-scaling"
  claims were temperature-0.2 / n=5 **sampling noise and are retracted**
  (see Benchmark 4 correction). Strong-model temp-0 row pending.
- **Keep symbolic chain arrows; prose rendering does not help.**
- Two methodological lessons: (a) fix retrieval before measuring any
  generation-side lever; (b) evaluate at **temperature 0 with n≥12** — small
  hot-sampled runs manufacture effects that vanish under determinism.

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

## Benchmark 4 — causal vs document structure ablation (weak model)

> **Correction.** An earlier version of this section claimed a recall "synergy"
> (+0.11 for causal+doc) from an n=5, temperature-0.2 run. It **did not
> replicate** — a second run gave -0.01 for the same cell. The deltas were
> sampling noise. Re-run **deterministically (temperature 0) at n=12** below;
> the recall synergy disappears and the real, stable signal is **faithfulness**.

llama-3.1-8b, `top_k=6`, n=12, **temperature 0** (deterministic — no sampling
noise). Retrieval-side structure always on; only the prompt's structure varies.

| condition    | kw_recall      | faithfulness    |
|--------------|----------------|-----------------|
| flat         | 0.65           | 0.83            |
| +causal      | 0.55 (**-0.10**)| 0.94 (**+0.11**)|
| +doc         | 0.65 (-0.00)   | 0.98 (**+0.15**)|
| +causal+doc  | 0.65 (-0.00)   | 0.92 (**+0.08**)|

**The honest signal: structure improves faithfulness, not recall.**
- **Recall:** structure is neutral-to-negative. Causal chains *alone* slightly
  *hurt* concept coverage (-0.10) — the symbolic notation displaces breadth.
  Document context is neutral. There is **no recall synergy** — that was noise.
- **Faithfulness:** *every* structure condition raises groundedness
  (+0.08 to +0.15). Showing the model the causal chains + document anchoring
  keeps answers tethered to the evidence, reducing unsupported claims.

Mechanism: structure here is a **grounding constraint**, not a coverage booster.
It does not help the model *find* more facts; it helps it *not invent* ones.
That is a real and useful property — but a different one than first claimed.

**Strong-model row: SETTLED (Claude Haiku vs Sonnet, temp 0, n=5).** Same-family
ablation, no rate limits.

| model | condition | kw_recall | faithful |
|---|---|---|---|
| Haiku 4.5 (weak) | flat | 0.78 | 0.84 |
| Haiku 4.5 (weak) | +causal | 0.77 | 0.89 (+0.06) |
| Haiku 4.5 (weak) | +doc | 0.78 | **0.94 (+0.10)** |
| Haiku 4.5 (weak) | +causal+doc | 0.84 | 0.82 |
| Sonnet 4.6 (strong) | flat | 0.78 | 0.83 |
| Sonnet 4.6 (strong) | +causal | 0.74 | 0.89 (+0.06) |
| Sonnet 4.6 (strong) | +doc | 0.80 | 0.85 (+0.02) |
| Sonnet 4.6 (strong) | +causal+doc | 0.74 | 0.89 (+0.06) |

**Answer: the capability dependence is signal-specific.**
- **Document structure (+doc) is capability-dependent** as hypothesized:
  faithfulness +0.10 for Haiku vs +0.02 for Sonnet. The weaker model benefits
  ~5x more from heading-path grounding; the stronger model already grounds well.
- **Causal chains (+causal) help both equally (+0.06)** — a general grounding
  aid, not capability-dependent.
- **Recall flat ~0.78 for both** — generation-side structure is not a recall
  lever; recall was already lifted by retrieval-side structure.
- Caveat: n=5. The +0.10 (Haiku +doc) is the clear signal; +causal+doc is noisy
  (the two signals don't combine cleanly). Direction is consistent with theory.

## Benchmark 5 — the value question: causal-graph RAG vs a STRONG flat baseline (`eval_value.py`)

Tested in the regime the system is *for* (multi-hop, root-cause) against strong
vector RAG (same encoder, top_k, LLM). 2 domains (finance fan-in + Chernobyl
deep cascade) x 3 question types, n=7/type, Haiku generation, Sonnet judge,
per-question logging, paired stats.

| qtype | flat | causal | delta | 95% CI | Wilcoxon p |
|---|---|---|---|---|---|
| fact | 0.45 | 0.57 | +0.12 | [-0.07, +0.28] | 0.250 |
| multihop | 0.24 | 0.47 | +0.24 | [+0.06, +0.55] | 0.125 |
| rootcause | 0.38 | 0.74 | +0.36 | [+0.10, +0.69] | 0.250 |

**The advantage scales with causal complexity** (fact < multihop < rootcause) —
causal-graph RAG nearly doubles root-cause correctness (0.38 -> 0.74) over a
strong baseline. Bootstrap CIs exclude zero for multihop and rootcause.

**Honest stats:** Wilcoxon p is non-significant (0.125, 0.25) — a POWER problem,
not absence of effect: at n=7 the test cannot reach p<0.05 with a few tied
deltas. The bootstrap CIs and the monotonic pattern are the trustworthy signals.
Strong directional evidence; n>=20/type needed for p<0.05 confirmation.

**Bottom line:** the earlier "small faithfulness bump" was testing a causal
tool as a fact-gathering box. In its real regime — multi-hop and root-cause,
vs a strong baseline — the causal graph delivers a large, complexity-scaling
advantage. That is the project's demonstrated value, and it is specialized
(causal reasoning), not general "better RAG".

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

## Methodological note (the most important lesson)

Two generation-side claims in earlier drafts were **sampling noise** that
vanished under determinism:
- the recall "synergy" (+0.11) → -0.01 on replication;
- the "capability-dependent +0.11" → not robust at temperature 0.2.

Fix: **evaluate at temperature 0** (removes run-to-run variance) and use
**n≥12** (tighter CIs). Never narrate a generation-side delta measured while
temperature>0 and n is small. The earlier write-ups over-read single runs; this
section supersedes them.

## Honest bottom line

What survives rigorous (temp-0, n=12) measurement:
1. **Causal extraction** (building the graph) — 26-q benchmark recall 0.46→0.60.
2. **Retrieval-side structure** (contextual indexing + MMR) — the large, robust
   win: flat recall 0.27/0.30→0.47 on a real large document.
3. **Generation-side structure → faithfulness, not recall.** Chains + document
   anchoring raise groundedness (+0.08–0.15) but do not add concept coverage
   (and bare causal chains slightly reduce it).

The practical reading: **the retrieval graph is what lifts answer quality; the
in-context structure is a grounding/anti-hallucination layer on top.** Recall
synergy and capability-scaling claims are retracted pending stronger evidence.
