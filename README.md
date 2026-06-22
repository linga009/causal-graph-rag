# Consequential-Graph VSA-RAG

A retrieval engine built to fix the failure mode that pure similarity-search RAG
has *by construction*: **chunking + embedding destroy the consequential structure
of a document.** When you split text into chunks and embed each one, the
cause→effect edges between chunks vanish. Ask "what did X ultimately cause?" and
similarity search returns the chunk mentioning X but is blind to the consequence
several hops away in a different chunk with different vocabulary.

This system extracts the causal topology **at ingest**, stores it as a directed
graph with VSA-encoded edges (so direction is preserved), and **traverses the
graph to return whole causal chains as the retrieval unit** — not isolated chunks.

## The problem, concretely

Document: *"The reactor overheated. As a result, the coolant valve failed. This
triggered a shutdown. The shutdown caused an outage. The outage disrupted
hospital operations."*

- **Standard dense RAG** chunks this into 5 fragments. Query *"what did the
  overheating ultimately disrupt?"* retrieves the "overheated" chunk by keyword
  similarity. The `outage -> operations` link lives in a chunk with near-zero
  lexical overlap with the query. **The answer is structurally invisible.**
- **This system** returns the full chain:
  `reactor ->(lead_to) valve ->(trigger) shutdown ->(cause) outage -/->(disrupt) operations`

## Architecture

```
                    +- direct node-name match -+
INGEST              +- VSA structural (direction)|
  extract causal    +- BM25 (exact terms)        +- RRF fusion -> entry nodes
  edges  ---------> +- dense (paraphrase)        |                    |
  build VSA graph                                                     v
  + index channels                              TRAVERSE causal graph (chains)
                                                          |
                                                  RERANK chains (direction-aware)
                                                          |
                                                  LLM <- ordered causal chains
```

Four fused channels pick *entry nodes*; the graph traversal recovers the
*structure*. That division is the key idea: similarity search is good at finding
*where* to start, terrible at representing *how things connect*. The graph owns
the connections.

## Files

| File | Purpose |
|------|---------|
| `vsa_core.py` | Bipolar hypervector algebra, role-filler triple encoding |
| `parser.py` | Sentence -> (AGENT, ACTION, PATIENT) triples (spaCy + fallback) |
| `causal_extractor.py` | Directed cause->effect edges, intra- and inter-sentence |
| `causal_graph.py` | VSA-encoded directed graph + chain traversal (fwd/bwd/path) |
| `retrievers.py` | BM25, dense (hashed), Reciprocal Rank Fusion |
| `graph_rag.py` | Orchestrating engine: ingest, retrieve, rerank, generate |
| `demo_graph.py` | Runnable demonstration |
| `llm_adapters.py` | Optional Groq / OpenAI / Anthropic clients |

(`pipeline.py` is the earlier single-triple VSA-RAG; the graph engine supersedes
it but reuses its `MockLLM`.)

## Quick start

```bash
pip install numpy sentence-transformers
# optional, improves causal extraction (handles passive voice, clausal subjects):
pip install spacy && python -m spacy download en_core_web_sm
python demo_graph.py
```

```python
from graph_rag import GraphRAG
rag = GraphRAG(dim=10000, semantic_weight=0)
rag.ingest(open("incident_report.txt").read())

answer, chains = rag.answer("What did the sensor fault ultimately disrupt?")
for c in chains:
    print(c.text())          # the recovered causal chain
    print(c.provenance())    # the source sentences it spans
```

Plug in a real LLM:
```python
from llm_adapters import GroqLLM
rag = GraphRAG(llm=GroqLLM())          # set GROQ_API_KEY
```

## How direction is preserved

Each edge `(cause, relation, effect)` is encoded as a role-filler hypervector
`AGENT(x)cause + ACTION(x)relation + PATIENT(x)effect`. Swapping cause and effect
re-binds the fillers to different roles, yielding a near-orthogonal vector — so
"A causes B" and "B causes A" are distinguishable, which cosine similarity over
embeddings cannot do. Edge **polarity** (+1 promotes, -1 suppresses) is tracked
separately; chain net-polarity is the product along the path, so a chain of two
suppressions reads as a net promotion.

## Query types supported

| Query intent | Traversal | Example |
|---|---|---|
| Forward ("what does X cause / lead to / disrupt") | `forward_chain` | "What did overheating ultimately cause?" |
| Backward ("why / root cause / what caused X") | `backward_chain` | "Why did the outage happen?" |
| Connection ("how does X relate to Y") | `path_between` | shortest causal path X->Y |

Direction is inferred from query phrasing; the reranker rewards chains that
*originate* at the queried event (forward) or *terminate* at it (backward).

## When to use this

This is a precision instrument, not a general-purpose RAG replacement. It is
the right tool when:

- **The question is causal** — "what ultimately caused X?", "why did Y happen?",
  "how did A lead to B?" Standard RAG cannot answer these because the causal
  chain spans multiple chunks with different vocabulary.
- **The document has consequential structure** — incident reports, medical case
  studies, engineering post-mortems, legal briefs, policy analyses, risk reports.
- **You need the LLM to reason over a path**, not just retrieve similar sentences.

**Strong fit by domain:**

| Domain | Example question |
|---|---|
| Healthcare | "What sequence of events led to the adverse outcome?" |
| DevOps / SRE | "What was the root cause of the outage?" |
| Legal / compliance | "What chain of decisions led to the liability?" |
| Finance / risk | "How did the rate change propagate through the portfolio?" |
| Engineering | "What failure mode triggered the cascade?" |

**Not the right tool for:**
- General factoid Q&A ("what year was X founded?") — use standard hybrid RAG
- Documents without explicit or implicit causal structure
- High-volume ingestion without a real relation extractor (the rule-based
  extractor is the ceiling on recall)

For best results, run this **alongside** a standard retriever and route by query
type: causal questions → this system, factoid questions → standard RAG.

## Honest limitations

- **The extractor is the ceiling.** Causal edges come from a verb lexicon +
  discourse connectives + the dependency parser. The rule-based fallback will
  miss domain-specific causal language ("the catalyst degraded", "the margin
  compressed"). In production, replace with a trained relation-extraction model
  or an LLM-based OpenIE extractor.
- **Implicit causation is partially handled** via an adjacency + state-change
  heuristic: consecutive sentences where the second contains a state-change verb
  (failed, collapsed, skidded, …) and no explicit connective get a weak
  `implicit_trigger` edge. This recovers cases like "the bridge was wet. Cars
  skidded." but misses purely inferential causation with no state-change signal.
- **Coreference is not resolved.** Pronouns ("this", "it", "they") appear as
  graph nodes instead of the event they refer to. A coreference resolver
  (e.g. spaCy's neuralcoref or an LLM pass) would significantly clean up the
  graph.
- **The "dense" channel uses SentenceTransformers** (`all-MiniLM-L6-v2`) by
  default. Falls back to a hashed-trigram stand-in if the package is not
  installed. For best results use a domain-specific encoder or Voyage/OpenAI.
- **The reranker blends lexical overlap with semantic similarity** from the dense
  encoder: the query embedding is compared against the mean of the chain's node
  embeddings. The seam is `GraphRAG._rerank`; swap in a cross-encoder
  (Cohere/Voyage rerank) for even stronger results.

## Production swap-in points

- `causal_extractor.extract_edges` → trained relation extractor / LLM OpenIE
- `causal_extractor.STATE_CHANGE_VERBS` → extend with domain-specific verbs to
  improve implicit causation recall
- `retrievers.SentenceTransformerDense` → swap model name for a domain-specific
  encoder; or replace with an ANN-indexed encoder (FAISS/Qdrant)
- `GraphRAG._rerank` → replace with a cross-encoder (Cohere/Voyage rerank) for
  full semantic reranking; the current blended scorer is a strong baseline
- store the graph in a real graph DB (Neo4j) and the VSA edge vectors as packed
  bits with a popcount index for sub-millisecond structural lookup
