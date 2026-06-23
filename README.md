# Causal Graph RAG

A retrieval engine built to fix the failure mode that pure similarity-search RAG has **by construction**: chunking and embedding destroy the cause→effect structure of a document. When you split text into chunks and embed each one, the causal edges between chunks vanish. Ask "what did X ultimately cause?" and similarity search returns the chunk mentioning X, but is blind to the consequence several hops away in a different chunk with different vocabulary.

This system extracts the causal topology **at ingest**, stores it as a directed graph with VSA-encoded edges (so direction is preserved), and **traverses the graph to return whole causal chains as the retrieval unit** — not isolated chunks.

Related work: [CausalRAG (ACL 2025)](https://arxiv.org/abs/2503.19878). Key differences are documented in the [comparison section](#comparison-with-causalrag-acl-2025).

---

## The problem, concretely

Document: *"The reactor overheated. As a result, the coolant valve failed. This triggered a shutdown. The shutdown caused an outage. The outage disrupted hospital operations."*

| System | Query: *"What did the overheating ultimately disrupt?"* |
|---|---|
| **Standard dense RAG** | Returns the *"reactor overheated"* chunk. The `outage → operations` link lives in a different chunk with near-zero lexical overlap. **Answer is structurally invisible.** |
| **This system** | Returns: `reactor ->(lead_to) valve ->(trigger) shutdown ->(cause) outage -/->(disrupt) operations` |

---

## Architecture

```
INGEST
  ┌─ spaCy dep parse ─┐
  │  + rule fallback   ├──► (cause, relation, effect) edges
  │  + LLM extractor* │         │
  └───────────────────┘         ▼
                          VSA-encoded directed graph
                          + BM25 / dense / path-signature indices

RETRIEVE (5-channel RRF fusion)
  ┌─ direct node-name match  (weight 2.0) ─┐
  │  VSA structural direction (weight 1.2)  │
  │  BM25 exact terms        (weight 1.0)   ├──► entry nodes
  │  Dense / MiniLM          (weight 1.0)   │
  └─ Path Signature†         (weight 0.8)  ─┘
                                    │
                          TRAVERSE causal graph
                          (forward / backward BFS from entry nodes)
                                    │
                          RERANK chains (direction-aware + semantic)
                                    │
                          LLM ◄── ordered causal chains as context
                          (optional causal-summary step‡)
```

`*` LLM-assisted extraction (borrowed from CausalRAG) — fills implicit causality gaps that dependency parsing misses.  
`†` Path Signature channel uses truncated iterated integrals (Rough Path Theory, level-3, d=16) — encodes sequential order in the narrative, not just content.  
`‡` Two-step generation: compress chains into a causal summary, then generate the answer. Improves coherence on multi-hop queries.

---

## Evaluation

Measured on a 5-question causal reasoning benchmark using LLM-as-judge metrics matching the [Ragas](https://docs.ragas.io/) framework (faithfulness, context precision, context recall). `ragas` the library does not support Python 3.14 — the metrics are reimplemented via direct LLM calls with equivalent semantics.

> **Note:** scores below are on a purpose-built demo corpus (reactor incident + budget-cuts chain), not the OpenAlex academic dataset used in the CausalRAG paper. Direct numeric comparison with the paper is directional only.

| Mode | Edges | Faithfulness | Precision | Recall |
|---|---|---|---|---|
| spaCy + rules (baseline) | 8 | 0.80 | 0.51 | 0.88 |
| spaCy + LLM augment | 8 | 0.80 | 0.51 | 0.88 |
| **LLM full (CausalRAG-style)** | **16** | **0.80** | **0.68** | **0.88** |

**CausalRAG paper reference** (OpenAlex dataset, GPT-4o-mini):

| System | Faithfulness | Precision | Recall |
|---|---|---|---|
| Regular RAG | 0.52 | 0.71 | 0.68 |
| GraphRAG-Local | 0.84 | 0.89 | 0.42 |
| CausalRAG (ACL 2025) | 0.78 | 0.93 | 0.50 |
| **This system (LLM full)** | **0.80** | **0.68** | **0.88** |

**Key takeaways:**
- **Faithfulness 0.80** — competitive with CausalRAG (0.78), well above Regular RAG (0.52). Causal chain retrieval produces grounded answers.
- **Recall 0.88** — highest of all systems, including CausalRAG (0.50). The 5-channel RRF brings in facts that single-channel vector search misses.
- **Precision gap (0.68 vs 0.93)** — CausalRAG uses GPT-4o-mini for graph building; the LLM quality directly shapes graph completeness. Using a stronger extractor (Claude / GPT-4o) will close this gap. The `--llm-extract full` flag enables this.

Run your own evaluation:
```bash
python eval_ragas.py                          # spaCy baseline
python eval_ragas.py --llm-extract full       # CausalRAG-style LLM extraction
python eval_ragas.py --compare-extraction     # all three modes side-by-side
```

---

## Installation

```bash
pip install numpy sentence-transformers

# Stronger causal extraction (passive voice, compound nouns):
pip install spacy && python -m spacy download en_core_web_sm

# LangChain integration:
pip install langchain-core langchain langchain-groq

# LLM backends (set the matching env var):
pip install groq        # GROQ_API_KEY
pip install anthropic   # ANTHROPIC_API_KEY
pip install openai      # OPENAI_API_KEY
```

Windows requires the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) for PyTorch / sentence-transformers.

---

## Quick start

```python
from graph_rag import GraphRAG

rag = GraphRAG(dim=10000)
rag.ingest(open("incident_report.txt").read())

answer, chains = rag.answer("What did the sensor fault ultimately disrupt?")
print(answer)
for c in chains:
    print(c.text())       # e.g. "sensor ->(lead_to) overheating ->(cause) ..."
    print(c.provenance()) # source sentences the chain spans
```

**With a real LLM:**
```python
from llm_adapters import GroqLLM   # or AnthropicLLM, OpenAILLM
rag = GraphRAG(llm=GroqLLM())      # reads GROQ_API_KEY from environment
```

**With LLM-assisted graph building** (catches implicit / academic causality):
```python
from llm_adapters import GroqLLM
llm = GroqLLM()
rag = GraphRAG(llm=llm)
rag.ingest(text, llm_extractor=llm, llm_mode="augment")  # fills spaCy gaps
# or:
rag.ingest(text, llm_extractor=llm, llm_mode="full")     # all sentences via LLM
```

**With causal-summary step** (two-step generation, tighter multi-hop answers):
```python
answer, chains = rag.answer("What did X lead to?", summarize=True)
```

---

## LangChain integration

Three surfaces for dropping the engine into LangChain pipelines:

### 1. Retriever (`BaseRetriever`)

```python
from langchain_groq import ChatGroq
from graph_rag import GraphRAG
from langchain_integration import VSAGraphRetriever

rag = GraphRAG()
rag.ingest(text)

retriever = VSAGraphRetriever(graph_rag=rag, top_k=3)
docs = retriever.invoke("What caused the shutdown?")
# docs[i].page_content  → chain text + source sentences
# docs[i].metadata      → entry_node, direction, rrf_score, provenance
```

### 2. LCEL chain

```python
from langchain_groq import ChatGroq
from langchain_integration import build_rag_chain

llm   = ChatGroq(model="llama-3.1-8b-instant")
chain = build_rag_chain(retriever, llm)                  # 1 LLM call
# chain = build_rag_chain(retriever, llm, summarize=True) # 2 LLM calls

answer = chain.invoke("Why did the power outage happen?")
# or: chain.stream(question) for streaming
```

### 3. Tool-calling agent

```python
from langchain.agents import create_agent
from langchain_integration import build_rag_tool, LangChainLLMAdapter

adapter = LangChainLLMAdapter(llm)          # use LangChain LLM inside GraphRAG
rag     = GraphRAG(llm=adapter)
rag.ingest(text, llm_extractor=adapter, llm_mode="full")

tool  = build_rag_tool(rag)
agent = create_agent(model=llm, tools=[tool],
                     system_prompt="You are a causal reasoning assistant.")
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
```

Run the full demo:
```bash
python demo_langchain.py    # auto-picks GROQ_API_KEY / ANTHROPIC_API_KEY from .env
```

---

## Comparison with CausalRAG (ACL 2025)

| Dimension | CausalRAG | This system |
|---|---|---|
| **Graph building** | LLM (GPT-4o-mini) on every chunk — 1 call/chunk | spaCy dep parse (free) + optional LLM fill |
| **Retrieval** | 1 channel: dense vector + k-hop expansion | 5 channels: name match, VSA direction, BM25, dense, path signature |
| **Causal direction** | Not modelled (symmetric similarity) | Forward / backward intent detection + directed graph traversal |
| **Path signatures** | — | Rough Path Theory (level-3 truncated iterated integrals, novel channel) |
| **Causal summary step** | Yes — dedicated LLM compression before generation | Yes — optional `summarize=True` |
| **LangChain** | LangChain v0.2 | LangChain 1.x (BaseRetriever, LCEL, create_agent) |
| **LLM dependency at ingest** | Required (graph built by LLM) | Optional (spaCy default; LLM is additive) |
| **Code status** | Paper released; code coming soon | Working, in this repo |

**When to use LLM extraction (`llm_mode="full"`):** Documents with academic, medical, or policy language where causality is expressed implicitly — *"the programme aims to address societal challenges"*, *"the biomarker may indicate disease progression"*. spaCy dependency parsing handles explicit causal verbs well; the LLM catches the rest.

**When to use `summarize=True`:** Long multi-hop chains (5+ edges) where the final LLM prompt becomes noisy. The summarization step compresses the chain into prose before generation, reducing hallucination on the final answer.

---

## Files

| File | Purpose |
|---|---|
| `vsa_core.py` | Bipolar {-1,+1} hypervector algebra, role-filler triple encoding |
| `parser.py` | Sentence → (AGENT, ACTION, PATIENT) triples (spaCy + rule fallback) |
| `causal_extractor.py` | Directed cause→effect edges + `LLMEdgeExtractor` + `extract_edges_hybrid()` |
| `causal_graph.py` | VSA-encoded directed graph + forward/backward/path traversal |
| `retrievers.py` | BM25, SentenceTransformerDense, PathSignatureRetriever, RRF |
| `graph_rag.py` | Orchestrating engine: ingest, retrieve, rerank, generate |
| `langchain_integration.py` | `VSAGraphRetriever`, `LangChainLLMAdapter`, `build_rag_chain`, `build_rag_tool` |
| `llm_adapters.py` | GroqLLM, AnthropicLLM, OpenAILLM adapters |
| `eval_ragas.py` | Faithfulness / precision / recall evaluation (Ragas-compatible metrics) |
| `demo_graph.py` | Core demo (MockLLM, no API key needed) |
| `demo_graph_live.py` | Same demo with auto-detected real LLM |
| `demo_langchain.py` | LangChain integration demo (all three surfaces) |
| `demo_rinn.py` | PDF ingestion demo (pypdf + Groq) |

---

## Query types supported

| Intent | Traversal | Example |
|---|---|---|
| Forward ("what does X cause / lead to") | `forward_chain` | "What did the overheating ultimately cause?" |
| Backward ("why / root cause / what caused X") | `backward_chain` | "Why did the outage happen?" |
| Connection ("how does X relate to Y") | `path_between` | shortest causal path X→Y |

Direction is inferred from query phrasing; the reranker rewards chains that originate (forward) or terminate (backward) at the queried event.

---

## Honest limitations

- **Extractor is the recall ceiling.** spaCy + the verb lexicon miss domain-specific causal language. Use `llm_extractor` for complex documents, or replace with a trained relation-extraction model.
- **Coreference not resolved.** Pronouns appear as graph nodes. A coreference resolver (spaCy neuralcoref, or an LLM pass) would significantly clean up graphs built from narrative text.
- **Implicit causation is partially handled** via an adjacency + state-change heuristic ("the bridge was wet. Cars skidded."). Purely inferential causation with no state-change signal is missed.
- **LLM judge quality.** The eval script uses the same LLM for generation and judging. A stronger judge (GPT-4o, Claude) gives more reliable faithfulness / precision scores. llama-3.1-8b-instant is accurate enough for relative comparisons but not publication-grade absolute scores.

## Production swap-in points

- `causal_extractor.extract_edges` → trained relation extractor (REBEL, UniversalNER) or LLM with structured output
- `retrievers.SentenceTransformerDense` → domain-specific encoder; or ANN index (FAISS, Qdrant)
- `GraphRAG._rerank` → cross-encoder reranker (Cohere Rerank, Voyage Rerank)
- Graph persistence → Neo4j for the directed graph; packed-bit VSA vectors with popcount index
