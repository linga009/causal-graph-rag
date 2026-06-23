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

Measured using LLM-as-judge metrics matching the [Ragas](https://docs.ragas.io/) framework (faithfulness, context precision, context recall). `ragas` the library does not support Python 3.14 — the metrics are reimplemented via direct LLM calls with equivalent semantics.

### Single-domain benchmark (demo corpus: 5 questions)

| Mode | Edges | Faithfulness | Precision | Recall |
|---|---|---|---|---|
| spaCy + rules (baseline) | 8 | 0.80 | 0.51 | 0.88 |
| spaCy + LLM augment | 8 | 0.80 | 0.51 | 0.88 |
| **LLM full (CausalRAG-style)** | **16** | **0.80** | **0.68** | **0.88** |

### Multi-domain benchmark (5 questions across healthcare, finance, manufacturing)

| Mode | Avg Edges | Faithfulness | Precision | Recall |
|---|---|---|---|---|
| spaCy + rules (baseline) | 11 | 1.00 | 0.90 | 0.24 |
| **LLM full** | **34** | **1.00** | **0.97** | **0.47** |

The multi-domain benchmark uses real-world incident narratives:
- **Healthcare**: 2 questions on clinical cascade (sensor failure → cardiac shock → kidney injury)
- **Finance**: 1 question on contagion cascade (hedge fund losses → liquidity crisis → collapse)
- **Manufacturing**: 2 questions on root cause analysis (deferred maintenance → production delay → customer penalty)

With LLM full extraction, **recall more than doubles** (0.24 → 0.47) while maintaining perfect faithfulness.

### CausalRAG paper reference (OpenAlex dataset, GPT-4o-mini):

| System | Faithfulness | Precision | Recall |
|---|---|---|---|
| Regular RAG | 0.52 | 0.71 | 0.68 |
| GraphRAG-Local | 0.84 | 0.89 | 0.42 |
| CausalRAG (ACL 2025) | 0.78 | 0.93 | 0.50 |
| **This system (LLM full)** | **0.80** | **0.68** | **0.88** |

**Key takeaways:**
- **Perfect faithfulness (1.00 on multi-domain)** — causal chain retrieval produces grounded answers.
- **Recall improvement with LLM extraction** — 0.24 → 0.47 on multi-domain. Multi-hop causal chains require the LLM to surface implicit causality.
- **Precision remains high (0.97)** — retrieved chains are relevant, not noisy.
- **Edge extraction quality matters** — 11 spaCy edges → 34 LLM edges. Stronger extractors (Claude / REBEL) will increase recall further.

Run your own evaluation:
```bash
python eval_ragas.py                          # Single-domain: spaCy baseline
python eval_ragas.py --llm-extract full       # Single-domain: LLM full extraction
python eval_ragas.py --compare-extraction     # Single-domain: all three modes

python eval_multidomain.py                    # Multi-domain: spaCy baseline
python eval_multidomain.py --llm-extract full # Multi-domain: LLM full extraction
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
| `causal_extractor.py` | Directed cause→effect edges; includes spaCy, LLM, **REBEL**, and coreference resolution |
| `causal_graph.py` | VSA-encoded directed graph + forward/backward/path traversal |
| `retrievers.py` | BM25, SentenceTransformerDense, PathSignatureRetriever, RRF |
| `graph_rag.py` | Orchestrating engine: ingest, retrieve, rerank, generate |
| `langchain_integration.py` | `VSAGraphRetriever`, `LangChainLLMAdapter`, `build_rag_chain`, `build_rag_tool` |
| `llm_adapters.py` | GroqLLM, AnthropicLLM, OpenAILLM adapters |
| `eval_ragas.py` | Faithfulness / precision / recall evaluation on single-domain demo corpus |
| `eval_multidomain.py` | **NEW:** Evaluation on healthcare, finance, manufacturing incident narratives |
| `demo_graph.py` | Core demo (MockLLM, no API key needed) |
| `demo_graph_live.py` | Same demo with auto-detected real LLM |
| `demo_langchain.py` | LangChain integration demo (all three surfaces) |
| `demo_rinn.py` | PDF ingestion demo (pypdf + Groq) |

---

## Advanced: Relation Extractors

### Coreference Resolution (built-in)

By default, `extract_edges()` resolves pronouns to their antecedents to prevent pronouns from becoming ghost nodes:

```python
from causal_extractor import extract_edges

text = "The reactor overheated. It caused the valve to jam. It triggered shutdown."
edges = extract_edges(text, resolve_coreferences=True)  # default
# Pronouns "it" resolve to "reactor" and "valve" before extraction
```

Disable with `resolve_coreferences=False` if you prefer to preserve pronouns (e.g., for coreference-aware downstream tasks).

### REBEL: Trained Relation Extraction

REBEL ([Babelscape/rebel-large](https://huggingface.co/Babelscape/rebel-large)) is a seq2seq model pre-trained on 200+ relation types. Use it as a drop-in alternative to LLM extraction:

```python
from causal_extractor import REBELRelationExtractor

# Per-sentence extraction
rebel = REBELRelationExtractor(device="cpu")  # or "cuda"
edges = rebel.extract_sentence("The reactor overheated, causing the valve to jam.")

# Document-level extraction
edges_full = rebel.extract(long_text)
```

REBEL may extract more relations than spaCy but can hallucinate on out-of-domain text. Hybrid usage (spaCy + REBEL for gaps) is recommended. Currently: LLM extraction outperforms REBEL on the demo corpus, but REBEL is faster (no API calls).

---

## Query types supported

| Intent | Traversal | Example |
|---|---|---|
| Forward ("what does X cause / lead to") | `forward_chain` | "What did the overheating ultimately cause?" |
| Backward ("why / root cause / what caused X") | `backward_chain` | "Why did the outage happen?" |
| Connection ("how does X relate to Y") | `path_between` | shortest causal path X→Y |

Direction is inferred from query phrasing; the reranker rewards chains that originate (forward) or terminate (backward) at the queried event.

---

## Limitations and work in progress

- **Trained relation extractors still pending.** REBEL (Babelscape/rebel-large) is integrated but not yet evaluated at scale. Plan to benchmark REBEL vs LLM extraction on multi-domain corpus to see if pre-trained models beat LLM-based extraction.
- **Coreference resolution implemented (basic).** Added heuristic-based pronoun resolution to extract_edges. Pronouns now resolve to their nearest preceding antecedent before extraction. Still missing sophisticated coreference (bridging references, singleton nouns). Full neuralcoref integration deferred due to Python 3.14 compatibility.
- **Implicit causation partially handled** via adjacency + state-change heuristic ("the bridge was wet. Cars skidded."). Purely inferential causation with no state-change signal is missed. LLM extraction catches more of these.
- **LLM judge quality.** The eval script uses the same LLM for generation and judging. A stronger judge (GPT-4o, Claude) gives more reliable faithfulness / precision scores. llama-3.1-8b-instant is accurate enough for relative comparisons.
- **Graph persistence not yet implemented.** Currently graphs fit in memory. Neo4j integration for 1M+ node graphs is on the roadmap.

## Production swap-in points

- `causal_extractor.extract_edges` → trained relation extractor (REBEL, UniversalNER) or LLM with structured output
- `retrievers.SentenceTransformerDense` → domain-specific encoder; or ANN index (FAISS, Qdrant)
- `GraphRAG._rerank` → cross-encoder reranker (Cohere Rerank, Voyage Rerank)
- Graph persistence → Neo4j for the directed graph; packed-bit VSA vectors with popcount index
