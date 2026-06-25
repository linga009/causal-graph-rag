"""
api.py — REST API for Causal Graph RAG
=======================================
FastAPI service exposing ingest, query, and graph inspection endpoints.

Usage
-----
  pip install fastapi uvicorn
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

  # Or with Docker:
  docker build -t causal-rag-api .
  docker run -p 8000:8000 -e GROQ_API_KEY=... causal-rag-api

Endpoints
---------
  POST /ingest        Ingest text into the causal graph
  POST /query         Answer a causal question
  GET  /graph         Graph statistics and edges
  DELETE /graph       Clear the graph
  GET  /health        Health check
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

# Load .env before any imports that read env vars
def _load_env(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass

_load_env()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.dirname(__file__))

from graph_rag import GraphRAG
from llm_adapters import GroqLLM, GeminiLLM, AnthropicLLM, OpenAILLM

# --------------------------------------------------------------------------- #
#  App setup
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Causal Graph RAG API",
    description="REST API for causal chain retrieval and reasoning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Shared RAG instance (in-memory; swap neo4j_uri for persistent backend)
# --------------------------------------------------------------------------- #

def _build_llm():
    """Pick an LLM from whatever API key is present (Groq → Gemini → Anthropic →
    OpenAI). Resilient: a key set without its SDK installed is skipped, not fatal."""
    for env, cls in (("GROQ_API_KEY", GroqLLM), ("GEMINI_API_KEY", GeminiLLM),
                     ("ANTHROPIC_API_KEY", AnthropicLLM), ("OPENAI_API_KEY", OpenAILLM)):
        if os.environ.get(env):
            try:
                return cls()
            except ImportError:
                continue
    return None

_llm = _build_llm()
_rag = GraphRAG(dim=10000, llm=_llm)

# GraphRAG mutates shared in-memory indices on ingest. FastAPI runs sync
# endpoints in a threadpool, so without a lock a concurrent /ingest and /query
# would read a half-rebuilt BM25/dense index. Serialize all graph access.
_rag_lock = threading.Lock()


def _edges(rag: GraphRAG) -> list:
    """Backend-agnostic edge list (in-memory list or Neo4j fetch)."""
    g = rag.graph
    return g._get_edges() if hasattr(g, "_get_edges") else list(g.edges)


# --------------------------------------------------------------------------- #
#  Request / Response models
# --------------------------------------------------------------------------- #

class IngestRequest(BaseModel):
    text: str = Field(..., description="Text to ingest into the causal graph")
    llm_mode: Optional[str] = Field(
        None,
        description="LLM extraction mode: 'augment' (fill spaCy gaps) or 'full' (all sentences). "
                    "Omit to use spaCy-only extraction (free, fast)."
    )
    schema_: str = Field(
        "general", alias="schema",
        description="Document-structure preset: 'general' (default, domain-agnostic), "
                    "'research' (IMRaD), 'clinical' (SOAP), 'incident' (RCA), or 'auto'."
    )

    model_config = {"populate_by_name": True}

class IngestResponse(BaseModel):
    edges_added: int
    total_nodes: int
    total_edges: int
    llm_mode_used: Optional[str]
    schema_used: str

class QueryRequest(BaseModel):
    question: str = Field(..., description="Causal question to answer")
    top_k: int = Field(3, ge=1, le=10, description="Number of causal chains to retrieve")
    summarize: bool = Field(False, description="Two-step generation: compress chains before answering")

class CausalChain(BaseModel):
    text: str
    entry_node: str
    direction: str
    score: float
    provenance: List[str]

class QueryResponse(BaseModel):
    answer: str
    chains: List[CausalChain]
    question: str

class GraphEdge(BaseModel):
    cause: str
    relation: str
    effect: str
    polarity: int
    source: str

class GraphStats(BaseModel):
    nodes: int
    edges: int
    sample_edges: List[GraphEdge]

class HealthResponse(BaseModel):
    status: str
    llm: str
    nodes: int
    edges: int
    available_schemas: List[str]


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Health check — LLM, graph size, and the document-structure presets a
    client can choose from for /ingest."""
    from doc_structure import AVAILABLE_SCHEMAS
    with _rag_lock:
        return HealthResponse(
            status="ok",
            llm=type(_llm).__name__ if _llm else "none (spaCy only)",
            nodes=len(_rag.graph.nodes()),
            edges=len(_edges(_rag)),
            available_schemas=list(AVAILABLE_SCHEMAS),
        )


@app.post("/ingest", response_model=IngestResponse, tags=["graph"])
def ingest(req: IngestRequest):
    """
    Ingest text into the causal graph.

    - **text**: The document to process (incident report, clinical note, financial report, etc.)
    - **llm_mode**: `"augment"` fills spaCy gaps with LLM; `"full"` uses LLM on every sentence.
      Omit for free spaCy-only extraction.
    - **schema**: document-structure preset (`general` default, or `research`/`clinical`/`incident`/`auto`).
      See `/health` → `available_schemas`.
    """
    from doc_structure import AVAILABLE_SCHEMAS
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="'text' must be a non-empty string.")
    if req.llm_mode and req.llm_mode not in ("augment", "full"):
        raise HTTPException(status_code=422, detail="llm_mode must be 'augment' or 'full'.")
    if req.schema_ not in AVAILABLE_SCHEMAS:
        raise HTTPException(
            status_code=422,
            detail=f"schema must be one of {list(AVAILABLE_SCHEMAS)}.",
        )

    with _rag_lock:
        if req.llm_mode and _llm:
            n = _rag.ingest(req.text, llm_extractor=_llm, llm_mode=req.llm_mode,
                            schema=req.schema_)
        else:
            n = _rag.ingest(req.text, schema=req.schema_)

        return IngestResponse(
            edges_added=n,
            total_nodes=len(_rag.graph.nodes()),
            total_edges=len(_edges(_rag)),
            llm_mode_used=req.llm_mode if req.llm_mode and _llm else None,
            schema_used=req.schema_,
        )


@app.post("/query", response_model=QueryResponse, tags=["retrieval"])
def query(req: QueryRequest):
    """
    Answer a causal question using the ingested graph.

    - **question**: Natural language causal question
    - **top_k**: Number of causal chains to retrieve (default 3)
    - **summarize**: Two-step generation for long multi-hop chains

    Returns the answer and the supporting causal chains with provenance.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="'question' must be a non-empty string.")
    if not _llm:
        raise HTTPException(
            status_code=503,
            detail="No LLM configured. Set GROQ_API_KEY, GEMINI_API_KEY, "
                   "ANTHROPIC_API_KEY, or OPENAI_API_KEY.",
        )

    # Retrieve under the lock (touches shared indices); run the stateless LLM
    # generation OUTSIDE the lock so concurrent queries don't serialize on it.
    with _rag_lock:
        if not _rag.graph.nodes():
            raise HTTPException(status_code=400, detail="Graph is empty. POST to /ingest first.")
        chains = _rag.retrieve(req.question, top_k=req.top_k)
    answer = _rag.generate(req.question, chains, summarize=req.summarize)

    chain_out = [
        CausalChain(
            text=c.text(),
            entry_node=c.entry_node,
            direction=c.direction,
            score=round(float(c.rerank_score), 4),
            provenance=c.provenance(),
        )
        for c in chains
    ]
    return QueryResponse(answer=answer, chains=chain_out, question=req.question)


@app.get("/graph", response_model=GraphStats, tags=["graph"])
def graph_stats():
    """Return graph statistics and a sample of edges."""
    with _rag_lock:
        edges = _edges(_rag)
        n_nodes = len(_rag.graph.nodes())
    sample = [
        GraphEdge(
            cause=e.cause,
            relation=e.relation,
            effect=e.effect,
            polarity=e.polarity,
            source=e.source_sent[:100] if e.source_sent else "",
        )
        for e in edges[:20]
    ]
    return GraphStats(nodes=n_nodes, edges=len(edges), sample_edges=sample)


@app.delete("/graph", tags=["graph"])
def clear_graph():
    """Clear all edges and nodes from the graph."""
    global _rag
    with _rag_lock:
        old = _rag
        _rag = GraphRAG(dim=10000, llm=_llm)
        if getattr(old, "using_neo4j", False):
            old.close()
    return {"status": "cleared"}
