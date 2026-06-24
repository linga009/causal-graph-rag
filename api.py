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
from typing import Any, Dict, List, Optional

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
from llm_adapters import GroqLLM, AnthropicLLM

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
    if os.environ.get("GROQ_API_KEY"):
        return GroqLLM()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
    return None

_llm = _build_llm()
_rag = GraphRAG(dim=10000, llm=_llm)


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

class IngestResponse(BaseModel):
    edges_added: int
    total_nodes: int
    total_edges: int
    llm_mode_used: Optional[str]

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


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Health check — returns LLM type and graph size."""
    edges = _rag.graph._get_edges() if hasattr(_rag.graph, "_get_edges") else list(_rag.graph.edges)
    return HealthResponse(
        status="ok",
        llm=type(_llm).__name__ if _llm else "none (spaCy only)",
        nodes=len(_rag.graph.nodes()),
        edges=len(edges),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["graph"])
def ingest(req: IngestRequest):
    """
    Ingest text into the causal graph.

    - **text**: The document to process (incident report, clinical note, financial report, etc.)
    - **llm_mode**: `"augment"` fills spaCy gaps with LLM; `"full"` uses LLM on every sentence.
      Omit for free spaCy-only extraction.
    """
    before = len(_rag.graph.nodes())

    if req.llm_mode and _llm:
        n = _rag.ingest(req.text, llm_extractor=_llm, llm_mode=req.llm_mode)
    else:
        n = _rag.ingest(req.text)

    edges = _rag.graph._get_edges() if hasattr(_rag.graph, "_get_edges") else list(_rag.graph.edges)
    after = len(_rag.graph.nodes())

    return IngestResponse(
        edges_added=n,
        total_nodes=after,
        total_edges=len(edges),
        llm_mode_used=req.llm_mode if req.llm_mode and _llm else None,
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
    if not _rag.graph.nodes():
        raise HTTPException(status_code=400, detail="Graph is empty. POST to /ingest first.")

    if not _llm:
        raise HTTPException(
            status_code=503,
            detail="No LLM configured. Set GROQ_API_KEY or ANTHROPIC_API_KEY."
        )

    answer, chains = _rag.answer(req.question, top_k=req.top_k, summarize=req.summarize)

    chain_out = []
    for c in chains:
        chain_out.append(CausalChain(
            text=c.text(),
            entry_node=getattr(c, "entry_node", ""),
            direction=getattr(c, "direction", "forward"),
            score=float(getattr(c, "score", 0.0)),
            provenance=c.provenance(),
        ))

    return QueryResponse(answer=answer, chains=chain_out, question=req.question)


@app.get("/graph", response_model=GraphStats, tags=["graph"])
def graph_stats():
    """Return graph statistics and a sample of edges."""
    edges = _rag.graph._get_edges() if hasattr(_rag.graph, "_get_edges") else list(_rag.graph.edges)
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
    return GraphStats(
        nodes=len(_rag.graph.nodes()),
        edges=len(edges),
        sample_edges=sample,
    )


@app.delete("/graph", tags=["graph"])
def clear_graph():
    """Clear all edges and nodes from the graph."""
    global _rag
    _rag = GraphRAG(dim=10000, llm=_llm)
    return {"status": "cleared"}
