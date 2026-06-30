"""
api.py — REST API for Causal Graph RAG
=======================================
FastAPI service exposing causal graph retrieval and reasoning endpoints.

Usage
-----
  pip install fastapi uvicorn
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

  # Or with Docker:
  docker build -t causal-rag-api .
  docker run -p 8000:8000 -e GROQ_API_KEY=... causal-rag-api

Environment variables
---------------------
  GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
      First key found is used. Omit all for spaCy-only ingest (no /query).
  ALLOWED_ORIGINS
      Comma-separated list of allowed CORS origins. Default: "*" (open).
      Production: set to "https://yourapp.com,https://api.yourapp.com".

Endpoints
---------
  POST /ingest        Ingest text into the causal graph
  POST /query         Answer a causal question (requires LLM)
  POST /retrieve      Retrieve causal chains without generating an answer
  POST /rootcause     Backward chains: trace root causes of an event (no LLM)
  POST /impact        Forward chains: trace downstream impact of an event (no LLM)
  POST /path          Shortest causal path between two events (no LLM)
  GET  /graph         Graph statistics and edge sample
  DELETE /graph       Clear the graph
  GET  /health        Health check + available schemas
"""

from __future__ import annotations

import os
import time
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

import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .graph_rag import GraphRAG
from .llm_adapters import build_llm

# --------------------------------------------------------------------------- #
#  App setup
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Causal Graph RAG API",
    description="REST API for causal chain retrieval and reasoning",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — configurable via ALLOWED_ORIGINS env var; defaults to open for dev
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# X-Process-Time header on every response (ms, 1 decimal place)
@app.middleware("http")
async def _add_process_time(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time"] = f"{ms:.1f}ms"
    return response


# --------------------------------------------------------------------------- #
#  Shared RAG instance
# --------------------------------------------------------------------------- #

_llm = build_llm()
_rag = GraphRAG(dim=10000, llm=_llm)

# All graph mutations and index reads must be serialized. LLM generation
# (stateless, slow) runs outside the lock so concurrent /query calls don't
# block each other on network I/O.
_rag_lock = threading.Lock()


def _edges(rag: GraphRAG) -> list:
    g = rag.graph
    return g._get_edges() if hasattr(g, "_get_edges") else list(g.edges)


def _chain_confidence(chain) -> float:
    if not chain.chain:
        return 1.0
    return sum(e.confidence for e in chain.chain) / len(chain.chain)


def _chain_polarity(chain) -> int:
    p = 1
    for e in chain.chain:
        p *= e.polarity
    return int(p)


# --------------------------------------------------------------------------- #
#  Request / Response models
# --------------------------------------------------------------------------- #

class IngestRequest(BaseModel):
    text: str = Field(..., description="Text to ingest into the causal graph")
    llm_mode: Optional[str] = Field(
        None,
        description="LLM extraction mode: 'augment' (fill spaCy gaps) or 'full' "
                    "(all sentences). Omit for free spaCy-only extraction."
    )
    schema_: str = Field(
        "general", alias="schema",
        description="Document-structure preset: general (default), research, "
                    "clinical, incident, or auto."
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
    summarize: bool = Field(
        False, description="Two-step generation: compress chains before answering"
    )


class RetrieveRequest(BaseModel):
    question: str = Field(..., description="Causal question to retrieve chains for")
    top_k: int = Field(3, ge=1, le=10, description="Number of causal chains to return")


class EventRequest(BaseModel):
    event: str = Field(..., description="Event or outcome to trace")
    depth: int = Field(6, ge=1, le=12, description="Maximum BFS depth")


class PathRequest(BaseModel):
    source: str = Field(..., description="Starting cause or event")
    target: str = Field(..., description="Target effect or outcome")
    depth: int = Field(6, ge=1, le=12, description="Maximum BFS depth")


class CausalChain(BaseModel):
    text: str
    entry_node: str
    direction: str
    score: float
    hop_count: int
    chain_confidence: float
    chain_polarity: int
    provenance: List[str]


class QueryResponse(BaseModel):
    answer: str
    chains: List[CausalChain]
    question: str


class RetrieveResponse(BaseModel):
    chains: List[CausalChain]
    question: str


class GraphTraversalResponse(BaseModel):
    node: str
    chains: List[str]   # human-readable chain texts
    count: int


class PathResponse(BaseModel):
    source_node: str
    target_node: str
    path: Optional[str]   # None if no path exists
    found: bool


class GraphEdge(BaseModel):
    cause: str
    relation: str
    effect: str
    polarity: int
    confidence: float
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
#  Helpers
# --------------------------------------------------------------------------- #

def _to_chain_model(c) -> CausalChain:
    return CausalChain(
        text=c.text(),
        entry_node=c.entry_node,
        direction=c.direction,
        score=round(float(c.rerank_score), 4),
        hop_count=len(c.chain),
        chain_confidence=round(_chain_confidence(c), 4),
        chain_polarity=_chain_polarity(c),
        provenance=c.provenance(),
    )


def _require_graph():
    if not _rag.graph.nodes():
        raise HTTPException(status_code=400, detail="Graph is empty. POST to /ingest first.")


def _require_llm():
    if not _llm:
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM configured. Set GROQ_API_KEY, GEMINI_API_KEY, "
                "ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            ),
        )


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Health check — LLM backend, graph size, and available document-structure presets."""
    from .doc_structure import AVAILABLE_SCHEMAS
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

    - **text**: Document to process (incident report, clinical note, financial report…)
    - **llm_mode**: `augment` fills spaCy gaps; `full` uses LLM on every sentence.
      Omit for free spaCy-only extraction.
    - **schema**: document-structure preset. See `/health` → `available_schemas`.
    """
    from .doc_structure import AVAILABLE_SCHEMAS
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="'text' must be non-empty.")
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

    Retrieves causal chains then calls the LLM to synthesise an answer.
    Requires an LLM key. For pure chain retrieval without generation, use `/retrieve`.

    - **question**: Natural language causal question
    - **top_k**: Number of chains to retrieve (1–10, default 3)
    - **summarize**: Add a compression step for long multi-hop chains (costs 1 extra LLM call)
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="'question' must be non-empty.")
    _require_llm()

    # Retrieve under lock (touches shared indices); generate outside (stateless + slow).
    # Replicate answer()'s real path: score gate + hybrid coverage sentences.
    # (Calling generate(q, chains) alone ships degraded chain-only answers and
    # tanks fact questions — must pass coverage_sentences.)
    _CHAIN_GATE = 2.0
    with _rag_lock:
        _require_graph()
        chains = _rag.retrieve(req.question, top_k=req.top_k)
        if chains and max(c.rerank_score for c in chains) < _CHAIN_GATE:
            chains = []                     # coverage-only fallback (factual queries)
        chain_nodes = {n for c in chains for e in c.chain
                       for n in (e.cause, e.effect)}
        coverage = _rag._retrieve_sentences(
            req.question, k=max(6, req.top_k * 2), chain_nodes=chain_nodes or None)

    try:
        answer = _rag.generate(req.question, chains, summarize=req.summarize,
                               coverage_sentences=coverage)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}")

    return QueryResponse(
        answer=answer,
        chains=[_to_chain_model(c) for c in chains],
        question=req.question,
    )


@app.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
def retrieve(req: RetrieveRequest):
    """
    Retrieve causal chains without calling an LLM.

    Returns the top-k scored causal chains for the question. Use this when you
    want to do your own generation, reranking, or just inspect what the graph found.
    No LLM required — pure graph + vector retrieval.

    - **question**: Causal question to retrieve chains for
    - **top_k**: Number of chains to return (1–10, default 3)
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="'question' must be non-empty.")

    with _rag_lock:
        _require_graph()
        chains = _rag.retrieve(req.question, top_k=req.top_k)

    return RetrieveResponse(
        chains=[_to_chain_model(c) for c in chains],
        question=req.question,
    )


@app.post("/rootcause", response_model=GraphTraversalResponse, tags=["graph-traversal"])
def rootcause(req: EventRequest):
    """
    Trace root causes of an event by traversing the causal graph backward.

    Returns up to 12 causal chains that end at the event. No LLM required —
    instant pure-graph BFS traversal.

    - **event**: The effect or outcome to investigate (fuzzy-matched to a graph node)
    - **depth**: Maximum BFS depth (default 6)
    """
    with _rag_lock:
        _require_graph()
        node, chains = _rag.root_causes(req.event, max_depth=req.depth)

    if node is None:
        raise HTTPException(
            status_code=404,
            detail=f"No graph node found matching '{req.event}'. "
                   "Check /graph for available nodes.",
        )
    return GraphTraversalResponse(
        node=node,
        chains=[c.text() for c in chains[:12]],
        count=len(chains),
    )


@app.post("/impact", response_model=GraphTraversalResponse, tags=["graph-traversal"])
def impact(req: EventRequest):
    """
    Trace downstream impacts of an event by traversing the causal graph forward.

    Returns up to 12 causal chains starting at the event (blast radius).
    No LLM required — instant pure-graph BFS traversal.

    - **event**: The cause or event to trace (fuzzy-matched to a graph node)
    - **depth**: Maximum BFS depth (default 6)
    """
    with _rag_lock:
        _require_graph()
        node, chains = _rag.impact(req.event, max_depth=req.depth)

    if node is None:
        raise HTTPException(
            status_code=404,
            detail=f"No graph node found matching '{req.event}'. "
                   "Check /graph for available nodes.",
        )
    return GraphTraversalResponse(
        node=node,
        chains=[c.text() for c in chains[:12]],
        count=len(chains),
    )


@app.post("/path", response_model=PathResponse, tags=["graph-traversal"])
def path(req: PathRequest):
    """
    Find the shortest causal path between two events in the graph.

    Returns the single shortest path (or `found: false` if none exists).
    No LLM required — instant BFS.

    - **source**: Starting cause or event
    - **target**: Target effect or outcome
    - **depth**: Maximum path length to search (default 6)
    """
    with _rag_lock:
        _require_graph()
        src_node, tgt_node, chain = _rag.connect(req.source, req.target, max_depth=req.depth)

    if src_node is None:
        raise HTTPException(
            status_code=404,
            detail=f"No graph node found matching source '{req.source}'.",
        )
    if tgt_node is None:
        raise HTTPException(
            status_code=404,
            detail=f"No graph node found matching target '{req.target}'.",
        )
    return PathResponse(
        source_node=src_node,
        target_node=tgt_node,
        path=chain.text() if chain else None,
        found=chain is not None,
    )


@app.get("/graph", response_model=GraphStats, tags=["graph"])
def graph_stats():
    """Graph statistics and a sample of up to 20 edges."""
    with _rag_lock:
        edges = _edges(_rag)
        n_nodes = len(_rag.graph.nodes())
    sample = [
        GraphEdge(
            cause=e.cause,
            relation=e.relation,
            effect=e.effect,
            polarity=e.polarity,
            confidence=round(getattr(e, "confidence", 0.85), 4),
            source=e.source_sent[:120] if e.source_sent else "",
        )
        for e in edges[:20]
    ]
    return GraphStats(nodes=n_nodes, edges=len(edges), sample_edges=sample)


@app.delete("/graph", tags=["graph"])
def clear_graph():
    """Clear all edges and nodes from the in-memory graph."""
    global _rag
    with _rag_lock:
        old = _rag
        _rag = GraphRAG(dim=10000, llm=_llm)
        if getattr(old, "using_external", False):
            old.close()
    return {"status": "cleared"}
