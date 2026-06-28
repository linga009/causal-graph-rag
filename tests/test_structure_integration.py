"""Phase 2: document structure wired into GraphRAG ingest/answer."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph_rag.graph_rag import GraphRAG

STRUCTURED_DOC = """# Incident Report: Plant Outage

## Timeline
The reactor overheated during the night shift. The overheating caused the
coolant valve to fail. The valve failure triggered an emergency shutdown.

## Impact
The shutdown reduced power output. Lower power output disrupted hospital
operations across the district.
"""


def _ingested(schema="general"):
    rag = GraphRAG(dim=10000)          # MockLLM
    rag.ingest(STRUCTURED_DOC, schema=schema)
    return rag


def test_structure_index_built_on_ingest():
    rag = _ingested()
    assert rag._struct_index, "ingest should populate the structure index"
    # every entry carries location metadata
    _, meta, _ = rag._struct_index[0]
    assert "heading_path" in meta and "position" in meta and "synthesis" in meta


def test_evidence_annotated_with_heading_path():
    rag = _ingested()
    chains = rag.retrieve("What did the overheating ultimately disrupt?", top_k=3)
    ctx = rag._build_context(chains, structured=True)
    # evidence carries its heading-path breadcrumb, whichever section matched
    assert "[Incident Report: Plant Outage >" in ctx, ctx
    # and no raw markdown heading lines leaked into the evidence
    assert "##" not in ctx and "# Incident" not in ctx, ctx


def test_locate_matches_despite_rewrite():
    rag = _ingested()
    # a paraphrase / coref-style rewrite still locates to the right section
    meta = rag._locate("the emergency shutdown reduced the power output")
    assert meta is not None
    assert meta["heading_path"] == ["Incident Report: Plant Outage", "Impact"]


def test_headingless_doc_no_annotation():
    # Plain text without headings: annotation is a no-op (no spurious tags).
    rag = GraphRAG(dim=10000)
    rag.ingest("The reactor overheated. It caused a shutdown.")
    chains = rag.retrieve("What caused the shutdown?", top_k=3)
    ctx = rag._build_context(chains, structured=True)
    assert "[" not in ctx.split("Evidence:")[-1], "no heading tags expected"


def test_incident_preset_tags_roles():
    rag = _ingested(schema="incident")
    roles = {meta["role"] for _, meta, _ in rag._struct_index if meta["role"]}
    assert {"timeline", "impact"} <= roles
