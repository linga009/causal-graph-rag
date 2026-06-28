"""Tests for the structure-preserving document parser."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph_rag.doc_structure import parse, detect_schema, detect_role, SYNTHESIS_ROLES


RESEARCH_DOC = """# Causal Graph RAG for Clinical Narratives

## Abstract
We present a retrieval method that preserves causal structure. It improves
recall on multi-hop questions. The main finding is a 71% recall gain over
flat retrieval on clinical cascades.

## Introduction
Standard RAG chunks documents and embeds them. This destroys causal links
between sentences. We address this gap.

## Methods
We extract causal edges with spaCy and an LLM. Edges are encoded with VSA.
The graph is traversed to return whole chains.

## Results
Recall improved from 0.31 to 0.53 on the healthcare benchmark. Faithfulness
rose to 0.77. Precision remained high.

## Conclusion
Causal structure matters for retrieval. The abstract's claim is supported by
the results section.
"""


def test_general_is_default_no_roles():
    # Default schema is general: structure is parsed, but no domain role labels.
    ds = parse(RESEARCH_DOC, doc_id="paper")
    assert ds.schema == "general"
    assert all(s.role is None for s in ds.sections())


def test_auto_detects_research():
    ds = parse(RESEARCH_DOC, doc_id="paper", schema="auto")
    assert ds.schema == "research"


def test_research_preset_roles():
    ds = parse(RESEARCH_DOC, doc_id="paper", schema="research")
    roles = {s.role for s in ds.sections() if s.role}
    assert {"abstract", "introduction", "methods", "results", "conclusion"} <= roles


def test_hierarchy_edges_are_domain_agnostic():
    # CONTAINS/FOLLOWS exist regardless of schema — structure is universal.
    for schema in ("general", "research"):
        ds = parse(RESEARCH_DOC, doc_id="paper", schema=schema)
        edges = ds.structural_edges()
        assert [e for e in edges if e[1] == "contains"], f"no CONTAINS ({schema})"
        assert [e for e in edges if e[1] == "follows"], f"no FOLLOWS ({schema})"


def test_sentence_role_lookup():
    ds = parse(RESEARCH_DOC, doc_id="paper", schema="research")
    abstract_sents = [s for s in ds.sentences() if ds.role_of(s.block_id) == "abstract"]
    assert any("main finding" in s.text for s in abstract_sents)


def test_synthesis_roles_present():
    ds = parse(RESEARCH_DOC, doc_id="paper", schema="research")
    section_roles = {s.role for s in ds.sections() if s.role}
    assert section_roles & SYNTHESIS_ROLES, "expected at least one synthesis-bearing section"


def test_no_content_dropped():
    ds = parse(RESEARCH_DOC, doc_id="paper")
    joined = " ".join(s.text for s in ds.sentences())
    for needle in ["71% recall gain", "0.31 to 0.53", "VSA", "abstract's claim"]:
        assert needle in joined, f"content lost: {needle}"


def test_clinical_preset_selected_by_user():
    note = (
        "Subjective\nPatient reports chest pain.\n\n"
        "Objective\nBP 150/95. ECG abnormal.\n\n"
        "Assessment\nLikely acute coronary syndrome.\n\n"
        "Plan\nAdmit and start anticoagulation.\n"
    )
    ds = parse(note, doc_id="note", schema="clinical")
    roles = {s.role for s in ds.sections() if s.role}
    assert {"subjective", "objective", "assessment", "plan"} <= roles


def test_unknown_schema_rejected():
    import pytest
    with pytest.raises(ValueError):
        parse(RESEARCH_DOC, schema="nonsense")


NESTED_DOC = """# Experiments
Intro to experiments.

## Results
We observed a clear effect.

## Ablations
Removing the module hurt performance.
"""


def test_heading_path_nesting():
    ds = parse(NESTED_DOC, doc_id="paper")
    # the sentence under "## Results" should carry the full breadcrumb
    target = next(s for s in ds.sentences() if "clear effect" in s.text)
    assert ds.heading_path(target.block_id) == ["Experiments", "Results"]
    ablation = next(s for s in ds.sentences() if "Removing the module" in s.text)
    assert ds.heading_path(ablation.block_id) == ["Experiments", "Ablations"]


def test_position_monotonic():
    ds = parse(RESEARCH_DOC, doc_id="paper")
    sents = ds.sentences()
    positions = [ds.position_of(s.block_id) for s in sents]
    assert positions == sorted(positions)
    assert 0.0 <= positions[0] <= positions[-1] <= 1.0


def test_synthesis_score_prefers_summarizing_sections():
    # Synthesis sections (abstract / conclusion) restate the paper briefly, so
    # they should outrank descriptive sections (methods / introduction) — with
    # NO domain role table involved (general signal). We do NOT privilege any
    # specific section name: the top section is simply whichever summarizes most.
    ds = parse(RESEARCH_DOC, doc_id="paper")  # schema='general'
    scores = ds.synthesis_scores()            # token-based, no embeddings
    by_title = {ds.blocks[sid].text: sc for sid, sc in scores.items()}
    assert by_title, "expected per-section scores"

    top = max(by_title, key=by_title.get)
    assert top in {"Abstract", "Conclusion"}, f"top should summarize, got {top!r} ({by_title})"
    # both synthesis sections beat the descriptive ones
    assert by_title["Abstract"] > by_title["Methods"]
    assert by_title["Conclusion"] > by_title["Introduction"]
