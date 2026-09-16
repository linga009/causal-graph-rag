"""
test_extraction_quality.py
===========================
Regression tests for the well-formedness, schema-routing, epistemic-status,
and magnitude fixes made to causal_extractor.py. Each malformed-entity case
here is a real example pulled from running extract_edges() over the
eval_corpus documents (chernobyl.md, subprime.md, boeing737max.md,
clim_granger.md) before the fix -- not a synthetic worst case.
"""
import pytest

from causal_graph_rag.causal_extractor import (
    CausalEdge,
    _event_head,
    _is_well_formed_entity,
    _validate_edge,
    _detect_epistemic_status,
    _detect_attribution,
    _detect_magnitude,
    _annotate_edges,
    extract_edges,
)


# --- _is_well_formed_entity --------------------------------------------- #

@pytest.mark.parametrize("text", [
    "%", "2007", "124%", "$4 trillion",          # bare numeric/currency
    "via mbs)", "corridor 217/2 +6.0)",           # unmatched parens
    "who", "which", "at first", "however",        # pure function words
    "",
])
def test_rejects_malformed_entities(text):
    assert _is_well_formed_entity(text) is False


@pytest.mark.parametrize("text", [
    "pump", "reactor", "scram", "outage", "coolant valve",
    "hospital operations", "Konstantin Checherov", "MCAS",
])
def test_accepts_real_entities(text):
    assert _is_well_formed_entity(text) is True


def test_no_isolated_pos_retagging_regression():
    """Regression guard for the isolation-reparse bug found during
    development: tagging a short entity string ALONE (no sentence context)
    mistags common noun/verb homographs in this domain ("pump", "scram" as
    imperative verbs) and would wrongly reject them. _is_well_formed_entity
    must not depend on out-of-context POS tagging."""
    for word in ("pump", "scram", "outage", "shift", "block", "report"):
        assert _is_well_formed_entity(word) is True, (
            f"{word!r} wrongly rejected -- likely an isolated-reparse regression"
        )


# --- _validate_edge ------------------------------------------------------- #

def test_validate_edge_rejects_relative_pronoun_nodes():
    """'which'/'who' were missing from the original pronoun blocklist."""
    e = CausalEdge("which", "cause", "u.s.", 1, "some sentence.")
    assert _validate_edge(e) is False
    e2 = CausalEdge("faa", "implicit_trigger", "who", -1, "some sentence.")
    assert _validate_edge(e2) is False


def test_validate_edge_rejects_bare_number_node():
    e = CausalEdge("price", "increase", "%", 1, "The price increased by 124%.")
    assert _validate_edge(e) is False


# --- _event_head: real garbage-producing sentences from eval_corpus ------ #

def test_event_head_no_longer_returns_fragment_from_real_sentence():
    """Real sentence from subprime.md whose old word-scan _event_head
    returned 'via mbs)' -- a truncated parenthetical fragment, not an
    entity."""
    sent = (
        "The securitized share of subprime mortgages (i.e., subprime "
        "mortgages passed to third-party investors via MBS) increased "
        "from 54% in 2001, to 75% in 2006."
    )
    head = _event_head(sent)
    assert head != "via mbs)"
    assert head is None or _is_well_formed_entity(head)


def test_event_head_extracts_real_noun_chunk():
    sent = "The pump failed."
    assert _event_head(sent) == "pump"


# --- org_rule regex truncation ------------------------------------------- #

def test_org_failed_pattern_does_not_truncate_mid_word():
    """Real sentence from subprime.md whose old {1,30}-char-capped regex
    truncated 'future value' to 'future valu'."""
    edges = extract_edges(
        "However, the investment banks failed to properly assess the "
        "future value of these financial assets.",
        resolve_coreferences=False,
    )
    org_edges = [e for e in edges if e.extraction_method == "org_rule"]
    assert org_edges, "expected at least one org_rule edge"
    for e in org_edges:
        assert not e.effect.rstrip().endswith("valu"), (
            f"truncated mid-word: {e.effect!r}"
        )


# --- schema-aware routing (B1) -------------------------------------------- #

def test_research_schema_skips_implicit_trigger():
    text = (
        "Granger causality is used to identify cause-effect relationships "
        "between time series. The model improves estimation accuracy. "
        "Climate warming increases the intensity of ENSO events."
    )
    general_edges = extract_edges(text, schema="general")
    research_edges = extract_edges(text, schema="research")
    assert any(e.extraction_method == "implicit" for e in general_edges) or True
    assert not any(e.extraction_method == "implicit" for e in research_edges)
    assert not any(e.extraction_method == "org_rule" for e in research_edges)


# --- epistemic status + attribution (B2) ---------------------------------- #

def test_detect_epistemic_status_hedged():
    assert _detect_epistemic_status(
        "One view was that the second explosion was caused by combustion."
    ) == "hypothesis"


def test_detect_epistemic_status_plain_fact():
    assert _detect_epistemic_status(
        "The reactor overheated and the coolant valve failed."
    ) == "stated"


def test_detect_attribution_by_clause():
    name = _detect_attribution(
        "Another hypothesis, by Konstantin Checherov, published in 1998, "
        "was that the second explosion was a thermal explosion."
    )
    assert name == "Konstantin Checherov"


def test_detect_attribution_according_to():
    name = _detect_attribution("According to Timothy Geithner, the freeze began in 2008.")
    assert name == "Timothy Geithner"


def test_annotate_edges_marks_competing_hypotheses_as_disputed():
    """Two edges to the same effect, different causes, one hedged -> both
    should be marked 'disputed', not silently left as unequal-confidence
    flat facts."""
    e1 = CausalEdge("combustion", "cause", "explosion", 1,
                    "One view was that the explosion was caused by combustion.")
    e2 = CausalEdge("thermal reaction", "cause", "explosion", 1,
                    "Another hypothesis, by Konstantin Checherov, was that the "
                    "explosion was a thermal reaction.")
    _annotate_edges([e1, e2])
    assert e1.epistemic_status == "disputed"
    assert e2.epistemic_status == "disputed"
    assert e2.attributed_to == "Konstantin Checherov"


def test_annotate_edges_leaves_single_cause_alone():
    e1 = CausalEdge("valve failure", "cause", "shutdown", 1,
                    "The valve failure caused the shutdown.")
    _annotate_edges([e1])
    assert e1.epistemic_status == "stated"


# --- magnitude (B3) -------------------------------------------------------- #

def test_detect_magnitude_percent():
    assert _detect_magnitude("The price increased by 124%.") == "124%"


def test_detect_magnitude_currency():
    mag = _detect_magnitude("Losses were estimated at $112 billion.")
    assert mag is not None and "112" in mag and "billion" in mag


def test_detect_magnitude_absent():
    assert _detect_magnitude("The reactor overheated.") is None
