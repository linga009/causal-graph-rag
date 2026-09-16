"""
test_parser.py
===============
Regression tests for parser.py's dependency-free fallback triple extractor
(_fallback_triples), which is the ONLY path exercised in CI (ci.yml never
installs spaCy -- see pyproject.toml's optional 'spacy' extra) and is also
reached whenever _spacy_triples finds nothing for a given sentence, even
with spaCy installed.
"""
from causal_graph_rag.parser import parse_triples, _fallback_triples


def test_plural_noun_not_mistaken_for_verb():
    """Real sentence from chernobyl.md: the old suffix-only ("-s"/"-es"/"-ed")
    pseudo-verb guesser mistook the plural noun 'eyewitnesses' for the
    clause's verb, producing a garbage triple (agent='calm according',
    action='eyewitnesses', patient='at time'). No known verb is actually
    present in this clause, so the correct answer is no triple at all, not
    a wrong guess."""
    s = ("The atmosphere in the control room at that point was calm, "
         "according to eyewitnesses, and there were no active emergency "
         "signals at that time.")
    triples = _fallback_triples(s)
    for t in triples:
        assert t.action != "eyewitnesses"


def test_ed_suffix_still_detected_as_verb():
    """-ed remains a reliable verb-only inflection and should still work
    as the fallback's suffix guess."""
    triples = _fallback_triples("The valve corroded rapidly.")
    assert any(t.action == "corroded" for t in triples)


def test_known_causal_verb_still_preferred_over_suffix_guess():
    triples = parse_triples("The pump caused the reactor to overheat.")
    assert any(t.action == "cause" for t in triples)
