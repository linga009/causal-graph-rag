import pytest
from vsa_core import Lexicon, Triple, encode_triple, hamming_similarity

def test_semantic_weight_schedule():
    """Semantic weight should increase with document complexity."""
    lex_short = Lexicon(dim=1000, semantic_weight=1)
    lex_long = Lexicon(dim=1000, semantic_weight=1)

    weight_short = lex_short.semantic_weight_schedule(doc_length=500)   # 50 sentences
    weight_long = lex_long.semantic_weight_schedule(doc_length=5000)    # 500 sentences

    assert weight_short < weight_long, "Long docs should have higher semantic weight"
    assert weight_short >= 1, "Minimum weight is 1 (identity only)"
    assert weight_long <= 5, "Maximum weight capped at 5"

def test_semantic_weight_similarity_robustness():
    """Higher semantic weight should increase synonym similarity."""
    lex1 = Lexicon(dim=1000, semantic_weight=1)
    lex2 = Lexicon(dim=1000, semantic_weight=4)

    t_unemp = Triple("unemployment", "causes", "social_unrest")
    t_jobless = Triple("joblessness", "causes", "civil_disorder")

    v1_unemp = encode_triple(t_unemp, lex1)
    v1_jobless = encode_triple(t_jobless, lex1)
    sim1 = hamming_similarity(v1_unemp, v1_jobless)

    v2_unemp = encode_triple(t_unemp, lex2)
    v2_jobless = encode_triple(t_jobless, lex2)
    sim2 = hamming_similarity(v2_unemp, v2_jobless)

    assert sim2 > sim1, "Higher semantic weight should increase synonym similarity"
