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

    # Use word pairs with strong trigram overlap (employment/unemployment share "employ")
    t1 = Triple("employment", "causes", "employment_gains")
    t2 = Triple("unemployment", "causes", "unemployment_crisis")

    v1_t1 = encode_triple(t1, lex1)
    v1_t2 = encode_triple(t2, lex1)
    sim1 = hamming_similarity(v1_t1, v1_t2)

    v2_t1 = encode_triple(t1, lex2)
    v2_t2 = encode_triple(t2, lex2)
    sim2 = hamming_similarity(v2_t1, v2_t2)

    assert sim2 > sim1, "Higher semantic weight should increase synonym similarity"
