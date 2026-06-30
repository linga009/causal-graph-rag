"""
Pytest session setup.

Warm the process-wide sentence-transformer model ONCE, before any test (and
before mongomock/pymongo are imported). Every later GraphRAG construction reuses
this cached model via `shared_st_model()` instead of re-loading it. This both
speeds the suite up and sidesteps a native crash seen on some toolchains where
loading torch's libraries *after* pymongo's C extension is imported aborts the
process — by ensuring the only torch load happens first.
"""
try:
    from causal_graph_rag.retrievers import shared_st_model
    shared_st_model()          # one-time load into the process-wide cache
except Exception:
    pass                       # no sentence-transformers -> HashingDense fallback
