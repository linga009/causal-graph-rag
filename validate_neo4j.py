"""
validate_neo4j.py — end-to-end validation against a REAL Neo4j server.

Prerequisite: a running Neo4j (e.g. via Docker):
    docker run -d --name neo4j-test -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/testpassword neo4j:latest

Run:
    python validate_neo4j.py

Validates:
  1. Connection + schema init
  2. add_edge assigns DISTINCT edge_ids (regression: all were 0)
  3. add_edges() bulk insert
  4. forward / backward traversal returns full chains
  5. VSA triple scoring
  6. GraphRAG end-to-end query through the Neo4j backend
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpassword")

CORPUS = (
    "The reactor overheated. The coolant valve failed. "
    "This triggered a shutdown. The shutdown caused an outage. "
    "The outage disrupted hospital operations."
)


def _ok(label, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    from causal_graph_rag.neo4j_graph import Neo4jCausalGraph
    from causal_graph_rag.causal_extractor import CausalEdge, extract_edges
    from causal_graph_rag.vsa_core import Lexicon, Triple
    from causal_graph_rag.graph_rag import GraphRAG

    lex = Lexicon(dim=10000, semantic_weight=0)

    print(f"Connecting to {URI} ...")
    g = Neo4jCausalGraph(uri=URI, user=USER, password=PASSWORD, lex=lex, clear_on_init=True)
    print("Connected + schema initialized.\n")

    all_pass = True

    # 1. add_edge distinct ids
    print("1. add_edge — distinct edge ids")
    for c, r, e in [("reactor", "overheat", "valve"), ("valve", "cause", "shutdown")]:
        g.add_edge(CausalEdge(c, r, e, 1, f"{c} {r} {e}"))
    edges = g._get_edges()
    ids = [e.edge_id for e in edges]
    all_pass &= _ok("edge_ids distinct", len(set(ids)) == len(ids), f"ids={ids}")

    # 2. bulk add_edges
    print("2. add_edges — bulk UNWIND insert")
    g.add_edges([
        CausalEdge("shutdown", "cause", "outage", 1, "shutdown caused outage"),
        CausalEdge("outage", "disrupt", "hospital operations", -1, "outage disrupted hospital operations"),
    ])
    edges = g._get_edges()
    ids = [e.edge_id for e in edges]
    all_pass &= _ok("all 4 edges present, ids distinct", len(edges) == 4 and len(set(ids)) == 4, f"ids={ids}")

    # 3. nodes
    print("3. nodes()")
    nodes = g.nodes()
    all_pass &= _ok("expected nodes present", {"reactor", "valve", "shutdown", "outage"} <= nodes, f"n={len(nodes)}")

    # 4. forward traversal
    print("4. forward_chain('reactor')")
    fwd = g.forward_chain("reactor", max_depth=6)
    reaches = any(p and p[-1].effect == "hospital operations" for p in fwd)
    all_pass &= _ok("reactor -> ... -> hospital operations", reaches, f"{len(fwd)} path(s)")

    # 5. backward traversal (root-cause order)
    print("5. backward_chain('hospital operations')")
    bwd = g.backward_chain("hospital operations", max_depth=6)
    roots = any(p and p[0].cause == "reactor" for p in bwd)
    all_pass &= _ok("root cause = reactor (chain in causal order)", roots, f"{len(bwd)} path(s)")

    # 6. VSA scoring
    print("6. score_edges_by_triple")
    scored = g.score_edges_by_triple(Triple("shutdown", "cause", "outage"), top_k=3)
    top_is_match = scored and scored[0][1].cause == "shutdown" and scored[0][1].effect == "outage"
    all_pass &= _ok("top VSA match is the shutdown->outage edge", bool(top_is_match), f"top sim={scored[0][0]:.3f}" if scored else "none")

    g.close()

    # 7. full GraphRAG through Neo4j backend
    print("7. GraphRAG end-to-end via Neo4j backend")
    rag = GraphRAG(dim=10000, neo4j_uri=URI, neo4j_user=USER, neo4j_password=PASSWORD)
    rag.graph.clear_on_init = False
    # clear and re-ingest cleanly
    rag.graph._clear_all()
    rag.graph._next_edge_id = 0
    n = rag.ingest(CORPUS)
    chains = rag.retrieve("What did the overheating ultimately disrupt?", top_k=3)
    joined = " ".join(c.text() for c in chains).lower()
    all_pass &= _ok("end-to-end retrieval finds the disruption", "operations" in joined, f"{n} edges, {len(chains)} chains")
    if chains:
        print(f"       top chain: {chains[0].text()}")
    rag.close()

    print()
    print("=" * 60)
    print("RESULT:", "ALL CHECKS PASSED ✓" if all_pass else "SOME CHECKS FAILED ✗")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
