"""
demo_graph.py — Consequential-Graph VSA-RAG demonstration.

Shows the capability that pure similarity-search RAG structurally cannot have:
recovering multi-hop cause->effect chains that chunking + embedding destroy.

    python demo_graph.py
"""

from causal_graph_rag.graph_rag import GraphRAG
from causal_graph_rag.causal_extractor import extract_edges

DOC = """
The reactor overheated due to a faulty temperature sensor.
As a result, the coolant valve failed.
This triggered an emergency shutdown.
The shutdown caused a regional power outage.
The power outage disrupted hospital operations.
Separately, budget cuts reduced routine inspection frequency.
Reduced inspections increased equipment failure risk.
"""

QUERIES = [
    ("What did the reactor overheating ultimately cause?", "forward multi-hop"),
    ("What did the overheating ultimately disrupt?",       "5-hop to endpoint"),
    ("Why did the power outage happen?",                   "backward root-cause"),
    ("What caused the emergency shutdown?",                "backward"),
    ("What did budget cuts lead to?",                      "separate chain"),
]


def main():
    rag = GraphRAG(dim=10000, semantic_weight=0)
    n = rag.ingest(DOC)

    print("EXTRACTED CAUSAL GRAPH")
    print("=" * 64)
    for e in extract_edges(DOC):
        print(f"  {e.text()}")
    print(f"\n{n} directed edges. Nodes: {sorted(rag.graph.nodes())}\n")

    print("QUERIES — each returns a whole chain, not isolated chunks")
    print("=" * 64)
    for q, label in QUERIES:
        ans, chains = rag.answer(q, top_k=1)
        print(f"\nQ ({label}): {q}")
        if chains:
            c = chains[0]
            print(f"   [{c.direction}] {c.text()}")
            print(f"   spans {len(c.provenance())} source sentence(s) / chunks")
        else:
            print("   (no causal match)")


if __name__ == "__main__":
    main()
