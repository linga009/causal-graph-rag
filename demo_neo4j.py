"""
demo_neo4j.py
=============
Demo of Neo4j-backed causal graph for persistent, large-scale graphs.

Requires a Neo4j instance running. Start with:
  docker run -p 7687:7687 -p 7474:7474 neo4j:latest

Then update the connection parameters below.

Usage
-----
  python demo_neo4j.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def demo_neo4j():
    """Demonstrate Neo4j backend with demo corpus."""
    from graph_rag import GraphRAG

    # Connection parameters (adjust for your setup)
    NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    print("=" * 70)
    print("  Causal Graph RAG — Neo4j Backend Demo")
    print("=" * 70)
    print(f"\nConnecting to: {NEO4J_URI}\n")

    try:
        # Initialize with Neo4j backend
        rag = GraphRAG(
            dim=10000,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
        )
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        print("\nMake sure Neo4j is running. You can start it with:")
        print("  docker run -p 7687:7687 -p 7474:7474 neo4j:latest")
        return

    # Demo corpus
    text = """
    The temperature sensor failed, which led to reactor overheating.
    The reactor overheated, which caused the coolant valve to jam.
    The jammed valve triggered the emergency shutdown.
    The emergency shutdown caused a power outage.
    The power outage disrupted hospital operations.
    Budget cuts reduced the inspection frequency.
    Reduced inspection frequency increased the risk of equipment failure.
    """

    print("Ingesting demo corpus...")
    n_edges = rag.ingest(text)
    print(f"  Extracted {n_edges} causal edges")
    print(f"  Graph nodes: {len(rag.graph.nodes())}\n")

    # Demo queries
    questions = [
        "What did the reactor overheating ultimately cause?",
        "Why did the power outage happen?",
        "What caused the emergency shutdown?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer, chains = rag.answer(q, top_k=3)
        print(f"A: {answer}\n")

        for i, chain in enumerate(chains, 1):
            print(f"  Chain {i}: {chain.text()}")
            print(f"    Provenance: {chain.provenance()}\n")

    # Cleanup
    print("\nClosing Neo4j connection...")
    rag.close()
    print("Done!")


if __name__ == "__main__":
    demo_neo4j()
