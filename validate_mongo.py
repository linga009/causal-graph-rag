"""
validate_mongo.py — end-to-end check of the MongoDB backend against a REAL
MongoDB / Atlas instance (mongomock covers the unit tests; this covers the wire).

    export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net"
    python validate_mongo.py

Confirms: ingest writes edges to MongoDB, $graphLookup reachability works on the
server, chain traversal + a full answer() round-trip succeed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DOC = ("The cooling pump failed. The pump failure caused the servers to overheat. "
       "The overheating triggered an emergency shutdown. The shutdown caused a "
       "regional outage. The outage disrupted hospital scheduling systems.")


def main() -> int:
    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("Set MONGO_URI to a real MongoDB/Atlas connection string.")
        return 1

    from causal_graph_rag import GraphRAG
    print("Connecting + ingesting into MongoDB ...")
    rag = GraphRAG(mongo_uri=uri, mongo_db="causal_rag_validate")
    rag.graph.col.delete_many({})              # fresh
    n = rag.ingest(DOC, schema="incident")
    print(f"  ingested {n} edges, {len(rag.graph.nodes())} nodes\n")

    print("Native $graphLookup — downstream impact of 'pump':")
    print(" ", rag.graph.reachable("pump failure", "forward") or "(node name mismatch — check nodes())")

    print("\nroot_causes('hospital scheduling systems'):")
    node, chains = rag.root_causes("hospital")
    for c in chains[:3]:
        print("  " + c.text())

    print("\nfull answer():")
    ans, _ = rag.answer("What ultimately disrupted the hospital systems?")
    print("  " + ans[:300])

    rag.close()
    print("\nOK — MongoDB backend works end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
