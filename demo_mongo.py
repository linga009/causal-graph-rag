"""
demo_mongo.py — Causal Graph RAG running on MongoDB.

Stores cause->effect edges as documents, then traverses them with MongoDB's
native $graphLookup (downstream impact set / upstream root-cause set) — no
client-side graph engine, no LLM in the path.

    python demo_mongo.py                                 # mongomock (no server)
    MONGO_URI="mongodb+srv://..." python demo_mongo.py   # real MongoDB / Atlas
"""
from __future__ import annotations
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Load the sentence-transformer once BEFORE pymongo is imported (some toolchains
# abort if torch's libs load after pymongo's C extension). Harmless if absent.
import io
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_err = sys.stderr
sys.stderr = io.StringIO()
try:
    from causal_graph_rag.retrievers import shared_st_model
    shared_st_model()
except Exception:
    pass
finally:
    sys.stderr = _err

B, DIM, GRN, CYN, YEL, MAG, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[36m", "\033[33m", "\033[35m", "\033[0m")


def line(s="", pause=0.5):
    print(s); sys.stdout.flush(); time.sleep(pause)


DOC = ("The reactor overheated. The coolant valve failed. This triggered an "
       "emergency shutdown. The shutdown caused a power outage. The power "
       "outage disrupted hospital operations.")


def main() -> int:
    from causal_graph_rag import GraphRAG
    from causal_graph_rag.mongo_graph import MongoCausalGraph

    uri = os.environ.get("MONGO_URI")
    if uri:
        rag = GraphRAG(dim=2000, mongo_uri=uri, mongo_db="causal_rag_demo")
        rag.graph.col.delete_many({})
        where = f"Atlas ({uri.split('@')[-1][:28]}…)"
    else:
        import mongomock
        rag = GraphRAG(dim=2000)                       # in-memory init (ST cached)
        rag.graph = MongoCausalGraph(client=mongomock.MongoClient(),
                                     db_name="causal_rag_demo", lex=rag.lex,
                                     clear_on_init=True)
        rag.using_mongo = rag.using_external = True
        where = "mongomock (in-process)"

    line(f"\n{B}{MAG}Causal Graph RAG on MongoDB{RST}  {DIM}— traverse cause→effect with $graphLookup{RST}\n", 0.8)
    line(f'{DIM}# document:  "{DOC}"{RST}\n', 1.0)

    n = rag.ingest(DOC, schema="incident")
    g = rag.graph
    line(f"{GRN}✓{RST} wrote {B}{n}{RST} causal edges as documents to "
         f"{CYN}'causal_edges'{RST}  {DIM}({where}){RST}\n", 1.0)

    # True source (cause, never an effect) and sink (effect, never a cause)
    causes = {e.cause for e in g.edges}
    effects = {e.effect for e in g.edges}
    seed = sorted(causes - effects)[0] if (causes - effects) else next(iter(causes), "")
    leaf = sorted(effects - causes)[0] if (effects - causes) else next(iter(effects), "")

    line(f"{DIM}# native MongoDB traversal — runs IN the database:{RST}")
    line(f"{CYN}db.causal_edges.aggregate([{RST}")
    line(f'{CYN}  {{ $match: {{ cause: "{seed}" }} }},{RST}')
    line(f'{CYN}  {{ $graphLookup: {{ from:"causal_edges", startWith:"$effect",{RST}')
    line(f'{CYN}      connectFromField:"effect", connectToField:"cause",{RST}')
    line(f'{CYN}      as:"downstream", maxDepth:5, depthField:"depth" }} }} ]){RST}\n', 0.6)

    t = time.perf_counter()
    impact = g.reachable(seed, "forward")
    ms = (time.perf_counter() - t) * 1000
    line(f"{B}Downstream impact of{RST} '{seed}'  {DIM}(hop depth):{RST}")
    for node, d in sorted((impact or {}).items(), key=lambda kv: kv[1]):
        line(f"  {YEL}{d}↳{RST} {node}", 0.18)
    line(f"  {DIM}$graphLookup in {ms:.2f} ms{RST}\n", 1.0)

    line(f"{B}Root causes of{RST} '{leaf}':  {CYN}" +
         ", ".join(sorted(g.reachable(leaf, 'backward') or {}, key=lambda k: -(g.reachable(leaf,'backward') or {}).get(k,0))) + RST, 0.8)

    chains = g.backward_chain(leaf, max_depth=8)
    if chains:
        line(f"\n{DIM}# full causal chain (BFS path enumeration):{RST}")
        line(f"  {CYN}{g.chain_text(max(chains, key=len))}{RST}\n", 1.0)

    line(f"{B}{YEL}→ graph traversal + causal reasoning, natively on MongoDB.{RST}", 0.6)
    line(f"{DIM}  GraphRAG(mongo_uri=...)  ·  pairs with Atlas Vector Search for the dense channel{RST}\n", 1.2)
    rag.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
