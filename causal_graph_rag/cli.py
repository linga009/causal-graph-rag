"""
cli.py — command-line interface for Causal Graph RAG.

Turns the engine into a tool you can use without writing Python:

    # Build a graph from a document and save it (warm startup)
    causal-rag ingest report.md --save graph.pkl --schema auto --llm-mode augment

    # Ask questions against a saved graph
    causal-rag query graph.pkl "What was the root cause of the outage?"

    # One-shot: ingest + ask without saving
    causal-rag ask report.md "What did the fire ultimately disrupt?"

    # Inspect a saved graph
    causal-rag info graph.pkl

    # Serve the REST API
    causal-rag serve --port 8000

LLM selection is automatic from whichever API key is in the environment / .env
(Groq → Gemini → Anthropic → OpenAI); spaCy-only (no LLM) works for ingest.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_env(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _build_llm():
    from .llm_adapters import build_llm
    return build_llm()


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def cmd_ingest(args) -> int:
    from .graph_rag import GraphRAG
    llm = _build_llm()
    rag = GraphRAG(llm=llm)
    n = rag.ingest(_read(args.file), schema=args.schema,
                   llm_extractor=(llm if args.llm_mode else None),
                   llm_mode=args.llm_mode or "augment")
    print(f"Ingested {n} causal edges, {len(rag.graph.nodes())} nodes "
          f"(schema={args.schema}, llm={'none' if llm is None else type(llm).__name__}).")
    if args.save:
        rag.save(args.save)
        print(f"Saved to {args.save}")
    return 0


def _answer_and_print(rag, question: str, top_k: int, show_chains: bool) -> int:
    if rag.llm is None or type(rag.llm).__name__ == "MockLLM":
        print("[warn] No LLM configured — answer will be a placeholder. "
              "Set GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY.",
              file=sys.stderr)
    answer, chains = rag.answer(question, top_k=top_k)
    print(answer)
    if show_chains and chains:
        print("\nSupporting causal chains:")
        for c in chains:
            print(f"  {c.text()}")
    return 0


def cmd_query(args) -> int:
    from .graph_rag import GraphRAG
    rag = GraphRAG.load(args.graph, llm=_build_llm())
    return _answer_and_print(rag, args.question, args.top_k, args.chains)


def cmd_ask(args) -> int:
    from .graph_rag import GraphRAG
    llm = _build_llm()
    rag = GraphRAG(llm=llm)
    rag.ingest(_read(args.file), schema=args.schema,
               llm_extractor=(llm if args.llm_mode else None),
               llm_mode=args.llm_mode or "augment")
    return _answer_and_print(rag, args.question, args.top_k, args.chains)


def cmd_agent(args) -> int:
    """Agentic mode: an LLM controller plans a sequence of graph-tool calls
    (rootcause / impact / path / retrieve) to answer complex or multi-intent
    questions. Requires an LLM."""
    from .graph_rag import GraphRAG
    from .agentic_rag import AgenticCausalRAG
    llm = _build_llm()
    if llm is None or type(llm).__name__ == "MockLLM":
        print("error: agentic mode requires an LLM. Set GROQ_API_KEY / "
              "ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY.", file=sys.stderr)
        return 1
    rag = GraphRAG.load(args.graph, llm=llm)
    agent = AgenticCausalRAG(rag, llm=llm, max_steps=args.max_steps)
    result = agent.run(args.question)
    if args.trace:
        print("Reasoning trace:")
        for s in result.steps:
            print(f"  {s}")
        print(f"  ({result.n_llm_calls} LLM calls)\n")
    print(result.answer)
    return 0


def cmd_info(args) -> int:
    from .graph_rag import GraphRAG
    rag = GraphRAG.load(args.graph)
    edges = rag.graph.edges
    print(f"Graph: {args.graph}")
    print(f"  nodes:           {len(rag.graph.nodes())}")
    print(f"  causal edges:    {len(edges)}")
    print(f"  indexed sents:   {len(rag._struct_index)}")
    roles = sorted({m['role'] for _, m, _ in rag._struct_index if m.get('role')})
    if roles:
        print(f"  discourse roles: {', '.join(roles)}")
    print("  sample edges:")
    for e in edges[:8]:
        print(f"    {e.cause} -> ({e.relation}) {e.effect}")
    return 0


def _print_chains(chains, limit=12):
    if not chains:
        print("  (no causal chains found)")
        return
    for c in chains[:limit]:
        print(f"  {c.text()}")
    if len(chains) > limit:
        print(f"  ... and {len(chains) - limit} more")


def cmd_rootcause(args) -> int:
    """Backward causal chains into an event — its root causes. No LLM needed."""
    from .graph_rag import GraphRAG
    rag = GraphRAG.load(args.graph)
    node, chains = rag.root_causes(args.event, max_depth=args.depth)
    if node is None:
        print(f"error: no node matching {args.event!r} in the graph", file=sys.stderr)
        return 1
    print(f"Root-cause chains into '{node}':")
    _print_chains(chains)
    return 0


def cmd_impact(args) -> int:
    """Forward causal chains out of an event — its downstream impact / blast radius."""
    from .graph_rag import GraphRAG
    rag = GraphRAG.load(args.graph)
    node, chains = rag.impact(args.event, max_depth=args.depth)
    if node is None:
        print(f"error: no node matching {args.event!r} in the graph", file=sys.stderr)
        return 1
    print(f"Impact chains from '{node}':")
    _print_chains(chains)
    return 0


def cmd_path(args) -> int:
    """Shortest causal path from one event to another."""
    from .graph_rag import GraphRAG
    rag = GraphRAG.load(args.graph)
    s, d, chain = rag.connect(args.src, args.dst, max_depth=args.depth)
    if s is None or d is None:
        miss = args.src if s is None else args.dst
        print(f"error: no node matching {miss!r} in the graph", file=sys.stderr)
        return 1
    if chain is None:
        print(f"No causal path from '{s}' to '{d}'.")
        return 0
    print(f"Causal path '{s}' -> '{d}':")
    print(f"  {chain.text()}")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn required: pip install fastapi uvicorn", file=sys.stderr)
        return 1
    uvicorn.run("causal_graph_rag.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="causal-rag",
                                description="Causal Graph RAG — build, query, and serve causal knowledge graphs.")
    p.add_argument("-v", "--verbose", action="store_true", help="enable info logging")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--schema", default="general",
                        help="document-structure preset: general/research/clinical/incident/auto")
        sp.add_argument("--llm-mode", choices=["augment", "full"], default=None,
                        help="use the LLM during extraction (default: spaCy only)")

    sp = sub.add_parser("ingest", help="build a graph from a document and optionally save it")
    sp.add_argument("file")
    sp.add_argument("--save", help="path to save the built graph (.pkl)")
    add_common(sp)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("query", help="ask a question against a saved graph")
    sp.add_argument("graph")
    sp.add_argument("question")
    sp.add_argument("--top-k", type=int, default=3)
    sp.add_argument("--chains", action="store_true", help="also print supporting chains")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("ask", help="ingest a document and ask in one shot (no save)")
    sp.add_argument("file")
    sp.add_argument("question")
    sp.add_argument("--top-k", type=int, default=3)
    sp.add_argument("--chains", action="store_true")
    add_common(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("agent", help="agentic mode: LLM plans multi-step graph-tool calls")
    sp.add_argument("graph")
    sp.add_argument("question")
    sp.add_argument("--max-steps", type=int, default=6)
    sp.add_argument("--trace", action="store_true", help="print the reasoning trace")
    sp.set_defaults(func=cmd_agent)

    sp = sub.add_parser("info", help="show stats for a saved graph")
    sp.add_argument("graph")
    sp.set_defaults(func=cmd_info)

    # -- direct causal graph queries (no LLM; what flat RAG cannot do) -------- #
    sp = sub.add_parser("rootcause", help="root-cause chains INTO an event (backward)")
    sp.add_argument("graph"); sp.add_argument("event")
    sp.add_argument("--depth", type=int, default=6)
    sp.set_defaults(func=cmd_rootcause)

    sp = sub.add_parser("impact", help="impact / blast-radius chains OUT of an event (forward)")
    sp.add_argument("graph"); sp.add_argument("event")
    sp.add_argument("--depth", type=int, default=6)
    sp.set_defaults(func=cmd_impact)

    sp = sub.add_parser("path", help="shortest causal path between two events")
    sp.add_argument("graph"); sp.add_argument("src"); sp.add_argument("dst")
    sp.add_argument("--depth", type=int, default=6)
    sp.set_defaults(func=cmd_path)

    sp = sub.add_parser("serve", help="run the REST API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}", file=sys.stderr)
        return 2
    except Exception as e:  # surface a clean message, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
