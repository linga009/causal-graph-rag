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
    """Pick an LLM from whatever key is present; None if no key (spaCy-only)."""
    from llm_adapters import GroqLLM, GeminiLLM, AnthropicLLM, OpenAILLM
    if os.environ.get("GROQ_API_KEY"):
        return GroqLLM()
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiLLM()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM()
    return None


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def cmd_ingest(args) -> int:
    from graph_rag import GraphRAG
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
    from graph_rag import GraphRAG
    rag = GraphRAG.load(args.graph, llm=_build_llm())
    return _answer_and_print(rag, args.question, args.top_k, args.chains)


def cmd_ask(args) -> int:
    from graph_rag import GraphRAG
    llm = _build_llm()
    rag = GraphRAG(llm=llm)
    rag.ingest(_read(args.file), schema=args.schema,
               llm_extractor=(llm if args.llm_mode else None),
               llm_mode=args.llm_mode or "augment")
    return _answer_and_print(rag, args.question, args.top_k, args.chains)


def cmd_info(args) -> int:
    from graph_rag import GraphRAG
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


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn required: pip install fastapi uvicorn", file=sys.stderr)
        return 1
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)
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

    sp = sub.add_parser("info", help="show stats for a saved graph")
    sp.add_argument("graph")
    sp.set_defaults(func=cmd_info)

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
