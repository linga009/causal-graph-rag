"""
demo_agentic.py — agentic causal RAG over a small incident report.

Shows the LLM controller decomposing a multi-intent question into a sequence of
LLM-free graph operations (rootcause / impact / path / retrieve), observing each
result, and synthesizing a final answer with a full reasoning trace.

Run:  python demo_agentic.py
      (auto-picks GROQ_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_env(path: str = ".env") -> None:
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _build_llm():
    from causal_graph_rag.llm_adapters import build_llm
    return build_llm()


REPORT = """
Incident Report: Regional Data Center Outage

A cooling pump failed during routine operation. The pump failure caused the
server racks to overheat. The overheating triggered an automatic thermal
shutdown of the primary cluster. The shutdown caused a 9-hour service outage.
The outage disrupted hospital scheduling systems across the region. Operators
had proceeded with maintenance without proper authorization, and the on-call
engineer was not notified of the rising temperatures.
"""

QUESTION = ("Why did the service outage happen, and what did the cooling pump "
            "failure ultimately disrupt?")


def main() -> int:
    _load_env()
    llm = _build_llm()
    if llm is None:
        print("No LLM key found. Set GROQ_API_KEY / ANTHROPIC_API_KEY / "
              "GEMINI_API_KEY / OPENAI_API_KEY in .env.")
        return 1

    from causal_graph_rag.graph_rag import GraphRAG
    from causal_graph_rag.agentic_rag import AgenticCausalRAG

    print(f"LLM: {type(llm).__name__}\nIngesting incident report ...\n")
    rag = GraphRAG(llm=llm)
    rag.ingest(REPORT, schema="incident")

    agent = AgenticCausalRAG(rag, llm=llm, max_steps=6)
    print(f"Question: {QUESTION}\n")
    print("=" * 70)
    result = agent.run(QUESTION)

    print("REASONING TRACE")
    print("=" * 70)
    for step in result.steps:
        print(f"  {step}")
    print(f"\n  ({result.n_llm_calls} LLM calls)\n")

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
