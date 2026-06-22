"""
demo_graph_live.py
==================
Same causal-graph demo as demo_graph.py but with a real LLM answering.

The LLM is selected in priority order:
  1. ANTHROPIC_API_KEY  -> AnthropicLLM (claude-opus-4-8)
  2. OPENAI_API_KEY     -> OpenAILLM    (gpt-4o)
  3. GROQ_API_KEY       -> GroqLLM      (llama-3.1-8b-instant)
  4. none found         -> MockLLM      (offline, with instructions printed)

Usage:
  # Windows
  set ANTHROPIC_API_KEY=sk-ant-...
  python demo_graph_live.py

  # Unix/macOS
  export ANTHROPIC_API_KEY=sk-ant-...
  python demo_graph_live.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from graph_rag import GraphRAG
from causal_extractor import extract_edges
from pipeline import MockLLM


# ---------------------------------------------------------------------------
# LLM auto-detection
# ---------------------------------------------------------------------------

def _pick_llm():
    """Return (llm_instance, label) for whichever key is present, else MockLLM."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from llm_adapters import AnthropicLLM
            llm = AnthropicLLM()
            return llm, f"AnthropicLLM ({llm.model})"
        except ImportError:
            print("[warn] ANTHROPIC_API_KEY set but 'anthropic' package missing.")
            print("       Run: pip install anthropic")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from llm_adapters import OpenAILLM
            llm = OpenAILLM()
            return llm, f"OpenAILLM ({llm.model})"
        except ImportError:
            print("[warn] OPENAI_API_KEY set but 'openai' package missing.")
            print("       Run: pip install openai")

    if os.environ.get("GROQ_API_KEY"):
        try:
            from llm_adapters import GroqLLM
            llm = GroqLLM()
            return llm, f"GroqLLM ({llm.model})"
        except ImportError:
            print("[warn] GROQ_API_KEY set but 'groq' package missing.")
            print("       Run: pip install groq")

    return None, None


# ---------------------------------------------------------------------------
# Demo document and queries (identical to demo_graph.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    llm, label = _pick_llm()

    if llm is None:
        print("=" * 64)
        print("No API key found — falling back to MockLLM (offline).")
        print()
        print("To use a real LLM, set one of these environment variables:")
        print("  set ANTHROPIC_API_KEY=sk-ant-...   # recommended (claude-opus-4-8)")
        print("  set OPENAI_API_KEY=sk-...           # gpt-4o")
        print("  set GROQ_API_KEY=gsk_...            # llama-3.1-8b-instant (free tier)")
        print("=" * 64)
        print()
        llm = MockLLM()
        label = "MockLLM (offline stand-in)"

    print(f"LLM: {label}")
    print()

    rag = GraphRAG(dim=10000, semantic_weight=0, llm=llm)
    n = rag.ingest(DOC)

    print("EXTRACTED CAUSAL GRAPH")
    print("=" * 64)
    for e in extract_edges(DOC):
        print(f"  {e.text()}")
    print(f"\n{n} directed edges. Nodes: {sorted(rag.graph.nodes())}\n")

    print("QUERIES — LLM answers grounded in retrieved causal chains")
    print("=" * 64)
    for q, label_q in QUERIES:
        ans, chains = rag.answer(q, top_k=2)
        print(f"\nQ ({label_q}): {q}")
        if chains:
            c = chains[0]
            print(f"   [{c.direction}] {c.text()}")
            print(f"   spans {len(c.provenance())} source sentence(s)")
        print(f"\nA: {ans}")
        print("-" * 64)


if __name__ == "__main__":
    main()
