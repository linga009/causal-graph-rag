"""
demo_structured.py — show the Phase 1 win.

Compares structured causal context (chain paths + polarity arrows) against the
legacy flat-sentence context on a polarity-sensitive question, using a real LLM.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_env(path=".env"):
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_env()

from graph_rag import GraphRAG
from llm_adapters import GroqLLM
from pipeline import MockLLM

DOC = """# Incident Report: Plant Outage

## Timeline
The reactor overheated during the night shift. The overheating caused the
coolant valve to fail. The valve failure triggered an emergency shutdown.

## Impact
The shutdown reduced power output. Lower power output disrupted hospital
operations across the district.
"""

QUESTION = "Did the emergency shutdown raise or lower power output, and what did that ultimately affect?"


def main():
    llm = GroqLLM() if os.environ.get("GROQ_API_KEY") else MockLLM()
    print(f"LLM: {type(llm).__name__}\n")

    rag = GraphRAG(dim=10000, llm=llm)
    rag.ingest(DOC)

    chains = rag.retrieve(QUESTION, top_k=3)
    print("Retrieved causal chains (note the polarity arrows):")
    for c in chains:
        print(f"  {c.text()}")
    print()
    print("Context handed to the LLM (note heading-path tags):")
    print(rag._build_context(chains, structured=True))
    print()

    ans_flat, _ = rag.answer(QUESTION, structured=False)
    ans_struct, _ = rag.answer(QUESTION, structured=True)

    print(f"Q: {QUESTION}\n")
    print(f"[flat sentences]   {ans_flat}\n")
    print(f"[structured chains] {ans_struct}")


if __name__ == "__main__":
    main()
