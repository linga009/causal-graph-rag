"""
test_rebel.py
=============
Quick test of REBEL relation extraction vs LLM extraction.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from causal_graph_rag.causal_extractor import extract_edges, REBELRelationExtractor
from causal_graph_rag.llm_adapters import GroqLLM, AnthropicLLM
from causal_graph_rag.pipeline import MockLLM

DEMO_TEXT = """
The temperature sensor failed, which led to reactor overheating.
The reactor overheated, which caused the coolant valve to jam.
The jammed valve triggered the emergency shutdown.
The emergency shutdown caused a power outage.
The power outage disrupted hospital operations.
Budget cuts reduced the inspection frequency.
Reduced inspection frequency increased the risk of equipment failure.
"""

def test_extraction_methods():
    print("=" * 70)
    print("CAUSAL EXTRACTION COMPARISON")
    print("=" * 70)

    # 1. Base extraction (spaCy + rules)
    print("\n[1] Base extraction (spaCy + rules):")
    base_edges = extract_edges(DEMO_TEXT)
    print(f"    Edges: {len(base_edges)}")
    for e in base_edges:
        print(f"      {e.text()}")

    # 2. REBEL extraction
    print("\n[2] REBEL extraction:")
    try:
        rebel = REBELRelationExtractor(device="cpu")
        rebel_edges = rebel.extract(DEMO_TEXT)
        print(f"    Edges: {len(rebel_edges)}")
        for e in rebel_edges:
            print(f"      {e.text()}")
    except ImportError as e:
        print(f"    SKIPPED: {e}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # 3. LLM extraction (if available)
    print("\n[3] LLM extraction (if available):")
    try:
        if os.environ.get("GROQ_API_KEY"):
            llm = GroqLLM()
        elif os.environ.get("ANTHROPIC_API_KEY"):
            llm = AnthropicLLM()
        else:
            llm = MockLLM()

        from causal_graph_rag.causal_extractor import LLMEdgeExtractor
        llm_extractor = LLMEdgeExtractor(llm, mode="full")
        llm_edges = llm_extractor.extract(DEMO_TEXT)
        print(f"    Edges: {len(llm_edges)}")
        for e in llm_edges:
            print(f"      {e.text()}")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_extraction_methods()
