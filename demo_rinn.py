"""
demo_rinn.py
============
Run GraphRAG on the Rinn AI PhD document using Groq as the LLM backend.
Loads API key from .env automatically.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))


# --------------------------------------------------------------------------- #
#  Load .env
# --------------------------------------------------------------------------- #
def _load_env(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val:
                    os.environ.setdefault(key, val)
    except FileNotFoundError:
        pass


_load_env(os.path.join(os.path.dirname(__file__), ".env"))


# --------------------------------------------------------------------------- #
#  Extract + clean PDF text
# --------------------------------------------------------------------------- #
def _load_pdf(path: str) -> str:
    import pypdf
    reader = pypdf.PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    raw = "\n".join(pages)

    # The PDF uses per-word spacing, producing blank lines between single words.
    # Step 1: collapse ALL whitespace (including newlines) to a single space.
    flat = re.sub(r"\s+", " ", raw).strip()

    # Step 2: remove bullet characters and stray symbols.
    flat = re.sub(r"[●•◆■]", " ", flat)
    flat = re.sub(r"\s+", " ", flat).strip()

    # Step 3: split into sentences on ". " followed by a capital letter,
    # "? " or "! " — keeping the terminator on the left sentence.
    # Also split on " Supervisor:" and "Research Theme:" which start new items.
    flat = re.sub(r"\.\s+(?=[A-Z])", ".\n", flat)
    flat = re.sub(r"\?\s+(?=[A-Z])", "?\n", flat)
    flat = re.sub(r"(?<=[a-z])\s+(Supervisor:|Research Theme:|PhD Topic:)", r"\n\1", flat)

    sentences = []
    for sent in flat.splitlines():
        sent = sent.strip()
        # Drop very short fragments, email addresses, URLs, headers
        if len(sent) < 25:
            continue
        if re.search(r"@|https?://|forms\.gle", sent):
            continue
        sentences.append(sent)

    return "\n".join(sentences)


# --------------------------------------------------------------------------- #
#  Build LLM
# --------------------------------------------------------------------------- #
def _build_llm():
    if os.environ.get("GROQ_API_KEY"):
        try:
            from llm_adapters import GroqLLM
            llm = GroqLLM()
            return llm, f"GroqLLM ({llm.model})"
        except ImportError:
            print("[warn] groq package missing — run: pip install groq")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from llm_adapters import AnthropicLLM
            llm = AnthropicLLM()
            return llm, f"AnthropicLLM ({llm.model})"
        except ImportError:
            pass

    from pipeline import MockLLM
    return MockLLM(), "MockLLM (no API key found)"


# --------------------------------------------------------------------------- #
#  Queries
# --------------------------------------------------------------------------- #
QUERIES = [
    ("What does the Rinn network enable?",                          "forward"),
    ("Why was the Rinn AI programme created?",                      "backward root-cause"),
    ("What will the enrichment experience lead to for students?",   "forward"),
    ("What does AI-driven digital expansion cause?",                "forward"),
    ("How does fetal heart rate analysis help?",                    "forward"),
    ("What causes differences in clinical decision quality?",       "backward"),
]


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    pdf_path = os.path.join(os.path.dirname(__file__), "Rinn ai phd.pdf")

    print("Loading PDF...")
    doc = _load_pdf(pdf_path)

    # Show a preview of what was extracted
    sentences = [s.strip() for s in doc.split("\n") if s.strip()]
    print(f"Extracted {len(sentences)} sentences from PDF.\n")
    print("DOCUMENT PREVIEW (first 8 sentences)")
    print("=" * 64)
    for s in sentences[:8]:
        print(f"  {s}")
    print()

    llm, label = _build_llm()
    print(f"LLM: {label}\n")

    from graph_rag import GraphRAG
    from causal_extractor import extract_edges

    rag = GraphRAG(dim=10000, semantic_weight=0, llm=llm)
    n = rag.ingest(doc)

    print("EXTRACTED CAUSAL GRAPH")
    print("=" * 64)
    edges = extract_edges(doc)
    for e in edges:
        print(f"  {e.text()}")
    print(f"\n{n} directed edges. {len(rag.graph.nodes())} nodes.\n")

    print("QUERIES")
    print("=" * 64)
    for q, label_q in QUERIES:
        ans, chains = rag.answer(q, top_k=2)
        print(f"\nQ ({label_q}): {q}")
        if chains:
            c = chains[0]
            print(f"  Chain [{c.direction}]: {c.text()}")
        print(f"  A: {ans}")
        print("-" * 64)


if __name__ == "__main__":
    main()
