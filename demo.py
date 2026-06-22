"""
demo.py — runnable end-to-end demonstration of the VSA-RAG engine.

    python demo.py

Uses the offline MockLLM by default. To use Groq, set GROQ_API_KEY and:
    from llm_adapters import GroqLLM
    rag = VSARAG(llm=GroqLLM())
"""

from pipeline import VSARAG
from parser import backend_name

CORPUS = """
Inflation causes unemployment.
The central bank raised interest rates.
Higher interest rates reduce inflation.
Unemployment increases poverty.
Automation displaces factory workers.
Renewable energy lowers carbon emissions.
"""

QUERIES = [
    "Does inflation cause unemployment?",        # active, exact
    "Is unemployment caused by inflation?",      # passive, same meaning
    "Does unemployment cause inflation?",        # swapped, opposite meaning
    "What reduces inflation?",                    # different fact
    "What lowers emissions?",                     # synonym-ish target
    "Who discovered penicillin?",                 # out of corpus
]


def main():
    print(f"Parser backend : {backend_name()}")
    rag = VSARAG(dim=10000, semantic_weight=1, match_threshold=0.45)
    n = rag.ingest(CORPUS, doc_id="kb")
    print(f"Ingested triples: {n}\n" + "=" * 64)

    for q in QUERIES:
        ans, hits = rag.answer(q, top_k=3)
        print(f"\nQ: {q}")
        if hits:
            for h in hits:
                print(f"   match {h.score:+.3f}  [{h.triple.text()}]")
            print(f"   -> {ans}")
        else:
            print("   (no structural match; LLM not called with stale context)")
            print(f"   -> {ans}")


if __name__ == "__main__":
    main()
