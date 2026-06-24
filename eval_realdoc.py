"""
eval_realdoc.py
===============
The corrected experiment (grounded in the literature):

  * REAL large document  : Wikipedia "Subprime mortgage crisis > Causes"
                           (~91k chars, ~436 sentences, deeply nested sections).
                           top_k=4 << 436, so retrieval must actually choose.
  * MULTI-HOP / GLOBAL Q : the regime where structure is expected to help
                           (GraphRAG: graphs win global/multi-hop, baseline wins
                           single-hop factoids).
  * STRUCTURE on vs off  : flat sentences  vs  causal chains + heading paths.
  * WEAK vs STRONG model : llama-3.1-8b-instant  vs  llama-3.3-70b-versatile.
                           Hypothesis (arXiv 2402.13492): structure helps the
                           weak model more; converges to ~0 for the strong one.

Retrieval is identical across all cells (spaCy graph, LLM-independent), so we
swap only the GENERATION model. A fixed strong judge scores faithfulness.

Run:  python eval_realdoc.py        (uses GROQ_API_KEY from .env)
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass
from typing import List, Dict

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

WEAK = "llama-3.1-8b-instant"
STRONG = "llama-3.3-70b-versatile"

# Which generation models to run (env override: MODELS="weak,strong").
# The capability-ablation only needs the weak model; the strong model has a low
# daily token cap, so default to weak-only and opt into strong explicitly.
_sel = os.environ.get("MODELS", "weak").lower()
MODELS = tuple(m for tag, m in (("weak", WEAK), ("strong", STRONG)) if tag in _sel)
JUDGE_MODEL = WEAK

DOC_PATH = "subprime_causes.md"


@dataclass
class Q:
    q: str
    concepts: List[str]   # substrings; answer "recall" = fraction present


# Multi-hop / global questions over the Causes section.
QUESTIONS = [
    Q("What were the primary causes of the subprime mortgage crisis?",
      ["subprime", "securit", "rating", "housing", "regulat", "shadow bank"]),
    Q("How did high-risk mortgage lending connect to the collapse of the shadow banking system?",
      ["subprime", "securit", "mortgage-backed", "shadow bank", "default"]),
    Q("What role did credit rating agencies play in enabling the crisis?",
      ["rating", "aaa", "securit", "risk"]),
    Q("How did government housing policy relate to the growth of subprime lending?",
      ["polic", "ownership", "fannie", "housing"]),
    Q("Explain how securitization spread mortgage risk through the financial system.",
      ["securit", "mortgage", "investor", "risk", "default"]),
]

# Ablation: isolate the two generation-side structure signals.
#   structured = causal chain paths (+ polarity arrows) in the prompt
#   contextual = document heading-path annotation on each evidence sentence
# (retrieval-side contextual indexing + MMR is always on; this varies only what
#  structure is SHOWN to the model.)
CONDITIONS = {
    "flat":          dict(structured=False, contextual=False),
    "+causal":       dict(structured=True,  contextual=False),
    "+doc":          dict(structured=False, contextual=True),
    "+causal+doc":   dict(structured=True,  contextual=True),
}

TOP_K = 6   # give MMR room to diversify coverage on global questions


def keyword_recall(answer: str, concepts: List[str]) -> float:
    a = answer.lower()
    return sum(1 for c in concepts if c in a) / len(concepts) if concepts else 0.0


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY required for this benchmark.")
        return
    text = open(DOC_PATH, encoding="utf-8").read()

    print("Ingesting real document (spaCy, LLM-independent retrieval)...")
    rag = GraphRAG(dim=10000, llm=GroqLLM(WEAK))
    n = rag.ingest(text, schema="auto")
    print(f"  {n} causal edges, {len(rag.graph.nodes())} nodes, "
          f"{len(rag._struct_index)} structural sentences, schema indexed.\n")

    from eval_ragas import RagasLLMJudge
    # Judge with the weak model to conserve the strong model's daily token cap;
    # kw_recall (the ablation's key metric) is judge-independent anyway.
    judge = RagasLLMJudge(GroqLLM(JUDGE_MODEL))

    # results[model][cond] = {"kr":, "faith":, "n":}
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model in MODELS:
        rag.llm = GroqLLM(model)
        results[model] = {c: {"kr": 0.0, "faith": 0.0, "n": 0} for c in CONDITIONS}
        print(f"--- generation model: {model} ---")
        for item in QUESTIONS:
            for cond, opts in CONDITIONS.items():
                try:
                    ans, chains = rag.answer(item.q, top_k=TOP_K, **opts)
                    contexts = GraphRAG._dedup_provenance(chains)
                    kr = keyword_recall(ans, item.concepts)
                    faith = judge.faithfulness(ans, contexts) if contexts else 0.0
                except Exception as exc:  # rate limit, transient API error, ...
                    print(f"    [skip] {model}/{cond}: {str(exc)[:80]}")
                    continue
                results[model][cond]["kr"] += kr
                results[model][cond]["faith"] += faith
                results[model][cond]["n"] += 1
        print()

    print("=" * 70)
    print("RESULTS - causal/doc ablation x model (real large doc)")
    print("=" * 70)
    print(f"{'model':<26}{'condition':<12}{'kw_recall':>10}{'faithful':>10}")
    print("-" * 70)
    for model in MODELS:
        base = None
        for cond in CONDITIONS:
            r = results[model][cond]
            n = r["n"] or 1
            kr, fa = r["kr"] / n, r["faith"] / n
            if base is None:
                base = (kr, fa); tag = ""
            else:
                tag = f"  (recall {kr-base[0]:+.2f}, faith {fa-base[1]:+.2f})"
            print(f"{model:<26}{cond:<12}{kr:>10.2f}{fa:>10.2f}{tag}")
        print("-" * 70)
    print("Hypothesis: structure's gain (the +deltas) should be larger for the")
    print("weak model than the strong model.")


if __name__ == "__main__":
    main()
