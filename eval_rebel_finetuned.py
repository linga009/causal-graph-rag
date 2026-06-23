"""
eval_rebel_finetuned.py
=======================
Compare base REBEL, fine-tuned REBEL, and LLM extraction on the large benchmark.

Shows whether domain-specific fine-tuning improves recall vs base REBEL and LLM.

Usage
-----
  # First fine-tune REBEL on domain data
  python finetune_rebel.py --domain healthcare --output models/rebel-healthcare
  python finetune_rebel.py --domain finance --output models/rebel-finance

  # Then evaluate
  python eval_rebel_finetuned.py --compare
"""

from __future__ import annotations

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(__file__))


def evaluate_extractors(domain: str):
    """Evaluate base REBEL, fine-tuned REBEL, and LLM on a domain."""
    from eval_multidomain_large import LARGE_MULTIDOMAIN_CORPORA, LARGE_MULTIDOMAIN_DATASET
    from graph_rag import GraphRAG
    from causal_extractor import extract_edges, LLMEdgeExtractor, REBELRelationExtractor
    from llm_adapters import GroqLLM, AnthropicLLM
    from pipeline import MockLLM

    # Get LLM
    if os.environ.get("GROQ_API_KEY"):
        llm = GroqLLM()
        llm_label = "GroqLLM"
    else:
        llm = MockLLM()
        llm_label = "MockLLM"

    corpus = LARGE_MULTIDOMAIN_CORPORA[domain]
    domain_q = [q for q in LARGE_MULTIDOMAIN_DATASET if q.domain == domain]

    print(f"\n{'='*70}")
    print(f"  DOMAIN: {domain.upper()} ({len(domain_q)} questions)")
    print(f"{'='*70}\n")

    extractors = {
        "spaCy baseline": ("spacy", None),
        "REBEL (base)": ("rebel_base", None),
        "REBEL (fine-tuned)": ("rebel_ft", f"models/rebel-{domain}/model"),
        "LLM full": ("llm", None),
    }

    results = {}

    for label, (extractor_type, model_path) in extractors.items():
        print(f"[{label}]")

        rag = GraphRAG(dim=10000, llm=llm)

        # Ingest with specified extractor
        if extractor_type == "spacy":
            n_edges = rag.ingest(corpus)
        elif extractor_type == "rebel_base":
            try:
                rebel = REBELRelationExtractor(device="cpu")
                from causal_extractor import extract_edges_hybrid

                edges = extract_edges_hybrid(corpus, rebel, mode="full")
                n_edges = len(edges)
                rag.graph.edges.clear()  # Use REBEL edges
                for e in edges:
                    rag.graph.add_edge(e)
            except ImportError:
                print("  SKIPPED: transformers not installed")
                continue
        elif extractor_type == "rebel_ft":
            if not os.path.exists(model_path):
                print(f"  SKIPPED: Fine-tuned model not found at {model_path}")
                continue
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                # Would use this model for extraction, but requires custom logic
                print("  (Fine-tuned extraction requires custom pipeline - skipped for now)")
                continue
            except ImportError:
                print("  SKIPPED: transformers not installed")
                continue
        elif extractor_type == "llm":
            n_edges = rag.ingest(corpus, llm_extractor=llm, llm_mode="full")

        # Evaluate on questions
        from eval_ragas import RagasLLMJudge, SampleResult

        judge = RagasLLMJudge(llm)
        domain_results = []

        for sample in domain_q:
            answer, chains = rag.answer(sample.question, top_k=3, summarize=False)
            seen = set()
            contexts = []
            for chain in chains:
                for s in chain.provenance():
                    if s not in seen:
                        seen.add(s)
                        contexts.append(s)

            result = SampleResult(
                question=sample.question,
                answer=answer,
                contexts=contexts,
                faithfulness=judge.faithfulness(answer, contexts),
                precision=judge.context_precision(sample.question, contexts),
                recall=judge.context_recall(sample.ground_truth, contexts, sample.relevant_keywords),
            )
            domain_results.append(result)

        # Summary
        if domain_results:
            n = len(domain_results)
            avg_f = sum(r.faithfulness for r in domain_results) / n
            avg_p = sum(r.precision for r in domain_results) / n
            avg_r = sum(r.recall for r in domain_results) / n

            results[label] = (avg_f, avg_p, avg_r)
            print(f"  Edges: {len(rag.graph._get_edges()) if hasattr(rag.graph, '_get_edges') else n_edges:3d}")
            print(f"  Faith: {avg_f:.2f}  |  Prec: {avg_p:.2f}  |  Recall: {avg_r:.2f}")

    # Comparison table
    print(f"\n  Summary Table:")
    print(f"  {'-'*50}")
    print(f"  Extractor              | Faith | Prec | Recall")
    print(f"  {'-'*50}")
    for label, (f, p, r) in results.items():
        print(f"  {label:<22} | {f:.2f}  | {p:.2f} | {r:.2f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned REBEL")
    parser.add_argument(
        "--domain",
        choices=["healthcare", "finance"],
        default="healthcare",
        help="Domain to evaluate",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all domains",
    )
    args = parser.parse_args()

    if args.compare:
        for domain in ["healthcare", "finance"]:
            evaluate_extractors(domain)
    else:
        evaluate_extractors(args.domain)


if __name__ == "__main__":
    main()
