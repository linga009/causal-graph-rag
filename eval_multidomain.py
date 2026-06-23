"""
eval_multidomain.py
===================
Evaluate Causal Graph RAG on multi-domain benchmark (healthcare, finance, manufacturing).
Extends eval_ragas.py with real-world incident narratives beyond the demo corpus.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.dirname(__file__))


# --------------------------------------------------------------------------- #
#  Multi-domain evaluation dataset
# --------------------------------------------------------------------------- #

@dataclass
class EvalSample:
    question: str
    ground_truth: str
    relevant_keywords: List[str] = field(default_factory=list)
    domain: str = "general"  # healthcare, finance, manufacturing, etc.


# DOMAIN 1: Healthcare / Clinical Incidents
HEALTHCARE_CORPUS = """
Patient arrived with severe chest pain and shortness of breath at 14:30.
Initial ECG showed ST elevation, suggesting acute myocardial infarction.
The cardiac care unit was at 95% capacity due to a surge in admissions.
Lack of available ICU beds delayed admission from the ER to monitored care.
The delay in admission worsened the patient's condition and increased cardiac necrosis.
Delayed interventional cardiology treatment resulted in larger infarct size.
Larger infarct size reduced cardiac output and ejection fraction.
Reduced ejection fraction led to cardiogenic shock.
Cardiogenic shock triggered acute kidney injury.
Acute kidney injury complicated post-infarction recovery and extended hospital stay.
"""

HEALTHCARE_QUESTIONS: List[EvalSample] = [
    EvalSample(
        question="Why did the patient develop cardiogenic shock?",
        ground_truth=(
            "The patient developed cardiogenic shock due to a cascade: "
            "ST elevation AMI → delayed ICU admission → delayed PCI → "
            "larger infarct size → reduced ejection fraction → cardiogenic shock."
        ),
        relevant_keywords=["cardiogenic shock", "ejection fraction", "infarct", "ICU", "delayed"],
        domain="healthcare"
    ),
    EvalSample(
        question="What was the root cause of the extended hospital stay?",
        ground_truth=(
            "The extended hospital stay resulted from acute kidney injury caused by "
            "cardiogenic shock, which itself stemmed from a large MI due to delayed "
            "intervention from ICU bed shortage."
        ),
        relevant_keywords=["kidney injury", "hospital stay", "cardiogenic shock", "ICU beds"],
        domain="healthcare"
    ),
]

# DOMAIN 2: Finance / Risk & Contagion
FINANCE_CORPUS = """
A major hedge fund experienced significant losses in the tech sector on March 15.
Losses triggered margin calls from their prime brokers.
Margin calls forced the fund to rapidly liquidate positions.
Rapid liquidation of large positions depressed tech stock prices.
Depressed tech prices triggered stop-loss orders across the market.
Stop-loss orders created a cascade of forced selling.
Forced selling accelerated the decline and triggered circuit breakers.
Circuit breaker halt paused trading for 15 minutes.
After resumption, retail investors panicked and sold.
Panic selling created a liquidity crisis.
Liquidity crisis increased bid-ask spreads and slowed transactions.
Slower transactions prevented the fund from covering margin calls on time.
Failed margin calls resulted in forced liquidation by the broker.
Forced liquidation wiped out the fund's remaining capital.
"""

FINANCE_QUESTIONS: List[EvalSample] = [
    EvalSample(
        question="How did the tech sector losses cascade into a fund collapse?",
        ground_truth=(
            "Initial losses triggered margin calls, which forced liquidation of positions. "
            "Large liquidations depressed prices, triggering stop-loss cascades and panic selling. "
            "This created a liquidity crisis that prevented timely margin call coverage, "
            "resulting in forced broker liquidation and total capital loss."
        ),
        relevant_keywords=["margin calls", "liquidation", "panic selling", "liquidity", "cascade"],
        domain="finance"
    ),
]

# DOMAIN 3: Manufacturing / Root Cause Analysis
MANUFACTURING_CORPUS = """
A servo motor in the main production line began overheating on shift 2.
Overheating was caused by inadequate lubrication due to a leaky bearing seal.
The bearing seal leak had been documented in the maintenance log but not addressed.
Lack of seal replacement was due to budget constraints in Q1.
Budget constraints delayed the procurement of replacement seals.
The servo motor overheating caused loss of precision in the CNC machine.
Loss of precision resulted in rejected parts with tolerance violations.
Rejected parts accumulated because quality control caught the defects late.
Part rejection triggered a production delay of 4 hours.
The 4-hour delay caused a shortage of components for downstream assembly.
Component shortage delayed final product assembly by 24 hours.
Delayed assembly resulted in missed delivery to a major customer.
Missed delivery triggered a $50k penalty clause in the supply contract.
"""

MANUFACTURING_QUESTIONS: List[EvalSample] = [
    EvalSample(
        question="What caused the customer delivery miss and penalty?",
        ground_truth=(
            "The $50k penalty resulted from a cascade starting with the bearing seal failure. "
            "Deferred maintenance due to budget constraints led to servo overheating, "
            "which caused CNC precision loss, rejected parts, production delays, "
            "component shortages, and ultimately a 24-hour assembly delay that "
            "triggered the supply contract penalty."
        ),
        relevant_keywords=["bearing seal", "budget", "servo", "precision", "delay", "penalty"],
        domain="manufacturing"
    ),
    EvalSample(
        question="What was the root cause of the production delay?",
        ground_truth=(
            "The root cause was the deferred bearing seal replacement due to Q1 budget constraints. "
            "This deferred maintenance led directly to servo overheating and precision loss."
        ),
        relevant_keywords=["bearing seal", "budget", "maintenance", "servo"],
        domain="manufacturing"
    ),
]

# Combine all datasets
MULTIDOMAIN_DATASET: List[EvalSample] = (
    HEALTHCARE_QUESTIONS + FINANCE_QUESTIONS + MANUFACTURING_QUESTIONS
)

MULTIDOMAIN_CORPORA = {
    "healthcare": HEALTHCARE_CORPUS,
    "finance": FINANCE_CORPUS,
    "manufacturing": MANUFACTURING_CORPUS,
}


# --------------------------------------------------------------------------- #
#  Evaluation runner
# --------------------------------------------------------------------------- #

def evaluate_multidomain(llm_extract: str | None = None, llm_mode: str = ""):
    """
    Run evaluation on all domains.
    """
    from graph_rag import GraphRAG
    from eval_ragas import RagasLLMJudge, SampleResult
    from llm_adapters import GroqLLM, AnthropicLLM
    from pipeline import MockLLM

    # Pick LLM
    if os.environ.get("GROQ_API_KEY"):
        llm = GroqLLM()
        llm_label = "GroqLLM"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        llm = AnthropicLLM()
        llm_label = "AnthropicLLM"
    else:
        llm = MockLLM()
        llm_label = "MockLLM"

    print(f"LLM: {llm_label}\n")
    judge = RagasLLMJudge(llm)

    # Run per-domain evaluations
    results_by_domain = {}

    for domain, corpus in MULTIDOMAIN_CORPORA.items():
        print(f"\n{'='*70}")
        print(f"  DOMAIN: {domain.upper()}")
        print(f"{'='*70}\n")

        # Build RAG for this domain
        rag = GraphRAG(dim=10000, llm=llm)
        if llm_extract:
            n_edges = rag.ingest(corpus, llm_extractor=llm, llm_mode=llm_mode)
        else:
            n_edges = rag.ingest(corpus)

        print(f"  Edges extracted: {n_edges}  |  Nodes: {len(rag.graph.nodes())}\n")

        # Evaluate questions for this domain
        domain_questions = [q for q in MULTIDOMAIN_DATASET if q.domain == domain]
        domain_results = []

        for sample in domain_questions:
            answer, chains = rag.answer(sample.question, top_k=3, summarize=False)

            # Collect contexts (provenance only, not chain text)
            seen_sents = set()
            contexts = []
            for chain in chains:
                for s in chain.provenance():
                    if s not in seen_sents:
                        seen_sents.add(s)
                        contexts.append(s)

            faith = judge.faithfulness(answer, contexts)
            prec = judge.context_precision(sample.question, contexts)
            rec = judge.context_recall(sample.ground_truth, contexts, sample.relevant_keywords)

            domain_results.append(SampleResult(
                question=sample.question,
                answer=answer,
                contexts=contexts,
                faithfulness=faith,
                precision=prec,
                recall=rec,
            ))

            print(f"  Q: {sample.question[:60]}...")
            print(f"     faith={faith:.2f}  prec={prec:.2f}  rec={rec:.2f}\n")

        results_by_domain[domain] = domain_results

    # Print summary table
    print("\n" + "="*70)
    print("  MULTI-DOMAIN EVALUATION SUMMARY")
    print("="*70)
    print(f"  {'Domain':<15}  {'Questions':<10}  Avg Faith  Avg Prec  Avg Recall")
    print(f"  {'-'*15}  {'-'*10}  ---------  --------  ----------")

    for domain, results in results_by_domain.items():
        n = len(results)
        avg_f = sum(r.faithfulness for r in results) / n if n > 0 else 0.0
        avg_p = sum(r.precision for r in results) / n if n > 0 else 0.0
        avg_r = sum(r.recall for r in results) / n if n > 0 else 0.0
        print(f"  {domain:<15}  {n:<10}  {avg_f:.2f}       {avg_p:.2f}      {avg_r:.2f}")

    print(f"  {'-'*15}  {'-'*10}  ---------  --------  ----------")
    all_results = [r for results in results_by_domain.values() for r in results]
    if all_results:
        overall_f = sum(r.faithfulness for r in all_results) / len(all_results)
        overall_p = sum(r.precision for r in all_results) / len(all_results)
        overall_r = sum(r.recall for r in all_results) / len(all_results)
        print(f"  {'OVERALL':<15}  {len(all_results):<10}  {overall_f:.2f}       {overall_p:.2f}      {overall_r:.2f}")

    print("="*70)


def _avg_metrics(results: List[SampleResult]) -> tuple[float, float, float]:
    """Average faith, prec, recall across results."""
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0
    return (
        sum(r.faithfulness for r in results) / n,
        sum(r.precision for r in results) / n,
        sum(r.recall for r in results) / n,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-domain causal RAG evaluation")
    parser.add_argument("--llm-extract", choices=["augment", "full"], default=None,
                        help="Enable LLM-assisted graph building")
    parser.add_argument("--compare-extraction", action="store_true",
                        help="Compare all three extraction modes side-by-side")
    args = parser.parse_args()

    if args.compare_extraction:
        # Run all three extraction modes and compare
        from llm_adapters import GroqLLM, AnthropicLLM
        from pipeline import MockLLM
        from graph_rag import GraphRAG
        from eval_ragas import RagasLLMJudge

        if os.environ.get("GROQ_API_KEY"):
            llm = GroqLLM()
            llm_label = "GroqLLM"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            llm = AnthropicLLM()
            llm_label = "AnthropicLLM"
        else:
            llm = MockLLM()
            llm_label = "MockLLM"

        judge = RagasLLMJudge(llm)
        runs = [
            ("spaCy baseline", None, ""),
            ("LLM augment", "augment", "augment"),
            ("LLM full", "full", "full"),
        ]

        print(f"LLM: {llm_label}\n")
        all_results_by_mode = {}

        for label, llm_ext, llm_mode in runs:
            print(f"\n[{label}]\n")
            results_by_domain = {}

            for domain, corpus in MULTIDOMAIN_CORPORA.items():
                rag = GraphRAG(dim=10000, llm=llm)
                if llm_ext:
                    n_edges = rag.ingest(corpus, llm_extractor=llm, llm_mode=llm_mode)
                else:
                    n_edges = rag.ingest(corpus)

                domain_results = []
                for sample in [q for q in MULTIDOMAIN_DATASET if q.domain == domain]:
                    answer, chains = rag.answer(sample.question, top_k=3, summarize=False)
                    seen_sents = set()
                    contexts = []
                    for chain in chains:
                        for s in chain.provenance():
                            if s not in seen_sents:
                                seen_sents.add(s)
                                contexts.append(s)

                    from eval_ragas import SampleResult
                    domain_results.append(SampleResult(
                        question=sample.question,
                        answer=answer,
                        contexts=contexts,
                        faithfulness=judge.faithfulness(answer, contexts),
                        precision=judge.context_precision(sample.question, contexts),
                        recall=judge.context_recall(sample.ground_truth, contexts, sample.relevant_keywords),
                    ))

                results_by_domain[domain] = domain_results

            all_results = [r for results in results_by_domain.values() for r in results]
            all_results_by_mode[label] = all_results

        # Summary table
        print("\n" + "="*80)
        print("  EXTRACTION MODE COMPARISON (Multi-Domain)")
        print("="*80)
        print(f"  {'Mode':<20}  Avg Edges  Avg Faith  Avg Prec  Avg Recall")
        print(f"  {'-'*20}  ---------  ---------  --------  ----------")

        for label in [l for l, _, _ in runs]:
            results = all_results_by_mode[label]
            if results:
                f, p, r = _avg_metrics(results)
                # Estimate avg edges (rough)
                print(f"  {label:<20}  {'~':<9}  {f:.2f}       {p:.2f}      {r:.2f}")

        print(f"  {'-'*20}  ---------  ---------  --------  ----------")
        print("="*80)

    else:
        evaluate_multidomain(args.llm_extract, args.llm_extract or "")
