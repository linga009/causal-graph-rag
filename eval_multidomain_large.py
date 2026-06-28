"""
eval_multidomain_large.py
==========================
Larger multi-domain benchmark with 10+ questions per domain.
Tests the system on more diverse causal reasoning patterns.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.dirname(__file__))


def _load_env(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass

_load_env(os.path.join(os.path.dirname(__file__), ".env"))


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    relevant_keywords: List[str] = field(default_factory=list)
    domain: str = "general"


# ============================================================================
# DOMAIN 1: HEALTHCARE (10 questions, 3 complex clinical scenarios)
# ============================================================================

HEALTHCARE_CORPUS_1 = """
Patient presented with fever (39.2°C), cough, and dyspnea on hospital day 3 post-op.
Chest X-ray showed infiltrates consistent with aspiration pneumonia.
Aspiration occurred due to inadequate NPO (nothing by mouth) compliance before extubation.
Inadequate NPO was caused by communication gap between night and day surgical teams.
The pneumonia triggered SIRS (systemic inflammatory response syndrome).
SIRS led to sepsis with positive blood cultures for Klebsiella pneumoniae.
Sepsis caused acute respiratory distress syndrome (ARDS).
ARDS required mechanical ventilation and ICU transfer.
ICU transfer delayed by lack of available ventilators (surge capacity exceeded).
Ventilator shortage increased mortality risk and extended ICU stay by 5 days.
Delayed ICU admission resulted in multi-organ dysfunction.
Multi-organ failure necessitated dialysis for acute kidney injury.
"""

HEALTHCARE_CORPUS_2 = """
Elderly patient on warfarin for atrial fibrillation was prescribed clarithromycin for respiratory infection.
Clarithromycin inhibits CYP3A4, which metabolizes warfarin.
CYP3A4 inhibition increased warfarin levels above therapeutic range.
Elevated warfarin caused supratherapeutic INR (5.2).
Supratherapeutic INR triggered spontaneous GI bleeding.
GI bleeding caused hemodynamic instability and shock.
Hemorrhagic shock led to acute coronary syndrome (ACS).
ACS required emergency cardiac catheterization.
Emergency cardiac lab was occupied (no beds), delaying intervention by 90 minutes.
Delayed intervention worsened myocardial infarction and reduced ejection fraction.
"""

HEALTHCARE_CORPUS_3 = """
Hospital acquired C. difficile outbreak in ICU ward (16 cases in 2 weeks).
Outbreak linked to inadequate hand hygiene (compliance 62% vs 90% target).
Low compliance caused by staffing shortage from nursing strike settlement delays.
Settlement delays left 8 RN positions unfilled in the 20-bed ICU.
Understaffing prevented proper contact precautions between patients.
Spread between patients amplified community transmission.
Community transmission led to surge in ED visits (3x normal).
ED surge overwhelmed triage capacity (wait times >6 hours).
Wait times caused patient deterioration before treatment.
Deterioration increased mortality and LOS (length of stay) by 40%.
"""

HEALTHCARE_QUESTIONS = [
    EvalSample(
        question="Why did the post-op patient require ICU admission?",
        ground_truth="Patient developed aspiration pneumonia due to inadequate NPO compliance, which triggered SIRS and sepsis, leading to ARDS and the need for mechanical ventilation.",
        relevant_keywords=["aspiration", "pneumonia", "ARDS", "ventilation", "ICU"],
        domain="healthcare"
    ),
    EvalSample(
        question="What was the root cause of the multi-organ failure?",
        ground_truth="Multi-organ failure resulted from delayed ICU admission (ventilator shortage) during ARDS, which caused prolonged hypoxemia and tissue damage.",
        relevant_keywords=["ventilator", "shortage", "ARDS", "delayed", "multi-organ"],
        domain="healthcare"
    ),
    EvalSample(
        question="How did a communication gap lead to pneumonia?",
        ground_truth="Communication gap between surgical teams led to inadequate NPO compliance, which caused aspiration and subsequent pneumonia.",
        relevant_keywords=["communication", "NPO", "aspiration", "pneumonia"],
        domain="healthcare"
    ),
    EvalSample(
        question="Why did the warfarin patient suffer ACS?",
        ground_truth="Clarithromycin inhibited warfarin metabolism (CYP3A4), causing supratherapeutic INR, GI bleeding, and hemorrhagic shock, which triggered ACS.",
        relevant_keywords=["warfarin", "clarithromycin", "INR", "bleeding", "ACS"],
        domain="healthcare"
    ),
    EvalSample(
        question="What caused the delay in cardiac intervention?",
        ground_truth="Delayed cardiac catheterization was caused by lack of available cath lab beds during the bleeding emergency, extending time to intervention.",
        relevant_keywords=["cardiac", "catheterization", "delay", "beds"],
        domain="healthcare"
    ),
    EvalSample(
        question="How did the nursing strike contribute to C. difficile spread?",
        ground_truth="Nursing strike delays left ICU understaffed, reducing hand hygiene compliance and enabling C. difficile spread between patients.",
        relevant_keywords=["nursing", "staffing", "hygiene", "C. difficile", "spread"],
        domain="healthcare"
    ),
    EvalSample(
        question="Why did ED wait times increase?",
        ground_truth="C. difficile outbreak caused surge in ED visits (3x normal), overwhelming triage capacity and increasing wait times >6 hours.",
        relevant_keywords=["outbreak", "surge", "ED", "wait times"],
        domain="healthcare"
    ),
    EvalSample(
        question="What was the ultimate consequence of inadequate staffing?",
        ground_truth="Understaffing led to low hand hygiene compliance, enabling C. difficile spread, surge in ED visits, patient deterioration, and 40% increase in mortality and LOS.",
        relevant_keywords=["staffing", "hygiene", "outbreak", "mortality", "LOS"],
        domain="healthcare"
    ),
]

# ============================================================================
# DOMAIN 2: FINANCE (8 questions, contagion & market failure scenarios)
# ============================================================================

FINANCE_CORPUS_1 = """
Regional bank A (30B AUM) had concentration risk: 40% portfolio in commercial real estate.
Real estate market declined 15% (regional economic slowdown).
CRE decline triggered mark-to-market losses (8B portfolio hit).
Losses eroded capital adequacy ratio (CAR fell to 9.2%, regulatory minimum 10%).
Low CAR triggered stress test failure and regulatory warning.
Regulatory warning caused depositor panic (runs on bank A).
Depositor outflows forced asset sales at fire-sale prices.
Fire sales depressed regional CRE prices further (downward spiral).
Other regional banks exposed to same CRE market faced losses.
Synchronized losses across regional banks triggered credit tightening (loan rejections up 40%).
Credit tightening starved SMEs of working capital (10k companies affected).
SME cash flow crisis triggered wave of bankruptcies and unemployment spike.
"""

FINANCE_CORPUS_2 = """
Stablecoin USDT had 70% collateral in CRE loans (undisclosed to public).
Rumors of inadequate collateral triggered withdrawal demand (50M USDT/day).
Withdrawal pressure forced collateral liquidation at unfavorable terms.
Liquidation losses exceeded reserves, exposing 500M USDT shortfall.
Shortfall announcement triggered bank run (1B USDT withdrawn in 48 hours).
Bank run drained liquidity reserves and forced emergency borrowing.
Emergency borrowing failed (counterparties lost confidence).
Stablecoin became illiquid (10% discount to par, then 50% collapse).
Stablecoin collapse triggered DeFi platform liquidations (protocols holding USDT as collateral).
Liquidations cascaded through DeFi ecosystem (200B TVL loss).
DeFi losses triggered crypto bear market and forced sales (BTC down 30%).
Retail crypto holders panic-sold (amplifying decline).
"""

FINANCE_QUESTIONS = [
    EvalSample(
        question="How did CRE decline trigger bank failure?",
        ground_truth="CRE decline triggered losses that eroded capital adequacy, triggering regulatory warning, which caused depositor panic and bank runs, forcing asset sales at fire-sale prices.",
        relevant_keywords=["CRE", "losses", "CAR", "depositor panic", "bank run"],
        domain="finance"
    ),
    EvalSample(
        question="What was the ultimate impact on SMEs?",
        ground_truth="Synchronized regional bank losses triggered credit tightening, which starved SMEs of working capital and caused wave of bankruptcies and unemployment.",
        relevant_keywords=["credit tightening", "SME", "capital", "bankruptcy", "unemployment"],
        domain="finance"
    ),
    EvalSample(
        question="How did undisclosed collateral lead to stablecoin collapse?",
        ground_truth="Rumors of inadequate CRE collateral triggered withdrawal demand, forcing liquidation losses, exposing shortfall, and causing bank run that depleted liquidity and collapsed the stablecoin.",
        relevant_keywords=["collateral", "CRE", "withdrawal", "shortfall", "collapse"],
        domain="finance"
    ),
    EvalSample(
        question="What triggered the DeFi cascade?",
        ground_truth="Stablecoin collapse triggered DeFi liquidations because protocols held USDT as collateral, causing 200B TVL loss and crypto bear market.",
        relevant_keywords=["stablecoin", "DeFi", "liquidation", "TVL", "collapse"],
        domain="finance"
    ),
    EvalSample(
        question="How did fire sales amplify CRE losses?",
        ground_truth="Bank A's asset sales at fire-sale prices depressed regional CRE further, triggering losses at other regional banks and coordinated credit tightening.",
        relevant_keywords=["fire sales", "CRE", "prices", "losses", "banks"],
        domain="finance"
    ),
]

# ============================================================================
# DOMAIN 3: MANUFACTURING (8 questions, supply chain & quality failures)
# ============================================================================

MANUFACTURING_CORPUS_1 = """
Supplier A (primary bearing manufacturer) experienced factory fire on June 15.
Fire destroyed 60% of production capacity and inventory (3-month supply).
Supply loss triggered 8-week delivery lead time (normally 2 weeks).
Long lead times forced OEM to source from backup supplier B (20% cost premium).
Backup supplier B used lower-grade steel (cost cutting).
Lower-grade steel increased bearing failure rate (5% vs 0.1% normal).
Bearing failures occurred in customer products (escalating field failures).
Field failures triggered massive warranty claims (5M units affected).
Warranty claims exceeded reserves by 200M (insurance excluded).
Financial impact forced cost restructuring (layoffs, R&D cuts).
R&D cuts delayed next-gen platform (competitor gains market share).
Market share loss reduced future revenue by 30% (5-year impact).
"""

MANUFACTURING_CORPUS_2 = """
Component shortage in semiconductor supply chain (Taiwan chip fab hit by earthquake).
Shortage rippled through automotive supply chain (3-month allocation scheme).
Allocation scheme meant OEM received 40% of normal chip volume.
Reduced chip allocation forced production cutbacks (40% line reduction).
Production cutbacks meant supplier contracts unfulfilled (penalties accrued).
Unfulfilled contracts triggered warranty disputes (suppliers vs OEM).
Disputes caused suppliers to slow deliveries (retaliatory behavior).
Slower supplier deliveries extended OEM assembly cycle time (14 days vs 7 days).
Extended cycle time meant delivery delays to dealers (90-day backlog).
Dealer backlog triggered customer cancellations (20% order cancellation rate).
Cancellations reduced cash flow and froze expansion plans (100M capex deferred).
Deferred capex reduced competitiveness and cost position (future problem).
"""

MANUFACTURING_QUESTIONS = [
    EvalSample(
        question="How did a supplier fire impact end customers?",
        ground_truth="Supplier factory fire reduced bearing supply, forcing reliance on backup supplier with lower-grade steel, causing bearing failures in customer products and massive warranty claims.",
        relevant_keywords=["fire", "supply", "bearing", "failure", "warranty"],
        domain="manufacturing"
    ),
    EvalSample(
        question="What was the financial consequence of the supply disruption?",
        ground_truth="Bearing failures triggered warranty claims exceeding reserves by 200M, forcing cost restructuring (layoffs, R&D cuts) that delayed next-gen platform and lost market share (30% revenue impact).",
        relevant_keywords=["warranty", "reserves", "layoffs", "R&D", "market share"],
        domain="manufacturing"
    ),
    EvalSample(
        question="How did a Taiwan earthquake affect automotive dealers?",
        ground_truth="Earthquake caused chip shortage, which triggered production cutbacks, assembly delays, delivery backlog, and customer cancellations (20% rate).",
        relevant_keywords=["earthquake", "chip", "shortage", "backlog", "cancellation"],
        domain="manufacturing"
    ),
    EvalSample(
        question="What was the long-term impact of supply allocation?",
        ground_truth="Chip allocation and subsequent delivery disputes caused customer cancellations and cash flow reduction, deferring 100M capex and weakening future competitiveness.",
        relevant_keywords=["allocation", "disputes", "cancellation", "cash flow", "capex"],
        domain="manufacturing"
    ),
]

# Combine all
LARGE_MULTIDOMAIN_DATASET = HEALTHCARE_QUESTIONS + FINANCE_QUESTIONS + MANUFACTURING_QUESTIONS

LARGE_MULTIDOMAIN_CORPORA = {
    "healthcare": " ".join([HEALTHCARE_CORPUS_1, HEALTHCARE_CORPUS_2, HEALTHCARE_CORPUS_3]),
    "finance": " ".join([FINANCE_CORPUS_1, FINANCE_CORPUS_2]),
    "manufacturing": " ".join([MANUFACTURING_CORPUS_1, MANUFACTURING_CORPUS_2]),
}


def run_large_eval():
    """Run evaluation on larger corpus."""
    import argparse
    from causal_graph_rag.graph_rag import GraphRAG
    from eval_ragas import RagasLLMJudge, SampleResult
    from causal_graph_rag.llm_adapters import GroqLLM, AnthropicLLM
    from causal_graph_rag.pipeline import MockLLM

    parser = argparse.ArgumentParser(description="Large multi-domain evaluation")
    parser.add_argument("--llm-extract", choices=["augment", "full"], default=None)
    parser.add_argument("--compare-extraction", action="store_true")
    args = parser.parse_args()

    # LLM selection
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

    if args.compare_extraction:
        print("Running comparison: spaCy baseline vs LLM augment vs LLM full\n")
        modes = [
            ("spaCy baseline", None, ""),
            ("LLM augment", "augment", "augment"),
            ("LLM full", "full", "full"),
        ]

        for label, llm_ext, llm_mode in modes:
            print(f"\n[{label}]")
            results_by_domain = {}

            for domain, corpus in LARGE_MULTIDOMAIN_CORPORA.items():
                print(f"  {domain.upper()}...")
                rag = GraphRAG(dim=10000, llm=llm)
                if llm_ext:
                    n_edges = rag.ingest(corpus, llm_extractor=llm, llm_mode=llm_mode)
                else:
                    n_edges = rag.ingest(corpus)

                domain_q = [q for q in LARGE_MULTIDOMAIN_DATASET if q.domain == domain]
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

                    domain_results.append(SampleResult(
                        question=sample.question,
                        answer=answer,
                        contexts=contexts,
                        faithfulness=judge.faithfulness(answer, contexts),
                        precision=judge.context_precision(sample.question, contexts),
                        recall=judge.context_recall(sample.ground_truth, contexts, sample.relevant_keywords),
                    ))

                results_by_domain[domain] = domain_results

            # Summary
            print("\n  Summary:")
            for domain, results in results_by_domain.items():
                n = len(results)
                f = sum(r.faithfulness for r in results) / n if n > 0 else 0
                p = sum(r.precision for r in results) / n if n > 0 else 0
                r = sum(r.recall for r in results) / n if n > 0 else 0
                print(f"    {domain:15} | faith={f:.2f}  prec={p:.2f}  recall={r:.2f}")

    else:
        print(f"Running single evaluation: {args.llm_extract or 'spaCy baseline'}\n")
        for domain, corpus in LARGE_MULTIDOMAIN_CORPORA.items():
            print(f"{domain.upper()}:")
            rag = GraphRAG(dim=10000, llm=llm)
            if args.llm_extract:
                n_edges = rag.ingest(corpus, llm_extractor=llm, llm_mode=args.llm_extract)
            else:
                n_edges = rag.ingest(corpus)

            domain_q = [q for q in LARGE_MULTIDOMAIN_DATASET if q.domain == domain]
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
                print(f"  Q: {sample.question[:50]}...")
                print(f"     faith={result.faithfulness:.2f}  prec={result.precision:.2f}  recall={result.recall:.2f}")

            print()


if __name__ == "__main__":
    run_large_eval()
