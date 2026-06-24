"""
eval_structure.py
=================
Does adding STRUCTURE (causal topology + document organization) to the context
actually improve answers? This benchmark measures it directly by holding
retrieval fixed and varying ONLY what structure the LLM is shown:

  flat        : evidence sentences only                     (structured=F, contextual=F)
  +causal     : + causal chain paths with polarity arrows   (structured=T, contextual=F)
  +causal+doc : + document heading-path annotations          (structured=T, contextual=T)

Because retrieval is identical across conditions, context-recall/precision do
not move — what moves is ANSWER quality. So we score the answers:

  answer_recall : fraction of ground-truth keywords present in the answer
  correctness   : LLM-judged 0..1 match of the answer to a reference answer

Run:  python eval_structure.py     (uses GROQ_API_KEY from .env)
"""
from __future__ import annotations
import os, re, sys
from dataclasses import dataclass, field
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
from pipeline import MockLLM


# --------------------------------------------------------------------------- #
#  Structured documents + structural questions
# --------------------------------------------------------------------------- #
RESEARCH_MD = """# Structure-Aware Causal Retrieval

## Abstract
We show that preserving causal structure during retrieval improves multi-hop
question answering. Our main finding is a large recall gain over flat retrieval.

## Methods
We extract causal edges and encode them with a vector-symbolic architecture.
The graph is traversed to return whole causal chains as the retrieval unit.

## Results
Recall improved from 0.31 to 0.53 on the healthcare benchmark. Faithfulness
rose to 0.77. Precision stayed high at 0.85.

## Conclusion
Causal structure is the key driver of the recall gain. Flat chunking discards
the cause-to-effect links that multi-hop questions depend on.
"""

INCIDENT_MD = """# Incident Report: District Power Outage

## Timeline
The reactor overheated during the night shift. The overheating caused the
coolant valve to fail. The valve failure triggered an emergency shutdown.

## Impact
The shutdown reduced power output. The reduced power output disrupted hospital
operations across the district for twelve hours.

## Root Cause
A deferred maintenance schedule left the cooling pump degraded, which allowed
the reactor to overheat under peak load.

## Remediation
Restore the maintenance schedule and add a redundant cooling pump. Install
overheating alarms with automatic load shedding.
"""


@dataclass
class Q:
    q: str
    reference: str
    keywords: List[str]


@dataclass
class Doc:
    doc_id: str
    schema: str
    text: str
    questions: List[Q]


DOCS = [
    Doc("paper", "research", RESEARCH_MD, [
        Q("What is the paper's main finding?",
          "Preserving causal structure improves recall; a large recall gain over flat retrieval.",
          ["recall", "causal structure"]),
        Q("What evidence supports the main claim, and what drives the improvement?",
          "Recall improved from 0.31 to 0.53 with faithfulness 0.77; causal structure is the driver.",
          ["0.31", "0.53", "causal structure"]),
        Q("Why does flat chunking underperform here?",
          "Flat chunking discards the cause-to-effect links that multi-hop questions depend on.",
          ["cause", "links", "multi-hop"]),
    ]),
    Doc("incident", "incident", INCIDENT_MD, [
        Q("What was the root cause of the outage and what did it ultimately impact?",
          "Deferred maintenance degraded the cooling pump, causing overheating; it disrupted hospital operations.",
          ["maintenance", "overheat", "hospital operations"]),
        Q("Did the emergency shutdown raise or lower power output, and what followed?",
          "The shutdown reduced power output, which disrupted hospital operations.",
          ["reduced", "power output", "hospital operations"]),
        Q("What corrective actions were recommended?",
          "Restore maintenance, add a redundant cooling pump, install overheating alarms with load shedding.",
          ["maintenance", "redundant", "alarms"]),
    ]),
]

CONDITIONS = {
    "flat":         dict(structured=False, contextual=False),
    "+causal":      dict(structured=True,  contextual=False),
    "+causal+doc":  dict(structured=True,  contextual=True),
}


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def keyword_recall(answer: str, keywords: List[str]) -> float:
    a = answer.lower()
    return sum(1 for k in keywords if k.lower() in a) / len(keywords) if keywords else 0.0


_NUM = re.compile(r"\b(0?\.\d+|\d+(?:\.\d+)?)\b")


def judge_correctness(llm, question: str, reference: str, candidate: str) -> float:
    prompt = (
        "You are grading a candidate answer against a reference answer. "
        "Score from 0.0 to 1.0 how well the CANDIDATE captures the key facts of "
        "the REFERENCE. Reply with ONLY a number between 0 and 1.\n\n"
        f"Question: {question}\nReference: {reference}\nCandidate: {candidate}\nScore:"
    )
    try:
        out = llm.generate(prompt)
        m = _NUM.search(out or "")
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
def main():
    llm = GroqLLM() if os.environ.get("GROQ_API_KEY") else MockLLM()
    print(f"LLM: {type(llm).__name__}\n")

    agg: Dict[str, Dict[str, float]] = {c: {"kr": 0.0, "corr": 0.0, "n": 0} for c in CONDITIONS}

    for doc in DOCS:
        rag = GraphRAG(dim=10000, llm=llm)
        rag.ingest(doc.text, schema=doc.schema)
        print(f"[{doc.doc_id}] schema={doc.schema}  ({len(doc.questions)} questions)")
        for item in doc.questions:
            for cond, opts in CONDITIONS.items():
                ans, _ = rag.answer(item.q, top_k=4, **opts)
                kr = keyword_recall(ans, item.keywords)
                corr = judge_correctness(llm, item.q, item.reference, ans)
                agg[cond]["kr"] += kr
                agg[cond]["corr"] += corr
                agg[cond]["n"] += 1

    print("\n" + "=" * 64)
    print("RESULTS - answer quality by structure shown to the LLM")
    print("=" * 64)
    print(f"{'condition':<14} {'answer_recall':>14} {'correctness':>13}")
    print("-" * 64)
    base = None
    for cond in CONDITIONS:
        n = agg[cond]["n"] or 1
        kr = agg[cond]["kr"] / n
        corr = agg[cond]["corr"] / n
        if base is None:
            base = (kr, corr)
        dkr = kr - base[0]
        dcorr = corr - base[1]
        tag = "" if cond == "flat" else f"   (vs flat: recall {dkr:+.2f}, correct {dcorr:+.2f})"
        print(f"{cond:<14} {kr:>14.2f} {corr:>13.2f}{tag}")
    print("=" * 64)


if __name__ == "__main__":
    main()
