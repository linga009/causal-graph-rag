"""
eval_value.py
=============
The experiment that answers "is the causal graph actually useful?" — by testing
it in the regime where it should win and against a STRONG baseline.

  Systems (apples-to-apples: same encoder, same top_k, same LLM):
    flat   = strong vector RAG — dense retrieval over sentences -> LLM.
    causal = this project — causal-graph traversal + structure -> LLM.

  Domains (different causal topology):
    finance   = Subprime crisis Causes  (shallow many-to-one fan-in, explicit).
    disaster  = Chernobyl accident       (deep linear cascade, multi-hop).

  Question types:
    fact      = single-hop fact lookup           (vector RAG expected to win/tie).
    multihop  = trace a cause to a distal effect  (causal graph expected to win).
    rootcause = backward "what underlying cause"  (causal graph expected to win).

  Metrics, logged PER QUESTION (so we can do real statistics):
    correctness = LLM-judged 0..1 match to a reference answer (Sonnet judge).
    kw_recall   = fraction of reference concepts present in the answer.

  Statistics (per question-type, paired causal-minus-flat on correctness):
    mean delta, 95% bootstrap CI, Wilcoxon signed-rank p-value.

Run:  python eval_value.py        (needs ANTHROPIC_API_KEY; ~$0.50 of Haiku+Sonnet)
"""
from __future__ import annotations
import os, re, sys, json
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

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
from llm_adapters import AnthropicLLM
from retrievers import SentenceTransformerDense

GEN_MODEL = "claude-haiku-4-5"      # weak/cheap generation model
JUDGE_MODEL = "claude-sonnet-4-6"   # stronger fixed judge
TOP_K = 6


# --------------------------------------------------------------------------- #
#  Strong flat-RAG baseline: dense retrieval over sentences (no graph).
# --------------------------------------------------------------------------- #
class FlatRAG:
    def __init__(self, llm):
        self.llm = llm
        self._sents: List[str] = []
        self._dense = SentenceTransformerDense()

    def ingest(self, text: str) -> None:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.strip()) > 15 and not s.strip().startswith("#")]
        self._sents = sents
        self._dense.index({str(i): s for i, s in enumerate(sents)})

    def answer(self, q: str, top_k: int = TOP_K):
        scored = self._dense.score(q)[:top_k]
        ctx = [self._sents[int(n)] for _, n in scored]
        prompt = ("You are a helpful assistant. Answer the question using ONLY the "
                  "evidence sentences provided. Be direct and concise.\n\n"
                  "Evidence:\n" + "\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:")
        return self.llm.generate(prompt), ctx


# --------------------------------------------------------------------------- #
#  Questions (grounded in the documents), tagged by type.
# --------------------------------------------------------------------------- #
@dataclass
class Q:
    qtype: str
    q: str
    reference: str
    concepts: List[str]


FINANCE = [
    Q("fact", "What were credit rating agencies criticized for during the crisis?",
      "Giving high (AAA) ratings to risky mortgage-backed securities.",
      ["rating", "securit"]),
    Q("fact", "What is securitization, as described here?",
      "Bundling mortgages into securities that are sold to investors.",
      ["securit", "investor"]),
    Q("fact", "What role did adjustable-rate or high-risk mortgages play?",
      "High-risk/subprime and adjustable-rate mortgages were issued to weak borrowers.",
      ["subprime", "mortgage"]),
    Q("multihop", "How did lax mortgage lending ultimately cause losses at financial institutions far removed from homebuyers?",
      "Lax lending produced subprime mortgages, securitized into mortgage-backed securities sold to institutions; when borrowers defaulted, those securities lost value, causing institution losses.",
      ["subprime", "securit", "default", "loss"]),
    Q("multihop", "Trace how the bursting of the housing bubble propagated into a wider financial crisis.",
      "The bubble burst, home prices fell, mortgage defaults and foreclosures rose, mortgage-backed security values collapsed, and financial institutions suffered losses and a credit freeze.",
      ["price", "default", "securit", "loss"]),
    Q("rootcause", "What underlying lending and incentive practices set the stage for the crisis?",
      "Lax and predatory lending, subprime loans, originate-to-distribute incentives, and weak regulation.",
      ["lending", "subprime", "regulat"]),
    Q("rootcause", "What was the underlying role of leverage in the shadow banking system?",
      "High leverage and debt in the shadow banking system amplified the collapse when asset values fell.",
      ["shadow", "leverage"]),
]

DISASTER = [
    Q("fact", "What was the AZ-5 button?",
      "The emergency shutdown (scram) button for the reactor.",
      ["az-5", "shutdown"]),
    Q("fact", "How high did the reactor power spike during the excursion?",
      "Around 30,000 MW thermal, about ten times normal operating output.",
      ["30,000", "mw"]),
    Q("fact", "What was the safety test intended to confirm?",
      "Whether the turbine run-down could power coolant pumps during a power loss.",
      ["test", "turbine"]),
    Q("multihop", "Trace the sequence from the safety test to the destruction of the reactor.",
      "The low-power test conditions led to pressing the AZ-5 scram, which caused a power excursion to ~30,000 MW, then a steam explosion and a second larger explosion that destroyed the reactor.",
      ["test", "scram", "excursion", "explosion"]),
    Q("multihop", "How did pressing the emergency shutdown button ultimately lead to the explosions?",
      "Inserting the control rods via AZ-5 caused a power surge/excursion, fuel overheating and steam pressure, producing a steam explosion and then a second explosion.",
      ["scram", "excursion", "steam", "explosion"]),
    Q("rootcause", "What underlying conditions primed the reactor for the accident before the scram?",
      "Reactor design flaws (positive void coefficient, control-rod design), very low power with xenon poisoning, and disabled/overridden safety systems during the test.",
      ["design", "power", "test"]),
    Q("rootcause", "What is believed to be the root cause of the second, larger explosion?",
      "Either combustion of hydrogen from steam-zirconium/graphite-steam reactions, or a thermal explosion from the uncontrolled power excursion.",
      ["hydrogen", "explosion"]),
]

DOCS = [
    ("finance", "subprime_causes.md", FINANCE),
    ("disaster", "chernobyl.md", DISASTER),
]


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def kw_recall(answer: str, concepts: List[str]) -> float:
    a = answer.lower()
    return sum(1 for c in concepts if c.lower() in a) / len(concepts) if concepts else 0.0


_NUM = re.compile(r"\b(0?\.\d+|\d(?:\.\d+)?)\b")


def judge_correctness(judge, question, reference, candidate) -> float:
    prompt = ("Grade the candidate answer against the reference. Score 0.0-1.0 for "
              "how well the candidate captures the reference's key facts. Reply with "
              f"ONLY a number 0-1.\n\nQuestion: {question}\nReference: {reference}\n"
              f"Candidate: {candidate}\nScore:")
    try:
        out = judge.generate(prompt)
        m = _NUM.search(out or "")
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
#  Statistics: paired causal-minus-flat on correctness, per question type.
# --------------------------------------------------------------------------- #
def paired_stats(deltas: List[float]) -> Dict[str, float]:
    from scipy.stats import wilcoxon
    arr = np.array(deltas, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else 0.0
    # 95% bootstrap CI of the mean delta
    rng = np.random.default_rng(0)
    boot = [rng.choice(arr, size=n, replace=True).mean() for _ in range(2000)] if n else [0.0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Wilcoxon signed-rank (needs some non-zero deltas)
    try:
        p = float(wilcoxon(arr)[1]) if np.any(arr != 0) else 1.0
    except Exception:
        p = float("nan")
    return {"n": n, "mean": mean, "ci_lo": float(lo), "ci_hi": float(hi), "p": p}


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required.")
        return
    gen = AnthropicLLM(GEN_MODEL, temperature=0.0)
    judge = AnthropicLLM(JUDGE_MODEL, temperature=0.0)

    rows = []   # per-question records
    for domain, path, qs in DOCS:
        text = open(path, encoding="utf-8").read()
        print(f"[{domain}] ingesting {path} ...")
        causal = GraphRAG(dim=10000, llm=gen)
        causal.ingest(text, schema="auto")
        flat = FlatRAG(gen)
        flat.ingest(text)

        for item in qs:
            for system, rag in (("flat", flat), ("causal", causal)):
                if system == "causal":
                    ans, _ = rag.answer(item.q, top_k=TOP_K)
                else:
                    ans, _ = rag.answer(item.q, top_k=TOP_K)
                corr = judge_correctness(judge, item.q, item.reference, ans)
                kr = kw_recall(ans, item.concepts)
                rows.append(dict(domain=domain, qtype=item.qtype, q=item.q,
                                 system=system, correctness=corr, kw=kr))
        print(f"  done ({len(qs)} questions x 2 systems)")

    json.dump(rows, open("eval_value_raw.json", "w"), indent=2)

    # -- aggregate + paired stats by question type --------------------------- #
    def avg(qtype, system, metric):
        vals = [r[metric] for r in rows if r["qtype"] == qtype and r["system"] == system]
        return sum(vals) / len(vals) if vals else 0.0

    qtypes = ["fact", "multihop", "rootcause"]
    print("\n" + "=" * 74)
    print("RESULTS - causal-graph RAG vs strong flat baseline (correctness)")
    print("=" * 74)
    print(f"{'qtype':<11}{'flat':>7}{'causal':>8}{'delta':>8}{'95% CI':>16}{'p':>8}")
    print("-" * 74)
    for qt in qtypes:
        flat_c = avg(qt, "flat", "correctness")
        caus_c = avg(qt, "causal", "correctness")
        # paired deltas per question (same question, causal - flat)
        qset = sorted({r["q"] for r in rows if r["qtype"] == qt})
        deltas = []
        for q in qset:
            c = next(r["correctness"] for r in rows if r["q"] == q and r["system"] == "causal")
            f = next(r["correctness"] for r in rows if r["q"] == q and r["system"] == "flat")
            deltas.append(c - f)
        st = paired_stats(deltas)
        ci = f"[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}]"
        print(f"{qt:<11}{flat_c:>7.2f}{caus_c:>8.2f}{st['mean']:>+8.2f}{ci:>16}{st['p']:>8.3f}")
    print("-" * 74)
    print("delta = causal - flat (correctness). p = Wilcoxon signed-rank (paired).")
    print("Raw per-question scores: eval_value_raw.json")


if __name__ == "__main__":
    main()
