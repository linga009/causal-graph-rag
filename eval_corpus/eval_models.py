"""
eval_models.py — causal-graph RAG vs strong flat baseline across THREE models.

Runs the same paired comparison as eval_value.py but over the 23-document
multi-field corpus AND across three generation models (Haiku, Sonnet, Opus),
with a fixed Sonnet judge. Answers two questions:

  1. Does causal-graph retrieval beat flat RAG on multi-hop / root-cause
     questions, and hold the line on facts — across many fields?
  2. CAPABILITY SCALING: does the causal advantage shrink as the generation
     model gets stronger (Haiku -> Opus)? If a weak model gains most, the tool
     is most valuable exactly where compute is cheapest.

Efficiency: the causal graph and the retrieved context are generation-model
INDEPENDENT (extraction is spaCy/rules; retrieval is vector/graph). So each
document is ingested once and each question retrieved once; only generation
and judging vary per model.

Run:  python eval_corpus/eval_models.py   (needs ANTHROPIC_API_KEY)
Writes: eval_corpus/results_raw.json
"""
from __future__ import annotations
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _load_env(path=os.path.join(ROOT, ".env")):
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_env()
from causal_graph_rag.graph_rag import GraphRAG
from causal_graph_rag.llm_adapters import AnthropicLLM
from causal_graph_rag.retrievers import SentenceTransformerDense

# The two validated retrieval components (proposition-aware rerank + calibrated
# fusion) are ON by default in GraphRAG — no per-run configuration needed.
TOP_K = 6
GEN_MODELS = [
    ("haiku",  "claude-haiku-4-5"),
    ("sonnet", "claude-sonnet-4-6"),
]
JUDGE_MODEL = "claude-sonnet-4-6"
QTYPES = ["fact", "multihop", "rootcause"]


# --------------------------------------------------------------------------- #
#  Flat baseline: dense retrieval over sentences (retrieval separated from gen)
# --------------------------------------------------------------------------- #
class FlatRAG:
    def __init__(self):
        self._sents: List[str] = []
        self._dense = SentenceTransformerDense()

    def ingest(self, text: str) -> None:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.strip()) > 15 and not s.strip().startswith("#")]
        self._sents = sents
        self._dense.index({str(i): s for i, s in enumerate(sents)})

    def context(self, q: str, top_k: int = TOP_K) -> List[str]:
        scored = self._dense.score(q)[:top_k]
        return [self._sents[int(n)] for _, n in scored]


def flat_prompt(q: str, ctx: List[str]) -> str:
    return ("You are a helpful assistant. Answer the question using ONLY the "
            "evidence sentences provided. Be direct and concise.\n\n"
            "Evidence:\n" + "\n".join(ctx) + f"\n\nQuestion: {q}\n\nAnswer:")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"\b(0?\.\d+|\d(?:\.\d+)?)\b")


def judge_correctness(judge, question, reference, candidate) -> float:
    prompt = ("Grade the candidate answer against the reference. Score 0.0-1.0 for "
              "how well the candidate captures the reference's key facts. Reply with "
              f"ONLY a number 0-1.\n\nQuestion: {question}\nReference: {reference}\n"
              f"Candidate: {candidate}\nScore:")
    try:
        m = _NUM.search(judge.generate(prompt) or "")
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0
    except Exception:
        return 0.0


def paired_stats(deltas: List[float]) -> Dict[str, float]:
    from scipy.stats import wilcoxon
    arr = np.array(deltas, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else 0.0
    rng = np.random.default_rng(0)
    boot = [rng.choice(arr, size=n, replace=True).mean() for _ in range(2000)] if n else [0.0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        p = float(wilcoxon(arr)[1]) if np.any(arr != 0) else 1.0
    except Exception:
        p = float("nan")
    return {"n": n, "mean": mean, "ci_lo": float(lo), "ci_hi": float(hi), "p": p}


# --------------------------------------------------------------------------- #
def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required.")
        return
    qpath = os.path.join(HERE, "corpus_questions.json")
    if not os.path.exists(qpath):
        print("corpus_questions.json missing — run gen_questions_corpus.py first.")
        return

    questions = json.load(open(qpath, encoding="utf-8"))
    by_slug: Dict[str, List[dict]] = {}
    for q in questions:
        by_slug.setdefault(q["slug"], []).append(q)

    llms = {name: AnthropicLLM(mid, temperature=0.0) for name, mid in GEN_MODELS}
    judge = AnthropicLLM(JUDGE_MODEL, temperature=0.0)

    print(f"{len(questions)} questions across {len(by_slug)} docs x "
          f"{len(GEN_MODELS)} models\n")

    rows = []
    for slug, qs in by_slug.items():
        schema = qs[0]["schema"]
        domain = qs[0]["domain"]
        text = open(os.path.join(HERE, f"{slug}.md"), encoding="utf-8").read()

        causal = GraphRAG(dim=10000, llm=llms["haiku"])
        causal.flag_proposition = True          # survived free screen
        causal.flag_calibrated_fusion = True    # survived free screen
        causal.ingest(text, schema=schema)
        flat = FlatRAG()
        flat.ingest(text)

        for item in qs:
            fctx = flat.context(item["q"], top_k=TOP_K)
            fprompt = flat_prompt(item["q"], fctx)

            for mname, _ in GEN_MODELS:
                llm = llms[mname]
                causal.llm = llm
                try:
                    # Use the REAL answer() path: score gate + hybrid coverage
                    # sentences + chains. (Calling generate(q, chains) directly
                    # bypasses the coverage hybrid and tanks fact questions.)
                    c_ans, _ = causal.answer(item["q"], top_k=TOP_K)
                except Exception as e:
                    c_ans = f"[gen error: {e}]"
                try:
                    f_ans = llm.generate(fprompt)
                except Exception as e:
                    f_ans = f"[gen error: {e}]"

                c_corr = judge_correctness(judge, item["q"], item["reference"], c_ans)
                f_corr = judge_correctness(judge, item["q"], item["reference"], f_ans)
                rows.append(dict(model=mname, slug=slug, domain=domain,
                                 qtype=item["qtype"], q=item["q"],
                                 causal=c_corr, flat=f_corr))
        print(f"  {slug:<18} done ({len(qs)} q x {len(GEN_MODELS)} models)")

    json.dump(rows, open(os.path.join(HERE, "results_raw.json"), "w"), indent=2)
    _report(rows)


def _report(rows):
    def deltas(model, qtype):
        sub = [r for r in rows if r["model"] == model and r["qtype"] == qtype]
        return [r["causal"] - r["flat"] for r in sub], sub

    print("\n" + "=" * 78)
    print("RESULTS — causal-graph RAG vs strong flat baseline, BY MODEL")
    print("=" * 78)
    for model, _ in GEN_MODELS:
        print(f"\n### {model.upper()}")
        print(f"{'qtype':<11}{'flat':>7}{'causal':>8}{'delta':>8}{'95% CI':>16}{'p':>8}")
        print("-" * 58)
        for qt in QTYPES:
            d, sub = deltas(model, qt)
            flat_m = float(np.mean([r["flat"] for r in sub])) if sub else 0.0
            caus_m = float(np.mean([r["causal"] for r in sub])) if sub else 0.0
            st = paired_stats(d)
            ci = f"[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}]"
            print(f"{qt:<11}{flat_m:>7.2f}{caus_m:>8.2f}{st['mean']:>+8.2f}{ci:>16}{st['p']:>8.3f}")

    # Capability scaling: delta per model per reasoning qtype
    print("\n" + "=" * 78)
    print("CAPABILITY SCALING — causal-minus-flat delta by model")
    print("=" * 78)
    print(f"{'qtype':<11}" + "".join(f"{m:>10}" for m, _ in GEN_MODELS))
    print("-" * 42)
    for qt in QTYPES:
        line = f"{qt:<11}"
        for model, _ in GEN_MODELS:
            d, _ = deltas(model, qt)
            line += f"{np.mean(d):>+10.2f}" if d else f"{'-':>10}"
        print(line)

    # Per-field breakdown pooled across models (reasoning questions only)
    print("\n" + "=" * 78)
    print("BY FIELD — causal-minus-flat delta (multihop+rootcause, pooled models)")
    print("=" * 78)
    fields = sorted({r["domain"] for r in rows})
    print(f"{'field':<14}{'delta':>8}{'n':>6}")
    print("-" * 28)
    for fld in fields:
        sub = [r for r in rows if r["domain"] == fld and r["qtype"] in ("multihop", "rootcause")]
        d = [r["causal"] - r["flat"] for r in sub]
        print(f"{fld:<14}{np.mean(d):>+8.2f}{len(d):>6}" if d else f"{fld:<14}{'-':>8}")
    print("\nRaw per-question scores: eval_corpus/results_raw.json")


if __name__ == "__main__":
    main()
