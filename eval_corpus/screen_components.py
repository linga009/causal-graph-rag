"""
screen_components.py — decide which proposed components survive, for FREE.

The expensive part of an eval is the LLM (generation + judge). But every
proposed component (real-embedding VSA, chain holography, log-signature rerank,
beam BFS, DPP selection) only changes WHAT IS RETRIEVED. So we screen them with
a no-LLM retrieval-quality proxy:

    concept_coverage(q) = fraction of the question's reference concepts (KW)
                          that appear anywhere in the retrieved evidence
                          (chain texts + provenance + coverage sentences).

The generator can only answer correctly if the evidence is present, so this
proxy is a fast, free, deterministic predictor of downstream correctness. We
run it over the 138-question corpus for the baseline and for each component
toggled on, and keep the ones that raise coverage on the reasoning questions
(multihop + rootcause) without hurting facts.

Run:  python eval_corpus/screen_components.py     (no API key needed)
"""
from __future__ import annotations
import json
import os
import sys
from typing import Callable, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import vsa_core
from graph_rag import GraphRAG

QTYPES = ["fact", "multihop", "rootcause"]
TOP_K = 6


# --------------------------------------------------------------------------- #
#  Config = a name + a function that sets flags on a fresh GraphRAG.
#  Flags default OFF / baseline; each config flips exactly one thing so the
#  screen attributes any coverage change to that component alone.
# --------------------------------------------------------------------------- #
def cfg_baseline(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = False
    _set_flags(rag, logsig=False, holo=False, beam=False, dpp=False)


def cfg_vsa_realemb(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = True
    _set_flags(rag, logsig=False, holo=False, beam=False, dpp=False)


def cfg_logsig(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = True
    _set_flags(rag, logsig=True, holo=False, beam=False, dpp=False)


def cfg_holo(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = True
    _set_flags(rag, logsig=False, holo=True, beam=False, dpp=False)


def cfg_beam(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = True
    _set_flags(rag, logsig=False, holo=False, beam=True, dpp=False)


def cfg_dpp(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = True
    _set_flags(rag, logsig=False, holo=False, beam=False, dpp=True)


def cfg_ppr(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = False     # baseline VSA (real-emb shown inert)
    _set_flags(rag, logsig=False, holo=False, beam=False, dpp=False)
    rag.flag_ppr = True


def cfg_proposition(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = False
    _set_flags(rag, logsig=False, holo=False, beam=False, dpp=False)
    rag.flag_proposition = True


def cfg_calibrated_fusion(rag: GraphRAG) -> None:
    vsa_core.USE_REAL_EMBEDDINGS = False
    _set_flags(rag, logsig=False, holo=False, beam=False, dpp=False)
    rag.flag_calibrated_fusion = True


def cfg_combined(rag: GraphRAG) -> None:
    """The two survivors stacked — the candidate shipping config."""
    vsa_core.USE_REAL_EMBEDDINGS = False
    _set_flags(rag, logsig=False, holo=False, beam=False, dpp=False)
    rag.flag_proposition = True
    rag.flag_calibrated_fusion = True


def _set_flags(rag: GraphRAG, *, logsig, holo, beam, dpp) -> None:
    rag.flag_logsig = logsig
    rag.flag_chain_holo = holo
    rag.flag_beam = beam
    rag.flag_dpp = dpp
    # round-2 leads default off; each cfg_* enables exactly one
    rag.flag_ppr = False
    rag.flag_proposition = False
    rag.flag_calibrated_fusion = False


CONFIGS: Dict[str, Callable[[GraphRAG], None]] = {
    "baseline (trigram VSA)": cfg_baseline,
    "+ proposition_rerank":   cfg_proposition,
    "+ calibrated_fusion":    cfg_calibrated_fusion,
    "+ both (ship candidate)": cfg_combined,
}


# --------------------------------------------------------------------------- #
def evidence_blobs(rag: GraphRAG, question: str):
    """Return (full_blob, chains_only_blob).

    full        = top-k chains' provenance + chain texts + coverage sentences
                  (what the generator actually sees).
    chains_only = ONLY the top-3 chains' provenance + node names (no coverage
                  sentences) — isolates the graph/rerank components' effect,
                  which the coverage sentences otherwise mask."""
    chains = rag.retrieve(question, top_k=TOP_K)
    full: List[str] = []
    chains_only: List[str] = []
    chain_nodes = set()
    for rank, c in enumerate(chains):
        full.append(c.text())
        full.extend(c.provenance())
        for e in c.chain:
            chain_nodes.add(e.cause)
            chain_nodes.add(e.effect)
        if rank < 3:                       # top-3 chains isolate ranking effects
            chains_only.append(c.text())
            chains_only.extend(c.provenance())
    full.extend(rag._retrieve_sentences(question, k=TOP_K * 2,
                                        chain_nodes=chain_nodes or None))
    return " ".join(full).lower(), " ".join(chains_only).lower()


def concept_coverage(evidence: str, concepts: List[str]) -> float:
    if not concepts:
        return 0.0
    return sum(1 for c in concepts if c.lower() in evidence) / len(concepts)


def run_config(name: str, setup, questions, by_slug) -> Dict[str, Dict[str, float]]:
    # Per-config coverage (full and chains-only), averaged within question type.
    full_s: Dict[str, List[float]] = {qt: [] for qt in QTYPES}
    chain_s: Dict[str, List[float]] = {qt: [] for qt in QTYPES}
    for slug, qs in by_slug.items():
        schema = qs[0]["schema"]
        text = open(os.path.join(HERE, f"{slug}.md"), encoding="utf-8").read()
        rag = GraphRAG(dim=10000)          # fresh Lexicon/caches per config
        setup(rag)                          # set flags BEFORE ingest (VSA edge hv)
        rag.ingest(text, schema=schema)
        for item in qs:
            full_ev, chain_ev = evidence_blobs(rag, item["q"])
            full_s[item["qtype"]].append(concept_coverage(full_ev, item["concepts"]))
            chain_s[item["qtype"]].append(concept_coverage(chain_ev, item["concepts"]))
    return {
        "full":  {qt: float(np.mean(v)) if v else 0.0 for qt, v in full_s.items()},
        "chain": {qt: float(np.mean(v)) if v else 0.0 for qt, v in chain_s.items()},
    }


def main():
    qpath = os.path.join(HERE, "corpus_questions.json")
    if not os.path.exists(qpath):
        print("corpus_questions.json missing — run gen_questions_corpus.py first.")
        return
    questions = json.load(open(qpath, encoding="utf-8"))
    by_slug: Dict[str, List[dict]] = {}
    for q in questions:
        by_slug.setdefault(q["slug"], []).append(q)

    print(f"screening {len(CONFIGS)} configs on {len(questions)} questions "
          f"({len(by_slug)} docs), no LLM\n")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, setup in CONFIGS.items():
        results[name] = run_config(name, setup, questions, by_slug)

    base = results["baseline (trigram VSA)"]

    def reasoning(res, metric):
        return res[metric]["multihop"] + res[metric]["rootcause"]

    # -- survival table: FULL evidence (what generator sees) ----------------- #
    print("\n" + "=" * 80)
    print("COMPONENT SCREEN (no LLM) — concept coverage of retrieved evidence")
    print("FULL = chains + coverage sentences   |   CHAINS = top-3 chains only")
    print("=" * 80)
    print(f"{'config':<26}{'full reason':>13}{'chain reason':>14}"
          f"{'full d':>9}{'chain d':>9}")
    print("-" * 80)
    bf, bc = reasoning(base, "full"), reasoning(base, "chain")
    for name, res in results.items():
        f_reason, c_reason = reasoning(res, "full"), reasoning(res, "chain")
        fd, cd = f_reason - bf, c_reason - bc
        if name == "baseline (trigram VSA)":
            verdict = ""
        else:
            # survives if it improves EITHER the full evidence the generator sees
            # OR (for rerank components) the isolated chain coverage
            verdict = "  SURVIVES" if (fd > 0.005 or cd > 0.005) else "  drop"
        print(f"{name:<26}{f_reason:>13.3f}{c_reason:>14.3f}{fd:>+9.3f}{cd:>+9.3f}{verdict}")
    print("-" * 80)
    print("reason = multihop+rootcause coverage. d = vs baseline. "
          "chain isolates graph/rerank effects the coverage sentences mask.")
    json.dump(results, open(os.path.join(HERE, "screen_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
