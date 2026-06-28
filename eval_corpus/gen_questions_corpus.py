"""
gen_questions_corpus.py — generate a typed question set for the 23-doc corpus.

For every document in registry.json, generate K questions of each type
(fact / multihop / rootcause) with a reference answer and concept keywords,
using a fixed strong model (Sonnet, temp 0). Paired design: any imperfection
in a generated question penalizes flat and causal equally, so it cancels in
the delta — letting us scale n cheaply without bias.

Run:  python eval_corpus/gen_questions_corpus.py   (needs ANTHROPIC_API_KEY)
Writes: eval_corpus/corpus_questions.json
"""
from __future__ import annotations
import json
import os
import re
import sys

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
from causal_graph_rag.llm_adapters import AnthropicLLM

K = 2  # questions per (doc, type) -> 23 docs x 3 types x 2 = 138 questions

TYPE_SPEC = {
    "fact": "a SINGLE-HOP factual lookup answerable from one sentence/passage "
            "(who/what/when/how-much/which-method). No causal chain needed.",
    "multihop": "a MULTI-HOP causal question whose answer requires tracing a cause "
                "through TWO OR MORE intermediate steps to a distal effect "
                "(a chain A->B->C->D). Phrase as 'how did X ultimately lead to Y' "
                "or 'trace how X propagated into Y'.",
    "rootcause": "a ROOT-CAUSE question: given a stated outcome, ask for the "
                 "underlying/root cause(s), requiring BACKWARD reasoning through the "
                 "causal chain. Phrase as 'what underlying ... caused/primed ...'.",
}

PROMPT = """You are creating an evaluation set from the document below.

Generate exactly {k} questions of this type: {spec}

Each question MUST be answerable from the document and must be SPECIFIC to its
content (not generic). For EACH question output exactly these three lines:
Q: <the question>
REF: <concise reference answer, key facts only>
KW: <2-4 lowercase keywords that must appear in a correct answer, comma-separated>

Separate consecutive questions with a line containing only: ---
No numbering, no other prose. Plain text, not JSON.

DOCUMENT TITLE: {title}
DOCUMENT:
{doc}
"""


def _parse_blocks(text: str):
    items = []
    for block in re.split(r"(?m)^\s*---\s*$", text):
        q = re.search(r"(?im)^Q:\s*(.+)", block)
        ref = re.search(r"(?im)^REF:\s*(.+)", block)
        kw = re.search(r"(?im)^KW:\s*(.+)", block)
        if q and ref and kw:
            items.append({
                "q": q.group(1).strip(),
                "reference": ref.group(1).strip(),
                "concepts": [c.strip().lower() for c in kw.group(1).split(",")
                             if c.strip()][:4],
            })
    return items


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required.")
        return
    registry = json.load(open(os.path.join(HERE, "registry.json"), encoding="utf-8"))
    llm = AnthropicLLM("claude-sonnet-4-6", temperature=0.0)

    out = []
    for entry in registry:
        slug = entry["slug"]
        doc = open(os.path.join(HERE, f"{slug}.md"), encoding="utf-8").read()
        for qtype, spec in TYPE_SPEC.items():
            prompt = PROMPT.format(k=K, spec=spec, title=slug, doc=doc[:60000])
            try:
                items = _parse_blocks(llm.generate(prompt))
            except Exception as e:
                print(f"[{slug}/{qtype}] failed: {e}", file=sys.stderr)
                continue
            for it in items[:K]:
                if it["q"] and it["reference"] and it["concepts"]:
                    out.append({"slug": slug, "domain": entry["domain"],
                                "schema": entry["schema"], "qtype": qtype, **it})
        n = sum(1 for r in out if r["slug"] == slug)
        print(f"  {slug:<18} {n} questions")

    json.dump(out, open(os.path.join(HERE, "corpus_questions.json"), "w"), indent=2)
    print(f"\nwrote corpus_questions.json: {len(out)} questions "
          f"({sum(1 for r in out if r['qtype']=='fact')} fact / "
          f"{sum(1 for r in out if r['qtype']=='multihop')} multihop / "
          f"{sum(1 for r in out if r['qtype']=='rootcause')} rootcause)")


if __name__ == "__main__":
    main()
