"""
gen_questions.py — generate a larger, typed question set for eval_value.py.

Why this is sound: eval_value uses a PAIRED design (same question to flat and
causal RAG). Any imperfection in a generated question or its reference penalizes
BOTH systems equally, so it cancels in the causal-minus-flat delta. That lets us
scale n cheaply without biasing the comparison — what we need to give the
Wilcoxon test real power (n>=20/type).

Generates K questions per (domain, type) grounded in the document, with a
reference answer and concept keywords, to questions.json.

Run:  python gen_questions.py        (needs ANTHROPIC_API_KEY; a few Sonnet calls)
"""
from __future__ import annotations
import os, re, json, sys

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
from llm_adapters import AnthropicLLM

K = 10  # questions per (domain, type) -> with 2 domains gives n=20/type

DOCS = [("finance", "subprime_causes.md"), ("disaster", "chernobyl.md")]

TYPE_SPEC = {
    "fact": "a SINGLE-HOP factual lookup answerable from one sentence/passage "
            "(who/what/when/how-much). No causal chain needed.",
    "multihop": "a MULTI-HOP causal question: the answer requires tracing a cause "
                "through TWO OR MORE intermediate steps to a distal effect (a chain "
                "A->B->C->D). Phrase it as 'how did X ultimately lead to Y' or "
                "'trace how X propagated into Y'.",
    "rootcause": "a ROOT-CAUSE question: given a stated outcome, ask for the "
                 "underlying/root cause(s), requiring BACKWARD reasoning through the "
                 "causal chain. Phrase as 'what underlying ... caused/primed ...'.",
}

PROMPT = """You are creating an evaluation set from the document below.

Generate exactly {k} questions of this type: {spec}

Each question MUST be answerable from the document. For EACH question output
exactly these three lines and nothing else:
Q: <the question>
REF: <concise reference answer, key facts only>
KW: <2-4 lowercase keywords, comma-separated>

Separate consecutive questions with a line containing only: ---
Do not number them. Do not add any other prose. (Plain text, not JSON — so
quotes and commas inside the answers are fine.)

DOCUMENT:
{doc}
"""


def _parse_blocks(text: str):
    """Parse the Q:/REF:/KW: block format. Robust to any punctuation in values."""
    items = []
    for block in re.split(r"(?m)^\s*---\s*$", text):
        q = re.search(r"(?im)^Q:\s*(.+)", block)
        ref = re.search(r"(?im)^REF:\s*(.+)", block)
        kw = re.search(r"(?im)^KW:\s*(.+)", block)
        if q and ref and kw:
            items.append({
                "q": q.group(1).strip(),
                "reference": ref.group(1).strip(),
                "concepts": [c.strip().lower() for c in kw.group(1).split(",") if c.strip()][:4],
            })
    return items


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required.")
        return
    llm = AnthropicLLM("claude-sonnet-4-6", temperature=0.0)
    out = []
    for domain, path in DOCS:
        doc = open(path, encoding="utf-8").read()
        for qtype, spec in TYPE_SPEC.items():
            prompt = PROMPT.format(k=K, spec=spec, doc=doc[:60000])
            try:
                items = _parse_blocks(llm.generate(prompt))
            except Exception as e:
                print(f"[{domain}/{qtype}] generation/parse failed: {e}")
                continue
            for it in items[:K]:
                if it["q"] and it["reference"] and it["concepts"]:
                    out.append({"domain": domain, "qtype": qtype, **it})
            print(f"[{domain}/{qtype}] {sum(1 for r in out if r['domain']==domain and r['qtype']==qtype)} questions")
    json.dump(out, open("questions.json", "w"), indent=2)
    print(f"\nwrote questions.json: {len(out)} questions")


if __name__ == "__main__":
    main()
