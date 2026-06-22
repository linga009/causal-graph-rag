"""
pipeline.py
===========
End-to-end relation-aware RAG built on the VSA core.

Flow:
  INGEST   : split doc -> sentences -> SVO triples -> encode -> VSAMemory,
             and keep plaintext chunks in a side store keyed by id.
  RETRIEVE : parse query -> triple(s) -> nearest stored triples by Hamming sim.
  AUGMENT  : pull plaintext chunks for the winning triples.
  GENERATE : hand context + question to an LLM (pluggable; mock by default).

Swap MockLLM for a real client (Groq / OpenAI / Anthropic) by implementing
the .generate(prompt) method.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple

from vsa_core import Lexicon, VSAMemory, Triple
from parser import parse_triples, backend_name


# --------------------------------------------------------------------------- #
#  Plaintext store (stand-in for Postgres/Mongo)
# --------------------------------------------------------------------------- #
class PlaintextStore:
    def __init__(self):
        self._d = {}
    def put(self, cid: str, text: str): self._d[cid] = text
    def get(self, cid: str) -> str:     return self._d.get(cid, "")


# --------------------------------------------------------------------------- #
#  LLM interface
# --------------------------------------------------------------------------- #
class MockLLM:
    """Deterministic stand-in so the pipeline is testable offline.
    Replace with a real client; signature is generate(prompt)->str."""
    def generate(self, prompt: str) -> str:
        ctx = ""
        m = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.S)
        if m:
            ctx = m.group(1).strip()
        q = ""
        mq = re.search(r"Question:\s*(.*)", prompt)
        if mq:
            q = mq.group(1).strip()
        first = ctx.split("\n")[0] if ctx else "(no context retrieved)"
        return f"[grounded answer to '{q}'] Based on retrieved context: {first}"


# --------------------------------------------------------------------------- #
#  The engine
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = (
    "Using ONLY the following retrieved context, answer the user's question. "
    "If the context does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
)


@dataclass
class Retrieved:
    score: float
    triple: Triple
    chunk_text: str
    doc_id: str


class VSARAG:
    def __init__(self, dim: int = 10000, semantic_weight: int = 1,
                 llm: Optional[object] = None, match_threshold: float = 0.45):
        self.lex = Lexicon(dim=dim, semantic_weight=semantic_weight)
        self.mem = VSAMemory(self.lex)
        self.store = PlaintextStore()
        self.llm = llm or MockLLM()
        self.match_threshold = match_threshold
        self._n = 0

    # -- ingest -------------------------------------------------------------- #
    def ingest(self, text: str, doc_id: str) -> int:
        added = 0
        for sent in _sentences(text):
            for tr in parse_triples(sent):
                cid = f"{doc_id}::chunk{self._n}"
                self.store.put(cid, sent.strip())
                self.mem.add(tr, doc_id, sent.strip())
                self._n += 1
                added += 1
        return added

    # -- retrieve ------------------------------------------------------------ #
    def retrieve(self, question: str, top_k: int = 3) -> List[Retrieved]:
        q_triples = parse_triples(question)
        if not q_triples:
            return []
        # score each query triple; keep best hits across all of them
        best = {}
        for qt in q_triples:
            for score, rec in self.mem.query(qt, top_k=top_k):
                key = id(rec)
                if key not in best or score > best[key][0]:
                    best[key] = (score, rec)
        ranked = sorted(best.values(), key=lambda x: -x[0])[:top_k]
        return [Retrieved(s, r.triple, r.chunk_text, r.doc_id)
                for s, r in ranked if s >= self.match_threshold]

    # -- generate ------------------------------------------------------------ #
    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[Retrieved]]:
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return ("No structurally-matching context was retrieved.", [])
        context = "\n".join(f"- {h.chunk_text}" for h in hits)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        return self.llm.generate(prompt), hits


# --------------------------------------------------------------------------- #
def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]
