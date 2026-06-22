"""
causal_extractor.py
===================
Extract DIRECTED causal/consequential edges from text — the structure that
chunking + embedding destroys.

Two sources of edges:

  (A) INTRA-SENTENCE   "X causes Y", "Y is triggered by X", "X prevents Y"
      Parsed via the SVO triple parser; the verb's class gives polarity.

  (B) INTER-SENTENCE   "The reactor overheated. As a result, the valve failed.
                        This triggered a shutdown."
      Discourse connectives ("as a result", "consequently", "this triggered",
      "therefore", "led to", "because of this") link the SUBJECT/event of one
      sentence to the next, rebuilding the chain across chunk boundaries.

Output: a list of CausalEdge(cause, relation, effect, polarity, source_sent).
Polarity: +1 promotes/produces, -1 prevents/reduces. Direction is always
cause -> effect.

Dependency-light: uses the existing rule/spaCy parser from parser.py plus a
connective lexicon. Swap in a trained relation-extraction model in production.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional

from parser import parse_triples, _clean, _split_clauses
from vsa_core import Triple


# --- verb -> (canonical relation, polarity) -------------------------------- #
# polarity +1 : cause brings effect about / increases it
# polarity -1 : cause suppresses / reduces / prevents effect
CAUSAL_VERBS = {
    "cause": ("cause", +1), "causes": ("cause", +1), "caused": ("cause", +1),
    "trigger": ("trigger", +1), "triggers": ("trigger", +1), "triggered": ("trigger", +1),
    "lead": ("lead_to", +1), "leads": ("lead_to", +1), "led": ("lead_to", +1),
    "produce": ("produce", +1), "produces": ("produce", +1), "produced": ("produce", +1),
    "drive": ("drive", +1), "drives": ("drive", +1), "drove": ("drive", +1),
    "increase": ("increase", +1), "increases": ("increase", +1), "increased": ("increase", +1),
    "create": ("create", +1), "creates": ("create", +1), "created": ("create", +1),
    "result": ("result_in", +1), "results": ("result_in", +1),
    "disrupt": ("disrupt", -1), "disrupts": ("disrupt", -1), "disrupted": ("disrupt", -1),
    "impair": ("impair", -1), "impairs": ("impair", -1), "impaired": ("impair", -1),
    "damage": ("damage", -1), "damages": ("damage", -1), "damaged": ("damage", -1),
    "affect": ("affect", +1), "affects": ("affect", +1), "affected": ("affect", +1),
    "worsen": ("worsen", -1), "worsens": ("worsen", -1), "worsened": ("worsen", -1),
    "improve": ("improve", +1), "improves": ("improve", +1), "improved": ("improve", +1),
    "enable": ("enable", +1), "enables": ("enable", +1), "enabled": ("enable", +1),
    "force": ("force", +1), "forces": ("force", +1), "forced": ("force", +1),
    "delay": ("delay", -1), "delays": ("delay", -1), "delayed": ("delay", -1),
    "reduce": ("reduce", -1), "reduces": ("reduce", -1), "reduced": ("reduce", -1),
    "lower": ("lower", -1), "lowers": ("lower", -1), "lowered": ("lower", -1),
    "prevent": ("prevent", -1), "prevents": ("prevent", -1), "prevented": ("prevent", -1),
    "suppress": ("suppress", -1), "suppresses": ("suppress", -1),
    "block": ("block", -1), "blocks": ("block", -1), "blocked": ("block", -1),
    "inhibit": ("inhibit", -1), "inhibits": ("inhibit", -1),
}

# Connectives that mean "the PREVIOUS event caused THIS clause".
FORWARD_CONNECTIVES = [
    "as a result", "consequently", "therefore", "thus", "hence",
    "this triggered", "this caused", "this led to", "which led to",
    "because of this", "so that", "resulting in", "leading to", "and so",
]
# Connectives that mean "THIS clause is caused by what FOLLOWS".
BACKWARD_CONNECTIVES = ["because", "due to", "owing to", "as a consequence of",
                        "caused by", "triggered by", "resulting from"]


@dataclass
class CausalEdge:
    cause: str
    relation: str
    effect: str
    polarity: int          # +1 promotes, -1 suppresses
    source_sent: str       # plaintext provenance for the LLM context

    def text(self) -> str:
        arrow = "==>" if self.polarity > 0 else "=/=>"
        return f"{self.cause} {arrow}[{self.relation}] {self.effect}"


# Words that must never be returned as an event anchor.
_NON_EVENTS = {
    "this", "that", "these", "those", "it", "they", "result", "results",
    "consequence", "the", "a", "an", "such", "which", "there", "here",
    "as", "so", "then", "thus", "hence", "therefore",
}
# Determiners/adjectives to skip when scanning for the head noun.
_SKIP_HEAD = {"the", "a", "an", "this", "that", "these", "those", "its",
              "their", "his", "her", "our", "your", "my", "some", "any",
              "emergency", "coolant", "power", "main", "first", "second",
              "subsequent", "resulting", "following", "entire", "whole"}


def _strip_connectives(sentence: str) -> str:
    low = sentence.lower().strip()
    for c in FORWARD_CONNECTIVES + BACKWARD_CONNECTIVES:
        if low.startswith(c):
            low = low[len(c):].lstrip(" ,")
    return low


def _event_head(sentence: str) -> Optional[str]:
    """The event/entity a clause is about: the head noun of its subject.
    Strips leading discourse connectives and skips pronouns/determiners so
    'As a result, the coolant valve failed' -> 'valve', not 'result'."""
    cleaned = _strip_connectives(sentence)

    # try the parser first, but reject non-event anchors
    trips = parse_triples(cleaned)
    if trips:
        cand = trips[0].agent or trips[0].patient
        if cand and cand not in _NON_EVENTS:
            return cand

    # fallback: first content noun-ish token after skipping determiners/adjs
    words = [_clean(w) for w in cleaned.split()]
    for w in words:
        if w and len(w) > 2 and w not in _NON_EVENTS and w not in _SKIP_HEAD:
            # stop at the verb — subject head is the last noun before it
            if w in CAUSAL_VERBS:
                break
            head = w
        else:
            continue
        # keep scanning: we want the noun closest to the verb (compound head)
    # simpler: collect candidate nouns before the first verb, take the last
    candidates = []
    for w in words:
        if w in CAUSAL_VERBS or w in ("failed", "happened", "occurred", "was", "were"):
            break
        if w and len(w) > 2 and w not in _NON_EVENTS and w not in _SKIP_HEAD:
            candidates.append(w)
    if candidates:
        return candidates[-1]
    return None


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def extract_edges(text: str) -> List[CausalEdge]:
    edges: List[CausalEdge] = []
    sents = _sentences(text)

    for idx, sent in enumerate(sents):
        low = sent.lower()

        # (A) intra-sentence causal triples
        for tr in parse_triples(sent):
            verb = tr.action.lower()
            if verb in CAUSAL_VERBS:
                rel, pol = CAUSAL_VERBS[verb]
                edges.append(CausalEdge(tr.agent, rel, tr.patient, pol, sent))

        # (B) inter-sentence chaining via forward connectives
        fwd_hit = next((c for c in FORWARD_CONNECTIVES if c in low), None)
        if fwd_hit and idx > 0:
            prev_event = _event_head(sents[idx - 1])
            this_event = _event_head(sent)
            if prev_event and this_event and prev_event != this_event:
                # polarity from any causal verb present, else default +1
                pol = +1
                rel = "leads_to"
                for v, (r, p) in CAUSAL_VERBS.items():
                    if v in low:
                        rel, pol = r, p
                        break
                edges.append(CausalEdge(prev_event, rel, this_event, pol,
                                        f"{sents[idx-1]} {sent}"))

        # (B') backward connectives inside one sentence:
        #      "Y happened because of X"  -> X causes Y
        bwd_hit = next((c for c in BACKWARD_CONNECTIVES if c in low), None)
        if bwd_hit:
            parts = re.split(re.escape(bwd_hit), low, maxsplit=1)
            if len(parts) == 2:
                effect_head = _event_head(parts[0])
                cause_head = _event_head(parts[1])
                if cause_head and effect_head and cause_head != effect_head:
                    edges.append(CausalEdge(cause_head, "lead_to", effect_head,
                                            +1, sent))

    # de-duplicate identical edges, keep first provenance
    seen = set()
    uniq = []
    for e in edges:
        key = (e.cause, e.relation, e.effect, e.polarity)
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq
