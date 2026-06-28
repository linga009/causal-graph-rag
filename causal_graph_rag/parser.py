"""
parser.py
=========
Turn raw sentences into (AGENT, ACTION, PATIENT) triples.

Two backends:
  * spaCy dependency parse (preferred) — handles passive voice, so
    "unemployment is caused by inflation" still yields AGENT=inflation.
  * A dependency-free rule fallback so the pipeline runs in any sandbox.

Both normalize the verb to its lemma-ish root and strip auxiliaries, so
active and passive phrasings of the same proposition collapse to one triple.
"""

from __future__ import annotations
from typing import List, Optional
from .vsa_core import Triple

try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = None
except Exception:
    _NLP = None


# Very small lemmatizer for the fallback path.
_VERB_NORMAL = {
    "causes": "cause", "caused": "cause", "causing": "cause",
    "increases": "increase", "increased": "increase",
    "reduces": "reduce", "reduced": "reduce", "lowers": "lower",
    "drives": "drive", "driven": "drive", "drove": "drive",
    "leads": "lead", "led": "lead", "produces": "produce", "produced": "produce",
    "triggers": "trigger", "triggered": "trigger",
}
_STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
         "do", "does", "did", "by", "of", "to", "that", "this", "it",
         "?", ".", ",", "'s"}
_PASSIVE_MARK = {"by"}


def _norm_verb(v: str) -> str:
    v = v.lower()
    return _VERB_NORMAL.get(v, v)


def _clean(tok: str) -> str:
    return tok.lower().strip(" ?.,'\"")


# --------------------------------------------------------------------------- #
#  spaCy backend
# --------------------------------------------------------------------------- #
def _spacy_triples(text: str) -> List[Triple]:
    doc = _NLP(text)
    triples: List[Triple] = []
    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            subj = obj = None
            passive = False
            for child in tok.children:
                dep = child.dep_
                if dep in ("nsubj",):
                    subj = child
                elif dep in ("nsubjpass",):
                    obj_candidate = child   # patient surfaced as subject
                    passive = True
                    obj = child
                elif dep in ("dobj", "dative", "attr", "oprd"):
                    obj = child
                elif dep == "pobj" and child.head.text.lower() == "by":
                    subj = child            # agent in passive "by X"
            # passive: nsubjpass is the patient, agent comes from "by"
            if passive:
                agent = subj
                patient = obj
            else:
                agent, patient = subj, obj
            if agent is not None and patient is not None:
                triples.append(Triple(
                    agent=_clean(agent.lemma_ if agent.lemma_ != "-PRON-" else agent.text),
                    action=_norm_verb(tok.lemma_),
                    patient=_clean(patient.lemma_ if patient.lemma_ != "-PRON-" else patient.text),
                ))
    return triples


# --------------------------------------------------------------------------- #
#  Rule-based fallback backend
# --------------------------------------------------------------------------- #
_AUX = {"does", "do", "did", "is", "are", "was", "were", "will", "can",
        "could", "would", "should", "has", "have", "had"}
_KNOWN_VERBS = set(_VERB_NORMAL.keys()) | set(_VERB_NORMAL.values())


def _fallback_triples(text: str) -> List[Triple]:
    """Handles common active/passive/interrogative forms:
        ACTIVE   : X <verb> Y               -> (X, verb, Y)
        QUESTION : Does X <verb> Y ?         -> (X, verb, Y)
        PASSIVE  : Y is <verb>ed by X        -> (X, verb, Y)
        WH-PASS  : What <verb>s X / by X      -> handled via 'by' marker
    Passive is detected by the 'by' agent marker."""
    triples: List[Triple] = []
    for clause in _split_clauses(text):
        words = [_clean(t) for t in clause.replace("?", " ").split()]
        words = [w for w in words if w]
        if not words:
            continue
        # drop a leading interrogative auxiliary ("does", "is", ...)
        if words and words[0] in _AUX:
            words = words[1:]
        # locate the main verb: prefer a known verb, else first -s/-ed token
        verb_idx = None
        for i, w in enumerate(words):
            if w in _KNOWN_VERBS:
                verb_idx = i
                break
        if verb_idx is None:
            for i, w in enumerate(words):
                if w in _STOP or w in _AUX:
                    continue
                if w.endswith(("es", "ed", "s")) and len(w) > 3:
                    verb_idx = i
                    break
        if verb_idx is None:
            continue
        verb = _norm_verb(words[verb_idx])
        left = [w for w in words[:verb_idx] if w not in _STOP and w not in _AUX]
        right_raw = words[verb_idx + 1:]
        passive = "by" in right_raw
        right = [w for w in right_raw if w not in _STOP and w not in _AUX]
        if not left or not right:
            continue
        # Capture up to two content words so compound nouns are preserved
        # ("coolant valve" instead of just "valve")
        left_head = " ".join(left[-2:])
        right_head = " ".join(right[-2:])
        if passive:
            agent, patient = right_head, left_head
        else:
            agent, patient = left_head, right_head
        triples.append(Triple(agent=agent, action=verb, patient=patient))
    return triples


def _split_clauses(text: str) -> List[str]:
    out = []
    for part in text.replace(";", ".").split("."):
        part = part.strip()
        if part:
            out.append(part)
    return out


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def parse_triples(text: str) -> List[Triple]:
    if _NLP is not None:
        t = _spacy_triples(text)
        if t:
            return t
    return _fallback_triples(text)


def backend_name() -> str:
    return "spaCy:en_core_web_sm" if _NLP is not None else "rule-based-fallback"
