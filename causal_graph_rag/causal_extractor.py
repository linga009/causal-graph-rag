"""
causal_extractor.py
===================
Extract DIRECTED causal/consequential edges from text.

Two backends for intra-sentence extraction:
  * spaCy  (preferred) — dependency-parse subject/object/passive properly,
    handles compound nouns, passive voice ("Y was triggered by X"), and
    xsubj relations. Requires: pip install spacy && python -m spacy download en_core_web_sm
  * Rule   (fallback)  — the original SVO + CAUSAL_VERBS approach; works
    offline with no extra packages.

Two sources of edges:
  (A) INTRA-SENTENCE   "X causes Y", "Y is triggered by X", "X prevents Y"
  (B) INTER-SENTENCE   Discourse connectives ("as a result", "consequently",
                       "because", "due to") linking events across sentences.

Output: List[CausalEdge(cause, relation, effect, polarity, source_sent)].
Polarity: +1 promotes/produces, -1 prevents/reduces.
"""

from __future__ import annotations
import json
import logging
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .parser import parse_triples, _clean, _split_clauses
from .vsa_core import Triple

log = logging.getLogger("causal_rag")


# --- verb -> (canonical relation, polarity) -------------------------------- #
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
    # Purpose / goal verbs (common in policy, programme, and research documents)
    "develop": ("develop", +1), "develops": ("develop", +1), "developed": ("develop", +1),
    "advance": ("advance", +1), "advances": ("advance", +1), "advanced": ("advance", +1),
    "investigate": ("investigate", +1), "investigates": ("investigate", +1),
    "address": ("address", +1), "addresses": ("address", +1), "addressed": ("address", +1),
    "equip": ("equip", +1), "equips": ("equip", +1), "equipped": ("equip", +1),
    "benefit": ("benefit", +1), "benefits": ("benefit", +1),
    "enhance": ("enhance", +1), "enhances": ("enhance", +1), "enhanced": ("enhance", +1),
    "predict": ("predict", +1), "predicts": ("predict", +1), "predicted": ("predict", +1),
    "integrate": ("integrate", +1), "integrates": ("integrate", +1), "integrated": ("integrate", +1),
    "inform": ("inform", +1), "informs": ("inform", +1), "informed": ("inform", +1),
    "transform": ("transform", +1), "transforms": ("transform", +1), "transformed": ("transform", +1),
    "empower": ("empower", +1), "empowers": ("empower", +1), "empowered": ("empower", +1),
}

# Connectives that mean "the PREVIOUS event caused THIS clause".
FORWARD_CONNECTIVES = [
    "as a result", "consequently", "therefore", "thus", "hence",
    "this triggered", "this caused", "this led to", "which led to",
    "because of this", "so that", "resulting in", "leading to", "and so",
    # temporal sequencing — strong enough signal to treat as causal
    "subsequently", "afterwards", "whereupon", "following this",
    "after which", "shortly after", "at that point", "after this",
]
# Connectives that mean "THIS clause is caused by what FOLLOWS".
BACKWARD_CONNECTIVES = ["because", "due to", "owing to", "as a consequence of",
                        "caused by", "triggered by", "resulting from"]

# State-change verbs: when sentence N+1 contains one of these and has no
# explicit connective, consecutive-sentence adjacency is treated as implicit
# causation from sentence N.
STATE_CHANGE_VERBS = {
    "failed", "fail", "fails", "failing",
    "broke", "break", "breaks", "broken", "breaking",
    "stopped", "stop", "stops", "stopping",
    "collapsed", "collapse", "collapses", "collapsing",
    "crashed", "crash", "crashes", "crashing",
    "died", "die", "dies", "dying",
    "halted", "halt", "halts", "halting",
    "fell", "fall", "falls", "falling",
    "rose", "rise", "rises", "rising",
    "dropped", "drop", "drops", "dropping",
    "surged", "surge", "surges", "surging",
    "spiked", "spike", "spikes", "spiking",
    "skidded", "skid", "skids", "skidding",
    "exploded", "explode", "explodes", "exploding",
    "ruptured", "rupture", "ruptures", "rupturing",
    "leaked", "leak", "leaks", "leaking",
    "overflowed", "overflow", "overflows",
    "froze", "freeze", "freezes", "frozen",
    "shut", "shuts", "shutting",
    "started", "start", "starts", "starting",
    "began", "begin", "begins",
    "went", "goes",
    "became", "become", "becomes", "becoming",
    "turned", "turn", "turns",
    "triggered", "trigger", "triggers",
    "activated", "activate", "activates",
    "initiated", "initiate", "initiates",
    "occurred", "occur", "occurs",
    "happened", "happen", "happens",
    "emerged", "emerge", "emerges",
    "appeared", "appear", "appears",
}


@dataclass
class CausalEdge:
    cause: str
    relation: str
    effect: str
    polarity: int          # +1 promotes, -1 suppresses
    source_sent: str       # plaintext provenance for the LLM context
    confidence: float = 0.85     # extraction confidence in [0, 1]
    extraction_method: str = "spacy"  # "spacy" | "rule" | "llm" | "rebel" | "implicit"
    # "stated" (asserted as fact) | "hypothesis" (hedged: "it has been
    # speculated...") | "disputed" (>=2 edges to the same effect disagree on
    # cause and at least one is a hypothesis). Set by _annotate_edges(), a
    # post-process pass -- not by individual extraction methods, so it's
    # consistent regardless of which of the four methods produced the edge.
    epistemic_status: str = "stated"
    attributed_to: Optional[str] = None  # named source of a claim, if detected
    magnitude: Optional[str] = None      # nearby numeric/%/currency span, if any

    def text(self) -> str:
        arrow = "==>" if self.polarity > 0 else "=/=>"
        out = f"{self.cause} {arrow}[{self.relation}] {self.effect}"
        if self.magnitude:
            out += f" ({self.magnitude})"
        if self.epistemic_status != "stated":
            out += f" [{self.epistemic_status}"
            out += f", per {self.attributed_to}]" if self.attributed_to else "]"
        return out


# Words that must never be returned as an event anchor.
_NON_EVENTS = {
    "this", "that", "these", "those", "it", "they", "result", "results",
    "consequence", "the", "a", "an", "such", "which", "there", "here",
    "as", "so", "then", "thus", "hence", "therefore",
}
_SKIP_HEAD = {"the", "a", "an", "this", "that", "these", "those", "its",
              "their", "his", "her", "our", "your", "my", "some", "any",
              "emergency", "coolant", "power", "main", "first", "second",
              "subsequent", "resulting", "following", "entire", "whole"}

# Pronoun words that must never appear as cause/effect nodes. Includes
# relative pronouns (which/who/whom/whose) -- omitting them let edges like
# ("faa", implicit_trigger, "who") and ("which", cause, "u.s.") through.
_PRONOUN_NODES = frozenset(
    "it this that these those they he she we i you them him her us "
    "its their his her our your my itself themselves ourselves "
    "which who whom whose".split()
)

# Words in an effect clause that signal the edge is suppressive (polarity -1).
_SUPPRESS_WORDS = frozenset(
    "not no never fail fails failed loss lose loses reduce reduces reduced "
    "prevent prevents prevented block blocks blocked inhibit inhibits "
    "decrease decreases decreased damage damages damaged deteriorate "
    "worsen worsens worsened collapse collapses collapsed crash crashes crashed "
    "disrupt disrupts disrupted impair impairs impaired degrade degrades degraded".split()
)

# Temporal / discourse scope markers that imply causality (weaker signal than
# explicit connectives, but reliable when combined with state-change verbs).
TEMPORAL_CONNECTIVES = [
    "as a result of", "in response to", "due to the",
    "following the", "after the", "prior to the",
    "during the", "in the wake of", "upon the",
    "once the", "when the", "as the",
]


_BARE_NUMERIC_RE = re.compile(r"^[\d\s.,%$€£]+$")

# Function words that don't count as a "real content word" on their own.
# Deliberately NOT using spaCy POS tagging here: tagging a 1-3 word span in
# isolation (no sentence context) is unreliable for common noun/verb
# homographs in this domain ("pump", "scram", "outage" all tag as VERB when
# handed to the tagger alone) -- that isolation-reparse bug briefly broke
# real edges like ("pump", cause, "reactor") during development. Real POS
# filtering happens where context is available: in _intra_spacy's in-context
# parse and _event_head's noun-chunk pass, both of which parse the whole
# sentence, never a bare extracted string.
_FRAGMENT_STOPWORDS = _NON_EVENTS | _SKIP_HEAD | _PRONOUN_NODES | frozenset(
    "is are was were be been being do does did by of to in on at for with "
    "however though meanwhile therefore moreover furthermore nonetheless "
    "otherwise at first after before".split()
)


def _is_well_formed_entity(text: str) -> bool:
    """Return True iff `text` looks like a real entity span rather than an
    extraction artifact: a bare number/percent/currency span, a dangling
    fragment with an unmatched parenthesis (truncation smell), or a span
    with no real content word at all (catches stray prepositions/function
    words/discourse markers that a raw word-scan can pick up).

    This is a single chokepoint deliberately kept independent of which of
    the four extraction methods (spacy/rule/implicit/org_rule) produced the
    span, since all of them can independently produce malformed spans. It
    is a coarse safety net, not a full grammaticality check -- it will not
    catch every fragment (e.g. "however investment" passes, since
    "investment" is a real content word), by design: see the isolation
    caveat above for why anything stronger needs sentence context.
    """
    t = text.strip()
    if not t:
        return False
    if _BARE_NUMERIC_RE.match(t):
        return False
    if _MAGNITUDE_RE.fullmatch(t):
        # A pure magnitude expression ("$4 trillion", "124%") is not a noun
        # phrase entity -- that information belongs in CausalEdge.magnitude
        # (see _detect_magnitude), not as a cause/effect node.
        return False
    if t.count("(") != t.count(")"):
        return False
    words = [w.strip(".,;:!?\"'()") for w in t.split()]
    words = [w for w in words if w]
    if not words:
        return False
    if not any(w.isalpha() and len(w) > 2 and w.lower() not in _FRAGMENT_STOPWORDS
               for w in words):
        return False
    return True


def _validate_edge(e: CausalEdge) -> bool:
    """Return True iff the edge is well-formed and should enter the graph.
    Rejects pronouns, empty strings, self-loops, overly long entity names,
    and (via _is_well_formed_entity) non-noun-phrase extraction artifacts."""
    c, eff = e.cause.strip(), e.effect.strip()
    if not c or not eff or not e.relation:
        return False
    if c in _PRONOUN_NODES or eff in _PRONOUN_NODES:
        return False
    if len(c) > 80 or len(eff) > 80:
        return False
    if c == eff:
        return False
    if not _is_well_formed_entity(c) or not _is_well_formed_entity(eff):
        return False
    return True


def _infer_polarity(effect_sentence: str) -> int:
    """Infer polarity from the effect clause. Returns -1 if suppression words
    are present, +1 otherwise. Used for implicit edges where polarity is unknown.
    Strips trailing punctuation so 'reduced.' / 'blocked,' still match."""
    words = {w.strip(".,;:!?\"'()") for w in effect_sentence.lower().split()}
    return -1 if words & _SUPPRESS_WORDS else +1


# --------------------------------------------------------------------------- #
#  spaCy-based intra-sentence extraction
# --------------------------------------------------------------------------- #

_spacy_nlp = None  # lazy singleton


def _get_nlp():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
        return _spacy_nlp
    except (ImportError, OSError):
        return None


def _compound_span(token) -> str:
    """Return '<compound modifiers> <head>' as a lowercase string."""
    parts = [c.text for c in token.lefts if c.dep_ == "compound"]
    parts.append(token.text)
    return " ".join(parts).lower().strip()


_PRONOUNS = {"this", "that", "these", "those", "it", "they", "he", "she", "we", "i",
             "which", "who", "whom", "whose"}


# --------------------------------------------------------------------------- #
#  Coreference resolution (pronouns -> antecedents)
# --------------------------------------------------------------------------- #

def _resolve_coreferences(text: str) -> str:
    """
    Resolve pronouns to their antecedents in text. This prevents pronouns from
    becoming ghost nodes in the causal graph.

    Uses spaCy neuralcoref if available (pip install neuralcoref); otherwise
    falls back to a simple heuristic (sentence-level antecedent matching).
    """
    nlp = _get_nlp()
    if nlp is None:
        return text

    try:
        # Try to use neuralcoref if installed
        import spacy_experimental
        spacy_experimental.component.set_extension("is_noun_phrase")
        # Neuralcoref is deprecated in recent spaCy versions; try the fallback instead
    except (ImportError, AttributeError):
        pass

    doc = nlp(text)

    # Heuristic coreference: for each pronoun, find the nearest preceding noun phrase
    # This is simple but works well for incident narratives
    replacements = {}  # (start, end) -> replacement_text
    noun_phrases = []  # (start, end, text) of recent NPs

    for token in doc:
        # Track noun phrases
        if token.pos_ in ("NOUN", "PROPN") and token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj"):
            # Get the compound span (e.g., "emergency shutdown" not just "shutdown")
            phrase_tokens = [t for t in token.subtree if t.pos_ in ("NOUN", "PROPN", "ADJ")]
            if phrase_tokens:
                phrase = " ".join(t.text for t in phrase_tokens).lower()
                noun_phrases.append((token.idx, token.idx + len(token.text), phrase, token))

        # Resolve pronouns
        if token.pos_ == "PRON" and token.lower_ in _PRONOUNS:
            # Find the most recent noun phrase (within last 100 tokens)
            candidates = [np for np in noun_phrases if np[0] < token.idx]
            if candidates:
                # Prefer recent and prominent NPs
                recent = candidates[-1]
                phrase_text = recent[2]
                # Replace the pronoun with the noun phrase
                replacements[(token.idx, token.idx + len(token.text))] = phrase_text

    # Apply replacements (in reverse order to preserve indices)
    result = text
    for (start, end), replacement in sorted(replacements.items(), reverse=True):
        result = result[:start] + replacement + result[end:]

    return result


def _intra_spacy(sent: str) -> Optional[List[CausalEdge]]:
    """Extract causal edges using spaCy dependency parse.
    Returns None if spaCy or the model is unavailable (triggers rule fallback).

    Handles three patterns:
      (1) Active:  nsubj -> VERB -> dobj/pobj
      (2) Passive: nsubjpass <- VERB (by-agent -> pobj)
      (3) Participial amod: "The outage disrupted operations"
          where small model tags the verb as amod of the dobj with
          an npadvmod for the subject-like argument.
    """
    nlp = _get_nlp()
    if nlp is None:
        return None

    edges: List[CausalEdge] = []
    doc = nlp(sent)

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in CAUSAL_VERBS:
            continue
        rel, pol = CAUSAL_VERBS[lemma]

        # Pattern 3: participial amod — small model misparses active sentences
        # e.g. "The power outage disrupted hospital operations."
        #   ROOT=operations, disrupted=amod(operations), outage=npadvmod(disrupted)
        # The nominal agent attaches to the verb token, not to the head noun.
        if token.dep_ == "amod" and token.head.pos_ in ("NOUN", "PROPN"):
            effect_txt = _compound_span(token.head)
            for child in token.children:
                if child.dep_ in ("npadvmod", "nsubj") and child.pos_ in ("NOUN", "PROPN"):
                    cause_txt = _compound_span(child)
                    if cause_txt and effect_txt and cause_txt != effect_txt:
                        edges.append(CausalEdge(cause_txt, rel, effect_txt, pol, sent,
                                                confidence=0.85, extraction_method="spacy"))
            continue  # don't also try patterns 1/2 for the same token

        # Find syntactic subject — filter pronouns that coreference resolution
        # would need to resolve (they add noise as standalone graph nodes),
        # and require a NOUN/PROPN head so numbers/symbols ("%", bare years)
        # carrying a subject/object dep label don't become graph nodes.
        subjects = [c for c in token.children
                    if c.dep_ in ("nsubj", "nsubjpass")
                    and c.lower_ not in _PRONOUNS
                    and c.pos_ in ("NOUN", "PROPN")]
        # Objects: direct object OR prepositional object ("led to X")
        objects = [c for c in token.children
                   if c.dep_ in ("dobj", "attr") and c.pos_ in ("NOUN", "PROPN")]
        if not objects:
            for prep in (c for c in token.children if c.dep_ == "prep"):
                objects += [gc for gc in prep.children
                           if gc.dep_ == "pobj" and gc.pos_ in ("NOUN", "PROPN")]

        is_passive = any(c.dep_ == "nsubjpass" for c in token.children)

        if is_passive:
            # Pattern 2: "Y was triggered by X" -> cause=X, effect=Y
            effect_tokens = subjects
            cause_tokens = []
            for c in token.children:
                if c.dep_ == "agent":
                    cause_tokens += [gc for gc in c.children
                                     if gc.dep_ == "pobj" and gc.pos_ in ("NOUN", "PROPN")]
            if cause_tokens:
                for eff in effect_tokens:
                    for cau in cause_tokens:
                        e_txt = _compound_span(eff)
                        c_txt = _compound_span(cau)
                        if e_txt and c_txt and e_txt != c_txt:
                            edges.append(CausalEdge(c_txt, rel, e_txt, pol, sent,
                                                    confidence=0.85, extraction_method="spacy"))
        else:
            # Pattern 1: "X caused Y"
            for subj in subjects:
                for obj in objects:
                    s_txt = _compound_span(subj)
                    o_txt = _compound_span(obj)
                    if s_txt and o_txt and s_txt != o_txt:
                        edges.append(CausalEdge(s_txt, rel, o_txt, pol, sent,
                                                confidence=0.85, extraction_method="spacy"))

    return edges


# --------------------------------------------------------------------------- #
#  Rule-based intra-sentence extraction (original approach, always available)
# --------------------------------------------------------------------------- #

def _intra_rule(sent: str) -> List[CausalEdge]:
    edges = []
    for tr in parse_triples(sent):
        verb = tr.action.lower()
        if verb in CAUSAL_VERBS and tr.agent.lower() not in _PRONOUNS:
            rel, pol = CAUSAL_VERBS[verb]
            edges.append(CausalEdge(tr.agent, rel, tr.patient, pol, sent,
                                    confidence=0.65, extraction_method="rule"))
    return edges


def _intra_edges(sent: str) -> List[CausalEdge]:
    """Try spaCy first; fall back to rule-based when spaCy finds nothing.
    This catches small-model parser errors (e.g. passive participials tagged amod)
    without losing the compound-noun benefit when spaCy parses correctly."""
    result = _intra_spacy(sent)
    if result:          # spaCy available AND found at least one edge
        return result
    return _intra_rule(sent)


# --------------------------------------------------------------------------- #
#  Inter-sentence chaining (connective-based, backend-agnostic)
# --------------------------------------------------------------------------- #

def _strip_connectives(sentence: str) -> str:
    low = sentence.lower().strip()
    for c in FORWARD_CONNECTIVES + BACKWARD_CONNECTIVES:
        if low.startswith(c):
            low = low[len(c):].lstrip(" ,")
    return low


def _event_head(sentence: str) -> Optional[str]:
    """The event/entity a clause is about: head noun of its subject.

    Three tiers, tried in order:
      1. parse_triples()'s agent/patient (cheap, already dependency-aware).
      2. spaCy noun-chunk nearest the cutoff point (a causal verb or a
         state-change word like "was"/"failed") -- a real NP, not just
         "whatever words happened to precede the cutoff".
      3. Raw word-scan (only reached when spaCy is unavailable). This tier
         is the one that historically produced fragments like "via mbs)" or
         "who", since it has no notion of grammatical structure at all;
         _validate_edge's _is_well_formed_entity check is the backstop for
         whatever it still lets through.
    """
    cleaned = _strip_connectives(sentence)

    trips = parse_triples(cleaned)
    if trips:
        cand = trips[0].agent or trips[0].patient
        if cand and cand not in _NON_EVENTS and _is_well_formed_entity(cand):
            return cand

    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(cleaned)
        cutoff = len(doc)
        for tok in doc:
            if tok.lemma_.lower() in CAUSAL_VERBS or tok.lower_ in (
                "failed", "happened", "occurred", "was", "were"):
                cutoff = tok.i
                break
        best = None
        for chunk in doc.noun_chunks:
            if chunk.end <= cutoff:
                best = chunk
        if best is not None:
            head = _compound_span(best.root)
            if head and head not in _NON_EVENTS and _is_well_formed_entity(head):
                return head

    words = [_clean(w) for w in cleaned.split()]
    candidates = []
    for w in words:
        if w in CAUSAL_VERBS or w in ("failed", "happened", "occurred", "was", "were"):
            break
        if w and len(w) > 2 and w not in _NON_EVENTS and w not in _SKIP_HEAD:
            candidates.append(w)
    if not candidates:
        return None
    # Return up to two content words so compound nouns are preserved
    # (e.g. "coolant valve" instead of just "valve")
    head = " ".join(candidates[-2:]) if len(candidates) >= 2 else candidates[-1]
    return head if _is_well_formed_entity(head) else None


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# --------------------------------------------------------------------------- #
#  Implicit causation: adjacency + state-change heuristic
# --------------------------------------------------------------------------- #

def _implicit_edges(sents: List[str], explicit_pairs: set) -> List[CausalEdge]:
    """Create weak implicit_trigger edges for sentences where:
      - sentence N+1 (or N+2) contains a state-change verb
      - no explicit causal or backward connective is present
      - the (cause_head, effect_head) pair is not already captured explicitly

    Also detects temporal scope markers ("during the ...", "following the ...")
    as implicit causation signals.

    Polarity is inferred from suppression words in the effect sentence rather
    than hardcoded to +1.
    """
    edges: List[CausalEdge] = []
    seen_pairs: set = set()

    for i in range(1, len(sents)):
        s_curr = sents[i]
        low = s_curr.lower()

        if any(c in low for c in FORWARD_CONNECTIVES + BACKWARD_CONNECTIVES):
            continue

        # Check for state-change verbs OR temporal connectives
        words = [w.rstrip(".,;:!?") for w in low.split()]
        has_state_change = any(w in STATE_CHANGE_VERBS for w in words)
        has_temporal = any(tc in low for tc in TEMPORAL_CONNECTIVES)
        if not has_state_change and not has_temporal:
            continue

        # Look back up to 4 sentences; confidence decays with distance.
        for lookback, conf in ((1, 0.50), (2, 0.50), (3, 0.45), (4, 0.40)):
            j = i - lookback
            if j < 0:
                continue
            s_prev = sents[j]
            prev_head = _event_head(s_prev)
            curr_head = _event_head(s_curr)

            if not prev_head or not curr_head or prev_head == curr_head:
                continue
            if (prev_head, curr_head) in explicit_pairs:
                continue
            if (prev_head, curr_head) in seen_pairs:
                continue

            pol = _infer_polarity(s_curr)
            seen_pairs.add((prev_head, curr_head))
            edges.append(CausalEdge(
                prev_head, "implicit_trigger", curr_head, pol,
                f"{s_prev} {s_curr}",
                confidence=conf, extraction_method="implicit",
            ))
            break  # prefer nearest cause; don't add both lookbacks

    return edges


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

# Organizational causality: captures governance and information-flow failures
# that spaCy/rules miss ("proceeded without oversight", "failed to notify", etc.)
# NOTE: capture groups are bounded by WORD count (\w+(?:\s+\w+){0,N}), not
# character count. A character-count cap ([\w\s]{1,30}) slices mid-word
# whenever a phrase's word boundaries don't happen to land on the count
# (e.g. "...future value" -> "...future valu"); a word-count cap can't ever
# do that, since \w+ only ever matches whole words. Groups are GREEDY (no
# trailing lookahead) and capped generously (10-12 words): an earlier
# non-greedy-plus-lookahead version could fail to match at all whenever a
# real phrase ran longer than its cap with no punctuation inside it (e.g.
# "properly assess the future value of these financial assets" -- 9 words,
# no internal punctuation) -- greedy matching just consumes up to the cap
# instead of requiring a nearby boundary that might not exist.
_ORG_WITHOUT = re.compile(
    r"(\w+(?:\s+\w+){0,7})\s+without\s+"
    r"(?:proper\s+|required\s+|adequate\s+|formal\s+|explicit\s+|prior\s+)?"
    r"(approval|authorization|oversight|notification|clearance|consent|review|supervision|knowledge)",
    re.I,
)
_ORG_FAILED = re.compile(
    r"(\w+(?:\s+\w+){0,5})\s+failed\s+to\s+(\w+(?:\s+\w+){0,11})",
    re.I,
)
_ORG_UNAWARE = re.compile(
    r"(\w+(?:\s+\w+){0,5})\s+(?:was|were)\s+(?:not\s+)?(?:unaware|informed|notified|told)\s+"
    r"(?:of\s+|about\s+)(\w+(?:\s+\w+){0,11})",
    re.I,
)
_ORG_DESPITE = re.compile(
    r"(\w+(?:\s+\w+){0,5})\s+(?:proceeded|continued|persisted|went\s+ahead|was\s+conducted|was\s+carried\s+out)\s+"
    r"despite\s+(\w+(?:\s+\w+){0,11})",
    re.I,
)
_ORG_PRESSURE = re.compile(
    r"(?:under\s+)?(?:\w+\s+){0,2}\bpressure\b\s+to\s+(\w+(?:\s+\w+){0,11})",
    re.I,
)


def _org_edges(sents: List[str]) -> List[CausalEdge]:
    """Extract organizational/procedural causality missed by dependency parsing:
    governance gaps, information failures, override of safety protocols."""
    edges: List[CausalEdge] = []

    for sent in sents:
        # "X proceeded/was done without [required] oversight/approval"
        # -> (lack of oversight) enabled X
        for m in _ORG_WITHOUT.finditer(sent):
            action = m.group(1).strip()
            prereq = m.group(2).strip().lower()
            cause_node = f"lack of {prereq}"
            if action not in _PRONOUN_NODES and len(action) > 3:
                edges.append(CausalEdge(
                    cause_node, "enabled", action, -1, sent,
                    confidence=0.60, extraction_method="org_rule",
                ))

        # "[actor] failed to [verb+object]"
        # -> (actor, failure_to, action)
        m = _ORG_FAILED.search(sent)
        if m:
            actor = m.group(1).strip().lower()
            action = m.group(2).strip().lower().rstrip(".,:;")
            if actor not in _PRONOUN_NODES and len(actor) > 2 and len(action) > 2:
                edges.append(CausalEdge(
                    actor, "failed_to", action, -1, sent,
                    confidence=0.60, extraction_method="org_rule",
                ))

        # "[X] was unaware of / not informed about [Y]"
        # -> (lack of Y awareness, affected, X)
        m = _ORG_UNAWARE.search(sent)
        if m:
            actor = m.group(1).strip().lower()
            thing = m.group(2).strip().lower().rstrip(".,:;")
            if actor not in _PRONOUN_NODES and len(actor) > 2 and len(thing) > 2:
                edges.append(CausalEdge(
                    f"lack of {thing} awareness", "affected", actor, -1, sent,
                    confidence=0.55, extraction_method="org_rule",
                ))

        # "[X] proceeded/continued despite [concern/warning]"
        # -> (concern, led_to, X risk)
        m = _ORG_DESPITE.search(sent)
        if m:
            actor = m.group(1).strip().lower()
            obstacle = m.group(2).strip().lower().rstrip(".,:;")
            if actor not in _PRONOUN_NODES and len(actor) > 2 and len(obstacle) > 3:
                edges.append(CausalEdge(
                    obstacle, "ignored_by", actor, -1, sent,
                    confidence=0.55, extraction_method="org_rule",
                ))

        # "under pressure to [action]"
        # -> (pressure, led_to, action)
        m = _ORG_PRESSURE.search(sent)
        if m:
            action = m.group(1).strip().lower().rstrip(".,:;")
            if len(action) > 3:
                edges.append(CausalEdge(
                    "schedule pressure", "led_to", action, -1, sent,
                    confidence=0.50, extraction_method="org_rule",
                ))

    return [e for e in edges if _validate_edge(e)]


# --------------------------------------------------------------------------- #
#  Epistemic status, attribution, and magnitude annotation (post-process)
# --------------------------------------------------------------------------- #

_HYPOTHESIS_CUES = (
    "it has been speculated", "it was speculated", "it is speculated",
    "one view was", "one view is", "one hypothesis", "another hypothesis",
    "another view", "some believe", "some argue", "it is believed",
    "it was believed", "has been suggested", "was suggested",
    "is thought", "was thought", "is thought to", "was thought to",
    "may have been", "might have been", "could have been",
)

# Named-source attribution. Kept deliberately narrow (two high-precision
# surface patterns) rather than general NER, since a wrong attribution is
# worse than none: it would misquote who made a claim.
_ATTRIBUTION_PATTERNS = (
    # (?i:...) scopes case-insensitivity to just the literal trigger phrase,
    # so the [A-Z] name requirement (the precision guard) stays case-sensitive.
    re.compile(r"\b(?i:according to) ((?:[A-Z][\w.\-]*\s*){1,4})"),
    re.compile(r",\s*(?i:by)\s+((?:[A-Z][\w.\-]*\s*){1,4}),"),
    re.compile(r"\b((?:[A-Z][\w.\-]*\s*){1,4})\s+"
               r"(?i:hypothesi[sz]ed|argued|claimed|suggested|speculated|theorized|posited)\s+that\b"),
)

_MAGNITUDE_RE = re.compile(
    r"(\$\s?[\d,]+(?:\.\d+)?\s?(?:trillion|billion|million|thousand)?|"
    r"[\d,]+(?:\.\d+)?\s?%|"
    r"[\d,]+(?:\.\d+)?\s?(?:percent|percentage\s+points?))",
    re.I,
)


def _detect_epistemic_status(sentence: str) -> str:
    low = sentence.lower()
    return "hypothesis" if any(cue in low for cue in _HYPOTHESIS_CUES) else "stated"


def _detect_attribution(sentence: str) -> Optional[str]:
    for pat in _ATTRIBUTION_PATTERNS:
        m = pat.search(sentence)
        if m:
            name = m.group(1).strip().rstrip(",")
            if name:
                return name
    return None


def _detect_magnitude(sentence: str) -> Optional[str]:
    m = _MAGNITUDE_RE.search(sentence)
    return m.group(1).strip() if m else None


def _annotate_edges(edges: List[CausalEdge]) -> None:
    """Mutate edges in place: epistemic_status/attributed_to/magnitude from
    the edge's own source sentence, then mark 'disputed' any group of edges
    that share an effect but disagree on cause where at least one is a
    hypothesis (e.g. two competing published explanations for one event —
    see the Chernobyl second-explosion case that motivated this)."""
    for e in edges:
        e.epistemic_status = _detect_epistemic_status(e.source_sent)
        e.attributed_to = _detect_attribution(e.source_sent)
        e.magnitude = _detect_magnitude(e.source_sent)

    by_effect: Dict[str, List[CausalEdge]] = defaultdict(list)
    for e in edges:
        by_effect[e.effect].append(e)
    for group in by_effect.values():
        causes = {g.cause for g in group}
        if len(causes) > 1 and any(g.epistemic_status == "hypothesis" for g in group):
            for g in group:
                g.epistemic_status = "disputed"


# Schemas whose prose is method/definition-oriented rather than narrated
# real-world events. The narrative-specific heuristics below (implicit
# adjacency+state-change chaining, org-governance patterns) are tuned on
# incident/general text and misfire on this kind of prose -- e.g. treating
# "sentence N+1 mentions a state-change verb" as implicit causation when
# sentence N+1 is actually describing what a statistical method does, not
# narrating a real event. Note this does NOT fix every research-schema
# extraction issue: intra-sentence CAUSAL_VERBS matching (_intra_edges) can
# still misfire on definitional sentences ("Pearl's causality provides a
# definition for...") -- that needs distinguishing generic/definitional
# statements from narrated events, which this schema gate does not attempt.
_NARRATIVE_HEURISTIC_SCHEMAS = frozenset({"general", "incident", "auto"})


def extract_edges(text: str, resolve_coreferences: bool = True,
                   schema: str = "general") -> List[CausalEdge]:
    """
    Extract causal edges from text.

    Parameters
    ----------
    text : str
        Input text to extract causal edges from.
    resolve_coreferences : bool (default True)
        If True, resolve pronouns to antecedents before extraction.
        This prevents pronouns from becoming ghost nodes in the graph.
    schema : str (default "general")
        Document-structure preset, same vocabulary as GraphRAG.ingest()'s
        `schema` param ("general"/"research"/"clinical"/"incident"/"auto").
        For "research"/"clinical" schemas, the narrative-tuned implicit
        adjacency+state-change chaining and org-governance heuristics are
        skipped (see _NARRATIVE_HEURISTIC_SCHEMAS docstring above) since
        they're tuned for incident narratives, not method/definition prose.
    """
    # Optionally resolve coreferences (pronouns -> antecedents)
    if resolve_coreferences:
        text = _resolve_coreferences(text)

    use_narrative_heuristics = schema in _NARRATIVE_HEURISTIC_SCHEMAS

    edges: List[CausalEdge] = []
    sents = _sentences(text)

    for idx, sent in enumerate(sents):
        low = sent.lower()

        # (A) intra-sentence causal triples
        edges.extend(_intra_edges(sent))

        # (B) inter-sentence chaining via forward connectives
        fwd_hit = next((c for c in FORWARD_CONNECTIVES if c in low), None)
        if fwd_hit and idx > 0:
            prev_event = _event_head(sents[idx - 1])
            this_event = _event_head(sent)
            if prev_event and this_event and prev_event != this_event:
                pol = +1
                rel = "leads_to"
                for v, (r, p) in CAUSAL_VERBS.items():
                    if v in low:
                        rel, pol = r, p
                        break
                edges.append(CausalEdge(prev_event, rel, this_event, pol,
                                        f"{sents[idx-1]} {sent}",
                                        confidence=0.80, extraction_method="rule"))

        # (B') backward connectives inside one sentence: "Y happened because of X"
        bwd_hit = next((c for c in BACKWARD_CONNECTIVES if c in low), None)
        if bwd_hit:
            parts = re.split(re.escape(bwd_hit), low, maxsplit=1)
            if len(parts) == 2:
                effect_head = _event_head(parts[0])
                cause_head = _event_head(parts[1])
                if cause_head and effect_head and cause_head != effect_head:
                    edges.append(CausalEdge(cause_head, "lead_to", effect_head,
                                            +1, sent,
                                            confidence=0.80, extraction_method="rule"))

    if use_narrative_heuristics:
        # Implicit causation pass: adjacency + state-change + temporal heuristics
        explicit_pairs = {(e.cause, e.effect) for e in edges}
        edges.extend(_implicit_edges(sents, explicit_pairs))

        # Organizational causality: governance gaps, information failures,
        # protocol overrides — patterns that dependency parsing misses.
        org = _org_edges(sents)
        org_pairs = {(e.cause, e.effect) for e in edges}
        edges.extend(e for e in org if (e.cause, e.effect) not in org_pairs)

    # Validate all edges (remove pronouns, empty strings, self-loops, >80 char names)
    edges = [e for e in edges if _validate_edge(e)]

    # de-duplicate identical edges
    seen = set()
    uniq = []
    for e in edges:
        key = (e.cause, e.relation, e.effect, e.polarity)
        if key not in seen:
            seen.add(key)
            uniq.append(e)

    _annotate_edges(uniq)
    return uniq


# --------------------------------------------------------------------------- #
#  LLM-assisted causal extraction  (borrowed from CausalRAG approach)
# --------------------------------------------------------------------------- #

class LLMEdgeExtractor:
    """
    Uses an LLM to extract causal edges from text, complementing the spaCy
    extractor on sentences with implicit, metaphorical, or academic causality
    that dependency parsing misses.

    Borrowed from the CausalRAG paper (ACL 2025) idea of LLM-as-graph-builder,
    but applied only to sentences where the spaCy/rule extractor found nothing
    (augment mode) or to all sentences (full mode).

    The LLM is prompted to return a strict JSON array so the output is
    machine-parseable without a second parsing call.

    Parameters
    ----------
    llm : any object with a .generate(prompt: str) -> str method
        Works with MockLLM, GroqLLM, AnthropicLLM, or LangChainLLMAdapter.
    mode : "augment" | "full"
        "augment" (default) — LLM only processes sentences where spaCy/rules
        found no edges.  Cheapest option: 0 LLM calls on well-parsed text.
        "full" — LLM processes every sentence regardless.  Same cost profile
        as CausalRAG, highest recall.
    """

    _PROMPT = textwrap.dedent("""\
    Extract every CAUSAL relationship from the text below.
    Return ONLY a valid JSON array, no explanation, no markdown.

    Rules:
    - Each item: {{"cause": "...", "relation": "...", "effect": "..."}}
    - cause / effect  : short noun phrases, 1-5 words, lowercase
    - relation        : single causal verb in base form (e.g. caused, triggered,
                        led_to, reduced, increased, enabled, disrupted)
    - Include explicit AND strongly implied causal links
    - Omit pronouns (it, this, they) as cause or effect
    - If no causal links exist, return []

    Text:
    {text}

    JSON:""")

    def __init__(self, llm: Any, mode: str = "augment") -> None:
        self.llm = llm
        self.mode = mode  # "augment" | "full"

    def _parse_response(self, raw: str, source_sent: str) -> List[CausalEdge]:
        """Parse LLM JSON output into CausalEdge objects."""
        try:
            # Strip markdown fences the LLM might add
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            # Find the JSON array
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if not match:
                return []
            items = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return []

        edges = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cause = str(item.get("cause", "")).strip().lower()
            relation = str(item.get("relation", "")).strip().lower().replace(" ", "_")
            effect = str(item.get("effect", "")).strip().lower()
            if not cause or not effect or not relation:
                continue
            if cause in _PRONOUNS or effect in _PRONOUNS:
                continue
            # Look up polarity from known verbs, default +1
            pol = CAUSAL_VERBS.get(relation, ("", +1))[1]
            e = CausalEdge(cause, relation, effect, pol, source_sent,
                           confidence=0.92, extraction_method="llm")
            if _validate_edge(e):
                edges.append(e)
        return edges

    def extract_sentence(self, sentence: str) -> List[CausalEdge]:
        """Run LLM extraction on a single sentence."""
        try:
            raw = self.llm.generate(self._PROMPT.format(text=sentence))
            return self._parse_response(raw, sentence)
        except Exception:
            return []

    def extract(self, text: str) -> List[CausalEdge]:
        """
        Extract causal edges from full text using the LLM.
        Processes sentence-by-sentence to keep prompts short and responses clean.
        """
        edges = []
        for sent in _sentences(text):
            edges.extend(self.extract_sentence(sent))
        return edges


# --------------------------------------------------------------------------- #
#  REBEL: Trained relation extraction (Babelscape/rebel-large)
# --------------------------------------------------------------------------- #

class REBELRelationExtractor:
    """
    Uses the REBEL seq2seq model (Babelscape/rebel-large on Hugging Face) for
    relation extraction. REBEL is trained on 200+ relation types and achieves
    SOTA performance on multiple RE benchmarks.

    Relation format: "REBEL outputs structured text like '< relation>'"
    We parse the model output and map recognized causal relations to CausalEdge.

    Parameters
    ----------
    device : str (default "cpu")
        Device for model inference: "cpu" or "cuda"
    batch_size : int (default 8)
        Batch size for inference on long documents
    """

    # Map REBEL relation names (and variants) to our canonical causal relations
    _REBEL_TO_CAUSAL = {
        "causes": "cause", "caused_by": "caused_by",
        "triggers": "trigger", "triggered_by": "triggered_by",
        "leads_to": "lead_to", "led_to": "lead_to",
        "produces": "produce", "produced_by": "produced_by",
        "results_in": "result_in", "resulted_in": "result_in",
        "increases": "increase", "increased_by": "increase",
        "decreases": "reduce", "reduced_by": "reduce",
        "affects": "affect", "affected_by": "affected_by",
        "disrupts": "disrupt", "disrupted_by": "disrupted_by",
        "prevents": "prevent", "prevented_by": "prevented_by",
        "enables": "enable", "enabled_by": "enabled_by",
        # Add more as needed — these are the most common in incident/causal text
    }

    def __init__(self, device: str = "cpu", batch_size: int = 8, model_name: str = "Babelscape/rebel-large") -> None:
        self.device = device
        self.batch_size = batch_size
        self._model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load the REBEL model and tokenizer."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self.device)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "transformers is required for REBEL. "
                "Install with: pip install transformers torch"
            )

    def _parse_rebel_output(self, text: str, source_sent: str) -> List[CausalEdge]:
        """
        Parse REBEL output format.
        REBEL outputs triplets as structured text: "entity1 <relation> entity2"
        We extract these and convert to CausalEdge format.
        """
        edges = []
        # REBEL format: "entity1 <relation> entity2" per line or space-separated
        # More commonly, it outputs a structured format with angle brackets
        # Example: "The reactor <causes> overheating"

        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to find patterns like "entity1 <relation> entity2"
            match = re.search(r"([^<>]+?)\s*<([^>]+)>\s*([^<>]+)", line)
            if match:
                cause_txt = match.group(1).strip().lower()
                rel_raw = match.group(2).strip().lower()
                effect_txt = match.group(3).strip().lower()

                # Skip pronouns
                if cause_txt in _PRONOUNS or effect_txt in _PRONOUNS:
                    continue
                if not cause_txt or not effect_txt or not rel_raw:
                    continue

                # Map REBEL relation to canonical form
                rel = self._REBEL_TO_CAUSAL.get(rel_raw.replace(" ", "_"), rel_raw)
                # Infer polarity: if it's in CAUSAL_VERBS, use that; else default +1
                pol = CAUSAL_VERBS.get(rel, ("", +1))[1]

                e = CausalEdge(cause_txt, rel, effect_txt, pol, source_sent,
                               confidence=0.78, extraction_method="rebel")
                if _validate_edge(e):
                    edges.append(e)

        return edges

    def extract_sentence(self, sentence: str) -> List[CausalEdge]:
        """Extract relations from a single sentence using REBEL."""
        self._load_model()

        try:
            import torch
            with torch.no_grad():
                inputs = self._tokenizer(
                    sentence,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self._model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=3,
                    temperature=1.0
                )

                output_text = self._tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
        except Exception as e:
            log.warning("REBEL extraction failed: %s", e)
            return []

        return self._parse_rebel_output(output_text, sentence)

    def extract(self, text: str) -> List[CausalEdge]:
        """Extract relations from full text using REBEL."""
        edges = []
        for sent in _sentences(text):
            edges.extend(self.extract_sentence(sent))
        return edges


def extract_edges_hybrid(
    text: str,
    llm: Any,
    mode: str = "augment",
    resolve_coreferences: bool = True,
    schema: str = "general",
) -> List[CausalEdge]:
    """
    Hybrid extraction: spaCy/rule extractor merged with LLM extractor.

    Parameters
    ----------
    text : str
        Document text to extract from.
    llm  : object with .generate(prompt) -> str
        Any LLM adapter.
    mode : "augment" | "full"
        "augment" — LLM only fills gaps (sentences where base extractor
                    found no edges). Recommended for cost-sensitive use.
        "full"    — LLM runs on all sentences. Higher recall, more API calls.
    resolve_coreferences : bool (default True)
        If True, resolve pronouns to antecedents before extraction.

    Returns
    -------
    Deduplicated list of CausalEdge, spaCy edges first then LLM-only edges.
    """
    # Resolve coreferences if requested
    if resolve_coreferences:
        text = _resolve_coreferences(text)

    llm_extractor = LLMEdgeExtractor(llm, mode=mode)
    sents = _sentences(text)

    # Base extraction (spaCy + rules) per sentence
    base_edges: List[CausalEdge] = []
    covered: set[int] = set()  # sentence indices where base found ≥1 edge
    for i, sent in enumerate(sents):
        sent_edges = _intra_edges(sent)
        if sent_edges:
            base_edges.extend(sent_edges)
            covered.add(i)

    # Inter-sentence edges from base extractor
    base_edges_full = extract_edges(text, schema=schema)  # includes inter-sentence chaining
    # Collect only the inter-sentence edges not already in per-sentence pass
    intra_keys = {(e.cause, e.relation, e.effect) for e in base_edges}
    for e in base_edges_full:
        if (e.cause, e.relation, e.effect) not in intra_keys:
            base_edges.append(e)

    # LLM extraction
    llm_edges: List[CausalEdge] = []
    if mode == "augment":
        # Only run LLM on sentences where base extractor found nothing
        for i, sent in enumerate(sents):
            if i not in covered:
                llm_edges.extend(llm_extractor.extract_sentence(sent))
    else:  # "full"
        llm_edges = llm_extractor.extract(text)

    # Merge, deduplicate — prefer base edges; LLM fills gaps
    seen: set[tuple] = {(e.cause, e.relation, e.effect) for e in base_edges}
    merged = list(base_edges)
    for e in llm_edges:
        key = (e.cause, e.relation, e.effect)
        if key not in seen:
            seen.add(key)
            merged.append(e)

    # Final validation pass (covers edges from all extraction paths)
    validated = [e for e in merged if _validate_edge(e)]
    _annotate_edges(validated)
    return validated
