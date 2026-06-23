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
import re
import textwrap
from dataclasses import dataclass
from typing import Any, List, Optional

from parser import parse_triples, _clean, _split_clauses
from vsa_core import Triple


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
    "improve": ("improve", +1), "improves": ("improve", +1), "improved": ("improve", +1),
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

    def text(self) -> str:
        arrow = "==>" if self.polarity > 0 else "=/=>"
        return f"{self.cause} {arrow}[{self.relation}] {self.effect}"


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


_PRONOUNS = {"this", "that", "these", "those", "it", "they", "he", "she", "we", "i"}


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
                if child.dep_ in ("npadvmod", "nsubj"):
                    cause_txt = _compound_span(child)
                    if cause_txt and effect_txt and cause_txt != effect_txt:
                        edges.append(CausalEdge(cause_txt, rel, effect_txt, pol, sent))
            continue  # don't also try patterns 1/2 for the same token

        # Find syntactic subject — filter pronouns that coreference resolution
        # would need to resolve (they add noise as standalone graph nodes)
        subjects = [c for c in token.children
                    if c.dep_ in ("nsubj", "nsubjpass")
                    and c.lower_ not in _PRONOUNS]
        # Objects: direct object OR prepositional object ("led to X")
        objects = [c for c in token.children if c.dep_ in ("dobj", "attr")]
        if not objects:
            for prep in (c for c in token.children if c.dep_ == "prep"):
                objects += [gc for gc in prep.children if gc.dep_ == "pobj"]

        is_passive = any(c.dep_ == "nsubjpass" for c in token.children)

        if is_passive:
            # Pattern 2: "Y was triggered by X" -> cause=X, effect=Y
            effect_tokens = subjects
            cause_tokens = []
            for c in token.children:
                if c.dep_ == "agent":
                    cause_tokens += [gc for gc in c.children if gc.dep_ == "pobj"]
            if cause_tokens:
                for eff in effect_tokens:
                    for cau in cause_tokens:
                        e_txt = _compound_span(eff)
                        c_txt = _compound_span(cau)
                        if e_txt and c_txt and e_txt != c_txt:
                            edges.append(CausalEdge(c_txt, rel, e_txt, pol, sent))
        else:
            # Pattern 1: "X caused Y"
            for subj in subjects:
                for obj in objects:
                    s_txt = _compound_span(subj)
                    o_txt = _compound_span(obj)
                    if s_txt and o_txt and s_txt != o_txt:
                        edges.append(CausalEdge(s_txt, rel, o_txt, pol, sent))

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
            edges.append(CausalEdge(tr.agent, rel, tr.patient, pol, sent))
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
    """The event/entity a clause is about: head noun of its subject."""
    cleaned = _strip_connectives(sentence)

    trips = parse_triples(cleaned)
    if trips:
        cand = trips[0].agent or trips[0].patient
        if cand and cand not in _NON_EVENTS:
            return cand

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
    return " ".join(candidates[-2:]) if len(candidates) >= 2 else candidates[-1]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# --------------------------------------------------------------------------- #
#  Implicit causation: adjacency + state-change heuristic
# --------------------------------------------------------------------------- #

def _implicit_edges(sents: List[str], explicit_pairs: set) -> List[CausalEdge]:
    """Create weak implicit_trigger edges for consecutive sentences where:
      - sentence N+1 contains a state-change verb
      - no explicit causal or backward connective is present in sentence N+1
      - the (cause_head, effect_head) pair is not already captured explicitly

    This catches "the bridge was wet. Cars skidded." — causation implied by
    adjacency and the state-change nature of the second event.
    """
    edges: List[CausalEdge] = []
    for i in range(1, len(sents)):
        s_prev = sents[i - 1]
        s_curr = sents[i]
        low = s_curr.lower()

        if any(c in low for c in FORWARD_CONNECTIVES + BACKWARD_CONNECTIVES):
            continue

        words = [w.rstrip(".,;:!?") for w in low.split()]
        if not any(w in STATE_CHANGE_VERBS for w in words):
            continue

        prev_head = _event_head(s_prev)
        curr_head = _event_head(s_curr)

        if (prev_head and curr_head and prev_head != curr_head
                and (prev_head, curr_head) not in explicit_pairs):
            edges.append(CausalEdge(
                prev_head, "implicit_trigger", curr_head, +1,
                f"{s_prev} {s_curr}",
            ))
    return edges


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def extract_edges(text: str, resolve_coreferences: bool = True) -> List[CausalEdge]:
    """
    Extract causal edges from text.

    Parameters
    ----------
    text : str
        Input text to extract causal edges from.
    resolve_coreferences : bool (default True)
        If True, resolve pronouns to antecedents before extraction.
        This prevents pronouns from becoming ghost nodes in the graph.
    """
    # Optionally resolve coreferences (pronouns -> antecedents)
    if resolve_coreferences:
        text = _resolve_coreferences(text)

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
                                        f"{sents[idx-1]} {sent}"))

        # (B') backward connectives inside one sentence: "Y happened because of X"
        bwd_hit = next((c for c in BACKWARD_CONNECTIVES if c in low), None)
        if bwd_hit:
            parts = re.split(re.escape(bwd_hit), low, maxsplit=1)
            if len(parts) == 2:
                effect_head = _event_head(parts[0])
                cause_head = _event_head(parts[1])
                if cause_head and effect_head and cause_head != effect_head:
                    edges.append(CausalEdge(cause_head, "lead_to", effect_head,
                                            +1, sent))

    # Implicit causation pass: adjacency + state-change heuristic
    explicit_pairs = {(e.cause, e.effect) for e in edges}
    edges.extend(_implicit_edges(sents, explicit_pairs))

    # de-duplicate identical edges
    seen = set()
    uniq = []
    for e in edges:
        key = (e.cause, e.relation, e.effect, e.polarity)
        if key not in seen:
            seen.add(key)
            uniq.append(e)
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
            edges.append(CausalEdge(cause, relation, effect, pol, source_sent))
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

                edges.append(CausalEdge(cause_txt, rel, effect_txt, pol, source_sent))

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
            print(f"REBEL extraction failed: {e}")
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
    base_edges_full = extract_edges(text)  # includes inter-sentence chaining
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

    return merged
