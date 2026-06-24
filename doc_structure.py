"""
doc_structure.py
================
Structure-preserving document parser.

Chunk-and-embed RAG throws away a document's organization — its headings,
section roles, paragraph grouping and sentence order. That skeleton *is*
knowledge: a claim in a "Conclusion" is a synthesized finding; the same words
in "Background" are context. This module recovers that skeleton.

It parses a document into a typed hierarchy

    Document
      └─ Section            (tagged with a discourse ROLE: abstract, methods,
           └─ Paragraph      results, conclusion, ... — domain-aware)
                └─ Sentence

and exposes it as typed edges (CONTAINS for hierarchy, FOLLOWS for order) that
feed the same VSA-encoded graph the causal layer uses. Each Sentence can later
be linked to the concept nodes the causal extractor produces (MENTIONS), so a
query can traverse *both* the document structure and the causal topology.

Increment #1 scope: Markdown / clean text where headings are explicit
(`#`, numbered, or ALL-CAPS). PDF layout parsing (via the `liteparse` skill /
PyMuPDF) plugs into `parse()` later without changing the data model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
#  Discourse-role taxonomy (per-domain document schemas)
# --------------------------------------------------------------------------- #
# A heading's text maps to a canonical role. Different document types have
# different skeletons; we register a schema per domain and detect which fits.

# Domain presets are OPT-IN. The core parser is domain-agnostic (hierarchy +
# order + position); selecting a preset adds discourse-role tagging on top.
#   schema="general" (default) -> structure only, no role labels, works anywhere
#   schema="research"|"clinical"|"incident" -> apply that preset's role map
#   schema="auto" -> pick the preset whose role keywords match the most headings
# Add a new domain = add one entry here. The core path never changes.
ROLE_PATTERNS: Dict[str, Dict[str, re.Pattern]] = {
    # IMRaD — research articles
    "research": {
        "abstract":     re.compile(r"\b(abstract|summary)\b", re.I),
        "introduction": re.compile(r"\b(introduction|background|motivation)\b", re.I),
        "related_work": re.compile(r"\b(related work|prior work|literature review)\b", re.I),
        "methods":      re.compile(r"\b(methods?|methodology|materials|approach|experimental setup)\b", re.I),
        "results":      re.compile(r"\b(results?|findings|evaluation|experiments?)\b", re.I),
        "discussion":   re.compile(r"\b(discussion|analysis|interpretation)\b", re.I),
        "conclusion":   re.compile(r"\b(conclusions?|concluding remarks|future work)\b", re.I),
        "references":   re.compile(r"\b(references|bibliography|works cited)\b", re.I),
    },
    # SOAP — clinical notes
    "clinical": {
        "subjective":   re.compile(r"\b(subjective|chief complaint|history of present illness|hpi)\b", re.I),
        "objective":    re.compile(r"\b(objective|vitals|physical exam|examination|labs?)\b", re.I),
        "assessment":   re.compile(r"\b(assessment|impression|diagnos[ei]s)\b", re.I),
        "plan":         re.compile(r"\b(plan|treatment|management|follow.?up|disposition)\b", re.I),
    },
    # Incident / RCA reports — industry
    "incident": {
        "summary":      re.compile(r"\b(summary|overview|executive summary)\b", re.I),
        "timeline":     re.compile(r"\b(timeline|sequence of events|chronology)\b", re.I),
        "root_cause":   re.compile(r"\b(root cause|cause analysis|contributing factors)\b", re.I),
        "impact":       re.compile(r"\b(impact|consequences|effects?)\b", re.I),
        "remediation":  re.compile(r"\b(remediation|corrective actions?|recommendations?|mitigation)\b", re.I),
    },
}

# Roles that carry synthesized / high-value knowledge — retrieval can prefer
# these for "what's the takeaway / main finding / conclusion" style queries.
SYNTHESIS_ROLES = frozenset({
    "abstract", "summary", "conclusion", "assessment", "discussion", "root_cause",
})


# --------------------------------------------------------------------------- #
#  Data model
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    """A node in the document hierarchy."""
    block_id: str                 # stable id, e.g. "sec:2", "para:5", "sent:11"
    kind: str                     # "section" | "paragraph" | "sentence"
    text: str
    order: int                    # global document order (sequence)
    level: int = 0                # heading depth for sections (1 = top)
    role: Optional[str] = None    # discourse role for sections
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class DocStructure:
    """Parsed document hierarchy + indices."""
    doc_id: str
    schema: str                       # detected domain schema name
    blocks: Dict[str, Block]          # block_id -> Block
    order: List[str]                  # block_ids in document order
    root_id: str

    # -- accessors ---------------------------------------------------------- #
    def sentences(self) -> List[Block]:
        return [self.blocks[b] for b in self.order if self.blocks[b].kind == "sentence"]

    def sections(self) -> List[Block]:
        return [self.blocks[b] for b in self.order if self.blocks[b].kind == "section"]

    def section_of(self, block_id: str) -> Optional[Block]:
        """Walk up to the enclosing section for any block."""
        cur = self.blocks.get(block_id)
        while cur is not None:
            if cur.kind == "section":
                return cur
            cur = self.blocks.get(cur.parent_id) if cur.parent_id else None
        return None

    def role_of(self, block_id: str) -> Optional[str]:
        sec = self.section_of(block_id)
        return sec.role if sec else None

    # -- typed structural edges (CONTAINS / FOLLOWS) ------------------------ #
    def structural_edges(self) -> List[Tuple[str, str, str]]:
        """Yield (head_id, relation, tail_id) for the hierarchy + sequence.

        CONTAINS: parent -> child (document skeleton)
        FOLLOWS : sentence_i -> sentence_{i+1} within the same section (order)
        """
        edges: List[Tuple[str, str, str]] = []
        for b in self.blocks.values():
            for child in b.children:
                edges.append((b.block_id, "contains", child))
        # FOLLOWS over sentences in reading order, scoped to a section
        sents = self.sentences()
        for a, b in zip(sents, sents[1:]):
            if self.section_of(a.block_id) is self.section_of(b.block_id):
                edges.append((a.block_id, "follows", b.block_id))
        return edges


# --------------------------------------------------------------------------- #
#  Parsing
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z].{0,80})$")
_CAPS_HEADING = re.compile(r"^([A-Z][A-Z0-9 \-/]{2,60})$")

# Every role keyword across all schemas — lets us recognize a bare section
# label ("Subjective", "Conclusion:") as a heading even when it isn't marked
# up with '#'. Clinical SOAP notes and many plain-text reports rely on this.
_ALL_ROLE_PATTERNS = [p for schema in ROLE_PATTERNS.values() for p in schema.values()]


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if len(s.strip()) > 1]


def _heading(line: str) -> Optional[Tuple[int, str]]:
    """Return (level, title) if the line is a heading, else None."""
    m = _MD_HEADING.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _NUM_HEADING.match(line)
    if m:
        return m.group(1).count(".") + 1, m.group(2).strip()
    m = _CAPS_HEADING.match(line.strip())
    if m and len(line.split()) <= 8:
        return 1, m.group(1).strip()
    # Bare section label: a short standalone line (optionally with a trailing
    # colon, no terminal punctuation) that matches a known role keyword.
    stripped = line.strip().rstrip(":").strip()
    if (stripped and len(stripped.split()) <= 5
            and not stripped.endswith((".", "!", "?"))
            and any(p.search(stripped) for p in _ALL_ROLE_PATTERNS)):
        return 1, stripped
    return None


# Selectable presets the user can pass to parse()/ingest. "general" is the
# default and applies no domain assumptions.
AVAILABLE_SCHEMAS: Tuple[str, ...] = ("general", "auto") + tuple(ROLE_PATTERNS)


def detect_schema(headings: List[str]) -> str:
    """Pick the domain preset whose role keywords match the most headings.
    Only used when the caller opts in with schema='auto'. Falls back to
    'general' when nothing matches, so we never force a domain."""
    best, best_hits = "general", 0
    for schema, patterns in ROLE_PATTERNS.items():
        hits = sum(1 for h in headings if any(p.search(h) for p in patterns.values()))
        if hits > best_hits:
            best, best_hits = schema, hits
    return best


def detect_role(heading: str, schema: str) -> Optional[str]:
    """Role label for a heading under a chosen preset. 'general' = no roles."""
    if schema in ("general", "auto"):
        return None
    for role, pat in ROLE_PATTERNS.get(schema, {}).items():
        if pat.search(heading):
            return role
    return None


def parse(text: str, doc_id: str = "doc", schema: str = "general") -> DocStructure:
    """Parse markdown / clean text into a typed document hierarchy.

    Heading detection: Markdown (`#`), numbered (`1.2 Title`), short ALL-CAPS
    lines, or bare section labels. Paragraphs split on blank lines; sentences
    on terminal punctuation. Content before the first heading goes under an
    implicit 'body' section so nothing is dropped.

    Parameters
    ----------
    schema : str (default "general")
        Domain preset for discourse-role tagging — the user's choice:
          "general"  — structure only, no role labels (works on any document)
          "research" — IMRaD (abstract/methods/results/conclusion/...)
          "clinical" — SOAP (subjective/objective/assessment/plan)
          "incident" — RCA (summary/timeline/root_cause/impact/remediation)
          "auto"     — detect the best-fitting preset from the headings
        Structure (hierarchy, order, position) is identical across all of
        these; only role labels differ.
    """
    if schema not in AVAILABLE_SCHEMAS:
        raise ValueError(
            f"unknown schema {schema!r}; choose from {AVAILABLE_SCHEMAS}"
        )
    lines = text.splitlines()

    # Resolve "auto" to a concrete preset from the headings (or 'general').
    if schema == "auto":
        heading_texts = [h[1] for ln in lines if (h := _heading(ln))]
        schema = detect_schema(heading_texts)

    blocks: Dict[str, Block] = {}
    order: List[str] = []
    counters = {"sec": 0, "para": 0, "sent": 0}

    def new_id(kind_prefix: str) -> str:
        counters[kind_prefix] += 1
        return f"{kind_prefix}:{counters[kind_prefix]}"

    root = Block(block_id="doc:0", kind="document", text=doc_id, order=0)
    blocks[root.block_id] = root
    order.append(root.block_id)

    # Implicit section for any preamble before the first heading.
    cur_section = Block(
        block_id=new_id("sec"), kind="section", text="(body)", order=len(order),
        level=1, role=None, parent_id=root.block_id,
    )
    blocks[cur_section.block_id] = cur_section
    root.children.append(cur_section.block_id)
    order.append(cur_section.block_id)

    para_buf: List[str] = []

    def flush_paragraph() -> None:
        nonlocal para_buf
        if not para_buf:
            return
        para_text = " ".join(para_buf).strip()
        para_buf = []
        if not para_text:
            return
        para = Block(
            block_id=new_id("para"), kind="paragraph", text=para_text,
            order=len(order), parent_id=cur_section.block_id,
        )
        blocks[para.block_id] = para
        cur_section.children.append(para.block_id)
        order.append(para.block_id)
        for s in _split_sentences(para_text):
            sent = Block(
                block_id=new_id("sent"), kind="sentence", text=s,
                order=len(order), parent_id=para.block_id,
            )
            blocks[sent.block_id] = sent
            para.children.append(sent.block_id)
            order.append(sent.block_id)

    for raw in lines:
        line = raw.rstrip()
        h = _heading(line)
        if h:
            flush_paragraph()
            level, title = h
            cur_section = Block(
                block_id=new_id("sec"), kind="section", text=title,
                order=len(order), level=level,
                role=detect_role(title, schema), parent_id=root.block_id,
            )
            blocks[cur_section.block_id] = cur_section
            root.children.append(cur_section.block_id)
            order.append(cur_section.block_id)
        elif not line.strip():
            flush_paragraph()
        else:
            para_buf.append(line.strip())
    flush_paragraph()

    return DocStructure(doc_id=doc_id, schema=schema, blocks=blocks,
                        order=order, root_id=root.block_id)
