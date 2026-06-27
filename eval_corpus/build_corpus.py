"""
build_corpus.py — assemble a 23-document multi-field causal-reasoning corpus.

Three source types, all written as markdown to eval_corpus/<slug>.md:

  local : curated docs already in the repo (chernobyl, subprime).
  wiki  : plain-text extract from the Wikipedia action API
          (disasters, engineering failures, financial crises).
  arxiv : full IMRaD paper text from ar5iv (LaTeX-rendered HTML), converted
          to markdown with section headings preserved — real scientific
          articles with abstract / methods / results / conclusions.

Each doc is tagged with the doc_structure schema it should ingest under
(incident / general / research) so the eval exercises structure-aware paths.

Run:  python eval_corpus/build_corpus.py
Writes: eval_corpus/<slug>.md (23 files) + eval_corpus/registry.json
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from html.parser import HTMLParser

import requests

_SESSION = requests.Session()


def _get(url: str, *, params=None, timeout=45, attempts=5) -> requests.Response:
    """GET with exponential backoff on 429/5xx/transient network errors."""
    last_exc = None
    for i in range(attempts):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": UA})
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except Exception as e:           # noqa: BLE001 — retry any transient failure
            last_exc = e
            if i < attempts - 1:
                time.sleep(1.5 * (2 ** i))   # 1.5, 3, 6, 12 s
    raise RuntimeError(f"GET failed after {attempts} attempts: {last_exc}")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAX_CHARS = 48000
UA = "causal-graph-rag-eval/1.0 (academic retrieval benchmark)"

# (slug, domain, schema, kind, ref)
CORPUS = [
    # --- disasters (incident schema) ---
    ("chernobyl",        "disaster",    "incident", "local", "chernobyl.md"),
    ("fukushima",        "disaster",    "incident", "wiki",  "Fukushima nuclear accident"),
    ("deepwater",        "disaster",    "incident", "wiki",  "Deepwater Horizon explosion"),
    ("bhopal",           "disaster",    "incident", "wiki",  "Bhopal disaster"),
    ("three_mile",       "disaster",    "incident", "wiki",  "Three Mile Island accident"),
    ("hindenburg",       "disaster",    "incident", "wiki",  "Hindenburg disaster"),
    # --- engineering failures (incident schema) ---
    ("challenger",       "engineering", "incident", "wiki",  "Space Shuttle Challenger disaster"),
    ("tacoma",           "engineering", "incident", "wiki",  "Tacoma Narrows Bridge (1940)"),
    ("boeing737max",     "engineering", "incident", "wiki",  "Boeing 737 MAX groundings"),
    # --- finance / economics (general schema) ---
    ("subprime",         "finance",     "general",  "local", "subprime_causes.md"),
    ("gfc2008",          "finance",     "general",  "wiki",  "2007–2008 financial crisis"),
    ("great_depression", "finance",     "general",  "wiki",  "Great Depression"),
    ("dotcom",           "finance",     "general",  "wiki",  "Dot-com bubble"),
    # --- scientific papers, IMRaD (research schema) ---
    ("ci_matching",      "causal-inf",  "research", "arxiv", "2504.09635"),
    ("ci_kernel",        "causal-inf",  "research", "arxiv", "2502.10958"),
    ("ci_network_gnn",   "causal-inf",  "research", "arxiv", "2211.07823"),
    ("epi_models",       "epidemiology","research", "arxiv", "2410.11743"),
    ("epi_timevarying",  "epidemiology","research", "arxiv", "2508.13427"),
    ("epi_diffgraphs",   "epidemiology","research", "arxiv", "2411.01292"),
    ("clim_extremeval",  "climate",     "research", "arxiv", "2308.07560"),
    ("clim_granger",     "climate",     "research", "arxiv", "2408.16004"),
    ("clim_potattr",     "climate",     "research", "arxiv", "1908.03107"),
    ("causalml_dynamics","causal-ml",   "research", "arxiv", "2505.16620"),
]


# --------------------------------------------------------------------------- #
#  Wikipedia
# --------------------------------------------------------------------------- #
def fetch_wiki(title: str) -> str:
    params = {"format": "json", "action": "query", "prop": "extracts",
              "explaintext": "1", "redirects": "1", "titles": title}
    data = _get("https://en.wikipedia.org/w/api.php", params=params, timeout=30).json()
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    if not text:
        raise RuntimeError(f"empty extract for {title!r}")
    return text[:MAX_CHARS]


# --------------------------------------------------------------------------- #
#  ar5iv (arXiv LaTeX -> HTML) -> markdown with headings
# --------------------------------------------------------------------------- #
_SKIP_TAGS = {"script", "style", "math", "figure", "table", "cite", "nav",
              "header", "footer"}
_HEAD_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####",
              "h5": "#####", "h6": "######"}
_STOP_HEADINGS = re.compile(r"^\s*(references|bibliography|acknowledg|appendix|"
                            r"supplementary)\b", re.I)


class _Ar5ivParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self._skip_depth = 0
        self._cur_tag: str | None = None     # "head"|"p" while capturing
        self._buf: list[str] = []
        self._head_prefix = ""
        self._stopped = False

    def handle_starttag(self, tag, attrs):
        if self._stopped:
            return
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in _HEAD_TAGS:
            self._flush()
            self._cur_tag = "head"
            self._head_prefix = _HEAD_TAGS[tag]
            self._buf = []
        elif tag == "p":
            self._flush()
            self._cur_tag = "p"
            self._buf = []

    def handle_endtag(self, tag):
        if self._stopped:
            return
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in _HEAD_TAGS and self._cur_tag == "head":
            self._flush()
        elif tag == "p" and self._cur_tag == "p":
            self._flush()

    def handle_data(self, data):
        if self._stopped or self._skip_depth or self._cur_tag is None:
            return
        self._buf.append(data)

    def _flush(self):
        if self._cur_tag is None:
            return
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        tag, self._cur_tag = self._cur_tag, None
        if not text:
            return
        if tag == "head":
            if _STOP_HEADINGS.match(text):
                self._stopped = True     # drop references onward
                return
            self.out.append(f"\n{self._head_prefix} {text}\n")
        else:
            self.out.append(text)


def fetch_arxiv(arxiv_id: str) -> str:
    html = _get(f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}", timeout=60).text
    p = _Ar5ivParser()
    p.feed(html)
    md = "\n".join(p.out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < 2500:
        raise RuntimeError(f"ar5iv text too short ({len(md)} chars) for {arxiv_id}")
    return md[:MAX_CHARS]


# --------------------------------------------------------------------------- #
def main() -> int:
    registry = []
    for slug, domain, schema, kind, ref in CORPUS:
        out_path = os.path.join(HERE, f"{slug}.md")
        try:
            # Skip docs already downloaded (idempotent re-runs, polite to hosts)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 2500:
                text = open(out_path, encoding="utf-8").read()
            elif kind == "local":
                text = open(os.path.join(ROOT, ref), encoding="utf-8").read()[:MAX_CHARS]
                open(out_path, "w", encoding="utf-8").write(text)
            elif kind == "wiki":
                text = fetch_wiki(ref)
                open(out_path, "w", encoding="utf-8").write(text)
                time.sleep(1.0)
            else:
                text = fetch_arxiv(ref)
                open(out_path, "w", encoding="utf-8").write(text)
                time.sleep(0.5)
            registry.append({"slug": slug, "domain": domain, "schema": schema,
                             "kind": kind, "ref": ref, "chars": len(text)})
            print(f"  [{domain:>12}/{schema:<8}] {slug:<18} {len(text):>6} chars")
        except Exception as e:
            print(f"  [FAIL] {slug} ({kind}:{ref}): {e}", file=sys.stderr)
    json.dump(registry, open(os.path.join(HERE, "registry.json"), "w"), indent=2)
    ok = len(registry)
    print(f"\nwrote {ok}/{len(CORPUS)} docs + registry.json")
    return 0 if ok == len(CORPUS) else 1


if __name__ == "__main__":
    sys.exit(main())
