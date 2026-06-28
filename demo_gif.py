"""
demo_gif.py — a clean, self-contained demo for recording a README GIF.

Tells the story in ~12s: build a causal graph from a short incident report
(no LLM), then answer "why / what-if / how-connected" by traversing it —
each query in well under a millisecond with zero LLM calls.

Record it with VHS:   vhs demo.tape         (-> assets/demo.gif)
or asciinema:         asciinema rec -c "python demo_gif.py" demo.cast
"""
from __future__ import annotations
import os
import sys
import time

# Silence model-load chatter BEFORE importing anything heavy.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Render unicode (→ ✓) regardless of the console's default codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ANSI colors
B, DIM, GRN, CYN, YEL, MAG, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[36m", "\033[33m", "\033[35m", "\033[0m")


def line(s="", pause=0.5):
    print(s)
    sys.stdout.flush()
    time.sleep(pause)


def typed(prompt, pause=0.7):
    print(f"{DIM}${RST} {B}{prompt}{RST}")
    sys.stdout.flush()
    time.sleep(pause)


DOC = ("The reactor overheated. The coolant valve failed. This triggered an "
       "emergency shutdown. The shutdown caused a power outage. The power "
       "outage disrupted hospital operations.")


def main():
    line(f"\n{B}{MAG}Causal Graph RAG{RST}  {DIM}— answer why / what-if by traversing cause→effect chains{RST}\n", 0.8)

    line(f"{DIM}# the incident report:{RST}", 0.3)
    line(f'{DIM}"{DOC}"{RST}\n', 1.2)

    # Build the graph (load noise suppressed)
    typed("causal-rag ingest incident.md --save graph.pkl", 0.4)
    import io
    _err = sys.stderr
    sys.stderr = io.StringIO()                 # hide any residual load warnings
    try:
        from causal_graph_rag.graph_rag import GraphRAG
        rag = GraphRAG()
        rag.ingest(DOC, schema="incident")
    finally:
        sys.stderr = _err
    line(f"  {GRN}✓{RST} {len(list(rag.graph.edges))} causal edges, "
         f"{len(rag.graph.nodes())} nodes  {DIM}(spaCy/rules — no LLM){RST}\n", 1.0)

    def run(label, cmd, fn):
        typed(cmd, 0.5)
        t = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t) * 1000
        for ln in out:
            line(f"  {CYN}{ln}{RST}", 0.15)
        line(f"  {DIM}{label} in {ms:.2f} ms · no LLM, no embedding search{RST}\n", 1.0)

    def chain_lines(chains):
        return [c.text() for c in chains[:1]] or ["(none)"]

    run("traced", 'causal-rag rootcause graph.pkl "hospital operations"',
        lambda: chain_lines(rag.root_causes("hospital operations")[1]))

    run("traced", 'causal-rag impact    graph.pkl "reactor overheated"',
        lambda: chain_lines(rag.impact("reactor overheated")[1]))

    def path_lines():
        s, d, ch = rag.connect("reactor overheated", "hospital operations")
        return [ch.text()] if ch else ["(no path)"]
    run("connected", 'causal-rag path     graph.pkl "reactor" "hospital operations"', path_lines)

    line(f"{B}{YEL}→ flat RAG can't do this at any price.{RST} "
         f"{DIM}The answer is 4 hops from the cause.{RST}", 1.0)
    line(f"{DIM}  pip install causal-graph-rag  ·  github.com/linga009/causal-graph-rag{RST}\n", 1.5)


if __name__ == "__main__":
    main()
