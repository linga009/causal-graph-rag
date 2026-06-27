"""
agentic_rag.py
==============
An OPT-IN agentic layer over the causal graph. Instead of the fixed
retrieve -> generate pipeline (GraphRAG.answer), an LLM controller plans and
executes a sequence of graph operations, observes the results, and decides what
to do next — query decomposition, iterative exploration, multi-hop bridging,
and self-verification.

Why this fits THIS project: the agent's action space is the set of **LLM-free,
instant, deterministic** causal-graph operations (root_causes / impact / connect
/ retrieve). So the agent spends LLM calls only on ORCHESTRATION — deciding which
graph op to run — while the retrieval itself stays free and exact. That is a far
cheaper agentic profile than typical agentic RAG, where every tool call is also a
vector search.

This is a MODE, not a replacement:
  - Default (fast, small-model friendly): GraphRAG.answer() — one LLM call.
  - Agentic (this module): N LLM calls for complex / multi-intent / exploratory
    questions where adaptive reasoning earns its cost.

No new dependency: works with any object exposing `.generate(prompt) -> str`
(the project's GroqLLM / AnthropicLLM / ..., or a LangChainLLMAdapter).

Quick start
-----------
    from graph_rag import GraphRAG
    from llm_adapters import GroqLLM
    from agentic_rag import AgenticCausalRAG

    rag = GraphRAG(llm=GroqLLM())
    rag.ingest(report_text, schema="incident")

    agent = AgenticCausalRAG(rag, llm=GroqLLM())
    result = agent.run("Why did the outage happen and what did it ultimately disrupt?")
    print(result.answer)
    for step in result.steps:        # full reasoning trace
        print(step)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

# --------------------------------------------------------------------------- #
#  Action parsing
# --------------------------------------------------------------------------- #
_ACTION_RE = re.compile(r"ACTION:\s*(\w+)\s*\((.*?)\)\s*$", re.I | re.M)
_FINAL_RE = re.compile(r"FINAL:\s*(.+)", re.I | re.S)
_THOUGHT_RE = re.compile(r"THOUGHT:\s*(.+)", re.I)


def _split_args(arg_str: str) -> List[str]:
    """Split tool arguments on commas, stripping quotes/whitespace."""
    return [a.strip().strip('"').strip("'") for a in arg_str.split(",") if a.strip()]


# --------------------------------------------------------------------------- #
#  Result types
# --------------------------------------------------------------------------- #
@dataclass
class AgentStep:
    kind: str                 # "action" | "final" | "invalid"
    thought: str = ""
    tool: str = ""
    args: str = ""
    observation: str = ""

    def __str__(self) -> str:
        if self.kind == "final":
            return f"FINAL: {self.observation}"
        if self.kind == "invalid":
            return f"INVALID: {self.thought[:80]}"
        return f"ACTION {self.tool}({self.args}) -> {self.observation[:120]}"


@dataclass
class AgentResult:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    n_llm_calls: int = 0


# --------------------------------------------------------------------------- #
#  Agentic controller
# --------------------------------------------------------------------------- #
_SYSTEM = """You are a causal-reasoning agent exploring a causal knowledge graph.
You answer questions by calling graph tools, observing results, and reasoning step
by step. The tools traverse a cause->effect graph; they are exact and free.

Tools:
- rootcause(event)        : trace what CAUSED an event (backward causal chains)
- impact(event)           : trace what an event LEADS TO (forward causal chains)
- path(source, target)    : the shortest causal path connecting two events
- retrieve(query)         : the most relevant causal chains + evidence for a query

Respond in EXACTLY this format, one step at a time:
THOUGHT: <your reasoning about what to do next>
ACTION: tool_name(arguments)

After each ACTION I will reply with OBSERVATION: <result>. Continue until you can
answer, then respond with:
THOUGHT: <why you can now answer>
FINAL: <your concise, direct answer grounded in the observations>

Rules: ONE action per step. Use rootcause/impact for "why"/"what results"
questions, path for "how is A connected to B", retrieve for general lookups.
Decompose multi-part questions into multiple actions. Do not invent facts not in
the observations."""


class AgenticCausalRAG:
    """ReAct-style controller over the causal graph's LLM-free operations."""

    def __init__(self, rag: Any, llm: Any, max_steps: int = 6,
                 max_obs_chars: int = 900):
        self.rag = rag
        self.llm = llm
        self.max_steps = max_steps
        self.max_obs_chars = max_obs_chars
        self.tools: Dict[str, Callable[[List[str]], str]] = {
            "rootcause": self._t_rootcause,
            "impact":    self._t_impact,
            "path":      self._t_path,
            "retrieve":  self._t_retrieve,
        }

    # -- tools (each returns a text observation; all are LLM-free) ----------- #
    def _fmt_chains(self, chains: list, limit: int = 6) -> str:
        if not chains:
            return "(no causal chains found)"
        lines = [f"  - {c.text()}" for c in chains[:limit]]
        if len(chains) > limit:
            lines.append(f"  ... ({len(chains) - limit} more)")
        return "\n".join(lines)

    def _t_rootcause(self, args: List[str]) -> str:
        if not args:
            return "rootcause needs an event argument."
        node, chains = self.rag.root_causes(args[0])
        if node is None:
            return f"No graph node matches '{args[0]}'."
        return f"Root causes of '{node}':\n{self._fmt_chains(chains)}"

    def _t_impact(self, args: List[str]) -> str:
        if not args:
            return "impact needs an event argument."
        node, chains = self.rag.impact(args[0])
        if node is None:
            return f"No graph node matches '{args[0]}'."
        return f"Downstream impacts of '{node}':\n{self._fmt_chains(chains)}"

    def _t_path(self, args: List[str]) -> str:
        if len(args) < 2:
            return "path needs source and target arguments."
        s, d, chain = self.rag.connect(args[0], args[1])
        if s is None or d is None:
            miss = args[0] if s is None else args[1]
            return f"No graph node matches '{miss}'."
        if chain is None:
            return (f"No causal path from '{s}' to '{d}'. "
                    "Try rootcause/impact on each separately.")
        return f"Causal path '{s}' -> '{d}':\n  {chain.text()}"

    def _t_retrieve(self, args: List[str]) -> str:
        query = args[0] if args else ""
        chains = self.rag.retrieve(query, top_k=4)
        if not chains:
            return "(no relevant chains)"
        out = []
        for c in chains[:4]:
            prov = c.provenance()
            ev = f"  evidence: {prov[0]}" if prov else ""
            out.append(f"  - {c.text()}\n{ev}".rstrip())
        return "Relevant chains:\n" + "\n".join(out)

    # -- controller loop ----------------------------------------------------- #
    def _execute(self, tool: str, arg_str: str) -> str:
        fn = self.tools.get(tool.lower())
        if fn is None:
            return (f"Unknown tool '{tool}'. Available: "
                    f"{', '.join(self.tools)}.")
        try:
            obs = fn(_split_args(arg_str))
        except Exception as exc:                      # never let a tool crash the loop
            return f"Tool '{tool}' errored: {exc}"
        return obs[: self.max_obs_chars]

    def _prompt(self, question: str, scratchpad: str) -> str:
        return (f"{_SYSTEM}\n\nQuestion: {question}\n{scratchpad}\n"
                "THOUGHT:")

    def run(self, question: str) -> AgentResult:
        """Run the agent loop. Returns the final answer + full reasoning trace."""
        scratchpad = ""
        steps: List[AgentStep] = []
        calls = 0

        for _ in range(self.max_steps):
            out = self.llm.generate(self._prompt(question, scratchpad))
            calls += 1
            thought_m = _THOUGHT_RE.search(out)
            thought = thought_m.group(1).strip() if thought_m else ""

            final_m = _FINAL_RE.search(out)
            if final_m:
                ans = final_m.group(1).strip()
                steps.append(AgentStep("final", thought=thought, observation=ans))
                return AgentResult(answer=ans, steps=steps, n_llm_calls=calls)

            act_m = _ACTION_RE.search(out)
            if not act_m:
                steps.append(AgentStep("invalid", thought=out.strip()[:200]))
                scratchpad += (f"\nTHOUGHT: {thought}\nOBSERVATION: I expected an "
                               "ACTION: tool(args) or FINAL: answer. Please follow "
                               "the format.\n")
                continue

            tool, arg_str = act_m.group(1), act_m.group(2)
            obs = self._execute(tool, arg_str)
            steps.append(AgentStep("action", thought=thought, tool=tool,
                                   args=arg_str, observation=obs))
            scratchpad += (f"\nTHOUGHT: {thought}\nACTION: {tool}({arg_str})\n"
                           f"OBSERVATION: {obs}\n")

        # Out of steps — force a final synthesis from what we gathered.
        ans = self._force_final(question, scratchpad)
        calls += 1
        steps.append(AgentStep("final", thought="(max steps reached)", observation=ans))
        return AgentResult(answer=ans, steps=steps, n_llm_calls=calls)

    def _force_final(self, question: str, scratchpad: str) -> str:
        prompt = (
            "You are a causal-reasoning assistant. Using ONLY the observations "
            "below, give a concise, direct answer to the question.\n\n"
            f"Observations:\n{scratchpad}\n\nQuestion: {question}\n\nAnswer:"
        )
        try:
            return self.llm.generate(prompt).strip()
        except Exception as exc:
            return f"(agent could not synthesize an answer: {exc})"


def build_agent(rag: Any, llm: Any, max_steps: int = 6) -> AgenticCausalRAG:
    """Convenience constructor."""
    return AgenticCausalRAG(rag, llm, max_steps=max_steps)
