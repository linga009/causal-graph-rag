"""
Tests for agentic_rag.py — the ReAct controller over the causal graph.

Uses a scripted fake LLM (no API key) to drive the loop deterministically and
verify that tools execute, observations feed back, multi-step decomposition
works, and the loop terminates safely.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph_rag.graph_rag import GraphRAG
from causal_graph_rag.agentic_rag import AgenticCausalRAG, _split_args, _ACTION_RE, _FINAL_RE

_TEXT = (
    "The pump failed. This caused the reactor to overheat. "
    "The overheating triggered an emergency scram. "
    "The scram led to a 12-hour outage. "
    "The outage disrupted hospital operations."
)


class ScriptLLM:
    """Returns canned responses in order; repeats the last one if exhausted."""
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate(self, prompt):
        i = min(len(self.calls), len(self.responses) - 1)
        self.calls.append(prompt)
        return self.responses[i]


def _rag():
    r = GraphRAG()
    r.ingest(_TEXT)
    return r


# --- parsing ---------------------------------------------------------------- #

def test_split_args():
    assert _split_args("scram") == ["scram"]
    assert _split_args('"pump", "outage"') == ["pump", "outage"]
    assert _split_args("") == []


def test_action_regex():
    m = _ACTION_RE.search("THOUGHT: find it\nACTION: rootcause(scram)")
    assert m and m.group(1) == "rootcause" and m.group(2) == "scram"


def test_final_regex():
    m = _FINAL_RE.search("THOUGHT: done\nFINAL: The pump failed.")
    assert m and "pump" in m.group(1)


# --- single-step loop ------------------------------------------------------- #

def test_agent_single_action_then_final():
    llm = ScriptLLM([
        " find the cause\nACTION: rootcause(scram)",
        " I can answer now\nFINAL: The pump failure caused the scram.",
    ])
    agent = AgenticCausalRAG(_rag(), llm)
    res = agent.run("What caused the scram?")
    assert "pump" in res.answer.lower()
    assert res.n_llm_calls == 2
    assert any(s.tool == "rootcause" for s in res.steps if s.kind == "action")
    # the observation from the tool must have been a real graph result
    act = next(s for s in res.steps if s.kind == "action")
    assert "scram" in act.observation.lower()


# --- multi-step decomposition ---------------------------------------------- #

def test_agent_decomposes_multi_intent():
    llm = ScriptLLM([
        " trace the cause\nACTION: rootcause(outage)",
        " now the effect\nACTION: impact(pump)",
        " synthesize\nFINAL: The pump failure cascaded to the outage, which "
        "disrupted hospital operations.",
    ])
    agent = AgenticCausalRAG(_rag(), llm)
    res = agent.run("Why did the outage happen and what did the pump failure disrupt?")
    tools_used = [s.tool for s in res.steps if s.kind == "action"]
    assert "rootcause" in tools_used and "impact" in tools_used
    assert "hospital" in res.answer.lower()


# --- robustness ------------------------------------------------------------- #

def test_agent_handles_unknown_tool():
    llm = ScriptLLM([
        " try a bad tool\nACTION: teleport(scram)",
        " recover\nFINAL: done.",
    ])
    agent = AgenticCausalRAG(_rag(), llm)
    res = agent.run("x?")
    act = next(s for s in res.steps if s.kind == "action")
    assert "unknown tool" in act.observation.lower()


def test_agent_handles_missing_node():
    llm = ScriptLLM([
        " look it up\nACTION: rootcause(nonexistent_xyz)",
        " nothing there\nFINAL: No information.",
    ])
    agent = AgenticCausalRAG(_rag(), llm)
    res = agent.run("x?")
    act = next(s for s in res.steps if s.kind == "action")
    assert "no graph node matches" in act.observation.lower()


def test_agent_max_steps_forces_final():
    # Never emits FINAL -> loop must hit max_steps and force a synthesis.
    llm = ScriptLLM([" loop\nACTION: retrieve(pump)"])
    agent = AgenticCausalRAG(_rag(), llm, max_steps=3)
    res = agent.run("What happened?")
    assert isinstance(res.answer, str) and res.answer
    # max_steps action calls + 1 forced-final call
    assert res.n_llm_calls == 4


def test_agent_invalid_format_is_nudged():
    llm = ScriptLLM([
        "I will just chat without the format.",
        " ok now properly\nFINAL: answered.",
    ])
    agent = AgenticCausalRAG(_rag(), llm)
    res = agent.run("x?")
    assert any(s.kind == "invalid" for s in res.steps)
    assert res.answer == "answered."
