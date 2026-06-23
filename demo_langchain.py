"""
demo_langchain.py
=================
Demonstrates the three LangChain integration surfaces:

  1. VSAGraphRetriever   — used standalone (retriever.invoke(query))
  2. build_rag_chain()   — LCEL chain:  question -> retriever -> LLM -> answer
  3. build_rag_tool()    — Tool in a ReAct / tool-calling agent

Loads GROQ_API_KEY from vsa_rag/.env.  Falls back to MockLLM if no key is set.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# --------------------------------------------------------------------------- #
#  .env loader (same as demo_rinn.py)
# --------------------------------------------------------------------------- #
def _load_env(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val:
                    os.environ.setdefault(key, val)
    except FileNotFoundError:
        pass


_load_env(os.path.join(os.path.dirname(__file__), ".env"))

# --------------------------------------------------------------------------- #
#  Sample document (same causal corpus as demo_graph.py)
# --------------------------------------------------------------------------- #
DEMO_TEXT = """
The temperature sensor failed, which led to reactor overheating.
The reactor overheated, which caused the coolant valve to jam.
The jammed valve triggered the emergency shutdown.
The emergency shutdown caused a power outage.
The power outage disrupted hospital operations.
Budget cuts reduced the inspection frequency.
Reduced inspection frequency increased the risk of equipment failure.
"""

QUERIES = [
    "What did the reactor overheating ultimately cause?",
    "Why did the power outage happen?",
    "What caused the emergency shutdown?",
    "What did budget cuts lead to?",
]


# --------------------------------------------------------------------------- #
#  Build LLM
# --------------------------------------------------------------------------- #
def _build_langchain_llm():
    if os.environ.get("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.environ["GROQ_API_KEY"],
            )
            return llm, "ChatGroq (llama-3.1-8b-instant)"
        except ImportError:
            print("[warn] langchain-groq not installed: pip install langchain-groq")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model="claude-opus-4-8",
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            return llm, "ChatAnthropic (claude-opus-4-8)"
        except ImportError:
            print("[warn] langchain-anthropic not installed: pip install langchain-anthropic")

    # No LLM key: use MockLLM wrapped so it fits LangChain expectations
    return None, "MockLLM"


# --------------------------------------------------------------------------- #
#  SURFACE 1 — retriever.invoke()
# --------------------------------------------------------------------------- #
def demo_retriever(retriever) -> None:
    print("\n" + "=" * 64)
    print("SURFACE 1: VSAGraphRetriever  (retriever.invoke)")
    print("=" * 64)
    for q in QUERIES[:2]:
        docs = retriever.invoke(q)
        print(f"\nQ: {q}")
        for i, doc in enumerate(docs, 1):
            print(f"  Doc {i}: {doc.page_content}")
            print(f"          meta: entry={doc.metadata['entry_node']!r}  "
                  f"dir={doc.metadata['direction']}  "
                  f"len={doc.metadata['chain_length']}")


# --------------------------------------------------------------------------- #
#  SURFACE 2 — LCEL chain  (chain.invoke)
# --------------------------------------------------------------------------- #
def demo_chain(retriever, lc_llm) -> None:
    from langchain_integration import build_rag_chain

    chain = build_rag_chain(retriever, lc_llm)

    print("\n" + "=" * 64)
    print("SURFACE 2: LCEL chain  (retriever | prompt | LLM | parser)")
    print("=" * 64)
    for q in QUERIES:
        print(f"\nQ: {q}")
        answer = chain.invoke(q)
        print(f"A: {answer}")
        print("-" * 64)


# --------------------------------------------------------------------------- #
#  SURFACE 3 — Tool-calling agent
# --------------------------------------------------------------------------- #
def demo_agent(rag, lc_llm) -> None:
    try:
        from langchain.agents import create_agent  # langchain 1.x API
    except ImportError:
        print("\n[skip] langchain agents require: pip install langchain")
        return

    from langchain_integration import build_rag_tool

    tool = build_rag_tool(rag)

    agent = create_agent(
        model=lc_llm,
        tools=[tool],
        system_prompt=(
            "You are a helpful assistant with access to a causal knowledge graph. "
            "Use the causal_graph_search tool to look up cause-effect relationships "
            "before answering questions about why things happened or what effects they had."
        ),
    )

    print("\n" + "=" * 64)
    print("SURFACE 3: Tool-calling agent  (langchain 1.x create_agent)")
    print("=" * 64)
    q = "I want to understand the full causal chain: why were hospital operations disrupted?"
    print(f"\nQ: {q}")
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    # Extract the last AI message from the messages list
    messages = result.get("messages", [])
    last_ai = next(
        (m for m in reversed(messages) if hasattr(m, "content") and hasattr(m, "type") and m.type == "ai"),
        None,
    )
    answer = last_ai.content if last_ai else str(result)
    print(f"A: {answer}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    from graph_rag import GraphRAG
    from langchain_integration import VSAGraphRetriever, LangChainLLMAdapter

    # Ingest the demo corpus
    print("Ingesting document...")
    rag = GraphRAG(dim=10000)
    n_edges = rag.ingest(DEMO_TEXT)
    print(f"  {n_edges} causal edges extracted, {len(rag.graph.nodes())} nodes indexed.")

    # Build retriever
    retriever = VSAGraphRetriever(graph_rag=rag, top_k=3)

    # Build LangChain LLM
    lc_llm, llm_label = _build_langchain_llm()
    print(f"  LLM: {llm_label}")

    # If no LangChain LLM is available, wire MockLLM through the adapter
    # for surface 1 (retriever), and show MockLLM answers for surfaces 2 & 3.
    if lc_llm is None:
        from pipeline import MockLLM
        from langchain_core.language_models import BaseLLM
        # Patch: use GraphRAG's built-in MockLLM for the chain via a simple wrapper
        class _MockChatLLM:
            """Minimal shim to let MockLLM work as a LangChain-style LLM in the chain."""
            def invoke(self, messages):
                # Extract the human message content
                from langchain_core.messages import AIMessage
                if isinstance(messages, list):
                    content = "\n".join(
                        m.content if hasattr(m, "content") else str(m)
                        for m in messages
                    )
                else:
                    content = str(messages)
                text = MockLLM().generate(content)
                return AIMessage(content=text)

            def stream(self, messages):
                yield self.invoke(messages)

        lc_llm = _MockChatLLM()

    # Run the three surfaces
    demo_retriever(retriever)
    demo_chain(retriever, lc_llm)

    # Agent demo only when a real LLM with tool-calling is available
    if os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        demo_agent(rag, lc_llm)
    else:
        print("\n[skip] Agent demo requires a real LLM key (GROQ_API_KEY or ANTHROPIC_API_KEY).")

    print("\nDone.")


if __name__ == "__main__":
    main()
