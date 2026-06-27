"""
langchain_integration.py
========================
LangChain-compatible wrappers for the VSA-RAG causal graph engine.

Five surfaces:

  VSAGraphRetriever     — BaseRetriever with three retrieval modes:
                           "chains"   (default) — structured causal chains as Documents
                           "hybrid"   — chains + coverage sentences (matches answer() quality)
                           "coverage" — flat coverage sentences only (like standard RAG)

  LangChainLLMAdapter   — Wraps any LangChain BaseChatModel / BaseLLM so it can
                          be passed as llm= to GraphRAG and also used as the
                          llm_extractor= for LLM-assisted graph building.

  build_rag_chain()     — Ready-made LCEL chain.
                          Plain mode:    retriever | prompt | llm | parser
                          Summarize mode (borrowed from CausalRAG):
                            retriever | summary_prompt | llm | answer_prompt | llm | parser
                          Returns a Runnable; call .invoke(question) or .stream(question).

  build_rag_tool()      — LangChain Tool wrapping GraphRAG.answer() for agents.

  build_graph_tools()   — Returns three pure-graph StructuredTools (no LLM, instant):
                           causal_rootcause — backward chain from an effect node
                           causal_impact    — forward chain from a cause node
                           causal_path      — shortest path between two nodes

Quick start
-----------
    from langchain_groq import ChatGroq
    from graph_rag import GraphRAG
    from langchain_integration import (
        VSAGraphRetriever, LangChainLLMAdapter, build_rag_chain, build_graph_tools
    )

    lc_llm  = ChatGroq(model="llama-3.1-8b-instant")
    adapter = LangChainLLMAdapter(lc_llm)

    rag = GraphRAG(llm=adapter)
    rag.ingest(text)

    # Hybrid retriever: chains + coverage sentences (recommended for LCEL chains)
    retriever = VSAGraphRetriever(graph_rag=rag, top_k=3, mode="hybrid")
    chain = build_rag_chain(retriever, lc_llm)
    print(chain.invoke("What caused the emergency shutdown?"))

    # Pure-graph tools for agents — no LLM, instant, free
    tools = build_graph_tools(rag) + [build_rag_tool(rag)]
    # tools: [causal_rootcause, causal_impact, causal_path, causal_graph_search]
"""

from __future__ import annotations

import re as _re
from typing import Any, Dict, List, Optional

_WORD_TOK = _re.compile(r"\w+", _re.UNICODE)   # unicode-aware: keeps accented words whole


def _simple_tokenize(s: str) -> List[str]:
    return _WORD_TOK.findall(s.lower())


# --------------------------------------------------------------------------- #
#  LangChain imports — langchain-core only; graceful stub if not installed
# --------------------------------------------------------------------------- #
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import (
        CallbackManagerForRetrieverRun,
        AsyncCallbackManagerForRetrieverRun,
    )
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.tools import StructuredTool, Tool
    from pydantic import BaseModel, ConfigDict, Field
    _LC_AVAILABLE = True
except ImportError as _lc_err:
    _LC_AVAILABLE = False
    _LC_MISSING = str(_lc_err)


def _require_lc() -> None:
    if not _LC_AVAILABLE:
        raise ImportError(
            "langchain-core is required for this module. "
            "Install it with: pip install langchain-core\n"
            f"Original error: {_LC_MISSING}"
        )


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #

def _find_node(graph_rag: Any, query_str: str) -> Optional[str]:
    """Return the graph node with the highest token overlap with query_str."""
    q_terms = set(_simple_tokenize(query_str))
    try:
        widx = graph_rag.graph.word_index()
    except AttributeError:
        return None
    candidates: Dict[str, int] = {}
    for term in q_terms:
        for node in widx.get(term, []):
            candidates[node] = candidates.get(node, 0) + 1
    return max(candidates, key=lambda n: candidates[n]) if candidates else None


# --------------------------------------------------------------------------- #
#  VSAGraphRetriever — BaseRetriever wrapper
# --------------------------------------------------------------------------- #
if _LC_AVAILABLE:
    class VSAGraphRetriever(BaseRetriever):
        """
        Wraps GraphRAG as a LangChain BaseRetriever.

        mode="chains"   — one Document per causal chain (default, backward-compatible)
        mode="hybrid"   — chain Documents + coverage sentence Documents
                          Matches the full answer() retrieval quality; recommended
                          for LCEL chains that do their own generation step.
        mode="coverage" — flat coverage sentences only (standard RAG behaviour)

        Chain Document schema
        ---------------------
        page_content : "Chain: A ->(cause) B\\nEvidence:\\n  [1] ..."
        metadata     : {
            type             : "chain",
            entry_node       : str,
            direction        : "forward" | "backward",
            rrf_score        : float,
            rerank_score     : float,
            hop_count        : int,         # number of edges in the chain
            chain_confidence : float,       # mean edge confidence
            chain_polarity   : int,         # +1 positive, -1 negative causation
            provenance       : List[str],   # raw source sentences
        }

        Coverage Document schema
        ------------------------
        page_content : raw sentence text
        metadata     : {type: "coverage"}
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        graph_rag: Any
        top_k: int = 3
        mode: str = "chains"   # "chains" | "hybrid" | "coverage"

        def _chain_docs(self, chains: list) -> List[Document]:
            docs = []
            for c in chains:
                chain_text = c.text()
                sources = c.provenance()
                source_block = "\n".join(
                    f"  [{i+1}] {s}" for i, s in enumerate(sources)
                )
                content = f"Chain: {chain_text}\nEvidence:\n{source_block}".strip()

                if c.chain:
                    conf = sum(e.confidence for e in c.chain) / len(c.chain)
                    polarity = 1
                    for e in c.chain:
                        polarity *= e.polarity
                else:
                    conf, polarity = 1.0, 1

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "type":             "chain",
                        "entry_node":       c.entry_node,
                        "direction":        c.direction,
                        "rrf_score":        round(c.rrf_score, 4),
                        "rerank_score":     round(c.rerank_score, 4),
                        "hop_count":        len(c.chain),
                        "chain_confidence": round(conf, 4),
                        "chain_polarity":   int(polarity),
                        "provenance":       sources,
                    },
                ))
            return docs

        def _coverage_docs(self, sentences: List[str]) -> List[Document]:
            return [
                Document(page_content=s, metadata={"type": "coverage"})
                for s in sentences
            ]

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            if self.mode == "coverage":
                sents = self.graph_rag._retrieve_sentences(
                    query, k=self.top_k * 2
                )
                return self._coverage_docs(sents)

            chains = self.graph_rag.retrieve(query, top_k=self.top_k)
            docs = self._chain_docs(chains)

            if self.mode == "hybrid":
                chain_nodes = {
                    n for c in chains
                    for e in c.chain
                    for n in (e.cause, e.effect)
                }
                sents = self.graph_rag._retrieve_sentences(
                    query,
                    k=self.top_k * 2,
                    chain_nodes=chain_nodes or None,
                )
                docs = docs + self._coverage_docs(sents)

            return docs

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._get_relevant_documents(query)
            )

else:
    class VSAGraphRetriever:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_lc()


# --------------------------------------------------------------------------- #
#  LangChainLLMAdapter — use any LC LLM inside GraphRAG
# --------------------------------------------------------------------------- #
class LangChainLLMAdapter:
    """
    Wraps a LangChain BaseChatModel or BaseLLM so it can be passed as the
    llm= or llm_extractor= argument to GraphRAG.

    Example
    -------
        from langchain_groq import ChatGroq
        from graph_rag import GraphRAG
        from langchain_integration import LangChainLLMAdapter

        lc_llm  = ChatGroq(model="llama-3.1-8b-instant")
        adapter = LangChainLLMAdapter(lc_llm)
        rag     = GraphRAG(llm=adapter)
        rag.ingest(text)
        answer, chains = rag.answer("What caused the shutdown?")
    """

    def __init__(self, lc_llm: Any) -> None:
        _require_lc()
        self._llm = lc_llm

    def generate(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage, BaseMessage
        result = self._llm.invoke([HumanMessage(content=prompt)])
        if isinstance(result, BaseMessage):
            return str(result.content)
        return str(result)


# --------------------------------------------------------------------------- #
#  build_rag_chain — LCEL convenience chain
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are a causal reasoning assistant. "
    "Answer using ONLY the evidence provided. "
    "Respect the causal direction (cause → effect). "
    "Be direct and concise — one to three sentences. "
    "Do not reference chain numbers or labels."
)

_HUMAN_TEMPLATE = "Evidence:\n{context}\n\nQuestion: {question}\n\nAnswer:"

_SUMMARY_SYSTEM = (
    "You are a causal reasoning assistant. "
    "Given the causal evidence below, write a single concise paragraph (3-5 sentences) "
    "that summarises the key cause-effect relationships relevant to the question. "
    "Preserve causal direction. Do NOT add information not in the evidence."
)

_SUMMARY_HUMAN = (
    "Causal evidence:\n{context}\n\nQuestion: {question}\n\nCausal summary:"
)

_ANSWER_FROM_SUMMARY_SYSTEM = (
    "You are a causal reasoning assistant. "
    "Answer the question using ONLY the causal summary provided. "
    "Be direct and concise."
)

_ANSWER_FROM_SUMMARY_HUMAN = (
    "Causal summary:\n{summary}\n\nQuestion: {question}\n\nAnswer:"
)


def build_rag_chain(retriever: Any, llm: Any, summarize: bool = False) -> Any:
    """
    Build a ready-to-use LCEL chain.

    Parameters
    ----------
    retriever : VSAGraphRetriever (or any BaseRetriever)
                Use mode="hybrid" for best results — gives the LLM both
                causal chains and coverage sentences.
    llm       : any LangChain BaseChatModel or BaseLLM
    summarize : bool (default False)
                When True, adds a causal-summary compression step before
                final generation (borrowed from CausalRAG). Uses 2 LLM
                calls per query but produces tighter answers on multi-hop chains.

    Returns
    -------
    A LangChain Runnable. Call .invoke(question) or .stream(question).

    Examples
    --------
        # Plain (1 LLM call):
        chain = build_rag_chain(retriever, llm)
        print(chain.invoke("What caused the power outage?"))

        # With causal-summary step (2 LLM calls, better on complex queries):
        chain = build_rag_chain(retriever, llm, summarize=True)
        print(chain.invoke("How did the sensor failure ultimately affect operations?"))
    """
    _require_lc()

    def _format_docs(docs: List[Document]) -> str:
        if not docs:
            return "No causal evidence found."
        return "\n\n".join(doc.page_content for doc in docs)

    if summarize:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", _SUMMARY_SYSTEM),
            ("human",  _SUMMARY_HUMAN),
        ])
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", _ANSWER_FROM_SUMMARY_SYSTEM),
            ("human",  _ANSWER_FROM_SUMMARY_HUMAN),
        ])

        def _summarize(inputs: dict) -> dict:
            summary = (summary_prompt | llm | StrOutputParser()).invoke(inputs)
            return {"summary": summary, "question": inputs["question"]}

        return (
            {
                "context":  retriever | RunnableLambda(_format_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(_summarize)
            | answer_prompt
            | llm
            | StrOutputParser()
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human",  _HUMAN_TEMPLATE),
    ])
    return (
        {
            "context":  retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# --------------------------------------------------------------------------- #
#  build_rag_tool — LangChain Tool for use in agents
# --------------------------------------------------------------------------- #
def build_rag_tool(graph_rag: Any, name: str = "causal_graph_search",
                   summarize: bool = False) -> Any:
    """
    Wrap GraphRAG.answer() as a LangChain Tool for use in tool-calling agents.

    For specialised no-LLM graph traversal, see build_graph_tools().

    Parameters
    ----------
    graph_rag : GraphRAG — an already-ingested GraphRAG instance.
    name      : str — tool name visible to the agent.
    summarize : bool — pass-through to GraphRAG.answer(summarize=).

    Returns
    -------
    langchain_core.tools.Tool
    """
    _require_lc()

    def _run(query: str) -> str:
        answer_text, chains = graph_rag.answer(query, top_k=3, summarize=summarize)
        if not chains:
            return "No causal chains found for this query."
        chain_lines = "\n".join(
            f"  Chain {i}: {c.text()}" for i, c in enumerate(chains, 1)
        )
        return f"{answer_text}\n\nSupporting chains:\n{chain_lines}"

    return Tool(
        name=name,
        func=_run,
        description=(
            "Search a causal knowledge graph built from the ingested document. "
            "Input: a natural-language question about causes or effects. "
            "Output: a synthesised answer and the supporting causal chains. "
            "Use when the question involves WHY something happened, "
            "WHAT caused something, or WHAT will result from something."
        ),
    )


# --------------------------------------------------------------------------- #
#  build_graph_tools — three pure-graph StructuredTools (no LLM, instant)
# --------------------------------------------------------------------------- #
def build_graph_tools(graph_rag: Any) -> List[Any]:
    """
    Return three StructuredTools for pure causal-graph traversal.

    These tools traverse the graph directly — no LLM call, no latency, free.
    Pair them with build_rag_tool() for a complete agent tool-belt.

    Tools
    -----
    causal_rootcause  — backward chain: trace what caused a given effect
    causal_impact     — forward chain: trace downstream impacts of a cause
    causal_path       — shortest causal path between two events

    Example
    -------
        tools = build_graph_tools(rag) + [build_rag_tool(rag)]
        agent = create_react_agent(llm, tools)
        agent.invoke({"messages": [
            {"role": "user", "content": "What caused the reactor to overheat?"}
        ]})
    """
    _require_lc()

    class _RootCauseInput(BaseModel):
        effect: str = Field(
            description="The effect or outcome to trace root causes for"
        )

    class _ImpactInput(BaseModel):
        cause: str = Field(
            description="The cause or event to trace downstream impacts for"
        )

    class _PathInput(BaseModel):
        source: str = Field(description="The starting cause or event")
        target: str = Field(description="The target effect or outcome")

    def _fmt_paths(paths: list, label: str, node: str) -> str:
        if not paths:
            return f"No {label} chains found for '{node}'."
        from causal_graph import CausalGraph
        lines = [
            f"  {i}. {CausalGraph.chain_text(p)}"
            for i, p in enumerate(paths[:5], 1)
        ]
        if len(paths) > 5:
            lines.append(f"  ... and {len(paths) - 5} more chains.")
        return "\n".join(lines)

    def _rootcause(effect: str) -> str:
        node = _find_node(graph_rag, effect)
        if node is None:
            return f"No graph node found matching '{effect}'."
        paths = graph_rag.graph.backward_chain(node, max_depth=6)
        return f"Root causes of '{node}':\n" + _fmt_paths(paths, "root-cause", node)

    def _impact(cause: str) -> str:
        node = _find_node(graph_rag, cause)
        if node is None:
            return f"No graph node found matching '{cause}'."
        paths = graph_rag.graph.forward_chain(node, max_depth=6)
        return (
            f"Downstream impacts of '{node}':\n"
            + _fmt_paths(paths, "impact", node)
        )

    def _path(source: str, target: str) -> str:
        src_node = _find_node(graph_rag, source)
        tgt_node = _find_node(graph_rag, target)
        if src_node is None:
            return f"No graph node found matching '{source}'."
        if tgt_node is None:
            return f"No graph node found matching '{target}'."
        from causal_graph import CausalGraph
        path = graph_rag.graph.path_between(src_node, tgt_node, max_depth=8)
        if path is None:
            return (
                f"No causal path found between '{src_node}' and '{tgt_node}'. "
                "Try causal_rootcause or causal_impact to explore each node separately."
            )
        return (
            f"Causal path from '{src_node}' to '{tgt_node}':\n"
            f"  {CausalGraph.chain_text(path)}"
        )

    rootcause_tool = StructuredTool.from_function(
        func=_rootcause,
        name="causal_rootcause",
        description=(
            "Trace the root causes of an effect by traversing the causal graph backward. "
            "Input: the effect or outcome to investigate. "
            "Returns up to 5 causal chains that explain why it happened. "
            "Use when asked WHY something happened or what caused an outcome. "
            "No LLM call — instant graph traversal."
        ),
        args_schema=_RootCauseInput,
    )

    impact_tool = StructuredTool.from_function(
        func=_impact,
        name="causal_impact",
        description=(
            "Trace the downstream impacts of a cause by traversing the causal graph forward. "
            "Input: the cause or event to trace. "
            "Returns what the cause leads to (blast radius). "
            "Use when asked WHAT WILL HAPPEN, what is affected, or the impact of an event. "
            "No LLM call — instant graph traversal."
        ),
        args_schema=_ImpactInput,
    )

    path_tool = StructuredTool.from_function(
        func=_path,
        name="causal_path",
        description=(
            "Find the shortest causal path connecting two events in the graph. "
            "Inputs: source (a cause or event) and target (an effect or outcome). "
            "Use when asked HOW event A is connected to or led to event B. "
            "No LLM call — instant graph traversal."
        ),
        args_schema=_PathInput,
    )

    return [rootcause_tool, impact_tool, path_tool]
