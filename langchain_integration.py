"""
langchain_integration.py
========================
LangChain-compatible wrappers for the VSA-RAG causal graph engine.

Four surfaces:

  VSAGraphRetriever   — BaseRetriever that converts ChainResult objects into
                        LangChain Documents.  Drop into any LCEL chain,
                        RetrievalQA, or agent that accepts a retriever.

  LangChainLLMAdapter — Wraps any LangChain BaseChatModel / BaseLLM so it can
                        be passed as llm= to GraphRAG and also used as the
                        llm_extractor= for LLM-assisted graph building.

  build_rag_chain()   — Ready-made LCEL chain.
                        Plain mode:    retriever | prompt | llm | parser
                        Summarize mode (borrowed from CausalRAG):
                          retriever | summary_prompt | llm | answer_prompt | llm | parser
                        Returns a Runnable; call .invoke(question) or .stream(question).

  build_rag_tool()    — LangChain Tool wrapping GraphRAG.answer() for agents.

Quick start
-----------
    from langchain_groq import ChatGroq
    from graph_rag import GraphRAG
    from langchain_integration import (
        VSAGraphRetriever, LangChainLLMAdapter, build_rag_chain
    )

    lc_llm  = ChatGroq(model="llama-3.1-8b-instant")
    adapter = LangChainLLMAdapter(lc_llm)

    rag = GraphRAG(llm=adapter)

    # Standard spaCy extraction (free, deterministic):
    rag.ingest(text)

    # OR: LLM-assisted extraction for academic / implicit causality
    #     (borrowed from CausalRAG — doubles edge count on complex documents):
    # rag.ingest(text, llm_extractor=adapter, llm_mode="augment")  # fills gaps
    # rag.ingest(text, llm_extractor=adapter, llm_mode="full")     # all sentences

    retriever = VSAGraphRetriever(graph_rag=rag, top_k=3)

    # Plain chain (1 LLM call per query):
    chain = build_rag_chain(retriever, lc_llm)

    # Causal-summary chain (2 LLM calls, tighter answers on multi-hop queries):
    # chain = build_rag_chain(retriever, lc_llm, summarize=True)

    print(chain.invoke("What caused the emergency shutdown?"))
"""

from __future__ import annotations

from typing import Any, List, Optional

# --------------------------------------------------------------------------- #
#  LangChain imports (langchain-core only — no heavy langchain dep required
#  for the retriever and adapter; the chain builder needs langchain-core too)
# --------------------------------------------------------------------------- #
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from pydantic import ConfigDict
    _LC_AVAILABLE = True
except ImportError as _lc_err:
    _LC_AVAILABLE = False
    _LC_MISSING = str(_lc_err)


def _require_lc() -> None:
    if not _LC_AVAILABLE:
        raise ImportError(
            f"langchain-core is required for this module. "
            f"Install it with: pip install langchain-core\n"
            f"Original error: {_LC_MISSING}"
        )


# --------------------------------------------------------------------------- #
#  VSAGraphRetriever — BaseRetriever wrapper
# --------------------------------------------------------------------------- #
if _LC_AVAILABLE:
    class VSAGraphRetriever(BaseRetriever):
        """
        Wraps GraphRAG as a LangChain BaseRetriever.

        Each retrieved ChainResult becomes a Document:
          page_content  — human-readable causal chain (e.g. "A ->(cause) B ->(trigger) C")
          metadata      — entry_node, direction, rrf_score, rerank_score, provenance

        Parameters
        ----------
        graph_rag : GraphRAG
            An already-ingested GraphRAG instance.
        top_k : int
            Number of causal chains to return per query (default 3).
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        graph_rag: Any        # GraphRAG instance
        top_k: int = 3

        def _make_docs(self, chains: list) -> List[Document]:
            docs = []
            for chain in chains:
                chain_text = chain.text()        # "A ->(cause) B ->(trigger) C"
                sources = chain.provenance()
                # Include BOTH the structured chain path AND the raw source sentences.
                # The chain arrows give the LLM the causal structure;
                # the sources give it verbatim evidence to quote from.
                source_block = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(sources))
                content = f"Chain: {chain_text}\nEvidence:\n{source_block}".strip()

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "entry_node":    chain.entry_node,
                        "direction":     chain.direction,
                        "rrf_score":     round(chain.rrf_score, 4),
                        "rerank_score":  round(chain.rerank_score, 4),
                        "provenance":    sources,
                        "chain_length":  len(chain.chain),
                    },
                ))
            return docs

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            return self._make_docs(self.graph_rag.retrieve(query, top_k=self.top_k))

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            # GraphRAG.retrieve is sync; run in executor to avoid blocking the event loop
            import asyncio
            loop = asyncio.get_event_loop()
            chains = await loop.run_in_executor(
                None, lambda: self.graph_rag.retrieve(query, top_k=self.top_k)
            )
            return self._make_docs(chains)

else:
    # Stub so the module is importable even without langchain-core
    class VSAGraphRetriever:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_lc()


# --------------------------------------------------------------------------- #
#  LangChainLLMAdapter — use any LC LLM inside GraphRAG
# --------------------------------------------------------------------------- #
class LangChainLLMAdapter:
    """
    Wraps a LangChain BaseChatModel or BaseLLM so it can be passed as the
    llm= argument to GraphRAG (which expects a .generate(prompt: str) -> str
    interface).

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

_HUMAN_TEMPLATE = (
    "Evidence:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

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
    llm       : any LangChain BaseChatModel or BaseLLM
    summarize : bool (default False)
        When True, adds a causal-summary compression step before the final
        generation (borrowed from CausalRAG).  Uses 2 LLM calls per query
        instead of 1, but produces tighter answers on multi-hop chains.

    Returns
    -------
    A LangChain Runnable.  Call .invoke(question) or .stream(question).

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
        # Pass full page_content: structured chain path + numbered evidence sentences.
        # This gives the LLM both the causal structure AND verbatim source text.
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

        chain = (
            {
                "context":  retriever | RunnableLambda(_format_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(_summarize)
            | answer_prompt
            | llm
            | StrOutputParser()
        )
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human",  _HUMAN_TEMPLATE),
        ])
        chain = (
            {
                "context":  retriever | RunnableLambda(_format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    return chain


# --------------------------------------------------------------------------- #
#  build_rag_tool — LangChain Tool for use in agents
# --------------------------------------------------------------------------- #
def build_rag_tool(graph_rag: Any, name: str = "causal_graph_search",
                   summarize: bool = False) -> Any:
    """
    Wrap GraphRAG.answer() as a LangChain Tool for use in tool-calling agents.

    The tool accepts a free-text question and returns the LLM-synthesised
    answer over the top causal chains — the same output as rag.answer().

    Parameters
    ----------
    graph_rag : GraphRAG
        An already-ingested GraphRAG instance.
    name : str
        Tool name visible to the agent (default "causal_graph_search").

    Returns
    -------
    langchain_core.tools.Tool

    Example (langchain 1.x)
    -----------------------
        from langchain.agents import create_agent

        tool  = build_rag_tool(rag)
        agent = create_agent(
            llm, [tool],
            system_prompt="You are a causal reasoning assistant.",
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Why did the power outage happen?"}]}
        )
        print(result["messages"][-1].content)
    """
    _require_lc()
    from langchain_core.tools import Tool

    def _run(query: str) -> str:
        answer, chains = graph_rag.answer(query, top_k=3, summarize=summarize)
        if not chains:
            return "No causal chains found for this query."
        chain_summary = "\n".join(
            f"  Chain {i}: {c.text()}" for i, c in enumerate(chains, 1)
        )
        return f"{answer}\n\nSupporting chains:\n{chain_summary}"

    return Tool(
        name=name,
        func=_run,
        description=(
            "Search a causal knowledge graph built from the ingested document. "
            "Input: a natural-language question about causes or effects. "
            "Output: a synthesised answer and the supporting causal chains. "
            "Use this tool when the question involves WHY something happened, "
            "WHAT caused something, or WHAT will result from something."
        ),
    )
