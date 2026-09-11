from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph

from core import get_model, settings
from knowledge import KnowledgeBaseError, SearchResult, search_documents


class LocalRagState(MessagesState, total=False):
    """State used by the local document retrieval agent."""

    retrieved_documents: list[dict[str, Any]]


def _last_user_query(state: LocalRagState) -> str:
    human_messages = [message for message in state["messages"] if isinstance(message, HumanMessage)]
    if not human_messages:
        raise KnowledgeBaseError("A user message is required for knowledge-base search.")
    content = human_messages[-1].content
    return content if isinstance(content, str) else str(content)


async def retrieve_documents(state: LocalRagState, config: RunnableConfig) -> LocalRagState:
    configurable = config.get("configurable", {})
    knowledge_base_id = configurable.get("knowledge_base_id")
    if not isinstance(knowledge_base_id, str) or not knowledge_base_id:
        raise KnowledgeBaseError(
            "knowledge_base_id is required in agent_config for local-rag-agent."
        )

    top_k = configurable.get("top_k", settings.KNOWLEDGE_DEFAULT_TOP_K)
    try:
        top_k = max(1, min(20, int(top_k)))
    except (TypeError, ValueError) as exc:
        raise KnowledgeBaseError("agent_config.top_k must be an integer between 1 and 20.") from exc

    matches: list[SearchResult] = await asyncio.to_thread(
        search_documents,
        knowledge_base_id,
        _last_user_query(state),
        top_k,
    )
    return {
        "retrieved_documents": [
            {
                "document_id": match.document_id,
                "source": match.source,
                "page": match.page,
                "content": match.content,
                "score": match.score,
            }
            for match in matches
        ],
        "messages": [],
    }


def _context_prompt(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return (
            "No relevant documents were found. Tell the user that the knowledge base has no answer."
        )

    sections = []
    for index, document in enumerate(documents, start=1):
        page = document.get("page")
        location = f"{document.get('source', 'unknown')}"
        if page is not None:
            location += f", page {page}"
        sections.append(f"[Source {index}: {location}]\n{document.get('content', '')}")
    return "\n\n".join(sections)


def _source_footer(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return ""
    citations = []
    seen: set[tuple[str, Any]] = set()
    for document in documents:
        source = str(document.get("source", "unknown"))
        page = document.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        location = source if page is None else f"{source}, page {page}"
        citations.append(f"- {location}")
    return "\n\nSources:\n" + "\n".join(citations)


async def generate_response(state: LocalRagState, config: RunnableConfig) -> LocalRagState:
    documents = state.get("retrieved_documents", [])
    system_prompt = (
        "You are a retrieval-augmented assistant. Answer using only the supplied document "
        "context. If the context does not contain the answer, say so clearly instead of "
        "guessing. Keep the answer concise and mention the relevant source name and page "
        "when making factual claims.\n\nDocument context:\n"
        f"{_context_prompt(documents)}"
    )
    model: BaseChatModel = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await model.ainvoke(
        [SystemMessage(content=system_prompt), *state["messages"]], config
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    return {"messages": [AIMessage(content=content + _source_footer(documents))]}


graph = StateGraph(LocalRagState)
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("generate_response", generate_response)
graph.set_entry_point("retrieve_documents")
graph.add_edge("retrieve_documents", "generate_response")
graph.add_edge("generate_response", END)

local_rag_agent = graph.compile()
