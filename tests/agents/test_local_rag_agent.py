from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.local_rag_agent import generate_response, retrieve_documents
from knowledge import SearchResult


@pytest.mark.asyncio
async def test_retrieve_documents_passes_agent_config_to_knowledge_base():
    match = SearchResult(
        document_id="document-1",
        source="policy.md",
        page=2,
        content="Remote work policy",
        score=0.2,
        metadata={},
    )
    state = {"messages": [HumanMessage(content="What is the policy?")]}
    config = RunnableConfig(configurable={"knowledge_base_id": "handbook", "top_k": 3})

    with patch("agents.local_rag_agent.search_documents", return_value=[match]) as search:
        result = await retrieve_documents(state, config)

    assert result["retrieved_documents"][0]["source"] == "policy.md"
    search.assert_called_once_with("handbook", "What is the policy?", 3)


@pytest.mark.asyncio
async def test_generate_response_appends_sources():
    fake_model = SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="The policy says yes."))
    )
    state = {
        "messages": [HumanMessage(content="What is the policy?")],
        "retrieved_documents": [
            {"source": "policy.md", "page": 2, "content": "Remote work policy"}
        ],
    }
    config = RunnableConfig(configurable={"model": "fake"})

    with patch("agents.local_rag_agent.get_model", return_value=fake_model):
        result = await generate_response(state, config)

    assert "The policy says yes." in result["messages"][0].content
    assert "Sources:" in result["messages"][0].content
    assert "policy.md, page 2" in result["messages"][0].content
