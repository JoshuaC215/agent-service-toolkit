from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response

from schema import KnowledgeDocumentResponse, KnowledgeSearchResponse


def test_upload_knowledge_document(agent_client):
    response = Response(
        201,
        json={
            "knowledge_base_id": "handbook",
            "document_id": "document-1",
            "filename": "policy.md",
            "content_type": "text/markdown",
            "size_bytes": 12,
            "chunk_count": 1,
        },
        request=Request("POST", "http://test/knowledge-bases/handbook/documents"),
    )
    with patch("httpx.post", return_value=response) as post:
        result = agent_client.upload_knowledge_document(
            "handbook", "policy.md", b"hello policy", "text/markdown"
        )

    assert isinstance(result, KnowledgeDocumentResponse)
    assert result.document_id == "document-1"
    assert post.call_args.kwargs["files"]["file"] == (
        "policy.md",
        b"hello policy",
        "text/markdown",
    )


@pytest.mark.asyncio
async def test_async_search_knowledge_base(agent_client):
    response = Response(
        200,
        json={
            "knowledge_base_id": "handbook",
            "query": "remote work",
            "results": [
                {
                    "document_id": "document-1",
                    "source": "policy.md",
                    "page": 1,
                    "content": "Remote work policy",
                    "score": 0.1,
                    "metadata": {},
                }
            ],
        },
        request=Request("GET", "http://test/knowledge-bases/handbook/search"),
    )
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get = AsyncMock(return_value=response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await agent_client.asearch_knowledge_base("handbook", "remote work", 3)

    assert isinstance(result, KnowledgeSearchResponse)
    assert result.results[0].source == "policy.md"
    mock_client.get.assert_awaited_once()
    assert mock_client.get.call_args.kwargs["params"] == {"q": "remote work", "k": 3}
