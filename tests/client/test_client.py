import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import Response

from client import AgentClient
from schema import ChatHistory, ChatMessage


def test_init(mock_env):
    """Test client initialization with different parameters."""
    # Test default values
    client = AgentClient()
    assert client.base_url == "http://localhost"
    assert client.agent == "research-assistant"
    assert client.timeout is None

    # Test custom values
    client = AgentClient(
        base_url="http://test",
        agent="custom-agent",
        timeout=30.0,
    )
    assert client.base_url == "http://test"
    assert client.agent == "custom-agent"
    assert client.timeout == 30.0


def test_headers(mock_env):
    """Test header generation with and without auth."""
    # Test without auth
    client = AgentClient()
    assert client._headers == {}

    # Test with auth
    with patch.dict(os.environ, {"AUTH_SECRET": "test-secret"}, clear=True):
        client = AgentClient()
        assert client._headers == {"Authorization": "Bearer test-secret"}


def test_invoke(agent_client):
    """Test synchronous invocation."""
    QUESTION = "What is the weather?"
    ANSWER = "The weather is sunny."

    # Mock successful response
    mock_response = Response(
        200,
        json={"type": "ai", "content": ANSWER},
    )
    with patch("httpx.post", return_value=mock_response):
        response = agent_client.invoke(QUESTION)
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER

    # Test with model and thread_id
    with patch("httpx.post", return_value=mock_response) as mock_post:
        response = agent_client.invoke(
            QUESTION,
            model="gpt-4o",
            thread_id="test-thread",
        )
        assert isinstance(response, ChatMessage)
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["message"] == QUESTION
        assert kwargs["json"]["model"] == "gpt-4o"
        assert kwargs["json"]["thread_id"] == "test-thread"

    # Test error response
    error_response = Response(500, text="Internal Server Error")
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(Exception) as exc:
            agent_client.invoke(QUESTION)
        assert "Error: 500" in str(exc.value)


@pytest.mark.asyncio
async def test_ainvoke(agent_client):
    """Test asynchronous invocation."""
    QUESTION = "What is the weather?"
    ANSWER = "The weather is sunny."

    # Test successful response
    mock_response = Response(200, json={"type": "ai", "content": ANSWER})
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        response = await agent_client.ainvoke(QUESTION)
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER

    # Test with model and thread_id
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        response = await agent_client.ainvoke(
            QUESTION,
            model="gpt-4o",
            thread_id="test-thread",
        )
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["message"] == QUESTION
        assert kwargs["json"]["model"] == "gpt-4o"
        assert kwargs["json"]["thread_id"] == "test-thread"

    # Test error response
    with patch("httpx.AsyncClient.post", return_value=Response(500, text="Internal Server Error")):
        with pytest.raises(Exception) as exc:
            await agent_client.ainvoke(QUESTION)
        assert "Error: 500" in str(exc.value)


def test_stream(agent_client):
    """Test synchronous streaming."""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # Create mock response with streaming events
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [
            f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"
        ]
        + ["data: [DONE]"]
    )

    # Mock the streaming response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = events
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)

    with patch("httpx.stream", return_value=mock_response):
        # Collect all streamed responses
        responses = list(agent_client.stream(QUESTION))

        # Verify tokens were streamed
        assert len(responses) == len(TOKENS) + 1  # tokens + final message
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # Verify final message
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # Test error response
    error_response = Mock()
    error_response.status_code = 500
    error_response.text = "Internal Server Error"
    error_response.__enter__ = Mock(return_value=error_response)
    error_response.__exit__ = Mock(return_value=None)
    with patch("httpx.stream", return_value=error_response):
        with pytest.raises(Exception) as exc:
            list(agent_client.stream(QUESTION))
        assert "Error: 500" in str(exc.value)


@pytest.mark.asyncio
async def test_astream(agent_client):
    """Test asynchronous streaming."""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # Create mock response with streaming events
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [
            f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"
        ]
        + ["data: [DONE]"]
    )

    # Create an async iterator for the events
    async def async_events():
        for event in events:
            yield event

    # Mock the streaming response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = Mock(return_value=async_events())
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = Mock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Collect all streamed responses
        responses = []
        async for response in agent_client.astream(QUESTION):
            responses.append(response)

        # Verify tokens were streamed
        assert len(responses) == len(TOKENS) + 1  # tokens + final message
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # Verify final message
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # Test error response
    error_response = AsyncMock()
    error_response.status_code = 500
    error_response.text = "Internal Server Error"
    error_response.__aenter__ = AsyncMock(return_value=error_response)

    mock_client.stream.return_value = error_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(Exception) as exc:
            async for _ in agent_client.astream(QUESTION):
                pass
        assert "Error: 500" in str(exc.value)


@pytest.mark.asyncio
async def test_acreate_feedback(agent_client):
    """Test asynchronous feedback creation."""
    RUN_ID = "test-run"
    KEY = "test-key"
    SCORE = 0.8
    KWARGS = {"comment": "Great response!"}

    # Test successful response
    with patch("httpx.AsyncClient.post", return_value=Response(200, json={})) as mock_post:
        await agent_client.acreate_feedback(RUN_ID, KEY, SCORE, KWARGS)
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["run_id"] == RUN_ID
        assert kwargs["json"]["key"] == KEY
        assert kwargs["json"]["score"] == SCORE
        assert kwargs["json"]["kwargs"] == KWARGS

    # Test error response
    with patch("httpx.AsyncClient.post", return_value=Response(500, text="Internal Server Error")):
        with pytest.raises(Exception) as exc:
            await agent_client.acreate_feedback(RUN_ID, KEY, SCORE)
        assert "Error: 500" in str(exc.value)


def test_get_history(agent_client):
    """Test chat history retrieval."""
    THREAD_ID = "test-thread"
    HISTORY = {
        "messages": [
            {"type": "human", "content": "What is the weather?"},
            {"type": "ai", "content": "The weather is sunny."},
        ]
    }

    # Mock successful response
    mock_response = Response(200, json=HISTORY)
    with patch("httpx.post", return_value=mock_response):
        history = agent_client.get_history(THREAD_ID)
        assert isinstance(history, ChatHistory)
        assert len(history.messages) == 2
        assert history.messages[0].type == "human"
        assert history.messages[1].type == "ai"

    # Test error response
    error_response = Response(500, text="Internal Server Error")
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(Exception) as exc:
            agent_client.get_history(THREAD_ID)
        assert "Error: 500" in str(exc.value)
