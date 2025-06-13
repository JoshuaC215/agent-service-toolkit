import asyncio
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from ag_ui.core import EventType
from ag_ui.encoder import EventEncoder
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.messages import ChatMessage as LangchainChatMessage
from langgraph.types import Interrupt

from service.utils import convert_message_to_agui_events


@pytest.fixture
def encoder():
    return EventEncoder()

def run_async_gen(gen):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(_collect_async_gen(gen))

def _collect_async_gen(gen):
    result = []
    async def collect():
        async for item in gen:
            result.append(item)
        return result
    return collect()

# ============ Unit Tests ============

def test_ai_message_with_tool_and_text(encoder):
    msg = AIMessage(
        content="hello",
        tool_calls=[{"name": "Calculator", "args": {"x": 1, "y": 2}, "id": str(uuid4())}]
    )
    events = run_async_gen(convert_message_to_agui_events(msg, encoder))
    assert any(EventType.TOOL_CALL_START.value in e for e in events)
    assert any(EventType.TOOL_CALL_ARGS.value in e for e in events)
    assert any(EventType.TOOL_CALL_END.value in e for e in events)
    assert any(EventType.TEXT_MESSAGE_START.value in e for e in events)
    assert any(EventType.TEXT_MESSAGE_CONTENT.value in e for e in events)
    assert any(EventType.TEXT_MESSAGE_END.value in e for e in events)

def test_ai_message_only_text(encoder):
    msg = AIMessage(content="just text", tool_calls=[])
    events = run_async_gen(convert_message_to_agui_events(msg, encoder))
    assert any(EventType.TEXT_MESSAGE_START.value in e for e in events)
    assert any(EventType.TEXT_MESSAGE_CONTENT.value in e for e in events)
    assert any(EventType.TEXT_MESSAGE_END.value in e for e in events)
    assert not any(EventType.TOOL_CALL_START.value in e for e in events)

def test_ai_message_only_tool(encoder):
    msg = AIMessage(content="", tool_calls=[{"name": "Search", "args": {"q": "foo"}, "id": str(uuid4())}])
    events = run_async_gen(convert_message_to_agui_events(msg, encoder))
    assert any(EventType.TOOL_CALL_START.value in e for e in events)
    assert any(EventType.TOOL_CALL_ARGS.value in e for e in events)
    assert any(EventType.TOOL_CALL_END.value in e for e in events)
    assert not any(EventType.TEXT_MESSAGE_START.value in e for e in events)

def test_tool_message_no_event(encoder):
    msg = ToolMessage(content="result", name="Calculator", tool_call_id=str(uuid4()))
    events = run_async_gen(convert_message_to_agui_events(msg, encoder))
    assert len(events) == 0

def test_custom_langchain_message(encoder):
    custom_data = {"name": "my_custom", "foo": 123}
    msg = LangchainChatMessage(role="custom", content=[custom_data])
    events = run_async_gen(convert_message_to_agui_events(msg, encoder))
    assert any(EventType.CUSTOM.value in e for e in events)
    assert any("my_custom" in e for e in events)

def test_human_message_to_raw_event(encoder):
    msg = HumanMessage(content="user input")
    events = run_async_gen(convert_message_to_agui_events(msg, encoder))
    assert any(EventType.RAW.value in e for e in events)
    assert any("user input" in e for e in events)

class Dummy: 
    pass

def test_invalid_message_to_raw_event(encoder):
    events = run_async_gen(convert_message_to_agui_events(Dummy(), encoder))
    assert any(EventType.RAW.value in e for e in events)

# ============ API Tests ============

def test_stream_agui_basic(test_client, mock_agent) -> None:
    """Test basic streaming interface with AG-UI protocol"""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."

    # Configure mock agent to return event stream
    events = [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make streaming request with AG-UI protocol
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_protocol": "agui"}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and not line.startswith("data: [DONE]"):
                try:
                    event_data = line.lstrip("data: ")
                    messages.append(event_data)
                except json.JSONDecodeError:
                    continue

        # Verify AG-UI protocol events are present
        event_content = "".join(messages)
        assert EventType.RUN_STARTED.value in event_content
        assert EventType.RUN_FINISHED.value in event_content


def test_stream_agui_with_tokens(test_client, mock_agent) -> None:
    """Test AG-UI protocol token streaming"""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " is", " sunny"]
    FINAL_ANSWER = "The weather is sunny"

    # Configure mock agent to return token event stream
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    with test_client.stream(
        "POST", "/stream", json={
            "message": QUESTION, 
            "stream_protocol": "agui", 
            "stream_tokens": True
        }
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and not line.startswith("data: [DONE]"):
                try:
                    event_data = line.lstrip("data: ")
                    messages.append(event_data)
                except json.JSONDecodeError:
                    continue

        # Verify token streaming events are present
        event_content = "".join(messages)
        assert EventType.TEXT_MESSAGE_CHUNK.value in event_content
        assert EventType.RUN_STARTED.value in event_content
        assert EventType.RUN_FINISHED.value in event_content


def test_stream_agui_no_tokens(test_client, mock_agent) -> None:
    """Test AG-UI protocol without token streaming"""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " is", " sunny"]
    FINAL_ANSWER = "The weather is sunny"

    # Configure mock agent to return event stream
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    with test_client.stream(
        "POST", "/stream", json={
            "message": QUESTION, 
            "stream_protocol": "agui", 
            "stream_tokens": False
        }
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and not line.startswith("data: [DONE]"):
                try:
                    event_data = line.lstrip("data: ")
                    messages.append(event_data)
                except json.JSONDecodeError:
                    continue

        # Verify no token streaming events but other AG-UI events are present
        event_content = "".join(messages)
        assert EventType.TEXT_MESSAGE_CHUNK.value not in event_content
        assert EventType.RUN_STARTED.value in event_content
        assert EventType.RUN_FINISHED.value in event_content


def test_stream_agui_with_tools(test_client, mock_agent) -> None:
    """Test AG-UI protocol tool calls"""
    QUESTION = "Calculate 2 + 3"
    
    # Create AI message with tool calls
    tool_message = AIMessage(
        content="I'll calculate that for you.",
        tool_calls=[{
            "name": "Calculator", 
            "args": {"expression": "2 + 3"}, 
            "id": str(uuid4())
        }]
    )

    events = [
        (
            "updates",
            {"math_tool": {"messages": [tool_message]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_protocol": "agui"}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and not line.startswith("data: [DONE]"):
                try:
                    event_data = line.lstrip("data: ")
                    messages.append(event_data)
                except json.JSONDecodeError:
                    continue

        # Verify tool call related AG-UI events are present
        event_content = "".join(messages)
        assert EventType.TOOL_CALL_START.value in event_content
        assert EventType.TOOL_CALL_ARGS.value in event_content
        assert EventType.TOOL_CALL_END.value in event_content
        assert EventType.RUN_STARTED.value in event_content
        assert EventType.RUN_FINISHED.value in event_content


def test_stream_agui_custom_agent(test_client, mock_agent) -> None:
    """Test AG-UI protocol with custom agent"""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather?"
    ANSWER = "It's sunny."

    events = [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return mock_agent
        return AsyncMock()

    with patch("service.service.get_agent", side_effect=agent_lookup):
        with test_client.stream(
            "POST", f"/{CUSTOM_AGENT}/stream", 
            json={"message": QUESTION, "stream_protocol": "agui"}
        ) as response:
            assert response.status_code == 200

            # Collect all SSE messages
            messages = []
            for line in response.iter_lines():
                if line and not line.startswith("data: [DONE]"):
                    try:
                        event_data = line.lstrip("data: ")
                        messages.append(event_data)
                    except json.JSONDecodeError:
                        continue

            # Verify AG-UI events are generated correctly
            event_content = "".join(messages)
            assert EventType.RUN_STARTED.value in event_content
            assert EventType.RUN_FINISHED.value in event_content


def test_stream_agui_interrupt(test_client, mock_agent) -> None:
    """Test AG-UI protocol interrupt handling"""
    QUESTION = "Confirm this action"
    INTERRUPT = "Please confirm: Continue with operation?"

    events = [
        (
            "updates",
            {"__interrupt__": [Interrupt(value=INTERRUPT)]},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_protocol": "agui"}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and not line.startswith("data: [DONE]"):
                try:
                    event_data = line.lstrip("data: ")
                    messages.append(event_data)
                except json.JSONDecodeError:
                    continue

        # Verify interrupt message is handled as text message
        event_content = "".join(messages)
        assert EventType.RUN_STARTED.value in event_content
        assert EventType.TEXT_MESSAGE_START.value in event_content
        assert INTERRUPT in event_content
        assert EventType.RUN_FINISHED.value in event_content


def test_stream_agui_custom_message(test_client, mock_agent) -> None:
    """Test AG-UI protocol custom messages"""
    QUESTION = "Custom request"
    
    # Create custom message
    custom_data = {"action": "special_operation", "data": {"key": "value"}}

    events = [
        (
            "custom",
            custom_data,
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_protocol": "agui"}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and not line.startswith("data: [DONE]"):
                try:
                    event_data = line.lstrip("data: ")
                    messages.append(event_data)
                except json.JSONDecodeError:
                    continue

        # Verify custom events
        event_content = "".join(messages)
        assert EventType.RUN_STARTED.value in event_content
        assert EventType.RAW.value in event_content
        assert "special_operation" in event_content
        assert EventType.RUN_FINISHED.value in event_content


def test_stream_invalid_protocol(test_client, mock_agent) -> None:
    """Test invalid stream protocol"""
    QUESTION = "What is the weather?"

    response = test_client.post(
        "/stream", 
        json={"message": QUESTION, "stream_protocol": "invalid"}
    )
    assert response.status_code == 422  # Validation error


def test_stream_agui_model_param(test_client, mock_agent) -> None:
    """Test AG-UI protocol model parameter passing"""
    QUESTION = "What is the weather?"
    CUSTOM_MODEL = "claude-3.5-sonnet"
    ANSWER = "The weather is sunny."

    events = [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        # Verify model parameter passing is correct
        config = kwargs.get("config", {})
        configurable = config.get("configurable", {})
        assert configurable.get("model") == CUSTOM_MODEL
        
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    with test_client.stream(
        "POST", "/stream", 
        json={
            "message": QUESTION, 
            "stream_protocol": "agui",
            "model": CUSTOM_MODEL
        }
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages to verify response is normal
        messages = []
        for line in response.iter_lines():
            if line and not line.startswith("data: [DONE]"):
                try:
                    event_data = line.lstrip("data: ")
                    messages.append(event_data)
                except json.JSONDecodeError:
                    continue

        event_content = "".join(messages)
        assert EventType.RUN_STARTED.value in event_content
        assert EventType.RUN_FINISHED.value in event_content 