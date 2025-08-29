from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from agents.agents import Agent
from agents.utils import CustomData
from client import AgentClient
from schema.schema import ChatMessage
from service.utils import langchain_to_chat_message

START_MESSAGE = CustomData(type="start", data={"key1": "value1", "key2": 123})

STATIC_MESSAGES = [
    AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="test_tool",
                args={"arg1": "value1"},
                id="test_call_id",
            ),
        ],
    ),
    ToolMessage(content="42", tool_call_id="test_call_id"),
    AIMessage(content="The answer is 42"),
    CustomData(type="end", data={"time": "end"}).to_langchain(),
]


EXPECTED_OUTPUT_MESSAGES = [
    langchain_to_chat_message(m) for m in [START_MESSAGE.to_langchain()] + STATIC_MESSAGES
]


def test_messages_conversion() -> None:
    """Verify that our list of messages is converted to the expected output."""

    messages = EXPECTED_OUTPUT_MESSAGES

    # Verify the sequence of messages
    assert len(messages) == 5

    # First message: Custom data start marker
    assert messages[0].type == "custom"
    assert messages[0].custom_data == {"key1": "value1", "key2": 123}

    # Second message: AI with tool call
    assert messages[1].type == "ai"
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0]["name"] == "test_tool"
    assert messages[1].tool_calls[0]["args"] == {"arg1": "value1"}

    # Third message: Tool response
    assert messages[2].type == "tool"
    assert messages[2].content == "42"
    assert messages[2].tool_call_id == "test_call_id"

    # Fourth message: Final AI response
    assert messages[3].type == "ai"
    assert messages[3].content == "The answer is 42"

    # Fifth message: Custom data end marker
    assert messages[4].type == "custom"
    assert messages[4].custom_data == {"time": "end"}


async def static_messages(state: MessagesState, writer: StreamWriter) -> MessagesState:
    START_MESSAGE.dispatch(writer)
    return {"messages": STATIC_MESSAGES}


agent = StateGraph(MessagesState)
agent.add_node("static_messages", static_messages)
agent.set_entry_point("static_messages")
agent.add_edge("static_messages", END)
static_agent = agent.compile(checkpointer=MemorySaver())


@pytest.fixture
def mock_database_settings(mock_env):
    """Fixture to ensure database settings are clean"""
    with patch("memory.settings") as mock_settings:
        yield mock_settings


def test_agent_stream(mock_database_settings, mock_httpx):
    """Test that streaming from our static agent works correctly with token streaming."""
    agent_meta = Agent(description="A static agent.", graph=static_agent)
    with patch.dict("agents.agents.agents", {"static-agent": agent_meta}, clear=True):
        client = AgentClient(agent="static-agent")

    # Use stream to get intermediate responses
    messages = []

    def agent_lookup(agent_id):
        if agent_id == "static-agent":
            return static_agent
        return None

    with patch("service.service.get_agent", side_effect=agent_lookup):
        for response in client.stream("Test message", stream_tokens=False):
            if isinstance(response, ChatMessage):
                messages.append(response)

    for expected, actual in zip(EXPECTED_OUTPUT_MESSAGES, messages):
        actual.run_id = None
        assert expected == actual
