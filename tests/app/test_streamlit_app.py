from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest
from streamlit.testing.v1 import AppTest

from client import AgentClientError
from schema import ChatHistory, ChatMessage
from schema.models import OpenAIModelName


def test_app_simple_non_streaming(mock_agent_client):
    """Test the full app - happy path"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    WELCOME_START = "Hello! I'm an AI agent. Ask me anything!"
    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE),
    )

    assert at.chat_message[0].avatar == "assistant"
    assert at.chat_message[0].markdown[0].value.startswith(WELCOME_START)

    at.sidebar.toggle[0].set_value(False)  # Use Streaming = False
    at.chat_input[0].set_value(PROMPT).run()
    print(at)
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == RESPONSE
    assert not at.exception


def test_app_settings(mock_agent_client):
    """Test the full app - happy path"""
    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["user_id"] = "1234"
    at.run()

    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE),
    )

    at.sidebar.toggle[0].set_value(False)  # Use Streaming = False
    assert at.sidebar.selectbox[0].value == "gpt-4o"
    assert mock_agent_client.agent == "test-agent"
    at.sidebar.selectbox[0].set_value("gpt-4o-mini")
    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    # Basic checks
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == RESPONSE

    # Check the args match the settings
    assert mock_agent_client.agent == "chatbot"
    mock_agent_client.ainvoke.assert_called_with(
        message=PROMPT,
        model=OpenAIModelName.GPT_4O_MINI,
        thread_id=at.session_state.thread_id,
        user_id="1234",
    )
    assert not at.exception


def test_app_thread_id_history(mock_agent_client):
    """Test the thread_id is generated"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Reset and set thread_id
    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["thread_id"] = "1234"
    HISTORY = [
        ChatMessage(type="human", content="What is the weather?"),
        ChatMessage(type="ai", content="The weather is sunny."),
    ]
    mock_agent_client.get_history.return_value = ChatHistory(messages=HISTORY)
    at.run()
    print(at)
    assert at.session_state.thread_id == "1234"
    mock_agent_client.get_history.assert_called_with(thread_id="1234")
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == "What is the weather?"
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == "The weather is sunny."
    assert not at.exception


def test_app_feedback(mock_agent_client):
    """TODO: Can't figure out how to interact with st.feedback"""

    pass


@pytest.mark.asyncio
async def test_app_streaming(mock_agent_client):
    """Test the app with streaming enabled - including tool messages"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Setup mock streaming response
    PROMPT = "What is 6 * 7?"
    ai_with_tool = ChatMessage(
        type="ai",
        content="",
        tool_calls=[{"name": "calculator", "id": "test_call_id", "args": {"expression": "6 * 7"}}],
    )
    tool_message = ChatMessage(type="tool", content="42", tool_call_id="test_call_id")
    final_ai_message = ChatMessage(type="ai", content="The answer is 42")

    messages = [ai_with_tool, tool_message, final_ai_message]

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)  # Use Streaming = True
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    response = at.chat_message[1]
    tool_status = response.status[0]
    assert response.avatar == "assistant"
    assert tool_status.label == "🛠️ Tool Call: calculator"
    assert tool_status.icon == ":material/check:"
    assert tool_status.markdown[0].value == "Input:"
    assert tool_status.json[0].value == '{"expression": "6 * 7"}'
    assert tool_status.markdown[1].value == "Output:"
    assert tool_status.markdown[2].value == "42"
    assert response.markdown[-1].value == "The answer is 42"
    assert not at.exception


@pytest.mark.asyncio
async def test_app_init_error(mock_agent_client):
    """Test the app with an error in the agent initialization"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Setup mock streaming response
    PROMPT = "What is 6 * 7?"
    mock_agent_client.astream.side_effect = AgentClientError("Error connecting to agent")

    at.toggle[0].set_value(True)  # Use Streaming = True
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    assert at.chat_message[0].avatar == "assistant"
    assert at.chat_message[1].avatar == "user"
    assert at.chat_message[1].markdown[0].value == PROMPT
    assert at.error[0].value == "Error generating response: Error connecting to agent"
    assert not at.exception


def test_app_new_chat_btn(mock_agent_client):
    at = AppTest.from_file("../../src/streamlit_app.py").run()
    thread_id_a = at.session_state.thread_id

    at.sidebar.button[0].click().run()

    assert at.session_state.thread_id != thread_id_a
    assert not at.exception


@pytest.fixture
def multi_agent_messages():
    """Fixture providing reusable messages for multi-agent tests"""
    from schema import ChatMessage

    # tool 1
    tool_1 = ChatMessage(
        type="ai",
        content="Starting tool 1...",
        tool_calls=[{"name": "do_work_1", "id": "tool-1", "args": {"my-arg": "value"}}]
    )
    tool_1_result = ChatMessage(type="tool", content="Tool 1 complete", tool_call_id="tool-1")

    # tool 2
    tool_2 = ChatMessage(
        type="ai",
        content="Starting tool 2...",
        tool_calls=[{"name": "do_work_2", "id": "tool-2", "args": {"my-arg-2": "value"}}]
    )
    tool_2_result = ChatMessage(type="tool", content="Tool 2 complete", tool_call_id="tool-2")

    # Transfer to agent A
    transfer_a = ChatMessage(
        type="ai",
        content="Transferring to agent A...",
        tool_calls=[{"name": "transfer_to_agent_a", "id": "transfer-a", "args": {"task": "task_1"}}],
    )
    transfer_a_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_a",
        tool_call_id="transfer-a",
    )

    # Agent A transfers back to supervisor
    transfer_back_a = ChatMessage(
        type="ai",
        content="Agent A finished.",
        tool_calls=[{"name": "transfer_back_to_supervisor", "id": "back-a-super", "args": {"result": "result_1"}}]
    )
    transfer_back_a_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_supervisor",
        tool_call_id="back-a-super",
    )

    # Final response
    supervisor_final = ChatMessage(
        type="ai",
        content="All agents have completed their tasks successfully."
    )

    return {
        'tool_1': tool_1,
        'tool_1_result': tool_1_result,
        'tool_2': tool_2,
        'tool_2_result': tool_2_result,
        'transfer_a': transfer_a,
        'transfer_a_success': transfer_a_success,
        'transfer_back_a': transfer_back_a,
        'transfer_back_a_success': transfer_back_a_success,
        'supervisor_final': supervisor_final,
    }


@pytest.mark.asyncio
async def test_app_streaming_hierarchical_sub_agents(mock_agent_client, multi_agent_messages):
    """Test hierarchical sub-agent UI with proper status containers and visual indicators"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    PROMPT = "Test hierarchical sub-agents with proper UI"

    # Supervisor -> Agent A (with tool_1 and tool_2) -> Supervisor
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages['transfer_a'], messages['transfer_a_success'],
            messages['tool_1'], messages['tool_1_result'],
            messages['tool_2'], messages['tool_2_result'],
            messages['transfer_back_a'], messages['transfer_back_a_success'],
            messages['supervisor_final']
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    # Verify the transfer message is displayed
    assert ai_message.children[0].value == "Transferring to agent A...", "First child should be transfer message"

    # Verify the sub-agent status container
    status_agent = ai_message.status[0]
    assert status_agent == ai_message.children[1], "Second child should be the status container"
    assert "💼 Sub Agent:" in status_agent.label or "transfer_to_agent_a" in status_agent.label

    # Verify tool calls are displayed as popovers within the status
    assert status_agent.children[0].value == "Starting tool 1...", "First tool message should be displayed"

    popover_1 = status_agent.children[1]
    assert hasattr(popover_1, 'type') and popover_1.type == 'popover', \
        "Tool calls should be displayed as popovers"
    assert popover_1.proto.popover.label == "do_work_1"
    assert popover_1.proto.popover.icon == "🛠️"
    assert popover_1.markdown[0].value == "**Tool:** do_work_1"
    assert popover_1.markdown[2].value == "**Output:**"
    assert popover_1.markdown[3].value == "Tool 1 complete"

    # Verify second tool call
    assert status_agent.children[2].value == "Starting tool 2...", "Second tool message should be displayed"

    popover_2 = status_agent.children[3]
    assert popover_2.type == "popover"
    assert popover_2.proto.popover.label == "do_work_2"
    assert popover_2.proto.popover.icon == "🛠️"

    # Verify final supervisor message
    assert ai_message.children[2].value == "All agents have completed their tasks successfully.", \
        "Final supervisor message should be displayed"

    assert not at.exception
