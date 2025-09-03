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
        tool_calls=[{"name": "do_work_1", "id": "tool-1", "args": {"my-arg": "value"}}],
    )
    tool_1_result = ChatMessage(type="tool", content="Tool 1 complete", tool_call_id="tool-1")

    # tool 2
    tool_2 = ChatMessage(
        type="ai",
        content="Starting tool 2...",
        tool_calls=[{"name": "do_work_2", "id": "tool-2", "args": {"my-arg-2": "value"}}],
    )
    tool_2_result = ChatMessage(type="tool", content="Tool 2 complete", tool_call_id="tool-2")

    # Transfer to agent A
    transfer_a = ChatMessage(
        type="ai",
        content="Transferring to agent A...",
        tool_calls=[
            {"name": "transfer_to_agent_a", "id": "transfer-a", "args": {"task": "task_1"}}
        ],
    )
    transfer_a_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_a",
        tool_call_id="transfer-a",
    )

    # Agent A transfers to agent B (sub-agent)
    transfer_b_from_a = ChatMessage(
        type="ai",
        content="Agent A delegating to agent B...",
        tool_calls=[
            {"name": "transfer_to_agent_b", "id": "transfer-a-b", "args": {"sub_task": "task_2"}}
        ],
    )
    transfer_b_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_b",
        tool_call_id="transfer-a-b",
    )

    # Agent B transfers back to A
    transfer_back_b = ChatMessage(
        type="ai",
        content="Agent B finished.",
        tool_calls=[
            {"name": "transfer_back_to_agent_a", "id": "back-b-a", "args": {"result": "result_2"}}
        ],
    )
    transfer_back_b_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_agent_a",
        tool_call_id="back-b-a",
    )

    # Agent A transfers back to supervisor
    transfer_back_a = ChatMessage(
        type="ai",
        content="Agent A finished.",
        tool_calls=[
            {
                "name": "transfer_back_to_supervisor",
                "id": "back-a-super",
                "args": {"result": "result_1"},
            }
        ],
    )
    transfer_back_a_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_supervisor",
        tool_call_id="back-a-super",
    )

    # Supervisor continues and transfers to agent C (sibling to A)
    supervisor_continues = ChatMessage(
        type="ai",
        content="Now transferring to agent C...",
        tool_calls=[
            {"name": "transfer_to_agent_c", "id": "transfer-c", "args": {"task": "task_3"}}
        ],
    )
    transfer_c_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_c",
        tool_call_id="transfer-c",
    )

    # Agent C transfers back
    transfer_back_c = ChatMessage(
        type="ai",
        content="Agent C finished.",
        tool_calls=[
            {
                "name": "transfer_back_to_supervisor",
                "id": "back-c-super",
                "args": {"result": "result_3"},
            }
        ],
    )
    transfer_back_c_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_supervisor",
        tool_call_id="back-c-super",
    )

    # Final response
    supervisor_final = ChatMessage(
        type="ai", content="All agents have completed their tasks successfully."
    )

    return {
        "tool_1": tool_1,
        "tool_1_result": tool_1_result,
        "tool_2": tool_2,
        "tool_2_result": tool_2_result,
        "transfer_a": transfer_a,
        "transfer_a_success": transfer_a_success,
        "transfer_b_from_a": transfer_b_from_a,
        "transfer_b_success": transfer_b_success,
        "transfer_back_b": transfer_back_b,
        "transfer_back_b_success": transfer_back_b_success,
        "transfer_back_a": transfer_back_a,
        "transfer_back_a_success": transfer_back_a_success,
        "supervisor_continues": supervisor_continues,
        "transfer_c_success": transfer_c_success,
        "transfer_back_c": transfer_back_c,
        "transfer_back_c_success": transfer_back_c_success,
        "supervisor_final": supervisor_final,
    }


@pytest.mark.asyncio
async def test_app_streaming_single_sub_agent(mock_agent_client, multi_agent_messages):
    """Test a single sub-agent with multiple tool calls to verify popover functionality"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    PROMPT = "Test single sub-agent with multiple tools"

    # Use the fixture and include multiple work tools to test multiple popovers
    # Supervisor -> Agent A (with tool_1 and tool_2) -> Supervisor
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message"
    )

    status_agent = ai_message.status[0]
    assert status_agent == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_agent.label

    assert status_agent.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )

    popover_1 = status_agent.children[1]
    assert hasattr(popover_1, "type") and popover_1.type == "popover", (
        "Second child of status should be a popover for the first tool call"
    )
    assert popover_1.proto.popover.label == "do_work_1"
    assert popover_1.proto.popover.icon == "🛠️"
    assert popover_1.markdown[0].value == "**Tool:** do_work_1"
    assert popover_1.markdown[1].value == "**Input:**"
    assert '"my-arg": "value"' in popover_1.json[0].value
    assert popover_1.markdown[2].value == "**Output:**"
    assert popover_1.markdown[3].value == "Tool 1 complete"

    assert status_agent.children[2].value == "Starting tool 2...", (
        "Third child of status should be tool 2 message"
    )

    popover_2 = status_agent.children[3]
    assert hasattr(popover_2, "type") and popover_2.type == "popover", (
        "Fourth child of the status should be a popover for the second tool call"
    )
    assert popover_2.proto.popover.label == "do_work_2"

    assert not at.exception


@pytest.mark.asyncio
async def test_app_streaming_sequential_sub_agents(mock_agent_client, multi_agent_messages):
    """Test when the supervisor agent transfers to sub agent A, then back to supervisor, then transfers to sub agent C, and back again"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    PROMPT = "Test multiple transfer back patterns"

    # Create message flow for sequential agents: Supervisor -> Agent A (using tool 1) -> Supervisor -> Agent C (using tool 2) -> Supervisor
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_continues"],
            messages["transfer_c_success"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_c"],
            messages["transfer_back_c_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message to agent A"
    )

    status_a = ai_message.status[0]
    assert status_a == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_a.label

    assert status_a.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )
    # Second child of status should be the popover for the first tool call
    popover_a = status_a.children[1]
    assert popover_a.type == "popover"
    assert popover_a.proto.popover.label == "do_work_1"
    assert popover_a.proto.popover.icon == "🛠️"
    assert popover_a.markdown[0].value == "**Tool:** do_work_1"
    assert popover_a.markdown[1].value == "**Input:**"
    assert popover_a.json[0].value == '{"my-arg": "value"}'
    assert popover_a.markdown[2].value == "**Output:**"
    assert popover_a.markdown[3].value == "Tool 1 complete"

    assert ai_message.children[2].value == "Now transferring to agent C...", (
        "Third child should be transfer message to agent C"
    )

    status_c = ai_message.status[1]
    assert status_c == ai_message.children[3], "Fourth child should be the second status"
    assert "transfer_to_agent_c" in status_c.label

    assert status_c.children[0].value == "Starting tool 2...", (
        "First child of next status should be tool 2 message"
    )
    popover_c = status_c.children[1]
    assert popover_c.type == "popover"
    assert popover_c.proto.popover.label == "do_work_2"
    assert popover_c.proto.popover.icon == "🛠️"
    assert popover_c.markdown[0].value == "**Tool:** do_work_2"
    assert popover_c.markdown[1].value == "**Input:**"
    assert popover_c.json[0].value == '{"my-arg-2": "value"}'
    assert popover_c.markdown[2].value == "**Output:**"
    assert popover_c.markdown[3].value == "Tool 2 complete"

    assert ai_message.children[4].value == "All agents have completed their tasks successfully.", (
        "Fifth child should be final supervisor message"
    )

    assert len(ai_message.children) == 6, (
        f"Should have 6 children: transfer to a, status for a, transfer to c, status for c, final message, feedback stars - got {len(ai_message.children)}"
    )

    assert not at.exception


@pytest.mark.asyncio
async def test_app_streaming_nested_sub_agents(mock_agent_client, multi_agent_messages):
    """Test nested sub-agents where agent B is a sub-agent of agent A"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    PROMPT = "Test nested sub-agents"

    # Create message flow for nested sub-agents: Supervisor -> Agent A (using tool 1) -> Agent B (using tool 2) -> Agent A -> Supervisor
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["transfer_b_from_a"],
            messages["transfer_b_success"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_b"],
            messages["transfer_back_b_success"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message to agent A"
    )

    status_a = ai_message.status[0]
    assert status_a == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_a.label

    assert status_a.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )
    # Second child of status should be the popover for the first tool call
    popover_a = status_a.children[1]
    assert popover_a.type == "popover"
    assert popover_a.proto.popover.label == "do_work_1"
    assert popover_a.proto.popover.icon == "🛠️"
    assert popover_a.markdown[0].value == "**Tool:** do_work_1"
    assert popover_a.markdown[1].value == "**Input:**"
    assert popover_a.json[0].value == '{"my-arg": "value"}'
    assert popover_a.markdown[2].value == "**Output:**"
    assert popover_a.markdown[3].value == "Tool 1 complete"

    assert status_a.children[2].value == "Agent A delegating to agent B...", (
        "Third child of status should be transfer message to agent B"
    )

    # Fourth child of status should be the nested status for Agent B
    nested_status_b = status_a.children[3]
    assert "transfer_to_agent_b" in nested_status_b.label

    assert nested_status_b.children[0].value == "Starting tool 2...", (
        "First child of nested status should be tool 2 message"
    )
    # Second child of nested status should be the popover for task 2 tool call
    popover_b = nested_status_b.children[1]
    assert popover_b.type == "popover"
    assert popover_b.proto.popover.label == "do_work_2"
    assert popover_b.proto.popover.icon == "🛠️"
    assert popover_b.markdown[0].value == "**Tool:** do_work_2"
    assert popover_b.markdown[1].value == "**Input:**"
    assert popover_b.json[0].value == '{"my-arg-2": "value"}'
    assert popover_b.markdown[2].value == "**Output:**"
    assert popover_b.markdown[3].value == "Tool 2 complete"

    assert ai_message.children[2].value == "All agents have completed their tasks successfully.", (
        "Third child should be final supervisor message"
    )

    assert len(ai_message.children) == 4, (
        f"Should have 4 children: transfer to a, status for a (with nested b), final message, feedback stars - got {len(ai_message.children)}"
    )

    assert not at.exception
