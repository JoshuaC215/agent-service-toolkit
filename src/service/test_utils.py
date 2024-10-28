from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage

from service.utils import langchain_to_chat_message


def test_messages_from_langchain() -> None:
    lc_human_message = HumanMessage(content="Hello, world!")
    human_message = langchain_to_chat_message(lc_human_message)
    assert human_message.type == "human"
    assert human_message.content == "Hello, world!"

    lc_ai_message = AIMessage(content="Hello, world!")
    ai_message = langchain_to_chat_message(lc_ai_message)
    assert ai_message.type == "ai"
    assert ai_message.content == "Hello, world!"

    lc_tool_message = ToolMessage(content="Hello, world!", tool_call_id="123")
    tool_message = langchain_to_chat_message(lc_tool_message)
    assert tool_message.type == "tool"
    assert tool_message.content == "Hello, world!"
    assert tool_message.tool_call_id == "123"

    lc_system_message = SystemMessage(content="Hello, world!")
    try:
        _ = langchain_to_chat_message(lc_system_message)
    except ValueError as e:
        assert str(e) == "Unsupported message type: SystemMessage"


def test_message_run_id_usage() -> None:
    run_id = "847c6285-8fc9-4560-a83f-4e6285809254"
    lc_message = AIMessage(content="Hello, world!")
    ai_message = langchain_to_chat_message(lc_message)
    ai_message.run_id = run_id
    assert ai_message.run_id == run_id


def test_messages_tool_calls() -> None:
    tool_call = ToolCall(name="test_tool", args={"x": 1, "y": 2}, id="call_Jja7")
    lc_ai_message = AIMessage(content="", tool_calls=[tool_call])
    ai_message = langchain_to_chat_message(lc_ai_message)
    assert ai_message.tool_calls[0]["id"] == "call_Jja7"
    assert ai_message.tool_calls[0]["name"] == "test_tool"
    assert ai_message.tool_calls[0]["args"] == {"x": 1, "y": 2}
