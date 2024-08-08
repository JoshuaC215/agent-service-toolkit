from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, ToolCall
from schema import ChatMessage

def test_messages_to_langchain():
    human_message = ChatMessage(type="human", content="Hello, world!")
    lc_message = human_message.to_langchain()
    assert isinstance(lc_message, HumanMessage)
    assert lc_message.type == "human"
    assert lc_message.content == "Hello, world!"

def test_messages_from_langchain():
    lc_human_message = HumanMessage(content="Hello, world!")
    human_message = ChatMessage.from_langchain(lc_human_message)
    assert human_message.type == "human"
    assert human_message.content == "Hello, world!"
    assert lc_human_message == human_message.to_langchain()

    lc_ai_message = AIMessage(content="Hello, world!")
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert ai_message.type == "ai"
    assert ai_message.content == "Hello, world!"
    assert lc_ai_message == ai_message.to_langchain()

    lc_tool_message = ToolMessage(content="Hello, world!", tool_call_id="123")
    tool_message = ChatMessage.from_langchain(lc_tool_message)
    assert tool_message.type == "tool"
    assert tool_message.content == "Hello, world!"
    assert tool_message.tool_call_id == "123"
    assert lc_tool_message == tool_message.to_langchain()

    lc_system_message = SystemMessage(content="Hello, world!")
    try:
        _ = ChatMessage.from_langchain(lc_system_message)
    except ValueError as e:
        assert str(e) == "Unsupported message type: SystemMessage"

def test_message_run_id_usage():
    run_id = "847c6285-8fc9-4560-a83f-4e6285809254"
    lc_message = AIMessage(content="Hello, world!")
    ai_message = ChatMessage.from_langchain(lc_message)
    ai_message.run_id = run_id
    assert ai_message.run_id == run_id

def test_messages_tool_calls():
    tool_call = ToolCall(name="test_tool", args={"x": 1, "y": 2}, id="call_Jja7")
    lc_ai_message = AIMessage(content="", tool_calls=[tool_call])
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert ai_message.tool_calls[0]["id"] == "call_Jja7"
    assert ai_message.tool_calls[0]["name"] == "test_tool"
    assert ai_message.tool_calls[0]["args"] == {"x": 1, "y": 2}
    assert lc_ai_message == ai_message.to_langchain()
