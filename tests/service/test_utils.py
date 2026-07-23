import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from service.utils import ensure_thread_ownership, langchain_to_chat_message


@pytest.mark.asyncio
async def test_ensure_thread_ownership_against_real_checkpointer() -> None:
    """Regression test for the #305 bug: the stored user_id must be read from
    checkpoint metadata, not state.config. A real checkpointer (not a hand-built
    mock) is required to catch this - #305's own mock-based tests placed user_id
    under `config`, which a real checkpointer never does, so they passed while the
    shipped check was dead code. This also pins the LangGraph contract itself, so
    it fails loudly (here) rather than silently (as a live 403 that never fires) if
    a future LangGraph version changes where configurable values end up.
    """
    graph = StateGraph(MessagesState)
    graph.add_node("noop", lambda state: {})
    graph.set_entry_point("noop")
    graph.add_edge("noop", END)
    compiled = graph.compile(checkpointer=MemorySaver())

    write_config = RunnableConfig(
        configurable={"thread_id": "ownership-thread", "user_id": "owner-id"}
    )
    await compiled.ainvoke({"messages": []}, config=write_config)

    read_config = RunnableConfig(configurable={"thread_id": "ownership-thread"})
    state = await compiled.aget_state(read_config)

    # Pin the actual contract ensure_thread_ownership relies on.
    assert state.config.get("configurable", {}).get("user_id") is None
    assert state.metadata is not None
    assert state.metadata.get("user_id") == "owner-id"

    # And exercise the real behavior against that real state.
    ensure_thread_ownership(state.metadata, "owner-id")  # no raise: matching owner
    ensure_thread_ownership(state.metadata, None)  # no raise: no user_id supplied

    with pytest.raises(HTTPException) as exc_info:
        ensure_thread_ownership(state.metadata, "different-user-id")
    assert exc_info.value.status_code == 403


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
