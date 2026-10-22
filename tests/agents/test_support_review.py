from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.constants import TAG_NOSTREAM

from agents import order_agent as module
from agents.support_review import (
    GroundingDecision,
    classify_request,
    current_turn,
    review_draft,
    structured_model,
)


class FakeToolsModel(FakeMessagesListChatModel):
    def bind_tools(self, tools):
        return self


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(module.settings, "ORDER_AGENT_TRIAGE_ENABLED", True)
    monkeypatch.setattr(module.settings, "ORDER_AGENT_REVIEW_ENABLED", True)
    monkeypatch.setattr(module, "classify_request", AsyncMock(return_value="solve"))

    def configure(responses, reviews):
        model = FakeToolsModel(responses=responses)
        monkeypatch.setattr(module, "get_model", lambda _: model)
        reviewer = AsyncMock(side_effect=reviews)
        monkeypatch.setattr(module, "review_draft", reviewer)
        return reviewer

    return configure


@pytest.mark.asyncio
async def test_rejected_draft_rewritten_before_public_response(runtime):
    review = runtime(
        [AIMessage(content="UNSUPPORTED"), AIMessage(content="Corrected answer")],
        [
            GroundingDecision(supported=False, feedback="Remove unsupported claim"),
            GroundingDecision(supported=True, feedback=""),
        ],
    )
    result = (
        await module.build_order_graph()
        .compile()
        .ainvoke({"messages": [HumanMessage(content="question")]})
    )
    assert review.await_count == 2
    assert result["messages"][-1].content == "Corrected answer"
    assert all(m.content != "UNSUPPORTED" for m in result["messages"])


@pytest.mark.asyncio
async def test_review_failure_is_bounded_and_never_publishes_draft(runtime):
    review = runtime([AIMessage(content="UNSAFE")], [RuntimeError("offline")] * 3)
    result = (
        await module.build_order_graph()
        .compile()
        .ainvoke({"messages": [HumanMessage(content="question")]})
    )
    assert review.await_count == 1
    assert result["review_attempts"] == 1
    assert "UNSAFE" not in result["messages"][-1].content


@pytest.mark.asyncio
async def test_review_retries_never_repeat_mutations(runtime):
    calls = []

    @tool("create_ticket")
    async def create_ticket() -> dict:
        """Record a fake mutation."""
        calls.append(1)
        return {"success": True, "ticket": {"ticket_id": "T1"}}

    request = AIMessage(
        content="UNREVIEWED CLAIM",
        tool_calls=[{"name": "create_ticket", "args": {}, "id": "call1"}],
    )
    runtime(
        [request, AIMessage(content="UNSAFE"), request],
        [GroundingDecision(supported=False, feedback="Unsupported")] * 3,
    )
    result = (
        await module.build_order_graph([create_ticket])
        .compile()
        .ainvoke({"messages": [HumanMessage(content="Create a ticket")]})
    )
    assert calls == [1]
    assert "UNREVIEWED CLAIM" not in str([m.content for m in result["messages"]])


@pytest.mark.asyncio
async def test_internal_llm_messages_are_not_streamed(runtime):
    runtime(
        [AIMessage(content="UNREVIEWED"), AIMessage(content="Reviewed answer")],
        [
            GroundingDecision(supported=False, feedback="Rewrite"),
            GroundingDecision(supported=True, feedback=""),
        ],
    )
    streamed = [
        event
        async for event in module.build_order_graph()
        .compile()
        .astream({"messages": [HumanMessage(content="question")]}, stream_mode="messages")
    ]
    assert "UNREVIEWED" not in str([message.content for message, _ in streamed])
    assert any(message.content == "Reviewed answer" for message, _ in streamed)


@pytest.mark.asyncio
async def test_handoff_does_not_execute_solver(runtime, monkeypatch):
    review = runtime([AIMessage(content="NEVER")], [])
    monkeypatch.setattr(module, "classify_request", AsyncMock(return_value="handoff"))
    result = (
        await module.build_order_graph()
        .compile()
        .ainvoke({"messages": [HumanMessage(content="Speak to a human")]})
    )
    review.assert_not_awaited()
    assert "NEVER" not in result["messages"][-1].content


@pytest.mark.asyncio
async def test_structured_review_receives_only_current_turn_and_nostream_tag():
    messages = [
        HumanMessage(content="old"),
        ToolMessage(content="OLD", tool_call_id="1"),
        HumanMessage(content="new"),
        ToolMessage(content="NEW", tool_call_id="2"),
    ]
    assert len(current_turn(messages)) == 2
    model = Mock()
    runner = Mock(ainvoke=AsyncMock(return_value=GroundingDecision(supported=True, feedback="")))
    model.with_structured_output.return_value = runner
    await review_draft(model, messages, "draft", {})
    args = runner.ainvoke.call_args.args
    assert "OLD" not in args[0][-1].content
    assert "NEW" in args[0][-1].content
    assert TAG_NOSTREAM in args[1]["tags"]


@pytest.mark.asyncio
async def test_triage_structured_response():
    model = Mock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value={"route": "solve"})
    assert await classify_request(model, [HumanMessage(content="my orders")], {}) == "solve"


def test_deepseek_structured_calls_disable_thinking_without_mutating_solver():
    from langchain_openai import ChatOpenAI

    solver = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key="test-key",
        extra_body={"thinking": {"type": "enabled"}},
    )
    stage = structured_model(solver)
    assert stage.extra_body["thinking"]["type"] == "disabled"
    assert solver.extra_body["thinking"]["type"] == "enabled"
    assert structured_model(ChatOpenAI(model="other-model", api_key="test-key")).extra_body is None
