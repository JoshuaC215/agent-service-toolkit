import json
from unittest.mock import patch

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langgraph.checkpoint.memory import MemorySaver

from schema import ChatMessage, StreamInput


class FakeToolModel(FakeMessagesListChatModel):
    """A fake model that supports tool calls."""

    def __init__(self, responses: list[BaseMessage]):
        super().__init__(responses=responses)

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_three_layer_supervisor_hierarchy_agent_with_fake_model():
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="transfer_to_sub-agent-research_expert", args={}, id="call-1")
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[ToolCall(name="transfer_to_sub-agent-math_expert", args={}, id="call-2")],
        ),
        AIMessage(
            content="", tool_calls=[ToolCall(name="add", args={"a": 2, "b": 3}, id="call-3")]
        ),
        AIMessage(content="2+3 is 5"),  # This is the response from the math expert,
        AIMessage(
            content="The Maths Expert says the answer is 5."
        ),  # This is the response from the research expert
        AIMessage(content="The result is 5."),
    ]

    from agents.langgraph_supervisor_hierarchy_agent import workflow

    agent = workflow(FakeToolModel(responses))
    agent.checkpointer = MemorySaver()

    with patch("service.service.get_agent", return_value=agent):
        from service.service import message_generator

        messages = []
        async for chunk in message_generator(
            StreamInput(message="Add 2 and 3"), agent_id="langgraph-supervisor-hierarchy-agent"
        ):
            if chunk and chunk.strip() != "data: [DONE]":  # Skip [DONE] message
                chat_message = json.loads(chunk.lstrip("data: "))["content"]
                messages.append(ChatMessage.model_validate(chat_message))

        for msg in messages:
            print(msg)

        assert len(messages) > 0
        assert messages[0].type == "ai"
        assert len(messages[0].tool_calls) > 0
        assert messages[1].type == "tool"
        assert messages[2].type == "ai"
        assert messages[-1].type == "ai"
        assert "2+3 is 5" in messages[-1].content
